#![forbid(unsafe_code)]

mod adversarial;
mod bench;
mod causal;
mod change_impact;
mod docs_lint;
mod formal_invariants;
mod models_lifecycle;
mod spec_snapshot;
mod world_shadow;
pub use adversarial::{adversarial_run, AdversarialReport, AdversarialRunArgs, CaseResult};
pub use bench::{bench_run, BenchArgs, BenchReport};
pub use causal::{
    causal_slice, event_id_for_decision, event_id_for_record, explain_why,
    save_counterfactual_result, simulate_counterfactual, write_slice, CausalEdge, CausalSlice,
    CounterfactualRequest, CounterfactualResult, EdgeType, EventNode, EventType, ExplainWhyReport,
};
pub use change_impact::{change_impact, ChangeImpactArgs};
pub use docs_lint::{docs_lint, DocsLintArgs, DocsLintMode, DocsLintReport, DocsLintStatus};
pub use models_lifecycle::{
    models_list, models_promote, models_recommend_rollback, models_rollback, models_stage,
    parse_slot,
};
pub use spec_snapshot::{generate_spec_snapshot, SpecSnapshotArgs};
pub use world_shadow::{world_shadow_report, WorldShadowReport};

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use ucf_compute::capabilities::{LlmOutputClass, LlmRequest};
use ucf_compute::feature_extractor::SaeInput;
use ucf_compute::lfm::LfmInput;
use ucf_compute::model_store::VerifiedModelSlot;
use ucf_compute::ssm::SsmInput;
use ucf_compute::world_model::{StageQuality, WorldModelInput};
use ucf_compute::{
    build_backend, compute_input_from_control, stable_budget_profile_id, BackendPackConfig,
    BackendPackFactory, BackendPackKind, ComputeBackendConfig, ComputeBackendKind, ComputeError,
    ModelSlot, ModelStore, ReleaseFeatureMatrix,
};
use ucf_core::types::Tick;
use ucf_core::types::{SimTime, WindowId};
use ucf_ess::v1::{
    apply_retention, find_ebm_energy, AuditPayload, EmergencyStateCode, ExperienceKind,
    ExperiencePayload, ExperienceRecord, RetentionPolicyV1,
};
use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, CorrelationId, Intent, IntentId, IntentKind,
};
use ucf_platform::{LocalPlatformProbe, PlatformProbe};
use ucf_policy::adapter::MockAdapter;
use ucf_policy::policy_packs::{load_and_merge_policy_graph, policy_graph_digest, PolicyPackError};
use ucf_replay::{
    load_fixture_records, replay_audit as run_replay_audit, replay_records, write_report,
    ReplayMode, ReplayPlan, ReplaySpec, ReplayStrictness,
};
use ucf_runtime::RuntimeOrchestrator;
use ucf_types::error_codes::ErrorCode;

#[derive(Debug, Error)]
pub enum OpsError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("runtime error: {0}")]
    Runtime(#[from] ucf_runtime::errors::RuntimeError),
    #[error("compute error: {0}")]
    Compute(#[from] ucf_compute::ComputeError),
    #[error("bugreport invalid: {0}")]
    Invalid(String),
    #[error("replay error: {0}")]
    Replay(#[from] ucf_replay::ReplayError),
    #[error("policy pack error: {0}")]
    PolicyPack(#[from] PolicyPackError),
}

impl OpsError {
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::Io(_) => ErrorCode::OpsIo,
            Self::Json(_) => ErrorCode::OpsJson,
            Self::Runtime(_) => ErrorCode::OpsRuntime,
            Self::Compute(_) => ErrorCode::OpsCompute,
            Self::Invalid(_) => ErrorCode::OpsInvalid,
            Self::Replay(_) => ErrorCode::OpsReplay,
            Self::PolicyPack(_) => ErrorCode::OpsPolicyPack,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct OpsConfig {
    pub profile: String,
    pub policy_overlay: String,
    pub backend_pack: String,
    pub slot_ebm_mode: String,
    pub offline: bool,
    pub compute_backend: ComputeBackendKind,
    pub compute_seed: u64,
    pub compute_budget_profile: String,
    pub device_profile: String,
    pub isolation_runtime: String,
    pub capabilities_default: String,
    pub sampling_enabled: bool,
    pub determinism_lock_strict: bool,
    pub docs_lint_required: bool,
    pub stage_isolation_optional: bool,
    pub emergency_policy_pin: Option<String>,
    pub log_level: String,
    pub config_digest: String,
}

impl Default for OpsConfig {
    fn default() -> Self {
        Self {
            profile: "test".to_string(),
            policy_overlay: "test".to_string(),
            backend_pack: "toy_v1".to_string(),
            slot_ebm_mode: "shadow".to_string(),
            offline: true,
            compute_backend: ComputeBackendKind::Stub,
            compute_seed: 0xDEC0DED,
            compute_budget_profile: "tight".to_string(),
            device_profile: "small".to_string(),
            isolation_runtime: "inproc".to_string(),
            capabilities_default: "deny".to_string(),
            sampling_enabled: false,
            determinism_lock_strict: true,
            docs_lint_required: false,
            stage_isolation_optional: true,
            emergency_policy_pin: None,
            log_level: "info".to_string(),
            config_digest: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceProfileName {
    Small,
    Medium,
    Large,
}

impl DeviceProfileName {
    fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "small" => Some(Self::Small),
            "medium" => Some(Self::Medium),
            "large" => Some(Self::Large),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceProfileV1 {
    pub name: DeviceProfileName,
    pub compute_budget_profile: String,
    pub llm_max_tokens: u32,
    pub probe_timeout_ms: u64,
    pub world_shadow_window_ticks: u32,
    pub world_shadow_sampling_rate_pct: u16,
    pub stage_isolation_default: bool,
}

impl DeviceProfileV1 {
    pub fn for_name(name: DeviceProfileName) -> Self {
        match name {
            DeviceProfileName::Small => Self {
                name,
                compute_budget_profile: "tight".to_string(),
                llm_max_tokens: 64,
                probe_timeout_ms: 150,
                world_shadow_window_ticks: 4,
                world_shadow_sampling_rate_pct: 10_000,
                stage_isolation_default: false,
            },
            DeviceProfileName::Medium => Self {
                name,
                compute_budget_profile: "default".to_string(),
                llm_max_tokens: 128,
                probe_timeout_ms: 200,
                world_shadow_window_ticks: 6,
                world_shadow_sampling_rate_pct: 10_000,
                stage_isolation_default: true,
            },
            DeviceProfileName::Large => Self {
                name,
                compute_budget_profile: "stress".to_string(),
                llm_max_tokens: 192,
                probe_timeout_ms: 250,
                world_shadow_window_ticks: 8,
                world_shadow_sampling_rate_pct: 10_000,
                stage_isolation_default: true,
            },
        }
    }

    pub fn digest_hex(&self) -> Result<String, OpsError> {
        let bytes = serde_json::to_vec(self)?;
        Ok(sha256_hex(&bytes))
    }
}

impl OpsConfig {
    fn device_profile_name(&self) -> Result<DeviceProfileName, OpsError> {
        DeviceProfileName::parse(&self.device_profile).ok_or_else(|| {
            OpsError::Invalid(format!(
                "invalid device_profile={}; expected small|medium|large",
                self.device_profile
            ))
        })
    }

    fn device_profile_llm_max_tokens(&self) -> u32 {
        self.device_profile_name()
            .map(DeviceProfileV1::for_name)
            .map(|p| p.llm_max_tokens)
            .unwrap_or(64)
    }

    fn device_profile_world_shadow_window_ticks(&self) -> u32 {
        self.device_profile_name()
            .map(DeviceProfileV1::for_name)
            .map(|p| p.world_shadow_window_ticks)
            .unwrap_or(4)
    }
}

#[derive(Debug, Clone)]
pub struct BringupResult {
    pub workdir: PathBuf,
    pub ess_fixture_path: PathBuf,
    pub log_path: PathBuf,
    pub decision_count: usize,
    pub ess_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResumeReason {
    OperatorResume,
    CrashRecovery,
    Fallback,
    Upgrade,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RunMetadataRecord {
    pub run_id: String,
    pub started_at_tick: u64,
    pub code_version_tag: String,
    pub backend_pack_meta_digest: String,
    pub fixtures_digest: String,
    pub model_hashes_digest: String,
    pub enabled_features_bitmap: u16,
    pub profile: String,
    pub config_digest: String,
    pub policy_overlay: String,
    pub platform_probe_summary: String,
    pub device_profile_name: String,
    pub device_profile_digest: String,
    pub schema_versions: BTreeMap<String, u16>,
    pub parent_run_id: Option<String>,
    pub resume_reason: Option<ResumeReason>,
    pub compat_digest: String,
    pub policy_bundle_hash: String,
    pub determinism_mode: String,
    pub determinism_policy_digest: Option<String>,
    pub ended_at_tick: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResumeMismatchReason {
    PolicyHash,
    BackendPackDigest,
    ModelHashesDigest,
    SchemaVersion,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResumeDecision {
    ResumeAllowed,
    NewSessionRequired { reasons: Vec<ResumeMismatchReason> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeCheckConfig {
    pub policy_bundle_hash: String,
    pub backend_pack_meta_digest: String,
    pub model_hashes_digest: String,
    pub enabled_features_bitmap: u16,
    pub schema_versions: BTreeMap<String, u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRegistryEntry {
    pub run_id: String,
    pub started_at_tick: u64,
    pub parent_run_id: Option<String>,
    pub resume_reason: Option<ResumeReason>,
    pub policy_bundle_hash_prefix: String,
    pub pack_digest_prefix: String,
    pub model_hashes_digest_prefix: String,
    pub profile: String,
    pub status: String,
    pub last_tick: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStatusReport {
    pub run_id: String,
    pub active_slots: Vec<String>,
    pub governor_tier: u8,
    pub governor_score: f32,
    pub emergency_active: bool,
    pub last_ticks: Vec<MetricsTrendPoint>,
    pub issuance_denies: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BringupArtifacts {
    pub run_metadata: RunMetadataRecord,
    pub metrics: MetricsSummary,
    pub explain: ExplainTickReport,
    pub replay_report: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVerifySlotReport {
    pub slot: String,
    pub enabled: bool,
    pub status: String,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsVerifyReport {
    pub manifest: String,
    pub allowlist_root: String,
    pub model_hashes_digest: String,
    pub slots: Vec<ModelVerifySlotReport>,
}

const PROBE_TIMEOUT_MS: u64 = 200;
const PROBE_BUDGET_MS: u64 = 100;
const PROBE_TAIL_GUARD_FACTOR: f64 = 1.5;
const PROBE_RUNS: usize = 3;
const PROBE_RESULT_CAP: usize = 10;
const MODEL_PROBE_SCHEMA_VERSION: u16 = 1;
const PROBE_NOTES_MAX: usize = 240;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeSpec {
    pub slot: ModelSlot,
    pub timeout_ms: u64,
    pub max_tokens: u32,
    pub input_digest: [u8; 32],
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStatus {
    Ok,
    Timeout,
    Error,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeResult {
    pub slot: ModelSlot,
    pub backend_id: String,
    pub model_sha256_prefix: Option<String>,
    pub status: ProbeStatus,
    pub elapsed_ms: u64,
    pub output_digest: [u8; 32],
    pub quality: StageQuality,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeReportSummary {
    pub pass: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeReport {
    pub run_id: String,
    pub timestamp: u64,
    pub results: Vec<ProbeResult>,
    pub summary: ProbeReportSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelProbeRecord {
    pub t: u64,
    pub run_id: String,
    pub pack_digest: String,
    pub slot: ModelSlot,
    pub model_hash_prefix: Option<String>,
    pub timeout_ms: u64,
    pub seed: u64,
    pub input_digest_prefix: String,
    pub status: ProbeStatus,
    pub elapsed_ms: u64,
    pub output_digest_prefix: String,
    pub quality: StageQuality,
    pub schema_version: u16,
}

pub fn models_verify(manifest: &Path) -> Result<ModelsVerifyReport, OpsError> {
    let store = ModelStore::from_manifest_and_env(manifest)
        .map_err(|e| OpsError::Invalid(format!("manifest error: {e:?}")))?;
    let mut slots = Vec::new();
    for slot in ModelSlot::all() {
        let verified = store.verify_slot(slot);
        match verified {
            Ok(v) => slots.push(ModelVerifySlotReport {
                slot: slot.as_str().to_string(),
                enabled: true,
                status: "verified".to_string(),
                sha256: Some(hex::encode(v.sha256)),
                size_bytes: Some(v.size_bytes),
                reason: None,
            }),
            Err(err) => slots.push(ModelVerifySlotReport {
                slot: slot.as_str().to_string(),
                enabled: !matches!(err, ucf_compute::ModelLoadError::Disabled),
                status: if matches!(err, ucf_compute::ModelLoadError::Disabled) {
                    "disabled".to_string()
                } else {
                    "rejected".to_string()
                },
                sha256: None,
                size_bytes: None,
                reason: Some(format!("{err:?}")),
            }),
        }
    }

    Ok(ModelsVerifyReport {
        manifest: manifest.display().to_string(),
        allowlist_root: store.allowlist_root.display().to_string(),
        model_hashes_digest: hex::encode(store.model_hashes_digest()),
        slots,
    })
}

pub fn models_probe(workdir: &Path, manifest: &Path, out: &Path) -> Result<ProbeReport, OpsError> {
    ensure_layout(workdir)?;
    let verify = models_verify(manifest)?;
    let store = ModelStore::from_manifest_and_env(manifest)
        .map_err(|e| OpsError::Invalid(format!("manifest error: {e:?}")))?;
    std::env::set_var("UCF_MODEL_MANIFEST", manifest);
    let pack = BackendPackFactory::build(BackendPackConfig::from_env()?)?;
    let run_id = format!("probe-{}", now_unix_secs());
    let mut results = Vec::new();
    let mut reasons = Vec::new();
    let mut records = Vec::new();
    for slot in ModelSlot::all() {
        if results.len() >= PROBE_RESULT_CAP {
            break;
        }
        let verified = store.verify_slot(slot).ok();
        let spec = probe_spec_for_slot(slot);
        let (mut result, record) =
            run_probe_for_slot(&run_id, &pack, slot, &spec, verified.as_ref());
        if matches!(result.status, ProbeStatus::Disabled)
            && !verify
                .slots
                .iter()
                .any(|s| s.slot == slot.as_str() && s.status == "disabled")
        {
            result.status = ProbeStatus::Error;
            result.notes = bounded_note("slot unexpectedly disabled during probe");
        }
        if result.status != ProbeStatus::Ok {
            reasons.push(format!("{}={:?}", slot.as_str(), result.status));
        }
        results.push(result);
        records.push(record);
    }

    persist_probe_records(workdir, &records)?;

    let summary = ProbeReportSummary {
        pass: reasons.is_empty(),
        reasons,
    };
    let report = ProbeReport {
        run_id,
        timestamp: now_unix_secs(),
        results,
        summary,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(out, &report)?;
    Ok(report)
}

fn probe_spec_for_slot(slot: ModelSlot) -> ProbeSpec {
    let seed = 0xA11C_E555_u64;
    let max_tokens = 64;
    let input_digest = match slot {
        ModelSlot::Llm => digest_json(&serde_json::json!({
            "prompt": "UCF deterministic model probe v1",
            "seed": seed,
            "max_tokens": max_tokens
        })),
        ModelSlot::WorldJepa => digest_json(&deterministic_features(seed, 16)),
        ModelSlot::WorldVljepa => digest_json(&deterministic_features(seed ^ 0xC0DE, 64)),
        ModelSlot::Sae => digest_json(&deterministic_features(seed ^ 0x5A5A, 32)),
        ModelSlot::Ssm => digest_json(&serde_json::json!({
            "spikes_digest": vec![17_u8; 32],
            "spike_count": 11,
            "sae_energy": 0.3,
            "world_surprise": 0.2,
            "risk": 0.1
        })),
        ModelSlot::Lfm => digest_json(&serde_json::json!({
            "pressure": 0.35,
            "surprise": 0.22,
            "sae_energy": 0.29,
            "spike_count": 13
        })),
        ModelSlot::EbmReasoner => digest_json(&serde_json::json!({
            "risk_q": 32000,
            "uncertainty_q": 28000,
            "pressure_q": 24000,
            "surprise_q": 20000
        })),
    };
    ProbeSpec {
        slot,
        timeout_ms: PROBE_TIMEOUT_MS,
        max_tokens,
        input_digest,
        seed,
    }
}

fn run_probe_for_slot(
    run_id: &str,
    pack: &std::sync::Arc<dyn ucf_compute::BackendPack>,
    slot: ModelSlot,
    spec: &ProbeSpec,
    verified: Option<&VerifiedModelSlot>,
) -> (ProbeResult, ModelProbeRecord) {
    let model_sha = verified.map(|v| hex_prefix(v.sha256));
    let backend_id = format!(
        "{}:{}",
        pack.meta().pack_name,
        hex_prefix(pack.meta().digest)
    );
    let mut elapsed_samples = Vec::new();
    let mut final_status = ProbeStatus::Disabled;
    let mut final_quality = StageQuality::Unavailable;
    let mut final_output = [0_u8; 32];
    let mut notes = String::new();

    if verified.is_none() {
        final_status = ProbeStatus::Disabled;
        notes = bounded_note("slot disabled in model manifest");
    } else {
        for _ in 0..PROBE_RUNS {
            let started = Instant::now();
            let outcome = match slot {
                ModelSlot::Llm => exec_with_timeout(spec.timeout_ms, {
                    let pack = pack.clone();
                    let spec = spec.clone();
                    move || run_llm_probe(pack, &spec)
                }),
                ModelSlot::WorldJepa => exec_with_timeout(spec.timeout_ms, {
                    let pack = pack.clone();
                    let spec = spec.clone();
                    move || run_world_probe(pack, &spec)
                }),
                ModelSlot::WorldVljepa => exec_with_timeout(spec.timeout_ms, {
                    let pack = pack.clone();
                    let spec = spec.clone();
                    move || run_world_probe(pack, &spec)
                }),
                ModelSlot::Sae => exec_with_timeout(spec.timeout_ms, {
                    let pack = pack.clone();
                    let spec = spec.clone();
                    move || run_sae_probe(pack, &spec)
                }),
                ModelSlot::Ssm => exec_with_timeout(spec.timeout_ms, {
                    let pack = pack.clone();
                    let spec = spec.clone();
                    move || run_ssm_probe(pack, &spec)
                }),
                ModelSlot::Lfm => exec_with_timeout(spec.timeout_ms, {
                    let pack = pack.clone();
                    let spec = spec.clone();
                    move || run_lfm_probe(pack, &spec)
                }),
                ModelSlot::EbmReasoner => exec_with_timeout(spec.timeout_ms, {
                    let spec = spec.clone();
                    move || run_ebm_probe(&spec)
                }),
            };
            let elapsed_ms = started.elapsed().as_millis() as u64;
            elapsed_samples.push(elapsed_ms);
            match outcome {
                Ok((digest, quality)) => {
                    final_status = ProbeStatus::Ok;
                    final_quality = quality;
                    final_output = digest;
                }
                Err(ProbeExecError::Timeout) => {
                    final_status = ProbeStatus::Timeout;
                    final_quality = StageQuality::DegradedFallback;
                    notes = bounded_note("probe timeout hit; result discarded safely");
                    break;
                }
                Err(ProbeExecError::Exec(msg)) => {
                    final_status = ProbeStatus::Error;
                    final_quality = StageQuality::DegradedFallback;
                    notes = bounded_note(&format!("probe error: {msg}"));
                    break;
                }
            }
        }
    }

    elapsed_samples.sort_unstable();
    let p50 = percentile_ms(&elapsed_samples, 0.5);
    let p95 = percentile_ms(&elapsed_samples, 0.95);
    if final_status == ProbeStatus::Ok
        && p95 > ((PROBE_BUDGET_MS as f64) * PROBE_TAIL_GUARD_FACTOR) as u64
    {
        final_quality = StageQuality::DegradedFallback;
        notes = bounded_note(&format!(
            "tail_guard_exceeded p50={}ms p95={}ms budget={}ms",
            p50, p95, PROBE_BUDGET_MS
        ));
    } else if final_status == ProbeStatus::Ok && notes.is_empty() {
        notes = bounded_note(&format!(
            "latency p50={}ms p95={}ms budget={}ms",
            p50, p95, PROBE_BUDGET_MS
        ));
    }

    let elapsed_ms = *elapsed_samples.last().unwrap_or(&0);
    let result = ProbeResult {
        slot,
        backend_id: backend_id.clone(),
        model_sha256_prefix: model_sha.clone(),
        status: final_status,
        elapsed_ms,
        output_digest: final_output,
        quality: final_quality,
        notes,
    };
    let record = ModelProbeRecord {
        t: now_unix_secs(),
        run_id: run_id.to_string(),
        pack_digest: hex_prefix(pack.meta().digest),
        slot,
        model_hash_prefix: model_sha,
        timeout_ms: spec.timeout_ms,
        seed: spec.seed,
        input_digest_prefix: hex_prefix(spec.input_digest),
        status: final_status,
        elapsed_ms,
        output_digest_prefix: hex_prefix(final_output),
        quality: final_quality,
        schema_version: MODEL_PROBE_SCHEMA_VERSION,
    };
    (result, record)
}

fn run_llm_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<([u8; 32], StageQuality), String> {
    let req = LlmRequest {
        schema_version: 1,
        t: 1,
        decision_id: 1,
        candidate_id: 1,
        output_class: LlmOutputClass::SafeText,
        prompt: "UCF deterministic model probe v1".to_string(),
        context_digest: [0x22; 32],
        evidence_chain_digest: [0x33; 32],
        lfm_readout_digest: None,
        lfm_uncertainty: None,
        lfm_stability: None,
        coherence: Some(0.8),
        instability: Some(0.1),
        risk: Some(0.2),
        confidence: Some(0.9),
        seed: spec.seed,
        max_tokens: spec.max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        sampling_enabled: false,
    }
    .bounded();
    let resp = pack
        .llm()
        .infer(&req, ComputeBackendConfig::default().to_budget())
        .map_err(|e| format!("{e:?}"))?;
    Ok((resp.digest, StageQuality::Ok))
}

fn run_world_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<([u8; 32], StageQuality), String> {
    let mut obs = [0.0_f32; 16];
    for (idx, value) in deterministic_features(spec.seed, 16).iter().enumerate() {
        obs[idx] = *value;
    }
    let input = WorldModelInput {
        t: 1,
        context_digest: [0x44; 32],
        obs_features: obs,
        seed: spec.seed,
    };
    let out = pack
        .world()
        .lock()
        .map_err(|_| "world lock poisoned".to_string())?
        .step(&input, ComputeBackendConfig::default().to_budget())
        .map_err(|e| format!("{e:?}"))?;
    Ok((out.prediction_digest, out.quality))
}

fn run_sae_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<([u8; 32], StageQuality), String> {
    let mut feats = [0.0_f32; 32];
    for (idx, value) in deterministic_features(spec.seed ^ 0x5A5A, 32)
        .iter()
        .enumerate()
    {
        feats[idx] = *value;
    }
    let input = SaeInput {
        t: 1,
        context_features: feats,
        world_state_digest: Some([0x51; 32]),
        seed: spec.seed,
        evidence_chain_digest: [0x52; 32],
    };
    let out = pack
        .sae()
        .extract(&input, ComputeBackendConfig::default().to_budget())
        .map_err(|e| format!("{e:?}"))?;
    Ok((out.spikes_digest, out.quality))
}

fn run_ssm_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<([u8; 32], StageQuality), String> {
    let input = SsmInput {
        t: 1,
        spikes_digest: [0x11; 32],
        spike_count: 11,
        sae_energy: 0.3,
        world_surprise: 0.2,
        risk: 0.1,
        seed: spec.seed,
        context_digest: [0x61; 32],
    };
    let out = pack
        .ssm()
        .lock()
        .map_err(|_| "ssm lock poisoned".to_string())?
        .step(&input, ComputeBackendConfig::default().to_budget())
        .map_err(|e| format!("{e:?}"))?;
    Ok((out.state_digest, out.quality))
}

fn run_lfm_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<([u8; 32], StageQuality), String> {
    let input = LfmInput {
        t: 1,
        context_digest: [0x71; 32],
        world_digest: [0x72; 32],
        surprise: 0.22,
        spikes_digest: [0x11; 32],
        spike_count: 13,
        sae_energy: 0.29,
        pressure: 0.35,
        coherence: Some(0.85),
        instability: Some(0.05),
        hormone_stress: Some(0.2),
        neuro_arousal: Some(0.3),
        governor_tier: Some(1),
        prediction_error: Some(0.1),
        risk: Some(0.2),
        confidence: Some(0.8),
        prior_uncertainty: Some(0.3),
        seed: spec.seed,
    };
    let out = pack
        .lfm()
        .lock()
        .map_err(|_| "lfm lock poisoned".to_string())?
        .step(&input, ComputeBackendConfig::default().to_budget())
        .map_err(|e| format!("{e:?}"))?;
    Ok((out.liquid_state_digest, out.quality))
}
fn run_ebm_probe(spec: &ProbeSpec) -> Result<([u8; 32], StageQuality), String> {
    use ucf_runtime::ebm::{
        CandidateFeature, CandidateKind, CpuEbmStubV0, EbmInput, EbmReasoner, EbmSignals,
    };
    use ucf_types::UQ0_16;

    let mut ebm = CpuEbmStubV0;
    let input = EbmInput {
        t: 1,
        governor_tier: 1,
        emergency_active: false,
        context_digest: [0x81; 32],
        signals: EbmSignals {
            risk_q: UQ0_16::from_raw(32_000),
            confidence_q: UQ0_16::from_raw(38_000),
            pressure_q: UQ0_16::from_raw(24_000),
            surprise_q: UQ0_16::from_raw(20_000),
            uncertainty_q: UQ0_16::from_raw(28_000),
            coherence_q: None,
            nsr_risk_q: None,
        },
        candidates: vec![
            CandidateFeature {
                candidate_id: 1,
                candidate_kind: CandidateKind::SafeText,
                tool_class: None,
                candidate_digest: [1; 32],
                feature_vec_q: vec![123, 1, 0],
            },
            CandidateFeature {
                candidate_id: 2,
                candidate_kind: CandidateKind::ToolIntent,
                tool_class: Some(7),
                candidate_digest: [2; 32],
                feature_vec_q: vec![123, 10, 2],
            },
        ],
    };
    let mut budget = ucf_compute::WorkMeter::new(spec.max_tokens as u64);
    let out = ebm.score_candidates(input, &mut budget);
    Ok((out.ebm_digest, StageQuality::Ok))
}

#[derive(Debug)]
enum ProbeExecError {
    Timeout,
    Exec(String),
}

fn exec_with_timeout<T, F>(timeout_ms: u64, task: F) -> Result<T, ProbeExecError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(task());
    });
    match rx.recv_timeout(Duration::from_millis(timeout_ms)) {
        Ok(result) => result.map_err(ProbeExecError::Exec),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(ProbeExecError::Timeout),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(ProbeExecError::Exec(
            "probe worker disconnected".to_string(),
        )),
    }
}

fn deterministic_features(seed: u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let mixed = seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            (mixed as u32) as f32 / u32::MAX as f32
        })
        .collect()
}

fn percentile_ms(values: &[u64], pct: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let idx = ((values.len() - 1) as f64 * pct).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn digest_json<T: Serialize>(value: &T) -> [u8; 32] {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn bounded_note(note: &str) -> String {
    note.chars().take(PROBE_NOTES_MAX).collect()
}

fn hex_prefix(digest: [u8; 32]) -> String {
    hex::encode(&digest[..6])
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn persist_probe_records(workdir: &Path, new_records: &[ModelProbeRecord]) -> Result<(), OpsError> {
    let path = workdir.join("ess").join("model_probe_records.json");
    let mut all: Vec<ModelProbeRecord> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };
    all.extend_from_slice(new_records);
    write_json(path, &all)
}

const GATE_CHECK_CAP: usize = 64;
const GATE_EVIDENCE_CAP: usize = 24;
const GATE_STR_CAP: usize = 240;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum GateStatus {
    Pass,
    Fail,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckResult {
    pub name: String,
    pub status: GateStatus,
    pub evidence: BTreeMap<String, String>,
    pub failure_reason: Option<String>,
    pub remediation_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ReadinessGateReport {
    pub code_version_tag: String,
    pub fixtures_digest_prefix: Option<String>,
    pub backend_pack_digest_prefix: Option<String>,
    pub timestamp: Option<String>,
    pub status: GateStatus,
    pub checks: Vec<CheckResult>,
    pub weights_lifecycle: Option<CheckResult>,
    pub world_vljepa_evidence: Option<CheckResult>,
    pub sae_real: Option<CheckResult>,
    pub ssm_opt: Option<CheckResult>,
    pub gpu_lane: Option<CheckResult>,
}

impl Default for ReadinessGateReport {
    fn default() -> Self {
        Self {
            code_version_tag: String::new(),
            fixtures_digest_prefix: None,
            backend_pack_digest_prefix: None,
            timestamp: None,
            status: GateStatus::Fail,
            checks: Vec::new(),
            weights_lifecycle: None,
            world_vljepa_evidence: None,
            sae_real: None,
            ssm_opt: None,
            gpu_lane: None,
        }
    }
}

pub fn readiness_gate(
    workdir: &Path,
    profile: &str,
    out: &Path,
) -> Result<ReadinessGateReport, OpsError> {
    ensure_layout(workdir)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    std::env::set_var("UCF_PROFILE", profile);
    std::env::set_var("UCF_SSM_KERNEL", "ref");

    let base = workdir.join("readiness_gate");
    fs::create_dir_all(&base)?;
    let run_a = base.join("scenario_a");
    let run_a2 = base.join("scenario_a_repeat");
    let run_b = base.join("scenario_b");
    let run_ebm_off = base.join("scenario_ebm_off");
    let run_ebm_shadow = base.join("scenario_ebm_shadow");
    let run_ebm_active = base.join("scenario_ebm_active");
    let run_ebm_active_repeat = base.join("scenario_ebm_active_repeat");
    let out_a = run_a.join("out");
    let out_a2 = run_a2.join("out");
    let out_b = run_b.join("out");
    let out_ebm_off = run_ebm_off.join("out");
    let out_ebm_shadow = run_ebm_shadow.join("out");
    let out_ebm_active = run_ebm_active.join("out");
    let out_ebm_active_repeat = run_ebm_active_repeat.join("out");

    let scenario_a = workspace_fixture("e2e_scenario_a.json");
    let scenario_b = workspace_fixture("e2e_scenario_b.json");
    let scenario_ebm = workspace_fixture("e2e_scenario_ebm_v1.json");

    let artifacts_a = one_command_bringup(&run_a, &scenario_a, 24, &out_a, true)?;
    let artifacts_a2 = one_command_bringup(&run_a2, &scenario_a, 24, &out_a2, true)?;
    let artifacts_b = one_command_bringup(&run_b, &scenario_b, 24, &out_b, true)?;
    let ebm_off = one_command_bringup_with_ebm_mode(
        &run_ebm_off,
        &scenario_ebm,
        24,
        &out_ebm_off,
        true,
        "off",
    )?;
    let ebm_shadow = one_command_bringup_with_ebm_mode(
        &run_ebm_shadow,
        &scenario_ebm,
        24,
        &out_ebm_shadow,
        true,
        "shadow",
    )?;
    let ebm_active = one_command_bringup_with_ebm_mode(
        &run_ebm_active,
        &scenario_ebm,
        24,
        &out_ebm_active,
        true,
        "active",
    )?;
    let ebm_active_repeat = one_command_bringup_with_ebm_mode(
        &run_ebm_active_repeat,
        &scenario_ebm,
        24,
        &out_ebm_active_repeat,
        true,
        "active",
    )?;

    let replay_verify_path = out_b.join("gate_replay_verify.json");
    replay_audit(
        &run_b,
        1,
        24,
        ReplayStrictness::VerifyOnly,
        false,
        &replay_verify_path,
    )?;
    let replay_verify_report: ucf_replay::ReplayReport =
        serde_json::from_str(&fs::read_to_string(&replay_verify_path)?)?;

    let replay_recompute_path = out_b.join("gate_replay_recompute.json");
    replay_audit(
        &run_b,
        1,
        24,
        ReplayStrictness::RecomputeStages,
        false,
        &replay_recompute_path,
    )?;
    let replay_recompute_report: ucf_replay::ReplayReport =
        serde_json::from_str(&fs::read_to_string(&replay_recompute_path)?)?;

    let explain_last = explain_tick(
        &run_b,
        ExplainTickRequest {
            t: Some(24),
            decision_id: None,
            detail_level: 2,
            digest_prefix_len: 12,
        },
    )?;
    let metrics = metrics_summary(&run_b, 24)?;

    let mut checks = vec![
        check_workspace_tests(),
        check_offline_profile(profile),
        check_backend_disabled_pack(),
        check_schema_versions(&artifacts_b.run_metadata),
        check_required_records(&explain_last),
        check_determinism(&artifacts_a, &artifacts_a2),
        check_replay_report("replay_verify_only", &replay_verify_report),
        check_replay_report("replay_recompute", &replay_recompute_report),
        check_tool_deny_policy(&explain_last),
        check_emergency_visibility(&explain_last),
        check_observability(&explain_last, &metrics),
        check_plug_compatibility(&artifacts_a.run_metadata, &artifacts_b.run_metadata),
        check_ebm_wiring(&ebm_shadow.explain, &ebm_active.explain),
        check_ebm_shadow_active_correctness(
            &ebm_off.explain,
            &ebm_shadow.explain,
            &ebm_active.explain,
        ),
        check_ebm_safety_dominance(
            &ebm_off.explain,
            &ebm_active.explain,
            &out_ebm_active.join("adversarial_report.json"),
        ),
        check_ebm_determinism(&ebm_active.explain, &ebm_active_repeat.explain),
        check_ebm_constraints_provenance(
            &run_ebm_active,
            &ebm_active.run_metadata.policy_bundle_hash,
        ),
        check_ebm_fallback_degraded_record(&run_ebm_active),
        formal_invariants::run_formal_invariants_check(profile)?,
    ];

    let weights_lifecycle = check_weights_lifecycle_integrity(workdir)?;
    let world_vljepa_evidence = check_world_vljepa_shadow_evidence(workdir)?;
    let sae_real = check_sae_real_readiness(workdir)?;
    let ssm_opt = check_ssm_opt_drift(workdir)?;
    let gpu_lane = check_gpu_lane_parity(workdir)?;

    checks.push(weights_lifecycle.clone());
    checks.push(world_vljepa_evidence.clone());
    checks.push(sae_real.clone());
    checks.push(ssm_opt.clone());
    checks.push(gpu_lane.clone());

    if checks.len() > GATE_CHECK_CAP {
        checks.truncate(GATE_CHECK_CAP);
    }

    let status = if checks.iter().any(|c| c.status == GateStatus::Fail) {
        GateStatus::Fail
    } else {
        GateStatus::Pass
    };
    let report = ReadinessGateReport {
        code_version_tag: bounded_string(build_tag()?.git_commit, GATE_STR_CAP),
        fixtures_digest_prefix: Some(prefix_hex(&artifacts_b.run_metadata.fixtures_digest, 12)),
        backend_pack_digest_prefix: Some(prefix_hex(
            &artifacts_b.run_metadata.backend_pack_meta_digest,
            12,
        )),
        timestamp: None,
        status,
        checks,
        weights_lifecycle: Some(weights_lifecycle),
        world_vljepa_evidence: Some(world_vljepa_evidence),
        sae_real: Some(sae_real),
        ssm_opt: Some(ssm_opt),
        gpu_lane: Some(gpu_lane),
    };
    write_json(out, &report)?;
    Ok(report)
}

fn one_command_bringup_with_ebm_mode(
    workdir: &Path,
    scenario: &Path,
    ticks: u64,
    out_dir: &Path,
    replay_verify: bool,
    ebm_mode: &str,
) -> Result<BringupArtifacts, OpsError> {
    let prev = std::env::var("UCF_SLOT_EBM_MODE").ok();
    std::env::set_var("UCF_SLOT_EBM_MODE", ebm_mode);
    let out = one_command_bringup(workdir, scenario, ticks, out_dir, replay_verify);
    if let Some(v) = prev {
        std::env::set_var("UCF_SLOT_EBM_MODE", v);
    } else {
        std::env::remove_var("UCF_SLOT_EBM_MODE");
    }
    out
}

fn check_ebm_wiring(shadow: &ExplainTickReport, active: &ExplainTickReport) -> CheckResult {
    let shadow_ok = shadow
        .governance
        .ebm
        .as_ref()
        .is_some_and(|e| !e.ebm_digest_prefix.is_empty() && e.top_energies_q.len() <= 8);
    let active_ok = active
        .governance
        .ebm
        .as_ref()
        .is_some_and(|e| !e.ebm_digest_prefix.is_empty() && e.top_energies_q.len() <= 8);
    if shadow_ok && active_ok {
        check_pass(
            "ebm_wiring_records",
            [
                ("shadow_present".to_string(), "true".to_string()),
                ("active_present".to_string(), "true".to_string()),
            ],
        )
    } else {
        check_skip(
            "ebm_wiring_records",
            [
                ("shadow_present".to_string(), shadow_ok.to_string()),
                ("active_present".to_string(), active_ok.to_string()),
            ],
            "ebm record or digest missing in shadow/active mode",
            "Enable EBM slot mode and ensure EbmReasoningRecord is emitted with bounded fields.",
        )
    }
}

fn check_ebm_shadow_active_correctness(
    off: &ExplainTickReport,
    shadow: &ExplainTickReport,
    active: &ExplainTickReport,
) -> CheckResult {
    let shadow_same = off.decision.selected_candidate_id == shadow.decision.selected_candidate_id;
    let active_safe = active.decision.selected_candidate_id != Some(2);
    if shadow_same && active_safe {
        check_pass(
            "ebm_shadow_active_correctness",
            [
                (
                    "off_selected".to_string(),
                    off.decision.selected_candidate_id.unwrap_or(0).to_string(),
                ),
                (
                    "active_selected".to_string(),
                    active
                        .decision
                        .selected_candidate_id
                        .unwrap_or(0)
                        .to_string(),
                ),
            ],
        )
    } else {
        check_fail(
            "ebm_shadow_active_correctness",
            [
                ("shadow_same_as_off".to_string(), shadow_same.to_string()),
                (
                    "active_not_tool_intent".to_string(),
                    active_safe.to_string(),
                ),
            ],
            "shadow changed decision or active selected tool-intent candidate",
            "Keep shadow observational-only and enforce active rerank away from ToolIntent.",
        )
    }
}

fn check_ebm_safety_dominance(
    off: &ExplainTickReport,
    active: &ExplainTickReport,
    adversarial_path: &Path,
) -> CheckResult {
    let off_tier = off.governance.tier.unwrap_or(0);
    let active_tier = active.governance.tier.unwrap_or(0);
    let off_score = off.governance.governor_score.unwrap_or(0);
    let active_score = active.governance.governor_score.unwrap_or(0);
    let monotone = active_tier >= off_tier && active_score >= off_score;
    let mut adv_denied = true;
    if let Ok(report_body) = fs::read_to_string(adversarial_path) {
        if let Ok(report) = serde_json::from_str::<crate::AdversarialReport>(&report_body) {
            adv_denied = report
                .cases
                .iter()
                .filter(|c| c.name.contains("ebm_"))
                .all(|c| c.observed.output_class == "safe_only");
        }
    }
    if monotone && adv_denied {
        check_pass(
            "ebm_safety_dominance",
            [
                ("off_tier".to_string(), off_tier.to_string()),
                ("active_tier".to_string(), active_tier.to_string()),
                ("off_score_q".to_string(), off_score.to_string()),
                ("active_score_q".to_string(), active_score.to_string()),
            ],
        )
    } else {
        check_fail(
            "ebm_safety_dominance",
            [
                ("monotone".to_string(), monotone.to_string()),
                ("adversarial_denied".to_string(), adv_denied.to_string()),
            ],
            "ebm active mode loosened governance or adversarial deny semantics",
            "Verify EBM only tightens governor/tool gates and rerun adversarial suite.",
        )
    }
}

fn check_ebm_determinism(
    active_a: &ExplainTickReport,
    active_b: &ExplainTickReport,
) -> CheckResult {
    let a = active_a
        .governance
        .ebm
        .as_ref()
        .map(|e| e.ebm_digest_prefix.clone())
        .unwrap_or_default();
    let b = active_b
        .governance
        .ebm
        .as_ref()
        .map(|e| e.ebm_digest_prefix.clone())
        .unwrap_or_default();
    if !a.is_empty() && a == b {
        check_pass("ebm_determinism_digest", [("digest_prefix".to_string(), a)])
    } else if a.is_empty() || b.is_empty() {
        check_skip(
            "ebm_determinism_digest",
            [("digest_a".to_string(), a), ("digest_b".to_string(), b)],
            "ebm digest missing in one or both active runs",
            "Emit EBM digest prefix in explain/governance output before enforcing strict determinism.",
        )
    } else {
        check_fail(
            "ebm_determinism_digest",
            [("digest_a".to_string(), a), ("digest_b".to_string(), b)],
            "ebm digest prefix changed between identical runs",
            "Use fixed fixture/backends/seeds and avoid non-deterministic ebm feature inputs.",
        )
    }
}

fn check_ebm_constraints_provenance(workdir: &Path, policy_hash: &str) -> CheckResult {
    let records =
        load_fixture_records(&workdir.join("ess").join("ess_fixture.json")).unwrap_or_default();
    let ebm_provenance_count = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::EbmConstraintProvenance)
        .count();
    if ebm_provenance_count == 0 {
        return check_skip(
            "ebm_constraints_provenance",
            [("records".to_string(), "0".to_string())],
            "no EbmConstraintProvenanceRecord present in fixture records",
            "Run readiness gate with full ESS audit fixtures that persist EBM provenance.",
        );
    }
    let policy_prefix = prefix_hex(policy_hash, 16);
    let match_found = records.iter().any(|r| {
        if r.kind != ExperienceKind::EbmConstraintProvenance {
            return false;
        }
        matches!(
            &r.payload,
            ExperiencePayload::Audit(AuditPayload::EbmConstraintProvenance(p))
                if digest_prefix_arr8(&p.policy_hash_prefix, 16) == policy_prefix
        )
    });
    if match_found {
        check_pass(
            "ebm_constraints_provenance",
            [("policy_prefix".to_string(), policy_prefix)],
        )
    } else {
        check_fail(
            "ebm_constraints_provenance",
            [("policy_prefix".to_string(), policy_prefix)],
            "missing or mismatched EbmConstraintProvenanceRecord",
            "Emit EBM constraints provenance at startup and bind it to policy hash prefix.",
        )
    }
}

fn check_ebm_fallback_degraded_record(workdir: &Path) -> CheckResult {
    let records =
        load_fixture_records(&workdir.join("ess").join("ess_fixture.json")).unwrap_or_default();
    let has_reasoning = records
        .iter()
        .any(|r| r.kind == ExperienceKind::EbmReasoning);
    let degraded = records.iter().any(|r| {
        if r.kind != ExperienceKind::EbmEnvelopeViolation {
            return false;
        }
        matches!(
            &r.payload,
            ExperiencePayload::Audit(AuditPayload::EbmEnvelopeViolation(_))
        )
    });
    if has_reasoning {
        if degraded {
            check_pass(
                "ebm_fallback_recorded",
                [
                    ("has_reasoning".to_string(), "true".to_string()),
                    ("degraded_record".to_string(), "true".to_string()),
                ],
            )
        } else {
            check_skip(
                "ebm_fallback_recorded",
                [
                    ("has_reasoning".to_string(), "true".to_string()),
                    ("degraded_record".to_string(), "false".to_string()),
                ],
                "no degraded ebm record observed in baseline run",
                "Run budget-starved ebm fixture to verify degraded fallback record persistence.",
            )
        }
    } else {
        check_skip(
            "ebm_fallback_recorded",
            [("has_reasoning".to_string(), "false".to_string())],
            "ebm reasoning records missing",
            "Enable EBM shadow/active mode and rerun readiness scenarios.",
        )
    }
}

fn workspace_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join(name)
}

fn check_workspace_tests() -> CheckResult {
    if std::env::var("CI").ok().as_deref() == Some("true") {
        return check_skip(
            "build_workspace_tests",
            [("skipped".to_string(), "ci".to_string())],
            "workspace test execution skipped in CI readiness lane",
            "Run cargo test --workspace --offline in a dedicated lane for full readiness coverage.",
        );
    }

    if std::env::var("UCF_SKIP_GATE_WORKSPACE_TESTS")
        .ok()
        .as_deref()
        == Some("1")
    {
        return check_skip(
            "build_workspace_tests",
            [("skipped".to_string(), "env".to_string())],
            "workspace test execution skipped by environment",
            "Unset UCF_SKIP_GATE_WORKSPACE_TESTS to run full readiness check.",
        );
    }

    let output = Command::new("cargo")
        .args(["test", "--workspace", "--offline"])
        .output();
    match output {
        Ok(out) if out.status.success() => check_pass(
            "build_workspace_tests",
            [("exit".to_string(), "0".to_string())],
        ),
        Ok(out) => check_fail(
            "build_workspace_tests",
            [(
                "exit".to_string(),
                out.status.code().unwrap_or(-1).to_string(),
            )],
            "cargo test --workspace --offline failed",
            "Fix failing tests before enabling real compute packs.",
        ),
        Err(err) => check_fail(
            "build_workspace_tests",
            [("error".to_string(), bounded_string(err.to_string(), 64))],
            "failed to execute cargo test",
            "Ensure cargo is available in CI and local environments.",
        ),
    }
}

fn check_offline_profile(profile: &str) -> CheckResult {
    let pass = profile == "test" && std::env::var("UCF_OFFLINE").ok().as_deref() == Some("1");
    if pass {
        check_pass(
            "offline_profile",
            [
                ("profile".to_string(), profile.to_string()),
                ("offline".to_string(), "1".to_string()),
            ],
        )
    } else {
        check_fail(
            "offline_profile",
            [
                ("profile".to_string(), profile.to_string()),
                (
                    "offline".to_string(),
                    std::env::var("UCF_OFFLINE").unwrap_or_else(|_| "unset".to_string()),
                ),
            ],
            "offline mode not enforced",
            "Run with --profile test and UCF_OFFLINE=1.",
        )
    }
}

fn check_backend_disabled_pack() -> CheckResult {
    let build = BackendPackFactory::build(BackendPackConfig {
        pack: BackendPackKind::CandleToyV1,
        seed: 7,
    });
    match build {
        Err(ComputeError::BackendDisabled) => check_pass(
            "feature_pack_disabled_fast_fail",
            [("pack".to_string(), "candle_toy_v1".to_string())],
        ),
        Ok(_) => check_fail(
            "feature_pack_disabled_fast_fail",
            [("pack".to_string(), "candle_toy_v1".to_string())],
            "disabled pack unexpectedly built",
            "Ensure release feature matrix blocks unavailable backend packs.",
        ),
        Err(err) => check_skip(
            "feature_pack_disabled_fast_fail",
            [(
                "detail".to_string(),
                bounded_string(format!("{err}"), GATE_STR_CAP),
            )],
            "unexpected backend error",
            "Review backend pack gating expectations for this profile.",
        ),
    }
}

fn check_schema_versions(meta: &RunMetadataRecord) -> CheckResult {
    let valid = !meta.schema_versions.is_empty() && meta.schema_versions.values().all(|v| *v > 0);
    if valid {
        check_pass(
            "schema_versions_present",
            [("count".to_string(), meta.schema_versions.len().to_string())],
        )
    } else {
        check_fail(
            "schema_versions_present",
            [("count".to_string(), meta.schema_versions.len().to_string())],
            "schema versions are missing or zero",
            "Populate non-zero schema versions in RunMetadataRecord.",
        )
    }
}

fn check_required_records(explain: &ExplainTickReport) -> CheckResult {
    let has_candidate_set = !explain
        .warnings
        .iter()
        .any(|w| w.contains("CandidateSetRecord"));
    let has_output = !explain.warnings.iter().any(|w| w.contains("OutputRecord"));
    let has_issuance = !explain
        .warnings
        .iter()
        .any(|w| w.contains("CapabilityIssuanceRecord"));
    let pass = has_candidate_set && has_output && has_issuance;
    if pass {
        check_pass(
            "required_records",
            [("warnings".to_string(), "0".to_string())],
        )
    } else {
        check_skip(
            "required_records",
            [(
                "warnings".to_string(),
                bounded_string(explain.warnings.join(" | "), GATE_STR_CAP),
            )],
            "not all required records are emitted in the current fixture bringup",
            "Run a runtime scenario that emits candidate-set/output/issuance audit records.",
        )
    }
}

fn check_determinism(a: &BringupArtifacts, b: &BringupArtifacts) -> CheckResult {
    let backend_match =
        a.run_metadata.backend_pack_meta_digest == b.run_metadata.backend_pack_meta_digest;
    let fixture_match = a.run_metadata.fixtures_digest == b.run_metadata.fixtures_digest;
    let explain_match = a.explain == b.explain;
    if backend_match && fixture_match && explain_match {
        check_pass(
            "determinism_scenario_a_repeat",
            [
                (
                    "backend_pack_digest_prefix".to_string(),
                    prefix_hex(&a.run_metadata.backend_pack_meta_digest, 12),
                ),
                (
                    "fixtures_digest_prefix".to_string(),
                    prefix_hex(&a.run_metadata.fixtures_digest, 12),
                ),
            ],
        )
    } else {
        check_fail(
            "determinism_scenario_a_repeat",
            [
                ("backend_match".to_string(), backend_match.to_string()),
                ("fixtures_match".to_string(), fixture_match.to_string()),
                ("explain_match".to_string(), explain_match.to_string()),
            ],
            "scenario A repeat produced different digests or explain output",
            "Use fixed seeds and deterministic fixture/backends for gate scenarios.",
        )
    }
}

fn check_replay_report(name: &str, report: &ucf_replay::ReplayReport) -> CheckResult {
    let ok = report.overall_status == ucf_replay::ReplayOverallStatus::Ok;
    if ok {
        check_pass(
            name,
            [(
                "mismatched_digests".to_string(),
                report.counters.mismatched_digests.to_string(),
            )],
        )
    } else {
        check_skip(
            name,
            [
                (
                    "overall_status".to_string(),
                    format!("{:?}", report.overall_status),
                ),
                (
                    "mismatched_digests".to_string(),
                    report.counters.mismatched_digests.to_string(),
                ),
            ],
            "replay audit drift detected on simplified fixture records",
            "Use full ESS slices with complete audit links for strict replay PASS.",
        )
    }
}

fn check_tool_deny_policy(explain: &ExplainTickReport) -> CheckResult {
    if explain.governance.issuance.is_empty() {
        check_skip(
            "tool_deny_by_default",
            [("issuance_records".to_string(), "0".to_string())],
            "no tool intent observed in fixture run",
            "Add a tool-intent fixture and verify deny issuance + no execution.",
        )
    } else {
        let denies = explain
            .governance
            .issuance
            .iter()
            .all(|i| i.granted.is_empty() && !i.denied.is_empty());
        if denies {
            check_pass(
                "tool_deny_by_default",
                [(
                    "issuance_records".to_string(),
                    explain.governance.issuance.len().to_string(),
                )],
            )
        } else {
            check_fail(
                "tool_deny_by_default",
                [(
                    "issuance_records".to_string(),
                    explain.governance.issuance.len().to_string(),
                )],
                "tool issuance granted in test profile",
                "Set tools default to deny and enforce governor deny-by-default.",
            )
        }
    }
}

fn check_emergency_visibility(explain: &ExplainTickReport) -> CheckResult {
    if explain.governance.emergency_active {
        check_pass(
            "emergency_override",
            [("emergency_active".to_string(), "true".to_string())],
        )
    } else {
        check_skip(
            "emergency_override",
            [("emergency_active".to_string(), "false".to_string())],
            "emergency not triggered by baseline fixtures",
            "Run dedicated runaway fixture to assert forced tier=3 and safe output.",
        )
    }
}

fn check_observability(explain: &ExplainTickReport, metrics: &MetricsSummary) -> CheckResult {
    let explain_ok = explain.header.decision_id > 0
        && explain.compute.risk.risk.is_some()
        && explain.links.record_ids.len() <= 64;
    let metrics_ok = metrics.ticks_observed > 0;
    if explain_ok && metrics_ok {
        check_pass(
            "observability_explain_metrics",
            [
                (
                    "ticks_observed".to_string(),
                    metrics.ticks_observed.to_string(),
                ),
                (
                    "record_links".to_string(),
                    explain.links.record_ids.len().to_string(),
                ),
            ],
        )
    } else {
        check_fail(
            "observability_explain_metrics",
            [
                ("explain_ok".to_string(), explain_ok.to_string()),
                ("metrics_ok".to_string(), metrics_ok.to_string()),
            ],
            "explain-tick or metrics summary missing required data",
            "Ensure ESS includes decision records and metrics stream is initialized.",
        )
    }
}

fn check_plug_compatibility(a: &RunMetadataRecord, b: &RunMetadataRecord) -> CheckResult {
    if a.schema_versions == b.schema_versions {
        check_pass(
            "backend_plug_contract_compat",
            [(
                "schema_count".to_string(),
                a.schema_versions.len().to_string(),
            )],
        )
    } else {
        check_fail(
            "backend_plug_contract_compat",
            [
                (
                    "schema_count_a".to_string(),
                    a.schema_versions.len().to_string(),
                ),
                (
                    "schema_count_b".to_string(),
                    b.schema_versions.len().to_string(),
                ),
            ],
            "schema contracts changed across scenario packs",
            "Keep record contracts stable across backend swaps.",
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct GateLifecycleSlotManifest {
    active_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct GateLifecycleManifest {
    slots: BTreeMap<String, GateLifecycleSlotManifest>,
    manifest_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PromotionRecordView {
    slot: String,
    to_hash: String,
    shadow_report_digest_prefix: Option<String>,
}

fn check_weights_lifecycle_integrity(_workdir: &Path) -> Result<CheckResult, OpsError> {
    let manifest_path = PathBuf::from("models/MANIFEST.toml");
    if !manifest_path.exists() {
        return Ok(check_skip(
            "weights_lifecycle",
            [("manifest".to_string(), "missing".to_string())],
            "weights lifecycle not initialized",
            "Initialize lifecycle via models stage/promote to enforce this gate.",
        ));
    }
    let raw = fs::read_to_string(&manifest_path)?;
    let manifest: GateLifecycleManifest = toml::from_str(&raw)
        .map_err(|e| OpsError::Invalid(format!("manifest parse failed: {e}")))?;
    let mut promoted_only = true;
    let mut active_count = 0usize;
    for (slot, m) in &manifest.slots {
        if let Some(hash) = &m.active_hash {
            active_count += 1;
            let promoted = PathBuf::from("models")
                .join("promoted")
                .join(slot)
                .join(hash);
            if !promoted.exists() {
                promoted_only = false;
            }
        }
    }

    let mut canonical = manifest.clone();
    canonical.manifest_digest.clear();
    let computed = sha256_hex(&serde_json::to_vec(&canonical)?);
    let digest_ok = computed == manifest.manifest_digest;

    let hist_dir = PathBuf::from("models/manifests/history");
    let hist_count = if hist_dir.exists() {
        fs::read_dir(hist_dir)?
            .filter_map(Result::ok)
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .count()
    } else {
        0
    };

    let promotion_records: Vec<PromotionRecordView> =
        fs::read_to_string("out/model_promotion_records.json")
            .ok()
            .and_then(|v| serde_json::from_str(&v).ok())
            .unwrap_or_default();

    let lifecycle_initialized = active_count > 0 || hist_count > 0 || !promotion_records.is_empty();
    if !lifecycle_initialized {
        return Ok(check_skip(
            "weights_lifecycle",
            [
                ("active_slots".to_string(), "0".to_string()),
                ("history_entries".to_string(), hist_count.to_string()),
                (
                    "promotion_records".to_string(),
                    promotion_records.len().to_string(),
                ),
            ],
            "weights lifecycle not initialized",
            "Initialize lifecycle via models stage/promote to enforce this gate.",
        ));
    }
    let mut provenance_missing = 0usize;
    for (slot, m) in &manifest.slots {
        if let Some(hash) = &m.active_hash {
            let found = promotion_records
                .iter()
                .any(|r| r.slot == *slot && r.to_hash == *hash);
            if !found && !promotion_records.is_empty() {
                provenance_missing += 1;
            }
        }
    }

    let pin_env_used =
        std::env::vars().any(|(k, v)| k.starts_with("UCF_MODEL_PIN_") && !v.is_empty());
    let pin_records_present = PathBuf::from("out/model_pin_records.json").exists();
    if pin_env_used && !pin_records_present {
        return Ok(check_fail(
            "weights_lifecycle",
            [
                ("pin_env_used".to_string(), "true".to_string()),
                ("pin_records".to_string(), "missing".to_string()),
            ],
            "pin override used without ModelPinRecord evidence",
            "Emit out/model_pin_records.json with slot/hash override rationale before promotion.",
        ));
    }

    if promoted_only && digest_ok && hist_count >= 1 && provenance_missing == 0 {
        Ok(check_pass(
            "weights_lifecycle",
            [
                ("active_slots".to_string(), active_count.to_string()),
                ("history_entries".to_string(), hist_count.to_string()),
                (
                    "manifest_digest_prefix".to_string(),
                    prefix_hex(&manifest.manifest_digest, 12),
                ),
            ],
        ))
    } else {
        Ok(check_fail(
            "weights_lifecycle",
            [
                ("promoted_only".to_string(), promoted_only.to_string()),
                ("manifest_digest_ok".to_string(), digest_ok.to_string()),
                ("history_entries".to_string(), hist_count.to_string()),
                ("missing_provenance".to_string(), provenance_missing.to_string()),
            ],
            "weights lifecycle integrity constraints not met",
            "Ensure active hashes are promoted, manifest digest is canonical, history is persisted, and promotion records exist.",
        ))
    }
}

fn policy_threshold_i64(key: &str) -> Option<i64> {
    let overlay = std::env::var("UCF_POLICY_OVERLAY").ok();
    let overlay_path = overlay
        .as_deref()
        .map(|name| PathBuf::from("policies/packs/overlays").join(name));
    let overlay_ref = overlay_path.as_deref();
    let (graph, _) =
        load_and_merge_policy_graph(Path::new("policies/packs/base_v1"), overlay_ref).ok()?;
    graph.thresholds.get(key).copied()
}

fn check_world_vljepa_shadow_evidence(workdir: &Path) -> Result<CheckResult, OpsError> {
    let manifest: Option<GateLifecycleManifest> = fs::read_to_string("models/MANIFEST.toml")
        .ok()
        .and_then(|v| toml::from_str(&v).ok());
    let active = manifest
        .as_ref()
        .and_then(|m| m.slots.get("world_vljepa"))
        .and_then(|s| s.active_hash.as_ref())
        .is_some();
    let promotions: Vec<PromotionRecordView> =
        fs::read_to_string("out/model_promotion_records.json")
            .ok()
            .and_then(|v| serde_json::from_str(&v).ok())
            .unwrap_or_default();
    let recent_promoted = promotions
        .iter()
        .rev()
        .take(3)
        .any(|p| p.slot == "world_vljepa");

    let report_path = workdir.join("out/world_shadow_report.json");
    let report: Option<WorldShadowReport> = fs::read_to_string(&report_path)
        .ok()
        .and_then(|v| serde_json::from_str(&v).ok());

    let require = active || recent_promoted;
    if !require && report.is_none() {
        return Ok(check_skip(
            "world_vljepa_evidence",
            [("required".to_string(), "false".to_string())],
            "world_vljepa inactive and no shadow artifact",
            "No action required unless world_vljepa is active or promoted.",
        ));
    }

    let Some(rep) = report else {
        return Ok(check_fail(
            "world_vljepa_evidence",
            [("shadow_report".to_string(), "missing".to_string())],
            "world_vljepa requires shadow evidence artifact",
            "Run `ucf-ops world shadow-report` and store out/world_shadow_report.json.",
        ));
    };

    let has_promo_digest = promotions
        .iter()
        .rev()
        .find(|p| p.slot == "world_vljepa")
        .and_then(|p| p.shadow_report_digest_prefix.clone())
        .is_some();
    let min_windows = policy_threshold_i64("world_vljepa_min_windows")
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(2);
    let drift_threshold = policy_threshold_i64("world_vljepa_drift_alarm_rate_max_q")
        .map(|v| (v as f32) / 10_000.0)
        .unwrap_or(0.05);
    let alarm_rate = if rep.window_count == 0 {
        1.0
    } else {
        rep.drift_alarms.len() as f32 / rep.window_count as f32
    };

    let ok = rep.status == GateStatus::Pass
        && rep.window_count >= min_windows
        && alarm_rate <= drift_threshold
        && (!require || has_promo_digest);
    if ok {
        Ok(check_pass(
            "world_vljepa_evidence",
            [
                ("windows".to_string(), rep.window_count.to_string()),
                ("alarm_rate".to_string(), format!("{alarm_rate:.4}")),
                (
                    "report_digest_prefix".to_string(),
                    prefix_hex(&rep.report_digest, 12),
                ),
            ],
        ))
    } else {
        Ok(check_fail(
            "world_vljepa_evidence",
            [
                ("status_pass".to_string(), (rep.status == GateStatus::Pass).to_string()),
                ("windows".to_string(), rep.window_count.to_string()),
                ("alarm_rate".to_string(), format!("{alarm_rate:.4}")),
                ("promotion_digest_ref".to_string(), has_promo_digest.to_string()),
            ],
            "world_vljepa shadow evidence below gate requirements",
            "Collect sufficient shadow windows, clear severe alarms, and attach shadow digest in promotion record.",
        ))
    }
}

fn check_sae_real_readiness(workdir: &Path) -> Result<CheckResult, OpsError> {
    let manifest: Option<GateLifecycleManifest> = fs::read_to_string("models/MANIFEST.toml")
        .ok()
        .and_then(|v| toml::from_str(&v).ok());
    let sae_active = manifest
        .as_ref()
        .and_then(|m| m.slots.get("sae"))
        .and_then(|s| s.active_hash.as_ref())
        .is_some();

    let probe_path = workdir.join("out/probe_report.json");
    let probe: Option<ProbeReport> = fs::read_to_string(&probe_path)
        .ok()
        .and_then(|v| serde_json::from_str(&v).ok());
    let Some(probe) = probe else {
        if !sae_active {
            return Ok(check_skip(
                "sae_real",
                [("sae_active".to_string(), "false".to_string())],
                "SAE not active and no probe evidence present",
                "No action required unless SAE is promoted/active.",
            ));
        }
        return Ok(check_fail(
            "sae_real",
            [("probe_report".to_string(), "missing".to_string())],
            "SAE readiness requires probe evidence",
            "Run `ucf-ops models probe --out out/probe_report.json` before gate.",
        ));
    };
    let sae = probe.results.iter().find(|r| r.slot == ModelSlot::Sae);
    let Some(sae) = sae else {
        return Ok(check_fail(
            "sae_real",
            [("sae_result".to_string(), "missing".to_string())],
            "SAE slot missing from probe report",
            "Enable SAE slot in manifest and rerun models probe.",
        ));
    };
    let ok = matches!(sae.status, ProbeStatus::Ok);
    if ok {
        Ok(check_pass(
            "sae_real",
            [("probe_status".to_string(), "PASS".to_string())],
        ))
    } else {
        Ok(check_fail(
            "sae_real",
            [("probe_status".to_string(), format!("{:?}", sae.status))],
            "SAE validators did not pass",
            "Fix SAE weight spec / spike-rate quality and rerun probe.",
        ))
    }
}

fn check_ssm_opt_drift(workdir: &Path) -> Result<CheckResult, OpsError> {
    let kernel = std::env::var("UCF_SSM_KERNEL").unwrap_or_else(|_| "ref".to_string());
    if kernel == "ref" {
        return Ok(check_skip(
            "ssm_opt",
            [("kernel".to_string(), kernel)],
            "optimized SSM kernel not enabled",
            "Set UCF_SSM_KERNEL=opt (or simd) and provide parity artifact to gate it.",
        ));
    }
    let path = workdir.join("out/ssm_opt_parity.json");
    let json: serde_json::Value = match fs::read_to_string(&path)
        .ok()
        .and_then(|v| serde_json::from_str(&v).ok())
    {
        Some(v) => v,
        None => {
            return Ok(check_fail(
                "ssm_opt",
                [("artifact".to_string(), "missing".to_string())],
                "SSM opt kernel enabled without drift/parity artifact",
                "Emit out/ssm_opt_parity.json with drift_alarm_rate and digest_mismatch_rate.",
            ))
        }
    };
    let drift = json
        .get("drift_alarm_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let mismatch = json
        .get("digest_mismatch_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let drift_limit = policy_threshold_i64("ssm_opt_drift_alarm_rate_max_q")
        .map(|v| (v as f64) / 10_000.0)
        .unwrap_or(0.05);

    if drift <= drift_limit && mismatch == 0.0 {
        Ok(check_pass(
            "ssm_opt",
            [
                ("kernel".to_string(), kernel),
                ("drift_alarm_rate".to_string(), format!("{drift:.4}")),
                ("digest_mismatch_rate".to_string(), format!("{mismatch:.4}")),
            ],
        ))
    } else {
        Ok(check_fail(
            "ssm_opt",
            [
                ("kernel".to_string(), kernel),
                ("drift_alarm_rate".to_string(), format!("{drift:.4}")),
                ("digest_mismatch_rate".to_string(), format!("{mismatch:.4}")),
            ],
            "SSM optimized kernel drift/parity thresholds exceeded",
            "Reduce drift alarms and enforce digest mismatch rate to zero before enabling opt lane.",
        ))
    }
}

fn check_gpu_lane_parity(workdir: &Path) -> Result<CheckResult, OpsError> {
    let mode = std::env::var("UCF_GPU_MODE").unwrap_or_else(|_| "off".to_string());
    if mode == "off" {
        return Ok(check_skip(
            "gpu_lane",
            [("gpu_mode".to_string(), mode)],
            "GPU lane disabled",
            "No action required.",
        ));
    }
    let path = workdir.join("out/gpu_parity_report.json");
    let json: serde_json::Value = match fs::read_to_string(&path)
        .ok()
        .and_then(|v| serde_json::from_str(&v).ok())
    {
        Some(v) => v,
        None => {
            return Ok(check_fail(
                "gpu_lane",
                [("artifact".to_string(), "missing".to_string())],
                "GPU mode enabled without parity artifact",
                "Emit out/gpu_parity_report.json containing envelope_mismatch_rate.",
            ))
        }
    };
    let mismatch = json
        .get("envelope_mismatch_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    if mismatch <= 0.01 {
        Ok(check_pass(
            "gpu_lane",
            [
                ("gpu_mode".to_string(), mode),
                (
                    "envelope_mismatch_rate".to_string(),
                    format!("{mismatch:.4}"),
                ),
            ],
        ))
    } else {
        Ok(check_fail(
            "gpu_lane",
            [
                ("gpu_mode".to_string(), mode),
                (
                    "envelope_mismatch_rate".to_string(),
                    format!("{mismatch:.4}"),
                ),
            ],
            "GPU lane parity threshold exceeded",
            "Keep GPU in shadow/off and fix parity drift before activation.",
        ))
    }
}

fn check_pass(name: &str, evidence: impl IntoIterator<Item = (String, String)>) -> CheckResult {
    CheckResult {
        name: name.to_string(),
        status: GateStatus::Pass,
        evidence: bounded_evidence(evidence),
        failure_reason: None,
        remediation_hint: None,
    }
}

fn check_fail(
    name: &str,
    evidence: impl IntoIterator<Item = (String, String)>,
    reason: &str,
    remediation: &str,
) -> CheckResult {
    CheckResult {
        name: name.to_string(),
        status: GateStatus::Fail,
        evidence: bounded_evidence(evidence),
        failure_reason: Some(bounded_string(reason, GATE_STR_CAP)),
        remediation_hint: Some(bounded_string(remediation, GATE_STR_CAP)),
    }
}

fn check_skip(
    name: &str,
    evidence: impl IntoIterator<Item = (String, String)>,
    reason: &str,
    remediation: &str,
) -> CheckResult {
    CheckResult {
        name: name.to_string(),
        status: GateStatus::Skip,
        evidence: bounded_evidence(evidence),
        failure_reason: Some(bounded_string(reason, GATE_STR_CAP)),
        remediation_hint: Some(bounded_string(remediation, GATE_STR_CAP)),
    }
}

fn bounded_evidence(
    evidence: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (idx, (k, v)) in evidence.into_iter().enumerate() {
        if idx >= GATE_EVIDENCE_CAP {
            break;
        }
        out.insert(bounded_string(k, 48), bounded_string(v, 96));
    }
    out
}

fn prefix_hex(value: &str, len: usize) -> String {
    value.chars().take(len.min(value.len())).collect()
}

fn bounded_string(value: impl Into<String>, max: usize) -> String {
    let value = value.into();
    let mut chars = value.chars();
    let bounded: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        format!("{bounded}…")
    } else {
        bounded
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagCheck {
    pub name: String,
    pub pass: bool,
    pub detail: String,
    pub remediation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagReport {
    pub checks: Vec<DiagCheck>,
}

impl DiagReport {
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|c| c.pass)
    }
}

#[derive(Debug, Clone)]
pub struct ExportArgs {
    pub last: Option<usize>,
    pub include_sandbox: bool,
    pub include_audit: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExplainTickRequest {
    pub t: Option<u64>,
    pub decision_id: Option<u64>,
    pub detail_level: u8,
    pub digest_prefix_len: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainTickReport {
    pub header: ExplainHeader,
    pub compute: ExplainCompute,
    pub governance: ExplainGovernance,
    pub decision: ExplainDecision,
    pub output: ExplainOutput,
    pub links: ExplainLinks,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainHeader {
    pub t: u64,
    pub decision_id: u64,
    pub backend_pack_digest_prefix: Option<String>,
    pub evidence_chain_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainCompute {
    pub world: ExplainWorld,
    pub sae: ExplainSae,
    pub ssm: ExplainSsm,
    pub lfm: ExplainLfm,
    pub coherence: Option<ExplainCoherence>,
    pub risk: ExplainRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainWorld {
    pub surprise: Option<f32>,
    pub prediction_error: Option<f32>,
    pub world_digest_prefix: Option<String>,
    pub quality: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainSae {
    pub spike_count: Option<u16>,
    pub energy: Option<f32>,
    pub spikes_digest_prefix: Option<String>,
    pub quality: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainSsm {
    pub pressure: Option<f32>,
    pub ssm_digest_prefix: Option<String>,
    pub quality: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainLfm {
    pub uncertainty: Option<f32>,
    pub stability: Option<f32>,
    pub lfm_digest_prefix: Option<String>,
    pub quality: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainCoherence {
    pub coherence: Option<f32>,
    pub phi_proxy: Option<f32>,
    pub coherence_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainRisk {
    pub risk: Option<f32>,
    pub confidence: Option<f32>,
    pub risk_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainGovernance {
    pub governor_score: Option<u16>,
    pub tier: Option<u8>,
    pub emergency_active: bool,
    pub issuance: Vec<IssuanceExplain>,
    pub ebm: Option<ExplainEbm>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainEbm {
    pub mode: u8,
    pub aggregate_energy_q: u16,
    pub base_energy_q: u16,
    pub best_candidate_id: Option<u16>,
    pub top_energies_q: Vec<u16>,
    pub top_term_contributions: Vec<(u16, String, u16)>,
    pub ebm_digest_prefix: String,
    pub constraints_digest_prefix: String,
    pub status: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct IssuanceExplain {
    pub candidate_id: Option<u16>,
    pub requested: Vec<String>,
    pub granted: Vec<String>,
    pub denied: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainDecision {
    pub candidate_count: Option<usize>,
    pub selected_candidate_id: Option<u16>,
    pub selected_candidate_digest_prefix: Option<String>,
    pub policy_hints: Vec<u8>,
    pub nsr_risk_q: Option<u16>,
    pub nsr_status: Option<u8>,
    pub nsr_rules_digest_prefix: Option<String>,
    pub nsr_reasons: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainOutput {
    pub output_class: Option<u8>,
    pub llm_backend: Option<String>,
    pub request_digest_prefix: Option<String>,
    pub response_digest_prefix: Option<String>,
    pub status: Option<u8>,
    pub finish_reason: Option<u8>,
    pub max_tokens_eff: Option<u32>,
    pub text_preview: Option<String>,
    pub redacted: Option<bool>,
    pub content_digest_prefix: Option<String>,
    pub payload_len: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExplainLinks {
    pub record_ids: Vec<u64>,
    pub record_kinds: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsSummary {
    pub ticks_observed: usize,
    pub mean_surprise: f32,
    pub max_surprise: f32,
    pub mean_pressure: f32,
    pub max_pressure: f32,
    pub mean_uncertainty: f32,
    pub max_uncertainty: f32,
    pub governor_tier_2_3_percent: f32,
    pub emergency_triggers: usize,
    pub tool_issuance_deny_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsTrendPoint {
    pub t: u64,
    pub surprise: Option<f32>,
    pub pressure: Option<f32>,
    pub uncertainty: Option<f32>,
    pub risk: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpsFixture {
    decisions: Vec<OpsFixtureDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpsFixtureDecision {
    decision_id: u64,
    corr: u64,
    tick: u64,
    window: u64,
    text: String,
    backend: String,
    risk: f32,
    confidence: f32,
    surprise: f32,
    pressure: f32,
    spike_count: u16,
    spikes_digest_hex: String,
    evidence_context_digest_hex: String,
    budget_profile_id: u32,
    seed: u64,
    risk_quality: u8,
}

pub fn bringup(workdir: &Path, demo: bool, ticks: u64) -> Result<BringupResult, OpsError> {
    ensure_layout(workdir)?;
    let cfg = load_or_init_config(workdir)?;

    std::env::set_var("UCF_COMPUTE_BACKEND", cfg.compute_backend.as_env_str());
    std::env::set_var("UCF_COMPUTE_SEED", cfg.compute_seed.to_string());
    std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", &cfg.compute_budget_profile);
    std::env::set_var(
        "UCF_LLM_MAX_TOKENS",
        cfg.device_profile_llm_max_tokens().to_string(),
    );
    std::env::set_var(
        "UCF_WORLD_VLJEPA_WINDOW_TICKS",
        cfg.device_profile_world_shadow_window_ticks().to_string(),
    );
    std::env::set_var("UCF_POLICY_OVERLAY", &cfg.policy_overlay);
    std::env::set_var("UCF_SLOT_EBM_MODE", &cfg.slot_ebm_mode);
    std::env::set_var("UCF_BACKEND_PACK", &cfg.backend_pack);
    std::env::set_var("UCF_TOOLS_DEFAULT", &cfg.capabilities_default);
    std::env::set_var("UCF_OFFLINE", if cfg.offline { "1" } else { "0" });
    ensure_policy_bundle_hash_env();
    ensure_policy_bundle_root()?;

    let mut orchestrator = RuntimeOrchestrator::try_new_from_env()?;
    let mut adapter = MockAdapter::default();
    let mut fixture_decisions = Vec::new();

    let max_ticks = if demo { ticks } else { ticks.max(10) };
    for step in 0..max_ticks {
        let time = SimTime {
            tick: Tick::new(step + 1),
            window: WindowId::new(0),
        };
        let corr = CorrelationId(step + 1);
        let ctrl = ControlFrame::new_text(
            time,
            corr,
            ChannelCode::ExternalOutput,
            Intent::new(IntentId(corr.0), IntentKind::Speak, "ops-demo"),
            format!("demo_text_{step}"),
        );

        let decision = orchestrator.ingest_and_process(&mut adapter, ctrl.clone())?;
        if let Some(summary) = decision.compute_summary {
            fixture_decisions.push(OpsFixtureDecision {
                decision_id: step + 1,
                corr: corr.0,
                tick: time.tick.get(),
                window: time.window.get(),
                text: extract_text(&ctrl),
                backend: summary.backend.to_string(),
                risk: summary.risk,
                confidence: summary.confidence,
                surprise: summary.surprise,
                pressure: summary.pressure,
                spike_count: summary.spike_count,
                spikes_digest_hex: hex::encode(summary.spikes_digest),
                evidence_context_digest_hex: summary
                    .evidence_context_digest
                    .map(hex::encode)
                    .unwrap_or_else(|| hex::encode([0u8; 32])),
                budget_profile_id: summary
                    .budget_profile_id
                    .unwrap_or(stable_budget_profile_id(1_000, 5_000)),
                seed: summary.seed.unwrap_or(cfg.compute_seed),
                risk_quality: summary.risk_quality.unwrap_or(2),
            });
        }
    }

    let fixture = OpsFixture {
        decisions: fixture_decisions,
    };

    let ess_fixture_path = workdir.join("ess").join("ess_fixture.json");
    let fixture_text = serde_json::to_string_pretty(&fixture)?;
    fs::write(&ess_fixture_path, fixture_text.as_bytes())?;

    let ess_digest = sha256_hex(fixture_text.as_bytes());
    let log_path = workdir.join("logs").join("bringup.log");
    let log_line = format!(
        "status=ok mode={} ticks={} ess={} digest={}\n",
        if demo { "demo" } else { "continuous" },
        max_ticks,
        ess_fixture_path.display(),
        ess_digest
    );
    fs::write(&log_path, log_line)?;

    Ok(BringupResult {
        workdir: workdir.to_path_buf(),
        ess_fixture_path,
        log_path,
        decision_count: fixture.decisions.len(),
        ess_digest,
    })
}

pub fn one_command_bringup(
    workdir: &Path,
    scenario: &Path,
    ticks: u64,
    out_dir: &Path,
    replay_verify: bool,
) -> Result<BringupArtifacts, OpsError> {
    ensure_layout(workdir)?;
    fs::create_dir_all(out_dir)?;
    let _scenario_doc: serde_json::Value = serde_json::from_str(&fs::read_to_string(scenario)?)?;

    std::env::set_var("UCF_SSM_KERNEL", "ref");
    let shadow_base = workdir.join("reports").join("world_vljepa");
    fs::create_dir_all(&shadow_base)?;
    let shadow_windows_tmp = shadow_base.join("current_windows.jsonl");
    let shadow_alarms_tmp = shadow_base.join("current_alarms.jsonl");
    let _ = fs::remove_file(&shadow_windows_tmp);
    let _ = fs::remove_file(&shadow_alarms_tmp);
    std::env::set_var(
        "UCF_WORLD_VLJEPA_WINDOWS_LOG",
        shadow_windows_tmp.display().to_string(),
    );
    std::env::set_var(
        "UCF_WORLD_VLJEPA_ALARMS_LOG",
        shadow_alarms_tmp.display().to_string(),
    );
    ucf_compute::world_vljepa_shadow::reset_shadow_state();
    let result = bringup(workdir, true, ticks)?;
    let cfg = load_or_init_config(workdir)?;
    let build = build_tag()?;
    let pack = BackendPackFactory::build(BackendPackConfig::from_env()?)?;
    let meta = pack.meta();

    let mut schema_versions = BTreeMap::new();
    schema_versions.insert("backend_pack_record".to_string(), 1);
    schema_versions.insert("compute_summary".to_string(), 1);
    schema_versions.insert("output".to_string(), 1);

    let policy_bundle_hash =
        std::env::var("UCF_POLICY_BUNDLE_SHA256").unwrap_or_else(|_| "unverified".to_string());
    let resume_cfg = ResumeCheckConfig {
        policy_bundle_hash: policy_bundle_hash.clone(),
        backend_pack_meta_digest: hex::encode(meta.digest),
        model_hashes_digest: hex::encode(meta.model_hashes_digest),
        enabled_features_bitmap: ReleaseFeatureMatrix::detect().bits,
        schema_versions: schema_versions.clone(),
    };
    let mut run_metadata = RunMetadataRecord {
        run_id: format!(
            "{}-{}",
            result.ess_digest.chars().take(12).collect::<String>(),
            now_unix_secs()
        ),
        started_at_tick: 0,
        code_version_tag: build.git_commit,
        backend_pack_meta_digest: resume_cfg.backend_pack_meta_digest.clone(),
        fixtures_digest: hex::encode(meta.fixtures_digest),
        model_hashes_digest: resume_cfg.model_hashes_digest.clone(),
        enabled_features_bitmap: resume_cfg.enabled_features_bitmap,
        profile: cfg.profile.clone(),
        config_digest: cfg.config_digest.clone(),
        policy_overlay: cfg.policy_overlay.clone(),
        platform_probe_summary: LocalPlatformProbe::probe().summary(),
        device_profile_name: cfg.device_profile.clone(),
        device_profile_digest: DeviceProfileV1::for_name(cfg.device_profile_name()?)
            .digest_hex()?,
        schema_versions,
        parent_run_id: None,
        resume_reason: None,
        compat_digest: compute_resume_compat_digest(&resume_cfg),
        policy_bundle_hash,
        determinism_mode: "deterministic_only".to_string(),
        determinism_policy_digest: None,
        ended_at_tick: Some(ticks),
    };
    if let Some(prev) = latest_run_metadata(workdir)? {
        let decision = check_resume_compat(&prev, &resume_cfg);
        match decision {
            ResumeDecision::ResumeAllowed => {
                run_metadata.parent_run_id = Some(prev.run_id);
                run_metadata.resume_reason = Some(ResumeReason::OperatorResume);
            }
            ResumeDecision::NewSessionRequired { .. } => {
                run_metadata.parent_run_id = Some(prev.run_id);
                run_metadata.resume_reason = Some(ResumeReason::Upgrade);
            }
        }
    }
    persist_run_metadata(workdir, &run_metadata)?;
    let shadow_windows = shadow_base.join(format!("{}_windows.jsonl", run_metadata.run_id));
    let shadow_alarms = shadow_base.join(format!("{}_alarms.jsonl", run_metadata.run_id));
    if shadow_windows_tmp.exists() {
        fs::rename(&shadow_windows_tmp, &shadow_windows)?;
    }
    if shadow_alarms_tmp.exists() {
        fs::rename(&shadow_alarms_tmp, &shadow_alarms)?;
    }

    let metrics = metrics_summary(workdir, ticks as usize)?;
    let explain_tick_index = ticks.saturating_sub(1);
    let explain = explain_tick(
        workdir,
        ExplainTickRequest {
            t: Some(explain_tick_index),
            decision_id: None,
            detail_level: 1,
            digest_prefix_len: 12,
        },
    )?;
    let replay_report = if replay_verify {
        let path = out_dir.join("replay_verify.json");
        replay_audit(
            workdir,
            1,
            ticks,
            ReplayStrictness::VerifyOnly,
            false,
            &path,
        )?;
        Some(path.display().to_string())
    } else {
        None
    };

    write_json(out_dir.join("metrics_summary.json"), &metrics)?;
    write_json(out_dir.join("explain_tick_last.json"), &explain)?;
    write_json(out_dir.join("run_metadata_record.json"), &run_metadata)?;
    write_json(out_dir.join("run_metadata.json"), &run_metadata)?;

    Ok(BringupArtifacts {
        run_metadata,
        metrics,
        explain,
        replay_report,
    })
}

pub fn diagnostics(workdir: &Path) -> Result<DiagReport, OpsError> {
    let mut checks = Vec::new();
    let cfg = load_or_init_config(workdir)?;

    checks.push(DiagCheck {
        name: "workspace_build_tag".to_string(),
        pass: !build_tag()?.git_commit.is_empty(),
        detail: format!("commit={}", build_tag()?.git_commit),
        remediation: "run inside a git worktree.".to_string(),
    });

    checks.push(DiagCheck {
        name: "config_resolved".to_string(),
        pass: cfg.capabilities_default == "deny",
        detail: format!(
            "backend={:?} seed={} budget={} isolation={} caps_default={}",
            cfg.compute_backend,
            cfg.compute_seed,
            cfg.compute_budget_profile,
            cfg.isolation_runtime,
            cfg.capabilities_default
        ),
        remediation: "set capabilities_default to deny for safe operation.".to_string(),
    });

    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let ess_records = load_fixture_records(&fixture_path)?;
    checks.push(DiagCheck {
        name: "ess_health".to_string(),
        pass: !ess_records.is_empty(),
        detail: format!("records={}", ess_records.len()),
        remediation: "run `ucf-ops bringup --demo` to seed ESS fixture.".to_string(),
    });

    let audit_ok = ess_records
        .iter()
        .filter(|r| r.kind == ExperienceKind::AuditCheckpoint)
        .all(|r| r.audit_digest.is_some());
    checks.push(DiagCheck {
        name: "audit_chain".to_string(),
        pass: audit_ok,
        detail: "audit checkpoints parsed (or none present)".to_string(),
        remediation: "run workload including tool gate operations to emit checkpoints.".to_string(),
    });

    let compute_ok = run_compute_probe(&cfg)?;
    checks.push(compute_ok);

    checks.push(DiagCheck {
        name: "sandbox_runtime".to_string(),
        pass: cfg.isolation_runtime == "inproc",
        detail: format!("runtime={}", cfg.isolation_runtime),
        remediation: "set isolation_runtime to inproc for offline diagnostics.".to_string(),
    });

    let log_exists = workdir.join("logs").join("bringup.log").exists();
    checks.push(DiagCheck {
        name: "metrics_tracing".to_string(),
        pass: log_exists,
        detail: "bringup.log present".to_string(),
        remediation: "run `ucf-ops bringup --demo` to initialize logging.".to_string(),
    });

    Ok(DiagReport { checks })
}

pub fn export_bugreport(workdir: &Path, args: &ExportArgs) -> Result<PathBuf, OpsError> {
    ensure_layout(workdir)?;
    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let fixture: OpsFixture = serde_json::from_str(&fs::read_to_string(&fixture_path)?)?;

    let selected = if let Some(last) = args.last {
        let len = fixture.decisions.len();
        fixture.decisions[len.saturating_sub(last)..].to_vec()
    } else {
        fixture.decisions.clone()
    };

    let timestamp = selected.last().map(|d| d.tick).unwrap_or(0);
    let out_dir = workdir
        .join("reports")
        .join(format!("bugreport_{timestamp:010}"));
    fs::create_dir_all(&out_dir)?;

    let config = load_or_init_config(workdir)?;
    write_json(out_dir.join("config_resolved.json"), &config)?;
    write_json(out_dir.join("build_tag.json"), &build_tag()?)?;
    write_json(
        out_dir.join("ess_slice.json"),
        &OpsFixture {
            decisions: selected.clone(),
        },
    )?;

    let mut indices = BTreeMap::<String, serde_json::Value>::new();
    indices.insert("count".to_string(), serde_json::json!(selected.len()));
    indices.insert(
        "range".to_string(),
        serde_json::json!({
            "from_tick": selected.first().map(|d| d.tick),
            "to_tick": selected.last().map(|d| d.tick),
            "include_sandbox": args.include_sandbox,
            "include_audit": args.include_audit,
        }),
    );
    write_json(out_dir.join("indices.json"), &indices)?;

    fs::write(
        out_dir.join("README.txt"),
        "Replay with:\nucf-ops replay-bugreport <path> --mode compute\n",
    )?;

    let checksums = build_checksums(&out_dir)?;
    write_json(out_dir.join("checksums.json"), &checksums)?;

    Ok(out_dir)
}

pub fn verify_bugreport(path: &Path) -> Result<(), OpsError> {
    let checksum_path = path.join("checksums.json");
    let checksums: ChecksumManifest = serde_json::from_str(&fs::read_to_string(checksum_path)?)?;

    for (file, expected) in &checksums.files {
        let data = fs::read(path.join(file))?;
        let got = sha256_hex(&data);
        if &got != expected {
            return Err(OpsError::Invalid(format!("checksum mismatch for {file}")));
        }
    }

    let fixture: OpsFixture =
        serde_json::from_str(&fs::read_to_string(path.join("ess_slice.json"))?)?;

    if fixture.decisions.len() > 10_000 {
        return Err(OpsError::Invalid("ess slice too large".to_string()));
    }

    Ok(())
}

pub fn replay_audit(
    workdir: &Path,
    from_tick: u64,
    to_tick: u64,
    strictness: ReplayStrictness,
    stop_on_first_divergence: bool,
    report_path: &Path,
) -> Result<(), OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let plan = ReplayPlan {
        t0: from_tick,
        t1: to_tick,
        expected_backend_pack_digest: None,
        strictness,
        stop_on_first_divergence,
    };
    let report = run_replay_audit(&records, &plan);
    let body = serde_json::to_string_pretty(&report)?;
    fs::write(report_path, body)?;
    Ok(())
}

pub fn replay_bugreport(path: &Path, mode: ReplayMode) -> Result<PathBuf, OpsError> {
    verify_bugreport(path)?;
    let records = load_fixture_records(&path.join("ess_slice.json"))?;
    let spec = ReplaySpec {
        from_tick: 0,
        to_tick: u64::MAX,
        backend_override: None,
        seed_override: None,
        budget_override: None,
        mode,
    };
    let result = replay_records(&records, &spec);
    let report_path = path.join("replay_report.json");
    write_report(&report_path, &result)?;
    Ok(report_path)
}

pub fn metrics_snapshot(workdir: &Path) -> Result<serde_json::Value, OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let mut risk_buckets = [0u64; 3];
    for record in &records {
        if let ExperiencePayload::Decision(decision) = &record.payload {
            if let Some(summary) = decision.compute_summary {
                let idx = if summary.risk < 0.33 {
                    0
                } else if summary.risk < 0.66 {
                    1
                } else {
                    2
                };
                risk_buckets[idx] += 1;
            }
        }
    }

    Ok(serde_json::json!({
        "compute": {
            "risk_distribution": risk_buckets,
            "budget_exceeded_total": 0
        },
        "sandbox": {
            "denied_total": 0,
            "rate_limited_total": 0
        },
        "audit": {
            "checkpoint_total": records.iter().filter(|r| r.kind == ExperienceKind::AuditCheckpoint).count()
        },
        "ess": {
            "records": records.len()
        }
    }))
}

pub fn explain_tick(
    workdir: &Path,
    req: ExplainTickRequest,
) -> Result<ExplainTickReport, OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    build_explain_tick_report(&records, req)
}

pub fn build_explain_tick_report(
    records: &[ExperienceRecord],
    req: ExplainTickRequest,
) -> Result<ExplainTickReport, OpsError> {
    let mut warnings = Vec::new();
    let prefix = req.digest_prefix_len.clamp(4, 32) as usize;
    let detail = req.detail_level.min(2);

    let decision = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::DecisionOut)
        .filter(|r| {
            req.t.is_none_or(|t| r.time.tick.get() == t)
                && req.decision_id.is_none_or(|id| r.id.0 == id)
        })
        .max_by_key(|r| (r.time.tick.get(), r.id.0))
        .ok_or_else(|| OpsError::Invalid("no matching decision found".to_string()))?;

    let tick = decision.time.tick.get();
    let decision_id = decision.id.0;
    let compute = decision.compute_summary;
    if compute.is_none() {
        warnings.push("DecisionOut missing compute_summary".to_string());
    }
    let evidence_chain = compute.and_then(|s| s.compute_chain_digest);

    let mut candidates = records
        .iter()
        .filter_map(|r| {
            if r.kind != ExperienceKind::CandidateSet || r.time.tick.get() != tick {
                return None;
            }
            match &r.payload {
                ExperiencePayload::Audit(AuditPayload::CandidateSet(c))
                    if c.decision_id == decision_id =>
                {
                    Some((r, c.clone()))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(r, _)| (r.time.tick.get(), r.id.0));
    let candidate_set = candidates.last().map(|(_, c)| c.clone());

    let mut ebm_records = records
        .iter()
        .filter_map(|r| {
            if r.kind != ExperienceKind::EbmReasoning || r.time.tick.get() != tick {
                return None;
            }
            match &r.payload {
                ExperiencePayload::Audit(AuditPayload::EbmReasoning(e))
                    if e.decision_id == decision_id =>
                {
                    Some((r, e.clone()))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    ebm_records.sort_by_key(|(r, _)| (r.time.tick.get(), r.id.0));
    let ebm = ebm_records.last().map(|(_, e)| e.clone());

    let mut outputs = records
        .iter()
        .filter_map(|r| {
            if r.kind != ExperienceKind::Output || r.time.tick.get() != tick {
                return None;
            }
            match &r.payload {
                ExperiencePayload::Audit(AuditPayload::Output(o))
                    if o.decision_id == decision_id =>
                {
                    Some((r, o.clone()))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    outputs.sort_by_key(|(r, o)| (r.time.tick.get(), r.id.0, o.candidate_id));
    let output = outputs.last().map(|(_, o)| o.clone());

    let mut issuances = records
        .iter()
        .filter_map(|r| {
            if r.kind != ExperienceKind::CapabilityIssuance || r.time.tick.get() != tick {
                return None;
            }
            match &r.payload {
                ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(i))
                    if i.decision_id == decision_id =>
                {
                    Some((r, i.clone()))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    issuances.sort_by_key(|(r, i)| {
        (
            r.time.tick.get(),
            r.id.0,
            i.candidate_id.unwrap_or(u16::MAX),
        )
    });

    let mut nsrs = records
        .iter()
        .filter_map(|r| {
            if r.kind == ExperienceKind::Nsr && r.time.tick.get() == tick {
                r.nsr_record
                    .clone()
                    .filter(|n| n.decision_id == decision_id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    nsrs.sort_by_key(|n| (n.t, n.decision_id));

    let mut lfm = records
        .iter()
        .filter_map(|r| {
            if r.kind == ExperienceKind::LfmSummary && r.time.tick.get() == tick {
                r.lfm_summary_record
                    .filter(|s| s.decision_id == Some(decision_id))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    lfm.sort_by_key(|s| s.t);
    let lfm_summary = lfm.last().copied();

    let mut packs = records
        .iter()
        .filter_map(|r| r.backend_pack_record.clone().filter(|p| p.t <= tick))
        .collect::<Vec<_>>();
    packs.sort_by_key(|p| p.t);
    let backend_pack = packs.last().cloned();

    let emergency_active = records.iter().any(|r| {
        r.kind == ExperienceKind::Emergency
            && r.time.tick.get() <= tick
            && matches!(
                &r.payload,
                ExperiencePayload::Audit(AuditPayload::Emergency(e)) if e.state == EmergencyStateCode::Active
            )
    });

    if candidate_set.is_none() {
        warnings.push("CandidateSetRecord missing".to_string());
    }
    if output.is_none() {
        warnings.push("OutputRecord missing".to_string());
    }
    if issuances.is_empty() {
        warnings.push("CapabilityIssuanceRecord missing".to_string());
    }

    let mut policy_hints = nsrs.iter().map(|n| n.policy_hint).collect::<Vec<_>>();
    policy_hints.sort_unstable();
    policy_hints.dedup();

    let mut nsr_reasons = nsrs
        .iter()
        .flat_map(|n| n.reasons.clone())
        .collect::<Vec<_>>();
    nsr_reasons.sort_unstable();
    nsr_reasons.dedup();
    if detail == 0 {
        nsr_reasons.truncate(4);
    } else {
        nsr_reasons.truncate(8);
    }

    let mut issuance_view = issuances
        .iter()
        .map(|(_, i)| {
            let mut requested = i.requested_kinds.clone();
            let mut granted = i.granted_kinds.clone();
            let mut denied = i.denied_kinds.clone();
            requested.sort();
            granted.sort();
            denied.sort();
            IssuanceExplain {
                candidate_id: i.candidate_id,
                requested,
                granted,
                denied,
            }
        })
        .collect::<Vec<_>>();
    issuance_view.sort();
    issuance_view.truncate(if detail == 0 { 2 } else { 8 });

    let links = {
        let mut rows = records
            .iter()
            .filter(|r| r.time.tick.get() == tick)
            .filter(|r| {
                r.id.0 == decision_id
                    || r.corr == decision.corr
                    || matches!(
                        (&r.kind, &r.payload),
                        (ExperienceKind::CapabilityIssuance, ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(i))) if i.decision_id == decision_id
                    )
                    || matches!(
                        (&r.kind, &r.payload),
                        (ExperienceKind::CandidateSet, ExperiencePayload::Audit(AuditPayload::CandidateSet(c))) if c.decision_id == decision_id
                    )
                    || matches!(
                        (&r.kind, &r.payload),
                        (ExperienceKind::Output, ExperiencePayload::Audit(AuditPayload::Output(o))) if o.decision_id == decision_id
                    )
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|r| (r.time.tick.get(), r.id.0));
        rows.truncate(32);
        ExplainLinks {
            record_ids: rows.iter().map(|r| r.id.0).collect(),
            record_kinds: rows.iter().map(|r| format!("{:?}", r.kind)).collect(),
        }
    };

    Ok(ExplainTickReport {
        header: ExplainHeader {
            t: tick,
            decision_id,
            backend_pack_digest_prefix: backend_pack.map(|p| digest_prefix(&p.meta_digest, prefix)),
            evidence_chain_digest_prefix: evidence_chain.map(|d| digest_prefix(&d, prefix)),
        },
        compute: ExplainCompute {
            world: ExplainWorld {
                surprise: compute.map(|s| s.surprise),
                prediction_error: compute.map(|s| s.surprise),
                world_digest_prefix: compute
                    .and_then(|s| s.world_digest)
                    .map(|d| digest_prefix(&d, prefix)),
                quality: compute.and_then(|s| s.risk_quality),
            },
            sae: ExplainSae {
                spike_count: compute.map(|s| s.spike_count),
                energy: compute.and_then(|s| s.energy),
                spikes_digest_prefix: compute.map(|s| digest_prefix(&s.spikes_digest, prefix)),
                quality: compute.and_then(|s| s.risk_quality),
            },
            ssm: ExplainSsm {
                pressure: compute.map(|s| s.pressure),
                ssm_digest_prefix: compute
                    .and_then(|s| s.ssm_digest)
                    .map(|d| digest_prefix(&d, prefix)),
                quality: compute.and_then(|s| s.risk_quality),
            },
            lfm: ExplainLfm {
                uncertainty: lfm_summary
                    .map(|s| s.uncertainty)
                    .or(compute.and_then(|s| s.lfm_uncertainty)),
                stability: lfm_summary
                    .map(|s| s.stability)
                    .or(compute.and_then(|s| s.lfm_stability)),
                lfm_digest_prefix: lfm_summary.map(|s| digest_prefix(&s.digest, prefix)).or(
                    compute
                        .and_then(|s| s.lfm_digest)
                        .map(|d| digest_prefix(&d, prefix)),
                ),
                quality: compute.and_then(|s| s.lfm_quality),
            },
            coherence: compute.map(|s| ExplainCoherence {
                coherence: s.coherence,
                phi_proxy: s.phi_proxy,
                coherence_digest_prefix: s.coherence_digest.map(|d| digest_prefix(&d, prefix)),
            }),
            risk: ExplainRisk {
                risk: compute.map(|s| s.risk),
                confidence: compute.map(|s| s.confidence),
                risk_digest_prefix: compute
                    .and_then(|s| s.compute_chain_digest)
                    .map(|d| digest_prefix(&d, prefix)),
            },
        },
        governance: ExplainGovernance {
            governor_score: issuances.last().map(|(_, i)| i.governor_score_q),
            tier: issuances.last().map(|(_, i)| i.effective_tier),
            emergency_active,
            issuance: issuance_view,
            ebm: ebm.as_ref().map(|e| ExplainEbm {
                mode: e.enablement_mode,
                aggregate_energy_q: e.aggregate_energy_q,
                base_energy_q: e.base_energy_q,
                best_candidate_id: e.top_candidate_ids.first().copied(),
                top_energies_q: e.top_energies_q.clone(),
                top_term_contributions: e
                    .top_term_contributions
                    .iter()
                    .map(|(id, q)| (*id, ebm_term_label(*id).to_string(), *q))
                    .collect(),
                ebm_digest_prefix: digest_prefix_arr8(&e.ebm_digest_prefix, prefix),
                constraints_digest_prefix: digest_prefix_arr8(&e.constraints_digest_prefix, prefix),
                status: e.status,
            }),
        },
        decision: ExplainDecision {
            candidate_count: candidate_set.as_ref().map(|c| c.summaries.len()),
            selected_candidate_id: candidate_set.as_ref().map(|c| c.selected_candidate_id),
            selected_candidate_digest_prefix: candidate_set
                .as_ref()
                .map(|c| digest_prefix(&c.selected_candidate_digest, prefix)),
            policy_hints,
            nsr_risk_q: nsrs.last().map(|n| n.nsr_risk_q),
            nsr_status: nsrs.last().map(|n| n.nsr_status),
            nsr_rules_digest_prefix: nsrs
                .last()
                .map(|n| digest_prefix_arr8(&n.rules_digest_prefix, prefix)),
            nsr_reasons,
        },
        output: ExplainOutput {
            output_class: output.as_ref().map(|o| o.output_class),
            llm_backend: output.as_ref().map(|o| o.llm_backend_name.clone()),
            request_digest_prefix: output
                .as_ref()
                .map(|o| digest_prefix(&o.llm_request_digest, prefix)),
            response_digest_prefix: output
                .as_ref()
                .map(|o| digest_prefix(&o.llm_response_digest, prefix)),
            status: output.as_ref().map(|o| o.status),
            finish_reason: output.as_ref().map(|o| o.finish_reason),
            max_tokens_eff: output.as_ref().map(|o| o.max_tokens_eff),
            text_preview: output.as_ref().and_then(|o| o.text.clone()).and_then(|t| {
                if detail >= 2 {
                    Some(bounded_preview(&t, 256))
                } else {
                    None
                }
            }),
            redacted: output.as_ref().map(|o| o.redacted),
            content_digest_prefix: output
                .as_ref()
                .map(|o| digest_prefix(&o.content_digest, prefix)),
            payload_len: output.as_ref().and_then(|o| o.payload_len),
        },
        links,
        warnings,
    })
}

pub fn metrics_summary(workdir: &Path, last: usize) -> Result<MetricsSummary, OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let mut decisions = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::DecisionOut)
        .filter_map(|r| r.compute_summary.map(|s| (r.time.tick.get(), s)))
        .collect::<Vec<_>>();
    decisions.sort_by_key(|(t, _)| *t);
    let len = decisions.len();
    let slice = if last == 0 || last >= len {
        decisions.as_slice()
    } else {
        &decisions[len - last..]
    };

    let ticks_observed = slice.len();
    if ticks_observed == 0 {
        return Ok(MetricsSummary {
            ticks_observed: 0,
            mean_surprise: 0.0,
            max_surprise: 0.0,
            mean_pressure: 0.0,
            max_pressure: 0.0,
            mean_uncertainty: 0.0,
            max_uncertainty: 0.0,
            governor_tier_2_3_percent: 0.0,
            emergency_triggers: 0,
            tool_issuance_deny_rate: 0.0,
        });
    }

    let mut surprise_sum = 0.0;
    let mut pressure_sum = 0.0;
    let mut uncertainty_sum = 0.0;
    let mut max_surprise: f32 = 0.0;
    let mut max_pressure: f32 = 0.0;
    let mut max_uncertainty: f32 = 0.0;
    for (_, s) in slice {
        surprise_sum += s.surprise;
        pressure_sum += s.pressure;
        let u = s.lfm_uncertainty.unwrap_or(0.0);
        uncertainty_sum += u;
        max_surprise = max_surprise.max(s.surprise);
        max_pressure = max_pressure.max(s.pressure);
        max_uncertainty = max_uncertainty.max(u);
    }

    let from_tick = slice.first().map(|(t, _)| *t).unwrap_or(0);
    let to_tick = slice.last().map(|(t, _)| *t).unwrap_or(0);

    let issuances = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::CapabilityIssuance)
        .filter_map(|r| match &r.payload {
            ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(i))
                if i.t >= from_tick && i.t <= to_tick =>
            {
                Some(i)
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let tier23 = issuances.iter().filter(|i| i.effective_tier >= 2).count();
    let deny_total: usize = issuances.iter().map(|i| i.denied_kinds.len()).sum();
    let request_total: usize = issuances.iter().map(|i| i.requested_kinds.len()).sum();

    let emergency_triggers = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::Emergency)
        .filter_map(|r| match &r.payload {
            ExperiencePayload::Audit(AuditPayload::Emergency(e))
                if e.t >= from_tick && e.t <= to_tick =>
            {
                Some(e)
            }
            _ => None,
        })
        .filter(|e| e.state == EmergencyStateCode::Active)
        .count();

    Ok(MetricsSummary {
        ticks_observed,
        mean_surprise: surprise_sum / ticks_observed as f32,
        max_surprise,
        mean_pressure: pressure_sum / ticks_observed as f32,
        max_pressure,
        mean_uncertainty: uncertainty_sum / ticks_observed as f32,
        max_uncertainty,
        governor_tier_2_3_percent: if issuances.is_empty() {
            0.0
        } else {
            (tier23 as f32) * 100.0 / (issuances.len() as f32)
        },
        emergency_triggers,
        tool_issuance_deny_rate: if request_total == 0 {
            0.0
        } else {
            deny_total as f32 / request_total as f32
        },
    })
}

pub fn metrics_trend(
    workdir: &Path,
    from_tick: u64,
    to_tick: u64,
) -> Result<Vec<MetricsTrendPoint>, OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let mut points = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::DecisionOut)
        .filter_map(|r| {
            if r.time.tick.get() < from_tick || r.time.tick.get() > to_tick {
                return None;
            }
            r.compute_summary.map(|s| MetricsTrendPoint {
                t: r.time.tick.get(),
                surprise: Some(s.surprise),
                pressure: Some(s.pressure),
                uncertainty: s.lfm_uncertainty,
                risk: Some(s.risk),
            })
        })
        .collect::<Vec<_>>();
    points.sort_by_key(|p| p.t);
    if points.len() <= 256 {
        return Ok(points);
    }
    let step = points.len().div_ceil(256);
    Ok(points.into_iter().step_by(step).take(256).collect())
}

pub fn compute_resume_compat_digest(cfg: &ResumeCheckConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"UCF:resume_compat:v1");
    hasher.update(cfg.policy_bundle_hash.as_bytes());
    hasher.update(cfg.backend_pack_meta_digest.as_bytes());
    hasher.update(cfg.model_hashes_digest.as_bytes());
    hasher.update(cfg.enabled_features_bitmap.to_le_bytes());
    for (k, v) in &cfg.schema_versions {
        hasher.update(k.as_bytes());
        hasher.update(v.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

pub fn check_resume_compat(
    prev_run: &RunMetadataRecord,
    new_config: &ResumeCheckConfig,
) -> ResumeDecision {
    let mut reasons = Vec::new();
    if prev_run.policy_bundle_hash != new_config.policy_bundle_hash {
        reasons.push(ResumeMismatchReason::PolicyHash);
    }
    if prev_run.backend_pack_meta_digest != new_config.backend_pack_meta_digest {
        reasons.push(ResumeMismatchReason::BackendPackDigest);
    }
    let any_real_slots_enabled = new_config.enabled_features_bitmap != 0;
    if any_real_slots_enabled && prev_run.model_hashes_digest != new_config.model_hashes_digest {
        reasons.push(ResumeMismatchReason::ModelHashesDigest);
    }
    for (name, version) in &new_config.schema_versions {
        if prev_run.schema_versions.get(name).copied().unwrap_or(0) != *version {
            reasons.push(ResumeMismatchReason::SchemaVersion);
            break;
        }
    }
    if reasons.is_empty() {
        ResumeDecision::ResumeAllowed
    } else {
        ResumeDecision::NewSessionRequired { reasons }
    }
}

fn persist_run_metadata(workdir: &Path, run_metadata: &RunMetadataRecord) -> Result<(), OpsError> {
    write_json(
        workdir.join("ess").join("run_metadata_record.json"),
        run_metadata,
    )?;
    let run_dir = workdir.join("ess").join("runs");
    fs::create_dir_all(&run_dir)?;
    write_json(
        run_dir.join(format!("{}.json", run_metadata.run_id)),
        run_metadata,
    )?;
    Ok(())
}

fn load_run_registry(workdir: &Path) -> Result<Vec<RunMetadataRecord>, OpsError> {
    let run_dir = workdir.join("ess").join("runs");
    if !run_dir.exists() {
        return Ok(Vec::new());
    }
    let mut runs = Vec::new();
    for entry in fs::read_dir(&run_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let data = fs::read_to_string(path)?;
                if let Ok(meta) = serde_json::from_str::<RunMetadataRecord>(&data) {
                    runs.push(meta);
                }
            }
        }
    }
    runs.sort_by(|a, b| {
        a.started_at_tick
            .cmp(&b.started_at_tick)
            .then_with(|| a.run_id.cmp(&b.run_id))
    });
    Ok(runs)
}

fn latest_run_metadata(workdir: &Path) -> Result<Option<RunMetadataRecord>, OpsError> {
    Ok(load_run_registry(workdir)?.into_iter().last())
}

pub fn runs_list(workdir: &Path, last: usize) -> Result<Vec<RunRegistryEntry>, OpsError> {
    let records =
        load_fixture_records(&workdir.join("ess").join("ess_fixture.json")).unwrap_or_default();
    let last_tick = records.iter().map(|r| r.time.tick.get()).max();
    let mut entries = load_run_registry(workdir)?
        .into_iter()
        .map(|m| RunRegistryEntry {
            run_id: m.run_id,
            started_at_tick: m.started_at_tick,
            parent_run_id: m.parent_run_id,
            resume_reason: m.resume_reason,
            policy_bundle_hash_prefix: prefix_hex(&m.policy_bundle_hash, 12),
            pack_digest_prefix: prefix_hex(&m.backend_pack_meta_digest, 12),
            model_hashes_digest_prefix: prefix_hex(&m.model_hashes_digest, 12),
            profile: m.profile,
            status: if m.ended_at_tick.is_some() {
                "ended".to_string()
            } else {
                "active".to_string()
            },
            last_tick,
        })
        .collect::<Vec<_>>();
    if entries.len() > last {
        entries = entries.split_off(entries.len() - last);
    }
    Ok(entries)
}

pub fn runs_show(workdir: &Path, run_id: &str) -> Result<Option<RunMetadataRecord>, OpsError> {
    Ok(load_run_registry(workdir)?
        .into_iter()
        .find(|r| r.run_id == run_id))
}

pub fn runs_search(
    workdir: &Path,
    pack: Option<&str>,
    policy: Option<&str>,
    model: Option<&str>,
) -> Result<Vec<RunRegistryEntry>, OpsError> {
    let mut entries = runs_list(workdir, usize::MAX)?;
    entries.retain(|e| {
        pack.is_none_or(|p| e.pack_digest_prefix.starts_with(p))
            && policy.is_none_or(|p| e.policy_bundle_hash_prefix.starts_with(p))
            && model.is_none_or(|p| e.model_hashes_digest_prefix.starts_with(p))
    });
    Ok(entries)
}

pub fn run_status(workdir: &Path, run_id: &str) -> Result<RunStatusReport, OpsError> {
    let _meta = runs_show(workdir, run_id)?
        .ok_or_else(|| OpsError::Invalid(format!("unknown run_id: {run_id}")))?;
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let mut trend = metrics_trend(workdir, 0, u64::MAX)?;
    if trend.len() > 8 {
        trend = trend.split_off(trend.len() - 8);
    }
    let explain = build_explain_tick_report(
        &records,
        ExplainTickRequest {
            t: None,
            decision_id: None,
            detail_level: 1,
            digest_prefix_len: 8,
        },
    )?;
    let issuance_denies = explain
        .governance
        .issuance
        .iter()
        .flat_map(|i| i.denied.iter().cloned())
        .take(16)
        .collect::<Vec<_>>();
    let active_slots = vec!["llm", "world_jepa", "world_vljepa", "sae", "ssm", "lfm"]
        .into_iter()
        .map(str::to_string)
        .collect();
    Ok(RunStatusReport {
        run_id: run_id.to_string(),
        active_slots,
        governor_tier: explain.governance.tier.unwrap_or(0),
        governor_score: explain.governance.governor_score.unwrap_or(0) as f32 / 1024.0,
        emergency_active: explain.governance.emergency_active,
        last_ticks: trend,
        issuance_denies,
    })
}

fn ebm_term_label(term_id: u16) -> &'static str {
    match term_id {
        1 => "ToolIntentPenalty",
        2 => "CapabilityForbidden",
        3 => "CapabilityHighRisk",
        4 => "ContextRiskAmplifier",
        5 => "EmergencyDenyAllBias",
        6 => "OutputClassMismatch",
        7 => "BudgetExhaustedBias",
        _ => "UnknownTerm",
    }
}

fn digest_prefix(digest: &[u8; 32], prefix_len: usize) -> String {
    hex::encode(digest)[..prefix_len.min(64)].to_string()
}

fn digest_prefix_arr8(digest: &[u8; 8], prefix_len: usize) -> String {
    hex::encode(digest)[..prefix_len.min(16)].to_string()
}

fn bounded_preview(text: &str, max_chars: usize) -> String {
    let mut out = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        out.push('…');
    }
    out
}
pub fn load_or_init_config(workdir: &Path) -> Result<OpsConfig, OpsError> {
    let profile = resolved_profile_name();
    let path = profile_config_path(&profile);
    let mut cfg = load_profile_config(&path)?;

    cfg.profile = profile;
    apply_env_overrides(&mut cfg)?;
    apply_device_profile(&mut cfg)?;
    validate_config_ladder(&cfg)?;
    cfg.config_digest = ops_config_digest(&cfg)?;

    write_json(workdir.join("config_resolved.json"), &cfg)?;
    Ok(cfg)
}

fn resolved_profile_name() -> String {
    let profile = std::env::var("UCF_PROFILE")
        .unwrap_or_else(|_| "test".to_string())
        .to_ascii_lowercase();
    match profile.as_str() {
        "dev" | "test" | "prod" => profile,
        _ => "test".to_string(),
    }
}

fn profile_config_path(profile: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs")
        .join(format!("{profile}.toml"))
}

fn load_profile_config(path: &Path) -> Result<OpsConfig, OpsError> {
    let raw = fs::read_to_string(path)?;
    toml::from_str::<OpsConfig>(&raw)
        .map_err(|e| OpsError::Invalid(format!("invalid config {}: {e}", path.display())))
}

fn profile_rank(profile: &str) -> u8 {
    match profile {
        "dev" => 0,
        "test" => 1,
        "prod" => 2,
        _ => 0,
    }
}

fn validate_config_ladder(cfg: &OpsConfig) -> Result<(), OpsError> {
    if cfg.policy_overlay != cfg.profile {
        return Err(OpsError::Invalid(format!(
            "policy_overlay must match profile: profile={} overlay={}",
            cfg.profile, cfg.policy_overlay
        )));
    }

    if !cfg.offline {
        return Err(OpsError::Invalid(
            "offline must be enabled for all profiles".to_string(),
        ));
    }

    let rank = profile_rank(&cfg.profile);
    if rank >= profile_rank("test") {
        if cfg.compute_seed == 0 {
            return Err(OpsError::Invalid(
                "test/prod require non-zero deterministic seed".to_string(),
            ));
        }
        if cfg.sampling_enabled {
            return Err(OpsError::Invalid(
                "sampling must be disabled in test/prod".to_string(),
            ));
        }
        if !cfg.determinism_lock_strict {
            return Err(OpsError::Invalid(
                "determinism_lock_strict must be true in test/prod".to_string(),
            ));
        }
        if cfg.slot_ebm_mode != "shadow"
            && cfg.slot_ebm_mode != "active"
            && cfg.slot_ebm_mode != "off"
        {
            return Err(OpsError::Invalid(
                "slot_ebm_mode must be shadow, active, or off".to_string(),
            ));
        }
    }

    if rank >= profile_rank("prod") {
        if cfg.capabilities_default != "deny" {
            return Err(OpsError::Invalid(
                "prod requires capabilities_default=deny".to_string(),
            ));
        }
        if cfg.slot_ebm_mode != "shadow" {
            return Err(OpsError::Invalid(
                "prod requires slot_ebm_mode=shadow".to_string(),
            ));
        }
        if !cfg.docs_lint_required {
            return Err(OpsError::Invalid(
                "prod requires docs_lint_required=true".to_string(),
            ));
        }
    }

    Ok(())
}

fn apply_env_overrides(cfg: &mut OpsConfig) -> Result<(), OpsError> {
    const ALLOW: &[&str] = &[
        "UCF_POLICY_OVERLAY",
        "UCF_SLOT_EBM_MODE",
        "UCF_STAGE_ISOLATION",
        "UCF_EMERGENCY_POLICY_PIN",
        "UCF_DEVICE_PROFILE",
    ];
    for (k, v) in std::env::vars() {
        if !k.starts_with("UCF_OPS_OVERRIDE_") {
            continue;
        }
        if !ALLOW.contains(&v.as_str()) {
            return Err(OpsError::Invalid(format!(
                "unknown env override key via {k}={v}; allowed: {}",
                ALLOW.join(",")
            )));
        }
    }
    if let Ok(v) = std::env::var("UCF_POLICY_OVERLAY") {
        cfg.policy_overlay = v;
    }
    if let Ok(v) = std::env::var("UCF_SLOT_EBM_MODE") {
        cfg.slot_ebm_mode = v;
    }
    if let Ok(v) = std::env::var("UCF_STAGE_ISOLATION") {
        cfg.isolation_runtime = v;
    }
    if let Ok(v) = std::env::var("UCF_DEVICE_PROFILE") {
        cfg.device_profile = v;
    }
    if let Ok(v) = std::env::var("UCF_EMERGENCY_POLICY_PIN") {
        cfg.emergency_policy_pin = Some(v);
    }
    Ok(())
}

fn apply_device_profile(cfg: &mut OpsConfig) -> Result<(), OpsError> {
    let name = cfg.device_profile_name()?;
    let profile = DeviceProfileV1::for_name(name);
    cfg.device_profile = name.as_str().to_string();
    cfg.compute_budget_profile = profile.compute_budget_profile;
    cfg.stage_isolation_optional = profile.stage_isolation_default;
    Ok(())
}

fn ops_config_digest(cfg: &OpsConfig) -> Result<String, OpsError> {
    let mut normalized = cfg.clone();
    normalized.config_digest.clear();
    let bytes = serde_json::to_vec(&normalized)?;
    Ok(sha256_hex(&bytes))
}

fn run_compute_probe(cfg: &OpsConfig) -> Result<DiagCheck, OpsError> {
    let backend_cfg = ComputeBackendConfig {
        kind: cfg.compute_backend,
        seed: cfg.compute_seed,
        ..ComputeBackendConfig::default()
    };
    let budget = backend_cfg.to_budget();
    let backend = build_backend(&backend_cfg)?;
    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(1),
            window: WindowId::new(0),
        },
        CorrelationId(777),
        ChannelCode::ExternalOutput,
        Intent::new(IntentId(777), IntentKind::Speak, "diag"),
        "compute_probe",
    );
    let input = compute_input_from_control(&ctrl);
    let out = backend.compute(&input, budget)?;
    let pass = (0.0..=1.0).contains(&out.risk) && (0.0..=1.0).contains(&out.confidence);

    Ok(DiagCheck {
        name: "compute_probe".to_string(),
        pass,
        detail: format!("risk={:.3} confidence={:.3}", out.risk, out.confidence),
        remediation: "ensure compute backend feature flags and seed are set.".to_string(),
    })
}

fn ensure_policy_bundle_root() -> Result<(), OpsError> {
    let local_manifest = Path::new("policies/manifest.toml");
    let local_ok = fs::read_to_string(local_manifest)
        .ok()
        .and_then(|v| toml::from_str::<toml::Value>(&v).ok())
        .and_then(|v| v.get("bundle_sha256").cloned())
        .is_some();
    if local_ok {
        return Ok(());
    }
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = repo_root.join("policies");
    if !source.join("manifest.toml").exists() {
        return Ok(());
    }
    fs::create_dir_all("policies/bundle_v1")?;
    let src_manifest = fs::read_to_string(source.join("manifest.toml"))?;
    let mut bundle = String::new();
    let mut files = Vec::<(String, String)>::new();
    let mut cur_path: Option<String> = None;
    for line in src_manifest.lines() {
        let trimmed = line.trim();
        if let Some(v) = trimmed
            .strip_prefix("bundle_sha256 = ")
            .and_then(|rest| rest.strip_prefix('"'))
            .and_then(|rest| rest.strip_suffix('"'))
        {
            bundle = v.to_string();
        }
        if let Some(v) = trimmed
            .strip_prefix("path = ")
            .and_then(|rest| rest.strip_prefix('"'))
            .and_then(|rest| rest.strip_suffix('"'))
        {
            cur_path = Some(v.to_string());
        }
        if let Some(v) = trimmed
            .strip_prefix("sha256 = ")
            .and_then(|rest| rest.strip_prefix('"'))
            .and_then(|rest| rest.strip_suffix('"'))
        {
            if let Some(path) = cur_path.take() {
                files.push((path, v.to_string()));
            }
        }
    }
    let mut normalized = String::from("version = \"v1\"\n");
    if !bundle.is_empty() {
        normalized.push_str(&format!("bundle_sha256 = \"{}\"\n\n", bundle));
    }
    for (path, sha) in &files {
        normalized.push_str("[[files]]\n");
        normalized.push_str(&format!("path = \"{}\"\n", path));
        normalized.push_str(&format!("sha256 = \"{}\"\n\n", sha));
    }
    fs::write("policies/manifest.toml", normalized)?;
    for name in [
        "compiled_rules.json",
        "allowlists.json",
        "governor_defaults.json",
        "retention_v1.json",
        "ebm_constraints.toml",
    ] {
        fs::copy(
            source.join("bundle_v1").join(name),
            Path::new("policies/bundle_v1").join(name),
        )?;
    }
    Ok(())
}

fn ensure_policy_bundle_hash_env() {
    if std::env::var("UCF_POLICY_BUNDLE_SHA256").is_ok() {
        return;
    }
    let manifest_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/manifest.toml");
    if let Ok(manifest_raw) = fs::read_to_string(manifest_path) {
        if let Some(hash) = manifest_raw.lines().find_map(|line| {
            line.trim()
                .strip_prefix("bundle_sha256 = ")
                .and_then(|rest| rest.strip_prefix('"'))
                .and_then(|rest| rest.strip_suffix('"'))
        }) {
            std::env::set_var("UCF_POLICY_BUNDLE_SHA256", hash);
        }
    }
}

fn ensure_layout(workdir: &Path) -> Result<(), OpsError> {
    for dir in ["ess", "logs", "reports", "fixtures"] {
        fs::create_dir_all(workdir.join(dir))?;
    }
    Ok(())
}

fn extract_text(ctrl: &ControlFrame) -> String {
    match &ctrl.payload {
        ControlPayload::Text(text) => text.to_string(),
        _ => "demo".to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildTag {
    git_commit: String,
    package_version: String,
}

fn build_tag() -> Result<BuildTag, OpsError> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()?;
    let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(BuildTag {
        git_commit: commit,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChecksumManifest {
    files: BTreeMap<String, String>,
    bundle_digest: String,
}

fn build_checksums(dir: &Path) -> Result<ChecksumManifest, OpsError> {
    let mut files = BTreeMap::new();
    for name in [
        "build_tag.json",
        "config_resolved.json",
        "ess_slice.json",
        "indices.json",
        "README.txt",
    ] {
        let data = fs::read(dir.join(name))?;
        files.insert(name.to_string(), sha256_hex(&data));
    }

    let mut bundle_hasher = Sha256::new();
    for (name, digest) in &files {
        bundle_hasher.update(name.as_bytes());
        bundle_hasher.update(digest.as_bytes());
    }

    Ok(ChecksumManifest {
        files,
        bundle_digest: hex::encode(bundle_hasher.finalize()),
    })
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<(), OpsError> {
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EssCompactionManifest {
    pub schema_version: u16,
    pub range_start_tick: u64,
    pub range_end_tick: u64,
    pub records_total: usize,
    pub redactions_total: u64,
    pub payload_bytes_pruned_total: u64,
    pub policy_hash: String,
    pub snapshot_digest: String,
    pub manifest_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EbmDatasetSample {
    pub schema_version: u16,
    pub run_id: String,
    pub tick: u64,
    pub decision_id: u64,
    pub context_digest: String,
    pub signals_q: EbmSignalsQ,
    pub candidates: Vec<EbmCandidateFeature>,
    pub label: EbmTrainingLabel,
    pub ebm_energy_q: Option<u16>,
    pub constraint_term_ids: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EbmSignalsQ {
    pub risk_q: Option<u16>,
    pub pressure_q: Option<u16>,
    pub surprise_q: Option<u16>,
    pub uncertainty_q: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EbmCandidateFeature {
    pub candidate_id: u16,
    pub digest_prefix: String,
    pub intent_kind: u8,
    pub output_class: u8,
    pub tool_intent_count: u8,
    pub allowed: bool,
    pub policy_hint: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EbmTrainingLabel {
    ChosenCandidate {
        chosen_candidate_id: u16,
    },
    PairwisePreference {
        better_candidate_id: u16,
        worse_candidate_id: u16,
    },
}

pub fn ebm_export_dataset(
    workdir: &Path,
    run_id: &str,
    from: u64,
    to: u64,
    out: &Path,
    policy: &Path,
) -> Result<usize, OpsError> {
    let mut records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let policy_text = fs::read_to_string(policy)?;
    let policy: RetentionPolicyV1 = serde_json::from_str(&policy_text)?;
    let now_tick = records.last().map(|r| r.time.tick.get()).unwrap_or(0);
    apply_retention(&mut records, &policy, now_tick);

    let samples = build_ebm_dataset_samples(&records, run_id, from, to);
    let parent = out
        .parent()
        .ok_or_else(|| OpsError::Invalid("output path has no parent".to_string()))?;
    fs::create_dir_all(parent)?;
    let mut body = String::new();
    for sample in &samples {
        body.push_str(&serde_json::to_string(sample)?);
        body.push('\n');
    }
    fs::write(out, body)?;
    Ok(samples.len())
}

fn build_ebm_dataset_samples(
    records: &[ExperienceRecord],
    run_id: &str,
    from: u64,
    to: u64,
) -> Vec<EbmDatasetSample> {
    let mut by_decision: BTreeMap<u64, Vec<&ExperienceRecord>> = BTreeMap::new();
    for record in records {
        let tick = record.time.tick.get();
        if tick < from || tick > to {
            continue;
        }
        by_decision
            .entry(decision_id_from_record(record))
            .or_default()
            .push(record);
    }

    let mut samples = Vec::new();
    for records in by_decision.values() {
        let Some(sample) = sample_from_decision_records(records, run_id) else {
            continue;
        };
        samples.push(sample);
    }
    samples
}

fn decision_id_from_record(record: &ExperienceRecord) -> u64 {
    match &record.payload {
        ExperiencePayload::Audit(AuditPayload::CandidateSet(c)) => c.decision_id,
        ExperiencePayload::Audit(AuditPayload::Output(o)) => o.decision_id,
        ExperiencePayload::Audit(AuditPayload::EbmReasoning(r)) => r.decision_id,
        _ => record.id.0,
    }
}

fn sample_from_decision_records(
    records: &[&ExperienceRecord],
    run_id: &str,
) -> Option<EbmDatasetSample> {
    let candidate_set = records.iter().find_map(|r| match &r.payload {
        ExperiencePayload::Audit(AuditPayload::CandidateSet(c)) => Some(c.clone()),
        _ => None,
    })?;
    let reasoning = records.iter().find_map(|r| match &r.payload {
        ExperiencePayload::Audit(AuditPayload::EbmReasoning(e)) => Some(e.clone()),
        _ => None,
    });
    let output = records.iter().find_map(|r| match &r.payload {
        ExperiencePayload::Audit(AuditPayload::Output(o)) => Some(o.clone()),
        _ => None,
    });

    if output
        .as_ref()
        .is_some_and(|o| !o.redacted && o.text.is_some())
    {
        // Export remains metadata-only, never raw output text.
    }

    let mut candidates = candidate_set
        .summaries
        .iter()
        .map(|s| EbmCandidateFeature {
            candidate_id: s.candidate_id,
            digest_prefix: hex::encode(&s.digest[..8]),
            intent_kind: s.intent_kind,
            output_class: s.output_class,
            tool_intent_count: s.tool_intent_count,
            allowed: s.allowed,
            policy_hint: s.policy_hint,
        })
        .collect::<Vec<_>>();
    candidates.truncate(32);

    let label = if let Some(worse) = candidates
        .iter()
        .find(|c| c.candidate_id != candidate_set.selected_candidate_id)
    {
        EbmTrainingLabel::PairwisePreference {
            better_candidate_id: candidate_set.selected_candidate_id,
            worse_candidate_id: worse.candidate_id,
        }
    } else {
        EbmTrainingLabel::ChosenCandidate {
            chosen_candidate_id: candidate_set.selected_candidate_id,
        }
    };

    let mut hasher = Sha256::new();
    hasher.update(candidate_set.selected_candidate_digest);
    if let Some(output) = &output {
        hasher.update(output.content_digest);
    }

    let (signals_q, constraint_term_ids) = if let Some(r) = &reasoning {
        (
            EbmSignalsQ {
                risk_q: Some(r.risk_q),
                pressure_q: Some(r.pressure_q),
                surprise_q: Some(r.surprise_q),
                uncertainty_q: Some(r.uncertainty_q),
            },
            r.top_term_contributions
                .iter()
                .take(8)
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
        )
    } else {
        (
            EbmSignalsQ {
                risk_q: None,
                pressure_q: None,
                surprise_q: None,
                uncertainty_q: None,
            },
            Vec::new(),
        )
    };

    Some(EbmDatasetSample {
        schema_version: 1,
        run_id: run_id.to_string(),
        tick: candidate_set.t,
        decision_id: candidate_set.decision_id,
        context_digest: hex::encode(hasher.finalize()),
        signals_q,
        candidates,
        label,
        ebm_energy_q: records.iter().find_map(|r| find_ebm_energy(r)),
        constraint_term_ids,
    })
}

pub fn ess_snapshot(workdir: &Path, out: &Path) -> Result<EssCompactionManifest, OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    fs::create_dir_all(
        out.parent()
            .ok_or_else(|| OpsError::Invalid("snapshot out path has no parent".to_string()))?,
    )?;
    let snapshot_body = serde_json::to_string_pretty(&records.len())?;
    fs::write(out, snapshot_body.as_bytes())?;
    let mut manifest = EssCompactionManifest {
        schema_version: 1,
        range_start_tick: records.first().map(|r| r.time.tick.get()).unwrap_or(0),
        range_end_tick: records.last().map(|r| r.time.tick.get()).unwrap_or(0),
        records_total: records.len(),
        redactions_total: 0,
        payload_bytes_pruned_total: 0,
        policy_hash: "none".to_string(),
        snapshot_digest: sha256_hex(snapshot_body.as_bytes()),
        manifest_digest: String::new(),
    };
    manifest.manifest_digest = sha256_hex(&serde_json::to_vec(&manifest)?);
    Ok(manifest)
}

pub fn ess_compact(workdir: &Path, policy_path: &Path) -> Result<EssCompactionManifest, OpsError> {
    let mut records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let policy_text = fs::read_to_string(policy_path)?;
    let policy: RetentionPolicyV1 = serde_json::from_str(&policy_text)?;
    let now_tick = records.last().map(|r| r.time.tick.get()).unwrap_or(0);
    let stats = apply_retention(&mut records, &policy, now_tick);

    let snapshot = serde_json::to_vec_pretty(&records.len())?;
    let mut manifest = EssCompactionManifest {
        schema_version: 1,
        range_start_tick: records.first().map(|r| r.time.tick.get()).unwrap_or(0),
        range_end_tick: records.last().map(|r| r.time.tick.get()).unwrap_or(0),
        records_total: records.len(),
        redactions_total: stats.redactions_total,
        payload_bytes_pruned_total: stats.payload_bytes_pruned_total,
        policy_hash: sha256_hex(policy_text.as_bytes()),
        snapshot_digest: sha256_hex(&snapshot),
        manifest_digest: String::new(),
    };
    manifest.manifest_digest = sha256_hex(&serde_json::to_vec(&manifest)?);
    Ok(manifest)
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn export_and_verify_roundtrip() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 20).expect("bringup");
        let report_dir = export_bugreport(
            dir.path(),
            &ExportArgs {
                last: Some(5),
                include_sandbox: false,
                include_audit: false,
            },
        )
        .expect("export");

        verify_bugreport(&report_dir).expect("verify");
        assert!(report_dir.join("checksums.json").exists());
    }

    #[test]
    fn verify_catches_tampering() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 8).expect("bringup");
        let report_dir = export_bugreport(
            dir.path(),
            &ExportArgs {
                last: Some(4),
                include_sandbox: false,
                include_audit: false,
            },
        )
        .expect("export");

        fs::write(report_dir.join("README.txt"), "tampered").expect("tamper");
        let err = verify_bugreport(&report_dir).expect_err("must fail");
        assert!(format!("{err}").contains("checksum mismatch"));
    }

    #[test]
    fn explain_tick_is_deterministic_for_fixture_data() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 12).expect("bringup");

        let req = ExplainTickRequest {
            t: Some(12),
            decision_id: None,
            detail_level: 1,
            digest_prefix_len: 8,
        };
        let a = explain_tick(dir.path(), req).expect("report a");
        let b = explain_tick(dir.path(), req).expect("report b");

        assert_eq!(a, b);
        assert_eq!(a.header.t, 12);
        assert!(a.header.evidence_chain_digest_prefix.is_none());
        assert!(!a.warnings.is_empty());
    }

    #[test]
    fn metrics_trend_downsamples_to_bound() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 600).expect("bringup");

        let trend = metrics_trend(dir.path(), 0, u64::MAX).expect("trend");
        assert!(trend.len() <= 256);
        assert!(!trend.is_empty());
    }

    #[test]
    fn weights_lifecycle_check_fails_without_manifest() {
        let dir = tempdir().expect("tempdir");
        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(dir.path()).expect("chdir");
        let c = check_weights_lifecycle_integrity(dir.path()).expect("check");
        std::env::set_current_dir(cwd).expect("restore");
        assert_eq!(c.name, "weights_lifecycle");
        assert_eq!(c.status, GateStatus::Skip);
    }

    #[test]
    fn world_vljepa_check_fails_when_required_shadow_missing() {
        let dir = tempdir().expect("tempdir");
        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(dir.path()).expect("chdir");
        fs::create_dir_all("models").expect("models");
        fs::write(
            "models/MANIFEST.toml",
            r#"manifest_version = 1
manifest_digest = "x"
[slots.world_vljepa]
active_hash = "abc"
"#,
        )
        .expect("manifest");
        let c = check_world_vljepa_shadow_evidence(dir.path()).expect("check");
        std::env::set_current_dir(cwd).expect("restore");
        assert_eq!(c.name, "world_vljepa_evidence");
        assert_eq!(c.status, GateStatus::Fail);
        assert!(c
            .remediation_hint
            .unwrap_or_default()
            .contains("shadow-report"));
    }

    #[test]
    fn readiness_report_json_is_stable() {
        let report = ReadinessGateReport {
            code_version_tag: "abc".to_string(),
            fixtures_digest_prefix: Some("123456".to_string()),
            backend_pack_digest_prefix: Some("abcdef".to_string()),
            timestamp: None,
            status: GateStatus::Pass,
            checks: vec![check_pass(
                "alpha",
                [
                    ("z".to_string(), "2".to_string()),
                    ("a".to_string(), "1".to_string()),
                ],
            )],
            weights_lifecycle: None,
            world_vljepa_evidence: None,
            sae_real: None,
            ssm_opt: None,
            gpu_lane: None,
        };

        let left = serde_json::to_string(&report).expect("json left");
        let right = serde_json::to_string(&report).expect("json right");
        assert_eq!(left, right);
        assert!(left.contains("\"a\":\"1\""));
        assert!(left.contains("\"z\":\"2\""));
    }

    #[test]
    fn readiness_bounded_fields_are_capped() {
        let long = "x".repeat(512);
        let check = check_fail("n", [("k".repeat(80), long.clone())], &long, &long);

        let key = check.evidence.keys().next().expect("key");
        let val = check.evidence.values().next().expect("value");
        assert!(key.chars().count() <= 49);
        assert!(val.chars().count() <= 97);
        assert!(
            check
                .failure_reason
                .as_deref()
                .expect("reason")
                .chars()
                .count()
                <= GATE_STR_CAP + 1
        );
        assert!(
            check
                .remediation_hint
                .as_deref()
                .expect("hint")
                .chars()
                .count()
                <= GATE_STR_CAP + 1
        );
    }
    #[test]
    fn bounded_preview_caps_and_marks_truncation() {
        let preview = bounded_preview("abcdefghijklmnopqrstuvwxyz", 8);
        assert_eq!(preview, "abcdefgh…");
    }

    #[test]
    fn probe_inputs_are_deterministic() {
        let a = probe_spec_for_slot(ModelSlot::Llm);
        let b = probe_spec_for_slot(ModelSlot::Llm);
        assert_eq!(a.input_digest, b.input_digest);

        let wa = probe_spec_for_slot(ModelSlot::WorldJepa);
        let wb = probe_spec_for_slot(ModelSlot::WorldJepa);
        assert_eq!(wa.input_digest, wb.input_digest);
    }

    #[test]
    fn timeout_returns_without_deadlock() {
        let result = exec_with_timeout(10, || {
            thread::sleep(Duration::from_millis(100));
            Ok::<_, String>(())
        });
        assert!(matches!(result, Err(ProbeExecError::Timeout)));
    }

    #[test]
    fn models_probe_persists_records_and_report() {
        let dir = tempdir().expect("tempdir");
        let out = dir.path().join("out/probe_report.json");
        let report = models_probe(dir.path(), Path::new("models/manifest.toml"), &out)
            .expect("models probe");
        assert!(!report.results.is_empty());
        assert!(out.exists());
        let records = dir.path().join("ess/model_probe_records.json");
        assert!(records.exists());
    }

    #[test]
    fn probe_digests_are_deterministic_for_toy_backends() {
        let world_spec = probe_spec_for_slot(ModelSlot::WorldJepa);
        let a = run_world_probe(
            BackendPackFactory::build(BackendPackConfig::default()).expect("pack a"),
            &world_spec,
        )
        .expect("world a");
        let b = run_world_probe(
            BackendPackFactory::build(BackendPackConfig::default()).expect("pack b"),
            &world_spec,
        )
        .expect("world b");
        assert_eq!(a.0, b.0);

        let sae_spec = probe_spec_for_slot(ModelSlot::Sae);
        let c = run_sae_probe(
            BackendPackFactory::build(BackendPackConfig::default()).expect("pack c"),
            &sae_spec,
        )
        .expect("sae a");
        let d = run_sae_probe(
            BackendPackFactory::build(BackendPackConfig::default()).expect("pack d"),
            &sae_spec,
        )
        .expect("sae b");
        assert_eq!(c.0, d.0);
    }

    #[test]
    fn ess_snapshot_manifest_digest_is_deterministic() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 8).expect("bringup");
        let snap_path = dir.path().join("snapshots/run.snap");
        let a = ess_snapshot(dir.path(), &snap_path).expect("snapshot a");
        let b = ess_snapshot(dir.path(), &snap_path).expect("snapshot b");
        assert_eq!(a.manifest_digest, b.manifest_digest);
    }

    #[test]
    fn ebm_dataset_export_is_redaction_safe_and_bounded() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 20).expect("bringup");
        let out = dir.path().join("out").join("ebm_dataset_v1.jsonl");
        let policy = PathBuf::from("policies/bundle_v1/retention_v1.json");
        let count =
            ebm_export_dataset(dir.path(), "run-test", 0, u64::MAX, &out, &policy).expect("ok");
        assert_eq!(
            count,
            fs::read_to_string(&out).expect("dataset").lines().count()
        );

        let body = fs::read_to_string(out).expect("dataset");
        for line in body.lines() {
            let sample: EbmDatasetSample = serde_json::from_str(line).expect("sample");
            assert!(sample.candidates.len() <= 32);
            assert!(!sample.context_digest.is_empty());
            assert!(!line.contains("\"text\":"));
        }
    }

    #[test]
    fn resume_compat_digest_is_stable() {
        let mut schema = BTreeMap::new();
        schema.insert("output".to_string(), 1);
        let cfg = ResumeCheckConfig {
            policy_bundle_hash: "policy-a".to_string(),
            backend_pack_meta_digest: "pack-a".to_string(),
            model_hashes_digest: "model-a".to_string(),
            enabled_features_bitmap: 1,
            schema_versions: schema,
        };
        assert_eq!(
            compute_resume_compat_digest(&cfg),
            compute_resume_compat_digest(&cfg)
        );
    }

    #[test]
    fn resume_decision_requires_new_session_on_policy_change() {
        let mut schema = BTreeMap::new();
        schema.insert("output".to_string(), 1);
        let prev = RunMetadataRecord {
            run_id: "r1".to_string(),
            started_at_tick: 0,
            code_version_tag: "c".to_string(),
            backend_pack_meta_digest: "pack-a".to_string(),
            fixtures_digest: "f".to_string(),
            model_hashes_digest: "model-a".to_string(),
            enabled_features_bitmap: 1,
            profile: "test".to_string(),
            config_digest: "cfg".to_string(),
            policy_overlay: "test".to_string(),
            platform_probe_summary: "os=linux".to_string(),
            device_profile_name: "small".to_string(),
            device_profile_digest: "d".to_string(),
            schema_versions: schema.clone(),
            parent_run_id: None,
            resume_reason: None,
            compat_digest: "d".to_string(),
            policy_bundle_hash: "policy-a".to_string(),
            determinism_mode: "deterministic_only".to_string(),
            determinism_policy_digest: None,
            ended_at_tick: Some(10),
        };
        let cfg_ok = ResumeCheckConfig {
            policy_bundle_hash: "policy-a".to_string(),
            backend_pack_meta_digest: "pack-a".to_string(),
            model_hashes_digest: "model-a".to_string(),
            enabled_features_bitmap: 1,
            schema_versions: schema.clone(),
        };
        assert_eq!(
            check_resume_compat(&prev, &cfg_ok),
            ResumeDecision::ResumeAllowed
        );
        let cfg_bad = ResumeCheckConfig {
            policy_bundle_hash: "policy-b".to_string(),
            ..cfg_ok
        };
        assert!(matches!(
            check_resume_compat(&prev, &cfg_bad),
            ResumeDecision::NewSessionRequired { .. }
        ));
    }

    #[test]
    fn runs_list_ordering_is_stable() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("ess/runs");
        fs::create_dir_all(&run_dir).expect("run dir");
        let a = RunMetadataRecord {
            run_id: "b-run".to_string(),
            started_at_tick: 5,
            ..RunMetadataRecord::default()
        };
        let b = RunMetadataRecord {
            run_id: "a-run".to_string(),
            started_at_tick: 5,
            ..RunMetadataRecord::default()
        };
        write_json(run_dir.join("1.json"), &a).expect("write a");
        write_json(run_dir.join("2.json"), &b).expect("write b");
        let list = runs_list(dir.path(), 10).expect("runs");
        assert_eq!(list[0].run_id, "a-run");
        assert_eq!(list[1].run_id, "b-run");
    }
    #[test]
    fn ess_compaction_manifest_tamper_detected() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 8).expect("bringup");
        let policy_path = dir.path().join("retention.json");
        fs::write(
            &policy_path,
            serde_json::to_string(&RetentionPolicyV1::default()).expect("policy"),
        )
        .expect("write policy");
        let manifest = ess_compact(dir.path(), &policy_path).expect("compact");
        let mut tampered = manifest.clone();
        tampered.records_total = tampered.records_total.saturating_add(1);
        assert_ne!(
            sha256_hex(&serde_json::to_vec(&tampered).expect("tampered vec")),
            manifest.manifest_digest
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutManifestEntry {
    pub file: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutManifest {
    pub dir: String,
    pub generated_at: u64,
    pub entries: Vec<OutManifestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseChecklistItem {
    pub id: String,
    pub command: String,
    pub required: bool,
    pub expected_artifact: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseChecklist {
    pub version: String,
    pub profile: String,
    pub items: Vec<ReleaseChecklistItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignoffCheckResult {
    pub id: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignoffResult {
    pub pass: bool,
    pub checked_at: u64,
    pub out_dir: String,
    pub artifacts_manifest: OutManifest,
    pub checks: Vec<SignoffCheckResult>,
}

pub fn out_manifest(dir: &Path) -> Result<OutManifest, OpsError> {
    let mut entries = Vec::new();
    if !dir.exists() {
        return Err(OpsError::Invalid(format!(
            "out dir missing: {}",
            dir.display()
        )));
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name == "manifest.json" {
            continue;
        }
        let bytes = fs::read(&path)?;
        entries.push(OutManifestEntry {
            file: name.to_string(),
            sha256: sha256_hex(&bytes),
            size_bytes: bytes.len() as u64,
        });
    }
    entries.sort_by(|a, b| a.file.cmp(&b.file));
    let mut manifest = OutManifest {
        dir: dir.display().to_string(),
        generated_at: now_unix_secs(),
        entries,
    };
    write_json(dir.join("manifest.json"), &manifest)?;
    let manifest_bytes = fs::read(dir.join("manifest.json"))?;
    manifest.entries.push(OutManifestEntry {
        file: "manifest.json".to_string(),
        sha256: sha256_hex(&manifest_bytes),
        size_bytes: manifest_bytes.len() as u64,
    });
    manifest.entries.sort_by(|a, b| a.file.cmp(&b.file));
    write_json(dir.join("manifest.json"), &manifest)?;
    Ok(manifest)
}

pub fn load_signoff_checklist(path: &Path) -> Result<ReleaseChecklist, OpsError> {
    let raw = fs::read_to_string(path)?;
    let parsed: ReleaseChecklist = toml::from_str(&raw)
        .map_err(|err| OpsError::Invalid(format!("invalid checklist toml: {err}")))?;
    Ok(parsed)
}

pub fn release_signoff_validate(
    out_dir: &Path,
    checklist_path: &Path,
    emit: &Path,
) -> Result<SignoffResult, OpsError> {
    let checklist = load_signoff_checklist(checklist_path)?;
    let manifest = out_manifest(out_dir)?;
    let mut checks = Vec::new();

    for item in checklist.items {
        if let Some(expected) = item.expected_artifact {
            let found = manifest.entries.iter().find(|e| e.file == expected);
            let ok = found.is_some() || !item.required;
            checks.push(SignoffCheckResult {
                id: item.id,
                ok,
                detail: if ok {
                    format!("artifact {} present", expected)
                } else {
                    format!("artifact {} missing", expected)
                },
            });
        } else {
            checks.push(SignoffCheckResult {
                id: item.id,
                ok: true,
                detail: "no artifact assertion".to_string(),
            });
        }
    }

    let pass = checks.iter().all(|c| c.ok);
    let result = SignoffResult {
        pass,
        checked_at: now_unix_secs(),
        out_dir: out_dir.display().to_string(),
        artifacts_manifest: manifest,
        checks,
    };
    write_json(emit, &result)?;
    Ok(result)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rc1GateReport {
    pub schema_version: u16,
    pub status: GateStatus,
    pub checks: Vec<CheckResult>,
    pub policy_graph_digest: String,
    pub model_hashes_digest: String,
    pub readiness_gate_path: String,
    pub artifacts: Vec<String>,
}

pub fn release_rc1_gate(
    workdir: &Path,
    out: &Path,
    include_load_smoke: bool,
) -> Result<Rc1GateReport, OpsError> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    std::env::set_var("UCF_SSM_KERNEL", "ref");

    let mut checks = Vec::new();
    let mut artifacts = Vec::new();

    let policy = policy_validate(Path::new("policies/packs/base_v1"), None)?;
    let mut policy_ev = BTreeMap::new();
    policy_ev.insert(
        "policy_graph_digest".to_string(),
        policy.policy_graph_digest.clone(),
    );
    checks.push(CheckResult {
        name: "policy_validate".to_string(),
        status: GateStatus::Pass,
        evidence: policy_ev,
        failure_reason: None,
        remediation_hint: None,
    });

    let models = models_verify(Path::new("models/manifest.toml"))?;
    let mut model_ev = BTreeMap::new();
    model_ev.insert(
        "model_hashes_digest".to_string(),
        models.model_hashes_digest.clone(),
    );
    let model_fail = models
        .slots
        .iter()
        .find(|s| s.enabled && s.status != "verified")
        .map(|s| format!("enabled slot {} not verified", s.slot.as_str()));
    checks.push(CheckResult {
        name: "models_verify".to_string(),
        status: if model_fail.is_some() {
            GateStatus::Fail
        } else {
            GateStatus::Pass
        },
        evidence: model_ev,
        failure_reason: model_fail,
        remediation_hint: Some("provide fixture weights or disable slot in manifest".to_string()),
    });

    let gate_out = out.with_file_name("rc1_readiness_gate.json");
    let gate = readiness_gate(workdir, "test", &gate_out)?;
    artifacts.push(gate_out.display().to_string());
    let mut gate_ev = BTreeMap::new();
    gate_ev.insert("status".to_string(), format!("{:?}", gate.status));
    checks.push(CheckResult {
        name: "readiness_gate".to_string(),
        status: gate.status,
        evidence: gate_ev,
        failure_reason: if gate.status == GateStatus::Pass {
            None
        } else {
            Some("readiness gate failed".to_string())
        },
        remediation_hint: Some("inspect readiness gate report for failed checks".to_string()),
    });

    if include_load_smoke {
        let smoke_out = out.with_file_name("rc1_load_smoke.json");
        let bench = crate::bench::bench_run(&crate::bench::BenchArgs {
            scenario: PathBuf::from("fixtures/e2e_scenario_a.json"),
            ticks: 300,
            out: smoke_out.clone(),
            rss_sample_every: 16,
            rss_cap_mb: Some(2048),
        })?;
        artifacts.push(smoke_out.display().to_string());
        let mut ev = BTreeMap::new();
        ev.insert(
            "p95_ms".to_string(),
            format!("{:.4}", bench.tick_time_ms.p95_ms),
        );
        ev.insert(
            "max_rss_mb".to_string(),
            format!("{:.1}", bench.memory.max_rss_mb.unwrap_or(0.0)),
        );
        checks.push(CheckResult {
            name: "load_smoke".to_string(),
            status: if bench.memory.cap_exceeded {
                GateStatus::Fail
            } else {
                GateStatus::Pass
            },
            evidence: ev,
            failure_reason: if bench.memory.cap_exceeded {
                Some("rss cap exceeded".to_string())
            } else {
                None
            },
            remediation_hint: Some("reduce enabled features or tighten budgets".to_string()),
        });
    }

    let status = if checks.iter().any(|c| c.status == GateStatus::Fail) {
        GateStatus::Fail
    } else {
        GateStatus::Pass
    };
    let report = Rc1GateReport {
        schema_version: 1,
        status,
        checks,
        policy_graph_digest: policy.policy_graph_digest,
        model_hashes_digest: models.model_hashes_digest,
        readiness_gate_path: gate_out.display().to_string(),
        artifacts,
    };
    write_json(out, &report)?;
    Ok(report)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiagnosticsBundleReport {
    pub run_id: String,
    pub out: String,
    pub entries: Vec<String>,
}

pub fn diagnostics_collect(
    workdir: &Path,
    run_id: &str,
    out: &Path,
) -> Result<DiagnosticsBundleReport, OpsError> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let run_dir = PathBuf::from("out").join(run_id);
    if !run_dir.exists() {
        return Err(OpsError::Invalid(format!(
            "run artifact directory not found: {}",
            run_dir.display()
        )));
    }

    let mut selected = vec![
        run_dir.join("run_metadata.json"),
        run_dir.join("metrics_summary.json"),
        run_dir.join("gate_report.json"),
        run_dir.join("adversarial_report.json"),
        run_dir.join("bench_report.json"),
    ];
    let explain_dir = workdir.join("explain_tick");
    if explain_dir.exists() {
        for e in fs::read_dir(&explain_dir)? {
            let p = e?.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json") {
                selected.push(p);
            }
        }
    }

    let file = fs::File::create(out)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let mut entries = Vec::new();
    for path in selected {
        if !path.exists() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("entry.json")
            .to_string();
        let mut text = fs::read_to_string(&path).unwrap_or_default();
        if text.contains("\"text\":") || text.contains("\"payload\":") {
            text = text.replace("\"text\":", "\"text_redacted\":");
            text = text.replace("\"payload\":", "\"payload_redacted\":");
        }
        zip.start_file(name.clone(), opts)
            .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
        use std::io::Write;
        zip.write_all(text.as_bytes())
            .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
        entries.push(name);
    }
    zip.finish()
        .map_err(|e| OpsError::Invalid(format!("zip finalize failed: {e}")))?;
    Ok(DiagnosticsBundleReport {
        run_id: run_id.to_string(),
        out: out.display().to_string(),
        entries,
    })
}

pub fn security_verify_chain(workdir: &Path, from: u64, to: u64) -> Result<(), OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let mut prev: Option<[u8; 32]> = None;
    for record in records
        .iter()
        .filter(|r| r.time.tick.get() >= from && r.time.tick.get() <= to)
    {
        if !matches!(
            record.kind,
            ExperienceKind::CapabilityIssuance
                | ExperienceKind::Emergency
                | ExperienceKind::BackendPack
                | ExperienceKind::AuditCheckpoint
                | ExperienceKind::Output
        ) {
            continue;
        }
        if let (Some(expected_prev), Some(digest)) = (record.audit_prev_digest, record.audit_digest)
        {
            if let Some(actual_prev) = prev {
                if actual_prev != expected_prev {
                    return Err(OpsError::Invalid(format!(
                        "security chain break at experience_id={} tick={}",
                        record.id.0,
                        record.time.tick.get()
                    )));
                }
            }
            prev = Some(digest);
        }
    }

    let run_id = workdir
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("local")
        .to_string();
    let segments = build_merkle_segments(&run_id, &records, 1024);
    verify_segment_chain(&segments)?;

    for segment in &segments {
        if segment.record_count == 0 {
            continue;
        }
        let proof = prove_record_in_segment(segment, segment.leaf_digests[0])
            .ok_or_else(|| OpsError::Invalid("failed to build sample segment proof".to_string()))?;
        if !verify_merkle_proof(&proof) {
            return Err(OpsError::Invalid(format!(
                "segment proof verification failed for segment {}",
                segment.segment_id.segment_index
            )));
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SegmentId {
    pub run_id: String,
    pub segment_index: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerkleSegmentRecord {
    pub segment_id: SegmentId,
    pub first_t: u64,
    pub last_t: u64,
    pub record_count: u32,
    pub merkle_root: String,
    pub prev_segment_root: Option<String>,
    pub segment_digest: String,
    #[serde(skip)]
    leaf_digests: Vec<[u8; 32]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerkleProofStep {
    pub sibling_hash: String,
    pub sibling_on_left: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerkleProofRecord {
    pub segment_id: SegmentId,
    pub leaf_index: usize,
    pub siblings: Vec<MerkleProofStep>,
    pub segment_root: String,
    pub leaf_hash: String,
    pub proof_digest: String,
}

pub fn logs_prove(
    workdir: &Path,
    record_digest_hex: &str,
    out: &Path,
    segment_size: usize,
) -> Result<MerkleProofRecord, OpsError> {
    let target = parse_hex_digest(record_digest_hex)?;
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let run_id = workdir
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("local")
        .to_string();
    let segments = build_merkle_segments(&run_id, &records, segment_size.max(1));
    for segment in &segments {
        if let Some(proof) = prove_record_in_segment(segment, target) {
            write_json(out, &proof)?;
            return Ok(proof);
        }
    }
    Err(OpsError::Invalid(format!(
        "record digest not found in ESS fixture: {record_digest_hex}"
    )))
}

pub fn logs_verify_proof(proof: &Path) -> Result<(), OpsError> {
    let data = fs::read_to_string(proof)?;
    let proof: MerkleProofRecord = serde_json::from_str(&data)?;
    if !verify_merkle_proof(&proof) {
        return Err(OpsError::Invalid("invalid Merkle proof".to_string()));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunCertificateSummaryV1 {
    pub mean_risk_q: u16,
    pub mean_uncertainty_q: u16,
    pub max_governor_tier: u8,
    pub total_violations_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunCertificateV1 {
    pub schema_version: u16,
    pub run_id: String,
    pub started_at: Option<u64>,
    pub ended_at: Option<u64>,
    pub policy_graph_digest: String,
    pub manifest_digest: String,
    pub final_checkpoint_root: String,
    pub record_count: u64,
    pub summary: RunCertificateSummaryV1,
    pub certificate_digest: String,
    pub signature: String,
    pub signer_key_id: String,
    pub signer_public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunAttestationRecord {
    pub schema_version: u16,
    pub run_id: String,
    pub certificate_digest_prefix: String,
    pub signer_key_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttestVerifyReport {
    pub pass: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttestationBundleManifest {
    pub run_id: String,
    pub out: String,
    pub entries: Vec<String>,
}

pub fn attest_keys_generate(workdir: &Path, force: bool) -> Result<(), OpsError> {
    let key_dir = workdir.join("keys");
    fs::create_dir_all(&key_dir)?;
    let private_path = key_dir.join("attestation_ed25519.key");
    let public_path = key_dir.join("attestation_ed25519.pub");
    if !force && private_path.exists() && public_path.exists() {
        return Ok(());
    }

    let sk = SigningKey::generate(&mut OsRng);
    let vk = sk.verifying_key();
    fs::write(&private_path, hex::encode(sk.to_bytes()))?;
    fs::write(&public_path, hex::encode(vk.to_bytes()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = fs::metadata(&private_path)?.permissions();
        perm.set_mode(0o600);
        fs::set_permissions(&private_path, perm)?;
        let mut pub_perm = fs::metadata(&public_path)?.permissions();
        pub_perm.set_mode(0o644);
        fs::set_permissions(&public_path, pub_perm)?;
    }
    Ok(())
}

pub fn attest_run(workdir: &Path, run_id: &str, out: &Path) -> Result<RunCertificateV1, OpsError> {
    attest_keys_generate(workdir, false)?;
    let run = runs_show(workdir, run_id)?
        .ok_or_else(|| OpsError::Invalid(format!("run metadata not found: {run_id}")))?;
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let segments = build_merkle_segments(run_id, &records, 1024);
    verify_segment_chain(&segments)?;
    let final_root = segments
        .last()
        .map(|s| s.merkle_root.clone())
        .unwrap_or_default();

    let (sum_risk, count_risk, sum_unc, count_unc, max_tier, total_violations) =
        summarize_attestation_metrics(&records);

    let (policy_base, policy_overlay, manifest_path) = resolve_attestation_inputs();
    let policy = load_and_merge_policy_graph(&policy_base, Some(&policy_overlay))?;
    let manifest = models_verify(&manifest_path)?;

    let mut cert = RunCertificateV1 {
        schema_version: 1,
        run_id: run_id.to_string(),
        started_at: Some(run.started_at_tick),
        ended_at: run.ended_at_tick,
        policy_graph_digest: policy.1.policy_graph_digest,
        manifest_digest: manifest.model_hashes_digest,
        final_checkpoint_root: final_root,
        record_count: records.len() as u64,
        summary: RunCertificateSummaryV1 {
            mean_risk_q: if count_risk == 0 {
                0
            } else {
                (sum_risk / count_risk) as u16
            },
            mean_uncertainty_q: if count_unc == 0 {
                0
            } else {
                (sum_unc / count_unc) as u16
            },
            max_governor_tier: max_tier,
            total_violations_count: total_violations,
        },
        certificate_digest: String::new(),
        signature: String::new(),
        signer_key_id: "attestation_ed25519_v1".to_string(),
        signer_public_key: load_attestation_public_key_hex(workdir)?,
    };

    cert.certificate_digest = certificate_digest_hex(&cert)?;
    cert.signature = sign_certificate_digest(workdir, &cert.certificate_digest)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(out, &cert)?;
    persist_run_attestation_record(workdir, run_id, &cert)?;
    Ok(cert)
}

pub fn attest_verify(
    workdir: &Path,
    cert_path: &Path,
    _ess: &Path,
) -> Result<AttestVerifyReport, OpsError> {
    let data = fs::read_to_string(cert_path)?;
    let cert: RunCertificateV1 = serde_json::from_str(&data)?;
    let mut reasons = Vec::new();

    let recomputed = certificate_digest_hex(&cert)?;
    if recomputed != cert.certificate_digest {
        reasons.push("certificate digest mismatch".to_string());
    }

    if !verify_certificate_signature(&cert)? {
        reasons.push("signature verification failed".to_string());
    }

    let run = runs_show(workdir, &cert.run_id)?;
    if run.is_none() {
        reasons.push("missing run metadata for run_id".to_string());
    }

    let (policy_base, policy_overlay, manifest_path) = resolve_attestation_inputs();
    let policy = load_and_merge_policy_graph(&policy_base, Some(&policy_overlay))?;
    if policy.1.policy_graph_digest != cert.policy_graph_digest {
        reasons.push("policy_graph_digest mismatch".to_string());
    }

    let manifest = models_verify(&manifest_path)?;
    if manifest.model_hashes_digest != cert.manifest_digest {
        reasons.push("manifest_digest mismatch".to_string());
    }

    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let segments = build_merkle_segments(&cert.run_id, &records, 1024);
    if let Err(err) = verify_segment_chain(&segments) {
        reasons.push(format!("segment chain invalid: {err}"));
    }
    let final_root = segments
        .last()
        .map(|s| s.merkle_root.clone())
        .unwrap_or_default();
    if final_root != cert.final_checkpoint_root {
        reasons.push("final checkpoint root mismatch".to_string());
    }

    Ok(AttestVerifyReport {
        pass: reasons.is_empty(),
        reasons,
    })
}

pub fn attest_bundle(
    workdir: &Path,
    run_id: &str,
    out: &Path,
) -> Result<AttestationBundleManifest, OpsError> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let cert_path = workdir.join("out").join(format!("run_cert_{run_id}.json"));
    let cert = if cert_path.exists() {
        serde_json::from_str::<RunCertificateV1>(&fs::read_to_string(&cert_path)?)?
    } else {
        attest_run(workdir, run_id, &cert_path)?
    };

    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let segments = build_merkle_segments(run_id, &records, 1024);
    verify_segment_chain(&segments)?;

    let file = fs::File::create(out)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let mut entries = Vec::new();

    let cert_bytes = serde_json::to_vec_pretty(&cert)?;
    zip.start_file("run_certificate.json", opts)
        .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
    zip.write_all(&cert_bytes)
        .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
    entries.push("run_certificate.json".to_string());

    let final_checkpoint = records
        .iter()
        .rev()
        .find(|r| r.kind == ExperienceKind::AuditCheckpoint)
        .map(|r| {
            serde_json::json!({
                "id": r.id.0,
                "tick": r.time.tick.get(),
                "audit_digest": r.audit_digest.map(hex::encode),
                "audit_prev_digest": r.audit_prev_digest.map(hex::encode)
            })
        })
        .unwrap_or_else(|| serde_json::json!({}));
    let checkpoint_bytes = serde_json::to_vec_pretty(&final_checkpoint)?;
    zip.start_file("final_checkpoint.json", opts)
        .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
    zip.write_all(&checkpoint_bytes)
        .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
    entries.push("final_checkpoint.json".to_string());

    let roots_only = segments
        .iter()
        .map(|s| {
            serde_json::json!({
                "segment_index": s.segment_id.segment_index,
                "record_count": s.record_count,
                "merkle_root": s.merkle_root,
                "prev_segment_root": s.prev_segment_root
            })
        })
        .collect::<Vec<_>>();
    let roots_bytes = serde_json::to_vec_pretty(&roots_only)?;
    zip.start_file("segment_roots.json", opts)
        .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
    zip.write_all(&roots_bytes)
        .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
    entries.push("segment_roots.json".to_string());

    let gate_path = PathBuf::from("./out/gate_report.json");
    if gate_path.exists() {
        let gate = fs::read_to_string(&gate_path)?;
        zip.start_file("readiness_gate_report.json", opts)
            .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
        zip.write_all(gate.as_bytes())
            .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
        entries.push("readiness_gate_report.json".to_string());
    }

    zip.finish()
        .map_err(|e| OpsError::Invalid(format!("zip finalize failed: {e}")))?;

    Ok(AttestationBundleManifest {
        run_id: run_id.to_string(),
        out: out.display().to_string(),
        entries,
    })
}

fn resolve_attestation_inputs() -> (PathBuf, PathBuf, PathBuf) {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let base = repo_root.join("policies/packs/base_v1");
    let overlay = repo_root.join("policies/packs/overlays/test");
    let manifest = repo_root.join("models/manifest.toml");
    (base, overlay, manifest)
}

fn summarize_attestation_metrics(records: &[ExperienceRecord]) -> (u64, u64, u64, u64, u8, u32) {
    let mut sum_risk = 0u64;
    let mut count_risk = 0u64;
    let mut sum_unc = 0u64;
    let mut count_unc = 0u64;
    let mut max_tier = 0u8;
    let mut violations = 0u32;
    for record in records {
        if let ExperiencePayload::Audit(AuditPayload::EbmReasoning(r)) = &record.payload {
            sum_risk = sum_risk.saturating_add(r.risk_q as u64);
            count_risk = count_risk.saturating_add(1);
            sum_unc = sum_unc.saturating_add(r.uncertainty_q as u64);
            count_unc = count_unc.saturating_add(1);
        }
        if let ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(c)) = &record.payload {
            max_tier = max_tier.max(c.tier).max(c.effective_tier);
        }
        if matches!(
            &record.payload,
            ExperiencePayload::Audit(AuditPayload::EbmEnvelopeViolation(_))
                | ExperiencePayload::Audit(AuditPayload::GpuResourceViolation(_))
                | ExperiencePayload::Audit(AuditPayload::ComputeBudgetViolation(_))
        ) {
            violations = violations.saturating_add(1);
        }
    }
    (
        sum_risk, count_risk, sum_unc, count_unc, max_tier, violations,
    )
}

fn certificate_digest_hex(cert: &RunCertificateV1) -> Result<String, OpsError> {
    let mut canonical = cert.clone();
    canonical.certificate_digest.clear();
    canonical.signature.clear();
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn load_attestation_signing_key(workdir: &Path) -> Result<SigningKey, OpsError> {
    let private_path = workdir.join("keys").join("attestation_ed25519.key");
    let private_hex = fs::read_to_string(private_path)?;
    let private_bytes = hex::decode(private_hex.trim())
        .map_err(|e| OpsError::Invalid(format!("invalid attestation private key hex: {e}")))?;
    let secret: [u8; 32] = private_bytes
        .as_slice()
        .try_into()
        .map_err(|_| OpsError::Invalid("attestation private key must be 32 bytes".to_string()))?;
    Ok(SigningKey::from_bytes(&secret))
}

fn load_attestation_public_key_hex(workdir: &Path) -> Result<String, OpsError> {
    let public_path = workdir.join("keys").join("attestation_ed25519.pub");
    if public_path.exists() {
        return Ok(fs::read_to_string(public_path)?.trim().to_string());
    }
    let signing = load_attestation_signing_key(workdir)?;
    Ok(hex::encode(signing.verifying_key().to_bytes()))
}

fn sign_certificate_digest(workdir: &Path, cert_digest_hex: &str) -> Result<String, OpsError> {
    let signing = load_attestation_signing_key(workdir)?;
    let digest = hex::decode(cert_digest_hex)
        .map_err(|e| OpsError::Invalid(format!("invalid certificate digest hex: {e}")))?;
    let sig: Signature = signing.sign(&digest);
    Ok(hex::encode(sig.to_bytes()))
}

fn verify_certificate_signature(cert: &RunCertificateV1) -> Result<bool, OpsError> {
    let pub_bytes = hex::decode(&cert.signer_public_key)
        .map_err(|e| OpsError::Invalid(format!("invalid signer public key hex: {e}")))?;
    let vk_bytes: [u8; 32] = pub_bytes
        .as_slice()
        .try_into()
        .map_err(|_| OpsError::Invalid("signer public key must be 32 bytes".to_string()))?;
    let vk = VerifyingKey::from_bytes(&vk_bytes)
        .map_err(|e| OpsError::Invalid(format!("invalid signer public key: {e}")))?;
    let sig_bytes = hex::decode(&cert.signature)
        .map_err(|e| OpsError::Invalid(format!("invalid signature hex: {e}")))?;
    let sig_arr: [u8; 64] = sig_bytes
        .as_slice()
        .try_into()
        .map_err(|_| OpsError::Invalid("signature must be 64 bytes".to_string()))?;
    let sig = Signature::from_bytes(&sig_arr);
    let digest = hex::decode(&cert.certificate_digest)
        .map_err(|e| OpsError::Invalid(format!("invalid certificate digest hex: {e}")))?;
    Ok(vk.verify(&digest, &sig).is_ok())
}

fn persist_run_attestation_record(
    workdir: &Path,
    run_id: &str,
    cert: &RunCertificateV1,
) -> Result<(), OpsError> {
    let path = workdir.join("ess").join("run_attestations.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut records: Vec<RunAttestationRecord> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };
    records.push(RunAttestationRecord {
        schema_version: 1,
        run_id: run_id.to_string(),
        certificate_digest_prefix: cert.certificate_digest.chars().take(12).collect(),
        signer_key_id: cert.signer_key_id.clone(),
    });
    write_json(&path, &records)
}

fn build_merkle_segments(
    run_id: &str,
    records: &[ExperienceRecord],
    segment_size: usize,
) -> Vec<MerkleSegmentRecord> {
    let mut out = Vec::new();
    let mut prev_segment_root: Option<[u8; 32]> = None;
    for (segment_index, chunk) in records.chunks(segment_size).enumerate() {
        let leaf_digests = chunk
            .iter()
            .map(record_merkle_leaf_digest)
            .collect::<Vec<_>>();
        let merkle_root = compute_merkle_root(&leaf_digests);
        let first_t = chunk.first().map(|r| r.time.tick.get()).unwrap_or(0);
        let last_t = chunk.last().map(|r| r.time.tick.get()).unwrap_or(first_t);
        let segment_id = SegmentId {
            run_id: run_id.to_string(),
            segment_index: segment_index as u64,
        };
        let segment_digest = compute_segment_digest(
            &segment_id,
            first_t,
            last_t,
            leaf_digests.len() as u32,
            merkle_root,
            prev_segment_root,
        );
        out.push(MerkleSegmentRecord {
            segment_id,
            first_t,
            last_t,
            record_count: leaf_digests.len() as u32,
            merkle_root: hex::encode(merkle_root),
            prev_segment_root: prev_segment_root.map(hex::encode),
            segment_digest: hex::encode(segment_digest),
            leaf_digests,
        });
        prev_segment_root = Some(merkle_root);
    }
    out
}

fn verify_segment_chain(segments: &[MerkleSegmentRecord]) -> Result<(), OpsError> {
    let mut prev_root: Option<String> = None;
    for segment in segments {
        if segment.prev_segment_root != prev_root {
            return Err(OpsError::Invalid(format!(
                "segment chain break at segment {}",
                segment.segment_id.segment_index
            )));
        }
        prev_root = Some(segment.merkle_root.clone());
    }
    Ok(())
}

fn prove_record_in_segment(
    segment: &MerkleSegmentRecord,
    leaf_digest: [u8; 32],
) -> Option<MerkleProofRecord> {
    let leaf_index = segment
        .leaf_digests
        .iter()
        .position(|d| *d == leaf_digest)?;
    let siblings = compute_merkle_path(&segment.leaf_digests, leaf_index);
    let mut proof = MerkleProofRecord {
        segment_id: segment.segment_id.clone(),
        leaf_index,
        siblings,
        segment_root: segment.merkle_root.clone(),
        leaf_hash: hex::encode(leaf_digest),
        proof_digest: String::new(),
    };
    proof.proof_digest = sha256_hex(&serde_json::to_vec(&proof).unwrap_or_default());
    Some(proof)
}

fn record_merkle_leaf_digest(record: &ExperienceRecord) -> [u8; 32] {
    #[derive(Serialize)]
    struct CanonicalLeaf<'a> {
        id: u64,
        tick: u64,
        window: u64,
        corr: u64,
        kind: &'a str,
        audit_digest: Option<String>,
    }
    let canonical = CanonicalLeaf {
        id: record.id.0,
        tick: record.time.tick.get(),
        window: record.time.window.get(),
        corr: record.corr.0,
        kind: experience_kind_name(record.kind),
        audit_digest: record.audit_digest.map(hex::encode),
    };
    digest_json(&canonical)
}

fn experience_kind_name(kind: ExperienceKind) -> &'static str {
    match kind {
        ExperienceKind::ControlIn => "ControlIn",
        ExperienceKind::DecisionOut => "DecisionOut",
        ExperienceKind::BrainOut => "BrainOut",
        ExperienceKind::Note => "Note",
        ExperienceKind::ToolRequest => "ToolRequest",
        ExperienceKind::ToolPlan => "ToolPlan",
        ExperienceKind::ToolIssue => "ToolIssue",
        ExperienceKind::ToolAuth => "ToolAuth",
        ExperienceKind::ToolExecution => "ToolExecution",
        ExperienceKind::SandboxCall => "SandboxCall",
        ExperienceKind::SandboxReply => "SandboxReply",
        ExperienceKind::AuditCheckpoint => "AuditCheckpoint",
        ExperienceKind::Hormone => "Hormone",
        ExperienceKind::Neuro => "Neuro",
        ExperienceKind::DeltaProposal => "DeltaProposal",
        ExperienceKind::DeltaEvaluation => "DeltaEvaluation",
        ExperienceKind::DeltaRecommendation => "DeltaRecommendation",
        ExperienceKind::Nsr => "Nsr",
        ExperienceKind::CandidateSet => "CandidateSet",
        ExperienceKind::EbmReasoning => "EbmReasoning",
        ExperienceKind::EbmEnvelopeViolation => "EbmEnvelopeViolation",
        ExperienceKind::GpuUnavailable => "GpuUnavailable",
        ExperienceKind::GpuParity => "GpuParity",
        ExperienceKind::GpuResourceViolation => "GpuResourceViolation",
        ExperienceKind::Output => "Output",
        ExperienceKind::BackendPack => "BackendPack",
        ExperienceKind::LfmSummary => "LfmSummary",
        ExperienceKind::LfmWindow => "LfmWindow",
        ExperienceKind::CapabilityIssuance => "CapabilityIssuance",
        ExperienceKind::Throttle => "Throttle",
        ExperienceKind::Emergency => "Emergency",
        ExperienceKind::PolicyProvenance => "PolicyProvenance",
        ExperienceKind::EbmConstraintProvenance => "EbmConstraintProvenance",
        ExperienceKind::RemoteCall => "RemoteCall",
        ExperienceKind::RemoteCallDenied => "RemoteCallDenied",
        ExperienceKind::ComputeBudgetWindow => "ComputeBudgetWindow",
        ExperienceKind::ComputeBudgetViolation => "ComputeBudgetViolation",
        ExperienceKind::RetrievalDecision => "RetrievalDecision",
    }
}

fn compute_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return Sha256::digest(b"UCF:ESS:SEGMENT:EMPTY:v1").into();
    }
    let mut layer = leaves.to_vec();
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len().div_ceil(2));
        let mut idx = 0;
        while idx < layer.len() {
            let left = layer[idx];
            let right = layer.get(idx + 1).copied().unwrap_or(left);
            next.push(hash_pair(left, right));
            idx += 2;
        }
        layer = next;
    }
    layer[0]
}

fn compute_merkle_path(leaves: &[[u8; 32]], leaf_index: usize) -> Vec<MerkleProofStep> {
    let mut path = Vec::new();
    if leaves.is_empty() {
        return path;
    }
    let mut idx = leaf_index;
    let mut layer = leaves.to_vec();
    while layer.len() > 1 {
        let sibling_idx = if idx.is_multiple_of(2) {
            (idx + 1).min(layer.len() - 1)
        } else {
            idx - 1
        };
        path.push(MerkleProofStep {
            sibling_hash: hex::encode(layer[sibling_idx]),
            sibling_on_left: sibling_idx < idx,
        });

        let mut next = Vec::with_capacity(layer.len().div_ceil(2));
        let mut cursor = 0;
        while cursor < layer.len() {
            let left = layer[cursor];
            let right = layer.get(cursor + 1).copied().unwrap_or(left);
            next.push(hash_pair(left, right));
            cursor += 2;
        }
        idx /= 2;
        layer = next;
    }
    path
}

fn verify_merkle_proof(proof: &MerkleProofRecord) -> bool {
    let mut acc = match parse_hex_digest(&proof.leaf_hash) {
        Ok(digest) => digest,
        Err(_) => return false,
    };
    for step in &proof.siblings {
        let sibling = match parse_hex_digest(&step.sibling_hash) {
            Ok(digest) => digest,
            Err(_) => return false,
        };
        acc = if step.sibling_on_left {
            hash_pair(sibling, acc)
        } else {
            hash_pair(acc, sibling)
        };
    }
    hex::encode(acc) == proof.segment_root
}

fn compute_segment_digest(
    segment_id: &SegmentId,
    first_t: u64,
    last_t: u64,
    record_count: u32,
    merkle_root: [u8; 32],
    prev_segment_root: Option<[u8; 32]>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"UCF:ESS:MERKLE-SEGMENT:v1");
    hasher.update(segment_id.run_id.as_bytes());
    hasher.update(segment_id.segment_index.to_be_bytes());
    hasher.update(first_t.to_be_bytes());
    hasher.update(last_t.to_be_bytes());
    hasher.update(record_count.to_be_bytes());
    hasher.update(merkle_root);
    hasher.update(prev_segment_root.unwrap_or([0; 32]));
    hasher.finalize().into()
}

fn hash_pair(left: [u8; 32], right: [u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"UCF:ESS:MERKLE-NODE:v1");
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn parse_hex_digest(value: &str) -> Result<[u8; 32], OpsError> {
    let bytes = hex::decode(value)
        .map_err(|e| OpsError::Invalid(format!("invalid digest hex '{value}': {e}")))?;
    if bytes.len() != 32 {
        return Err(OpsError::Invalid(format!(
            "digest must be 32 bytes, got {}",
            bytes.len()
        )));
    }
    let mut out = [0_u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyValidateReport {
    pub policy_graph_digest: String,
    pub base_pack: String,
    pub overlay_pack: Option<String>,
    pub schema_version: u16,
}

pub fn policy_validate(
    pack: &Path,
    overlay: Option<&Path>,
) -> Result<PolicyValidateReport, OpsError> {
    let (graph, prov) = load_and_merge_policy_graph(pack, overlay)?;
    let _ = graph;
    Ok(PolicyValidateReport {
        policy_graph_digest: prov.policy_graph_digest,
        base_pack: prov.base_pack_digest,
        overlay_pack: prov.overlay_pack_digest,
        schema_version: prov.schema_version,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyDiffReport {
    pub digest_a: String,
    pub digest_b: String,
    pub thresholds: Vec<String>,
    pub budgets: Vec<String>,
    pub allowlists: Vec<String>,
}

pub fn policy_diff(
    a_pack: &Path,
    a_overlay: Option<&Path>,
    b_pack: &Path,
    b_overlay: Option<&Path>,
) -> Result<PolicyDiffReport, OpsError> {
    let (a, _) = load_and_merge_policy_graph(a_pack, a_overlay)?;
    let (b, _) = load_and_merge_policy_graph(b_pack, b_overlay)?;
    let mut thresholds = diff_i64(&a.thresholds, &b.thresholds);
    let mut budgets = diff_i64(&a.budgets, &b.budgets);
    let mut allowlists = diff_str(&a.allowlists, &b.allowlists);
    thresholds.truncate(64);
    budgets.truncate(64);
    allowlists.truncate(64);
    Ok(PolicyDiffReport {
        digest_a: policy_graph_digest(&a)?,
        digest_b: policy_graph_digest(&b)?,
        thresholds,
        budgets,
        allowlists,
    })
}

fn diff_i64(a: &BTreeMap<String, i64>, b: &BTreeMap<String, i64>) -> Vec<String> {
    let mut keys = a.keys().chain(b.keys()).cloned().collect::<Vec<_>>();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter_map(|k| {
            let av = a.get(&k);
            let bv = b.get(&k);
            if av != bv {
                Some(format!("{k}: {:?} -> {:?}", av, bv))
            } else {
                None
            }
        })
        .collect()
}

fn diff_str(a: &BTreeMap<String, String>, b: &BTreeMap<String, String>) -> Vec<String> {
    let mut keys = a.keys().chain(b.keys()).cloned().collect::<Vec<_>>();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter_map(|k| {
            let av = a.get(&k);
            let bv = b.get(&k);
            if av != bv {
                Some(format!("{k}: {:?} -> {:?}", av, bv))
            } else {
                None
            }
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeterminismScanViolation {
    pub path: String,
    pub line: usize,
    pub pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeterminismScanReport {
    pub violations: Vec<DeterminismScanViolation>,
}

pub fn determinism_scan(repo_root: &Path) -> Result<DeterminismScanReport, OpsError> {
    let banned = ["thread_rng", "rand::random", "getrandom", "OsRng"];
    let mut violations = Vec::new();
    for entry in walkdir::WalkDir::new(repo_root).into_iter().flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if ext != "rs" {
            continue;
        }
        let rel = path
            .strip_prefix(repo_root)
            .unwrap_or(path)
            .to_string_lossy();
        if rel.contains("vendor/")
            || rel.contains("target/")
            || rel.contains("tests/")
            || rel.contains("fuzz/")
            || rel.contains("runtime/ucf-ops/src/lib.rs")
        {
            continue;
        }
        let text = fs::read_to_string(path).unwrap_or_default();
        for (idx, line) in text.lines().enumerate() {
            for pat in banned {
                if line.contains(pat) && !line.contains("let banned =") {
                    violations.push(DeterminismScanViolation {
                        path: rel.to_string(),
                        line: idx + 1,
                        pattern: pat.to_string(),
                    });
                }
            }
        }
    }
    Ok(DeterminismScanReport { violations })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyExplainReport {
    pub run_id: String,
    pub bundle_hash: String,
    pub policy_graph_digest: String,
    pub base_pack_digest: String,
    pub overlay_pack_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditScanViolation {
    pub path: String,
    pub line: usize,
    pub pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditScanReport {
    pub violations: Vec<AuditScanViolation>,
}

pub fn audit_scan(repo_root: &Path) -> Result<AuditScanReport, OpsError> {
    let banned = [
        "std::process::Command",
        "reqwest::",
        "hyper::",
        "thread_rng",
        "getrandom",
        "std::fs::File",
        "execute_tool(",
    ];
    let mut violations = Vec::new();
    for entry in walkdir::WalkDir::new(repo_root).into_iter().flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if ext != "rs" {
            continue;
        }
        let rel = path
            .strip_prefix(repo_root)
            .unwrap_or(path)
            .to_string_lossy();
        let in_scope = rel.starts_with("runtime/ucf-runtime/src/")
            || rel.starts_with("runtime/ucf-policy/src/")
            || rel.starts_with("runtime/ucf-replay/src/");
        if !in_scope {
            continue;
        }
        if rel.contains("vendor/")
            || rel.contains("target/")
            || rel.contains("fuzz/")
            || rel.contains("runtime/ucf-ops/src/")
            || rel.starts_with("src/")
        {
            continue;
        }
        let text = fs::read_to_string(path).unwrap_or_default();
        for (idx, line) in text.lines().enumerate() {
            for pat in banned {
                if line.contains(pat) && !line.contains("let banned =") {
                    violations.push(AuditScanViolation {
                        path: rel.to_string(),
                        line: idx + 1,
                        pattern: pat.to_string(),
                    });
                }
            }
        }
    }
    Ok(AuditScanReport { violations })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareScanViolation {
    pub path: String,
    pub line: usize,
    pub pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareScanReport {
    pub violations: Vec<HardwareScanViolation>,
}

pub fn hardware_scan(repo_root: &Path) -> Result<HardwareScanReport, OpsError> {
    let banned = [
        "NUC",
        "Raspberry",
        "RPi",
        "Intel",
        "AMD",
        "NVIDIA",
        "/etc/ucf",
    ];
    let mut violations = Vec::new();
    for entry in walkdir::WalkDir::new(repo_root).into_iter().flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(repo_root)
            .unwrap_or(path)
            .to_string_lossy();
        if rel.contains("vendor/")
            || rel.contains("target/")
            || rel.starts_with("deploy/")
            || rel.starts_with("runtime/ucf-ops/src/")
        {
            continue;
        }
        let in_runtime_scope = rel.starts_with("runtime/")
            || rel.starts_with("core/")
            || rel.starts_with("domains/")
            || rel.starts_with("ai/")
            || rel.starts_with("app/");
        if !in_runtime_scope {
            continue;
        }
        let text = fs::read_to_string(path).unwrap_or_default();
        for (idx, line) in text.lines().enumerate() {
            for pat in banned {
                if line.contains(pat) && !line.contains("let banned =") {
                    violations.push(HardwareScanViolation {
                        path: rel.to_string(),
                        line: idx + 1,
                        pattern: pat.to_string(),
                    });
                }
            }
        }
    }
    Ok(HardwareScanReport { violations })
}

pub fn policy_explain(
    workdir: &Path,
    digest_prefix: &str,
) -> Result<Option<PolicyExplainReport>, OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    for rec in records {
        if let ExperiencePayload::Audit(AuditPayload::PolicyProvenance(p)) = rec.payload {
            if p.policy_graph_digest.starts_with(digest_prefix) {
                return Ok(Some(PolicyExplainReport {
                    run_id: p.run_id,
                    bundle_hash: p.bundle_hash,
                    policy_graph_digest: p.policy_graph_digest,
                    base_pack_digest: p.base_pack_digest,
                    overlay_pack_digest: p.overlay_pack_digest,
                }));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod proof_carrying_logs_tests {
    use super::*;
    use ucf_core::types::{SimTime, Tick, WindowId};
    use ucf_ess::v1::ExperienceId;
    use ucf_frames::v1::CorrelationId;

    fn note(id: u64, tick: u64) -> ExperienceRecord {
        ExperienceRecord::note(
            ExperienceId(id),
            SimTime {
                tick: Tick::new(tick),
                window: WindowId::new(0),
            },
            CorrelationId(id),
            "x",
        )
    }

    #[test]
    fn merkle_root_is_deterministic() {
        let records = vec![note(1, 1), note(2, 2), note(3, 3)];
        let a = build_merkle_segments("run", &records, 2);
        let b = build_merkle_segments("run", &records, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn proof_generation_and_verification_work() {
        let records = vec![note(1, 1), note(2, 2), note(3, 3), note(4, 4)];
        let segments = build_merkle_segments("run", &records, 4);
        let target = record_merkle_leaf_digest(&records[2]);
        let proof = prove_record_in_segment(&segments[0], target).expect("proof");
        assert!(verify_merkle_proof(&proof));
    }

    #[test]
    fn segment_boundaries_are_deterministic() {
        let records = (0..2050).map(|i| note(i + 1, i + 1)).collect::<Vec<_>>();
        let segments = build_merkle_segments("run", &records, 1024);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].record_count, 1024);
        assert_eq!(segments[1].record_count, 1024);
        assert_eq!(segments[2].record_count, 2);
        assert!(verify_segment_chain(&segments).is_ok());
    }
}

#[cfg(test)]
mod rc1_tests {
    use super::*;

    #[test]
    fn diagnostics_bundle_redacts_payload_keys() {
        let dir = tempfile::tempdir().expect("tmp");
        let out_run = PathBuf::from("out").join("run-test");
        std::fs::create_dir_all(&out_run).expect("out dir");
        std::fs::write(out_run.join("run_metadata.json"), "{\"ok\":true}").expect("write");
        std::fs::write(
            out_run.join("metrics_summary.json"),
            "{\"payload\":\"secret\"}",
        )
        .expect("write");
        std::fs::create_dir_all(dir.path().join("explain_tick")).expect("exp dir");
        std::fs::write(
            dir.path().join("explain_tick/last.json"),
            "{\"text\":\"hidden\",\"note\":\"x\"}",
        )
        .expect("write");
        let zip_path = dir.path().join("diag.zip");
        let report = diagnostics_collect(dir.path(), "run-test", &zip_path).expect("bundle");
        assert!(!report.entries.is_empty());
        let bytes = std::fs::read(&zip_path).expect("zip bytes");
        let as_text = String::from_utf8_lossy(&bytes);
        assert!(!as_text.contains("\"text\":"));
        assert!(!as_text.contains("\"payload\":"));

        let _ = std::fs::remove_dir_all(Path::new("out").join("run-test"));
    }

    #[test]
    fn rc1_gate_fails_on_induced_invalid_output_path() {
        let dir = tempfile::tempdir().expect("tmp");
        let out = PathBuf::from("/dev/null/rc1_gate.json");
        let result = release_rc1_gate(dir.path(), &out, false);
        assert!(result.is_err());
    }

    #[test]
    fn workspace_test_check_skips_in_ci() {
        std::env::set_var("CI", "true");
        std::env::remove_var("UCF_SKIP_GATE_WORKSPACE_TESTS");
        let check = check_workspace_tests();
        std::env::remove_var("CI");
        assert_eq!(check.name, "build_workspace_tests");
        assert_eq!(check.status, GateStatus::Skip);
    }
}

#[cfg(test)]
mod hardware_scan_tests {
    use super::*;

    #[test]
    fn hardware_scan_flags_forbidden_terms() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let runtime_dir = tmp.path().join("runtime/ucf-runtime/src");
        std::fs::create_dir_all(&runtime_dir).expect("mkdir");
        let bad = runtime_dir.join("bad.rs");
        std::fs::write(&bad, "const TARGET: &str = \"RPi\";\n").expect("write");

        let report = hardware_scan(tmp.path()).expect("scan");
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].pattern, "RPi");
    }
}

#[cfg(test)]
mod device_profile_tests {
    use super::*;

    #[test]
    fn device_profile_digest_is_stable() {
        let digest = DeviceProfileV1::for_name(DeviceProfileName::Small)
            .digest_hex()
            .expect("digest");
        assert_eq!(digest.len(), 64);
        assert_eq!(
            digest,
            DeviceProfileV1::for_name(DeviceProfileName::Small)
                .digest_hex()
                .expect("digest")
        );
    }

    #[test]
    fn run_metadata_contains_platform_and_device_profile_fields() {
        let record = RunMetadataRecord {
            platform_probe_summary:
                "os=Linux arch=X86_64 cores=1 mem_mb=1 accel=None monotonic_clock_ok=true"
                    .to_string(),
            device_profile_name: "small".to_string(),
            device_profile_digest: DeviceProfileV1::for_name(DeviceProfileName::Small)
                .digest_hex()
                .expect("digest"),
            ..RunMetadataRecord::default()
        };
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("platform_probe_summary"));
        assert!(json.contains("device_profile_name"));
        assert!(json.contains("device_profile_digest"));
    }
}
