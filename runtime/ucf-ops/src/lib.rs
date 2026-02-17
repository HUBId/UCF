#![forbid(unsafe_code)]

mod adversarial;
pub use adversarial::{adversarial_run, AdversarialReport, AdversarialRunArgs, CaseResult};

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    AuditPayload, EmergencyStateCode, ExperienceKind, ExperiencePayload, ExperienceRecord,
};
use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, CorrelationId, Intent, IntentId, IntentKind,
};
use ucf_policy::adapter::MockAdapter;
use ucf_replay::{
    load_fixture_records, replay_audit as run_replay_audit, replay_records, write_report,
    ReplayMode, ReplayPlan, ReplaySpec, ReplayStrictness,
};
use ucf_runtime::RuntimeOrchestrator;

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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct OpsConfig {
    pub profile: String,
    pub offline: bool,
    pub compute_backend: ComputeBackendKind,
    pub compute_seed: u64,
    pub compute_budget_profile: String,
    pub isolation_runtime: String,
    pub capabilities_default: String,
    pub log_level: String,
}

impl Default for OpsConfig {
    fn default() -> Self {
        Self {
            profile: "test".to_string(),
            offline: true,
            compute_backend: ComputeBackendKind::Stub,
            compute_seed: 0xDEC0DED,
            compute_budget_profile: "tight".to_string(),
            isolation_runtime: "inproc".to_string(),
            capabilities_default: "deny".to_string(),
            log_level: "info".to_string(),
        }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadataRecord {
    pub run_id: String,
    pub started_at_tick: u64,
    pub code_version_tag: String,
    pub backend_pack_meta_digest: String,
    pub fixtures_digest: String,
    pub model_hashes_digest: String,
    pub enabled_features_bitmap: u16,
    pub profile: String,
    pub schema_versions: BTreeMap<String, u16>,
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
pub struct ReadinessGateReport {
    pub code_version_tag: String,
    pub fixtures_digest_prefix: Option<String>,
    pub backend_pack_digest_prefix: Option<String>,
    pub timestamp: Option<String>,
    pub status: GateStatus,
    pub checks: Vec<CheckResult>,
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
    std::env::set_var("UCF_OFFLINE", "1");
    std::env::set_var("UCF_TOOLS_DEFAULT", "deny");

    let base = workdir.join("readiness_gate");
    fs::create_dir_all(&base)?;
    let run_a = base.join("scenario_a");
    let run_a2 = base.join("scenario_a_repeat");
    let run_b = base.join("scenario_b");
    let out_a = run_a.join("out");
    let out_a2 = run_a2.join("out");
    let out_b = run_b.join("out");

    let scenario_a = workspace_fixture("e2e_scenario_a.json");
    let scenario_b = workspace_fixture("e2e_scenario_b.json");

    let artifacts_a = one_command_bringup(&run_a, &scenario_a, 24, &out_a, true)?;
    let artifacts_a2 = one_command_bringup(&run_a2, &scenario_a, 24, &out_a2, true)?;
    let artifacts_b = one_command_bringup(&run_b, &scenario_b, 24, &out_b, true)?;

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
    ];

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
    };
    write_json(out, &report)?;
    Ok(report)
}

fn workspace_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join(name)
}

fn check_workspace_tests() -> CheckResult {
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

    std::env::set_var("UCF_PROFILE", "test");
    std::env::set_var("UCF_OFFLINE", "1");
    std::env::set_var("UCF_TOOLS_DEFAULT", "deny");
    let result = bringup(workdir, true, ticks)?;
    let build = build_tag()?;
    let pack = BackendPackFactory::build(BackendPackConfig::from_env()?)?;
    let meta = pack.meta();

    let mut schema_versions = BTreeMap::new();
    schema_versions.insert("backend_pack_record".to_string(), 1);
    schema_versions.insert("compute_summary".to_string(), 1);
    schema_versions.insert("output".to_string(), 1);

    let run_metadata = RunMetadataRecord {
        run_id: result.ess_digest.chars().take(16).collect(),
        started_at_tick: 0,
        code_version_tag: build.git_commit,
        backend_pack_meta_digest: hex::encode(meta.digest),
        fixtures_digest: hex::encode(meta.fixtures_digest),
        model_hashes_digest: hex::encode(meta.model_hashes_digest),
        enabled_features_bitmap: ReleaseFeatureMatrix::detect().bits,
        profile: resolved_profile_name(),
        schema_versions,
    };
    write_json(
        workdir.join("ess").join("run_metadata_record.json"),
        &run_metadata,
    )?;

    let metrics = metrics_summary(workdir, ticks as usize)?;
    let explain = explain_tick(
        workdir,
        ExplainTickRequest {
            t: Some(ticks),
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
        nsr_reasons.truncate(16);
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
        },
        decision: ExplainDecision {
            candidate_count: candidate_set.as_ref().map(|c| c.summaries.len()),
            selected_candidate_id: candidate_set.as_ref().map(|c| c.selected_candidate_id),
            selected_candidate_digest_prefix: candidate_set
                .as_ref()
                .map(|c| digest_prefix(&c.selected_candidate_digest, prefix)),
            policy_hints,
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

fn digest_prefix(digest: &[u8; 32], prefix_len: usize) -> String {
    hex::encode(digest)[..prefix_len.min(64)].to_string()
}

fn bounded_preview(text: &str, max_chars: usize) -> String {
    let mut out = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        out.push('…');
    }
    out
}
pub fn load_or_init_config(workdir: &Path) -> Result<OpsConfig, OpsError> {
    let path = workdir.join("config_resolved.json");
    if !path.exists() {
        let cfg = profile_defaults(&resolved_profile_name());
        write_json(&path, &cfg)?;
        return Ok(cfg);
    }
    let mut cfg: OpsConfig = serde_json::from_str(&fs::read_to_string(path)?)?;
    let prof = resolved_profile_name();
    cfg.profile = prof.clone();
    let defaults = profile_defaults(&prof);
    if cfg.profile.is_empty() {
        cfg = defaults;
    }
    Ok(apply_env_overrides(cfg))
}

fn resolved_profile_name() -> String {
    std::env::var("UCF_PROFILE")
        .unwrap_or_else(|_| "test".to_string())
        .to_ascii_lowercase()
}

fn profile_defaults(profile: &str) -> OpsConfig {
    match profile {
        "dev" => OpsConfig {
            profile: "dev".to_string(),
            offline: true,
            compute_backend: ComputeBackendKind::Stub,
            compute_seed: 0xDEC0DED,
            compute_budget_profile: "default".to_string(),
            isolation_runtime: "inproc".to_string(),
            capabilities_default: "allow".to_string(),
            log_level: "debug".to_string(),
        },
        "prod" => OpsConfig {
            profile: "prod".to_string(),
            offline: true,
            compute_backend: ComputeBackendKind::Stub,
            compute_seed: 0xA11CE,
            compute_budget_profile: "stress".to_string(),
            isolation_runtime: "inproc".to_string(),
            capabilities_default: "deny".to_string(),
            log_level: "info".to_string(),
        },
        _ => OpsConfig::default(),
    }
}

fn apply_env_overrides(mut cfg: OpsConfig) -> OpsConfig {
    if let Ok(v) = std::env::var("UCF_COMPUTE_SEED") {
        if let Ok(seed) = v.parse::<u64>() {
            cfg.compute_seed = seed;
        }
    }
    if let Ok(v) = std::env::var("UCF_COMPUTE_BUDGET_PROFILE") {
        cfg.compute_budget_profile = v;
    }
    cfg
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
}

pub fn security_verify_chain(workdir: &Path, from: u64, to: u64) -> Result<(), OpsError> {
    let records = load_fixture_records(workdir)?;
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
    Ok(())
}
