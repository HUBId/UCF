use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ucf_compute::ModelSlot;
use ucf_ess::v1::{AuditPayload, ExperiencePayload};
use ucf_replay::load_fixture_records;

use crate::remediation::merge_canonical_remediations;
use crate::second_slot_parity::{OptionalBackendSupportStateV1, SecondSlotParityReportV1};
use crate::{prefix_hex, sha256_hex, OpsError};

const MANIFEST_HISTORY_KEEP: usize = 20;
const MODEL_FILE_NAME: &str = "model.safetensors";
const MANIFEST_PATH: &str = "models/MANIFEST.toml";
const LEGACY_MANIFEST_PATH: &str = "models/lifecycle_manifest.toml";
const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_SLOT_FILES: usize = 512;
const PROBE_SCHEMA_VERSION: u16 = 1;
const PROBE_DIGEST_PREFIX_LEN: usize = 16;
const SHADOW_READY_SCHEMA_VERSION: u16 = 1;
const SHADOW_READY_MAX_SLOTS: usize = 2;
const ELIGIBILITY_SCHEMA_VERSION: u16 = 1;
const BACKEND_EVIDENCE_SNAPSHOT_SCHEMA_VERSION: u16 = 1;
const SUPPORTED_REAL_SLOT_SET_VERSION: &str = "v3_supported_real_slots_max2";
const SLOT_SET_MAX: usize = 2;
const PROBE_OUTPUT_CAP: usize = 8;
const PROBE_NOTES_CAP: usize = 8;
const SAE_SPIKE_COUNT_KMAX: u32 = 1024;
const Q01_MAX: u16 = 10_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelFileEntry {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LifecycleSlotManifest {
    pub slot_id: String,
    pub active_hash: Option<String>,
    pub files: Vec<ModelFileEntry>,
    pub max_bytes: u64,
    pub contract_versions_supported: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LifecycleManifest {
    pub manifest_version: u16,
    pub created_at: Option<u64>,
    pub slots: Vec<LifecycleSlotManifest>,
    pub manifest_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageResult {
    pub slot: String,
    pub hash: String,
    pub files: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LifecycleActionReport {
    pub shadow_report_digest_prefix: Option<String>,
    pub action: String,
    pub slot: String,
    pub from_hash: Option<String>,
    pub to_hash: String,
    pub old_manifest_digest_prefix: Option<String>,
    pub new_manifest_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub probe_report_digest_prefix: Option<String>,
    pub readiness_gate_digest_prefix: Option<String>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsListReport {
    pub slot: String,
    pub active_hash: Option<String>,
    pub staged_hashes: Vec<String>,
    pub promoted_hashes: Vec<String>,
    pub current_mode: String,
    pub probe_ready: bool,
    pub shadow_ready: bool,
    pub active_eligible: bool,
    pub denial_reason_probe: Option<String>,
    pub denial_reason_shadow: Option<String>,
    pub denial_reason_active: Option<String>,
    pub last_evidence_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EligibilityOverallStatusV1 {
    NoneReady,
    ProbeOnly,
    ShadowReadyPartial,
    ShadowReadyAll,
    ActiveEligiblePartial,
    ActiveEligibleAll,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnifiedEligibilityStatusV1 {
    pub slot_id: String,
    pub target_hash_prefix: String,
    pub manifest_digest_prefix: String,
    pub probe_ready: bool,
    pub shadow_ready: bool,
    pub active_eligible: bool,
    pub latest_probe_digest_prefix: String,
    pub latest_shadow_evidence_digest_prefix: String,
    pub latest_active_evidence_digest_prefix: String,
    pub latest_drift_status: DriftStatusV1,
    pub burn_support_state: OptionalBackendSupportStateV1,
    pub burn_parity_present: bool,
    pub denial_reason_probe: Option<String>,
    pub denial_reason_shadow: Option<String>,
    pub denial_reason_active: Option<String>,
    pub remediation_codes: Vec<String>,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
    pub status_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedRealSlotSetV1 {
    pub schema_version: u16,
    pub slots: Vec<String>,
    pub source: String,
    pub set_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceFreshnessPolicyV1 {
    pub probe_max_age_ticks: u64,
    pub compare_max_age_ticks: u64,
    pub no_impact_max_age_ticks: u64,
    pub drift_max_age_ticks: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceDenialCodeV1 {
    NoProbe,
    StaleProbe,
    NoCompare,
    StaleCompare,
    HashMismatch,
    DriftSevere,
    DriftWarn,
    ActiveNotEnabled,
    UnsupportedSlotSet,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SlotEvidenceSnapshotV1 {
    pub slot_id: String,
    pub manifest_digest_prefix: String,
    pub target_hash_prefix: String,
    pub latest_probe_report_digest_prefix: String,
    pub latest_compare_window_digest_prefix: String,
    pub latest_shadow_ready_digest_prefix: String,
    pub latest_active_evidence_digest_prefix: String,
    pub latest_drift_status: DriftStatusV1,
    pub freshness_probe_age_ticks: Option<u64>,
    pub freshness_compare_age_ticks: Option<u64>,
    pub freshness_no_impact_age_ticks: Option<u64>,
    pub freshness_drift_status_age_ticks: Option<u64>,
    pub hash_consistent: bool,
    pub probe_missing: bool,
    pub compare_missing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BackendSupportStateV1 {
    Supported,
    Unsupported,
    NotBuilt,
    NotConfigured,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendSupportMatrixV1 {
    pub stub: BackendSupportStateV1,
    pub candle: BackendSupportStateV1,
    pub burn: BackendSupportStateV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendEvidenceSlotReadinessV1 {
    pub probe_ready: bool,
    pub shadow_ready: bool,
    pub active_eligible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendEvidenceSlotDenialsV1 {
    pub probe: Option<EvidenceDenialCodeV1>,
    pub shadow: Option<EvidenceDenialCodeV1>,
    pub active: Option<EvidenceDenialCodeV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendEvidenceSlotEvidenceV1 {
    pub latest_probe_report_digest_prefix: String,
    pub latest_compare_window_digest_prefix: String,
    pub latest_shadow_ready_digest_prefix: String,
    pub latest_active_evidence_digest_prefix: String,
    pub latest_drift_status: DriftStatusV1,
    pub freshness_probe_age_ticks: Option<u64>,
    pub freshness_compare_age_ticks: Option<u64>,
    pub freshness_no_impact_age_ticks: Option<u64>,
    pub freshness_drift_status_age_ticks: Option<u64>,
    pub hash_consistency_ok: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendEvidenceSlotSnapshotV1 {
    pub slot_id: String,
    pub target_hash_prefix: String,
    pub backend_support: BackendSupportMatrixV1,
    pub evidence: BackendEvidenceSlotEvidenceV1,
    pub readiness: BackendEvidenceSlotReadinessV1,
    pub denials: BackendEvidenceSlotDenialsV1,
    pub remediation_codes: Vec<String>,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendEvidenceSnapshotV1 {
    pub schema_version: u16,
    pub supported_slot_set_digest: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub slots: Vec<BackendEvidenceSlotSnapshotV1>,
    pub snapshot_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsConsistencyCheckReportV1 {
    pub schema_version: u16,
    pub status: String,
    pub slot_set_digest: String,
    pub mismatch_categories: Vec<String>,
    pub checked_slots: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EligibilityGeneratedFromV1 {
    pub probe_report_digests: Vec<String>,
    pub shadow_ready_report_digest: String,
    pub active_evidence_report_digest: String,
    pub second_slot_parity_report_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AggregatedEligibilityReportV1 {
    pub schema_version: u16,
    pub overall_status: EligibilityOverallStatusV1,
    pub slots: Vec<UnifiedEligibilityStatusV1>,
    pub report_digest: String,
    pub policy_graph_digest_prefix: String,
    pub generated_from: EligibilityGeneratedFromV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EligibilitySnapshotRecordV1 {
    pub invocation_id: u64,
    pub slots: Vec<EligibilitySnapshotSlotV1>,
    pub report_digest_prefix: String,
    pub policy_graph_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EligibilitySnapshotSlotV1 {
    pub slot_id: String,
    pub probe_ready: bool,
    pub shadow_ready: bool,
    pub active_eligible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DriftStatusV1 {
    Ok,
    Warn,
    Severe,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveEnablementEvidenceV1 {
    pub slot_id: String,
    pub target_hash: String,
    pub latest_probe_report_digest_prefix: String,
    pub latest_probe_status: ProbeReportStatus,
    pub latest_compare_window_digest_prefix: String,
    pub shadow_no_impact_verified: bool,
    pub latest_drift_status: DriftStatusV1,
    pub evidence_window_ticks: u64,
    pub freshness_probe_age_ticks: u64,
    pub freshness_compare_age_ticks: u64,
    pub freshness_no_impact_age_ticks: u64,
    pub freshness_drift_status_age_ticks: u64,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnifiedActiveEvidencePolicyV1 {
    pub freshness_probe_max_age_ticks: u64,
    pub freshness_compare_max_age_ticks: u64,
    pub freshness_no_impact_max_age_ticks: u64,
    pub freshness_drift_status_max_age_ticks: u64,
    pub allow_warn_drift_for_active: bool,
    pub require_matching_target_hash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShadowReadyEvidenceV1 {
    pub slot_id: String,
    pub target_hash: String,
    pub manifest_digest_prefix: String,
    pub latest_probe_report_digest_prefix: String,
    pub latest_probe_status: ProbeReportStatus,
    pub latest_compare_window_digest_prefix: String,
    pub compare_window_present: bool,
    pub no_impact_verified: bool,
    pub latest_drift_status: DriftStatusV1,
    pub shadow_ready: bool,
    pub denial_reason_code: Option<String>,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShadowReadyCheckRecordV1 {
    pub slot_id: String,
    pub target_hash: String,
    pub status: ActiveCheckStatus,
    pub evidence_digest_prefix: Option<String>,
    pub denial_reason_code: Option<String>,
    pub policy_graph_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AggregatedStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AggregatedEvidenceReportV1 {
    pub schema_version: u16,
    pub overall_status: AggregatedStatusV1,
    pub slots: Vec<ShadowReadyEvidenceV1>,
    pub generated_at: u64,
    pub report_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActiveEnablementDeniedCode {
    ActiveDeniedNoProbe,
    ActiveDeniedStaleProbe,
    ActiveDeniedNoCompare,
    ActiveDeniedStaleCompare,
    ActiveDeniedNoNoimpact,
    ActiveDeniedStaleNoimpact,
    ActiveDeniedDriftSevere,
    ActiveDeniedDriftWarn,
    ActiveDeniedHashMismatch,
    ActiveDeniedStrictMode,
    ActiveDeniedBackendNotYetAllowed,
    ActiveNotEnabledForSlotStage,
    ProbeRequired,
    BackendDisabled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnablementDenied {
    pub code: ActiveEnablementDeniedCode,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActiveCheckStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveEnablementCheckRecordV1 {
    pub slot_id: String,
    pub target_hash: String,
    pub status: ActiveCheckStatus,
    pub denial_code: Option<ActiveEnablementDeniedCode>,
    pub evidence_digest_prefix: Option<String>,
    pub freshness_probe_age_ticks: Option<u64>,
    pub freshness_compare_age_ticks: Option<u64>,
    pub freshness_no_impact_age_ticks: Option<u64>,
    pub freshness_drift_status_age_ticks: Option<u64>,
    pub timestamp: u64,
    pub supported_slot_set_scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedRealSlotActiveViewEntryV1 {
    pub slot_id: String,
    pub target_hash: String,
    pub active_eligible: bool,
    pub denial_reason_code: Option<String>,
    pub evidence_digest_prefix: Option<String>,
    pub freshness_probe_age_ticks: Option<u64>,
    pub freshness_compare_age_ticks: Option<u64>,
    pub freshness_no_impact_age_ticks: Option<u64>,
    pub freshness_drift_status_age_ticks: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedRealSlotsActiveViewV1 {
    pub schema_version: u16,
    pub slot_set_scope: String,
    pub slots: Vec<SupportedRealSlotActiveViewEntryV1>,
    pub all_supported_slots_active_eligible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsActiveCheckReport {
    pub slot_id: String,
    pub target_hash: String,
    pub status: ActiveCheckStatus,
    pub evidence: Option<ActiveEnablementEvidenceV1>,
    pub denied: Option<EnablementDenied>,
    pub remediation: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsVerifyReport {
    pub manifest: String,
    pub manifest_present: bool,
    pub digest_match: bool,
    pub promoted_hashes_exist: bool,
    pub files_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RollbackRecommendation {
    pub slot: String,
    pub tick: u64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsRollbackRecommendationReport {
    pub slot: String,
    pub recommendations: Vec<RollbackRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProbeMode {
    Active,
    Hash,
    Stub,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProbeCheckStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeEnvelopeCheck {
    pub code: String,
    pub status: ProbeCheckStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeDigestOutput {
    pub key: String,
    pub digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeScalarOutput {
    pub key: String,
    pub value_q: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeCounterOutput {
    pub key: String,
    pub value: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeOutputs {
    pub digests: Vec<ProbeDigestOutput>,
    pub scalars: Vec<ProbeScalarOutput>,
    pub counters: Vec<ProbeCounterOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProbeReportStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeReportV1 {
    pub schema_version: u16,
    pub slot_id: String,
    pub mode: ProbeMode,
    pub manifest_digest_prefix: String,
    pub model_hash_prefix: Option<String>,
    pub backend_id: String,
    pub contract_version: String,
    pub outputs: ProbeOutputs,
    pub latency_ms: u64,
    pub envelope_checks: Vec<ProbeEnvelopeCheck>,
    pub status: ProbeReportStatus,
}

impl ProbeReportV1 {
    pub const fn pass(&self) -> bool {
        matches!(self.status, ProbeReportStatus::Pass)
    }
}

pub fn models_stage(slot: ModelSlot, src_dir: &Path) -> Result<StageResult, OpsError> {
    if !src_dir.exists() {
        return Err(OpsError::Invalid(format!(
            "stage source missing: {}",
            src_dir.display()
        )));
    }
    let model_path = src_dir.join(MODEL_FILE_NAME);
    if !model_path.exists() {
        return Err(OpsError::Invalid(format!(
            "required {} missing",
            MODEL_FILE_NAME
        )));
    }
    let source_entries = collect_file_entries(src_dir)?;
    if source_entries.is_empty() {
        return Err(OpsError::Invalid("stage source has no files".to_string()));
    }
    if source_entries.len() > MAX_SLOT_FILES {
        return Err(OpsError::Invalid(format!(
            "stage source has too many files: {} > {}",
            source_entries.len(),
            MAX_SLOT_FILES
        )));
    }
    let hash = digest_entries(&source_entries);
    let dst = PathBuf::from("models")
        .join("staging")
        .join(slot.as_str())
        .join(&hash);
    copy_tree(src_dir, &dst)?;
    ensure_manifest_slot(slot)?;
    Ok(StageResult {
        slot: slot.as_str().to_string(),
        hash,
        files: collect_files(&dst)?.len(),
    })
}

pub fn models_promote(
    slot: ModelSlot,
    hash: &str,
    history_keep: Option<usize>,
) -> Result<LifecycleActionReport, OpsError> {
    let staged = PathBuf::from("models")
        .join("staging")
        .join(slot.as_str())
        .join(hash);
    if !staged.exists() {
        return Err(OpsError::Invalid(format!(
            "staged artifact not found for {hash}"
        )));
    }
    let staged_verify = verify_staged_candidate(slot, hash)?;
    if !staged_verify.pass {
        return Err(OpsError::Invalid(format!(
            "staged artifact verification failed: {}",
            staged_verify.reason
        )));
    }

    let promoted = PathBuf::from("models")
        .join("promoted")
        .join(slot.as_str())
        .join(hash);
    copy_tree(&staged, &promoted)?;

    let mut manifest = load_or_init_manifest()?;
    let old_manifest_digest_prefix = Some(manifest.manifest_digest.chars().take(12).collect());
    let slot_key = slot.as_str().to_string();
    let from_hash = manifest
        .slots
        .iter()
        .find(|s| s.slot_id == slot_key)
        .and_then(|s| s.active_hash.clone());
    let files = collect_file_entries(&promoted)?;
    upsert_slot_manifest(
        &mut manifest,
        slot_key.clone(),
        Some(hash.to_string()),
        files,
    );
    persist_manifest_with_history(&mut manifest, history_keep)?;
    persist_action_record("promotion", &slot_key, &from_hash, hash, &manifest)?;
    Ok(LifecycleActionReport {
        action: "promotion".to_string(),
        shadow_report_digest_prefix: None,
        slot: slot_key,
        from_hash,
        to_hash: hash.to_string(),
        old_manifest_digest_prefix,
        new_manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        probe_report_digest_prefix: None,
        readiness_gate_digest_prefix: None,
        timestamp: now_secs(),
    })
}

pub fn models_rollback(
    slot: ModelSlot,
    to_hash: Option<&str>,
    steps: Option<usize>,
    history_keep: Option<usize>,
) -> Result<LifecycleActionReport, OpsError> {
    let target_hash = resolve_rollback_target(slot, to_hash, steps)?;
    let promoted = PathBuf::from("models")
        .join("promoted")
        .join(slot.as_str())
        .join(&target_hash);
    if !promoted.exists() {
        return Err(OpsError::Invalid(
            "rollback hash is not present in promoted".to_string(),
        ));
    }
    let mut manifest = load_or_init_manifest()?;
    let old_manifest_digest_prefix = Some(manifest.manifest_digest.chars().take(12).collect());
    let slot_key = slot.as_str().to_string();
    let from_hash = manifest
        .slots
        .iter()
        .find(|s| s.slot_id == slot_key)
        .and_then(|s| s.active_hash.clone());
    let files = collect_file_entries(&promoted)?;
    upsert_slot_manifest(
        &mut manifest,
        slot_key.clone(),
        Some(target_hash.clone()),
        files,
    );
    persist_manifest_with_history(&mut manifest, history_keep)?;
    persist_rollback_record(&slot_key, &from_hash, &target_hash, &manifest)?;
    Ok(LifecycleActionReport {
        action: "rollback".to_string(),
        shadow_report_digest_prefix: None,
        slot: slot_key,
        from_hash,
        to_hash: target_hash,
        old_manifest_digest_prefix,
        new_manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        probe_report_digest_prefix: None,
        readiness_gate_digest_prefix: None,
        timestamp: now_secs(),
    })
}

pub fn models_list(slot: ModelSlot) -> Result<ModelsListReport, OpsError> {
    let manifest = load_or_init_manifest()?;
    let slot_key = slot.as_str().to_string();
    let active_hash = manifest
        .slots
        .iter()
        .find(|s| s.slot_id == slot_key)
        .and_then(|s| s.active_hash.clone());
    let eligibility = supported_real_slots(None)
        .ok()
        .filter(|slots| slots.contains(&slot))
        .and_then(|_| derive_unified_eligibility_status(slot, Path::new("."), &manifest).ok());

    Ok(ModelsListReport {
        slot: slot_key.clone(),
        active_hash,
        staged_hashes: list_hash_dirs(
            &PathBuf::from("models").join("staging").join(slot.as_str()),
        )?,
        promoted_hashes: list_hash_dirs(
            &PathBuf::from("models").join("promoted").join(slot.as_str()),
        )?,
        current_mode: slot_mode_from_env(slot).to_string(),
        probe_ready: eligibility.as_ref().is_some_and(|e| e.probe_ready),
        shadow_ready: eligibility.as_ref().is_some_and(|e| e.shadow_ready),
        active_eligible: eligibility.as_ref().is_some_and(|e| e.active_eligible),
        denial_reason_probe: eligibility
            .as_ref()
            .and_then(|e| e.denial_reason_probe.clone()),
        denial_reason_shadow: eligibility
            .as_ref()
            .and_then(|e| e.denial_reason_shadow.clone()),
        denial_reason_active: eligibility
            .as_ref()
            .and_then(|e| e.denial_reason_active.clone()),
        last_evidence_digest_prefix: eligibility
            .as_ref()
            .map(|e| e.latest_active_evidence_digest_prefix.clone())
            .filter(|v| v != "missing"),
    })
}

pub fn models_eligibility(
    workdir: &Path,
    requested_slot: Option<ModelSlot>,
    out: &Path,
) -> Result<AggregatedEligibilityReportV1, OpsError> {
    let snapshot = models_evidence_snapshot(workdir, requested_slot, None)?;
    let manifest = load_or_init_manifest()?;
    let mut statuses = snapshot
        .slots
        .iter()
        .map(|slot| unified_eligibility_from_backend_snapshot(slot, &manifest))
        .collect::<Vec<_>>();
    statuses.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    let overall_status = derive_eligibility_overall_status(&statuses);
    let second_slot = crate::detect_second_slot(workdir).ok();
    let second_slot_parity_report_digest = second_slot
        .and_then(|slot| {
            let path = workdir
                .join("out")
                .join(format!("{}_parity_report.json", slot.as_str()));
            fs::read(path)
                .ok()
                .map(|bytes| crate::prefix_hex(&crate::sha256_hex(&bytes), 16))
        })
        .unwrap_or_else(|| "missing".to_string());

    let generated_from = EligibilityGeneratedFromV1 {
        probe_report_digests: statuses
            .iter()
            .map(|s| s.latest_probe_digest_prefix.clone())
            .collect(),
        shadow_ready_report_digest: digest_shadow_generated_from(&statuses),
        active_evidence_report_digest: digest_active_generated_from(&statuses),
        second_slot_parity_report_digest,
    };
    let policy_graph_digest_prefix = read_policy_graph_digest_prefix();

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(ELIGIBILITY_SCHEMA_VERSION.to_string().as_bytes());
    digest_source.extend_from_slice(format!("{:?}", overall_status).as_bytes());
    digest_source.extend_from_slice(policy_graph_digest_prefix.as_bytes());
    digest_source.extend_from_slice(generated_from.shadow_ready_report_digest.as_bytes());
    digest_source.extend_from_slice(generated_from.active_evidence_report_digest.as_bytes());
    digest_source.extend_from_slice(generated_from.second_slot_parity_report_digest.as_bytes());
    for probe_digest in &generated_from.probe_report_digests {
        digest_source.extend_from_slice(probe_digest.as_bytes());
    }
    for slot in &statuses {
        digest_source.extend_from_slice(slot.status_digest.as_bytes());
    }

    let report = AggregatedEligibilityReportV1 {
        schema_version: ELIGIBILITY_SCHEMA_VERSION,
        overall_status,
        slots: statuses,
        report_digest: sha256_hex(&digest_source),
        policy_graph_digest_prefix,
        generated_from,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    append_eligibility_snapshot_record(workdir, &report)?;
    Ok(report)
}

pub fn models_consistency_check(
    workdir: &Path,
    out: &Path,
) -> Result<ModelsConsistencyCheckReportV1, OpsError> {
    let slot_set = supported_real_slot_set_v1()?;
    let manifest = load_or_init_manifest()?;
    let mut mismatch = BTreeSet::new();
    let mut checked_slots = Vec::new();
    for slot_id in &slot_set.slots {
        let slot = parse_slot(slot_id)?;
        checked_slots.push(slot.as_str().to_string());
        let eligibility = derive_unified_eligibility_status(slot, workdir, &manifest)?;
        let strict_out = workdir.join("out").join("strict_check_consistency.json");
        let strict = crate::strict_check(workdir, true, &strict_out).ok();
        if let Some(strict) = strict.as_ref().and_then(|r| r.report.v3.as_ref()) {
            let strict_slot_checks = strict
                .checks
                .iter()
                .filter(|c| c.slot_id.as_deref() == Some(slot.as_str()))
                .collect::<Vec<_>>();
            let strict_probe = strict_slot_checks
                .iter()
                .find(|c| c.check_id == "STRICT_PROBE_READY")
                .map(|c| matches!(c.status, crate::StrictCheckV3Status::Pass));
            if let Some(v) = strict_probe {
                if v != eligibility.probe_ready {
                    mismatch.insert("READINESS_BOOL_MISMATCH".to_string());
                }
            }
        }
        if eligibility.target_hash_prefix == "missing" {
            mismatch.insert("TARGET_HASH_MISMATCH".to_string());
        }
        let mut reasons = [
            eligibility.denial_reason_probe.as_deref(),
            eligibility.denial_reason_shadow.as_deref(),
            eligibility.denial_reason_active.as_deref(),
        ]
        .into_iter()
        .flatten()
        .filter_map(|r| map_denial_reason_to_code(Some(r)).map(|c| format!("{:?}", c)))
        .collect::<Vec<_>>();
        reasons.sort();
        reasons.dedup();
        if reasons.len() > 3 {
            mismatch.insert("DENIAL_REASON_MISMATCH".to_string());
        }
    }
    checked_slots.sort();
    checked_slots.dedup();
    let report = ModelsConsistencyCheckReportV1 {
        schema_version: 1,
        status: if mismatch.is_empty() { "PASS" } else { "FAIL" }.to_string(),
        slot_set_digest: prefix_hex(&slot_set.set_digest, 16),
        mismatch_categories: mismatch.into_iter().collect(),
        checked_slots,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn models_active_check(
    slot: ModelSlot,
    workdir: &Path,
    out: &Path,
) -> Result<ModelsActiveCheckReport, OpsError> {
    let manifest = load_or_init_manifest()?;
    let slot_id = slot.as_str().to_string();
    let target_hash = manifest
        .slots
        .iter()
        .find(|s| s.slot_id == slot_id)
        .and_then(|s| s.active_hash.clone())
        .ok_or_else(|| {
            OpsError::Invalid("no active_hash for slot in lifecycle manifest".to_string())
        })?;
    let strict_mode = std::env::var("UCF_STRICT_MODE").ok().as_deref() == Some("1");
    let bypass = std::env::var("UCF_DEV_ACTIVE_BYPASS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    let decision = can_enable_active(slot, &target_hash, workdir, strict_mode, bypass);
    let (status, evidence, denied) = match decision {
        Ok(evidence) => (ActiveCheckStatus::Pass, Some(evidence), None),
        Err(denied) => (ActiveCheckStatus::Fail, None, Some(denied)),
    };
    let report = ModelsActiveCheckReport {
        slot_id: slot_id.clone(),
        target_hash: target_hash.clone(),
        status: status.clone(),
        evidence,
        denied,
        remediation: vec![
            format!(
                "cargo run -p ucf-ops -- models probe --slot {} --out ./out/probe_{}.json",
                slot.as_str(),
                slot.as_str()
            ),
            "cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json"
                .to_string(),
            "cargo run -p ucf-ops -- drift report --run <run_id> --windows 20 --out ./out/drift_report.json".to_string(),
            "rollback or keep slot in shadow mode until active-check PASS".to_string(),
        ],
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    append_active_check_record(workdir, &slot_id, &target_hash, &report)?;
    Ok(report)
}

pub fn models_active_evidence(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedRealSlotsActiveViewV1, OpsError> {
    let slots = supported_real_slots(None)?;
    let manifest = load_or_init_manifest()?;
    let policy = unified_active_evidence_policy();
    let mut entries = Vec::new();

    for slot in slots {
        let slot_id = slot.as_str().to_string();
        let target_hash = manifest
            .slots
            .iter()
            .find(|s| s.slot_id == slot_id)
            .and_then(|s| s.active_hash.clone())
            .unwrap_or_else(|| "missing".to_string());
        let eval = if target_hash == "missing" {
            Err(EnablementDenied {
                code: ActiveEnablementDeniedCode::ActiveDeniedHashMismatch,
                detail: "ACTIVE_DENIED_HASH_MISMATCH: slot has no active hash".to_string(),
            })
        } else {
            evaluate_active_evidence(slot, &target_hash, workdir, &policy)
        };
        match eval {
            Ok(ev) => entries.push(SupportedRealSlotActiveViewEntryV1 {
                slot_id,
                target_hash,
                active_eligible: true,
                denial_reason_code: None,
                evidence_digest_prefix: Some(prefix_hex(&ev.evidence_digest, 16)),
                freshness_probe_age_ticks: Some(ev.freshness_probe_age_ticks),
                freshness_compare_age_ticks: Some(ev.freshness_compare_age_ticks),
                freshness_no_impact_age_ticks: Some(ev.freshness_no_impact_age_ticks),
                freshness_drift_status_age_ticks: Some(ev.freshness_drift_status_age_ticks),
            }),
            Err(denied) => entries.push(SupportedRealSlotActiveViewEntryV1 {
                slot_id,
                target_hash,
                active_eligible: false,
                denial_reason_code: Some(format!("{:?}", denied.code)),
                evidence_digest_prefix: None,
                freshness_probe_age_ticks: None,
                freshness_compare_age_ticks: None,
                freshness_no_impact_age_ticks: None,
                freshness_drift_status_age_ticks: None,
            }),
        }
    }
    entries.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    let report = SupportedRealSlotsActiveViewV1 {
        schema_version: 1,
        slot_set_scope: SUPPORTED_REAL_SLOT_SET_VERSION.to_string(),
        all_supported_slots_active_eligible: entries.iter().all(|e| e.active_eligible),
        slots: entries,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn models_shadow_ready(
    workdir: &Path,
    requested_slot: Option<ModelSlot>,
    out: &Path,
) -> Result<AggregatedEvidenceReportV1, OpsError> {
    let slots = supported_real_slots(requested_slot)?;
    let mut slot_reports = Vec::new();
    for slot in slots {
        slot_reports.push(build_shadow_ready_evidence(slot, workdir)?);
    }
    slot_reports.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    let overall_status = if slot_reports.iter().all(|s| s.shadow_ready) {
        AggregatedStatusV1::Pass
    } else {
        AggregatedStatusV1::Fail
    };
    let generated_at = now_secs();
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(SHADOW_READY_SCHEMA_VERSION.to_string().as_bytes());
    digest_source.extend_from_slice(format!("{:?}", overall_status).as_bytes());
    for slot in &slot_reports {
        digest_source.extend_from_slice(slot.evidence_digest.as_bytes());
    }
    let report_digest = sha256_hex(&digest_source);
    let report = AggregatedEvidenceReportV1 {
        schema_version: SHADOW_READY_SCHEMA_VERSION,
        overall_status,
        slots: slot_reports,
        generated_at,
        report_digest,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    append_shadow_ready_records(workdir, &report)?;
    Ok(report)
}

pub fn can_enable_active(
    slot: ModelSlot,
    target_hash: &str,
    workdir: &Path,
    strict_mode: bool,
    dev_bypass: bool,
) -> Result<ActiveEnablementEvidenceV1, EnablementDenied> {
    if strict_mode && dev_bypass {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedStrictMode,
            detail: "dev bypass is forbidden in strict mode".to_string(),
        });
    }
    if cfg!(feature = "backend-burn") && slot == ModelSlot::WorldJepa {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedBackendNotYetAllowed,
            detail: "ACTIVE_DENIED_BACKEND_NOT_YET_ALLOWED: burn world active mode is not supported in v0".to_string(),
        });
    }
    if slot == ModelSlot::Sae {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveNotEnabledForSlotStage,
            detail:
                "ACTIVE_NOT_ENABLED_FOR_SLOT_STAGE: sae remains shadow-only in tiny real fixture v2"
                    .to_string(),
        });
    }
    let manifest = load_or_init_manifest().map_err(|e| EnablementDenied {
        code: ActiveEnablementDeniedCode::ActiveDeniedHashMismatch,
        detail: e.to_string(),
    })?;
    let slot_id = slot.as_str().to_string();
    let manifest_hash = manifest
        .slots
        .iter()
        .find(|s| s.slot_id == slot_id)
        .and_then(|s| s.active_hash.clone())
        .ok_or_else(|| EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedHashMismatch,
            detail: "slot has no active hash in lifecycle manifest".to_string(),
        })?;
    let promoted_path = PathBuf::from("models")
        .join("promoted")
        .join(slot.as_str())
        .join(&manifest_hash);
    if target_hash != manifest_hash || !promoted_path.exists() {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedHashMismatch,
            detail: "target hash not aligned with promoted active hash".to_string(),
        });
    }

    let policy = unified_active_evidence_policy();
    evaluate_active_evidence(slot, target_hash, workdir, &policy)
}

fn unified_active_evidence_policy() -> UnifiedActiveEvidencePolicyV1 {
    let cfg = crate::load_or_init_config(Path::new(".")).unwrap_or_default();
    UnifiedActiveEvidencePolicyV1 {
        freshness_probe_max_age_ticks: cfg.active_evidence_probe_max_age_ticks.max(1),
        freshness_compare_max_age_ticks: cfg.active_evidence_compare_max_age_ticks.max(1),
        freshness_no_impact_max_age_ticks: cfg.active_evidence_no_impact_max_age_ticks.max(1),
        freshness_drift_status_max_age_ticks: cfg.active_evidence_drift_status_max_age_ticks.max(1),
        allow_warn_drift_for_active: cfg.active_evidence_allow_warn_drift_for_active,
        require_matching_target_hash: cfg.active_evidence_require_matching_target_hash,
    }
}

fn evaluate_active_evidence(
    slot: ModelSlot,
    target_hash: &str,
    workdir: &Path,
    policy: &UnifiedActiveEvidencePolicyV1,
) -> Result<ActiveEnablementEvidenceV1, EnablementDenied> {
    let probe_path = PathBuf::from("out").join(format!("probe_{}.json", slot.as_str()));
    let probe: ProbeReportV1 = fs::read_to_string(&probe_path)
        .ok()
        .and_then(|body| serde_json::from_str(&body).ok())
        .ok_or_else(|| EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedNoProbe,
            detail: "ACTIVE_DENIED_NO_PROBE: missing probe report".to_string(),
        })?;
    let expected_hash_prefix = target_hash
        .chars()
        .take(PROBE_DIGEST_PREFIX_LEN)
        .collect::<String>();
    if !probe.pass() || probe.slot_id != slot.as_str() {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedNoProbe,
            detail: "ACTIVE_DENIED_NO_PROBE: latest probe missing PASS for slot".to_string(),
        });
    }
    if policy.require_matching_target_hash
        && probe.model_hash_prefix.as_deref() != Some(expected_hash_prefix.as_str())
    {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedHashMismatch,
            detail: "ACTIVE_DENIED_HASH_MISMATCH: probe hash prefix does not match target hash"
                .to_string(),
        });
    }
    let latest_probe_digest_prefix = prefix_hex(
        &sha256_hex(&serde_json::to_vec(&probe).unwrap_or_default()),
        16,
    );

    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let records = load_fixture_records(&fixture_path).unwrap_or_default();
    let mut latest_compare: Option<ucf_ess::v1::SlotCompareWindowRecordV1> = None;
    let mut max_tick = 0_u64;
    let mut drift_status = DriftStatusV1::Unknown;
    let mut latest_drift_tick = 0_u64;
    for record in records {
        let tick = record.time.tick.get();
        max_tick = max_tick.max(tick);
        if let ExperiencePayload::Audit(payload) = record.payload {
            match payload {
                AuditPayload::SlotCompareWindow(window) if window.slot_id == slot.as_str() => {
                    latest_compare = Some(window);
                }
                AuditPayload::DriftAlarm(alarm) if alarm.slot_id == slot.as_str() => {
                    latest_drift_tick = tick;
                    drift_status = if alarm.severity.eq_ignore_ascii_case("severe") {
                        DriftStatusV1::Severe
                    } else {
                        DriftStatusV1::Warn
                    };
                }
                _ => {}
            }
        }
    }

    let Some(compare) = latest_compare else {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedNoCompare,
            detail: "ACTIVE_DENIED_NO_COMPARE: missing compare/parity window".to_string(),
        });
    };
    let freshness_compare_age_ticks = max_tick.saturating_sub(compare.t1);
    if matches!(
        crate::compare_freshness(
            Some(compare.t1),
            max_tick,
            policy.freshness_compare_max_age_ticks,
        ),
        crate::CompareWindowFreshnessV1::StaleCompare
    ) {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedStaleCompare,
            detail: "ACTIVE_DENIED_STALE_COMPARE: compare/parity window evidence is stale"
                .to_string(),
        });
    }

    let gate_path = PathBuf::from("out").join("gate_report.json");
    let gate_json = fs::read_to_string(&gate_path)
        .ok()
        .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok());
    let shadow_no_impact_verified = gate_json
        .as_ref()
        .and_then(|v| v.get("checks").and_then(|c| c.as_array()))
        .map(|checks| {
            checks.iter().any(|c| {
                c.get("name").and_then(|v| v.as_str()) == Some("shadow_no_decision_impact")
                    && c.get("status").and_then(|v| v.as_str()) == Some("PASS")
            })
        })
        .unwrap_or(false);
    if !shadow_no_impact_verified {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedNoNoimpact,
            detail: "ACTIVE_DENIED_NO_NOIMPACT: no-decision-impact proof missing or failed"
                .to_string(),
        });
    }
    let gate_tick = gate_json
        .as_ref()
        .and_then(|v| v.get("run_id").and_then(|r| r.as_str()))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(max_tick);
    let freshness_no_impact_age_ticks = max_tick.saturating_sub(gate_tick);
    if freshness_no_impact_age_ticks > policy.freshness_no_impact_max_age_ticks {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedStaleNoimpact,
            detail: "ACTIVE_DENIED_STALE_NOIMPACT: no-decision-impact evidence is stale"
                .to_string(),
        });
    }

    let freshness_probe_age_ticks = max_tick.saturating_sub(compare.t1);
    if freshness_probe_age_ticks > policy.freshness_probe_max_age_ticks {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedStaleProbe,
            detail: "ACTIVE_DENIED_STALE_PROBE: probe evidence is stale".to_string(),
        });
    }

    if matches!(drift_status, DriftStatusV1::Unknown) {
        drift_status = DriftStatusV1::Ok;
        latest_drift_tick = max_tick;
    }
    let freshness_drift_status_age_ticks = max_tick.saturating_sub(latest_drift_tick);
    if freshness_drift_status_age_ticks > policy.freshness_drift_status_max_age_ticks {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedStaleCompare,
            detail: "ACTIVE_DENIED_STALE_COMPARE: drift status evidence is stale".to_string(),
        });
    }
    if matches!(drift_status, DriftStatusV1::Severe) {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedDriftSevere,
            detail: "ACTIVE_DENIED_DRIFT_SEVERE: drift status is severe".to_string(),
        });
    }
    if matches!(drift_status, DriftStatusV1::Warn) && !policy.allow_warn_drift_for_active {
        return Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedDriftWarn,
            detail: "ACTIVE_DENIED_DRIFT_WARN: warn drift not allowed by policy".to_string(),
        });
    }

    let compare_digest = sha256_hex(
        format!(
            "{}:{}:{}:{}:{}:{}",
            compare.slot_id,
            compare.t0,
            compare.t1,
            compare.sample_count,
            compare.mean_delta_q,
            compare.p95_delta_q
        )
        .as_bytes(),
    );
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(slot.as_str().as_bytes());
    digest_source.extend_from_slice(target_hash.as_bytes());
    digest_source.extend_from_slice(latest_probe_digest_prefix.as_bytes());
    digest_source.extend_from_slice(format!("{:?}", probe.status).as_bytes());
    digest_source.extend_from_slice(prefix_hex(&compare_digest, 16).as_bytes());
    digest_source.extend_from_slice(if shadow_no_impact_verified {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(format!("{:?}", drift_status).as_bytes());
    digest_source.extend_from_slice(
        policy
            .freshness_compare_max_age_ticks
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(freshness_probe_age_ticks.to_string().as_bytes());
    digest_source.extend_from_slice(freshness_compare_age_ticks.to_string().as_bytes());
    digest_source.extend_from_slice(freshness_no_impact_age_ticks.to_string().as_bytes());
    digest_source.extend_from_slice(freshness_drift_status_age_ticks.to_string().as_bytes());
    let evidence_digest = sha256_hex(&digest_source);

    Ok(ActiveEnablementEvidenceV1 {
        slot_id: slot.as_str().to_string(),
        target_hash: target_hash.to_string(),
        latest_probe_report_digest_prefix: latest_probe_digest_prefix,
        latest_probe_status: probe.status,
        latest_compare_window_digest_prefix: prefix_hex(&compare_digest, 16),
        shadow_no_impact_verified,
        latest_drift_status: drift_status,
        evidence_window_ticks: policy.freshness_compare_max_age_ticks,
        freshness_probe_age_ticks,
        freshness_compare_age_ticks,
        freshness_no_impact_age_ticks,
        freshness_drift_status_age_ticks,
        evidence_digest,
    })
}

fn append_active_check_record(
    workdir: &Path,
    slot_id: &str,
    target_hash: &str,
    report: &ModelsActiveCheckReport,
) -> Result<(), OpsError> {
    let path = workdir.join("out").join("active_enablement_checks.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut all: Vec<ActiveEnablementCheckRecordV1> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };
    all.push(ActiveEnablementCheckRecordV1 {
        slot_id: slot_id.to_string(),
        target_hash: target_hash.to_string(),
        status: report.status.clone(),
        denial_code: report.denied.as_ref().map(|d| d.code.clone()),
        evidence_digest_prefix: report
            .evidence
            .as_ref()
            .map(|e| prefix_hex(&e.evidence_digest, 16)),
        freshness_probe_age_ticks: report
            .evidence
            .as_ref()
            .map(|e| e.freshness_probe_age_ticks),
        freshness_compare_age_ticks: report
            .evidence
            .as_ref()
            .map(|e| e.freshness_compare_age_ticks),
        freshness_no_impact_age_ticks: report
            .evidence
            .as_ref()
            .map(|e| e.freshness_no_impact_age_ticks),
        freshness_drift_status_age_ticks: report
            .evidence
            .as_ref()
            .map(|e| e.freshness_drift_status_age_ticks),
        timestamp: now_secs(),
        supported_slot_set_scope: Some(SUPPORTED_REAL_SLOT_SET_VERSION.to_string()),
    });
    fs::write(path, serde_json::to_vec_pretty(&all)?)?;
    Ok(())
}

fn append_shadow_ready_records(
    workdir: &Path,
    report: &AggregatedEvidenceReportV1,
) -> Result<(), OpsError> {
    let path = workdir.join("out").join("shadow_ready_checks.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut all: Vec<ShadowReadyCheckRecordV1> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };
    let policy_graph_digest_prefix = read_policy_graph_digest_prefix();
    for slot in &report.slots {
        all.push(ShadowReadyCheckRecordV1 {
            slot_id: slot.slot_id.clone(),
            target_hash: slot.target_hash.clone(),
            status: if slot.shadow_ready {
                ActiveCheckStatus::Pass
            } else {
                ActiveCheckStatus::Fail
            },
            evidence_digest_prefix: Some(prefix_hex(&slot.evidence_digest, 16)),
            denial_reason_code: slot.denial_reason_code.clone(),
            policy_graph_digest_prefix: policy_graph_digest_prefix.clone(),
        });
    }
    fs::write(path, serde_json::to_vec_pretty(&all)?)?;
    Ok(())
}

fn evidence_freshness_policy_v1() -> EvidenceFreshnessPolicyV1 {
    let cfg = crate::load_or_init_config(Path::new(".")).unwrap_or_default();
    EvidenceFreshnessPolicyV1 {
        probe_max_age_ticks: cfg.active_evidence_probe_max_age_ticks.max(1),
        compare_max_age_ticks: cfg.active_evidence_compare_max_age_ticks.max(1),
        no_impact_max_age_ticks: cfg.active_evidence_no_impact_max_age_ticks.max(1),
        drift_max_age_ticks: cfg.active_evidence_drift_status_max_age_ticks.max(1),
    }
}

pub fn map_denial_reason_to_code(reason: Option<&str>) -> Option<EvidenceDenialCodeV1> {
    let reason = reason?;
    if reason.contains("NO_PROBE")
        || reason.contains("PROBE_REQUIRED")
        || reason.contains("PROBE_REPORT_MISSING")
    {
        return Some(EvidenceDenialCodeV1::NoProbe);
    }
    if reason.contains("STALE_PROBE") {
        return Some(EvidenceDenialCodeV1::StaleProbe);
    }
    if reason.contains("NO_COMPARE") || reason.contains("COMPARE_WINDOW_MISSING") {
        return Some(EvidenceDenialCodeV1::NoCompare);
    }
    if reason.contains("STALE_COMPARE") {
        return Some(EvidenceDenialCodeV1::StaleCompare);
    }
    if reason.contains("HASH_MISMATCH") || reason.contains("HashMismatch") {
        return Some(EvidenceDenialCodeV1::HashMismatch);
    }
    if reason.contains("DRIFT_SEVERE") {
        return Some(EvidenceDenialCodeV1::DriftSevere);
    }
    if reason.contains("DRIFT_WARN") {
        return Some(EvidenceDenialCodeV1::DriftWarn);
    }
    if reason.contains("ACTIVE_NOT_ENABLED") {
        return Some(EvidenceDenialCodeV1::ActiveNotEnabled);
    }
    if reason.contains("UNSUPPORTED_SLOT_SET") {
        return Some(EvidenceDenialCodeV1::UnsupportedSlotSet);
    }
    None
}

pub fn resolve_slot_evidence(
    slot: ModelSlot,
    target_hash: &str,
    workdir: &Path,
    manifest: &LifecycleManifest,
) -> Result<SlotEvidenceSnapshotV1, OpsError> {
    let slot_id = slot.as_str().to_string();
    let policy = evidence_freshness_policy_v1();
    let target_hash_prefix = prefix_hex(target_hash, 16);
    let manifest_digest_prefix = prefix_hex(&manifest.manifest_digest, 16);

    let probe_path = PathBuf::from("out").join(format!("probe_{}.json", slot.as_str()));
    let probe = fs::read_to_string(&probe_path)
        .ok()
        .and_then(|body| serde_json::from_str::<ProbeReportV1>(&body).ok());
    let latest_probe_report_digest_prefix = probe
        .as_ref()
        .map(|p| prefix_hex(&sha256_hex(&serde_json::to_vec(p).unwrap_or_default()), 16))
        .unwrap_or_else(|| "missing".to_string());

    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let records = load_fixture_records(&fixture_path).unwrap_or_default();
    let mut latest_compare: Option<ucf_ess::v1::SlotCompareWindowRecordV1> = None;
    let mut max_tick = 0_u64;
    let mut drift_status = DriftStatusV1::Unknown;
    let mut latest_drift_tick = 0_u64;
    for record in records {
        let tick = record.time.tick.get();
        max_tick = max_tick.max(tick);
        if let ExperiencePayload::Audit(payload) = record.payload {
            match payload {
                AuditPayload::SlotCompareWindow(window) if window.slot_id == slot.as_str() => {
                    latest_compare = Some(window);
                }
                AuditPayload::DriftAlarm(alarm) if alarm.slot_id == slot.as_str() => {
                    latest_drift_tick = tick;
                    drift_status = if alarm.severity.eq_ignore_ascii_case("severe") {
                        DriftStatusV1::Severe
                    } else {
                        DriftStatusV1::Warn
                    };
                }
                _ => {}
            }
        }
    }
    if matches!(drift_status, DriftStatusV1::Unknown) {
        drift_status = DriftStatusV1::Ok;
        latest_drift_tick = max_tick;
    }

    let latest_compare_window_digest_prefix = latest_compare
        .as_ref()
        .map(compare_digest_prefix)
        .unwrap_or_else(|| "missing".to_string());
    let freshness_compare_age_ticks = latest_compare
        .as_ref()
        .map(|w| max_tick.saturating_sub(w.t1));
    let freshness_probe_age_ticks = freshness_compare_age_ticks;
    let freshness_drift_status_age_ticks = Some(max_tick.saturating_sub(latest_drift_tick));

    let shadow = build_shadow_ready_evidence(slot, workdir)?;
    let latest_shadow_ready_digest_prefix = prefix_hex(&shadow.evidence_digest, 16);

    let active_eval = if target_hash == "missing" {
        Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedHashMismatch,
            detail: "ACTIVE_DENIED_HASH_MISMATCH: slot has no active hash".to_string(),
        })
    } else {
        evaluate_active_evidence(
            slot,
            target_hash,
            workdir,
            &UnifiedActiveEvidencePolicyV1 {
                freshness_probe_max_age_ticks: policy.probe_max_age_ticks,
                freshness_compare_max_age_ticks: policy.compare_max_age_ticks,
                freshness_no_impact_max_age_ticks: policy.no_impact_max_age_ticks,
                freshness_drift_status_max_age_ticks: policy.drift_max_age_ticks,
                allow_warn_drift_for_active: crate::load_or_init_config(Path::new("."))
                    .unwrap_or_default()
                    .active_evidence_allow_warn_drift_for_active,
                require_matching_target_hash: crate::load_or_init_config(Path::new("."))
                    .unwrap_or_default()
                    .active_evidence_require_matching_target_hash,
            },
        )
    };
    let (latest_active_evidence_digest_prefix, freshness_no_impact_age_ticks) = match active_eval {
        Ok(ev) => (
            prefix_hex(&ev.evidence_digest, 16),
            Some(ev.freshness_no_impact_age_ticks),
        ),
        Err(_) => ("missing".to_string(), None),
    };

    let expected_hash_prefix = target_hash
        .chars()
        .take(PROBE_DIGEST_PREFIX_LEN)
        .collect::<String>();
    let hash_consistent = probe
        .as_ref()
        .and_then(|p| p.model_hash_prefix.as_ref())
        .is_some_and(|h| h == &expected_hash_prefix);

    Ok(SlotEvidenceSnapshotV1 {
        slot_id,
        manifest_digest_prefix,
        target_hash_prefix,
        latest_probe_report_digest_prefix,
        latest_compare_window_digest_prefix,
        latest_shadow_ready_digest_prefix,
        latest_active_evidence_digest_prefix,
        latest_drift_status: drift_status,
        freshness_probe_age_ticks,
        freshness_compare_age_ticks,
        freshness_no_impact_age_ticks,
        freshness_drift_status_age_ticks,
        hash_consistent,
        probe_missing: probe.is_none(),
        compare_missing: latest_compare.is_none(),
    })
}

fn read_policy_graph_digest_prefix() -> String {
    fs::read_to_string(PathBuf::from("out").join("gate_report.json"))
        .ok()
        .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
        .and_then(|v| {
            v.get("policy_graph_digest")
                .and_then(|x| x.as_str())
                .map(|x| x.to_string())
        })
        .map(|value| prefix_hex(&value, 16))
        .unwrap_or_else(|| "unknown".to_string())
}

fn derive_unified_eligibility_status(
    slot: ModelSlot,
    workdir: &Path,
    manifest: &LifecycleManifest,
) -> Result<UnifiedEligibilityStatusV1, OpsError> {
    let slot_id = slot.as_str().to_string();
    let target_hash = manifest
        .slots
        .iter()
        .find(|s| s.slot_id == slot_id)
        .and_then(|s| s.active_hash.clone())
        .unwrap_or_else(|| "missing".to_string());

    let snapshot = resolve_slot_evidence(slot, &target_hash, workdir, manifest)?;
    let probe_path = PathBuf::from("out").join(format!("probe_{}.json", slot.as_str()));
    let probe = fs::read_to_string(&probe_path)
        .ok()
        .and_then(|body| serde_json::from_str::<ProbeReportV1>(&body).ok());
    let expected_hash_prefix = target_hash
        .chars()
        .take(PROBE_DIGEST_PREFIX_LEN)
        .collect::<String>();
    let probe_hash_match = probe
        .as_ref()
        .and_then(|p| p.model_hash_prefix.as_deref())
        .map(|v| v == expected_hash_prefix)
        .unwrap_or(false);

    let probe_ready = probe
        .as_ref()
        .map(|p| p.pass() && p.slot_id == slot.as_str() && probe_hash_match)
        .unwrap_or(false);
    let denial_reason_probe = if probe_ready {
        None
    } else if probe.is_none() {
        Some("NO_PROBE".to_string())
    } else if !probe_hash_match {
        Some("HASH_MISMATCH".to_string())
    } else {
        Some("NO_PROBE".to_string())
    };

    let shadow = build_shadow_ready_evidence(slot, workdir)?;
    let second_slot = crate::detect_second_slot(workdir).ok();
    let parity_path = workdir
        .join("out")
        .join(format!("{}_parity_report.json", slot.as_str()));
    let parity_report = fs::read_to_string(&parity_path)
        .ok()
        .and_then(|body| serde_json::from_str::<SecondSlotParityReportV1>(&body).ok());
    let (burn_support_state, burn_parity_present) = if second_slot == Some(slot) {
        parity_report
            .map(|p| (p.burn_support_state, p.burn_parity_present))
            .unwrap_or((OptionalBackendSupportStateV1::NotConfigured, false))
    } else {
        (OptionalBackendSupportStateV1::Unsupported, false)
    };
    let active_eval = if target_hash == "missing" {
        Err(EnablementDenied {
            code: ActiveEnablementDeniedCode::ActiveDeniedHashMismatch,
            detail: "ACTIVE_DENIED_HASH_MISMATCH: slot has no active hash".to_string(),
        })
    } else {
        evaluate_active_evidence(
            slot,
            &target_hash,
            workdir,
            &unified_active_evidence_policy(),
        )
    };
    let (active_eligible, denial_reason_active, drift_status) = match active_eval {
        Ok(ev) => (true, None, ev.latest_drift_status),
        Err(denied) => (
            false,
            map_denial_reason_to_code(Some(&format!("{:?}", denied.code)))
                .map(|c| format!("{:?}", c))
                .or(Some(format!("{:?}", denied.code))),
            shadow.latest_drift_status.clone(),
        ),
    };

    let denial_reason_shadow = map_denial_reason_to_code(shadow.denial_reason_code.as_deref())
        .map(|c| format!("{:?}", c))
        .or(shadow.denial_reason_code.clone());

    let mut remediation_codes = [
        denial_reason_probe.clone(),
        denial_reason_shadow.clone(),
        denial_reason_active.clone(),
    ]
    .into_iter()
    .flatten()
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect::<Vec<_>>();
    remediation_codes.truncate(4);

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(snapshot.slot_id.as_bytes());
    digest_source.extend_from_slice(snapshot.target_hash_prefix.as_bytes());
    digest_source.extend_from_slice(snapshot.manifest_digest_prefix.as_bytes());
    digest_source.extend_from_slice(if probe_ready { b"1" } else { b"0" });
    digest_source.extend_from_slice(if shadow.shadow_ready { b"1" } else { b"0" });
    digest_source.extend_from_slice(if active_eligible { b"1" } else { b"0" });
    digest_source.extend_from_slice(snapshot.latest_probe_report_digest_prefix.as_bytes());
    digest_source.extend_from_slice(snapshot.latest_shadow_ready_digest_prefix.as_bytes());
    digest_source.extend_from_slice(snapshot.latest_active_evidence_digest_prefix.as_bytes());
    digest_source.extend_from_slice(format!("{:?}", drift_status).as_bytes());
    digest_source.extend_from_slice(format!("{:?}", burn_support_state).as_bytes());
    digest_source.extend_from_slice(if burn_parity_present { b"1" } else { b"0" });
    for code in &remediation_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(UnifiedEligibilityStatusV1 {
        slot_id: snapshot.slot_id,
        target_hash_prefix: snapshot.target_hash_prefix,
        manifest_digest_prefix: snapshot.manifest_digest_prefix,
        probe_ready,
        shadow_ready: shadow.shadow_ready,
        active_eligible,
        latest_probe_digest_prefix: snapshot.latest_probe_report_digest_prefix,
        latest_shadow_evidence_digest_prefix: snapshot.latest_shadow_ready_digest_prefix,
        latest_active_evidence_digest_prefix: snapshot.latest_active_evidence_digest_prefix,
        latest_drift_status: drift_status,
        burn_support_state,
        burn_parity_present,
        denial_reason_probe,
        denial_reason_shadow,
        denial_reason_active,
        canonical_remediation_codes: merge_canonical_remediations(remediation_codes.iter(), 4),
        remediation_codes,
        status_digest: sha256_hex(&digest_source),
    })
}

fn unified_eligibility_from_backend_snapshot(
    slot: &BackendEvidenceSlotSnapshotV1,
    manifest: &LifecycleManifest,
) -> UnifiedEligibilityStatusV1 {
    let manifest_digest_prefix = prefix_hex(&manifest.manifest_digest, 16);
    let burn_support_state = match slot.backend_support.burn {
        BackendSupportStateV1::Supported => OptionalBackendSupportStateV1::Supported,
        BackendSupportStateV1::Unsupported => OptionalBackendSupportStateV1::Unsupported,
        BackendSupportStateV1::NotBuilt => OptionalBackendSupportStateV1::NotBuilt,
        BackendSupportStateV1::NotConfigured => OptionalBackendSupportStateV1::NotConfigured,
    };
    let denial_reason_probe = slot.denials.probe.as_ref().map(|d| format!("{:?}", d));
    let denial_reason_shadow = slot.denials.shadow.as_ref().map(|d| format!("{:?}", d));
    let denial_reason_active = slot.denials.active.as_ref().map(|d| format!("{:?}", d));

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(slot.slot_id.as_bytes());
    digest_source.extend_from_slice(slot.target_hash_prefix.as_bytes());
    digest_source.extend_from_slice(manifest_digest_prefix.as_bytes());
    digest_source.extend_from_slice(if slot.readiness.probe_ready {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(if slot.readiness.shadow_ready {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(if slot.readiness.active_eligible {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(slot.evidence.latest_probe_report_digest_prefix.as_bytes());
    digest_source.extend_from_slice(slot.evidence.latest_shadow_ready_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        slot.evidence
            .latest_active_evidence_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(format!("{:?}", slot.evidence.latest_drift_status).as_bytes());
    digest_source.extend_from_slice(format!("{:?}", burn_support_state).as_bytes());
    digest_source.extend_from_slice(
        if slot.backend_support.burn == BackendSupportStateV1::Supported {
            b"1"
        } else {
            b"0"
        },
    );
    for code in &slot.remediation_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    UnifiedEligibilityStatusV1 {
        slot_id: slot.slot_id.clone(),
        target_hash_prefix: slot.target_hash_prefix.clone(),
        manifest_digest_prefix,
        probe_ready: slot.readiness.probe_ready,
        shadow_ready: slot.readiness.shadow_ready,
        active_eligible: slot.readiness.active_eligible,
        latest_probe_digest_prefix: slot.evidence.latest_probe_report_digest_prefix.clone(),
        latest_shadow_evidence_digest_prefix: slot
            .evidence
            .latest_shadow_ready_digest_prefix
            .clone(),
        latest_active_evidence_digest_prefix: slot
            .evidence
            .latest_active_evidence_digest_prefix
            .clone(),
        latest_drift_status: slot.evidence.latest_drift_status.clone(),
        burn_support_state,
        burn_parity_present: slot.backend_support.burn == BackendSupportStateV1::Supported,
        denial_reason_probe,
        denial_reason_shadow,
        denial_reason_active,
        remediation_codes: slot.remediation_codes.clone(),
        canonical_remediation_codes: merge_canonical_remediations(slot.remediation_codes.iter(), 4),
        status_digest: sha256_hex(&digest_source),
    }
}

pub fn models_evidence_snapshot(
    workdir: &Path,
    requested_slot: Option<ModelSlot>,
    run_id: Option<&str>,
) -> Result<BackendEvidenceSnapshotV1, OpsError> {
    let slot_set = supported_real_slot_set_v1()?;
    let manifest = load_or_init_manifest()?;
    let slots = supported_real_slots(requested_slot)?;
    let second_slot = crate::detect_second_slot(workdir).ok();
    let mut snapshots = Vec::new();

    for slot in slots {
        let slot_id = slot.as_str().to_string();
        let target_hash = manifest
            .slots
            .iter()
            .find(|s| s.slot_id == slot_id)
            .and_then(|s| s.active_hash.clone())
            .unwrap_or_else(|| "missing".to_string());
        let slot_evidence = resolve_slot_evidence(slot, &target_hash, workdir, &manifest)?;
        let eligibility = derive_unified_eligibility_status(slot, workdir, &manifest)?;
        let parity_report = read_second_slot_parity_report(workdir, run_id, slot.as_str());
        let backend_support = BackendSupportMatrixV1 {
            stub: BackendSupportStateV1::Supported,
            candle: resolve_candle_support_state(slot, second_slot, parity_report.as_ref()),
            burn: resolve_burn_support_state(slot, second_slot, parity_report.as_ref()),
        };
        let mut remediation_codes = eligibility.remediation_codes;
        remediation_codes.sort();
        remediation_codes.dedup();
        remediation_codes.truncate(4);
        snapshots.push(BackendEvidenceSlotSnapshotV1 {
            slot_id,
            target_hash_prefix: slot_evidence.target_hash_prefix,
            backend_support,
            evidence: BackendEvidenceSlotEvidenceV1 {
                latest_probe_report_digest_prefix: slot_evidence.latest_probe_report_digest_prefix,
                latest_compare_window_digest_prefix: slot_evidence
                    .latest_compare_window_digest_prefix,
                latest_shadow_ready_digest_prefix: slot_evidence.latest_shadow_ready_digest_prefix,
                latest_active_evidence_digest_prefix: slot_evidence
                    .latest_active_evidence_digest_prefix,
                latest_drift_status: slot_evidence.latest_drift_status,
                freshness_probe_age_ticks: slot_evidence.freshness_probe_age_ticks,
                freshness_compare_age_ticks: slot_evidence.freshness_compare_age_ticks,
                freshness_no_impact_age_ticks: slot_evidence.freshness_no_impact_age_ticks,
                freshness_drift_status_age_ticks: slot_evidence.freshness_drift_status_age_ticks,
                hash_consistency_ok: slot_evidence.hash_consistent,
            },
            readiness: BackendEvidenceSlotReadinessV1 {
                probe_ready: eligibility.probe_ready,
                shadow_ready: eligibility.shadow_ready,
                active_eligible: eligibility.active_eligible,
            },
            denials: BackendEvidenceSlotDenialsV1 {
                probe: map_denial_reason_to_code(eligibility.denial_reason_probe.as_deref()),
                shadow: map_denial_reason_to_code(eligibility.denial_reason_shadow.as_deref()),
                active: map_denial_reason_to_code(eligibility.denial_reason_active.as_deref()),
            },
            canonical_remediation_codes: merge_canonical_remediations(remediation_codes.iter(), 4),
            remediation_codes,
        });
    }

    snapshots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        BACKEND_EVIDENCE_SNAPSHOT_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(slot_set.set_digest.as_bytes());
    let policy_graph_digest_prefix = read_policy_graph_digest_prefix();
    digest_source.extend_from_slice(policy_graph_digest_prefix.as_bytes());
    let manifest_digest_prefix = prefix_hex(&manifest.manifest_digest, 16);
    digest_source.extend_from_slice(manifest_digest_prefix.as_bytes());
    for slot in &snapshots {
        digest_source.extend_from_slice(slot.slot_id.as_bytes());
        digest_source.extend_from_slice(slot.target_hash_prefix.as_bytes());
        digest_source.extend_from_slice(format!("{:?}", slot.backend_support.stub).as_bytes());
        digest_source.extend_from_slice(format!("{:?}", slot.backend_support.candle).as_bytes());
        digest_source.extend_from_slice(format!("{:?}", slot.backend_support.burn).as_bytes());
        digest_source.extend_from_slice(slot.evidence.latest_probe_report_digest_prefix.as_bytes());
        digest_source
            .extend_from_slice(slot.evidence.latest_compare_window_digest_prefix.as_bytes());
        digest_source.extend_from_slice(slot.evidence.latest_shadow_ready_digest_prefix.as_bytes());
        digest_source.extend_from_slice(
            slot.evidence
                .latest_active_evidence_digest_prefix
                .as_bytes(),
        );
        digest_source
            .extend_from_slice(format!("{:?}", slot.evidence.latest_drift_status).as_bytes());
    }

    Ok(BackendEvidenceSnapshotV1 {
        schema_version: BACKEND_EVIDENCE_SNAPSHOT_SCHEMA_VERSION,
        supported_slot_set_digest: prefix_hex(&slot_set.set_digest, 16),
        policy_graph_digest_prefix,
        manifest_digest_prefix,
        slots: snapshots,
        snapshot_digest: sha256_hex(&digest_source),
    })
}

fn read_second_slot_parity_report(
    workdir: &Path,
    run_id: Option<&str>,
    slot_id: &str,
) -> Option<SecondSlotParityReportV1> {
    let run_path = run_id.map(|rid| {
        workdir
            .join("out")
            .join(rid)
            .join(format!("{}_parity_report.json", slot_id))
    });
    let default_path = workdir
        .join("out")
        .join(format!("{}_parity_report.json", slot_id));
    run_path
        .into_iter()
        .chain(std::iter::once(default_path))
        .find_map(|path| fs::read_to_string(path).ok())
        .and_then(|body| serde_json::from_str::<SecondSlotParityReportV1>(&body).ok())
}

fn resolve_candle_support_state(
    slot: ModelSlot,
    second_slot: Option<ModelSlot>,
    parity_report: Option<&SecondSlotParityReportV1>,
) -> BackendSupportStateV1 {
    if !cfg!(feature = "backend-candle") {
        return BackendSupportStateV1::NotBuilt;
    }
    if second_slot == Some(slot) && parity_report.is_none() {
        return BackendSupportStateV1::NotConfigured;
    }
    BackendSupportStateV1::Supported
}

fn resolve_burn_support_state(
    slot: ModelSlot,
    second_slot: Option<ModelSlot>,
    parity_report: Option<&SecondSlotParityReportV1>,
) -> BackendSupportStateV1 {
    if second_slot != Some(slot) {
        return BackendSupportStateV1::Unsupported;
    }
    match parity_report
        .map(|r| r.burn_support_state.clone())
        .unwrap_or(OptionalBackendSupportStateV1::NotConfigured)
    {
        OptionalBackendSupportStateV1::Supported => BackendSupportStateV1::Supported,
        OptionalBackendSupportStateV1::Unsupported => BackendSupportStateV1::Unsupported,
        OptionalBackendSupportStateV1::NotBuilt => BackendSupportStateV1::NotBuilt,
        OptionalBackendSupportStateV1::NotConfigured => BackendSupportStateV1::NotConfigured,
    }
}

fn derive_eligibility_overall_status(
    slots: &[UnifiedEligibilityStatusV1],
) -> EligibilityOverallStatusV1 {
    let all_active = slots.iter().all(|s| s.active_eligible);
    let any_active = slots.iter().any(|s| s.active_eligible);
    if all_active {
        return EligibilityOverallStatusV1::ActiveEligibleAll;
    }
    if any_active {
        return EligibilityOverallStatusV1::ActiveEligiblePartial;
    }
    let all_shadow = slots.iter().all(|s| s.shadow_ready);
    let any_shadow = slots.iter().any(|s| s.shadow_ready);
    if all_shadow {
        return EligibilityOverallStatusV1::ShadowReadyAll;
    }
    if any_shadow {
        return EligibilityOverallStatusV1::ShadowReadyPartial;
    }
    if slots.iter().any(|s| s.probe_ready) {
        return EligibilityOverallStatusV1::ProbeOnly;
    }
    EligibilityOverallStatusV1::NoneReady
}

fn digest_shadow_generated_from(slots: &[UnifiedEligibilityStatusV1]) -> String {
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(SHADOW_READY_SCHEMA_VERSION.to_string().as_bytes());
    for slot in slots {
        digest_source.extend_from_slice(slot.latest_shadow_evidence_digest_prefix.as_bytes());
    }
    sha256_hex(&digest_source)
}

fn digest_active_generated_from(slots: &[UnifiedEligibilityStatusV1]) -> String {
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(SUPPORTED_REAL_SLOT_SET_VERSION.as_bytes());
    for slot in slots {
        digest_source.extend_from_slice(slot.latest_active_evidence_digest_prefix.as_bytes());
        digest_source.extend_from_slice(if slot.active_eligible { b"1" } else { b"0" });
        if let Some(reason) = slot.denial_reason_active.as_ref() {
            digest_source.extend_from_slice(reason.as_bytes());
        }
    }
    sha256_hex(&digest_source)
}

fn append_eligibility_snapshot_record(
    workdir: &Path,
    report: &AggregatedEligibilityReportV1,
) -> Result<(), OpsError> {
    let path = workdir
        .join("out")
        .join("records")
        .join("models_eligibility_records.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut all: Vec<EligibilitySnapshotRecordV1> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?).unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut slots = report
        .slots
        .iter()
        .map(|slot| EligibilitySnapshotSlotV1 {
            slot_id: slot.slot_id.clone(),
            probe_ready: slot.probe_ready,
            shadow_ready: slot.shadow_ready,
            active_eligible: slot.active_eligible,
        })
        .collect::<Vec<_>>();
    slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    all.push(EligibilitySnapshotRecordV1 {
        invocation_id: now_secs(),
        slots,
        report_digest_prefix: prefix_hex(&report.report_digest, 16),
        policy_graph_digest_prefix: report.policy_graph_digest_prefix.clone(),
    });
    fs::write(path, serde_json::to_vec_pretty(&all)?)?;
    Ok(())
}

pub fn supported_real_slot_set_v1() -> Result<SupportedRealSlotSetV1, OpsError> {
    let second_slot = detect_second_real_slot_from_docs()?;
    let mut slots = vec![ModelSlot::WorldJepa, second_slot];
    slots.sort_by_key(|s| s.as_str().to_string());
    slots.dedup();
    if slots.len() != SHADOW_READY_MAX_SLOTS || slots.len() > SLOT_SET_MAX {
        return Err(OpsError::Invalid(
            "UNSUPPORTED_SLOT_SET: expected exactly world_jepa plus one secondary slot".to_string(),
        ));
    }
    let slots = slots
        .into_iter()
        .map(|s| s.as_str().to_string())
        .collect::<Vec<_>>();
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(SUPPORTED_REAL_SLOT_SET_VERSION.as_bytes());
    digest_source.extend_from_slice(b"docs/series_state_snapshot.md");
    for slot in &slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    Ok(SupportedRealSlotSetV1 {
        schema_version: 1,
        slots,
        source: "docs/series_state_snapshot.md#Second supported slot".to_string(),
        set_digest: sha256_hex(&digest_source),
    })
}

fn supported_real_slots(requested_slot: Option<ModelSlot>) -> Result<Vec<ModelSlot>, OpsError> {
    let set = supported_real_slot_set_v1()?;
    let slots = set
        .slots
        .iter()
        .map(|slot| parse_slot(slot))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(slot) = requested_slot {
        if slots.contains(&slot) {
            Ok(vec![slot])
        } else {
            Err(OpsError::Invalid(format!(
                "SHADOW_READY_SLOT_NOT_SUPPORTED: slot {} not in supported set [{}]",
                slot.as_str(),
                set.slots.join(", ")
            )))
        }
    } else {
        Ok(slots)
    }
}

fn detect_second_real_slot_from_docs() -> Result<ModelSlot, OpsError> {
    let direct = PathBuf::from("docs/series_state_snapshot.md");
    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("docs/series_state_snapshot.md");
    let body = fs::read_to_string(&direct)
        .or_else(|_| fs::read_to_string(&fallback))
        .map_err(|_| {
            OpsError::Invalid(
                "SHADOW_READY_SECOND_SLOT_UNKNOWN: missing docs/series_state_snapshot.md; set tiny real fixture second slot to sae or ssm and regenerate docs".to_string(),
            )
        })?;
    for line in body.lines() {
        if line.contains("Second supported slot") {
            let lower = line.to_ascii_lowercase();
            if lower.contains("`sae`") || lower.contains(" sae") {
                return Ok(ModelSlot::Sae);
            }
            if lower.contains("`ssm`") || lower.contains(" ssm") {
                return Ok(ModelSlot::Ssm);
            }
            return Err(OpsError::Invalid(
                "SHADOW_READY_SECOND_SLOT_UNKNOWN: only sae or ssm are allowed; update docs/series_state_snapshot.md and rerun".to_string(),
            ));
        }
    }
    Err(OpsError::Invalid(
        "SHADOW_READY_SECOND_SLOT_UNKNOWN: unable to locate second-slot declaration; add it to docs/series_state_snapshot.md".to_string(),
    ))
}

fn build_shadow_ready_evidence(
    slot: ModelSlot,
    workdir: &Path,
) -> Result<ShadowReadyEvidenceV1, OpsError> {
    let manifest = load_or_init_manifest()?;
    let slot_id = slot.as_str().to_string();
    let target_hash = manifest
        .slots
        .iter()
        .find(|s| s.slot_id == slot_id)
        .and_then(|s| s.active_hash.clone())
        .unwrap_or_else(|| "missing".to_string());
    let manifest_digest_prefix = prefix_hex(&manifest.manifest_digest, 16);
    let probe_path = PathBuf::from("out").join(format!("probe_{}.json", slot.as_str()));
    let probe: Option<ProbeReportV1> = fs::read_to_string(&probe_path)
        .ok()
        .and_then(|body| serde_json::from_str(&body).ok());
    let latest_probe_report_digest_prefix = probe
        .as_ref()
        .map(|p| prefix_hex(&sha256_hex(&serde_json::to_vec(p).unwrap_or_default()), 16))
        .unwrap_or_else(|| "missing".to_string());
    let latest_probe_status = probe
        .as_ref()
        .map(|p| p.status.clone())
        .unwrap_or(ProbeReportStatus::Fail);

    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let records = load_fixture_records(&fixture_path).unwrap_or_default();
    let mut latest_compare: Option<ucf_ess::v1::SlotCompareWindowRecordV1> = None;
    let mut drift_status = DriftStatusV1::Unknown;
    let mut max_tick = 0_u64;
    for record in records {
        max_tick = max_tick.max(record.time.tick.get());
        if let ExperiencePayload::Audit(payload) = record.payload {
            match payload {
                AuditPayload::SlotCompareWindow(window) if window.slot_id == slot.as_str() => {
                    latest_compare = Some(window);
                }
                AuditPayload::DriftAlarm(alarm) if alarm.slot_id == slot.as_str() => {
                    drift_status = if alarm.severity.eq_ignore_ascii_case("severe") {
                        DriftStatusV1::Severe
                    } else {
                        DriftStatusV1::Warn
                    };
                }
                _ => {}
            }
        }
    }
    let compare_window_present = latest_compare.is_some();
    let latest_compare_window_digest_prefix = latest_compare
        .as_ref()
        .map(compare_digest_prefix)
        .unwrap_or_else(|| "missing".to_string());
    let compare_max_age = crate::load_or_init_config(workdir)
        .map(|cfg| cfg.active_evidence_compare_max_age_ticks.max(1))
        .unwrap_or(256);
    let compare_freshness = crate::compare_freshness(
        latest_compare.as_ref().map(|w| w.t1),
        max_tick,
        compare_max_age,
    );
    if matches!(drift_status, DriftStatusV1::Unknown) {
        drift_status = DriftStatusV1::Ok;
    }
    let no_impact_verified = fs::read_to_string(PathBuf::from("out").join("gate_report.json"))
        .ok()
        .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
        .and_then(|v| v.get("checks").and_then(|c| c.as_array()).cloned())
        .map(|checks| {
            checks.iter().any(|c| {
                c.get("name").and_then(|v| v.as_str()) == Some("shadow_no_decision_impact")
                    && c.get("status").and_then(|v| v.as_str()) == Some("PASS")
            })
        })
        .unwrap_or(false);

    let denial_reason_code = if target_hash == "missing" {
        Some("SHADOW_READY_TARGET_HASH_MISSING".to_string())
    } else if !matches!(latest_probe_status, ProbeReportStatus::Pass) {
        Some("SHADOW_READY_PROBE_REQUIRED".to_string())
    } else if !compare_window_present {
        Some("SHADOW_READY_COMPARE_WINDOW_MISSING".to_string())
    } else if matches!(
        compare_freshness,
        crate::CompareWindowFreshnessV1::StaleCompare
    ) {
        Some("SHADOW_READY_STALE_COMPARE".to_string())
    } else if !no_impact_verified {
        Some("SHADOW_READY_NO_IMPACT_MISSING".to_string())
    } else if matches!(drift_status, DriftStatusV1::Severe) {
        Some("SHADOW_READY_DRIFT_SEVERE".to_string())
    } else {
        None
    };
    let shadow_ready = denial_reason_code.is_none();

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(slot_id.as_bytes());
    digest_source.extend_from_slice(target_hash.as_bytes());
    digest_source.extend_from_slice(manifest_digest_prefix.as_bytes());
    digest_source.extend_from_slice(latest_probe_report_digest_prefix.as_bytes());
    digest_source.extend_from_slice(format!("{:?}", latest_probe_status).as_bytes());
    digest_source.extend_from_slice(latest_compare_window_digest_prefix.as_bytes());
    digest_source.extend_from_slice(if compare_window_present { b"1" } else { b"0" });
    digest_source.extend_from_slice(format!("{:?}", compare_freshness).as_bytes());
    digest_source.extend_from_slice(if no_impact_verified { b"1" } else { b"0" });
    digest_source.extend_from_slice(format!("{:?}", drift_status).as_bytes());
    digest_source.extend_from_slice(if shadow_ready { b"1" } else { b"0" });
    if let Some(reason) = denial_reason_code.as_ref() {
        digest_source.extend_from_slice(reason.as_bytes());
    }
    let evidence_digest = sha256_hex(&digest_source);
    Ok(ShadowReadyEvidenceV1 {
        slot_id,
        target_hash,
        manifest_digest_prefix,
        latest_probe_report_digest_prefix,
        latest_probe_status,
        latest_compare_window_digest_prefix,
        compare_window_present,
        no_impact_verified,
        latest_drift_status: drift_status,
        shadow_ready,
        denial_reason_code,
        evidence_digest,
    })
}

fn compare_digest_prefix(compare: &ucf_ess::v1::SlotCompareWindowRecordV1) -> String {
    let window_id = crate::derive_window_id("compat", &compare.slot_id, compare.t0, compare.t1);
    let compare_digest = sha256_hex(
        format!(
            "{}:{}:{}:{}:{}:{}:{}",
            compare.slot_id,
            compare.t0,
            compare.t1,
            window_id,
            compare.sample_count,
            compare.mean_delta_q,
            compare.p95_delta_q
        )
        .as_bytes(),
    );
    prefix_hex(&compare_digest, 16)
}

fn slot_mode_from_env(slot: ModelSlot) -> &'static str {
    let key = format!("UCF_SLOT_{}_MODE", slot.env_key());
    match std::env::var(key) {
        Ok(v) if v.eq_ignore_ascii_case("active") => "active",
        Ok(v) if v.eq_ignore_ascii_case("shadow") => "shadow",
        Ok(v) if v.eq_ignore_ascii_case("off") => "off",
        _ => "shadow",
    }
}

pub fn models_verify(manifest: &Path) -> Result<ModelsVerifyReport, OpsError> {
    if !manifest.exists() {
        return Ok(ModelsVerifyReport {
            manifest: manifest.display().to_string(),
            manifest_present: false,
            digest_match: false,
            promoted_hashes_exist: false,
            files_verified: false,
        });
    }
    let raw = fs::read_to_string(manifest)?;
    if raw.len() > MAX_MANIFEST_BYTES {
        return Err(OpsError::Invalid(format!(
            "manifest too large: {} > {} bytes",
            raw.len(),
            MAX_MANIFEST_BYTES
        )));
    }
    let parsed: LifecycleManifest = toml::from_str(&raw)
        .map_err(|e| OpsError::Invalid(format!("manifest parse failed: {e}")))?;
    let digest_match = compute_manifest_digest(&parsed)? == parsed.manifest_digest;
    let mut promoted_hashes_exist = true;
    let mut files_verified = true;
    for slot in &parsed.slots {
        if slot.files.len() > MAX_SLOT_FILES {
            return Err(OpsError::Invalid(format!(
                "slot {} has too many files: {} > {}",
                slot.slot_id,
                slot.files.len(),
                MAX_SLOT_FILES
            )));
        }
        if let Some(active_hash) = slot.active_hash.as_ref() {
            let root = PathBuf::from("models")
                .join("promoted")
                .join(&slot.slot_id)
                .join(active_hash);
            if !root.exists() {
                promoted_hashes_exist = false;
                continue;
            }
            for file in &slot.files {
                let path = root.join(&file.path);
                let metadata = match fs::metadata(&path) {
                    Ok(v) => v,
                    Err(_) => {
                        files_verified = false;
                        continue;
                    }
                };
                if metadata.len() != file.size_bytes {
                    files_verified = false;
                    continue;
                }
                if file_sha256_hex(&path)? != file.sha256 {
                    files_verified = false;
                }
            }
        }
    }
    Ok(ModelsVerifyReport {
        manifest: manifest.display().to_string(),
        manifest_present: true,
        digest_match,
        promoted_hashes_exist,
        files_verified,
    })
}

pub fn models_recommend_rollback(
    slot: ModelSlot,
    workdir: &Path,
) -> Result<ModelsRollbackRecommendationReport, OpsError> {
    let ess_path = workdir.join("ess").join("ess_fixture.json");
    let slot_key = slot.as_str().to_string();
    if !ess_path.exists() {
        return Ok(ModelsRollbackRecommendationReport {
            slot: slot_key,
            recommendations: Vec::new(),
        });
    }
    let mut recommendations = Vec::new();
    for record in load_fixture_records(&ess_path)? {
        let ExperiencePayload::Text(note) = record.payload else {
            continue;
        };
        let note = note.to_string();
        if note.starts_with("model_rollback_recommendation")
            && note.contains(&format!("slot={}", slot.as_str()))
        {
            recommendations.push(RollbackRecommendation {
                slot: slot.as_str().to_string(),
                tick: record.time.tick.get(),
                note,
            });
        }
    }
    recommendations.sort_by_key(|r| r.tick);
    recommendations.reverse();
    recommendations.truncate(8);
    Ok(ModelsRollbackRecommendationReport {
        slot: slot_key,
        recommendations,
    })
}

pub fn parse_slot(value: &str) -> Result<ModelSlot, OpsError> {
    match value {
        "llm" => Ok(ModelSlot::Llm),
        "world" => Ok(ModelSlot::WorldJepa),
        "world_jepa" => Ok(ModelSlot::WorldJepa),
        "world_vljepa" => Ok(ModelSlot::WorldVljepa),
        "sae" => Ok(ModelSlot::Sae),
        "lfm" => Ok(ModelSlot::Lfm),
        "ssm" => Ok(ModelSlot::Ssm),
        "ebm_reasoner" | "ebm" => Ok(ModelSlot::EbmReasoner),
        _ => Err(OpsError::Invalid(format!("unknown slot: {value}"))),
    }
}

pub fn models_probe_slot(
    slot: ModelSlot,
    hash: Option<&str>,
    out: &Path,
) -> Result<ProbeReportV1, OpsError> {
    let started = Instant::now();
    let manifest = load_or_init_manifest()?;
    let slot_id = slot.as_str().to_string();
    let slot_manifest = manifest.slots.iter().find(|s| s.slot_id == slot_id);

    let (mode, source_dir, model_hash) = if let Some(probe_hash) = hash {
        let staged = PathBuf::from("models")
            .join("staging")
            .join(slot.as_str())
            .join(probe_hash);
        if !staged.exists() {
            return Err(OpsError::Invalid(format!(
                "UCF_OPS_MODELS_PROBE_STAGED_HASH_MISSING: {probe_hash}"
            )));
        }
        (ProbeMode::Hash, Some(staged), Some(probe_hash.to_string()))
    } else if let Some(active_hash) = slot_manifest.and_then(|s| s.active_hash.clone()) {
        let promoted = PathBuf::from("models")
            .join("promoted")
            .join(slot.as_str())
            .join(&active_hash);
        if !promoted.exists() {
            return Err(OpsError::Invalid(format!(
                "UCF_OPS_MODELS_PROBE_ACTIVE_HASH_MISSING: {}",
                active_hash
            )));
        }
        (ProbeMode::Active, Some(promoted), Some(active_hash))
    } else {
        (ProbeMode::Stub, None, None)
    };

    let contract_version = slot_manifest
        .and_then(|s| s.contract_versions_supported.first().cloned())
        .unwrap_or_else(|| "v1".to_string());

    let (outputs, backend_id) = build_probe_outputs(
        slot,
        source_dir.as_deref(),
        model_hash.as_deref(),
        &manifest,
        mode.clone(),
    )?;
    let mut envelope_checks = build_envelope_checks(slot, &outputs);
    let pass = envelope_checks
        .iter()
        .all(|check| matches!(check.status, ProbeCheckStatus::Pass));
    if envelope_checks.len() > PROBE_NOTES_CAP {
        envelope_checks.truncate(PROBE_NOTES_CAP);
    }

    let report = ProbeReportV1 {
        schema_version: PROBE_SCHEMA_VERSION,
        slot_id,
        mode,
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        model_hash_prefix: model_hash
            .as_ref()
            .map(|value| value.chars().take(PROBE_DIGEST_PREFIX_LEN).collect()),
        backend_id,
        contract_version,
        outputs,
        latency_ms: started.elapsed().as_millis() as u64,
        envelope_checks,
        status: if pass {
            ProbeReportStatus::Pass
        } else {
            ProbeReportStatus::Fail
        },
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn build_probe_outputs(
    slot: ModelSlot,
    source_dir: Option<&Path>,
    model_hash: Option<&str>,
    manifest: &LifecycleManifest,
    mode: ProbeMode,
) -> Result<(ProbeOutputs, String), OpsError> {
    let slot_id = slot.as_str();
    let mut seed_material = Vec::new();
    seed_material.extend_from_slice(slot_id.as_bytes());
    if let Some(hash) = model_hash {
        seed_material.extend_from_slice(hash.as_bytes());
    }
    if let Some(dir) = source_dir {
        let model_path = dir.join(MODEL_FILE_NAME);
        if !model_path.exists() {
            return Err(OpsError::Invalid(format!(
                "UCF_OPS_MODELS_PROBE_MODEL_FILE_MISSING: {}",
                model_path.display()
            )));
        }
        let bytes = fs::read(&model_path)?;
        if let ProbeMode::Active = mode {
            if let Some(expected) = manifest
                .slots
                .iter()
                .find(|s| s.slot_id == slot_id)
                .and_then(|s| s.files.iter().find(|f| f.path == MODEL_FILE_NAME))
            {
                let found = sha256_hex(&bytes);
                if found != expected.sha256 {
                    return Err(OpsError::Invalid(
                        "UCF_OPS_MODELS_PROBE_ACTIVE_SHA_MISMATCH".to_string(),
                    ));
                }
            }
        }
        seed_material.extend_from_slice(&bytes);
    }
    let slot_digest = sha256_hex(&seed_material);
    let alt_digest = sha256_hex(format!("{slot_id}:alt:{slot_digest}").as_bytes());
    let scalar_a = scalar_q(&slot_digest, 0);
    let scalar_b = scalar_q(&slot_digest, 4);
    let spike_count = (u32::from(scalar_q(&slot_digest, 8)) % SAE_SPIKE_COUNT_KMAX) + 1;

    let mut digests = match slot {
        ModelSlot::Llm => vec![
            ProbeDigestOutput {
                key: "response_digest".to_string(),
                digest_prefix: slot_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
            },
            ProbeDigestOutput {
                key: "contract_digest".to_string(),
                digest_prefix: alt_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
            },
        ],
        ModelSlot::WorldJepa | ModelSlot::WorldVljepa => vec![ProbeDigestOutput {
            key: "prediction_digest".to_string(),
            digest_prefix: slot_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
        }],
        ModelSlot::Sae => vec![ProbeDigestOutput {
            key: "spikes_digest".to_string(),
            digest_prefix: slot_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
        }],
        ModelSlot::Ssm => vec![ProbeDigestOutput {
            key: "state_digest".to_string(),
            digest_prefix: slot_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
        }],
        ModelSlot::Lfm => vec![
            ProbeDigestOutput {
                key: "uncertainty_digest".to_string(),
                digest_prefix: slot_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
            },
            ProbeDigestOutput {
                key: "stability_digest".to_string(),
                digest_prefix: alt_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
            },
        ],
        ModelSlot::EbmReasoner => vec![
            ProbeDigestOutput {
                key: "energy_digest".to_string(),
                digest_prefix: slot_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
            },
            ProbeDigestOutput {
                key: "risk_digest".to_string(),
                digest_prefix: alt_digest.chars().take(PROBE_DIGEST_PREFIX_LEN).collect(),
            },
        ],
    };
    digests.sort_by(|a, b| a.key.cmp(&b.key));
    digests.truncate(PROBE_OUTPUT_CAP);

    let mut scalars = match slot {
        ModelSlot::Llm => vec![ProbeScalarOutput {
            key: "completion_confidence_q".to_string(),
            value_q: scalar_a,
        }],
        ModelSlot::WorldJepa | ModelSlot::WorldVljepa => vec![ProbeScalarOutput {
            key: "prediction_error_q".to_string(),
            value_q: scalar_a,
        }],
        ModelSlot::Sae => vec![ProbeScalarOutput {
            key: "spike_ratio_q".to_string(),
            value_q: scalar_a,
        }],
        ModelSlot::Ssm => vec![ProbeScalarOutput {
            key: "pressure_q".to_string(),
            value_q: scalar_a,
        }],
        ModelSlot::Lfm => vec![
            ProbeScalarOutput {
                key: "uncertainty_q".to_string(),
                value_q: scalar_a,
            },
            ProbeScalarOutput {
                key: "stability_q".to_string(),
                value_q: scalar_b,
            },
        ],
        ModelSlot::EbmReasoner => vec![
            ProbeScalarOutput {
                key: "energy_q".to_string(),
                value_q: scalar_a,
            },
            ProbeScalarOutput {
                key: "risk_q".to_string(),
                value_q: scalar_b,
            },
        ],
    };
    scalars.sort_by(|a, b| a.key.cmp(&b.key));
    scalars.truncate(PROBE_OUTPUT_CAP);

    let mut counters = match slot {
        ModelSlot::Llm => vec![
            ProbeCounterOutput {
                key: "prompt_tokens".to_string(),
                value: 8,
            },
            ProbeCounterOutput {
                key: "completion_tokens".to_string(),
                value: 12,
            },
        ],
        ModelSlot::Sae => vec![ProbeCounterOutput {
            key: "spike_count".to_string(),
            value: spike_count,
        }],
        _ => Vec::new(),
    };
    if let Some(dir) = source_dir {
        let model_path = dir.join(MODEL_FILE_NAME);
        let model_len = fs::metadata(model_path)?.len() as u32;
        counters.push(ProbeCounterOutput {
            key: "model_bytes".to_string(),
            value: model_len,
        });
    }
    counters.sort_by(|a, b| a.key.cmp(&b.key));
    counters.truncate(PROBE_OUTPUT_CAP);

    #[cfg(feature = "backend-candle")]
    if slot == ModelSlot::Sae {
        if let Some(dir) = source_dir {
            let path = dir.join(MODEL_FILE_NAME);
            if path.exists() {
                let model_bytes = fs::read(&path)?;
                let mut hash = [0_u8; 32];
                if let Some(h) = model_hash {
                    if let Ok(decoded) = hex::decode(h) {
                        if decoded.len() == 32 {
                            hash.copy_from_slice(&decoded);
                        }
                    }
                }
                if let Ok(adapter) =
                    ucf_compute::stage_v1_candle::CandleSaeAdapterV0::from_safetensors_bytes(
                        hash,
                        &model_bytes,
                    )
                {
                    let input = ucf_compute::stage_v1::SaeInputV1 {
                        context_digest: [0x2Au8; 32],
                        prediction_digest: [0x51u8; 32],
                        top_k: 8,
                    };
                    if let Ok(out) = ucf_compute::stage_v1::SaeExtractorV1::infer(&adapter, &input)
                    {
                        let mut candle_digests = vec![ProbeDigestOutput {
                            key: "spikes_digest".to_string(),
                            digest_prefix: prefix_hex(
                                &hex::encode(out.spikes_digest),
                                PROBE_DIGEST_PREFIX_LEN,
                            ),
                        }];
                        candle_digests.sort_by(|a, b| a.key.cmp(&b.key));
                        let candle_scalars = vec![ProbeScalarOutput {
                            key: "spike_ratio_q".to_string(),
                            value_q: (u16::try_from(out.spikes.len())
                                .unwrap_or(0)
                                .saturating_mul(1000)
                                / u16::try_from(SAE_SPIKE_COUNT_KMAX).unwrap_or(1))
                            .min(Q01_MAX),
                        }];
                        let mut candle_counters = vec![ProbeCounterOutput {
                            key: "spike_count".to_string(),
                            value: out.spikes.len() as u32,
                        }];
                        if let Some(dir2) = source_dir {
                            let model_path = dir2.join(MODEL_FILE_NAME);
                            let model_len = fs::metadata(model_path)?.len() as u32;
                            candle_counters.push(ProbeCounterOutput {
                                key: "model_bytes".to_string(),
                                value: model_len,
                            });
                            candle_counters.sort_by(|a, b| a.key.cmp(&b.key));
                        }
                        return Ok((
                            ProbeOutputs {
                                digests: candle_digests,
                                scalars: candle_scalars,
                                counters: candle_counters,
                            },
                            format!(
                                "candle:sae:{}",
                                ucf_compute::stage_v1::SaeExtractorV1::backend_id(&adapter)
                            ),
                        ));
                    }
                }
            }
        }
    }

    Ok((
        ProbeOutputs {
            digests,
            scalars,
            counters,
        },
        "offline_probe_stub_v1".to_string(),
    ))
}

fn build_envelope_checks(slot: ModelSlot, outputs: &ProbeOutputs) -> Vec<ProbeEnvelopeCheck> {
    let mut checks = Vec::new();
    let scalar_bounds_ok = outputs.scalars.iter().all(|item| item.value_q <= Q01_MAX);
    checks.push(ProbeEnvelopeCheck {
        code: "PROBE_SCALAR_BOUNDS".to_string(),
        status: if scalar_bounds_ok {
            ProbeCheckStatus::Pass
        } else {
            ProbeCheckStatus::Fail
        },
    });

    let digest_non_zero = outputs.digests.iter().all(|item| {
        !item.digest_prefix.is_empty() && item.digest_prefix.chars().any(|ch| ch != '0')
    });
    checks.push(ProbeEnvelopeCheck {
        code: "PROBE_DIGEST_NON_ZERO".to_string(),
        status: if digest_non_zero {
            ProbeCheckStatus::Pass
        } else {
            ProbeCheckStatus::Fail
        },
    });

    let model_bytes_ok = outputs
        .counters
        .iter()
        .find(|v| v.key == "model_bytes")
        .map(|v| v.value > 0)
        .unwrap_or(true);
    checks.push(ProbeEnvelopeCheck {
        code: "PROBE_MODEL_BYTES_NON_ZERO".to_string(),
        status: if model_bytes_ok {
            ProbeCheckStatus::Pass
        } else {
            ProbeCheckStatus::Fail
        },
    });
    let output_caps_ok = outputs.digests.len() <= PROBE_OUTPUT_CAP
        && outputs.scalars.len() <= PROBE_OUTPUT_CAP
        && outputs.counters.len() <= PROBE_OUTPUT_CAP;
    checks.push(ProbeEnvelopeCheck {
        code: "PROBE_OUTPUT_CAP".to_string(),
        status: if output_caps_ok {
            ProbeCheckStatus::Pass
        } else {
            ProbeCheckStatus::Fail
        },
    });

    if slot == ModelSlot::Sae {
        let spike_ok = outputs
            .counters
            .iter()
            .find(|v| v.key == "spike_count")
            .map(|v| v.value <= SAE_SPIKE_COUNT_KMAX)
            .unwrap_or(false);
        checks.push(ProbeEnvelopeCheck {
            code: "PROBE_SAE_SPIKE_COUNT_BOUNDED".to_string(),
            status: if spike_ok {
                ProbeCheckStatus::Pass
            } else {
                ProbeCheckStatus::Fail
            },
        });
    }
    checks
}

fn scalar_q(hex: &str, offset: usize) -> u16 {
    let start = offset.min(hex.len());
    let end = (start + 4).min(hex.len());
    let fragment = &hex[start..end];
    let raw = u16::from_str_radix(fragment, 16).unwrap_or(0);
    raw % (Q01_MAX + 1)
}

fn persist_manifest_with_history(
    manifest: &mut LifecycleManifest,
    history_keep: Option<usize>,
) -> Result<(), OpsError> {
    manifest.manifest_digest = compute_manifest_digest(manifest)?;
    manifest.created_at = Some(now_secs());
    fs::create_dir_all("models")?;
    let body = toml::to_string_pretty(manifest)
        .map_err(|e| OpsError::Invalid(format!("manifest serialize failed: {e}")))?;
    fs::write(MANIFEST_PATH, body.as_bytes())?;
    fs::write(
        LEGACY_MANIFEST_PATH,
        legacy_manifest_toml(manifest)?.as_bytes(),
    )?;
    let hist_dir = PathBuf::from("models/manifests/history");
    fs::create_dir_all(&hist_dir)?;
    let name = format!(
        "{}_{}.toml",
        now_millis(),
        manifest
            .manifest_digest
            .chars()
            .take(12)
            .collect::<String>()
    );
    fs::write(hist_dir.join(name), body.as_bytes())?;
    trim_history(&hist_dir, resolve_history_keep(history_keep))?;
    Ok(())
}

fn load_or_init_manifest() -> Result<LifecycleManifest, OpsError> {
    let canonical_path = PathBuf::from(MANIFEST_PATH);
    let legacy_path = PathBuf::from(LEGACY_MANIFEST_PATH);
    if !canonical_path.exists() && !legacy_path.exists() {
        let mut out = LifecycleManifest {
            manifest_version: 1,
            created_at: None,
            slots: Vec::new(),
            manifest_digest: String::new(),
        };
        out.manifest_digest = compute_manifest_digest(&out)?;
        return Ok(out);
    }
    let path = if canonical_path.exists() {
        canonical_path
    } else {
        legacy_path
    };
    let raw = fs::read_to_string(path)?;
    if let Ok(parsed) = toml::from_str::<LifecycleManifest>(&raw) {
        return Ok(parsed);
    }
    #[derive(Deserialize)]
    struct LegacySlot {
        active_hash: Option<String>,
    }
    #[derive(Deserialize)]
    struct LegacyManifest {
        slots: std::collections::BTreeMap<String, LegacySlot>,
        manifest_digest: Option<String>,
    }
    let legacy: LegacyManifest = toml::from_str(&raw)
        .map_err(|e| OpsError::Invalid(format!("manifest parse failed: {e}")))?;
    Ok(LifecycleManifest {
        manifest_version: 1,
        created_at: None,
        slots: legacy
            .slots
            .into_iter()
            .map(|(slot_id, slot)| LifecycleSlotManifest {
                slot_id,
                active_hash: slot.active_hash,
                files: Vec::new(),
                max_bytes: 64 * 1024 * 1024,
                contract_versions_supported: vec!["v1".to_string()],
            })
            .collect(),
        manifest_digest: legacy.manifest_digest.unwrap_or_default(),
    })
}

fn legacy_manifest_toml(manifest: &LifecycleManifest) -> Result<String, OpsError> {
    #[derive(Serialize)]
    struct LegacySlotManifest {
        active_hash: Option<String>,
    }
    #[derive(Serialize)]
    struct LegacyManifest {
        slots: std::collections::BTreeMap<String, LegacySlotManifest>,
        manifest_digest: String,
    }
    let slots = manifest
        .slots
        .iter()
        .map(|s| {
            (
                s.slot_id.clone(),
                LegacySlotManifest {
                    active_hash: s.active_hash.clone(),
                },
            )
        })
        .collect();
    toml::to_string_pretty(&LegacyManifest {
        slots,
        manifest_digest: manifest.manifest_digest.clone(),
    })
    .map_err(|e| OpsError::Invalid(format!("legacy manifest serialize failed: {e}")))
}

fn compute_manifest_digest(manifest: &LifecycleManifest) -> Result<String, OpsError> {
    #[derive(Serialize)]
    struct Canonical<'a> {
        manifest_version: u16,
        slots: &'a [LifecycleSlotManifest],
    }
    let mut slots = manifest.slots.clone();
    slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    for slot in &mut slots {
        slot.files.sort_by(|a, b| a.path.cmp(&b.path));
        slot.contract_versions_supported.sort();
    }
    let canonical = serde_json::to_vec(&Canonical {
        manifest_version: manifest.manifest_version,
        slots: &slots,
    })?;
    Ok(sha256_hex(&canonical))
}

fn ensure_manifest_slot(slot: ModelSlot) -> Result<(), OpsError> {
    let mut manifest = load_or_init_manifest()?;
    let slot_id = slot.as_str().to_string();
    if !manifest.slots.iter().any(|s| s.slot_id == slot_id) {
        manifest.slots.push(LifecycleSlotManifest {
            slot_id,
            active_hash: None,
            files: Vec::new(),
            max_bytes: 64 * 1024 * 1024,
            contract_versions_supported: vec!["v1".to_string()],
        });
        persist_manifest_with_history(&mut manifest, None)?;
    }
    Ok(())
}

fn upsert_slot_manifest(
    manifest: &mut LifecycleManifest,
    slot_id: String,
    active_hash: Option<String>,
    files: Vec<ModelFileEntry>,
) {
    if let Some(entry) = manifest.slots.iter_mut().find(|s| s.slot_id == slot_id) {
        entry.active_hash = active_hash;
        entry.files = files;
        entry.max_bytes = 64 * 1024 * 1024;
        entry.contract_versions_supported = vec!["v1".to_string()];
        return;
    }
    manifest.slots.push(LifecycleSlotManifest {
        slot_id,
        active_hash,
        files,
        max_bytes: 64 * 1024 * 1024,
        contract_versions_supported: vec!["v1".to_string()],
    });
}

fn digest_entries(entries: &[ModelFileEntry]) -> String {
    let mut hasher = Sha256::new();
    for entry in entries {
        hasher.update(entry.path.as_bytes());
        hasher.update([0]);
        hasher.update(entry.sha256.as_bytes());
        hasher.update([0]);
        hasher.update(entry.size_bytes.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>, OpsError> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let p = entry.path();
            if entry.file_type()?.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn collect_file_entries(root: &Path) -> Result<Vec<ModelFileEntry>, OpsError> {
    let files = collect_files(root)?;
    let mut out = Vec::new();
    for file in files {
        let rel = file
            .strip_prefix(root)
            .map_err(|e| OpsError::Invalid(e.to_string()))?;
        out.push(ModelFileEntry {
            path: rel.to_string_lossy().to_string(),
            sha256: file_sha256_hex(&file)?,
            size_bytes: fs::metadata(&file)?.len(),
        });
    }
    Ok(out)
}

fn copy_tree(src: &Path, dst: &Path) -> Result<(), OpsError> {
    fs::create_dir_all(dst)?;
    let files = collect_files(src)?;
    for file in files {
        let rel = file
            .strip_prefix(src)
            .map_err(|e| OpsError::Invalid(e.to_string()))?;
        let out = dst.join(rel);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(file, out)?;
    }
    Ok(())
}

fn file_sha256_hex(path: &Path) -> Result<String, OpsError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 16 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn list_hash_dirs(base: &Path) -> Result<Vec<String>, OpsError> {
    if !base.exists() {
        return Ok(Vec::new());
    }
    let mut out = fs::read_dir(base)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    out.sort();
    Ok(out)
}

fn trim_history(dir: &Path, keep: usize) -> Result<(), OpsError> {
    let mut entries = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .collect::<Vec<_>>();
    entries.sort_by_key(|e| e.file_name());
    if entries.len() > keep {
        for entry in &entries[..entries.len() - keep] {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

fn resolve_history_keep(history_keep: Option<usize>) -> usize {
    history_keep
        .or_else(|| {
            std::env::var("UCF_MODELS_MANIFEST_HISTORY_KEEP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        })
        .filter(|v| *v > 0)
        .unwrap_or(MANIFEST_HISTORY_KEEP)
}

#[derive(Debug)]
struct StagedVerifyResult {
    pass: bool,
    reason: String,
}

fn verify_staged_candidate(slot: ModelSlot, hash: &str) -> Result<StagedVerifyResult, OpsError> {
    let staged = PathBuf::from("models")
        .join("staging")
        .join(slot.as_str())
        .join(hash);
    let entries = collect_file_entries(&staged)?;
    if entries.is_empty() {
        return Ok(StagedVerifyResult {
            pass: false,
            reason: "staged directory has no files".to_string(),
        });
    }
    if entries.len() > MAX_SLOT_FILES {
        return Ok(StagedVerifyResult {
            pass: false,
            reason: format!("too many files: {} > {}", entries.len(), MAX_SLOT_FILES),
        });
    }
    if digest_entries(&entries) != hash {
        return Ok(StagedVerifyResult {
            pass: false,
            reason: "digest mismatch for staged artifact".to_string(),
        });
    }
    let mut manifest = load_or_init_manifest()?;
    let slot_id = slot.as_str().to_string();
    let max_bytes = manifest
        .slots
        .iter_mut()
        .find(|s| s.slot_id == slot_id)
        .map(|s| s.max_bytes)
        .unwrap_or(64 * 1024 * 1024);
    let total = entries.iter().map(|e| e.size_bytes).sum::<u64>();
    if total > max_bytes {
        return Ok(StagedVerifyResult {
            pass: false,
            reason: format!("size cap exceeded: {total} > {max_bytes}"),
        });
    }
    Ok(StagedVerifyResult {
        pass: true,
        reason: "PASS".to_string(),
    })
}

fn resolve_rollback_target(
    slot: ModelSlot,
    to_hash: Option<&str>,
    steps: Option<usize>,
) -> Result<String, OpsError> {
    if let Some(to_hash) = to_hash {
        return Ok(to_hash.to_string());
    }
    let steps = steps.unwrap_or(1);
    if steps == 0 {
        return Err(OpsError::Invalid("--steps must be >= 1".to_string()));
    }
    let current_hash = load_or_init_manifest()?
        .slots
        .into_iter()
        .find(|s| s.slot_id == slot.as_str())
        .and_then(|s| s.active_hash);
    let mut history = load_history_manifests()?;
    history.reverse();
    let mut candidates = Vec::<String>::new();
    for manifest in history {
        let Some(slot_entry) = manifest.slots.iter().find(|s| s.slot_id == slot.as_str()) else {
            continue;
        };
        let Some(hash) = slot_entry.active_hash.clone() else {
            continue;
        };
        if candidates.last() != Some(&hash) {
            candidates.push(hash);
        }
    }
    let mut previous = Vec::<String>::new();
    for hash in candidates {
        if current_hash.as_deref() == Some(hash.as_str()) {
            continue;
        }
        if previous.last() != Some(&hash) {
            previous.push(hash);
        }
    }
    if previous.len() < steps {
        return Err(OpsError::Invalid(format!(
            "rollback target unavailable for --steps {steps}"
        )));
    }
    Ok(previous[steps - 1].clone())
}

fn load_history_manifests() -> Result<Vec<LifecycleManifest>, OpsError> {
    let dir = PathBuf::from("models/manifests/history");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .collect::<Vec<_>>();
    entries.sort_by_key(|e| e.file_name());
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let parsed = toml::from_str::<LifecycleManifest>(&fs::read_to_string(entry.path())?)
            .map_err(|e| OpsError::Invalid(format!("history manifest parse failed: {e}")))?;
        out.push(parsed);
    }
    Ok(out)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn persist_action_record(
    action: &str,
    slot: &str,
    from_hash: &Option<String>,
    to_hash: &str,
    manifest: &LifecycleManifest,
) -> Result<(), OpsError> {
    let report = LifecycleActionReport {
        action: action.to_string(),
        shadow_report_digest_prefix: None,
        slot: slot.to_string(),
        from_hash: from_hash.clone(),
        to_hash: to_hash.to_string(),
        old_manifest_digest_prefix: None,
        new_manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        probe_report_digest_prefix: None,
        readiness_gate_digest_prefix: None,
        timestamp: now_secs(),
    };
    let path = PathBuf::from("out").join("model_promotion_records.json");
    append_action(path, &report)
}

fn persist_rollback_record(
    slot: &str,
    from_hash: &Option<String>,
    to_hash: &str,
    manifest: &LifecycleManifest,
) -> Result<(), OpsError> {
    let report = LifecycleActionReport {
        action: "rollback".to_string(),
        shadow_report_digest_prefix: None,
        slot: slot.to_string(),
        from_hash: from_hash.clone(),
        to_hash: to_hash.to_string(),
        old_manifest_digest_prefix: None,
        new_manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        manifest_digest_prefix: manifest.manifest_digest.chars().take(12).collect(),
        probe_report_digest_prefix: None,
        readiness_gate_digest_prefix: None,
        timestamp: now_secs(),
    };
    let path = PathBuf::from("out").join("model_rollback_records.json");
    append_action(path, &report)
}

fn append_action(path: PathBuf, report: &LifecycleActionReport) -> Result<(), OpsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut all: Vec<LifecycleActionReport> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };
    all.push(report.clone());
    fs::write(path, serde_json::to_string_pretty(&all)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CwdGuard {
        prev: PathBuf,
    }

    impl CwdGuard {
        fn enter(path: &Path) -> Self {
            let prev = std::env::current_dir().expect("cwd");
            std::env::set_current_dir(path).expect("chdir");
            Self { prev }
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.prev);
        }
    }

    #[test]
    fn manifest_digest_canonical_stable() {
        let mut a = LifecycleManifest {
            manifest_version: 1,
            created_at: Some(10),
            slots: vec![
                LifecycleSlotManifest {
                    slot_id: "sae".to_string(),
                    active_hash: Some("bb".to_string()),
                    files: vec![ModelFileEntry {
                        path: "b.txt".to_string(),
                        sha256: "2".repeat(64),
                        size_bytes: 2,
                    }],
                    max_bytes: 1024,
                    contract_versions_supported: vec!["v2".to_string(), "v1".to_string()],
                },
                LifecycleSlotManifest {
                    slot_id: "llm".to_string(),
                    active_hash: Some("aa".to_string()),
                    files: vec![ModelFileEntry {
                        path: "a.txt".to_string(),
                        sha256: "1".repeat(64),
                        size_bytes: 1,
                    }],
                    max_bytes: 1024,
                    contract_versions_supported: vec!["v1".to_string()],
                },
            ],
            manifest_digest: String::new(),
        };
        let mut b = LifecycleManifest {
            manifest_version: 1,
            created_at: Some(99),
            slots: vec![a.slots[1].clone(), a.slots[0].clone()],
            manifest_digest: String::new(),
        };
        a.manifest_digest = compute_manifest_digest(&a).expect("digest");
        b.manifest_digest = compute_manifest_digest(&b).expect("digest");
        assert_eq!(a.manifest_digest, b.manifest_digest);
    }

    #[test]
    fn stage_and_verify_detects_tamper() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("src");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"abc").expect("write");
        let staged = models_stage(ModelSlot::Llm, &src).expect("stage");
        let staged_path = dir
            .path()
            .join("models/staging/llm")
            .join(&staged.hash)
            .join("model.safetensors");
        assert!(staged_path.exists());

        let promoted = dir.path().join("models/promoted/llm").join(&staged.hash);
        copy_tree(
            &dir.path().join("models/staging/llm").join(&staged.hash),
            &promoted,
        )
        .expect("promoted copy");
        let mut manifest = load_or_init_manifest().expect("manifest");
        upsert_slot_manifest(
            &mut manifest,
            "llm".to_string(),
            Some(staged.hash.clone()),
            collect_file_entries(&promoted).expect("entries"),
        );
        persist_manifest_with_history(&mut manifest, None).expect("persist");

        let ok = models_verify(Path::new("models/MANIFEST.toml")).expect("verify");
        assert!(ok.digest_match && ok.promoted_hashes_exist && ok.files_verified);

        fs::write(promoted.join("model.safetensors"), b"tampered").expect("tamper");
        let bad = models_verify(Path::new("models/MANIFEST.toml")).expect("verify bad");
        assert!(!bad.files_verified);
    }

    #[test]
    fn stage_rejects_too_many_files() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
        let src = dir.path().join("src");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"abc").expect("model");
        for i in 0..MAX_SLOT_FILES {
            fs::write(src.join(format!("f_{i}.bin")), b"x").expect("file");
        }
        let err = models_stage(ModelSlot::Llm, &src).expect_err("must reject");
        assert!(format!("{err}").contains("too many files"));
    }

    #[test]
    fn promote_and_rollback_write_history_and_prune() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("src");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"v1").expect("write");
        let staged_v1 = models_stage(ModelSlot::Llm, &src).expect("stage v1");
        let promote_v1 =
            models_promote(ModelSlot::Llm, &staged_v1.hash, Some(2)).expect("promote v1");
        assert_eq!(promote_v1.to_hash, staged_v1.hash);

        fs::write(src.join("model.safetensors"), b"v2").expect("write");
        let staged_v2 = models_stage(ModelSlot::Llm, &src).expect("stage v2");
        let _promote_v2 =
            models_promote(ModelSlot::Llm, &staged_v2.hash, Some(2)).expect("promote v2");

        let rollback = models_rollback(ModelSlot::Llm, Some(&staged_v1.hash), None, Some(2))
            .expect("rollback");
        assert_eq!(rollback.to_hash, staged_v1.hash);

        let manifest = load_or_init_manifest().expect("manifest");
        let llm = manifest
            .slots
            .iter()
            .find(|s| s.slot_id == "llm")
            .expect("llm slot");
        assert_eq!(llm.active_hash.as_deref(), Some(staged_v1.hash.as_str()));

        let hist_count = fs::read_dir("models/manifests/history")
            .expect("history")
            .filter_map(|e| e.ok())
            .count();
        assert!(hist_count <= 2);
    }

    #[test]
    fn rollback_steps_selects_previous_hash() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("src");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"h1").expect("h1");
        let h1 = models_stage(ModelSlot::Llm, &src).expect("stage h1").hash;
        models_promote(ModelSlot::Llm, &h1, None).expect("promote h1");
        fs::write(src.join("model.safetensors"), b"h2").expect("h2");
        let h2 = models_stage(ModelSlot::Llm, &src).expect("stage h2").hash;
        models_promote(ModelSlot::Llm, &h2, None).expect("promote h2");

        let report = models_rollback(ModelSlot::Llm, None, Some(1), None).expect("rollback steps");
        assert_eq!(report.to_hash, h1);
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn materialize_world_real_tiny_fixture(dst_root: &Path) -> PathBuf {
        let fixture_hex = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/weights/world_real_tiny.safetensors.hex");
        let raw_hex = fs::read_to_string(&fixture_hex).expect("fixture hex");
        let bytes =
            hex::decode(raw_hex.trim()).expect("fixture hex must decode into deterministic bytes");
        let dir = dst_root.join("fixtures/weights/world_real_tiny_dir");
        fs::create_dir_all(&dir).expect("fixture dir");
        fs::write(dir.join("model.safetensors"), bytes).expect("fixture model");
        dir
    }

    fn materialize_sae_real_tiny_fixture(dst_root: &Path) -> PathBuf {
        let fixture_hex = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/weights/sae_real_tiny.safetensors.hex");
        let raw_hex = fs::read_to_string(&fixture_hex).expect("fixture hex");
        let bytes =
            hex::decode(raw_hex.trim()).expect("fixture hex must decode into deterministic bytes");
        let dir = dst_root.join("fixtures/weights/sae_real_tiny_dir");
        fs::create_dir_all(&dir).expect("fixture dir");
        fs::write(dir.join("model.safetensors"), bytes).expect("fixture model");
        dir
    }

    struct CwdGuard {
        prev: PathBuf,
    }

    impl CwdGuard {
        fn enter(path: &Path) -> Self {
            let prev = std::env::current_dir().expect("cwd");
            std::env::set_current_dir(path).expect("chdir");
            Self { prev }
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.prev);
        }
    }

    #[test]
    fn probe_report_serialization_is_stable() {
        let report = ProbeReportV1 {
            schema_version: 1,
            slot_id: "llm".to_string(),
            mode: ProbeMode::Stub,
            manifest_digest_prefix: "abcd1234abcd".to_string(),
            model_hash_prefix: None,
            backend_id: "offline_probe_stub_v1".to_string(),
            contract_version: "v1".to_string(),
            outputs: ProbeOutputs {
                digests: vec![ProbeDigestOutput {
                    key: "response_digest".to_string(),
                    digest_prefix: "1234abcd1234abcd".to_string(),
                }],
                scalars: vec![ProbeScalarOutput {
                    key: "completion_confidence_q".to_string(),
                    value_q: 42,
                }],
                counters: vec![ProbeCounterOutput {
                    key: "prompt_tokens".to_string(),
                    value: 8,
                }],
            },
            latency_ms: 1,
            envelope_checks: vec![ProbeEnvelopeCheck {
                code: "PROBE_SCALAR_BOUNDS".to_string(),
                status: ProbeCheckStatus::Pass,
            }],
            status: ProbeReportStatus::Pass,
        };
        let a = serde_json::to_string_pretty(&report).expect("json a");
        let b = serde_json::to_string_pretty(&report).expect("json b");
        assert_eq!(a, b);
    }

    #[test]
    fn probe_stage_promote_probe_pass() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("fixtures/models_dummy/llm");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"dummy-llm-v1").expect("model");

        let staged = models_stage(ModelSlot::Llm, &src).expect("stage");
        let _promoted = models_promote(ModelSlot::Llm, &staged.hash, None).expect("promote");
        let out = dir.path().join("out/probe_llm.json");
        let report = models_probe_slot(ModelSlot::Llm, None, &out).expect("probe");
        assert!(report.pass());
        assert!(out.exists());
    }

    #[test]
    fn probe_hash_mode_does_not_mutate_manifest() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("fixtures/models_dummy/sae");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"dummy-sae-v1").expect("model");
        let staged = models_stage(ModelSlot::Sae, &src).expect("stage");

        let before = load_or_init_manifest()
            .expect("manifest before")
            .manifest_digest;
        let out = dir.path().join("out/probe_sae.json");
        let report = models_probe_slot(ModelSlot::Sae, Some(&staged.hash), &out).expect("probe");
        let after = load_or_init_manifest()
            .expect("manifest after")
            .manifest_digest;

        assert!(report.pass());
        assert_eq!(before, after);
    }

    #[test]
    fn probe_tampered_dummy_fails_envelope() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("fixtures/models_dummy/sae_bad");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"").expect("model");
        let staged = models_stage(ModelSlot::Sae, &src).expect("stage");

        let out = dir.path().join("out/probe_sae_bad.json");
        let report = models_probe_slot(ModelSlot::Sae, Some(&staged.hash), &out).expect("probe");
        assert!(!report.pass());
        assert!(report
            .envelope_checks
            .iter()
            .any(|c| c.code == "PROBE_MODEL_BYTES_NON_ZERO"
                && matches!(c.status, ProbeCheckStatus::Fail)));
    }

    #[test]
    fn world_real_tiny_fixture_hash_is_stable() {
        let fixture_hex = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/weights/world_real_tiny.safetensors.hex");
        let raw_hex = fs::read_to_string(&fixture_hex).expect("fixture hex");
        let bytes = hex::decode(raw_hex.trim()).expect("fixture bytes");
        let digest = hex::encode(Sha256::digest(&bytes));
        assert_eq!(
            digest,
            "73b51575099cb45efb4a3fc66e1daf31157476c2f6ec3a2d8a313452cad024c6"
        );
    }

    #[test]
    fn world_real_tiny_stage_probe_promote_probe_flow() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = materialize_world_real_tiny_fixture(dir.path());
        let staged = models_stage(ModelSlot::WorldJepa, &src).expect("stage");

        let manifest_before = load_or_init_manifest()
            .expect("manifest before")
            .manifest_digest;
        let staged_probe_out = dir.path().join("out/probe_world_staged.json");
        let staged_report =
            models_probe_slot(ModelSlot::WorldJepa, Some(&staged.hash), &staged_probe_out)
                .expect("staged probe");
        let manifest_after = load_or_init_manifest()
            .expect("manifest after")
            .manifest_digest;
        assert!(staged_report.pass());
        assert_eq!(manifest_before, manifest_after);

        let promote = models_promote(ModelSlot::WorldJepa, &staged.hash, Some(4)).expect("promote");
        assert_eq!(promote.slot, "world_jepa");
        assert_eq!(promote.to_hash, staged.hash);
        assert!(promote.old_manifest_digest_prefix.is_some());
        assert_eq!(promote.new_manifest_digest_prefix.len(), 12);

        let active_probe_out = dir.path().join("out/probe_world_active.json");
        let active_report =
            models_probe_slot(ModelSlot::WorldJepa, None, &active_probe_out).expect("active probe");
        assert!(active_report.pass());
        assert_eq!(active_report.mode, ProbeMode::Active);
        assert_eq!(
            active_report.model_hash_prefix.as_deref(),
            Some(&staged.hash[..PROBE_DIGEST_PREFIX_LEN])
        );

        let history_count = fs::read_dir("models/manifests/history")
            .expect("history")
            .filter_map(|entry| entry.ok())
            .count();
        assert!(history_count >= 1);
    }

    #[test]
    fn sae_real_tiny_fixture_hash_is_stable() {
        let fixture_hex = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/weights/sae_real_tiny.safetensors.hex");
        let raw_hex = fs::read_to_string(&fixture_hex).expect("fixture hex");
        let bytes = hex::decode(raw_hex.trim()).expect("fixture bytes");
        let digest = hex::encode(Sha256::digest(&bytes));
        assert_eq!(
            digest,
            "0f1ea81381690179efb5058ff06379423142265b2e6e80ca731ecd8ad8330c57"
        );
    }

    #[test]
    fn sae_real_tiny_stage_probe_promote_probe_flow() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = materialize_sae_real_tiny_fixture(dir.path());
        let staged = models_stage(ModelSlot::Sae, &src).expect("stage");

        let manifest_before = load_or_init_manifest()
            .expect("manifest before")
            .manifest_digest;
        let staged_probe_out = dir.path().join("out/probe_sae_staged.json");
        let staged_report =
            models_probe_slot(ModelSlot::Sae, Some(&staged.hash), &staged_probe_out)
                .expect("staged probe");
        let manifest_after = load_or_init_manifest()
            .expect("manifest after")
            .manifest_digest;
        assert!(staged_report.pass());
        assert_eq!(manifest_before, manifest_after);

        let promote = models_promote(ModelSlot::Sae, &staged.hash, Some(4)).expect("promote");
        assert_eq!(promote.slot, "sae");
        assert_eq!(promote.to_hash, staged.hash);

        let active_probe_out = dir.path().join("out/probe_sae_active.json");
        let active_report =
            models_probe_slot(ModelSlot::Sae, None, &active_probe_out).expect("active probe");
        assert!(active_report.pass());
        assert_eq!(active_report.mode, ProbeMode::Active);
        assert_eq!(
            active_report.model_hash_prefix.as_deref(),
            Some(&staged.hash[..PROBE_DIGEST_PREFIX_LEN])
        );
        #[cfg(feature = "backend-candle")]
        assert!(active_report.backend_id.starts_with("candle:sae:"));
    }

    #[test]
    fn sae_tampered_promoted_fixture_fails_verify() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = materialize_sae_real_tiny_fixture(dir.path());
        let staged = models_stage(ModelSlot::Sae, &src).expect("stage");
        let _promoted = models_promote(ModelSlot::Sae, &staged.hash, None).expect("promote");

        let promoted_model = PathBuf::from("models")
            .join("promoted")
            .join("sae")
            .join(&staged.hash)
            .join("model.safetensors");
        fs::write(promoted_model, b"tampered").expect("tamper");

        let verify = models_verify(Path::new(MANIFEST_PATH)).expect("verify");
        assert!(verify.manifest_present);
        assert!(!verify.files_verified);
    }

    #[test]
    fn active_check_denies_without_probe() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("fixtures/models_dummy/world");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"dummy-world-v1").expect("model");
        let staged = models_stage(ModelSlot::WorldJepa, &src).expect("stage");
        let _promoted = models_promote(ModelSlot::WorldJepa, &staged.hash, None).expect("promote");

        let denied = can_enable_active(
            ModelSlot::WorldJepa,
            &staged.hash,
            Path::new("."),
            false,
            false,
        )
        .expect_err("must deny");
        #[cfg(feature = "backend-burn")]
        assert!(matches!(
            denied.code,
            ActiveEnablementDeniedCode::ActiveDeniedBackendNotYetAllowed
        ));
        #[cfg(not(feature = "backend-burn"))]
        assert!(matches!(
            denied.code,
            ActiveEnablementDeniedCode::ActiveDeniedNoProbe
        ));
    }

    #[test]
    fn active_check_denies_sae_for_slot_stage() {
        let denied = can_enable_active(ModelSlot::Sae, "deadbeef", Path::new("."), false, false)
            .expect_err("must deny");
        assert!(matches!(
            denied.code,
            ActiveEnablementDeniedCode::ActiveNotEnabledForSlotStage
        ));
    }

    #[test]
    fn strict_mode_active_hash_missing_promoted_artifact_fails() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        let src = dir.path().join("fixtures/models_dummy/ssm");
        fs::create_dir_all(&src).expect("src");
        fs::write(src.join("model.safetensors"), b"dummy-ssm-v1").expect("model");
        let staged = models_stage(ModelSlot::Ssm, &src).expect("stage");
        let _promoted = models_promote(ModelSlot::Ssm, &staged.hash, None).expect("promote");
        let promoted_dir = PathBuf::from("models")
            .join("promoted")
            .join("ssm")
            .join(&staged.hash);
        fs::remove_dir_all(promoted_dir).expect("remove promoted");

        let denied = can_enable_active(ModelSlot::Ssm, &staged.hash, Path::new("."), true, false)
            .expect_err("must deny");
        assert!(matches!(
            denied.code,
            ActiveEnablementDeniedCode::ActiveDeniedHashMismatch
        ));
    }

    #[cfg(feature = "backend-burn")]
    #[test]
    fn active_check_denies_burn_world_backend() {
        let denied = can_enable_active(
            ModelSlot::WorldJepa,
            "deadbeef",
            Path::new("."),
            false,
            false,
        )
        .expect_err("must deny");
        assert!(matches!(
            denied.code,
            ActiveEnablementDeniedCode::ActiveDeniedBackendNotYetAllowed
        ));
    }

    #[test]
    fn detects_second_supported_real_slot_from_docs() {
        let slot = detect_second_real_slot_from_docs().expect("second slot");
        assert_eq!(slot, ModelSlot::Sae);
    }

    #[test]
    fn unknown_second_slot_configuration_fails_with_hint() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
        fs::create_dir_all("docs").expect("docs dir");
        fs::write(
            "docs/series_state_snapshot.md",
            "- Second supported slot in this stage: `invalid_slot`
",
        )
        .expect("series snapshot");
        let err = supported_real_slots(None).expect_err("must fail");
        assert!(err.to_string().contains("sae or ssm"));
    }

    #[test]
    fn shadow_ready_evidence_digest_is_stable() {
        let ev = ShadowReadyEvidenceV1 {
            slot_id: "world_jepa".to_string(),
            target_hash: "abc123".to_string(),
            manifest_digest_prefix: "m1".to_string(),
            latest_probe_report_digest_prefix: "p1".to_string(),
            latest_probe_status: ProbeReportStatus::Pass,
            latest_compare_window_digest_prefix: "c1".to_string(),
            compare_window_present: true,
            no_impact_verified: true,
            latest_drift_status: DriftStatusV1::Ok,
            shadow_ready: true,
            denial_reason_code: None,
            evidence_digest: "d1".to_string(),
        };
        let mut digest_source = Vec::new();
        digest_source.extend_from_slice(ev.slot_id.as_bytes());
        digest_source.extend_from_slice(ev.target_hash.as_bytes());
        digest_source.extend_from_slice(ev.manifest_digest_prefix.as_bytes());
        digest_source.extend_from_slice(ev.latest_probe_report_digest_prefix.as_bytes());
        digest_source.extend_from_slice(format!("{:?}", ev.latest_probe_status).as_bytes());
        digest_source.extend_from_slice(ev.latest_compare_window_digest_prefix.as_bytes());
        digest_source.extend_from_slice(b"1");
        digest_source.extend_from_slice(b"1");
        digest_source.extend_from_slice(format!("{:?}", ev.latest_drift_status).as_bytes());
        digest_source.extend_from_slice(b"1");
        let a = sha256_hex(&digest_source);
        let b = sha256_hex(&digest_source);
        assert_eq!(a, b);
    }

    #[test]
    fn models_shadow_ready_fails_without_required_evidence() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());

        fs::create_dir_all("docs").expect("docs dir");
        fs::write(
            "docs/series_state_snapshot.md",
            "- Second supported slot in this stage: `sae`\n",
        )
        .expect("series snapshot");

        let world_src = materialize_world_real_tiny_fixture(dir.path());
        let sae_src = materialize_sae_real_tiny_fixture(dir.path());
        let world_hash = models_stage(ModelSlot::WorldJepa, &world_src)
            .expect("stage world")
            .hash;
        let sae_hash = models_stage(ModelSlot::Sae, &sae_src)
            .expect("stage sae")
            .hash;
        models_promote(ModelSlot::WorldJepa, &world_hash, None).expect("promote world");
        models_promote(ModelSlot::Sae, &sae_hash, None).expect("promote sae");

        let out = PathBuf::from("out/shadow_ready_report.json");
        let report = models_shadow_ready(Path::new("."), None, &out).expect("shadow-ready report");
        assert!(matches!(report.overall_status, AggregatedStatusV1::Fail));
        assert_eq!(report.slots.len(), 2);
        assert!(report.slots.iter().all(|slot| !slot.shadow_ready));
        assert!(out.exists());
    }
    #[test]
    fn active_evidence_digest_is_stable() {
        let ev = ActiveEnablementEvidenceV1 {
            slot_id: "world_jepa".to_string(),
            target_hash: "abc123".to_string(),
            latest_probe_report_digest_prefix: "1111".to_string(),
            latest_probe_status: ProbeReportStatus::Pass,
            latest_compare_window_digest_prefix: "2222".to_string(),
            shadow_no_impact_verified: true,
            latest_drift_status: DriftStatusV1::Ok,
            evidence_window_ticks: 128,
            freshness_probe_age_ticks: 2,
            freshness_compare_age_ticks: 3,
            freshness_no_impact_age_ticks: 4,
            freshness_drift_status_age_ticks: 5,
            evidence_digest: "".to_string(),
        };
        let mut digest_source = Vec::new();
        digest_source.extend_from_slice(ev.slot_id.as_bytes());
        digest_source.extend_from_slice(ev.target_hash.as_bytes());
        digest_source.extend_from_slice(ev.latest_probe_report_digest_prefix.as_bytes());
        digest_source.extend_from_slice(format!("{:?}", ev.latest_probe_status).as_bytes());
        digest_source.extend_from_slice(ev.latest_compare_window_digest_prefix.as_bytes());
        digest_source.extend_from_slice(b"1");
        digest_source.extend_from_slice(format!("{:?}", ev.latest_drift_status).as_bytes());
        digest_source.extend_from_slice(ev.evidence_window_ticks.to_string().as_bytes());
        digest_source.extend_from_slice(ev.freshness_probe_age_ticks.to_string().as_bytes());
        digest_source.extend_from_slice(ev.freshness_compare_age_ticks.to_string().as_bytes());
        digest_source.extend_from_slice(ev.freshness_no_impact_age_ticks.to_string().as_bytes());
        digest_source.extend_from_slice(ev.freshness_drift_status_age_ticks.to_string().as_bytes());
        let a = sha256_hex(&digest_source);
        let b = sha256_hex(&digest_source);
        assert_eq!(a, b);
    }

    #[test]
    fn supported_real_slots_active_view_ordering_stable() {
        let mut slots = [
            SupportedRealSlotActiveViewEntryV1 {
                slot_id: "world_jepa".to_string(),
                target_hash: "b".to_string(),
                active_eligible: true,
                denial_reason_code: None,
                evidence_digest_prefix: Some("1".to_string()),
                freshness_probe_age_ticks: Some(1),
                freshness_compare_age_ticks: Some(1),
                freshness_no_impact_age_ticks: Some(1),
                freshness_drift_status_age_ticks: Some(1),
            },
            SupportedRealSlotActiveViewEntryV1 {
                slot_id: "sae".to_string(),
                target_hash: "a".to_string(),
                active_eligible: false,
                denial_reason_code: Some("ACTIVE_DENIED_NO_PROBE".to_string()),
                evidence_digest_prefix: None,
                freshness_probe_age_ticks: None,
                freshness_compare_age_ticks: None,
                freshness_no_impact_age_ticks: None,
                freshness_drift_status_age_ticks: None,
            },
        ];
        slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
        assert_eq!(slots[0].slot_id, "sae");
        assert_eq!(slots[1].slot_id, "world_jepa");
    }

    #[test]
    fn unified_eligibility_overall_status_is_deterministic() {
        let slots = vec![
            UnifiedEligibilityStatusV1 {
                slot_id: "sae".to_string(),
                target_hash_prefix: "a".to_string(),
                manifest_digest_prefix: "m".to_string(),
                probe_ready: true,
                shadow_ready: true,
                active_eligible: false,
                latest_probe_digest_prefix: "p1".to_string(),
                latest_shadow_evidence_digest_prefix: "s1".to_string(),
                latest_active_evidence_digest_prefix: "missing".to_string(),
                latest_drift_status: DriftStatusV1::Warn,
                burn_support_state: OptionalBackendSupportStateV1::NotConfigured,
                burn_parity_present: false,
                denial_reason_probe: None,
                denial_reason_shadow: None,
                denial_reason_active: Some("ACTIVE_DENIED_DRIFT_WARN".to_string()),
                remediation_codes: vec!["ACTIVE_DENIED_DRIFT_WARN".to_string()],
                canonical_remediation_codes: vec![],
                status_digest: "d1".to_string(),
            },
            UnifiedEligibilityStatusV1 {
                slot_id: "world_jepa".to_string(),
                target_hash_prefix: "b".to_string(),
                manifest_digest_prefix: "m".to_string(),
                probe_ready: true,
                shadow_ready: true,
                active_eligible: true,
                latest_probe_digest_prefix: "p2".to_string(),
                latest_shadow_evidence_digest_prefix: "s2".to_string(),
                latest_active_evidence_digest_prefix: "a2".to_string(),
                latest_drift_status: DriftStatusV1::Ok,
                burn_support_state: OptionalBackendSupportStateV1::Unsupported,
                burn_parity_present: false,
                denial_reason_probe: None,
                denial_reason_shadow: None,
                denial_reason_active: None,
                remediation_codes: vec![],
                canonical_remediation_codes: vec![],
                status_digest: "d2".to_string(),
            },
        ];
        assert!(matches!(
            derive_eligibility_overall_status(&slots),
            EligibilityOverallStatusV1::ActiveEligiblePartial
        ));
        assert_eq!(
            digest_shadow_generated_from(&slots),
            digest_shadow_generated_from(&slots)
        );
        assert_eq!(
            digest_active_generated_from(&slots),
            digest_active_generated_from(&slots)
        );
    }

    #[test]
    fn unified_eligibility_overall_status_none_and_probe_only() {
        let none_ready = vec![UnifiedEligibilityStatusV1 {
            slot_id: "world_jepa".to_string(),
            target_hash_prefix: "a".to_string(),
            manifest_digest_prefix: "m".to_string(),
            probe_ready: false,
            shadow_ready: false,
            active_eligible: false,
            latest_probe_digest_prefix: "missing".to_string(),
            latest_shadow_evidence_digest_prefix: "missing".to_string(),
            latest_active_evidence_digest_prefix: "missing".to_string(),
            latest_drift_status: DriftStatusV1::Unknown,
            burn_support_state: OptionalBackendSupportStateV1::NotConfigured,
            burn_parity_present: false,
            denial_reason_probe: Some("PROBE_REPORT_MISSING".to_string()),
            denial_reason_shadow: Some("SHADOW_READY_PROBE_REQUIRED".to_string()),
            denial_reason_active: Some("ActiveDeniedNoProbe".to_string()),
            remediation_codes: vec!["PROBE_REPORT_MISSING".to_string()],
            canonical_remediation_codes: vec![],
            status_digest: "d".to_string(),
        }];
        assert!(matches!(
            derive_eligibility_overall_status(&none_ready),
            EligibilityOverallStatusV1::NoneReady
        ));

        let mut probe_only = none_ready.clone();
        probe_only[0].probe_ready = true;
        assert!(matches!(
            derive_eligibility_overall_status(&probe_only),
            EligibilityOverallStatusV1::ProbeOnly
        ));
    }

    #[test]
    fn unified_eligibility_report_serialization_is_stable() {
        let report = AggregatedEligibilityReportV1 {
            schema_version: 1,
            overall_status: EligibilityOverallStatusV1::ProbeOnly,
            slots: vec![UnifiedEligibilityStatusV1 {
                slot_id: "world_jepa".to_string(),
                target_hash_prefix: "abc".to_string(),
                manifest_digest_prefix: "def".to_string(),
                probe_ready: true,
                shadow_ready: false,
                active_eligible: false,
                latest_probe_digest_prefix: "p1".to_string(),
                latest_shadow_evidence_digest_prefix: "missing".to_string(),
                latest_active_evidence_digest_prefix: "missing".to_string(),
                latest_drift_status: DriftStatusV1::Warn,
                burn_support_state: OptionalBackendSupportStateV1::NotConfigured,
                burn_parity_present: false,
                denial_reason_probe: None,
                denial_reason_shadow: Some("SHADOW_READY_PROBE_REQUIRED".to_string()),
                denial_reason_active: Some("ActiveDeniedNoProbe".to_string()),
                remediation_codes: vec!["SHADOW_READY_PROBE_REQUIRED".to_string()],
                canonical_remediation_codes: vec![],
                status_digest: "status1".to_string(),
            }],
            report_digest: "deadbeef".to_string(),
            policy_graph_digest_prefix: "aa11bb22".to_string(),
            generated_from: EligibilityGeneratedFromV1 {
                probe_report_digests: vec!["p1".to_string()],
                shadow_ready_report_digest: "s1".to_string(),
                active_evidence_report_digest: "a1".to_string(),
                second_slot_parity_report_digest: "missing".to_string(),
            },
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
    }

    #[cfg(test)]
    mod consistency_v4_tests {
        use super::*;

        #[test]
        fn denial_mapping_is_stable() {
            assert_eq!(
                map_denial_reason_to_code(Some("ACTIVE_DENIED_STALE_COMPARE")),
                Some(EvidenceDenialCodeV1::StaleCompare)
            );
            assert_eq!(
                map_denial_reason_to_code(Some("SHADOW_READY_DRIFT_SEVERE")),
                Some(EvidenceDenialCodeV1::DriftSevere)
            );
        }

        #[test]
        fn supported_slot_set_digest_is_deterministic() {
            let a = supported_real_slot_set_v1().expect("slot set");
            let b = supported_real_slot_set_v1().expect("slot set");
            assert_eq!(a.slots, b.slots);
            assert_eq!(a.set_digest, b.set_digest);
        }

        #[test]
        fn backend_support_state_order_is_stable() {
            let support = BackendSupportMatrixV1 {
                stub: BackendSupportStateV1::Supported,
                candle: BackendSupportStateV1::NotBuilt,
                burn: BackendSupportStateV1::NotConfigured,
            };
            let encoded = serde_json::to_string(&support).expect("encode");
            let stub = encoded.find("stub").expect("stub");
            let candle = encoded.find("candle").expect("candle");
            let burn = encoded.find("burn").expect("burn");
            assert!(stub < candle && candle < burn);
        }

        #[test]
        fn backend_snapshot_digest_is_stable() {
            let snapshot = BackendEvidenceSnapshotV1 {
                schema_version: 1,
                supported_slot_set_digest: "abc".to_string(),
                policy_graph_digest_prefix: "def".to_string(),
                manifest_digest_prefix: "123".to_string(),
                slots: vec![BackendEvidenceSlotSnapshotV1 {
                    slot_id: "world_jepa".to_string(),
                    target_hash_prefix: "aaaa".to_string(),
                    backend_support: BackendSupportMatrixV1 {
                        stub: BackendSupportStateV1::Supported,
                        candle: BackendSupportStateV1::Supported,
                        burn: BackendSupportStateV1::Unsupported,
                    },
                    evidence: BackendEvidenceSlotEvidenceV1 {
                        latest_probe_report_digest_prefix: "p".to_string(),
                        latest_compare_window_digest_prefix: "c".to_string(),
                        latest_shadow_ready_digest_prefix: "s".to_string(),
                        latest_active_evidence_digest_prefix: "a".to_string(),
                        latest_drift_status: DriftStatusV1::Ok,
                        freshness_probe_age_ticks: Some(1),
                        freshness_compare_age_ticks: Some(2),
                        freshness_no_impact_age_ticks: Some(3),
                        freshness_drift_status_age_ticks: Some(4),
                        hash_consistency_ok: true,
                    },
                    readiness: BackendEvidenceSlotReadinessV1 {
                        probe_ready: true,
                        shadow_ready: true,
                        active_eligible: false,
                    },
                    denials: BackendEvidenceSlotDenialsV1 {
                        probe: None,
                        shadow: None,
                        active: Some(EvidenceDenialCodeV1::DriftWarn),
                    },
                    remediation_codes: vec!["DRIFT_WARN".to_string()],
                    canonical_remediation_codes: vec![],
                }],
                snapshot_digest: "beef".to_string(),
            };
            let a = serde_json::to_vec(&snapshot).expect("a");
            let b = serde_json::to_vec(&snapshot).expect("b");
            assert_eq!(a, b);
        }
    }
}
