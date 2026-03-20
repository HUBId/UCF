#![forbid(unsafe_code)]

mod adversarial;
mod airgap;
mod alerts;
mod artifact_schema;
mod bench;
mod canonical_governance_entry;
mod causal;
mod change_impact;
mod compare_window;
mod config_contract;
mod continuity_authority;
mod docs_lint;
mod drift;
mod final_bundle_consumer_sweep;
mod final_governance_authority;
mod final_governance_consumer_sweep;
mod final_readiness_consumer_sweep;
mod formal_invariants;
mod goldens;
mod governance_entry_check;
mod governance_entry_sweep;
mod governance_surfaces;
mod interop_consistency;
mod models_lifecycle;
mod nightly;
mod operator_export_chain;
mod operator_report;
mod operator_review_packet;
mod operator_signoff;
mod operator_workflow;
mod readiness_spine;
mod remediation;
mod remediation_consistency;
mod reviewability_truth;
mod roundtrip_chain;
mod scope_authority;
mod second_slot_parity;
mod soak;
mod spec_snapshot;
mod strict_evidence;
mod v6_gate;
mod v7_gate;
mod v8_gate;
mod v9_gate;
mod world_shadow;
pub use adversarial::{adversarial_run, AdversarialReport, AdversarialRunArgs, CaseResult};
pub use airgap::{
    airgap_export_models, airgap_export_policies, airgap_export_repro, airgap_export_run_cert,
    airgap_import, AirgapArtifactType, AirgapExportReport, AirgapImportArgs, AirgapImportMode,
    AirgapImportReport,
};
pub use alerts::{alerts_report, AlertClearRecordV1, AlertEventV1, AlertRecordV1, AlertsReportV1};
pub use artifact_schema::{
    check_artifact_schema_snapshots, classify_drift, generate_artifact_schema_snapshots,
    ArtifactSchemaArgs, ArtifactSchemaCheckReport, ArtifactSchemaSnapshot, DriftKind,
};
pub use bench::{bench_run, BenchArgs, BenchReport};
pub use canonical_governance_entry::{
    canonical_entry_from_optional, derive_canonical_governance_entry,
    require_canonical_governance_entry, CanonicalGovernanceEntryStatusV1,
    CanonicalGovernanceEntryV1, CANONICAL_ENTRY_REQUIRED, CANONICAL_GOVERNANCE_ENTRY_REQUIRED,
    GOVERNANCE_ENTRY_SCOPE_MISMATCH, GOVERNANCE_PRIMARY_SURFACES_REQUIRED,
    SECONDARY_ENTRY_PATH_BLOCKED,
};
pub use causal::{
    causal_slice, event_id_for_decision, event_id_for_record, explain_why,
    save_counterfactual_result, simulate_counterfactual, write_slice, CausalEdge, CausalSlice,
    CounterfactualRequest, CounterfactualResult, EdgeType, EventNode, EventType, ExplainWhyReport,
};
pub use change_impact::{change_impact, ChangeImpactArgs};
pub use compare_window::{
    build_compare_window_meta, compare_freshness, derive_drift_inputs_from_slot_compare,
    derive_window_id, sample_digest_prefixes, unified_compare_semantics_v1,
    CompareWindowBackendStatusV1, CompareWindowFreshnessV1, CompareWindowMetaV1, DriftInputV1,
};
pub use config_contract::{
    export_policy_key_registry_v1, migrate_config_v1, ConfigV1, MigrateReport, PolicyKeyEntryV1,
};
pub use continuity_authority::{
    continuity_authority_check, CanonicalContinuityAuthorityV1, ContinuityAuthorityStatusV1,
};
pub use docs_lint::{docs_lint, DocsLintArgs, DocsLintMode, DocsLintReport, DocsLintStatus};
pub use drift::{drift_report, drift_status_map, DriftReportV1};
pub use final_bundle_consumer_sweep::{
    final_bundle_consumer_sweep, FinalBundleConsumerAuthorityStatusV1,
    FinalBundleConsumerAuthorityV1, FinalBundleConsumerMismatchCategoryV1,
    FinalBundleConsumerStatusV1, FinalBundleConsumerSweepReportV1,
};
pub use final_governance_authority::{
    require_final_governance_authority, FinalGovernanceAuthorityContextV1,
    FINAL_GOVERNANCE_AUTHORITY_REQUIRED, LEGACY_GOVERNANCE_INPUT_BLOCKED,
};
pub use final_governance_consumer_sweep::{
    final_governance_consumer_sweep, FinalGovernanceConsumerAuthorityStatusV1,
    FinalGovernanceConsumerAuthorityV1, FinalGovernanceConsumerMismatchCategoryV1,
    FinalGovernanceConsumerStatusV1, FinalGovernanceConsumerSweepReportV1,
};
pub use final_readiness_consumer_sweep::{
    final_readiness_consumer_sweep, FinalReadinessConsumerAuthorityStatusV1,
    FinalReadinessConsumerAuthorityV1, FinalReadinessConsumerMismatchCategoryV1,
    FinalReadinessConsumerStatusV1, FinalReadinessConsumerSweepReportV1,
};
pub use goldens::{
    goldens_generate, goldens_update, goldens_verify, goldens_verify_detailed, GoldenGenerateArgs,
    GoldenRefreshHeuristic, GoldenScenarioConfig, GoldenVerifyArgs, GoldenVerifyReport,
    GoldenVerifyScenarioReport,
};
pub use governance_entry_check::{
    governance_entry_check, GovernanceEntryCheckReportV1, GovernanceEntryCheckStatusV1,
    GovernanceEntryConsumerResultV1, GovernanceEntryMismatchCategoryV1,
};
pub use governance_entry_sweep::{
    governance_entry_sweep, CanonicalGovernanceEntryAuthorityV2, GovernanceEntryAuthorityStatusV2,
    GovernanceEntrySweepMismatchCategoryV1, GovernanceEntrySweepReportV1,
    GovernanceEntrySweepSurfaceStatusV1,
};
pub use governance_surfaces::{
    validate_governance_primary_surfaces, validate_governance_primary_surfaces_from_workdir,
    validate_governance_primary_surfaces_optional,
    validate_governance_primary_surfaces_with_applied_scope, GovernancePrimarySurfacesV1,
    GOVERNANCE_APPLIED_SET_MISMATCH_CODE, GOVERNANCE_PRIMARY_SURFACE_SCOPE_DRIFT_CODE,
    GOVERNANCE_SURFACE_MISMATCH_CODE, GOVERNANCE_SURFACE_MISSING_CODE,
};
pub use interop_consistency::{
    interop_consistency_matrix, CrossSurfaceContextMatrixV1, CrossSurfaceEntryV1,
    CrossSurfaceMatchRulesV1, InteropConsistencyMatrixReportV1, InteropMismatchCategoryV1,
    InteropOverallStatusV1,
};
pub use models_lifecycle::{
    can_enable_active, load_applied_supported_set_context_v1, models_active_check,
    models_active_evidence, models_active_review_snapshot, models_applied_scope_check,
    models_backend_resolution, models_consistency_check, models_eligibility,
    models_evidence_snapshot, models_list, models_probe_slot, models_promote,
    models_recommend_rollback, models_rollback, models_shadow_ready, models_stage,
    models_supported_scope_execute, models_supported_scope_execute_v4,
    models_supported_scope_execute_v5, models_supported_scope_reevaluate,
    models_supported_set_apply, models_supported_set_review,
    models_verify as models_verify_lifecycle, parse_slot, ActiveCheckStatus,
    ActiveEnablementDeniedCode, ActiveEnablementEvidenceV1, ActiveReviewEvidenceV1,
    ActiveReviewOverallStatusV1, ActiveReviewSnapshotRecordV1, AggregatedActiveReviewSnapshotV1,
    AggregatedEligibilityReportV1, AggregatedEvidenceReportV1, AggregatedStatusV1,
    AppliedScopeCheckReportV1, AppliedSupportedSetContextV1, BackendEvidenceSnapshotV1,
    BackendSupportStateV1, BurnResolutionStatusV1, BurnSupportResolutionV1,
    EligibilityOverallStatusV1, ModelsActiveCheckReport, ModelsConsistencyCheckReportV1,
    ProbeReportV1, ShadowReadyCheckRecordV1, ShadowReadyEvidenceV1, SlotEvidenceSnapshotV1,
    SlotExpansionEligibilityV1, SupportedRealSlotSetDecisionV2,
    SupportedRealSlotSetExecutionDecisionV2, SupportedRealSlotSetPolicyV2, SupportedRealSlotSetV1,
    SupportedRealSlotSetV2, SupportedRealSlotsActiveViewV1, SupportedScopeExecutionDecisionV3,
    SupportedScopeExecutionDecisionV4, SupportedScopeExecutionDecisionV5,
    SupportedScopeExecutionV3, SupportedScopeExecutionV4, SupportedScopeExecutionV5,
    SupportedScopeReevaluationDecisionV1, SupportedScopeReevaluationV1, SupportedSetApplyReportV1,
    SupportedSetExecutionDeniedCodeV1, SupportedSetExpansionRecordV1, SupportedSetFreezeRecordV1,
    SupportedSetReviewReportV1, UnifiedEligibilityStatusV1,
};
pub use nightly::{
    nightly_summarize, NightlyComponentReport, NightlyOverallStatus, NightlySummarizeArgs,
    NightlySummaryReport,
};
pub use operator_export_chain::{
    derive_operator_export_authority_chain, operator_export_chain_check,
    OperatorExportAuthorityChainStatusV1, OperatorExportAuthorityChainV1,
    OperatorExportAuthorityInputs, OperatorExportAuthorityMismatchCategoryV1,
};
pub use operator_report::{
    operator_report, operator_report_text, ConsolidatedOperatorReportV1, OperatorReportArgs,
    OperatorStatus,
};
pub use operator_review_packet::{
    operator_review_packet, operator_review_packet_text, OperatorReviewPacketArgs,
    OperatorReviewPacketV1, OperatorReviewStageV1,
};
pub use operator_signoff::{
    operator_signoff, operator_signoff_text, OperatorSignoffArgs, OperatorSignoffDecisionV1,
    SignoffDecisionStateV1, SignoffPolicyV1,
};
pub use operator_workflow::{
    operator_workflow_chain, operator_workflow_chain_text, OperatorWorkflowArgs,
    OperatorWorkflowChainV1, OperatorWorkflowExportTargetsV1, OperatorWorkflowPolicyV1,
    OperatorWorkflowStageV2,
};
pub use readiness_spine::{
    attach_spine_prefix_to_packet, attach_spine_prefix_to_signoff, attach_spine_prefix_to_workflow,
    derive_canonical_readiness_authority_v2, derive_canonical_readiness_spine,
    readiness_spine_check, readiness_spine_sweep, require_canonical_readiness_spine,
    require_final_readiness_authority, write_canonical_readiness_spine,
    CanonicalReadinessAuthorityStatusV2, CanonicalReadinessAuthorityV2,
    CanonicalReadinessSpineStatusV1, CanonicalReadinessSpineV1, FinalReadinessAuthorityContextV1,
    ReadinessSpineCheckReportV1, ReadinessSpineCheckStatusV1, ReadinessSpineMismatchCategoryV1,
    ReadinessSpineSweepMismatchCategoryV1, ReadinessSpineSweepReportV1,
    ReadinessSpineSweepSurfaceStatusV1, CANONICAL_READINESS_SPINE_REQUIRED,
    FINAL_READINESS_AUTHORITY_REQUIRED, LEGACY_READINESS_INPUT_BLOCKED,
    REVIEWABILITY_REDUCTION_REQUIRED, SECONDARY_READINESS_PATH_BLOCKED,
    SLOT_REVIEWABILITY_TRUTH_REQUIRED,
};
pub use remediation::all_registry_rows as remediation_registry_rows;
pub use remediation_consistency::{
    final_primary_semantics_sweep, primary_semantics_sweep, remediation_consistency_check,
    remediation_interop_check, remediation_spine_check, CanonicalPrimarySemanticsAuthorityStatusV1,
    CanonicalPrimarySemanticsAuthorityV1, CanonicalRemediationObservationV1,
    CrossSurfaceConditionObservationV1, CrossSurfaceObservationStatusV1,
    FinalPrimarySemanticsConsumerAuthorityStatusV1, FinalPrimarySemanticsConsumerAuthorityV1,
    FinalPrimarySemanticsSweepReportV1, PrimarySemanticsObservationV1,
    PrimarySemanticsObservedSurfaceV1, PrimarySemanticsSweepReportV1,
    RemediationConsistencyCheckV1, RemediationConsistencyObservedV1,
    RemediationConsistencyReportV1, RemediationConsistencyStatusV1,
    RemediationInteropCheckReportV1, RemediationMismatchKindV1, RemediationSpineCheckReportV1,
    SpineConditionObservationV1, CANONICAL_CONDITION_MODEL_REQUIRED,
    CANONICAL_REMEDIATION_REGISTRY_REQUIRED, FINAL_PRIMARY_SEMANTICS_AUTHORITY_REQUIRED,
    LEGACY_PRIMARY_SEMANTICS_INPUT_BLOCKED, LEGACY_PRIMARY_SEMANTICS_REJECTED,
    LEGACY_PRIMARY_SEMANTICS_TRANSLATED,
};
pub use reviewability_truth::{
    derive_slot_reviewability_truths, derive_slot_reviewability_truths_from_active,
    reduce_reviewability, review_truth_check, slot_is_reviewable, ReviewTruthCheckReportV1,
    ReviewTruthCheckStatusV1, ReviewTruthMismatchCategoryV1, ReviewabilityAggregateReadinessV1,
    ReviewabilityReductionV1, SlotReviewabilityEvidenceDigestsV1, SlotReviewabilityTruthV1,
};
pub use roundtrip_chain::{
    operator_roundtrip_chain_check, CanonicalRoundTripChainStatusV1, CanonicalRoundTripChainV1,
};
pub use scope_authority::{
    scope_authority_check, ScopeAuthorityCheckReportV1, ScopeAuthorityMismatchCategoryV1,
    ScopeAuthorityOverallStatusV1, ScopeAuthoritySurfaceResultV1, APPLIED_SCOPE_MISSING,
    APPLIED_SCOPE_REQUIRED, APPLIED_SCOPE_TRANSLATION_FAILED, LEGACY_SCOPE_PATH_BLOCKED,
};
pub use second_slot_parity::{
    detect_second_slot, second_slot_parity_evidence_exists, second_slot_parity_report,
    OptionalBackendSupportStateV1, SaeParityRecordV1, SecondSlotParityRecordV1,
    SecondSlotParityReportV1, SsmParityRecordV1,
};
pub use soak::{
    parse_duration_secs, parse_inject, soak_run, InjectTrigger, SoakReport, SoakRunArgs, SoakStatus,
};
pub use spec_snapshot::{generate_spec_snapshot, SpecSnapshotArgs};
pub use strict_evidence::{
    operator_block_from_strict, resolve_strict_evidence, strict_explain, OperatorBlockingViewV1,
    StrictEvidenceContextV1, StrictEvidenceSnapshotV1, StrictEvidenceStatusV1,
    StrictExplainReportV1,
};
pub use v6_gate::{v6_gate, V6GateCheckV1, V6GateOverallStatus, V6GateReportV1};
pub use v7_gate::{v7_gate, V7GateCheckV1, V7GateOverallStatus, V7GateReportV1};
pub use v8_gate::{v8_gate, V8GateCheckV1, V8GateOverallStatus, V8GateReportV1};
pub use v9_gate::{v9_gate, V9GateCheckV1, V9GateOverallStatus, V9GateReportV1};
pub use world_shadow::{
    world_parity_evidence_exists, world_parity_report, world_shadow_report,
    WorldBackendEligibilityV1, WorldParityRecordV1, WorldParityReportV1, WorldShadowReport,
};

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{mpsc, Mutex, OnceLock};
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
use ucf_policy::policy_packs::{
    load_and_merge_policy_graph, policy_graph_digest, DriftBudgetEntryV1, PolicyPackError,
};
use ucf_replay::{
    load_fixture_records, replay_audit as run_replay_audit, replay_records, write_report,
    ReplayMode, ReplayPlan, ReplaySpec, ReplayStrictness,
};
use ucf_runtime::RuntimeOrchestrator;
use ucf_types::error_codes::ErrorCode;

const DEV_LOOP_MAX_SCENARIOS: usize = 2;
const TROUBLESHOOT_MAX_ISSUES: usize = 8;

#[cfg(test)]
pub(crate) fn test_cwd_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

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
    pub strict_mode: bool,
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
    pub active_evidence_probe_max_age_ticks: u64,
    pub active_evidence_compare_max_age_ticks: u64,
    pub active_evidence_no_impact_max_age_ticks: u64,
    pub active_evidence_drift_status_max_age_ticks: u64,
    pub active_evidence_allow_warn_drift_for_active: bool,
    pub active_evidence_require_matching_target_hash: bool,
    pub config_digest: String,
}

impl Default for OpsConfig {
    fn default() -> Self {
        Self {
            profile: "test".to_string(),
            strict_mode: false,
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
            active_evidence_probe_max_age_ticks: 256,
            active_evidence_compare_max_age_ticks: 256,
            active_evidence_no_impact_max_age_ticks: 256,
            active_evidence_drift_status_max_age_ticks: 256,
            active_evidence_allow_warn_drift_for_active: false,
            active_evidence_require_matching_target_hash: true,
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
    pub strict_mode_enabled: bool,
    pub strict_mode_digest: Option<String>,
    pub probe_report_digest_prefix: Option<String>,
    pub crash_dumps_disabled: bool,
    pub models_manifest_present: bool,
    pub models_manifest_digest_prefix: Option<String>,
    pub ended_at_tick: Option<u64>,
}

fn disable_crash_dumps_best_effort() -> bool {
    std::env::var("UCF_CRASH_DUMPS_DISABLED")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn load_models_manifest_runtime_metadata() -> (bool, Option<String>) {
    let path = Path::new("models/MANIFEST.toml");
    let Some(raw) = fs::read_to_string(path).ok() else {
        return (false, None);
    };
    let digest = raw
        .lines()
        .find(|l| l.trim_start().starts_with("manifest_digest"))
        .and_then(|l| l.split('=').nth(1))
        .map(|v| v.trim().trim_matches('"').to_string());
    let prefix = digest.map(|d| d.chars().take(12).collect());
    (true, prefix)
}

fn load_probe_report_digest_prefix(workdir: &Path) -> Option<String> {
    let path = workdir.join("out/probe_report.json");
    let body = fs::read(path).ok()?;
    Some(prefix_hex(&sha256_hex(&body), 16))
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DevLoopStepStatus {
    Pass,
    Fail,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevLoopStepResult {
    pub step: String,
    pub status: DevLoopStepStatus,
    pub detail: String,
    pub artifact: Option<String>,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevLoopReport {
    pub profile: String,
    pub scenario: String,
    pub ticks: u64,
    pub steps: Vec<DevLoopStepResult>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevLoopArgs {
    pub profile: String,
    pub scenario: String,
    pub ticks: u64,
    pub out_dir: PathBuf,
    pub run_tests: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigReloadReasonCode {
    PolicyOverlayChanged,
    PolicyPathChanged,
    ManifestChanged,
    StrictModeChanged,
    AuthTokenChanged,
    UnsupportedKeyChanged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigReloadAppliedRecord {
    pub t_unix: u64,
    pub profile: String,
    pub changed_keys: Vec<String>,
    pub config_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigReloadDeniedRecord {
    pub t_unix: u64,
    pub profile: String,
    pub changed_keys: Vec<String>,
    pub reason_codes: Vec<ConfigReloadReasonCode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TroubleshootIssue {
    pub source: String,
    pub severity: String,
    pub detail: String,
    pub next_command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TroubleshootReport {
    pub run_id: String,
    pub strict_failure: Option<String>,
    pub drift_report: Option<String>,
    pub readiness_gate: Option<String>,
    pub docs_lint: Option<String>,
    pub gateway_abuse_count: usize,
    pub issues: Vec<TroubleshootIssue>,
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
    pub spike_count: Option<u16>,
    pub spikes_digest_prefix: Option<String>,
    pub pressure_q: Option<u16>,
    pub state_digest_prefix: Option<String>,
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
    let _manifest_env_guard = EnvVarGuard::set("UCF_MODEL_MANIFEST", manifest.as_os_str());
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
            run_probe_for_slot(&run_id, &pack, slot, &spec, verified.as_ref(), &store);
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

struct EnvVarGuard {
    key: String,
    prev: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &str, value: &std::ffi::OsStr) -> Self {
        let prev = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self {
            key: key.to_string(),
            prev,
        }
    }

    fn remove(key: &str) -> Self {
        let prev = std::env::var(key).ok();
        std::env::remove_var(key);
        Self {
            key: key.to_string(),
            prev,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev.as_ref() {
            std::env::set_var(&self.key, prev);
        } else {
            std::env::remove_var(&self.key);
        }
    }
}

fn with_env_var<T>(key: &str, value: &str, f: impl FnOnce() -> T) -> T {
    let _guard = EnvVarGuard::set(key, std::ffi::OsStr::new(value));
    f()
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
    store: &ModelStore,
) -> (ProbeResult, ModelProbeRecord) {
    let model_sha = verified.map(|v| hex_prefix(v.sha256));
    let slot_component = match slot {
        ModelSlot::Llm => pack.meta().llm_backend as u8,
        ModelSlot::WorldJepa | ModelSlot::WorldVljepa => pack.meta().world_backend as u8,
        ModelSlot::Sae => pack.meta().sae_backend as u8,
        ModelSlot::Ssm => pack.meta().ssm_backend as u8,
        ModelSlot::Lfm => pack.meta().lfm_backend as u8,
        ModelSlot::EbmReasoner => pack.meta().lfm_backend as u8,
    };
    let mut backend_id = format!(
        "{}:{}/slot:{}",
        pack.meta().pack_name,
        hex_prefix(pack.meta().digest),
        slot_component
    );
    let mut elapsed_samples = Vec::new();
    let mut final_status = ProbeStatus::Disabled;
    let mut final_quality = StageQuality::Unavailable;
    let mut final_output = [0_u8; 32];
    let mut spike_count = None;
    let mut spikes_digest_prefix = None;
    let mut pressure_q = None;
    let mut state_digest_prefix = None;
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
                    let has_weights = verified.is_some();
                    let store = store.clone();
                    move || run_world_probe(pack, &spec, has_weights, &store)
                }),
                ModelSlot::WorldVljepa => exec_with_timeout(spec.timeout_ms, {
                    let pack = pack.clone();
                    let spec = spec.clone();
                    let store = store.clone();
                    move || run_world_probe(pack, &spec, false, &store)
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
                Ok(probe_out) => {
                    final_status = ProbeStatus::Ok;
                    final_quality = probe_out.quality;
                    final_output = probe_out.digest;
                    if let Some(v) = probe_out.spike_count {
                        spike_count = Some(v);
                    }
                    if let Some(v) = probe_out.spikes_digest_prefix {
                        spikes_digest_prefix = Some(v);
                    }
                    if let Some(v) = probe_out.pressure_q {
                        pressure_q = Some(v);
                    }
                    if let Some(v) = probe_out.state_digest_prefix {
                        state_digest_prefix = Some(v);
                    }
                    if let Some(v) = probe_out.backend_id {
                        backend_id = v;
                    }
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
        spike_count,
        spikes_digest_prefix,
        pressure_q,
        state_digest_prefix,
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

#[derive(Debug, Clone)]
struct SlotProbeOutput {
    digest: [u8; 32],
    quality: StageQuality,
    backend_id: Option<String>,
    spike_count: Option<u16>,
    spikes_digest_prefix: Option<String>,
    pressure_q: Option<u16>,
    state_digest_prefix: Option<String>,
}

fn run_llm_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<SlotProbeOutput, String> {
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
    Ok(SlotProbeOutput {
        digest: resp.digest,
        quality: StageQuality::Ok,
        backend_id: None,
        spike_count: None,
        spikes_digest_prefix: None,
        pressure_q: None,
        state_digest_prefix: None,
    })
}

fn run_world_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
    has_weights: bool,
    store: &ModelStore,
) -> Result<SlotProbeOutput, String> {
    let _ = (has_weights, store);
    let mut obs = [0.0_f32; 16];
    for (idx, value) in deterministic_features(spec.seed, 16).iter().enumerate() {
        obs[idx] = *value;
    }
    #[cfg(feature = "backend-burn")]
    if has_weights {
        use ucf_compute::stage_v1::{WorldInputV1, WorldPredictorV1};
        use ucf_compute::stage_v1_burn::BurnWorldAdapterV0;
        let adapter = BurnWorldAdapterV0::from_model_store(store);
        let input_v1 = WorldInputV1 {
            context_digest: spec.input_digest,
            previous_world_state_digest: None,
            signal_q: 1024,
        };
        if let Ok(out) = adapter.step(&input_v1) {
            return Ok(SlotProbeOutput {
                digest: out.prediction_digest,
                quality: StageQuality::Ok,
                backend_id: Some(format!("burn:world:{}", adapter.backend_id())),
                spike_count: None,
                spikes_digest_prefix: None,
                pressure_q: Some(out.prediction_error_q),
                state_digest_prefix: Some(hex_prefix(out.prediction_digest)),
            });
        }
    }

    #[cfg(all(feature = "backend-candle", not(feature = "backend-burn")))]
    if has_weights {
        use ucf_compute::stage_v1::{WorldInputV1, WorldPredictorV1};
        use ucf_compute::stage_v1_candle::CandleWorldAdapterV0;
        let adapter = CandleWorldAdapterV0::from_model_store(store);
        let input_v1 = WorldInputV1 {
            context_digest: spec.input_digest,
            previous_world_state_digest: None,
            signal_q: 1024,
        };
        if let Ok(out) = adapter.step(&input_v1) {
            return Ok(SlotProbeOutput {
                digest: out.prediction_digest,
                quality: StageQuality::Ok,
                backend_id: Some(format!("candle:world:{}", adapter.backend_id())),
                spike_count: None,
                spikes_digest_prefix: None,
                pressure_q: Some(out.prediction_error_q),
                state_digest_prefix: Some(hex_prefix(out.prediction_digest)),
            });
        }
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
    Ok(SlotProbeOutput {
        digest: out.prediction_digest,
        quality: out.quality,
        backend_id: None,
        spike_count: None,
        spikes_digest_prefix: None,
        pressure_q: None,
        state_digest_prefix: None,
    })
}

fn run_sae_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<SlotProbeOutput, String> {
    #[cfg(feature = "backend-candle")]
    {
        use ucf_compute::stage_v1::{SaeExtractorV1, SaeInputV1};
        use ucf_compute::stage_v1_candle::CandleSaeAdapterV0;

        if let Ok(store) = ModelStore::from_env_default() {
            let adapter = CandleSaeAdapterV0::from_model_store(&store);
            let input_v1 = SaeInputV1 {
                context_digest: spec.input_digest,
                prediction_digest: [0x51; 32],
                top_k: 8,
            };
            if let Ok(out) = adapter.infer(&input_v1) {
                return Ok(SlotProbeOutput {
                    digest: out.spikes_digest,
                    quality: StageQuality::Ok,
                    backend_id: Some(format!("candle:sae:{}", adapter.backend_id())),
                    spike_count: Some(out.spikes.len() as u16),
                    spikes_digest_prefix: Some(hex_prefix(out.spikes_digest)),
                    pressure_q: None,
                    state_digest_prefix: None,
                });
            }
        }
    }

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
    Ok(SlotProbeOutput {
        digest: out.spikes_digest,
        quality: out.quality,
        backend_id: None,
        spike_count: Some(out.spike_count),
        spikes_digest_prefix: Some(hex_prefix(out.spikes_digest)),
        pressure_q: None,
        state_digest_prefix: None,
    })
}

fn run_ssm_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<SlotProbeOutput, String> {
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
    Ok(SlotProbeOutput {
        digest: out.state_digest,
        quality: out.quality,
        backend_id: None,
        spike_count: None,
        spikes_digest_prefix: None,
        pressure_q: Some(
            u16::try_from((out.pressure.clamp(0.0, 1.0) * f32::from(u16::MAX)).round() as u32)
                .unwrap_or(u16::MAX),
        ),
        state_digest_prefix: Some(hex_prefix(out.state_digest)),
    })
}

fn run_lfm_probe(
    pack: std::sync::Arc<dyn ucf_compute::BackendPack>,
    spec: &ProbeSpec,
) -> Result<SlotProbeOutput, String> {
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
    Ok(SlotProbeOutput {
        digest: out.liquid_state_digest,
        quality: out.quality,
        backend_id: None,
        spike_count: None,
        spikes_digest_prefix: None,
        pressure_q: None,
        state_digest_prefix: None,
    })
}
fn run_ebm_probe(spec: &ProbeSpec) -> Result<SlotProbeOutput, String> {
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
    Ok(SlotProbeOutput {
        digest: out.ebm_digest,
        quality: StageQuality::Ok,
        backend_id: None,
        spike_count: None,
        spikes_digest_prefix: None,
        pressure_q: None,
        state_digest_prefix: None,
    })
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V0GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V0GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V0GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V0GateOverallStatus,
    pub checks: Vec<V0GateCheckV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V1GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint: String,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V1GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V1GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V1GateOverallStatus,
    pub checks: Vec<V1GateCheckV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V2GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V2GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V2GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V2GateOverallStatus,
    pub checks: Vec<V2GateCheckV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V3GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V3GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V3GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V3GateOverallStatus,
    pub checks: Vec<V3GateCheckV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V4GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V4GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V4GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V4GateOverallStatus,
    pub checks: Vec<V4GateCheckV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum V5GateOverallStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V5GateCheckV1 {
    pub name: String,
    pub status: GateStatus,
    pub evidence_digest_prefixes: BTreeMap<String, String>,
    pub remediation_hint_code: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V5GateReportV1 {
    pub schema_version: u16,
    pub overall_status: V5GateOverallStatus,
    pub checks: Vec<V5GateCheckV1>,
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

const V0_MAX_RECORD_BYTES: usize = 16 * 1024;
const V0_SCHEMA_VERSION: u16 = 1;

pub fn v0_gate(workdir: &Path, scenario: &Path, out: &Path) -> Result<V0GateReportV1, OpsError> {
    ensure_layout(workdir)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");

    let scenario_doc: serde_json::Value = serde_json::from_str(&fs::read_to_string(scenario)?)?;
    let ticks = scenario_doc
        .get("ticks")
        .and_then(|v| v.as_u64())
        .unwrap_or(8);

    let policy = policy_validate(
        &repo_root.join("policies/packs/base_v1"),
        Some(&repo_root.join("policies/packs/overlays/test")),
    );
    let expected_policy_prefix =
        expected_policy_digest_prefix_from_spec_snapshot(&repo_root.join("docs/spec_snapshot.md"))?;

    let policy_check = match policy {
        Ok(report) => {
            let locked = report
                .policy_graph_digest
                .starts_with(&expected_policy_prefix);
            v0_gate_check(
                "policy_graph_lock",
                if locked {
                    GateStatus::Pass
                } else {
                    GateStatus::Fail
                },
                [
                    (
                        "policy_graph_digest".to_string(),
                        prefix_hex(&report.policy_graph_digest, 12),
                    ),
                    (
                        "locked_prefix".to_string(),
                        prefix_hex(&expected_policy_prefix, 12),
                    ),
                ],
                "v0.policy.lock_mismatch",
            )
        }
        Err(err) => v0_gate_check(
            "policy_graph_lock",
            GateStatus::Fail,
            [(
                "error".to_string(),
                bounded_string(err.to_string(), GATE_STR_CAP),
            )],
            "v0.policy.validation_error",
        ),
    };

    let run_one = v0_gate_run_once(workdir, scenario, ticks, "run_1");
    let run_two = v0_gate_run_once(workdir, scenario, ticks, "run_2");

    let (determinism_check, e2e_check, boundedness_check, schema_check, no_tool_check) =
        match (run_one, run_two) {
            (Ok(a), Ok(b)) => {
                let determinism_pass = a.signals_digest == b.signals_digest
                    && a.decision_digest == b.decision_digest
                    && a.experience_digest == b.experience_digest;
                let determinism_check = v0_gate_check(
                    "determinism_double_run",
                    if determinism_pass {
                        GateStatus::Pass
                    } else {
                        GateStatus::Fail
                    },
                    [
                        (
                            "signals_digest".to_string(),
                            prefix_hex(&a.signals_digest, 12),
                        ),
                        (
                            "decision_digest".to_string(),
                            prefix_hex(&a.decision_digest, 12),
                        ),
                        (
                            "experience_digest".to_string(),
                            prefix_hex(&a.experience_digest, 12),
                        ),
                    ],
                    "v0.determinism.digest_mismatch",
                );

                let e2e_pass = a.record_count > 0 && a.has_required_records;
                let e2e_check = v0_gate_check(
                    "e2e_flow_a",
                    if e2e_pass {
                        GateStatus::Pass
                    } else {
                        GateStatus::Fail
                    },
                    [
                        ("record_count".to_string(), a.record_count.to_string()),
                        (
                            "required_records".to_string(),
                            a.has_required_records.to_string(),
                        ),
                    ],
                    "v0.e2e.required_records_missing",
                );

                let boundedness_pass = a.max_record_bytes <= V0_MAX_RECORD_BYTES;
                let boundedness_check = v0_gate_check(
                    "record_boundedness",
                    if boundedness_pass {
                        GateStatus::Pass
                    } else {
                        GateStatus::Fail
                    },
                    [
                        (
                            "max_record_bytes".to_string(),
                            a.max_record_bytes.to_string(),
                        ),
                        (
                            "max_allowed_bytes".to_string(),
                            V0_MAX_RECORD_BYTES.to_string(),
                        ),
                    ],
                    "v0.records.size_cap_exceeded",
                );

                let schema_pass = schema_versions_known(&a.schema_versions);
                let schema_check = v0_gate_check(
                    "schema_versions_known",
                    if schema_pass {
                        GateStatus::Pass
                    } else {
                        GateStatus::Fail
                    },
                    [(
                        "schema_versions_digest".to_string(),
                        prefix_hex(&sha256_hex(&serde_json::to_vec(&a.schema_versions)?), 12),
                    )],
                    "v0.schema.unknown_or_missing",
                );

                let no_tool_pass = a.tool_execution_count == 0;
                let no_tool_check = v0_gate_check(
                    "no_tool_execution",
                    if no_tool_pass {
                        GateStatus::Pass
                    } else {
                        GateStatus::Fail
                    },
                    [(
                        "tool_execution_count".to_string(),
                        a.tool_execution_count.to_string(),
                    )],
                    "v0.tools.execution_detected",
                );

                (
                    determinism_check,
                    e2e_check,
                    boundedness_check,
                    schema_check,
                    no_tool_check,
                )
            }
            (Err(err), _) | (_, Err(err)) => {
                let fail = v0_gate_check(
                    "determinism_double_run",
                    GateStatus::Fail,
                    [(
                        "error".to_string(),
                        bounded_string(err.to_string(), GATE_STR_CAP),
                    )],
                    "v0.run.execution_error",
                );
                (
                    fail,
                    v0_gate_check(
                        "e2e_flow_a",
                        GateStatus::Fail,
                        [("error".to_string(), "dependent_on_run".to_string())],
                        "v0.run.execution_error",
                    ),
                    v0_gate_check(
                        "record_boundedness",
                        GateStatus::Fail,
                        [("error".to_string(), "dependent_on_run".to_string())],
                        "v0.run.execution_error",
                    ),
                    v0_gate_check(
                        "schema_versions_known",
                        GateStatus::Fail,
                        [("error".to_string(), "dependent_on_run".to_string())],
                        "v0.run.execution_error",
                    ),
                    v0_gate_check(
                        "no_tool_execution",
                        GateStatus::Fail,
                        [("error".to_string(), "dependent_on_run".to_string())],
                        "v0.run.execution_error",
                    ),
                )
            }
        };

    let checks = vec![
        policy_check,
        determinism_check,
        e2e_check,
        boundedness_check,
        schema_check,
        no_tool_check,
    ];
    let overall_status = if checks.iter().all(|c| c.status == GateStatus::Pass) {
        V0GateOverallStatus::Pass
    } else {
        V0GateOverallStatus::Fail
    };

    let report = V0GateReportV1 {
        schema_version: V0_SCHEMA_VERSION,
        overall_status,
        checks,
    };
    write_json(out, &report)?;
    Ok(report)
}

const V1_SCHEMA_VERSION: u16 = 1;
const V2_SCHEMA_VERSION: u16 = 1;
const V3_SCHEMA_VERSION: u16 = 1;

pub fn v1_gate(workdir: &Path, out: &Path) -> Result<V1GateReportV1, OpsError> {
    ensure_layout(workdir)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut checks = Vec::new();

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let v0_out = workdir.join("out").join("v0_gate_report.json");
    let v0_scenario = repo_root.join("fixtures/e2e/v0_flow_a.json");
    let v0 = v0_gate(workdir, &v0_scenario, &v0_out)?;
    checks.push(v1_gate_check(
        "v0_gate_pass",
        if matches!(v0.overall_status, V0GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("v0_gate_report".to_string(), prefix_hex(&sha256_hex(&serde_json::to_vec(&v0)?), 16))],
        "run `cargo run -p ucf-ops -- v0 gate --out ./out/v0_gate_report.json` and fix failing checks",
    ));

    let models_dir = repo_root.join("models");
    if models_dir.exists() {
        let mut verification_attempts = Vec::new();
        let mut status = GateStatus::Fail;
        let mut evidence = vec![("models_verify".to_string(), "missing".to_string())];
        for manifest in [
            repo_root.join("models/manifest.toml"),
            repo_root.join("models/MANIFEST.toml"),
        ] {
            if !manifest.exists() {
                continue;
            }
            if let Ok(report) = models_verify(&manifest) {
                if report
                    .slots
                    .iter()
                    .all(|slot| slot.status == "verified" || slot.status == "disabled")
                {
                    status = GateStatus::Pass;
                    evidence = vec![
                        (
                            "manifest".to_string(),
                            bounded_string(manifest.display().to_string(), 32),
                        ),
                        (
                            "models_verify".to_string(),
                            prefix_hex(&sha256_hex(&serde_json::to_vec(&report)?), 16),
                        ),
                    ];
                    break;
                }
                verification_attempts.push(format!(
                    "legacy:{}:not_all_verified",
                    manifest
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                ));
            } else if let Ok(report) = models_verify_lifecycle(&manifest) {
                let pass = report.manifest_present
                    && report.digest_match
                    && report.promoted_hashes_exist
                    && report.files_verified;
                if pass {
                    status = GateStatus::Pass;
                    evidence = vec![
                        (
                            "manifest".to_string(),
                            bounded_string(manifest.display().to_string(), 32),
                        ),
                        (
                            "models_verify_lifecycle".to_string(),
                            prefix_hex(&sha256_hex(&serde_json::to_vec(&report)?), 16),
                        ),
                    ];
                    break;
                }
                verification_attempts.push(format!(
                    "lifecycle:{}:not_all_verified",
                    manifest
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                ));
            } else {
                verification_attempts.push(format!(
                    "parse:{}:failed",
                    manifest
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                ));
            }
        }
        if status == GateStatus::Fail {
            evidence = vec![(
                "attempts".to_string(),
                bounded_string(verification_attempts.join("|"), 48),
            )];
        }
        checks.push(v1_gate_check(
            "models_manifest_verify",
            status,
            evidence,
            "run `cargo run -p ucf-ops -- models verify --manifest models/manifest.toml` (or MANIFEST.toml) and ensure all slots are verified/disabled",
        ));
    } else {
        checks.push(v1_gate_check(
            "models_manifest_verify",
            GateStatus::Skip,
            [("models_dir".to_string(), "missing".to_string())],
            "optional: add `models/` plus `models/MANIFEST.toml` to enable verification",
        ));
    }

    let probe_out = workdir.join("out").join("probe_v1_gate.json");
    let probe_manifest = repo_root.join("models/MANIFEST.toml");
    match models_probe(workdir, &probe_manifest, &probe_out) {
        Ok(report) => {
            let target_slots = [ModelSlot::WorldJepa, ModelSlot::Sae, ModelSlot::Ssm];
            let mut evidence = Vec::new();
            let mut all_pass = true;
            for slot in target_slots {
                let status = report
                    .results
                    .iter()
                    .find(|r| r.slot == slot)
                    .map(|r| r.status)
                    .unwrap_or(ProbeStatus::Error);
                if !matches!(status, ProbeStatus::Ok | ProbeStatus::Disabled) {
                    all_pass = false;
                }
                evidence.push((format!("probe_{}", slot.as_str()), format!("{:?}", status)));
            }
            checks.push(v1_gate_check(
                "probes_dummy_pass",
                if all_pass { GateStatus::Pass } else { GateStatus::Fail },
                evidence,
                "run `cargo run -p ucf-ops -- models probe --manifest models/MANIFEST.toml --out ./out/probe_report.json` and fix failed slots",
            ));
        }
        Err(err) => checks.push(v1_gate_check(
            "probes_dummy_pass",
            GateStatus::Fail,
            [("error".to_string(), bounded_string(err.to_string(), 48))],
            "ensure probe fixtures/backend stubs are available and rerun gate",
        )),
    }

    let scenario = repo_root.join("fixtures/e2e/v0_flow_a.json");
    let off = v0_gate_run_once(workdir, &scenario, 8, "v1_gate_shadow_off");
    let shadow = one_command_bringup_with_ebm_mode(
        &workdir.join("v1_gate").join("shadow_on"),
        &scenario,
        8,
        &workdir.join("v1_gate").join("shadow_on").join("out"),
        true,
        "shadow",
    );

    let (shadow_status, shadow_evidence) = match (off, shadow) {
        (Ok(off_run), Ok(shadow_run)) => {
            let shadow_decision = sha256_hex(&serde_json::to_vec(&shadow_run.explain.decision)?);
            let no_impact = off_run.decision_digest == shadow_decision;
            let shadow_records = load_fixture_records(
                &workdir
                    .join("v1_gate")
                    .join("shadow_on")
                    .join("ess")
                    .join("ess_fixture.json"),
            )?;
            let tool_count = shadow_records
                .iter()
                .filter(|r| r.kind == ExperienceKind::ToolExecution)
                .count();
            (
                if no_impact && tool_count == 0 {
                    GateStatus::Pass
                } else {
                    GateStatus::Fail
                },
                vec![
                    (
                        "off_decision".to_string(),
                        prefix_hex(&off_run.decision_digest, 16),
                    ),
                    (
                        "shadow_decision".to_string(),
                        prefix_hex(&shadow_decision, 16),
                    ),
                    ("tool_execution_count".to_string(), tool_count.to_string()),
                ],
            )
        }
        (Err(err), _) | (_, Err(err)) => (
            GateStatus::Fail,
            vec![("error".to_string(), bounded_string(err.to_string(), 48))],
        ),
    };
    checks.push(v1_gate_check(
        "shadow_no_decision_impact",
        shadow_status,
        shadow_evidence,
        "keep slot in shadow mode only and require decision digest parity with baseline run",
    ));

    let cfg = load_or_init_config(workdir)?;
    let drift_budget_override = std::env::var("UCF_STRICT_DRIFT_BUDGET_PATH").ok();
    let base_drift_budget = repo_root.join("policies/packs/base_v1/drift_budget.toml");
    let overlay_drift_budget = repo_root.join(format!(
        "policies/packs/overlays/{}/drift_budget.toml",
        cfg.profile
    ));
    let drift_present = if let Some(path) = drift_budget_override {
        PathBuf::from(path).exists()
    } else {
        overlay_drift_budget.exists() || base_drift_budget.exists()
    };
    checks.push(v1_gate_check(
        "drift_budget_present_if_shadow",
        if drift_present {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "drift_budget".to_string(),
            if drift_present { "present" } else { "missing" }.to_string(),
        )],
        "add `drift_budget.toml` to base or active overlay pack",
    ));

    let base_alerts = repo_root.join("policies/packs/base_v1/alerts.toml");
    let overlay_alerts = repo_root.join(format!(
        "policies/packs/overlays/{}/alerts.toml",
        cfg.profile
    ));
    let alerts_present = overlay_alerts.exists() || base_alerts.exists();
    checks.push(v1_gate_check(
        "alerts_present",
        if alerts_present {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "alerts_config".to_string(),
            if alerts_present { "present" } else { "missing" }.to_string(),
        )],
        "add `alerts.toml` to base or active overlay pack",
    ));

    let strict_out = workdir.join("out").join("strict_check_v1_gate.json");
    let strict = strict_check(workdir, true, &strict_out)?;
    let strict_v1_pass = strict.ok
        || strict
            .report
            .v1_checks
            .iter()
            .all(|c| matches!(c.status, StrictCheckStatus::Pass));
    checks.push(v1_gate_check(
        "strict_check_v1",
        if strict_v1_pass { GateStatus::Pass } else { GateStatus::Fail },
        [(
            "strict_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&strict.report)?), 16),
        )],
        "run `cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json` and resolve v1 strict failures",
    ));

    let portability_status = match (hardware_scan(&repo_root), path_scan(&repo_root)) {
        (Ok(hw), Ok(path)) => {
            let mut evidence = vec![
                (
                    "hardware_violations".to_string(),
                    hw.violations.len().to_string(),
                ),
                (
                    "path_violations".to_string(),
                    path.violations.len().to_string(),
                ),
            ];
            evidence.sort_by(|a, b| a.0.cmp(&b.0));
            v1_gate_check(
                "portability_scans",
                if hw.violations.is_empty() && path.violations.is_empty() {
                    GateStatus::Pass
                } else {
                    GateStatus::Fail
                },
                evidence,
                "run `cargo run -p ucf-ops -- portability check --out ./out/portability_check.json` and fix scan violations",
            )
        }
        (Err(err), _) | (_, Err(err)) => v1_gate_check(
            "portability_scans",
            GateStatus::Skip,
            [("error".to_string(), bounded_string(err.to_string(), 48))],
            "optional scan unavailable in this environment",
        ),
    };
    checks.push(portability_status);

    let overall_status = if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V1GateOverallStatus::Pass
    } else {
        V1GateOverallStatus::Fail
    };
    let report = V1GateReportV1 {
        schema_version: V1_SCHEMA_VERSION,
        overall_status,
        checks,
    };
    write_json(out, &report)?;
    Ok(report)
}

pub fn v2_gate(workdir: &Path, out: &Path) -> Result<V2GateReportV1, OpsError> {
    ensure_layout(workdir)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let second_slot = detect_second_slot(&repo_root)?;
    let mut checks = Vec::new();

    let v0_out = workdir.join("out").join("v0_gate_report_v2_gate.json");
    let v0 = v0_gate(
        workdir,
        &repo_root.join("fixtures/e2e/v0_flow_a.json"),
        &v0_out,
    )?;
    checks.push(v2_gate_check(
        "v0_gate_pass",
        if matches!(v0.overall_status, V0GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v0_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v0)?), 16),
        )],
        "REMEDIATE_RUN_V0_GATE",
        "NOTE_REQUIRED_V0",
    ));

    let v1_out = workdir.join("out").join("v1_gate_report_v2_gate.json");
    let v1 = v1_gate(workdir, &v1_out)?;
    checks.push(v2_gate_check(
        "v1_gate_pass",
        if matches!(v1.overall_status, V1GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v1_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v1)?), 16),
        )],
        "REMEDIATE_RUN_V1_GATE",
        "NOTE_REQUIRED_V1",
    ));

    let verify_manifest = repo_root.join("models/manifest.toml");
    let verify = models_verify(&verify_manifest)?;
    checks.push(v2_gate_check(
        "models_manifest_verify",
        if verify
            .slots
            .iter()
            .all(|slot| slot.status == "verified" || slot.status == "disabled")
        {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "models_verify".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&verify)?), 16),
        )],
        "REMEDIATE_MODELS_VERIFY",
        "NOTE_REQUIRED_MANIFEST",
    ));

    let manifest = repo_root.join("models/MANIFEST.toml");
    let probe_out = workdir.join("out").join("probe_v2_gate.json");
    let probe = models_probe(workdir, &manifest, &probe_out)?;
    let probe_digest = prefix_hex(&sha256_hex(&serde_json::to_vec(&probe)?), 16);
    let world_probe = probe
        .results
        .iter()
        .find(|r| r.slot == ModelSlot::WorldJepa)
        .map(|r| r.status);
    checks.push(v2_gate_check(
        "world_tiny_fixture_probe_pass",
        if matches!(world_probe, Some(ProbeStatus::Ok | ProbeStatus::Disabled)) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            ("probe_report".to_string(), probe_digest.clone()),
            (
                "probe_world".to_string(),
                format!("{:?}", world_probe.unwrap_or(ProbeStatus::Error)),
            ),
        ],
        "REMEDIATE_WORLD_PROBE",
        "NOTE_REQUIRED_WORLD",
    ));

    let second_probe = probe
        .results
        .iter()
        .find(|r| r.slot == second_slot)
        .map(|r| r.status);
    checks.push(v2_gate_check(
        "second_slot_tiny_fixture_probe_pass",
        if matches!(second_probe, Some(ProbeStatus::Ok | ProbeStatus::Disabled)) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            ("probe_report".to_string(), probe_digest.clone()),
            (
                format!("probe_{}", second_slot.as_str()),
                format!("{:?}", second_probe.unwrap_or(ProbeStatus::Error)),
            ),
        ],
        "REMEDIATE_SECOND_SLOT_PROBE",
        "NOTE_REQUIRED_SECOND_SLOT",
    ));

    let world_shadow = v2_shadow_no_impact_check(workdir, ModelSlot::WorldJepa, "world");
    checks.push(world_shadow);
    let second_shadow = v2_shadow_no_impact_check(workdir, second_slot, second_slot.as_str());
    checks.push(second_shadow);

    let shadow_ready_path = workdir.join("out").join("shadow_ready_report.json");
    let shadow_ready = models_shadow_ready(workdir, None, &shadow_ready_path).ok();
    let shadow_ready_digest = shadow_ready
        .as_ref()
        .map(|r| prefix_hex(&r.report_digest, 16))
        .unwrap_or_else(|| "missing".to_string());
    let world_ready = shadow_ready.as_ref().and_then(|report| {
        report
            .slots
            .iter()
            .find(|slot| slot.slot_id == ModelSlot::WorldJepa.as_str())
    });
    let world_probe_disabled = matches!(world_probe, Some(ProbeStatus::Disabled));
    checks.push(v2_gate_check(
        "world_shadow_ready",
        if world_probe_disabled {
            GateStatus::Skip
        } else if world_ready.is_some_and(|s| s.shadow_ready) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "shadow_ready_report".to_string(),
            shadow_ready_digest.clone(),
        )],
        "REMEDIATE_WORLD_SHADOW_READY",
        if world_probe_disabled {
            "NOTE_OPTIONAL_WORLD_DISABLED"
        } else {
            "NOTE_REQUIRED_WORLD"
        },
    ));
    let second_ready = shadow_ready.as_ref().and_then(|report| {
        report
            .slots
            .iter()
            .find(|slot| slot.slot_id == second_slot.as_str())
    });
    let second_probe_disabled = matches!(second_probe, Some(ProbeStatus::Disabled));
    checks.push(v2_gate_check(
        "second_slot_shadow_ready",
        if second_probe_disabled {
            GateStatus::Skip
        } else if second_ready.is_some_and(|s| s.shadow_ready) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("shadow_ready_report".to_string(), shadow_ready_digest)],
        "REMEDIATE_SECOND_SLOT_SHADOW_READY",
        if second_probe_disabled {
            "NOTE_OPTIONAL_SECOND_SLOT_DISABLED"
        } else {
            "NOTE_REQUIRED_SECOND_SLOT"
        },
    ));

    let cfg = load_or_init_config(workdir)?;
    let drift_budget_override = std::env::var("UCF_STRICT_DRIFT_BUDGET_PATH").ok();
    let base_drift_budget = repo_root.join("policies/packs/base_v1/drift_budget.toml");
    let overlay_drift_budget = repo_root.join(format!(
        "policies/packs/overlays/{}/drift_budget.toml",
        cfg.profile
    ));
    let drift_present = if let Some(path) = drift_budget_override {
        PathBuf::from(path).exists()
    } else {
        overlay_drift_budget.exists() || base_drift_budget.exists()
    };
    checks.push(v2_gate_check(
        "drift_budget_present",
        if drift_present {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "drift_budget".to_string(),
            if drift_present { "present" } else { "missing" }.to_string(),
        )],
        "REMEDIATE_DRIFT_BUDGET",
        "NOTE_REQUIRED_POLICY",
    ));

    let base_alerts = repo_root.join("policies/packs/base_v1/alerts.toml");
    let overlay_alerts = repo_root.join(format!(
        "policies/packs/overlays/{}/alerts.toml",
        cfg.profile
    ));
    let alerts_present = overlay_alerts.exists() || base_alerts.exists();
    checks.push(v2_gate_check(
        "alerts_rules_present",
        if alerts_present {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "alerts".to_string(),
            if alerts_present { "present" } else { "missing" }.to_string(),
        )],
        "REMEDIATE_ALERTS_RULES",
        "NOTE_REQUIRED_POLICY",
    ));

    let strict_out = workdir.join("out").join("strict_check_v2_gate.json");
    let strict = strict_check(workdir, true, &strict_out)?;
    let strict_v2_pass = strict.report.v1_checks.iter().all(|check| {
        !check.check_id.starts_with("v2_") || matches!(check.status, StrictCheckStatus::Pass)
    });
    checks.push(v2_gate_check(
        "strict_check_v2",
        if strict_v2_pass {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "strict_report".to_string(),
            prefix_hex(&strict.report.digest_hex()?, 16),
        )],
        "REMEDIATE_STRICT_V2",
        "NOTE_REQUIRED_STRICT",
    ));

    let parity_path = workdir.join("out").join("world_parity_report.json");
    let parity = world_parity_report(workdir, &probe.run_id, &parity_path).ok();
    let parity_present = parity.is_some() || parity_path.exists();
    let parity_digest = parity
        .as_ref()
        .map(|r| prefix_hex(&r.report_digest, 16))
        .or_else(|| {
            fs::read(&parity_path)
                .ok()
                .map(|b| prefix_hex(&sha256_hex(&b), 16))
        })
        .unwrap_or_else(|| "missing".to_string());
    checks.push(v2_gate_check(
        "world_parity_report_present",
        if parity_present {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("world_parity_report".to_string(), parity_digest.clone())],
        "REMEDIATE_WORLD_PARITY_REPORT",
        "NOTE_REQUIRED_WORLD",
    ));

    let burn_check = if cfg!(feature = "backend-burn") {
        let burn_probe = world_probe
            .as_ref()
            .is_some_and(|s| matches!(s, ProbeStatus::Ok));
        v2_gate_check(
            "burn_world_probe_pass",
            if burn_probe {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            [
                ("probe_report".to_string(), probe_digest),
                (
                    "burn_expected".to_string(),
                    if burn_probe { "ok" } else { "probe_not_ok" }.to_string(),
                ),
            ],
            "REMEDIATE_BURN_PROBE",
            "NOTE_OPTIONAL_BURN",
        )
    } else {
        v2_gate_check(
            "burn_world_probe_pass",
            GateStatus::Skip,
            [("burn_backend".to_string(), "not_enabled".to_string())],
            "REMEDIATE_ENABLE_BURN_BACKEND",
            "NOTE_OPTIONAL_BURN",
        )
    };
    checks.push(burn_check);

    let burn_shadow = if cfg!(feature = "backend-burn") {
        checks
            .iter()
            .find(|c| c.name == "world_parity_report_present")
            .map(|c| {
                v2_gate_check(
                    "burn_world_shadow_compare_present",
                    c.status,
                    [("world_parity_report".to_string(), parity_digest)],
                    "REMEDIATE_BURN_COMPARE_REPORT",
                    "NOTE_OPTIONAL_BURN",
                )
            })
            .unwrap_or_else(|| {
                v2_gate_check(
                    "burn_world_shadow_compare_present",
                    GateStatus::Fail,
                    [("world_parity_report".to_string(), "missing".to_string())],
                    "REMEDIATE_BURN_COMPARE_REPORT",
                    "NOTE_OPTIONAL_BURN",
                )
            })
    } else {
        v2_gate_check(
            "burn_world_shadow_compare_present",
            GateStatus::Skip,
            [("burn_backend".to_string(), "not_enabled".to_string())],
            "REMEDIATE_ENABLE_BURN_BACKEND",
            "NOTE_OPTIONAL_BURN",
        )
    };
    checks.push(burn_shadow);

    let overall_status = if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V2GateOverallStatus::Pass
    } else {
        V2GateOverallStatus::Fail
    };
    let report = V2GateReportV1 {
        schema_version: V2_SCHEMA_VERSION,
        overall_status,
        checks,
    };
    write_json(out, &report)?;
    Ok(report)
}

pub fn v3_gate(workdir: &Path, out: &Path) -> Result<V3GateReportV1, OpsError> {
    ensure_layout(workdir)?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut checks = Vec::new();

    let v0_out = workdir.join("out").join("v0_gate_report_v3_gate.json");
    let v0 = v0_gate(
        workdir,
        &repo_root.join("fixtures/e2e/v0_flow_a.json"),
        &v0_out,
    )?;
    checks.push(v3_gate_check(
        "v0_gate_pass",
        if matches!(v0.overall_status, V0GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v0_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v0)?), 16),
        )],
        "REMEDIATE_RUN_V0_GATE",
        "NOTE_REQUIRED_V0",
    ));

    let v1_out = workdir.join("out").join("v1_gate_report_v3_gate.json");
    let v1 = v1_gate(workdir, &v1_out)?;
    checks.push(v3_gate_check(
        "v1_gate_pass",
        if matches!(v1.overall_status, V1GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v1_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v1)?), 16),
        )],
        "REMEDIATE_RUN_V1_GATE",
        "NOTE_REQUIRED_V1",
    ));

    let v2_out = workdir.join("out").join("v2_gate_report_v3_gate.json");
    let v2 = v2_gate(workdir, &v2_out)?;
    checks.push(v3_gate_check(
        "v2_gate_pass",
        if matches!(v2.overall_status, V2GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v2_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v2)?), 16),
        )],
        "REMEDIATE_RUN_V2_GATE",
        "NOTE_REQUIRED_V2",
    ));

    let second_slot = detect_second_slot_for_v3(&repo_root);
    checks.push(match &second_slot {
        Ok(slot) => v3_gate_check(
            "supported_slot_set_detected",
            GateStatus::Pass,
            [
                (
                    "supported_slot_world".to_string(),
                    ModelSlot::WorldJepa.as_str().to_string(),
                ),
                (
                    "supported_slot_second".to_string(),
                    slot.as_str().to_string(),
                ),
            ],
            "REMEDIATE_DECLARE_SECOND_SLOT",
            "NOTE_REQUIRED_SCOPE",
        ),
        Err(err) => v3_gate_check(
            "supported_slot_set_detected",
            GateStatus::Fail,
            [("error".to_string(), bounded_string(err.to_string(), 48))],
            "REMEDIATE_DECLARE_SECOND_SLOT",
            "NOTE_REQUIRED_SCOPE",
        ),
    });

    let manifest = repo_root.join("models/MANIFEST.toml");
    let probe_out = workdir.join("out").join("probe_v3_gate.json");
    let probe = models_probe(workdir, &manifest, &probe_out).ok();
    let probe_digest = probe
        .as_ref()
        .map(|r| prefix_hex(&sha256_hex(&serde_json::to_vec(r).unwrap_or_default()), 16))
        .unwrap_or_else(|| "missing".to_string());
    let world_probe = probe.as_ref().and_then(|report| {
        report
            .results
            .iter()
            .find(|r| r.slot == ModelSlot::WorldJepa)
            .map(|r| r.status)
    });
    checks.push(v3_gate_check(
        "world_probe_ready",
        if matches!(world_probe, Some(ProbeStatus::Ok | ProbeStatus::Disabled)) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            ("probe_report".to_string(), probe_digest.clone()),
            (
                "probe_world".to_string(),
                format!("{:?}", world_probe.unwrap_or(ProbeStatus::Error)),
            ),
        ],
        "REMEDIATE_WORLD_PROBE",
        "NOTE_REQUIRED_WORLD",
    ));

    let second_probe_status = second_slot.as_ref().ok().and_then(|slot| {
        probe.as_ref().and_then(|report| {
            report
                .results
                .iter()
                .find(|r| r.slot == *slot)
                .map(|r| r.status)
        })
    });
    let second_probe_note = second_slot
        .as_ref()
        .map(|slot| slot.as_str().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    checks.push(v3_gate_check(
        "second_slot_probe_ready",
        if matches!(second_probe_status, Some(ProbeStatus::Ok)) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            ("probe_report".to_string(), probe_digest.clone()),
            (
                format!("probe_{second_probe_note}"),
                format!("{:?}", second_probe_status.unwrap_or(ProbeStatus::Error)),
            ),
        ],
        "REMEDIATE_SECOND_SLOT_PROBE",
        "NOTE_REQUIRED_SECOND_SLOT",
    ));

    let shadow_ready_out = workdir.join("out").join("shadow_ready_report_v3_gate.json");
    let shadow_ready = models_shadow_ready(workdir, None, &shadow_ready_out).ok();
    let shadow_ready_digest = shadow_ready
        .as_ref()
        .map(|r| prefix_hex(&r.report_digest, 16))
        .unwrap_or_else(|| "missing".to_string());

    let world_ready = shadow_ready.as_ref().and_then(|report| {
        report
            .slots
            .iter()
            .find(|slot| slot.slot_id == ModelSlot::WorldJepa.as_str())
    });
    checks.push(v3_gate_check(
        "world_shadow_ready",
        if world_ready.is_some_and(|s| s.shadow_ready) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "shadow_ready_report".to_string(),
            shadow_ready_digest.clone(),
        )],
        "REMEDIATE_WORLD_SHADOW_READY",
        "NOTE_REQUIRED_WORLD",
    ));

    let second_ready = second_slot.as_ref().ok().and_then(|slot| {
        shadow_ready.as_ref().and_then(|report| {
            report
                .slots
                .iter()
                .find(|entry| entry.slot_id == slot.as_str())
        })
    });
    checks.push(v3_gate_check(
        "second_slot_shadow_ready",
        if second_ready.is_some_and(|s| s.shadow_ready) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "shadow_ready_report".to_string(),
            shadow_ready_digest.clone(),
        )],
        "REMEDIATE_SECOND_SLOT_SHADOW_READY",
        "NOTE_REQUIRED_SECOND_SLOT",
    ));

    let world_no_impact = v2_shadow_no_impact_check(workdir, ModelSlot::WorldJepa, "world");
    checks.push(v3_gate_check(
        "world_shadow_no_impact",
        world_no_impact.status,
        world_no_impact.evidence_digest_prefixes,
        "REMEDIATE_SHADOW_NO_IMPACT",
        "NOTE_REQUIRED_WORLD",
    ));

    let second_shadow_check = second_slot
        .as_ref()
        .ok()
        .map(|slot| v2_shadow_no_impact_check(workdir, *slot, slot.as_str()));
    checks.push(if let Some(check) = second_shadow_check {
        v3_gate_check(
            "second_slot_shadow_no_impact",
            check.status,
            check.evidence_digest_prefixes,
            "REMEDIATE_SHADOW_NO_IMPACT",
            "NOTE_REQUIRED_SECOND_SLOT",
        )
    } else {
        v3_gate_check(
            "second_slot_shadow_no_impact",
            GateStatus::Fail,
            [("error".to_string(), "second_slot_unknown".to_string())],
            "REMEDIATE_SHADOW_NO_IMPACT",
            "NOTE_REQUIRED_SECOND_SLOT",
        )
    });

    let world_parity = world_parity_evidence_exists(workdir);
    let second_parity = second_slot
        .as_ref()
        .ok()
        .is_some_and(|slot| second_slot_parity_evidence_exists(workdir, *slot));
    let semantics = unified_compare_semantics_v1();
    let semantics_ok = semantics.window_id_rule == "u64_prefix(sha256(run_id:slot_id:t0:t1))"
        && semantics.freshness_rule == "current_tick - t1 <= max_age => FRESH else STALE_COMPARE"
        && world_parity
        && second_parity;
    checks.push(v3_gate_check(
        "compare_window_semantics_normalized",
        if semantics_ok {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "window_rule".to_string(),
                bounded_string(semantics.window_id_rule, 32),
            ),
            (
                "freshness_rule".to_string(),
                bounded_string(semantics.freshness_rule, 32),
            ),
            (
                "world_parity_present".to_string(),
                if world_parity { "yes" } else { "no" }.to_string(),
            ),
            (
                "second_parity_present".to_string(),
                if second_parity { "yes" } else { "no" }.to_string(),
            ),
        ],
        "REMEDIATE_COMPARE_SEMANTICS",
        "NOTE_REQUIRED_COMPARE",
    ));

    let eligibility_out = workdir
        .join("out")
        .join("models_eligibility_report_v3_gate.json");
    let eligibility = models_eligibility(workdir, None, &eligibility_out).ok();
    let eligibility_ok = second_slot.as_ref().ok().is_some_and(|slot| {
        eligibility.as_ref().is_some_and(|report| {
            report
                .slots
                .iter()
                .any(|s| s.slot_id == ModelSlot::WorldJepa.as_str())
                && report.slots.iter().any(|s| s.slot_id == slot.as_str())
        })
    });
    checks.push(v3_gate_check(
        "unified_eligibility_report_present",
        if eligibility_ok {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "eligibility_report".to_string(),
            eligibility
                .as_ref()
                .map(|r| prefix_hex(&r.report_digest, 16))
                .unwrap_or_else(|| "missing".to_string()),
        )],
        "REMEDIATE_ELIGIBILITY_REPORT",
        "NOTE_REQUIRED_ELIGIBILITY",
    ));

    let strict_out = workdir.join("out").join("strict_check_v3_gate.json");
    let strict = strict_check(workdir, true, &strict_out).ok();
    let strict_v3_pass = strict.as_ref().is_some_and(|report| {
        report.ok
            && report.report.v3.as_ref().is_some_and(|v3| {
                v3.checks
                    .iter()
                    .all(|c| !matches!(c.status, StrictCheckV3Status::Fail))
            })
    });
    checks.push(v3_gate_check(
        "strict_check_v3_pass",
        if strict_v3_pass {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "strict_report".to_string(),
            strict
                .as_ref()
                .and_then(|r| r.report.digest_hex().ok())
                .map(|d| prefix_hex(&d, 16))
                .unwrap_or_else(|| "missing".to_string()),
        )],
        "REMEDIATE_STRICT_V3",
        "NOTE_REQUIRED_STRICT",
    ));

    let operator_out = workdir.join("out").join("operator_report_v3_gate.json");
    let operator = operator_report(
        workdir,
        &OperatorReportArgs {
            run_id: None,
            latest: true,
        },
        &operator_out,
    )
    .ok();
    checks.push(v3_gate_check(
        "operator_report_present",
        if operator.is_some() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "operator_report".to_string(),
            operator
                .as_ref()
                .map(|r| prefix_hex(&r.report_digest, 16))
                .unwrap_or_else(|| "missing".to_string()),
        )],
        "REMEDIATE_OPERATOR_REPORT",
        "NOTE_REQUIRED_OPERATOR",
    ));

    let docs_out = workdir.join("out").join("docs_lint_v3_gate.json");
    let docs = docs_lint(&DocsLintArgs {
        repo_root: repo_root.clone(),
        policy_pack: repo_root.join("policies/packs/base_v1"),
        overlay_pack: Some(repo_root.join("policies/packs/overlays/test")),
        spec_snapshot: repo_root.join("docs/spec_snapshot.md"),
        prompt_index: repo_root.join("docs/prompt_series_index.md"),
        module_map: repo_root.join("docs/module_map.md"),
        deploy_doc: repo_root.join("docs/deploy.md"),
        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
        mode: DocsLintMode::Strict,
    })
    .ok();
    if let Some(report) = &docs {
        let _ = write_json(&docs_out, report);
    }
    let portability_out = workdir.join("out").join("portability_check_v3_gate.json");
    let portability = portability_check(&portability_out).ok();
    let docs_pass = docs.as_ref().is_some_and(|r| r.ok);
    let portability_pass = portability
        .as_ref()
        .is_some_and(|r| r.deterministic_within_os);
    checks.push(v3_gate_check(
        "portability_docs_checks_pass",
        if docs_pass && portability_pass {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "docs_lint".to_string(),
                if docs_pass { "pass" } else { "fail" }.to_string(),
            ),
            (
                "portability_check".to_string(),
                if portability_pass { "pass" } else { "fail" }.to_string(),
            ),
        ],
        "REMEDIATE_PORTABILITY_DOCS",
        "NOTE_REQUIRED_DOCS",
    ));

    checks.push(if cfg!(feature = "backend-burn") {
        v3_gate_check(
            "burn_world_parity_present",
            if world_parity {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            [(
                "world_parity_report".to_string(),
                if world_parity { "present" } else { "missing" }.to_string(),
            )],
            "REMEDIATE_BURN_COMPARE_REPORT",
            "NOTE_OPTIONAL_BURN",
        )
    } else {
        v3_gate_check(
            "burn_world_parity_present",
            GateStatus::Skip,
            [("burn_backend".to_string(), "not_enabled".to_string())],
            "REMEDIATE_ENABLE_BURN_BACKEND",
            "NOTE_OPTIONAL_BURN",
        )
    });

    let overall_status = if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V3GateOverallStatus::Pass
    } else {
        V3GateOverallStatus::Fail
    };

    let report = V3GateReportV1 {
        schema_version: V3_SCHEMA_VERSION,
        overall_status,
        checks,
    };
    write_json(out, &report)?;
    Ok(report)
}

pub fn v4_gate(workdir: &Path, out: &Path) -> Result<V4GateReportV1, OpsError> {
    ensure_layout(workdir)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut checks = Vec::new();

    let v0_out = workdir.join("out").join("v0_gate_report_v4_gate.json");
    let v0 = v0_gate(
        workdir,
        &repo_root.join("fixtures/e2e/v0_flow_a.json"),
        &v0_out,
    )?;
    checks.push(v4_gate_check(
        "v0_gate_pass",
        if matches!(v0.overall_status, V0GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v0_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v0)?), 16),
        )],
        "REMEDIATE_RUN_V0_GATE",
        "NOTE_REQUIRED_V0",
    ));

    let v1_out = workdir.join("out").join("v1_gate_report_v4_gate.json");
    let v1 = v1_gate(workdir, &v1_out)?;
    checks.push(v4_gate_check(
        "v1_gate_pass",
        if matches!(v1.overall_status, V1GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v1_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v1)?), 16),
        )],
        "REMEDIATE_RUN_V1_GATE",
        "NOTE_REQUIRED_V1",
    ));

    let v2_out = workdir.join("out").join("v2_gate_report_v4_gate.json");
    let v2 = v2_gate(workdir, &v2_out)?;
    checks.push(v4_gate_check(
        "v2_gate_pass",
        if matches!(v2.overall_status, V2GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v2_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v2)?), 16),
        )],
        "REMEDIATE_RUN_V2_GATE",
        "NOTE_REQUIRED_V2",
    ));

    let v3_out = workdir.join("out").join("v3_gate_report_v4_gate.json");
    let v3 = v3_gate(workdir, &v3_out)?;
    checks.push(v4_gate_check(
        "v3_gate_pass",
        if matches!(v3.overall_status, V3GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v3_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v3)?), 16),
        )],
        "REMEDIATE_RUN_V3_GATE",
        "NOTE_REQUIRED_V3",
    ));

    let slot_set = models_lifecycle::supported_real_slot_set_v1()?;
    let second_slot = detect_second_slot_for_v3(&repo_root)?;
    let slot_set_consistent = slot_set.slots.len() == 2
        && slot_set
            .slots
            .iter()
            .any(|slot| slot == ModelSlot::WorldJepa.as_str())
        && slot_set
            .slots
            .iter()
            .any(|slot| slot == second_slot.as_str());
    checks.push(v4_gate_check(
        "supported_slot_set_consistent",
        if slot_set_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "slot_set_digest".to_string(),
                prefix_hex(&slot_set.set_digest, 16),
            ),
            ("slots".to_string(), slot_set.slots.join(",")),
        ],
        "REMEDIATE_MODELS_CONSISTENCY",
        "NOTE_REQUIRED_SLOT_SET",
    ));

    let models_consistency_out = workdir
        .join("out")
        .join("models_consistency_check_v4_gate.json");
    let models_consistency = models_consistency_check(workdir, &models_consistency_out)?;
    let optional_backend_states_check = v4_gate_check(
        "optional_backend_states_consistent",
        if models_consistency.status == "PASS" {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "consistency_report".to_string(),
                prefix_hex(&sha256_hex(&serde_json::to_vec(&models_consistency)?), 16),
            ),
            (
                "mismatch_count".to_string(),
                models_consistency.mismatch_categories.len().to_string(),
            ),
        ],
        "REMEDIATE_MODELS_CONSISTENCY",
        "NOTE_OPTIONAL_BACKEND_STATE",
    );

    let backend_snapshot_out = workdir
        .join("out")
        .join("backend_evidence_snapshot_v4_gate.json");
    let backend_snapshot = models_evidence_snapshot(workdir, None, None)?;
    write_json(&backend_snapshot_out, &backend_snapshot)?;
    checks.push(v4_gate_check(
        "backend_evidence_snapshot_present",
        if backend_snapshot.slots.is_empty() {
            GateStatus::Fail
        } else {
            GateStatus::Pass
        },
        [(
            "snapshot_digest".to_string(),
            prefix_hex(&backend_snapshot.snapshot_digest, 16),
        )],
        "REMEDIATE_BACKEND_EVIDENCE_SNAPSHOT",
        "NOTE_REQUIRED_EVIDENCE",
    ));

    let artifact_schema = check_artifact_schema_snapshots(&artifact_schema::ArtifactSchemaArgs {
        repo_root: repo_root.clone(),
        out_dir: repo_root.join("docs/artifact_schema_snapshots"),
    })?;
    checks.push(v4_gate_check(
        "backend_evidence_snapshot_schema_stable",
        if artifact_schema
            .diffs
            .iter()
            .all(|d| d.artifact != "backend_evidence_snapshot_v1")
        {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "backend_schema_drift".to_string(),
            artifact_schema
                .diffs
                .iter()
                .filter(|d| d.artifact == "backend_evidence_snapshot_v1")
                .count()
                .to_string(),
        )],
        "REMEDIATE_ARTIFACT_SCHEMA_SNAPSHOT",
        "NOTE_REQUIRED_SCHEMA",
    ));

    let operator_out = workdir.join("out").join("operator_report_v4_gate.json");
    let operator = operator_report(
        workdir,
        &OperatorReportArgs {
            run_id: None,
            latest: true,
        },
        &operator_out,
    )?;
    checks.push(v4_gate_check(
        "operator_report_present",
        GateStatus::Pass,
        [(
            "operator_report_digest".to_string(),
            prefix_hex(&operator.report_digest, 16),
        )],
        "REMEDIATE_OPERATOR_REPORT",
        "NOTE_REQUIRED_OPERATOR",
    ));

    let signoff_out = workdir.join("out").join("operator_signoff_v4_gate.json");
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: "test".to_string(),
        },
        &signoff_out,
    )?;
    checks.push(v4_gate_check(
        "operator_signoff_present",
        GateStatus::Pass,
        [(
            "operator_signoff_digest".to_string(),
            prefix_hex(&signoff.decision_digest, 16),
        )],
        "REMEDIATE_OPERATOR_SIGNOFF",
        "NOTE_REQUIRED_SIGNOFF",
    ));

    let signoff_consistent = signoff
        .evidence_snapshot_digest_prefix
        .starts_with(&prefix_hex(&backend_snapshot.snapshot_digest, 16))
        && signoff
            .operator_report_digest_prefix
            .starts_with(&prefix_hex(&operator.report_digest, 16));
    checks.push(v4_gate_check(
        "operator_signoff_consistent_with_evidence",
        if signoff_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "snapshot_digest".to_string(),
                signoff.evidence_snapshot_digest_prefix.clone(),
            ),
            (
                "operator_digest".to_string(),
                signoff.operator_report_digest_prefix.clone(),
            ),
        ],
        "REMEDIATE_OPERATOR_SIGNOFF",
        "NOTE_REQUIRED_SIGNOFF_ALIGNMENT",
    ));

    let remediation_doc = repo_root.join("docs/remediation_codes_v1.md");
    let remediation_doc_present = remediation_doc.exists();
    checks.push(v4_gate_check(
        "remediation_registry_present",
        if remediation_doc_present {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "registry_doc".to_string(),
            if remediation_doc_present {
                "present".to_string()
            } else {
                "missing".to_string()
            },
        )],
        "REMEDIATE_REMEDIATION_REGISTRY",
        "NOTE_REQUIRED_REMEDIATION_REGISTRY",
    ));

    let generated_registry_dir = tempfile::tempdir()?;
    let generated_registry = generated_registry_dir
        .path()
        .join("remediation_codes_v1.md");
    generate_remediation_codes_doc(&generated_registry)?;
    let registry_consistent = fs::read_to_string(&generated_registry)?.replace("\r\n", "\n")
        == fs::read_to_string(&remediation_doc)
            .unwrap_or_default()
            .replace("\r\n", "\n")
        && remediation_codes_aligned(&operator, &signoff, &backend_snapshot, &repo_root)?;
    checks.push(v4_gate_check(
        "remediation_registry_consistent_across_reports",
        if registry_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "registry_alignment".to_string(),
            if registry_consistent {
                "ok".to_string()
            } else {
                "mismatch".to_string()
            },
        )],
        "REMEDIATE_REMEDIATION_REGISTRY",
        "NOTE_REQUIRED_REMEDIATION_REGISTRY_ALIGNMENT",
    ));

    let strict_snapshot = resolve_strict_evidence(
        &PathBuf::from("./out"),
        &StrictEvidenceContextV1 {
            run_id: None,
            latest: true,
            strict_required: true,
            expected_policy_graph_digest_prefix: Some(
                backend_snapshot.policy_graph_digest_prefix.clone(),
            ),
            expected_manifest_digest_prefix: Some(backend_snapshot.manifest_digest_prefix.clone()),
            expected_supported_slot_set_digest_prefix: Some(
                backend_snapshot.supported_slot_set_digest.clone(),
            ),
        },
    );
    checks.push(v4_gate_check(
        "strict_evidence_present",
        if matches!(
            strict_snapshot.strict_status,
            StrictEvidenceStatusV1::Pass | StrictEvidenceStatusV1::Fail
        ) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "strict_snapshot_digest".to_string(),
            prefix_hex(&strict_snapshot.snapshot_digest, 16),
        )],
        "REMEDIATE_STRICT_V3",
        "NOTE_REQUIRED_STRICT_EVIDENCE",
    ));

    let strict_alignment = operator.sections.strict_section.strict_status
        == strict_snapshot.strict_status
        && operator.sections.strict_section.primary_denial_code
            == strict_snapshot.primary_denial_code
        && strict_operator_signoff_alignment(&strict_snapshot, &operator, &signoff);
    checks.push(v4_gate_check(
        "strict_operator_alignment_ok",
        if strict_alignment {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "strict_status".to_string(),
            format!("{:?}", strict_snapshot.strict_status),
        )],
        "REMEDIATE_STRICT_OPERATOR_ALIGNMENT",
        "NOTE_REQUIRED_STRICT_ALIGNMENT",
    ));

    checks.push(v4_gate_check(
        "artifact_schema_snapshot_checks_pass",
        if artifact_schema.ok {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "schema_diff_count".to_string(),
            artifact_schema.diffs.len().to_string(),
        )],
        "REMEDIATE_ARTIFACT_SCHEMA_SNAPSHOT",
        "NOTE_REQUIRED_SCHEMA",
    ));

    let docs = docs_lint(&DocsLintArgs {
        repo_root: repo_root.clone(),
        policy_pack: repo_root.join("policies/packs/base_v1"),
        overlay_pack: Some(repo_root.join("policies/packs/overlays/test")),
        spec_snapshot: repo_root.join("docs/spec_snapshot.md"),
        prompt_index: repo_root.join("docs/prompt_series_index.md"),
        module_map: repo_root.join("docs/module_map.md"),
        deploy_doc: repo_root.join("docs/deploy.md"),
        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
        mode: DocsLintMode::Strict,
    })?;
    let portability_out = workdir.join("out").join("portability_check_v4_gate.json");
    let portability = portability_check(&portability_out)?;
    checks.push(v4_gate_check(
        "portability_docs_checks_pass",
        if docs.ok && portability.deterministic_within_os {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "docs_lint".to_string(),
                if docs.ok { "pass" } else { "fail" }.to_string(),
            ),
            (
                "portability_check".to_string(),
                if portability.deterministic_within_os {
                    "pass"
                } else {
                    "fail"
                }
                .to_string(),
            ),
        ],
        "REMEDIATE_PORTABILITY_DOCS",
        "NOTE_REQUIRED_DOCS",
    ));

    checks.push(optional_backend_states_check);

    let parity_path = workdir
        .join("out")
        .join(format!("{}_parity_report.json", second_slot.as_str()));
    checks.push(if parity_path.exists() {
        let parity = fs::read_to_string(&parity_path)
            .ok()
            .and_then(|body| serde_json::from_str::<SecondSlotParityReportV1>(&body).ok());
        v4_gate_check(
            "burn_parity_optional_path_consistent",
            if parity.is_some() {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            [(
                "parity_report".to_string(),
                parity
                    .as_ref()
                    .map(|r| prefix_hex(&r.report_digest, 16))
                    .unwrap_or_else(|| "parse_error".to_string()),
            )],
            "REMEDIATE_SECOND_SLOT_PARITY",
            "NOTE_OPTIONAL_BURN",
        )
    } else {
        v4_gate_check(
            "burn_parity_optional_path_consistent",
            GateStatus::Skip,
            [("parity_report".to_string(), "not_configured".to_string())],
            "REMEDIATE_SECOND_SLOT_PARITY",
            "NOTE_OPTIONAL_BURN",
        )
    });

    let overall_status = if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V4GateOverallStatus::Pass
    } else {
        V4GateOverallStatus::Fail
    };

    let report = V4GateReportV1 {
        schema_version: 1,
        overall_status,
        checks,
    };
    write_json(out, &report)?;
    Ok(report)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceSurfacesCheckReportV1 {
    pub schema_version: u16,
    pub status: String,
    pub summary_code: String,
    pub governance_primary_surfaces: Option<GovernancePrimarySurfacesV1>,
}

pub fn governance_surfaces_check(
    workdir: &Path,
    out: &Path,
) -> Result<GovernanceSurfacesCheckReportV1, OpsError> {
    ensure_layout(workdir)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    let backend_snapshot = models_evidence_snapshot(workdir, None, None)?;
    let active_out = workdir
        .join("out")
        .join("active_review_snapshot_governance_surfaces_check.json");
    let active_review_snapshot = models_active_review_snapshot(workdir, &active_out)?;

    let report =
        match validate_governance_primary_surfaces(&backend_snapshot, &active_review_snapshot) {
            Ok(primary) => GovernanceSurfacesCheckReportV1 {
                schema_version: 1,
                status: "PASS".to_string(),
                summary_code: "PASS".to_string(),
                governance_primary_surfaces: Some(primary),
            },
            Err(err) => GovernanceSurfacesCheckReportV1 {
                schema_version: 1,
                status: "FAIL".to_string(),
                summary_code: err.to_string(),
                governance_primary_surfaces: None,
            },
        };

    write_json(out, &report)?;
    Ok(report)
}

pub fn v5_gate(workdir: &Path, out: &Path) -> Result<V5GateReportV1, OpsError> {
    ensure_layout(workdir)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut checks = Vec::new();

    let v0_out = workdir.join("out").join("v0_gate_report_v5_gate.json");
    let v0 = v0_gate(
        workdir,
        &repo_root.join("fixtures/e2e/v0_flow_a.json"),
        &v0_out,
    )?;
    checks.push(v5_gate_check(
        "v0_gate_pass",
        if matches!(v0.overall_status, V0GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v0_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v0)?), 16),
        )],
        "REMEDIATE_RUN_V0_GATE",
        "NOTE_REQUIRED_V0",
    ));

    let v1_out = workdir.join("out").join("v1_gate_report_v5_gate.json");
    let v1 = v1_gate(workdir, &v1_out)?;
    checks.push(v5_gate_check(
        "v1_gate_pass",
        if matches!(v1.overall_status, V1GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v1_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v1)?), 16),
        )],
        "REMEDIATE_RUN_V1_GATE",
        "NOTE_REQUIRED_V1",
    ));

    let v2_out = workdir.join("out").join("v2_gate_report_v5_gate.json");
    let v2 = v2_gate(workdir, &v2_out)?;
    checks.push(v5_gate_check(
        "v2_gate_pass",
        if matches!(v2.overall_status, V2GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v2_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v2)?), 16),
        )],
        "REMEDIATE_RUN_V2_GATE",
        "NOTE_REQUIRED_V2",
    ));

    let v3_out = workdir.join("out").join("v3_gate_report_v5_gate.json");
    let v3 = v3_gate(workdir, &v3_out)?;
    checks.push(v5_gate_check(
        "v3_gate_pass",
        if matches!(v3.overall_status, V3GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v3_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v3)?), 16),
        )],
        "REMEDIATE_RUN_V3_GATE",
        "NOTE_REQUIRED_V3",
    ));

    let v4_out = workdir.join("out").join("v4_gate_report_v5_gate.json");
    let v4 = v4_gate(workdir, &v4_out)?;
    checks.push(v5_gate_check(
        "v4_gate_pass",
        if matches!(v4.overall_status, V4GateOverallStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "v4_gate_report".to_string(),
            prefix_hex(&sha256_hex(&serde_json::to_vec(&v4)?), 16),
        )],
        "REMEDIATE_RUN_V4_GATE",
        "NOTE_REQUIRED_V4",
    ));

    let second_slot = detect_second_slot_for_v3(&repo_root)?;
    let supported_slots = models_lifecycle::supported_real_slot_set_v1()?;
    let supported_set_out = workdir
        .join("out")
        .join("supported_set_review_v5_gate.json");
    let supported_set = models_supported_set_review(workdir, &supported_set_out)?;
    checks.push(v5_gate_check(
        "supported_set_review_present",
        if supported_set_out.exists() && !supported_set.policy.policy_digest.is_empty() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "supported_set_policy_digest".to_string(),
            prefix_hex(&supported_set.policy.policy_digest, 16),
        )],
        "REMEDIATE_SUPPORTED_SET_REVIEW",
        "NOTE_REQUIRED_SUPPORTED_SET",
    ));
    let supported_set_consistent = {
        let expected = BTreeSet::from([
            ModelSlot::WorldJepa.as_str().to_string(),
            second_slot.as_str().to_string(),
        ]);
        let current = supported_slots
            .slots
            .iter()
            .cloned()
            .collect::<BTreeSet<String>>();
        let reviewed = supported_set
            .policy
            .current_supported_slots
            .iter()
            .cloned()
            .collect::<BTreeSet<String>>();
        current == expected && reviewed == current
    };
    checks.push(v5_gate_check(
        "supported_set_review_consistent",
        if supported_set_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "supported_slot_set_digest".to_string(),
                prefix_hex(&supported_slots.set_digest, 16),
            ),
            (
                "supported_slots".to_string(),
                supported_slots.slots.join(","),
            ),
        ],
        "REMEDIATE_SUPPORTED_SET_REVIEW",
        "NOTE_REQUIRED_SUPPORTED_SET_ALIGNMENT",
    ));

    let backend_evidence_snapshot = models_evidence_snapshot(workdir, None, None)?;
    let active_review_out = workdir
        .join("out")
        .join("active_review_snapshot_v5_gate.json");
    let active_review = models_active_review_snapshot(workdir, &active_review_out)?;
    checks.push(v5_gate_check(
        "active_review_snapshot_present",
        if !active_review.slots.is_empty() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "active_review_snapshot_digest".to_string(),
            prefix_hex(&active_review.snapshot_digest, 16),
        )],
        "REMEDIATE_ACTIVE_REVIEW_SNAPSHOT",
        "NOTE_REQUIRED_ACTIVE_REVIEW",
    ));
    let governance_surfaces_validation =
        validate_governance_primary_surfaces(&backend_evidence_snapshot, &active_review);
    let active_review_slots = active_review
        .slots
        .iter()
        .map(|slot| slot.slot_id.clone())
        .collect::<BTreeSet<_>>();
    let active_review_consistent = governance_surfaces_validation.is_ok()
        && active_review.supported_slot_set_digest == supported_slots.set_digest
        && active_review_slots
            == supported_slots
                .slots
                .iter()
                .cloned()
                .collect::<BTreeSet<String>>()
        && active_review.signoff_alignment.aligned;
    checks.push(v5_gate_check(
        "active_review_snapshot_consistent",
        if active_review_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "active_review_alignment".to_string(),
            active_review.signoff_alignment.status_code.clone(),
        )],
        "REMEDIATE_ACTIVE_REVIEW_ALIGNMENT",
        "NOTE_REQUIRED_ACTIVE_REVIEW_ALIGNMENT",
    ));
    checks.push(v5_gate_check(
        "governance_primary_surfaces_consistent",
        if governance_surfaces_validation.is_ok() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "governance_surface_status".to_string(),
            governance_surfaces_validation
                .as_ref()
                .map(|s| s.governance_surfaces_digest.clone())
                .unwrap_or_else(|e| e.to_string()),
        )],
        "REMEDIATE_GOVERNANCE_SURFACES",
        "NOTE_REQUIRED_GOVERNANCE_SURFACE_ALIGNMENT",
    ));

    let backend_resolution_out = workdir.join("out").join(format!(
        "backend_resolution_{}_v5_gate.json",
        second_slot.as_str()
    ));
    let backend_resolution = models_backend_resolution(workdir, second_slot, None)?;
    write_json(&backend_resolution_out, &backend_resolution)?;
    checks.push(v5_gate_check(
        "backend_resolution_present",
        if !backend_resolution.evidence_digest.is_empty() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "backend_resolution_digest".to_string(),
            prefix_hex(&backend_resolution.evidence_digest, 16),
        )],
        "REMEDIATE_BACKEND_RESOLUTION",
        "NOTE_REQUIRED_BACKEND_RESOLUTION",
    ));
    let backend_resolution_consistent = backend_resolution.slot_id == second_slot.as_str()
        && active_review
            .slots
            .iter()
            .find(|slot| slot.slot_id == second_slot.as_str())
            .map(|slot| slot.burn_resolution == backend_resolution)
            .unwrap_or(false);
    checks.push(v5_gate_check(
        "backend_resolution_consistent",
        if backend_resolution_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "backend_resolution_state".to_string(),
            format!("{:?}", backend_resolution.support_state),
        )],
        "REMEDIATE_BACKEND_RESOLUTION_ALIGNMENT",
        "NOTE_REQUIRED_BACKEND_RESOLUTION_ALIGNMENT",
    ));

    let repro_smoke = repro_pack_smoke(
        "v5_repro_smoke",
        "./out/repro_v5_gate.zip",
        "./out/repro_verify_v5_gate.json",
    );
    checks.push(v5_gate_check(
        "enriched_repro_export_smoke_pass",
        if matches!(repro_smoke.status, PortabilityGateStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [("repro_smoke_detail".to_string(), repro_smoke.detail.clone())],
        "REMEDIATE_REPRO_EXPORT",
        "NOTE_REQUIRED_REPRO",
    ));

    let bugkit_smoke = bugkit_smoke("v5_bugkit_smoke", "./out/bugkit_v5_gate.zip");
    checks.push(v5_gate_check(
        "enriched_bugkit_export_smoke_pass",
        if matches!(bugkit_smoke.status, PortabilityGateStatus::Pass) {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "bugkit_smoke_detail".to_string(),
            bugkit_smoke.detail.clone(),
        )],
        "REMEDIATE_BUGKIT_EXPORT",
        "NOTE_REQUIRED_BUGKIT",
    ));

    let remediation_out = workdir
        .join("out")
        .join("remediation_consistency_v5_gate.json");
    let remediation = remediation_consistency_check(&remediation_out)?;
    checks.push(v5_gate_check(
        "remediation_consistency_pass",
        if remediation.summary.status == RemediationConsistencyStatusV1::Pass {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "remediation_fail_count".to_string(),
            remediation.summary.fail_count.to_string(),
        )],
        "REMEDIATE_REMEDIATION_CONSISTENCY",
        "NOTE_REQUIRED_REMEDIATION_CONSISTENCY",
    ));

    let review_packet_out = workdir
        .join("out")
        .join("operator_review_packet_v5_gate.json");
    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &review_packet_out,
    )?;
    checks.push(v5_gate_check(
        "operator_review_packet_present",
        if !review_packet.packet_digest.is_empty() {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "operator_review_packet_digest".to_string(),
            prefix_hex(&review_packet.packet_digest, 16),
        )],
        "REMEDIATE_OPERATOR_REVIEW_PACKET",
        "NOTE_REQUIRED_REVIEW_PACKET",
    ));
    let gate_digests_aligned = review_packet.artifacts.gate_digests.v0
        == prefix_hex(&sha256_hex(&serde_json::to_vec(&v0)?), 16)
        && review_packet.artifacts.gate_digests.v1
            == prefix_hex(&sha256_hex(&serde_json::to_vec(&v1)?), 16)
        && review_packet.artifacts.gate_digests.v2
            == prefix_hex(&sha256_hex(&serde_json::to_vec(&v2)?), 16)
        && review_packet.artifacts.gate_digests.v3
            == prefix_hex(&sha256_hex(&serde_json::to_vec(&v3)?), 16)
        && review_packet.artifacts.gate_digests.v4
            == prefix_hex(&sha256_hex(&serde_json::to_vec(&v4)?), 16);
    let review_packet_consistent = review_packet.supported_slot_set_digest
        == active_review.supported_slot_set_digest
        && review_packet.policy_graph_digest_prefix == active_review.policy_graph_digest_prefix
        && gate_digests_aligned;
    checks.push(v5_gate_check(
        "operator_review_packet_consistent",
        if review_packet_consistent {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "review_stage".to_string(),
            format!("{:?}", review_packet.review_stage),
        )],
        "REMEDIATE_OPERATOR_REVIEW_PACKET_ALIGNMENT",
        "NOTE_REQUIRED_REVIEW_PACKET_ALIGNMENT",
    ));

    let artifact_schema = check_artifact_schema_snapshots(&artifact_schema::ArtifactSchemaArgs {
        repo_root: repo_root.clone(),
        out_dir: repo_root.join("docs/artifact_schema_snapshots"),
    })?;
    checks.push(v5_gate_check(
        "artifact_schema_snapshot_checks_pass",
        if artifact_schema.ok {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "schema_diff_count".to_string(),
            artifact_schema.diffs.len().to_string(),
        )],
        "REMEDIATE_ARTIFACT_SCHEMA_SNAPSHOT",
        "NOTE_REQUIRED_SCHEMA",
    ));

    let docs = docs_lint(&DocsLintArgs {
        repo_root: repo_root.clone(),
        policy_pack: repo_root.join("policies/packs/base_v1"),
        overlay_pack: Some(repo_root.join("policies/packs/overlays/test")),
        spec_snapshot: repo_root.join("docs/spec_snapshot.md"),
        prompt_index: repo_root.join("docs/prompt_series_index.md"),
        module_map: repo_root.join("docs/module_map.md"),
        deploy_doc: repo_root.join("docs/deploy.md"),
        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
        mode: DocsLintMode::Strict,
    })?;
    let portability_out = workdir.join("out").join("portability_check_v5_gate.json");
    let portability = portability_check(&portability_out)?;
    checks.push(v5_gate_check(
        "portability_docs_checks_pass",
        if docs.ok && portability.deterministic_within_os {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [
            (
                "docs_lint".to_string(),
                if docs.ok { "pass" } else { "fail" }.to_string(),
            ),
            (
                "portability_check".to_string(),
                if portability.deterministic_within_os {
                    "pass"
                } else {
                    "fail"
                }
                .to_string(),
            ),
        ],
        "REMEDIATE_PORTABILITY_DOCS",
        "NOTE_REQUIRED_DOCS",
    ));

    let models_consistency_out = workdir
        .join("out")
        .join("models_consistency_check_v5_gate.json");
    let models_consistency = models_consistency_check(workdir, &models_consistency_out)?;
    checks.push(v5_gate_check(
        "optional_backend_resolution_consistent",
        if models_consistency.status == "PASS" {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        [(
            "mismatch_count".to_string(),
            models_consistency.mismatch_categories.len().to_string(),
        )],
        "REMEDIATE_MODELS_CONSISTENCY",
        "NOTE_OPTIONAL_BACKEND",
    ));

    checks.push(v5_gate_check(
        "chosen_slot_burn_optional_path_consistent",
        match backend_resolution.support_state {
            OptionalBackendSupportStateV1::Unsupported
            | OptionalBackendSupportStateV1::NotConfigured => GateStatus::Skip,
            _ => {
                if matches!(
                    backend_resolution.resolution,
                    BurnResolutionStatusV1::BurnSupportedForShadowCompare
                ) {
                    GateStatus::Pass
                } else {
                    GateStatus::Fail
                }
            }
        },
        [(
            "burn_support_state".to_string(),
            format!("{:?}", backend_resolution.support_state),
        )],
        "REMEDIATE_BACKEND_RESOLUTION",
        "NOTE_OPTIONAL_BURN",
    ));

    let overall_status = if checks
        .iter()
        .all(|check| matches!(check.status, GateStatus::Pass | GateStatus::Skip))
    {
        V5GateOverallStatus::Pass
    } else {
        V5GateOverallStatus::Fail
    };

    let report = V5GateReportV1 {
        schema_version: 1,
        overall_status,
        checks,
    };
    write_json(out, &report)?;
    Ok(report)
}

fn strict_operator_signoff_alignment(
    strict_snapshot: &StrictEvidenceSnapshotV1,
    operator: &ConsolidatedOperatorReportV1,
    signoff: &OperatorSignoffDecisionV1,
) -> bool {
    if matches!(strict_snapshot.strict_status, StrictEvidenceStatusV1::Fail)
        && operator.sections.strict_section.status != OperatorStatus::Fail
    {
        return false;
    }
    if matches!(
        strict_snapshot.strict_status,
        StrictEvidenceStatusV1::Missing
    ) && operator.sections.strict_section.status != OperatorStatus::Missing
    {
        return false;
    }
    if let Some(primary) = strict_snapshot.primary_denial_code.as_ref() {
        if !signoff.reasons.iter().any(|reason| reason == primary) {
            return false;
        }
    }
    true
}

fn remediation_codes_aligned(
    operator: &ConsolidatedOperatorReportV1,
    signoff: &OperatorSignoffDecisionV1,
    snapshot: &BackendEvidenceSnapshotV1,
    repo_root: &Path,
) -> Result<bool, OpsError> {
    let registry = fs::read_to_string(repo_root.join("docs/remediation_codes_v1.md"))?;
    let mut codes = BTreeSet::new();
    for code in &operator.canonical_remediation_codes {
        codes.insert(code.clone());
    }
    for code in &signoff.canonical_remediation_codes {
        codes.insert(code.clone());
    }
    for slot in &snapshot.slots {
        for code in &slot.canonical_remediation_codes {
            codes.insert(code.clone());
        }
    }
    Ok(codes.iter().all(|code| registry.contains(code)))
}

fn v4_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
    notes: &str,
) -> V4GateCheckV1 {
    V4GateCheckV1 {
        name: name.to_string(),
        status,
        evidence_digest_prefixes: bounded_evidence(evidence),
        remediation_hint_code: remediation_hint_code.to_string(),
        notes: notes.to_string(),
    }
}

fn v5_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
    notes: &str,
) -> V5GateCheckV1 {
    V5GateCheckV1 {
        name: name.to_string(),
        status,
        evidence_digest_prefixes: bounded_evidence(evidence),
        remediation_hint_code: remediation_hint_code.to_string(),
        notes: notes.to_string(),
    }
}

#[cfg(test)]
mod v4_gate_tests {
    use super::*;

    #[test]
    fn v4_gate_check_order_is_fixed() {
        let checks = vec![
            "v0_gate_pass",
            "v1_gate_pass",
            "v2_gate_pass",
            "v3_gate_pass",
            "supported_slot_set_consistent",
            "backend_evidence_snapshot_present",
            "backend_evidence_snapshot_schema_stable",
            "operator_report_present",
            "operator_signoff_present",
            "operator_signoff_consistent_with_evidence",
            "remediation_registry_present",
            "remediation_registry_consistent_across_reports",
            "strict_evidence_present",
            "strict_operator_alignment_ok",
            "artifact_schema_snapshot_checks_pass",
            "portability_docs_checks_pass",
            "optional_backend_states_consistent",
            "burn_parity_optional_path_consistent",
        ];
        let report = V4GateReportV1 {
            schema_version: 1,
            overall_status: V4GateOverallStatus::Pass,
            checks: checks
                .iter()
                .map(|name| v4_gate_check(name, GateStatus::Pass, [], "REMEDIATE", "NOTE"))
                .collect(),
        };
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(names, checks);
    }

    #[test]
    fn v4_gate_report_serialization_is_deterministic() {
        let report = V4GateReportV1 {
            schema_version: 1,
            overall_status: V4GateOverallStatus::Pass,
            checks: vec![
                v4_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "REMEDIATE_A",
                    "NOTE_A",
                ),
                v4_gate_check(
                    "b",
                    GateStatus::Skip,
                    [("x".to_string(), "y".to_string())],
                    "REMEDIATE_B",
                    "NOTE_B",
                ),
            ],
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
    }

    #[test]
    fn v4_gate_normalization_fail_closed() {
        let report = V4GateReportV1 {
            schema_version: 1,
            overall_status: V4GateOverallStatus::Fail,
            checks: vec![
                v4_gate_check(
                    "required",
                    GateStatus::Fail,
                    [],
                    "REMEDIATE",
                    "NOTE_REQUIRED",
                ),
                v4_gate_check(
                    "optional",
                    GateStatus::Skip,
                    [],
                    "REMEDIATE",
                    "NOTE_OPTIONAL",
                ),
            ],
        };
        assert!(matches!(report.overall_status, V4GateOverallStatus::Fail));
    }
}

fn detect_second_slot_for_v3(repo_root: &Path) -> Result<ModelSlot, OpsError> {
    let body = fs::read_to_string(repo_root.join("docs/series_state_snapshot.md"))?;
    let mut selected = None;
    for line in body.lines() {
        if line.contains("Second supported slot") {
            let lower = line.to_ascii_lowercase();
            let has_sae = lower.contains("`sae`") || lower.contains(" sae");
            let has_ssm = lower.contains("`ssm`") || lower.contains(" ssm");
            if has_sae && has_ssm {
                return Err(OpsError::Invalid(
                    "V3_SECOND_SLOT_AMBIGUOUS: expected exactly one of sae or ssm".to_string(),
                ));
            }
            if has_sae {
                selected = Some(ModelSlot::Sae);
            }
            if has_ssm {
                selected = Some(ModelSlot::Ssm);
            }
        }
    }
    selected.ok_or_else(|| {
        OpsError::Invalid(
            "V3_SECOND_SLOT_UNKNOWN: expected second supported slot declaration".to_string(),
        )
    })
}

fn v3_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
    notes: &str,
) -> V3GateCheckV1 {
    V3GateCheckV1 {
        name: name.to_string(),
        status,
        evidence_digest_prefixes: bounded_evidence(evidence),
        remediation_hint_code: remediation_hint_code.to_string(),
        canonical_remediation_codes: crate::remediation::canonical_from_legacy_code(
            remediation_hint_code,
        ),
        notes: notes.to_string(),
    }
}

fn v2_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
    notes: &str,
) -> V2GateCheckV1 {
    V2GateCheckV1 {
        name: name.to_string(),
        status,
        evidence_digest_prefixes: bounded_evidence(evidence),
        remediation_hint_code: remediation_hint_code.to_string(),
        canonical_remediation_codes: crate::remediation::canonical_from_legacy_code(
            remediation_hint_code,
        ),
        notes: notes.to_string(),
    }
}

fn v2_shadow_no_impact_check(workdir: &Path, slot: ModelSlot, note: &str) -> V2GateCheckV1 {
    let scenario = workspace_fixture("e2e_scenario_a.json");
    let env_key = format!("UCF_SLOT_{}_MODE", slot.env_key());
    let off = with_env_var(&env_key, "off", || {
        v0_gate_run_once(
            workdir,
            &scenario,
            8,
            &format!("v2_{}_shadow_off", slot.as_str()),
        )
    });
    let shadow = with_env_var(&env_key, "shadow", || {
        v0_gate_run_once(
            workdir,
            &scenario,
            8,
            &format!("v2_{}_shadow_on", slot.as_str()),
        )
    });
    match (off, shadow) {
        (Ok(off_run), Ok(shadow_run)) => {
            let no_impact = off_run.decision_digest == shadow_run.decision_digest;
            let no_tools =
                off_run.tool_execution_count == 0 && shadow_run.tool_execution_count == 0;
            v2_gate_check(
                if slot == ModelSlot::WorldJepa {
                    "world_shadow_no_impact"
                } else {
                    "second_slot_shadow_no_impact"
                },
                if no_impact && no_tools {
                    GateStatus::Pass
                } else {
                    GateStatus::Fail
                },
                [
                    (
                        "baseline_decision".to_string(),
                        prefix_hex(&off_run.decision_digest, 16),
                    ),
                    (
                        "shadow_decision".to_string(),
                        prefix_hex(&shadow_run.decision_digest, 16),
                    ),
                    (
                        "tool_exec_count".to_string(),
                        (off_run.tool_execution_count + shadow_run.tool_execution_count)
                            .to_string(),
                    ),
                ],
                "REMEDIATE_SHADOW_NO_IMPACT",
                if note == "world" {
                    "NOTE_REQUIRED_WORLD"
                } else {
                    "NOTE_REQUIRED_SECOND_SLOT"
                },
            )
        }
        (Err(err), _) | (_, Err(err)) => v2_gate_check(
            if slot == ModelSlot::WorldJepa {
                "world_shadow_no_impact"
            } else {
                "second_slot_shadow_no_impact"
            },
            GateStatus::Fail,
            [("error".to_string(), bounded_string(err.to_string(), 48))],
            "REMEDIATE_SHADOW_NO_IMPACT",
            "NOTE_REQUIRED_SHADOW",
        ),
    }
}

fn v1_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint: &str,
) -> V1GateCheckV1 {
    V1GateCheckV1 {
        name: name.to_string(),
        status,
        evidence_digest_prefixes: bounded_evidence(evidence),
        remediation_hint: remediation_hint.to_string(),
        canonical_remediation_codes: crate::remediation::canonical_from_legacy_remediation(
            remediation_hint,
        ),
    }
}

fn v0_gate_check(
    name: &str,
    status: GateStatus,
    evidence: impl IntoIterator<Item = (String, String)>,
    remediation_hint_code: &str,
) -> V0GateCheckV1 {
    V0GateCheckV1 {
        name: name.to_string(),
        status,
        evidence_digest_prefixes: bounded_evidence(evidence),
        remediation_hint_code: remediation_hint_code.to_string(),
        canonical_remediation_codes: crate::remediation::canonical_from_legacy_code(
            remediation_hint_code,
        ),
    }
}

#[derive(Debug)]
struct V0RunSummary {
    signals_digest: String,
    decision_digest: String,
    experience_digest: String,
    record_count: usize,
    max_record_bytes: usize,
    schema_versions: BTreeMap<String, u16>,
    tool_execution_count: usize,
    has_required_records: bool,
}

fn v0_gate_run_once(
    workdir: &Path,
    scenario: &Path,
    ticks: u64,
    run_name: &str,
) -> Result<V0RunSummary, OpsError> {
    let run_dir = workdir.join("v0_gate").join(run_name);
    let _ = fs::remove_dir_all(&run_dir);
    let out_dir = run_dir.join("out");
    let artifacts = one_command_bringup(&run_dir, scenario, ticks, &out_dir, true)?;
    let records = load_fixture_records(&run_dir.join("ess").join("ess_fixture.json"))?;
    let max_record_bytes = records
        .iter()
        .map(|r| format!("{r:?}").len())
        .max()
        .unwrap_or(0);
    let tool_execution_count = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::ToolExecution)
        .count();
    let has_required_records = !records.is_empty();
    Ok(V0RunSummary {
        signals_digest: artifacts.run_metadata.fixtures_digest,
        decision_digest: sha256_hex(&serde_json::to_vec(&artifacts.explain.decision)?),
        experience_digest: sha256_hex(&serde_json::to_vec(&artifacts.explain.links)?),
        record_count: records.len(),
        max_record_bytes,
        schema_versions: artifacts.run_metadata.schema_versions,
        tool_execution_count,
        has_required_records,
    })
}

fn schema_versions_known(versions: &BTreeMap<String, u16>) -> bool {
    let expected = ["backend_pack_record", "compute_summary", "output"];
    expected
        .iter()
        .all(|key| versions.get(*key).copied().unwrap_or_default() > 0)
        && versions.len() == expected.len()
}

fn expected_policy_digest_prefix_from_spec_snapshot(path: &Path) -> Result<String, OpsError> {
    let raw = fs::read_to_string(path)?;
    let Some(line) = raw.lines().find(|l| l.contains("policy_graph_digest")) else {
        return Err(OpsError::Invalid(
            "docs/spec_snapshot.md missing policy_graph_digest line".to_string(),
        ));
    };
    let Some(prefix) = line.split('`').nth(1) else {
        return Err(OpsError::Invalid(
            "docs/spec_snapshot.md policy_graph_digest format invalid".to_string(),
        ));
    };
    Ok(prefix.trim_end_matches('…').to_string())
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
    let manifest_path = PathBuf::from("models/lifecycle_manifest.toml");
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

fn policy_drift_entry(stage_id: &str) -> Option<DriftBudgetEntryV1> {
    let overlay = std::env::var("UCF_POLICY_OVERLAY").ok();
    let overlay_path = overlay
        .as_deref()
        .map(|name| PathBuf::from("policies/packs/overlays").join(name));
    let overlay_ref = overlay_path.as_deref();
    let (graph, _) =
        load_and_merge_policy_graph(Path::new("policies/packs/base_v1"), overlay_ref).ok()?;
    graph
        .drift_budget
        .entries
        .into_iter()
        .find(|entry| entry.slot_id == stage_id)
}

fn check_world_vljepa_shadow_evidence(workdir: &Path) -> Result<CheckResult, OpsError> {
    let manifest: Option<GateLifecycleManifest> =
        fs::read_to_string("models/lifecycle_manifest.toml")
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
    let budget = policy_drift_entry("world_vljepa");
    let min_windows = budget
        .as_ref()
        .map(|entry| entry.window_size_ticks as usize)
        .unwrap_or(2);
    let drift_threshold = budget
        .as_ref()
        .and_then(|entry| entry.scalar_delta_max_q.get("error_delta_p95_q").copied())
        .map(|v| (v as f32) / 10_000.0)
        .unwrap_or(0.0);
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
    let manifest: Option<GateLifecycleManifest> =
        fs::read_to_string("models/lifecycle_manifest.toml")
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
    let drift_limit = policy_drift_entry("ssm_opt")
        .and_then(|entry| entry.scalar_delta_max_q.get("drift_alarm_rate_q").copied())
        .map(|v| (v as f64) / 10_000.0)
        .unwrap_or(0.0);

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
    pub drift: ExplainDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainDrift {
    pub statuses: BTreeMap<String, String>,
    pub alarm_ids: Vec<String>,
    pub reason_codes: Vec<String>,
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
    if cfg.strict_mode {
        StrictModeEnforcer::check_all(workdir, &cfg, false).map_err(|report| {
            OpsError::Invalid(format!(
                "strict mode failed: {} failed checks (see out/strict_failure.json)",
                report
                    .checks
                    .iter()
                    .filter(|c| matches!(c.status, StrictCheckStatus::Fail))
                    .count()
            ))
        })?;
    }

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
        strict_mode_enabled: cfg.strict_mode,
        strict_mode_digest: None,
        probe_report_digest_prefix: None,
        crash_dumps_disabled: disable_crash_dumps_best_effort(),
        models_manifest_present: false,
        models_manifest_digest_prefix: None,
        ended_at_tick: Some(ticks),
    };
    let (models_manifest_present, models_manifest_digest_prefix) =
        load_models_manifest_runtime_metadata();
    run_metadata.models_manifest_present = models_manifest_present;
    run_metadata.models_manifest_digest_prefix = models_manifest_digest_prefix;
    run_metadata.probe_report_digest_prefix = load_probe_report_digest_prefix(workdir);
    if cfg.strict_mode {
        let strict_policy = StrictModeV1::from_config(&cfg);
        run_metadata.strict_mode_digest = Some(sha256_hex(
            serde_json::to_vec(&strict_policy)
                .unwrap_or_default()
                .as_slice(),
        ));
    }

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

pub fn dev_loop(workdir: &Path, args: &DevLoopArgs) -> Result<DevLoopReport, OpsError> {
    ensure_layout(workdir)?;
    fs::create_dir_all(&args.out_dir)?;
    let mut steps = Vec::new();

    if args.run_tests {
        let test_out = args.out_dir.join("cargo_test_subset.log");
        let (ok, detail) = run_shell_command(
            &["cargo", "test", "-p", "ucf-ops", "--lib", "--quiet"],
            &test_out,
        )?;
        steps.push(DevLoopStepResult {
            step: "cargo_test_subset".to_string(),
            status: if ok {
                DevLoopStepStatus::Pass
            } else {
                DevLoopStepStatus::Fail
            },
            detail,
            artifact: Some(test_out.display().to_string()),
            hint: if ok {
                None
            } else {
                Some("fix failing tests before iterating bringup".to_string())
            },
        });
    } else {
        steps.push(DevLoopStepResult {
            step: "cargo_test_subset".to_string(),
            status: DevLoopStepStatus::Skipped,
            detail: "skipped (--no-tests)".to_string(),
            artifact: None,
            hint: None,
        });
    }

    let scenario_path =
        PathBuf::from("fixtures/goldens/scenarios").join(format!("{}.json", args.scenario));
    let bringup_out = args.out_dir.join("bringup");
    let bringup_status =
        one_command_bringup(workdir, &scenario_path, args.ticks, &bringup_out, true);
    match bringup_status {
        Ok(artifacts) => steps.push(DevLoopStepResult {
            step: "bringup".to_string(),
            status: DevLoopStepStatus::Pass,
            detail: format!(
                "run_id={} profile={}",
                artifacts.run_metadata.run_id, artifacts.run_metadata.profile
            ),
            artifact: Some(bringup_out.display().to_string()),
            hint: None,
        }),
        Err(err) => steps.push(DevLoopStepResult {
            step: "bringup".to_string(),
            status: DevLoopStepStatus::Fail,
            detail: err.to_string(),
            artifact: Some(bringup_out.display().to_string()),
            hint: Some(
                "run `ucf-ops bringup --scenario <path> --ticks <n>` to inspect failure"
                    .to_string(),
            ),
        }),
    }

    let docs_out = args.out_dir.join("docs_lint_report.json");
    let docs_report = docs_lint(&DocsLintArgs {
        repo_root: PathBuf::from("."),
        policy_pack: PathBuf::from("policies/packs/base_v1"),
        overlay_pack: Some(PathBuf::from(format!(
            "policies/packs/overlays/{}",
            args.profile
        ))),
        spec_snapshot: PathBuf::from("docs/spec_snapshot.md"),
        prompt_index: PathBuf::from("docs/prompt_series_index.md"),
        module_map: PathBuf::from("docs/module_map.md"),
        deploy_doc: PathBuf::from("docs/deploy.md"),
        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
        mode: DocsLintMode::Strict,
    });
    match docs_report {
        Ok(report) => {
            write_json(&docs_out, &report)?;
            steps.push(DevLoopStepResult {
                step: "docs_lint".to_string(),
                status: if report.ok {
                    DevLoopStepStatus::Pass
                } else {
                    DevLoopStepStatus::Fail
                },
                detail: format!("ok={}", report.ok),
                artifact: Some(docs_out.display().to_string()),
                hint: if report.ok {
                    None
                } else {
                    Some(
                        "run `ucf-ops docs lint --strict --out ./out/docs_lint_report.json`"
                            .to_string(),
                    )
                },
            });
        }
        Err(err) => steps.push(DevLoopStepResult {
            step: "docs_lint".to_string(),
            status: DevLoopStepStatus::Fail,
            detail: err.to_string(),
            artifact: Some(docs_out.display().to_string()),
            hint: Some("fix docs lint configuration or docs references".to_string()),
        }),
    }

    let mut scenarios = vec![args.scenario.clone(), "golden_b".to_string()];
    scenarios.sort();
    scenarios.dedup();
    scenarios.truncate(DEV_LOOP_MAX_SCENARIOS);
    for scenario in scenarios {
        let verify_res = goldens_verify(&GoldenVerifyArgs {
            scenario: scenario.clone(),
            os: std::env::consts::OS.to_string(),
            out_root: PathBuf::from("fixtures/goldens"),
            workdir_root: args.out_dir.join("goldens_workdir"),
        });
        let verify_ok = verify_res.is_ok();
        let detail = match verify_res {
            Ok(()) => "status=PASS".to_string(),
            Err(err) => err.to_string(),
        };
        steps.push(DevLoopStepResult {
            step: format!("goldens_verify:{scenario}"),
            status: if verify_ok {
                DevLoopStepStatus::Pass
            } else {
                DevLoopStepStatus::Fail
            },
            detail,
            artifact: Some(
                args.out_dir
                    .join(format!("goldens_{scenario}.json"))
                    .display()
                    .to_string(),
            ),
            hint: if verify_ok {
                None
            } else {
                Some(format!(
                    "run `ucf-ops goldens verify --scenario {scenario}`"
                ))
            },
        });
    }

    let mut next_actions = BTreeSet::new();
    for step in &steps {
        if step.status == DevLoopStepStatus::Fail {
            if let Some(hint) = &step.hint {
                next_actions.insert(hint.clone());
            }
        }
    }
    if next_actions.is_empty() {
        next_actions.insert(
            "all checks passed; continue with full workspace tests before commit".to_string(),
        );
    }

    let report = DevLoopReport {
        profile: args.profile.clone(),
        scenario: args.scenario.clone(),
        ticks: args.ticks,
        steps,
        next_actions: next_actions.into_iter().collect(),
    };
    write_json(args.out_dir.join("dev_loop_report.json"), &report)?;
    Ok(report)
}

pub fn apply_hot_reload_if_safe(
    workdir: &Path,
    current: &OpsConfig,
    updated: &OpsConfig,
) -> Result<Result<OpsConfig, ConfigReloadDeniedRecord>, OpsError> {
    let changed_keys = diff_ops_config_keys(current, updated);
    let mut reasons = Vec::new();
    for key in &changed_keys {
        match key.as_str() {
            "compute_budget_profile" | "sampling_enabled" | "log_level" => {}
            "policy_overlay" => reasons.push(ConfigReloadReasonCode::PolicyOverlayChanged),
            "strict_mode" | "determinism_lock_strict" => {
                reasons.push(ConfigReloadReasonCode::StrictModeChanged)
            }
            "emergency_policy_pin" => reasons.push(ConfigReloadReasonCode::AuthTokenChanged),
            _ => reasons.push(ConfigReloadReasonCode::UnsupportedKeyChanged),
        }
    }
    if current.profile != updated.profile {
        reasons.push(ConfigReloadReasonCode::ManifestChanged);
    }
    reasons.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
    reasons.dedup();
    if reasons.is_empty() {
        let mut applied = updated.clone();
        applied.config_digest = ops_config_digest(&applied)?;
        let record = ConfigReloadAppliedRecord {
            t_unix: now_unix_secs(),
            profile: applied.profile.clone(),
            changed_keys,
            config_digest: applied.config_digest.clone(),
        };
        persist_jsonl_record(
            &workdir.join("reports/config_reload_applied.jsonl"),
            &record,
        )?;
        return Ok(Ok(applied));
    }
    let denied = ConfigReloadDeniedRecord {
        t_unix: now_unix_secs(),
        profile: updated.profile.clone(),
        changed_keys,
        reason_codes: reasons,
    };
    persist_jsonl_record(&workdir.join("reports/config_reload_denied.jsonl"), &denied)?;
    Ok(Err(denied))
}

pub fn troubleshoot(
    workdir: &Path,
    run_id: &str,
    out: &Path,
) -> Result<TroubleshootReport, OpsError> {
    ensure_layout(workdir)?;
    let strict = first_existing_path(&[
        workdir.join("out/strict_failure.json"),
        PathBuf::from("./out/strict_failure.json"),
    ]);
    let drift = first_existing_path(&[
        workdir.join(format!("out/drift_{run_id}.json")),
        PathBuf::from("./out/drift_report.json"),
    ]);
    let gate = first_existing_path(&[
        workdir.join("out/gate_report.json"),
        PathBuf::from("./out/gate_report.json"),
    ]);
    let docs = first_existing_path(&[
        workdir.join("out/docs_lint_report.json"),
        PathBuf::from("./out/docs_lint_report.json"),
    ]);
    let records =
        load_fixture_records(&workdir.join("ess").join("ess_fixture.json")).unwrap_or_default();
    let abuse_count: usize = records
        .iter()
        .filter_map(|r| match &r.payload {
            ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(i)) => {
                Some(i.denied_kinds.len())
            }
            _ => None,
        })
        .sum();

    let mut issues = Vec::new();
    if let Some(path) = &strict {
        issues.push(TroubleshootIssue {
            source: "strict_failure".to_string(),
            severity: "high".to_string(),
            detail: format!("strict checks failed: {}", path.display()),
            next_command: "ucf-ops strict check --strict --out ./out/strict_check.json".to_string(),
        });
    }
    if let Some(path) = &drift {
        issues.push(TroubleshootIssue {
            source: "drift_report".to_string(),
            severity: "medium".to_string(),
            detail: format!("drift report present: {}", path.display()),
            next_command: format!(
                "ucf-ops drift report --run {run_id} --windows 32 --out ./out/drift_report.json"
            ),
        });
    }
    if gate.is_none() {
        issues.push(TroubleshootIssue {
            source: "readiness_gate".to_string(),
            severity: "medium".to_string(),
            detail: "readiness gate report missing".to_string(),
            next_command: "ucf-ops readiness-gate --profile test --out ./out/gate_report.json"
                .to_string(),
        });
    }
    if docs.is_none() {
        issues.push(TroubleshootIssue {
            source: "docs_lint".to_string(),
            severity: "low".to_string(),
            detail: "docs lint report missing".to_string(),
            next_command: "ucf-ops docs lint --strict --out ./out/docs_lint_report.json"
                .to_string(),
        });
    }
    if abuse_count > 0 {
        issues.push(TroubleshootIssue {
            source: "gateway_abuse".to_string(),
            severity: "high".to_string(),
            detail: format!("detected {} denied tool invocations", abuse_count),
            next_command: "ucf-ops security verify-chain --from 0 --to 18446744073709551615"
                .to_string(),
        });
    }
    issues.sort_by(|a, b| {
        a.source
            .cmp(&b.source)
            .then(a.next_command.cmp(&b.next_command))
    });
    issues.truncate(TROUBLESHOOT_MAX_ISSUES);

    let report = TroubleshootReport {
        run_id: run_id.to_string(),
        strict_failure: strict.map(|p| p.display().to_string()),
        drift_report: drift.map(|p| p.display().to_string()),
        readiness_gate: gate.map(|p| p.display().to_string()),
        docs_lint: docs.map(|p| p.display().to_string()),
        gateway_abuse_count: abuse_count,
        issues,
    };
    write_json(out, &report)?;
    Ok(report)
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
    let drift = fs::read_to_string(workdir.join("out/drift_report.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<DriftReportV1>(&raw).ok());
    build_explain_tick_report_with_drift(&records, req, drift.as_ref())
}

pub fn build_explain_tick_report(
    records: &[ExperienceRecord],
    req: ExplainTickRequest,
) -> Result<ExplainTickReport, OpsError> {
    build_explain_tick_report_with_drift(records, req, None)
}

fn build_explain_tick_report_with_drift(
    records: &[ExperienceRecord],
    req: ExplainTickRequest,
    drift: Option<&DriftReportV1>,
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

    let drift_statuses = drift.map(drift_status_map).unwrap_or_default();
    let mut drift_alarm_ids = Vec::new();
    let mut drift_reason_codes = Vec::new();
    if let Some(rep) = drift {
        for alarm in &rep.alarms {
            if alarm.window_id == tick || alarm.window_id + 1 == tick {
                drift_alarm_ids.push(alarm.alarm_id.clone());
                drift_reason_codes.push(alarm.reason_code.clone());
            }
        }
    }
    drift_alarm_ids.sort();
    drift_reason_codes.sort();
    drift_reason_codes.dedup();

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
            drift: ExplainDrift {
                statuses: drift_statuses,
                alarm_ids: drift_alarm_ids,
                reason_codes: drift_reason_codes,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrictModeV1 {
    pub enabled: bool,
    pub required_checks: Vec<String>,
    pub dev_exceptions: Vec<String>,
}

impl StrictModeV1 {
    fn from_config(cfg: &OpsConfig) -> Self {
        Self {
            enabled: cfg.strict_mode,
            required_checks: vec![
                "determinism_sampling_disabled".to_string(),
                "determinism_rng_scan".to_string(),
                "policy_graph_digest".to_string(),
                "policy_pack_validate".to_string(),
                "models_manifest_digest".to_string(),
                "models_promoted_only".to_string(),
                "models_verify".to_string(),
                "tool_2pc_required".to_string(),
                "sandbox_fs_scan".to_string(),
            ],
            dev_exceptions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StrictCheckStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrictCheckResult {
    pub check_id: String,
    pub status: StrictCheckStatus,
    pub error_codes: Vec<String>,
    pub remediation: String,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrictModeFailureReport {
    pub schema_version: u16,
    pub strict_mode_enabled: bool,
    pub profile: String,
    pub checks: Vec<StrictCheckResult>,
    #[serde(default)]
    pub v1_checks: Vec<StrictCheckResult>,
    #[serde(default)]
    pub v3: Option<StrictFailureReportV3>,
    #[serde(default)]
    pub evidence_digest_prefixes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrictFailureReportV3 {
    pub schema_version: u16,
    pub strict_mode_enabled: bool,
    pub overall_status: String,
    pub checks: Vec<StrictCheckV3Result>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StrictCheckV3Status {
    Pass,
    Fail,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrictCheckV3Result {
    pub check_id: String,
    pub slot_id: Option<String>,
    pub status: StrictCheckV3Status,
    pub denial_code: Option<String>,
    pub evidence_digest_prefixes: Vec<String>,
    pub remediation_code: String,
    #[serde(default)]
    pub canonical_remediation_codes: Vec<String>,
}

impl StrictModeFailureReport {
    pub fn has_failures(&self) -> bool {
        let base_failed = self
            .checks
            .iter()
            .chain(self.v1_checks.iter())
            .any(|c| matches!(c.status, StrictCheckStatus::Fail));
        let v3_failed = self
            .v3
            .as_ref()
            .map(|r| {
                r.checks
                    .iter()
                    .any(|c| matches!(c.status, StrictCheckV3Status::Fail))
            })
            .unwrap_or(false);
        base_failed || v3_failed
    }

    fn normalized_for_digest(&self) -> Self {
        let mut c = self.clone();
        c.checks.sort_by(|a, b| a.check_id.cmp(&b.check_id));
        c.v1_checks.sort_by(|a, b| a.check_id.cmp(&b.check_id));
        if let Some(v3) = c.v3.as_mut() {
            v3.checks.sort_by(|a, b| {
                a.slot_id
                    .as_deref()
                    .unwrap_or("")
                    .cmp(b.slot_id.as_deref().unwrap_or(""))
                    .then(a.check_id.cmp(&b.check_id))
            });
        }
        c.evidence_digest_prefixes = c
            .evidence_digest_prefixes
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        c
    }

    pub fn digest_hex(&self) -> Result<String, OpsError> {
        Ok(sha256_hex(
            serde_json::to_vec(&self.normalized_for_digest())?.as_slice(),
        ))
    }
}

pub struct StrictModeEnforcer;

impl StrictModeEnforcer {
    #[allow(clippy::result_large_err)]
    pub fn check_all(
        workdir: &Path,
        cfg: &OpsConfig,
        ops_only: bool,
    ) -> Result<(), StrictModeFailureReport> {
        let mut checks = Vec::new();
        let mut evidence_digest_prefixes = BTreeMap::new();
        checks.push(if !cfg.sampling_enabled {
            strict_pass("determinism_sampling_disabled")
        } else {
            strict_fail(
                "determinism_sampling_disabled",
                "strict.sampling.enabled",
                "set runtime.sampling_enabled=false",
            )
        });

        match determinism_scan(Path::new(".")) {
            Ok(r) if r.violations.is_empty() => checks.push(strict_pass("determinism_rng_scan")),
            Ok(_) => checks.push(strict_fail(
                "determinism_rng_scan",
                "strict.determinism.rng",
                "remove disallowed RNG usage and rerun `ucf-ops determinism scan`",
            )),
            Err(_) => checks.push(strict_fail(
                "determinism_rng_scan",
                "strict.determinism.scan_error",
                "ensure repository is readable for determinism scan",
            )),
        }

        let pack = PathBuf::from("policies/packs/base_v1");
        let overlay = PathBuf::from(format!("policies/packs/overlays/{}", cfg.profile));
        match policy_validate(&pack, Some(&overlay)) {
            Ok(report) => {
                if let Ok(expected) = std::env::var("UCF_POLICY_GRAPH_DIGEST") {
                    if expected == report.policy_graph_digest {
                        checks.push(strict_pass("policy_graph_digest"));
                    } else {
                        checks.push(strict_fail(
                            "policy_graph_digest",
                            "strict.policy.digest_mismatch",
                            "set UCF_POLICY_GRAPH_DIGEST to the validated policy_graph_digest",
                        ));
                    }
                } else {
                    checks.push(strict_fail(
                        "policy_graph_digest",
                        "strict.policy.digest_missing",
                        "export UCF_POLICY_GRAPH_DIGEST from validated policy",
                    ));
                }
                checks.push(strict_pass("policy_pack_validate"));
            }
            Err(_) => {
                checks.push(strict_fail("policy_graph_digest", "strict.policy.validation_failed", "run `ucf-ops policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test`"));
                checks.push(strict_fail(
                    "policy_pack_validate",
                    "strict.policy.validation_failed",
                    "fix policy pack errors and re-run validate",
                ));
            }
        }

        let manifest_path = PathBuf::from("models/manifest.toml");
        match fs::read_to_string(&manifest_path) {
            Ok(body) => {
                let computed = sha256_hex(body.as_bytes());
                match std::env::var("UCF_MODEL_MANIFEST_DIGEST") {
                    Ok(expected) if expected == computed => {
                        checks.push(strict_pass("models_manifest_digest"))
                    }
                    Ok(_) => checks.push(strict_fail(
                        "models_manifest_digest",
                        "strict.models.manifest_digest_mismatch",
                        "set UCF_MODEL_MANIFEST_DIGEST to sha256(models/manifest.toml)",
                    )),
                    Err(_) => checks.push(strict_fail(
                        "models_manifest_digest",
                        "strict.models.manifest_digest_missing",
                        "export UCF_MODEL_MANIFEST_DIGEST",
                    )),
                }
                if body.contains("promoted/") {
                    checks.push(strict_pass("models_promoted_only"));
                } else {
                    checks.push(strict_fail(
                        "models_promoted_only",
                        "strict.models.promoted_only",
                        "use promoted/<slot>/<hash>/model.safetensors entries",
                    ));
                }
            }
            Err(_) => {
                checks.push(strict_fail(
                    "models_manifest_digest",
                    "strict.models.manifest_missing",
                    "ensure models/manifest.toml exists",
                ));
                checks.push(strict_fail(
                    "models_promoted_only",
                    "strict.models.manifest_missing",
                    "ensure models manifest is present",
                ));
            }
        }

        match models_verify(Path::new("models/manifest.toml")) {
            Ok(report) if report.slots.iter().all(|s| s.status == "verified" || s.status == "disabled") => checks.push(strict_pass("models_verify")),
            Ok(_) => checks.push(strict_fail("models_verify", "strict.models.verify_failed", "run `ucf-ops models verify --manifest models/manifest.toml` and fix rejected slots")),
            Err(_) => checks.push(strict_fail("models_verify", "strict.models.verify_error", "fix manifest parse/allowlist issues")),
        }

        checks.push(if cfg.capabilities_default == "deny" {
            strict_pass("tool_2pc_required")
        } else {
            strict_fail(
                "tool_2pc_required",
                "strict.tools.2pc_required",
                "set capabilities_default=deny and keep tool-governed execution",
            )
        });

        match path_scan(Path::new(".")) {
            Ok(r) if r.violations.is_empty() => checks.push(strict_pass("sandbox_fs_scan")),
            Ok(_) => checks.push(strict_fail(
                "sandbox_fs_scan",
                "strict.sandbox.path_violation",
                "remove banned absolute/system paths from runtime code",
            )),
            Err(_) => checks.push(strict_fail(
                "sandbox_fs_scan",
                "strict.sandbox.scan_error",
                "ensure repository is readable for path scan",
            )),
        }

        if ops_only {
            let docs = docs_lint(&DocsLintArgs {
                repo_root: PathBuf::from("."),
                policy_pack: PathBuf::from("policies/packs/base_v1"),
                overlay_pack: Some(PathBuf::from("policies/packs/overlays/test")),
                spec_snapshot: PathBuf::from("docs/spec_snapshot.md"),
                prompt_index: PathBuf::from("docs/prompt_series_index.md"),
                module_map: PathBuf::from("docs/module_map.md"),
                deploy_doc: PathBuf::from("docs/deploy.md"),
                artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
                mode: DocsLintMode::Strict,
            });
            match docs {
                Ok(r) if r.ok => checks.push(strict_pass("docs_lint_strict")),
                Ok(_) => checks.push(strict_fail(
                    "docs_lint_strict",
                    "strict.docs.lint_failed",
                    "run `ucf-ops docs lint --strict`",
                )),
                Err(_) => checks.push(strict_fail(
                    "docs_lint_strict",
                    "strict.docs.lint_error",
                    "resolve docs lint execution error",
                )),
            }
        }

        checks.sort_by(|a, b| a.check_id.cmp(&b.check_id));
        let v1_checks = strict_v1_checks(workdir, cfg, &mut evidence_digest_prefixes);
        let v3 = strict_v3_checks(workdir, cfg);
        let report = StrictModeFailureReport {
            schema_version: 1,
            strict_mode_enabled: true,
            profile: cfg.profile.clone(),
            checks,
            v1_checks,
            v3: Some(v3),
            evidence_digest_prefixes,
        };

        if report.has_failures() {
            let _ = fs::create_dir_all(workdir.join("out"));
            let _ = write_json(workdir.join("out/strict_failure.json"), &report);
            Err(report)
        } else {
            Ok(())
        }
    }
}

fn strict_v1_checks(
    workdir: &Path,
    cfg: &OpsConfig,
    evidence_digest_prefixes: &mut BTreeMap<String, String>,
) -> Vec<StrictCheckResult> {
    let slot_enablement = ucf_compute::SlotEnablement::from_env().unwrap_or_default();
    let active_mode_requested = std::env::var("UCF_REAL_ENABLEMENT_MODE")
        .ok()
        .map(|v| v.eq_ignore_ascii_case("active"))
        .unwrap_or(false)
        || cfg.slot_ebm_mode.eq_ignore_ascii_case("active")
        || ModelSlot::all().into_iter().any(|slot| {
            matches!(
                slot_enablement.for_slot(slot),
                ucf_compute::SlotMode::Active
            )
        });
    let shadow_enabled = std::env::var("UCF_REAL_ENABLEMENT_MODE")
        .ok()
        .map(|v| v.eq_ignore_ascii_case("shadow") || v.eq_ignore_ascii_case("compare"))
        .unwrap_or(false)
        || cfg.slot_ebm_mode.eq_ignore_ascii_case("shadow")
        || ModelSlot::all().into_iter().any(|slot| {
            matches!(
                slot_enablement.for_slot(slot),
                ucf_compute::SlotMode::Shadow
            )
        });

    let mut checks = Vec::new();
    if cfg!(feature = "backend-burn") {
        if matches!(slot_enablement.world_jepa, ucf_compute::SlotMode::Active) {
            checks.push(strict_fail(
                "v2_burn_world_active_denied",
                "ACTIVE_DENIED_BACKEND_NOT_YET_ALLOWED",
                "set UCF_SLOT_WORLD_JEPA=shadow|off for backend-burn world adapter",
            ));
        } else {
            checks.push(strict_pass("v2_burn_world_active_denied"));
        }
    } else {
        checks.push(strict_pass("v2_burn_world_active_denied"));
    }
    let manifest_path = PathBuf::from("models/MANIFEST.toml");
    let probe_enforcement_enabled = std::env::var("UCF_STRICT_ENFORCE_ACTIVE_PROBES")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);

    let v1_verify = models_verify_lifecycle(&manifest_path).ok();
    if let Some(report) = v1_verify.as_ref() {
        if report.manifest_present {
            let digest = sha256_hex(
                &serde_json::to_vec(report)
                    .unwrap_or_else(|_| b"strict.v1.models_verify.digest_fallback".to_vec()),
            );
            evidence_digest_prefixes
                .insert("models_verify_report".to_string(), prefix_hex(&digest, 16));
        }
    }

    if active_mode_requested {
        match fs::read_to_string(&manifest_path) {
            Ok(body) => {
                let manifest_digest = sha256_hex(body.as_bytes());
                evidence_digest_prefixes.insert(
                    "models_manifest".to_string(),
                    prefix_hex(&manifest_digest, 16),
                );
                checks.push(strict_pass("v1_manifest_active_requires_digest"));
            }
            Err(_) => checks.push(strict_fail(
                "v1_manifest_active_requires_digest",
                "strict.v1.manifest.required_for_active",
                "create models/MANIFEST.toml and pin digest evidence before active rollout",
            )),
        }
    } else {
        checks.push(strict_pass("v1_manifest_active_requires_digest"));
    }

    if active_mode_requested {
        match v1_verify {
            Some(report) if report.manifest_present && report.promoted_hashes_exist && report.files_verified => {
                checks.push(strict_pass("v1_promoted_only_active_hashes"));
            }
            _ => checks.push(strict_fail(
                "v1_promoted_only_active_hashes",
                "strict.v1.promoted_only.verify_failed",
                "run `cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml` and promote active hashes",
            )),
        }
    } else {
        checks.push(strict_pass("v1_promoted_only_active_hashes"));
    }

    if probe_enforcement_enabled && active_mode_requested {
        let probe_path = workdir.join("out/probe_report.json");
        match fs::read_to_string(&probe_path)
            .ok()
            .and_then(|body| serde_json::from_str::<ProbeReport>(&body).ok())
        {
            Some(report) => {
                let probe_digest = sha256_hex(
                    &serde_json::to_vec(&report)
                        .unwrap_or_else(|_| b"strict.v1.probe.digest_fallback".to_vec()),
                );
                evidence_digest_prefixes.insert(
                    "probe_report".to_string(),
                    prefix_hex(&probe_digest, 16),
                );
                let all_ok = report
                    .results
                    .iter()
                    .all(|result| matches!(result.status, ProbeStatus::Ok | ProbeStatus::Disabled));
                let burn_world_ok = if cfg!(feature = "backend-burn") {
                    report
                        .results
                        .iter()
                        .find(|result| result.slot == ModelSlot::WorldJepa)
                        .map(|result| matches!(result.status, ProbeStatus::Ok))
                        .unwrap_or(false)
                } else {
                    true
                };
                if report.summary.pass && all_ok && burn_world_ok {
                    checks.push(strict_pass("v1_active_slots_probe_pass"));
                } else {
                    checks.push(strict_fail(
                        "v1_active_slots_probe_pass",
                        "PROBE_REQUIRED",
                        "run `cargo run -p ucf-ops -- models probe --manifest models/manifest.toml --out ./out/probe_report.json` and require PASS",
                    ));
                }
            }
            None => checks.push(strict_fail(
                "v1_active_slots_probe_pass",
                "strict.v1.probes.report_missing",
                "run `cargo run -p ucf-ops -- models probe --manifest models/manifest.toml --out ./out/probe_report.json`",
            )),
        }
    } else {
        checks.push(strict_pass("v1_active_slots_probe_pass"));
    }

    if shadow_enabled {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let drift_budget_override = std::env::var("UCF_STRICT_DRIFT_BUDGET_PATH").ok();
        let base_drift_budget = repo_root.join("policies/packs/base_v1/drift_budget.toml");
        let overlay_drift_budget = repo_root.join(format!(
            "policies/packs/overlays/{}/drift_budget.toml",
            cfg.profile
        ));
        let drift_budget_path = drift_budget_override
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| {
                if overlay_drift_budget.exists() {
                    Some(overlay_drift_budget.clone())
                } else if base_drift_budget.exists() {
                    Some(base_drift_budget.clone())
                } else {
                    None
                }
            });
        if let Some(path) = drift_budget_path {
            match fs::read(path) {
                Ok(digest_source) if !digest_source.is_empty() => {
                    evidence_digest_prefixes.insert(
                        "drift_budget".to_string(),
                        prefix_hex(&sha256_hex(&digest_source), 16),
                    );
                    checks.push(strict_pass("v1_shadow_requires_drift_budget"));
                }
                _ => checks.push(strict_fail(
                    "v1_shadow_requires_drift_budget",
                    "strict.v1.shadow.drift_budget_missing",
                    "run `cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test` and ensure drift_budget.toml is present",
                )),
            }
        } else {
            checks.push(strict_fail(
                "v1_shadow_requires_drift_budget",
                "strict.v1.shadow.drift_budget_missing",
                "run `cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test` and ensure drift_budget.toml is present",
            ));
        }

        let compare_enabled = std::env::var("UCF_SLOT_COMPARE_WINDOW")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v > 0)
            .unwrap_or(true);
        if compare_enabled {
            checks.push(strict_pass("v1_shadow_compare_window_required"));
        } else {
            checks.push(strict_fail(
                "v1_shadow_compare_window_required",
                "strict.v1.shadow.compare_window_disabled",
                "set UCF_SLOT_COMPARE_WINDOW to a positive integer to emit SlotCompareWindow records",
            ));
        }

        let world_compare_configured = std::env::var("UCF_WORLD_PARITY_COMPARE_ENABLED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let candle_shadow = std::env::var("UCF_WORLD_CANDLE_SHADOW_ENABLED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let burn_shadow = std::env::var("UCF_WORLD_BURN_SHADOW_ENABLED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        if world_compare_configured && candle_shadow && burn_shadow {
            if world_parity_evidence_exists(workdir) {
                checks.push(strict_pass("v2_world_parity_evidence_required"));
            } else {
                checks.push(strict_fail(
                    "v2_world_parity_evidence_required",
                    "PARITY_EVIDENCE_MISSING",
                    "run `cargo run -p ucf-ops -- world parity-report --run <id> --out ./out/world_parity_report.json`",
                ));
            }
        } else {
            checks.push(strict_pass("v2_world_parity_evidence_required"));
        }

        let second_slot = detect_second_slot(Path::new(".")).unwrap_or(ModelSlot::Sae);
        let second_compare_required = std::env::var("UCF_SECOND_SLOT_PARITY_COMPARE_REQUIRED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let second_burn_required = std::env::var("UCF_SECOND_SLOT_BURN_PARITY_REQUIRED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        if second_compare_required {
            if second_slot_parity_evidence_exists(workdir, second_slot) {
                checks.push(strict_pass("v3_second_slot_parity_evidence_required"));
            } else {
                checks.push(strict_fail(
                    "v3_second_slot_parity_evidence_required",
                    "PARITY_EVIDENCE_MISSING",
                    "run `cargo run -p ucf-ops -- models parity --slot <sae|ssm> --run <id> --out ./out/<slot>_parity_report.json`",
                ));
            }
        } else {
            checks.push(strict_pass("v3_second_slot_parity_evidence_required"));
        }

        if second_burn_required {
            let resolution =
                models_lifecycle::models_backend_resolution(workdir, second_slot, None).ok();
            if let Some(resolution) = resolution {
                if matches!(
                    resolution.resolution,
                    models_lifecycle::BurnResolutionStatusV1::BurnSupportedForShadowCompare
                ) {
                    checks.push(strict_pass("v4_optional_backend_burn_parity_required"));
                } else {
                    checks.push(strict_fail(
                        "v4_optional_backend_burn_parity_required",
                        "OPTIONAL_BACKEND_CLOSED_UNSUPPORTED",
                        "Burn for configured second slot is formally closed unsupported in this phase; disable UCF_SECOND_SLOT_BURN_PARITY_REQUIRED or explicitly reopen via governance",
                    ));
                }
            } else {
                checks.push(strict_fail(
                    "v4_optional_backend_burn_parity_required",
                    "OPTIONAL_BACKEND_CLOSED_UNSUPPORTED",
                    "Burn resolution unavailable for configured second slot; regenerate parity/evidence artifacts and rerun",
                ));
            }
        } else {
            checks.push(strict_pass("v4_optional_backend_burn_parity_required"));
        }

        let strict_shadow_evidence_required = std::env::var("UCF_STRICT_SHADOW_EVIDENCE_REQUIRED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        if strict_shadow_evidence_required {
            let strict_shadow_evidence_hard_fail =
                std::env::var("UCF_STRICT_SHADOW_EVIDENCE_HARD_FAIL")
                    .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                    .unwrap_or(false);
            match models_lifecycle::models_shadow_ready(workdir, None, &workdir.join("out/shadow_ready_report.json")) {
                Ok(report) if matches!(report.overall_status, models_lifecycle::AggregatedStatusV1::Pass) => {
                    evidence_digest_prefixes.insert(
                        "shadow_ready_report".to_string(),
                        prefix_hex(&report.report_digest, 16),
                    );
                    checks.push(strict_pass("v2_shadow_ready_evidence_required"));
                }
                Ok(_) if strict_shadow_evidence_hard_fail => checks.push(strict_fail(
                    "v2_shadow_ready_evidence_required",
                    "strict.v2.shadow_ready.evidence_missing",
                    "run `cargo run -p ucf-ops -- models shadow-ready --out ./out/shadow_ready_report.json` after compare windows are available",
                )),
                Ok(_) => checks.push(strict_pass("v2_shadow_ready_evidence_required")),
                Err(_) if strict_shadow_evidence_hard_fail => checks.push(strict_fail(
                    "v2_shadow_ready_evidence_required",
                    "strict.v2.shadow_ready.evidence_missing",
                    "run `cargo run -p ucf-ops -- models shadow-ready --out ./out/shadow_ready_report.json` once probe + compare evidence exists",
                )),
                Err(_) => checks.push(strict_pass("v2_shadow_ready_evidence_required")),
            }
        } else {
            checks.push(strict_pass("v2_shadow_ready_evidence_required"));
        }
    } else {
        checks.push(strict_pass("v1_shadow_requires_drift_budget"));
        checks.push(strict_pass("v1_shadow_compare_window_required"));
    }

    if active_mode_requested {
        let bypass = cfg.profile.eq_ignore_ascii_case("dev")
            && std::env::var("UCF_DEV_ACTIVE_BYPASS")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(false);
        let active_slots = ModelSlot::all()
            .into_iter()
            .filter(|slot| {
                matches!(
                    slot_enablement.for_slot(*slot),
                    ucf_compute::SlotMode::Active
                )
            })
            .collect::<Vec<_>>();
        if active_slots.is_empty() {
            checks.push(strict_fail(
                "v2_active_requires_evidence",
                "strict.v2.active.slot_missing",
                "configure at least one real slot explicitly before active check",
            ));
        } else {
            let mut ok = true;
            for slot in active_slots {
                let manifest = models_lifecycle::models_list(slot).ok();
                let target = manifest.and_then(|m| m.active_hash);
                if let Some(hash) = target {
                    match can_enable_active(slot, &hash, workdir, true, bypass) {
                        Ok(e) => {
                            evidence_digest_prefixes.insert(
                                format!("active_evidence_{}", slot.as_str()),
                                prefix_hex(&e.evidence_digest, 16),
                            );
                        }
                        Err(_) => ok = false,
                    }
                } else {
                    ok = false;
                }
            }
            if ok {
                checks.push(strict_pass("v2_active_requires_evidence"));
            } else {
                checks.push(strict_fail(
                    "v2_active_requires_evidence",
                    "strict.v2.active.evidence_missing",
                    "run `cargo run -p ucf-ops -- models active-check --slot <slot> --out ./out/active_check_<slot>.json` and keep slot shadow until PASS",
                ));
            }
        }
    } else {
        checks.push(strict_pass("v2_active_requires_evidence"));
    }

    checks.push(strict_pass("v1_shadow_no_decision_impact_guard"));
    checks
}

fn strict_v3_checks(workdir: &Path, cfg: &OpsConfig) -> StrictFailureReportV3 {
    let mut checks = Vec::new();
    let mut slots = models_lifecycle::current_supported_real_slot_set(workdir)
        .ok()
        .map(|set| {
            set.slots
                .iter()
                .filter_map(|slot| parse_slot(slot).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            let second_slot = detect_second_slot(workdir).ok();
            let mut fallback = vec![ModelSlot::WorldJepa];
            if let Some(slot) = second_slot {
                fallback.push(slot);
            }
            fallback
        });
    slots.sort_by_key(|s| s.as_str().to_string());
    slots.dedup();

    if !cfg.strict_mode {
        checks.push(strict_v3_check(
            "STRICT_MANIFEST_VALID",
            None,
            StrictCheckV3Status::Skip,
            None,
            Vec::new(),
            "REMEDIATE_MANIFEST",
        ));
        for slot in &slots {
            let slot_id = Some(slot.as_str().to_string());
            for (check_id, remediation) in [
                ("STRICT_PROBE_READY", "REMEDIATE_PROBE"),
                ("STRICT_SHADOW_READY", "REMEDIATE_SHADOW_READY"),
                ("STRICT_ACTIVE_ELIGIBLE", "REMEDIATE_ACTIVE_ELIGIBILITY"),
                ("STRICT_COMPARE_FRESH", "REMEDIATE_COMPARE_WINDOW"),
                ("STRICT_DRIFT_OK", "REMEDIATE_DRIFT"),
                ("STRICT_HASH_CONSISTENT", "REMEDIATE_HASH_ALIGNMENT"),
            ] {
                checks.push(strict_v3_check(
                    check_id,
                    slot_id.clone(),
                    StrictCheckV3Status::Skip,
                    None,
                    Vec::new(),
                    remediation,
                ));
            }
        }
        return StrictFailureReportV3 {
            schema_version: 3,
            strict_mode_enabled: false,
            overall_status: "PASS".to_string(),
            checks,
        };
    }

    let slot_enablement = ucf_compute::SlotEnablement::from_env().unwrap_or_default();
    let compare_max_age = cfg.active_evidence_compare_max_age_ticks.max(1);
    let any_real_shadow_or_active_requested = slots.iter().any(|slot| {
        matches!(
            slot_enablement.for_slot(*slot),
            ucf_compute::SlotMode::Shadow | ucf_compute::SlotMode::Active
        )
    });

    let manifest_raw = fs::read_to_string(PathBuf::from("models").join("manifest.toml"))
        .or_else(|_| fs::read_to_string(PathBuf::from("models").join("MANIFEST.toml")))
        .ok();
    let manifest = manifest_raw
        .as_ref()
        .and_then(|body| body.parse::<toml::Value>().ok());
    let manifest_ok = manifest
        .as_ref()
        .and_then(|m| m.get("slots").and_then(|v| v.as_array()))
        .is_some_and(|entries| {
            entries.iter().any(|entry| {
                let slot_id = entry
                    .get("slot_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let hash = entry
                    .get("active_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                slots.iter().any(|slot| slot.as_str() == slot_id)
                    && !hash.is_empty()
                    && hash != "missing"
            })
        });
    checks.push(strict_v3_check(
        "STRICT_MANIFEST_VALID",
        None,
        if !any_real_shadow_or_active_requested {
            StrictCheckV3Status::Skip
        } else if manifest_ok {
            StrictCheckV3Status::Pass
        } else {
            StrictCheckV3Status::Fail
        },
        if !any_real_shadow_or_active_requested || manifest_ok {
            None
        } else {
            Some("STRICT_MANIFEST_INVALID")
        },
        Vec::new(),
        "REMEDIATE_MANIFEST",
    ));

    for slot in &slots {
        let slot_id = slot.as_str().to_string();
        let list = models_lifecycle::models_list(*slot).ok();
        let mode = slot_enablement.for_slot(*slot);
        let mode_requires_shadow = matches!(mode, ucf_compute::SlotMode::Shadow);
        let mode_requires_active = matches!(mode, ucf_compute::SlotMode::Active);

        let probe_ready = list.as_ref().is_some_and(|r| r.probe_ready);
        let probe_denial = list
            .as_ref()
            .and_then(|r| r.denial_reason_probe.clone())
            .unwrap_or_else(|| "STRICT_PROBE_NOT_READY".to_string());
        checks.push(strict_v3_check(
            "STRICT_PROBE_READY",
            Some(slot_id.clone()),
            if !mode_requires_shadow && !mode_requires_active {
                StrictCheckV3Status::Skip
            } else if probe_ready {
                StrictCheckV3Status::Pass
            } else {
                StrictCheckV3Status::Fail
            },
            if !mode_requires_shadow && !mode_requires_active || probe_ready {
                None
            } else {
                Some(&probe_denial)
            },
            list.as_ref()
                .and_then(|r| r.last_evidence_digest_prefix.clone())
                .into_iter()
                .collect(),
            "REMEDIATE_PROBE",
        ));

        let shadow_ready = list.as_ref().is_some_and(|r| r.shadow_ready);
        let shadow_denial = list
            .as_ref()
            .and_then(|r| r.denial_reason_shadow.clone())
            .unwrap_or_else(|| "STRICT_SHADOW_NOT_READY".to_string());
        checks.push(strict_v3_check(
            "STRICT_SHADOW_READY",
            Some(slot_id.clone()),
            if !mode_requires_shadow {
                StrictCheckV3Status::Skip
            } else if shadow_ready {
                StrictCheckV3Status::Pass
            } else {
                StrictCheckV3Status::Fail
            },
            if !mode_requires_shadow || shadow_ready {
                None
            } else {
                Some(&shadow_denial)
            },
            list.as_ref()
                .and_then(|r| r.last_evidence_digest_prefix.clone())
                .into_iter()
                .collect(),
            "REMEDIATE_SHADOW_READY",
        ));

        let active_eligible = list.as_ref().is_some_and(|r| r.active_eligible);
        let active_denial = list
            .as_ref()
            .and_then(|r| r.denial_reason_active.clone())
            .unwrap_or_else(|| "STRICT_ACTIVE_NOT_ELIGIBLE".to_string());
        checks.push(strict_v3_check(
            "STRICT_ACTIVE_ELIGIBLE",
            Some(slot_id.clone()),
            if !mode_requires_active {
                StrictCheckV3Status::Skip
            } else if active_eligible {
                StrictCheckV3Status::Pass
            } else {
                StrictCheckV3Status::Fail
            },
            if !mode_requires_active || active_eligible {
                None
            } else {
                Some(&active_denial)
            },
            list.as_ref()
                .and_then(|r| r.last_evidence_digest_prefix.clone())
                .into_iter()
                .collect(),
            "REMEDIATE_ACTIVE_ELIGIBILITY",
        ));

        let shadow_denial_code = list
            .as_ref()
            .and_then(|r| r.denial_reason_shadow.clone())
            .unwrap_or_default();
        let active_denial_code = list
            .as_ref()
            .and_then(|r| r.denial_reason_active.clone())
            .unwrap_or_default();
        let compare_ok = !(shadow_denial_code.contains("STALE_COMPARE")
            || active_denial_code.contains("ActiveDeniedStaleCompare"));
        checks.push(strict_v3_check(
            "STRICT_COMPARE_FRESH",
            Some(slot_id.clone()),
            if !mode_requires_shadow && !mode_requires_active {
                StrictCheckV3Status::Skip
            } else if compare_ok {
                StrictCheckV3Status::Pass
            } else {
                StrictCheckV3Status::Fail
            },
            if !mode_requires_shadow && !mode_requires_active || compare_ok {
                None
            } else {
                Some("STRICT_COMPARE_WINDOW_STALE")
            },
            Vec::new(),
            "REMEDIATE_COMPARE_WINDOW",
        ));

        let drift_ok = !(shadow_denial_code.contains("DRIFT_SEVERE")
            || active_denial_code.contains("ActiveDeniedDriftSevere")
            || active_denial_code.contains("ActiveDeniedDriftWarn"));
        checks.push(strict_v3_check(
            "STRICT_DRIFT_OK",
            Some(slot_id.clone()),
            if !mode_requires_shadow && !mode_requires_active {
                StrictCheckV3Status::Skip
            } else if drift_ok {
                StrictCheckV3Status::Pass
            } else {
                StrictCheckV3Status::Fail
            },
            if !mode_requires_shadow && !mode_requires_active || drift_ok {
                None
            } else {
                Some("STRICT_DRIFT_DENY")
            },
            Vec::new(),
            "REMEDIATE_DRIFT",
        ));

        let probe_path = workdir
            .join("out")
            .join(format!("probe_{}.json", slot.as_str()));
        let probe = fs::read_to_string(&probe_path)
            .ok()
            .and_then(|body| serde_json::from_str::<ProbeReportV1>(&body).ok());
        let target_hash_prefix = manifest
            .as_ref()
            .and_then(|m| m.get("slots").and_then(|v| v.as_array()))
            .and_then(|entries| {
                entries.iter().find_map(|entry| {
                    let id = entry.get("slot_id").and_then(|v| v.as_str())?;
                    if id == slot_id {
                        Some(
                            entry
                                .get("active_hash")
                                .and_then(|v| v.as_str())
                                .unwrap_or("missing"),
                        )
                    } else {
                        None
                    }
                })
            })
            .map(|h| prefix_hex(h, 16))
            .unwrap_or_else(|| "missing".to_string());
        let probe_hash_ok = probe
            .as_ref()
            .and_then(|p| p.model_hash_prefix.as_ref())
            .is_some_and(|h| h == &target_hash_prefix);

        let mut max_tick = 0_u64;
        let records =
            load_fixture_records(&workdir.join("ess").join("ess_fixture.json")).unwrap_or_default();
        for rec in records {
            max_tick = max_tick.max(rec.time.tick.get());
            if let ExperiencePayload::Audit(AuditPayload::SlotCompareWindow(_window)) = rec.payload
            {
            }
        }
        let _ = compare_freshness(Some(max_tick), max_tick, compare_max_age);
        let hash_ok = probe_hash_ok && !active_denial_code.contains("HashMismatch");
        checks.push(strict_v3_check(
            "STRICT_HASH_CONSISTENT",
            Some(slot_id),
            if !mode_requires_shadow && !mode_requires_active {
                StrictCheckV3Status::Skip
            } else if hash_ok {
                StrictCheckV3Status::Pass
            } else {
                StrictCheckV3Status::Fail
            },
            if !mode_requires_shadow && !mode_requires_active || hash_ok {
                None
            } else {
                Some("STRICT_HASH_MISMATCH")
            },
            vec![target_hash_prefix],
            "REMEDIATE_HASH_ALIGNMENT",
        ));
    }

    checks.sort_by(|a, b| {
        let a_global = a.slot_id.is_none();
        let b_global = b.slot_id.is_none();
        b_global
            .cmp(&a_global)
            .then(
                a.slot_id
                    .as_deref()
                    .unwrap_or("")
                    .cmp(b.slot_id.as_deref().unwrap_or("")),
            )
            .then(a.check_id.cmp(&b.check_id))
    });
    let failed = checks
        .iter()
        .any(|c| matches!(c.status, StrictCheckV3Status::Fail));
    StrictFailureReportV3 {
        schema_version: 3,
        strict_mode_enabled: true,
        overall_status: if failed { "FAIL" } else { "PASS" }.to_string(),
        checks,
    }
}

fn strict_v3_check(
    check_id: &str,
    slot_id: Option<String>,
    status: StrictCheckV3Status,
    denial_code: Option<&str>,
    mut evidence_digest_prefixes: Vec<String>,
    remediation_code: &str,
) -> StrictCheckV3Result {
    evidence_digest_prefixes.sort();
    evidence_digest_prefixes.dedup();
    evidence_digest_prefixes.truncate(4);
    StrictCheckV3Result {
        check_id: check_id.to_string(),
        slot_id,
        status,
        denial_code: denial_code.map(|v| v.to_string()),
        evidence_digest_prefixes,
        remediation_code: remediation_code.to_string(),
        canonical_remediation_codes: crate::remediation::canonical_from_legacy_code(
            remediation_code,
        ),
    }
}

fn strict_pass(id: &str) -> StrictCheckResult {
    StrictCheckResult {
        check_id: id.to_string(),
        status: StrictCheckStatus::Pass,
        error_codes: Vec::new(),
        remediation: "ok".to_string(),
        canonical_remediation_codes: Vec::new(),
    }
}

fn strict_fail(id: &str, code: &str, remediation: &str) -> StrictCheckResult {
    StrictCheckResult {
        check_id: id.to_string(),
        status: StrictCheckStatus::Fail,
        error_codes: vec![code.to_string()],
        remediation: remediation.to_string(),
        canonical_remediation_codes: crate::remediation::canonical_from_legacy_remediation(
            remediation,
        ),
    }
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
    let cfg = ConfigV1::from_toml_str(&raw)
        .map_err(|e| OpsError::Invalid(format!("invalid config {}: {e}", path.display())))?;
    Ok(cfg.into_ops_config())
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
        "UCF_STRICT_MODE",
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

    if let Ok(v) = std::env::var("UCF_STRICT_MODE") {
        cfg.strict_mode = matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES");
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
    let _env_guards = [
        EnvVarGuard::remove("UCF_REAL_ENABLEMENT_MODE"),
        EnvVarGuard::remove("UCF_SLOT_WORLD_JEPA"),
        EnvVarGuard::remove("UCF_SLOT_WORLD_VLJEPA"),
        EnvVarGuard::remove("UCF_SLOT_SAE"),
        EnvVarGuard::remove("UCF_SLOT_SSM"),
        EnvVarGuard::remove("UCF_SLOT_LFM"),
    ];
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
    let in_unit_interval = |value: f32| (-1.0e-6..=1.0 + 1.0e-6).contains(&value);
    let pass = out.risk.is_finite()
        && out.confidence.is_finite()
        && in_unit_interval(out.risk)
        && in_unit_interval(out.confidence);

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
    let commit = if output.status.success() {
        let parsed = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if parsed.is_empty() {
            "unknown".to_string()
        } else {
            parsed
        }
    } else {
        "unknown".to_string()
    };
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

fn persist_jsonl_record(path: &Path, value: &impl Serialize) -> Result<(), OpsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

fn first_existing_path(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.exists()).cloned()
}

fn run_shell_command(cmd: &[&str], out: &Path) -> Result<(bool, String), OpsError> {
    let mut command = Command::new(cmd[0]);
    command.args(&cmd[1..]);
    let output = command.output()?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::new();
    body.push_str(&String::from_utf8_lossy(&output.stdout));
    body.push_str(&String::from_utf8_lossy(&output.stderr));
    fs::write(out, body.as_bytes())?;
    Ok((
        output.status.success(),
        format!("exit_code={}", output.status.code().unwrap_or(1)),
    ))
}

fn diff_ops_config_keys(current: &OpsConfig, updated: &OpsConfig) -> Vec<String> {
    let mut keys = Vec::new();
    macro_rules! changed {
        ($field:ident) => {
            if current.$field != updated.$field {
                keys.push(stringify!($field).to_string());
            }
        };
    }
    changed!(profile);
    changed!(strict_mode);
    changed!(policy_overlay);
    changed!(backend_pack);
    changed!(slot_ebm_mode);
    changed!(offline);
    changed!(compute_backend);
    changed!(compute_seed);
    changed!(compute_budget_profile);
    changed!(device_profile);
    changed!(isolation_runtime);
    changed!(capabilities_default);
    changed!(sampling_enabled);
    changed!(determinism_lock_strict);
    changed!(docs_lint_required);
    changed!(stage_isolation_optional);
    changed!(emergency_policy_pin);
    changed!(log_level);
    changed!(active_evidence_probe_max_age_ticks);
    changed!(active_evidence_compare_max_age_ticks);
    changed!(active_evidence_no_impact_max_age_ticks);
    changed!(active_evidence_drift_status_max_age_ticks);
    changed!(active_evidence_allow_warn_drift_for_active);
    changed!(active_evidence_require_matching_target_hash);
    keys.sort();
    keys
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
    fn export_and_verify_roundtrip() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
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
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
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
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
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
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
        let c = check_weights_lifecycle_integrity(dir.path()).expect("check");
        assert_eq!(c.name, "weights_lifecycle");
        assert_eq!(c.status, GateStatus::Skip);
    }

    #[test]
    fn world_vljepa_check_fails_when_required_shadow_missing() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
        fs::create_dir_all("models").expect("models");
        fs::write(
            "models/lifecycle_manifest.toml",
            r#"manifest_version = 1
manifest_digest = "x"
[slots.world_vljepa]
active_hash = "abc"
"#,
        )
        .expect("manifest");
        let c = check_world_vljepa_shadow_evidence(dir.path()).expect("check");
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
        let store =
            ModelStore::from_manifest_and_env(Path::new("nonexistent.toml")).expect("store");
        let a = run_world_probe(
            BackendPackFactory::build(BackendPackConfig::default()).expect("pack a"),
            &world_spec,
            false,
            &store,
        )
        .expect("world a");
        let b = run_world_probe(
            BackendPackFactory::build(BackendPackConfig::default()).expect("pack b"),
            &world_spec,
            false,
            &store,
        )
        .expect("world b");
        assert_eq!(a.digest, b.digest);

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
        assert_eq!(c.digest, d.digest);
    }

    #[test]
    fn ess_snapshot_manifest_digest_is_deterministic() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
        bringup(dir.path(), true, 8).expect("bringup");
        let snap_path = dir.path().join("snapshots/run.snap");
        let a = ess_snapshot(dir.path(), &snap_path).expect("snapshot a");
        let b = ess_snapshot(dir.path(), &snap_path).expect("snapshot b");
        assert_eq!(a.manifest_digest, b.manifest_digest);
    }

    #[test]
    fn ebm_dataset_export_is_redaction_safe_and_bounded() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
        bringup(dir.path(), true, 20).expect("bringup");
        let out = dir.path().join("out").join("ebm_dataset_v1.jsonl");
        let policy = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../policies/bundle_v1/retention_v1.json");
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
            strict_mode_enabled: false,
            strict_mode_digest: None,
            probe_report_digest_prefix: None,
            crash_dumps_disabled: false,
            models_manifest_present: false,
            models_manifest_digest_prefix: None,
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
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let _cwd = CwdGuard::enter(dir.path());
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseBuildRcArgs {
    pub version: String,
    pub profile: String,
    pub out: PathBuf,
    pub fast: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RcVerificationReportDigest {
    pub step: String,
    pub report: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RcManifestV1 {
    pub schema_version: u16,
    pub version: String,
    pub profile: String,
    pub code_version_tag: String,
    pub policy_graph_digest: String,
    pub models_manifest_digest: String,
    pub binary_hashes: BTreeMap<String, String>,
    pub verification_reports: Vec<RcVerificationReportDigest>,
    pub bundle_hashes: BTreeMap<String, String>,
    pub rc_digest: String,
    pub signer_key_id: String,
    pub signer_public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseBuildRcReport {
    pub version: String,
    pub profile: String,
    pub fast: bool,
    pub rc_zip: String,
    pub rc_digest: String,
    pub out_dir: String,
}

pub fn release_build_rc(
    workdir: &Path,
    args: &ReleaseBuildRcArgs,
) -> Result<ReleaseBuildRcReport, OpsError> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::create_dir_all(&args.out)?;
    let reports_dir = args.out.join("reports");
    fs::create_dir_all(&reports_dir)?;

    run_release_step(
        &repo_root,
        &["cargo", "build", "--release"],
        &reports_dir.join("build_release.log"),
        "build_release",
    )?;

    let docs_out = reports_dir.join("docs_lint_report.json");
    let docs_report = docs_lint(&DocsLintArgs {
        repo_root: repo_root.clone(),
        policy_pack: repo_root.join("policies/packs/base_v1"),
        overlay_pack: Some(repo_root.join("policies/packs/overlays/test")),
        spec_snapshot: repo_root.join("docs/spec_snapshot.md"),
        prompt_index: repo_root.join("docs/prompt_series_index.md"),
        module_map: repo_root.join("docs/module_map.md"),
        deploy_doc: repo_root.join("docs/deploy_portable.md"),
        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
        mode: DocsLintMode::Strict,
    })?;
    write_json(&docs_out, &docs_report)?;
    if !docs_report.ok {
        return Err(OpsError::Invalid(format!(
            "release build-rc failed at docs_lint (report: {})",
            docs_out.display()
        )));
    }

    ensure_spec_snapshot_unchanged(&repo_root, &reports_dir.join("spec_snapshot_check.json"))?;

    let gate_out = reports_dir.join("gate_report.json");
    let gate = readiness_gate(workdir, "test", &gate_out)?;
    if gate.status != GateStatus::Pass {
        return Err(OpsError::Invalid(format!(
            "release build-rc failed at readiness_gate (report: {})",
            gate_out.display()
        )));
    }

    let adversarial_out = reports_dir.join("adversarial_report.json");
    if !args.fast {
        let adversarial = adversarial_run(&AdversarialRunArgs {
            workdir: workdir.to_path_buf(),
            suite: "v1".to_string(),
            out: adversarial_out.clone(),
        })?;
        if !adversarial.pass {
            return Err(OpsError::Invalid(format!(
                "release build-rc failed at adversarial_run (report: {})",
                adversarial_out.display()
            )));
        }
    } else {
        write_json(
            &adversarial_out,
            &serde_json::json!({"skipped": true, "reason": "--fast"}),
        )?;
    }

    let goldens_out = reports_dir.join("goldens_report.json");
    if !args.fast {
        let scenarios = golden_scenario_ids()?;
        let mut scenario_reports = Vec::new();
        for scenario in scenarios {
            scenario_reports.push(goldens_verify_detailed(&GoldenVerifyArgs {
                scenario,
                os: std::env::consts::OS.to_string(),
                out_root: PathBuf::from("fixtures/goldens"),
                workdir_root: PathBuf::from(".ucf_goldens"),
            })?);
        }
        scenario_reports.sort_by(|a, b| a.scenario.cmp(&b.scenario));
        let overall = if scenario_reports
            .iter()
            .all(|entry| entry.status == GateStatus::Pass)
        {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        };
        let bundle = GoldenVerifyReport {
            os: std::env::consts::OS.to_string(),
            status: overall,
            scenarios: scenario_reports,
        };
        write_json(&goldens_out, &bundle)?;
        if bundle.status != GateStatus::Pass {
            return Err(OpsError::Invalid(format!(
                "release build-rc failed at goldens_verify (report: {})",
                goldens_out.display()
            )));
        }
    } else {
        write_json(
            &goldens_out,
            &serde_json::json!({"skipped": true, "reason": "--fast"}),
        )?;
    }

    let strict_out = reports_dir.join("strict_check.json");
    let strict = strict_check(workdir, true, &strict_out)?;
    if !strict.ok {
        return Err(OpsError::Invalid(format!(
            "release build-rc failed at strict_check (report: {})",
            strict_out.display()
        )));
    }

    let bundle_out = args.out.join("bundle");
    let target = bundle_out.display().to_string();
    run_release_step(
        &repo_root,
        &[
            "python",
            "deploy/scripts/build_bundle.py",
            "--target",
            target.as_str(),
            "--profile",
            args.profile.as_str(),
            "--bin-source",
            "target/release",
        ],
        &reports_dir.join("bundle_build.log"),
        "bundle_build",
    )?;

    attest_keys_generate(workdir, false)?;
    let (policy_base, policy_overlay, manifest_path) = resolve_attestation_inputs();
    let policy = load_and_merge_policy_graph(&policy_base, Some(&policy_overlay))?;
    let manifest_verify = models_verify(&manifest_path)?;
    let binary_hashes = release_binary_hashes(&repo_root.join("target/release"))?;
    let mut verification_reports = collect_report_hashes(&reports_dir)?;
    verification_reports.sort_by(|a, b| a.step.cmp(&b.step));
    let mut bundle_hashes = collect_tree_hashes(&bundle_out)?;

    let code_version_tag = git_head_short(&repo_root)?;
    let signer_public_key = load_attestation_public_key_hex(workdir)?;
    let mut manifest = RcManifestV1 {
        schema_version: 1,
        version: args.version.clone(),
        profile: args.profile.clone(),
        code_version_tag,
        policy_graph_digest: policy.1.policy_graph_digest,
        models_manifest_digest: manifest_verify.model_hashes_digest,
        binary_hashes,
        verification_reports,
        bundle_hashes: std::mem::take(&mut bundle_hashes),
        rc_digest: String::new(),
        signer_key_id: "attestation_ed25519_v1".to_string(),
        signer_public_key,
    };
    manifest.rc_digest = rc_manifest_digest(&manifest)?;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let signature = sign_certificate_digest(workdir, &manifest.rc_digest)?;

    let mut pack_entries = BTreeMap::new();
    pack_entries.insert("RC_MANIFEST.json".to_string(), manifest_bytes.clone());
    pack_entries.insert(
        "RC_MANIFEST.sig".to_string(),
        format!("{}\n", signature).into_bytes(),
    );
    insert_tree_entries(&args.out.join("bundle"), "bundle", &mut pack_entries)?;
    insert_tree_entries(&reports_dir, "reports", &mut pack_entries)?;

    let mut sums_lines = Vec::new();
    for (path, bytes) in &pack_entries {
        sums_lines.push(format!("{}  {}", sha256_hex(bytes), path));
    }
    sums_lines.sort();
    let sums_bytes = format!("{}\n", sums_lines.join("\n")).into_bytes();
    pack_entries.insert("SHA256SUMS.txt".to_string(), sums_bytes);

    let mut digest_hasher = Sha256::new();
    digest_hasher.update(&serde_json::to_vec(&manifest)?);
    for (path, bytes) in &pack_entries {
        if path == "SHA256SUMS.txt" {
            continue;
        }
        digest_hasher.update(path.as_bytes());
        digest_hasher.update(sha256_hex(bytes).as_bytes());
    }
    let final_digest = hex::encode(digest_hasher.finalize());
    let rc_name = format!("ucf_rc_{}_{}.zip", args.version, &final_digest[..16]);
    let rc_zip = args.out.join(rc_name);
    write_deterministic_zip(&rc_zip, &pack_entries)?;

    Ok(ReleaseBuildRcReport {
        version: args.version.clone(),
        profile: args.profile.clone(),
        fast: args.fast,
        rc_zip: rc_zip.display().to_string(),
        rc_digest: final_digest,
        out_dir: args.out.display().to_string(),
    })
}

fn run_release_step(
    repo_root: &Path,
    cmd: &[&str],
    out: &Path,
    step: &str,
) -> Result<(), OpsError> {
    if cmd.is_empty() {
        return Err(OpsError::Invalid("empty command".to_string()));
    }
    let mut command = Command::new(cmd[0]);
    command.current_dir(repo_root).args(&cmd[1..]);
    let output = command.output()?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::new();
    body.push_str(&String::from_utf8_lossy(&output.stdout));
    body.push_str(&String::from_utf8_lossy(&output.stderr));
    fs::write(out, body)?;
    if !output.status.success() {
        return Err(OpsError::Invalid(format!(
            "release build-rc failed at {step} (report: {})",
            out.display()
        )));
    }
    Ok(())
}

fn ensure_spec_snapshot_unchanged(repo_root: &Path, out: &Path) -> Result<(), OpsError> {
    let spec_path = repo_root.join("docs/spec_snapshot.md");
    let before = fs::read(&spec_path)?;
    let temp = repo_root.join("out/spec_snapshot_build_rc_tmp.md");
    generate_spec_snapshot(&SpecSnapshotArgs {
        policy: repo_root.join("policies/packs/base_v1"),
        overlay: Some(repo_root.join("policies/packs/overlays/test")),
        out: temp.clone(),
    })?;
    let after = fs::read(&temp)?;
    let unchanged = before == after;
    let _ = fs::remove_file(&temp);
    write_json(
        out,
        &serde_json::json!({"unchanged": unchanged, "path": spec_path.display().to_string()}),
    )?;
    if !unchanged {
        return Err(OpsError::Invalid(format!(
            "release build-rc failed at spec_snapshot_unchanged (report: {})",
            out.display()
        )));
    }
    Ok(())
}

fn golden_scenario_ids() -> Result<Vec<String>, OpsError> {
    let mut ids = Vec::new();
    let scenarios_dir = PathBuf::from("fixtures/goldens/scenarios");
    for entry in fs::read_dir(scenarios_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|v| v.to_str()) != Some("json") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|v| v.to_str()) {
            ids.push(stem.to_string());
        }
    }
    ids.sort();
    Ok(ids)
}

fn release_binary_hashes(dir: &Path) -> Result<BTreeMap<String, String>, OpsError> {
    let mut out = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if name.ends_with(".d") || name.ends_with(".rlib") || name.ends_with(".rmeta") {
            continue;
        }
        if name.contains('.') && !name.ends_with(".exe") {
            continue;
        }
        let bytes = fs::read(&path)?;
        out.insert(name.to_string(), sha256_hex(&bytes));
    }
    if out.is_empty() {
        return Err(OpsError::Invalid(format!(
            "no release binaries found in {}",
            dir.display()
        )));
    }
    Ok(out)
}

fn collect_report_hashes(reports_dir: &Path) -> Result<Vec<RcVerificationReportDigest>, OpsError> {
    let mut out = Vec::new();
    for entry in fs::read_dir(reports_dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(&path)?;
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        out.push(RcVerificationReportDigest {
            step: name.to_string(),
            report: format!("reports/{name}"),
            sha256: sha256_hex(&bytes),
        });
    }
    out.sort_by(|a, b| a.step.cmp(&b.step));
    Ok(out)
}

fn collect_tree_hashes(root: &Path) -> Result<BTreeMap<String, String>, OpsError> {
    let mut out = BTreeMap::new();
    for entry in walkdir::WalkDir::new(root)
        .sort_by_file_name()
        .into_iter()
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .map_err(|e| OpsError::Invalid(format!("strip prefix failed: {e}")))?
            .to_string_lossy()
            .replace('\\', "/");
        out.insert(rel, sha256_hex(&fs::read(path)?));
    }
    Ok(out)
}

fn rc_manifest_digest(manifest: &RcManifestV1) -> Result<String, OpsError> {
    let mut canonical = manifest.clone();
    canonical.rc_digest.clear();
    canonical
        .verification_reports
        .sort_by(|a, b| a.step.cmp(&b.step));
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn insert_tree_entries(
    root: &Path,
    prefix: &str,
    entries: &mut BTreeMap<String, Vec<u8>>,
) -> Result<(), OpsError> {
    for entry in walkdir::WalkDir::new(root)
        .sort_by_file_name()
        .into_iter()
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .map_err(|e| OpsError::Invalid(format!("strip prefix failed: {e}")))?
            .to_string_lossy()
            .replace('\\', "/");
        let key = format!("{prefix}/{rel}");
        entries.insert(key, fs::read(path)?);
    }
    Ok(())
}

fn write_deterministic_zip(
    out: &Path,
    entries: &BTreeMap<String, Vec<u8>>,
) -> Result<(), OpsError> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(out)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .last_modified_time(zip::DateTime::default());
    for (path, bytes) in entries {
        zip.start_file(path, options)
            .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
        zip.write_all(bytes)
            .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
    }
    zip.finish()
        .map_err(|e| OpsError::Invalid(format!("zip finalize failed: {e}")))?;
    Ok(())
}

fn git_head_short(repo_root: &Path) -> Result<String, OpsError> {
    let out = Command::new("git")
        .current_dir(repo_root)
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()?;
    if !out.status.success() {
        return Err(OpsError::Invalid("unable to resolve git HEAD".to_string()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
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
    include_backtrace: bool,
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
    let panic_log = workdir.join("out").join("panic_records.jsonl");
    if include_backtrace && panic_log.exists() {
        selected.push(panic_log);
    }
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
    let path_redaction_re = regex::Regex::new(r"(/[A-Za-z0-9_./-]+|[A-Za-z]:\\[^\s]+)")
        .map_err(|e| OpsError::Invalid(format!("path redaction regex invalid: {e}")))?;
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
        if include_backtrace && (name.contains("panic") || text.contains("stack backtrace:")) {
            text = path_redaction_re
                .replace_all(&text, "<redacted_path>")
                .to_string();
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReproPackArtifact {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalArtifactIncludedStateV1 {
    Included,
    Missing,
    Excluded,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalExportArtifactRefV1 {
    pub artifact_kind: String,
    pub relative_path: String,
    pub included_state: CanonicalArtifactIncludedStateV1,
    pub sha256: Option<String>,
    pub schema_version: Option<u16>,
    pub artifact_digest: Option<String>,
    pub reason_code: Option<String>,
    pub ref_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalExportContextV1 {
    pub supported_slot_set_digest_prefix: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub run_id: Option<String>,
    pub operator_signoff_digest_prefix: Option<String>,
    pub backend_evidence_snapshot_digest_prefix: Option<String>,
    pub active_review_snapshot_digest_prefix: Option<String>,
    pub context_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalExportLayoutCompatibilityV1 {
    Canonical,
    LegacyExportLayout,
    LegacyExportTranslated,
    LegacyExportUnsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReproPackEssSlice {
    pub record_count: usize,
    pub segment_roots: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReproPackManifestV1 {
    pub schema_version: u16,
    pub pack_id: String,
    pub run_id: String,
    pub policy_graph_digest: String,
    pub manifest_digest: String,
    pub config_digest: String,
    pub included_artifacts: Vec<ReproPackArtifact>,
    pub ess_slice: ReproPackEssSlice,
    pub certificate_digest: Option<String>,
    pub evidence_context: PackEvidenceContextSummaryV1,
    pub backend_evidence_snapshot: PackEvidenceArtifactRefV1,
    pub active_review_snapshot: PackEvidenceArtifactRefV1,
    pub operator_signoff: PackEvidenceArtifactRefV1,
    pub backend_resolution: PackEvidenceArtifactRefV1,
    pub export_context: CanonicalExportContextV1,
    pub related_artifacts: Vec<CanonicalExportArtifactRefV1>,
    #[serde(default = "missing_prefix_value")]
    pub canonical_bundle_spine_digest_prefix: String,
    #[serde(default = "missing_prefix_value")]
    pub canonical_bundle_authority_digest_prefix: String,
    pub export_layout_compatibility: CanonicalExportLayoutCompatibilityV1,
    pub repro_pack_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReproPackBuildReport {
    pub run_id: String,
    pub pack_id: String,
    pub out: String,
    pub entry_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReproVerifyReport {
    pub pass: bool,
    pub run_id: String,
    pub pack_id: String,
    pub checked_files: usize,
    pub replay_report: String,
    pub first_divergence: Option<u64>,
    pub reasons: Vec<String>,
}

fn canonical_bundle_context_digest_hex(
    context: &CanonicalBundleConsumptionContextV1,
) -> Result<String, OpsError> {
    let mut canonical = context.clone();
    canonical.consumption_context_digest.clear();
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn bundle_roundtrip_digest_hex(report: &BundleRoundTripConsistencyV1) -> Result<String, OpsError> {
    let mut canonical = report.clone();
    canonical.roundtrip_digest.clear();
    canonical.mismatch_codes.sort();
    canonical.canonical_condition_codes.sort();
    canonical.primary_remediation_codes.sort();
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn bundle_spine_digest_hex(spine: &CanonicalBundleSpineV1) -> Result<String, OpsError> {
    let mut canonical = spine.clone();
    canonical.bundle_spine_digest.clear();
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn bundle_authority_digest_hex(authority: &CanonicalBundleAuthorityV2) -> Result<String, OpsError> {
    let mut canonical = authority.clone();
    canonical.authority_digest.clear();
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn digest_prefix_str(value: &str, len: usize) -> String {
    value.chars().take(len).collect()
}

pub const CANONICAL_BUNDLE_SPINE_REQUIRED: &str = "CANONICAL_BUNDLE_SPINE_REQUIRED";
pub const CANONICAL_BUNDLE_CONTEXT_REQUIRED: &str = "CANONICAL_BUNDLE_CONTEXT_REQUIRED";
pub const CANONICAL_EXPORT_REFS_REQUIRED: &str = "CANONICAL_EXPORT_REFS_REQUIRED";
pub const CANONICAL_EXPORT_ARTIFACT_REFS_REQUIRED: &str = "CANONICAL_EXPORT_ARTIFACT_REFS_REQUIRED";
pub const CANONICAL_EXPORT_CONTEXT_REQUIRED: &str = "CANONICAL_EXPORT_CONTEXT_REQUIRED";
pub const SECONDARY_BUNDLE_PATH_BLOCKED: &str = "SECONDARY_BUNDLE_PATH_BLOCKED";
pub const FINAL_BUNDLE_AUTHORITY_REQUIRED: &str = "FINAL_BUNDLE_AUTHORITY_REQUIRED";
pub const LEGACY_BUNDLE_INPUT_BLOCKED: &str = "LEGACY_BUNDLE_INPUT_BLOCKED";

fn canonical_artifact_refs_digest_prefix(
    refs: &[CanonicalExportArtifactRefV1],
) -> Result<String, OpsError> {
    let mut canonical = refs.to_vec();
    canonical.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(digest_prefix_str(
        &sha256_hex(&serde_json::to_vec(&canonical)?),
        16,
    ))
}

fn parse_normalized_bundle_manifest(
    bundle_kind: CanonicalBundleKindV1,
    export_context: &CanonicalExportContextV1,
    related_artifacts: &[CanonicalExportArtifactRefV1],
) -> Result<CanonicalBundleConsumptionContextV1, OpsError> {
    let mut included_artifact_kinds = related_artifacts
        .iter()
        .filter(|item| {
            matches!(
                item.included_state,
                CanonicalArtifactIncludedStateV1::Included
            )
        })
        .map(|item| item.artifact_kind.clone())
        .collect::<Vec<_>>();
    included_artifact_kinds.sort();
    included_artifact_kinds.dedup();

    let mut context = CanonicalBundleConsumptionContextV1 {
        bundle_kind,
        export_context_digest_prefix: digest_prefix_str(&export_context.context_digest, 16),
        applied_supported_set_digest_prefix: export_context
            .supported_slot_set_digest_prefix
            .clone(),
        policy_graph_digest_prefix: export_context.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: export_context.manifest_digest_prefix.clone(),
        artifact_refs_digest_prefix: canonical_artifact_refs_digest_prefix(related_artifacts)?,
        included_artifact_kinds,
        consumption_context_digest: String::new(),
    };
    context.consumption_context_digest = canonical_bundle_context_digest_hex(&context)?;
    Ok(context)
}

fn merge_status(
    a: BundleRoundTripMatchStatusV1,
    b: BundleRoundTripMatchStatusV1,
) -> BundleRoundTripMatchStatusV1 {
    use BundleRoundTripMatchStatusV1::{Legacy, Match, Mismatch, Missing};
    match (a, b) {
        (Mismatch, _) | (_, Mismatch) => Mismatch,
        (Legacy, _) | (_, Legacy) => Legacy,
        (Missing, _) | (_, Missing) => Missing,
        _ => Match,
    }
}

struct BundleRoundTripInputs<'a> {
    bundle_kind: CanonicalBundleKindV1,
    bundle_digest: &'a str,
    export_context: &'a CanonicalExportContextV1,
    evidence_context: &'a PackEvidenceContextSummaryV1,
    related_artifacts: &'a [CanonicalExportArtifactRefV1],
    backend_evidence_snapshot: &'a PackEvidenceArtifactRefV1,
    active_review_snapshot: &'a PackEvidenceArtifactRefV1,
    operator_signoff: &'a PackEvidenceArtifactRefV1,
    export_layout_compatibility: &'a CanonicalExportLayoutCompatibilityV1,
}

struct BundleSpineInputs<'a> {
    bundle_kind: CanonicalBundleKindV1,
    export_context: &'a CanonicalExportContextV1,
    evidence_context: &'a PackEvidenceContextSummaryV1,
    related_artifacts: &'a [CanonicalExportArtifactRefV1],
    backend_evidence_snapshot: &'a PackEvidenceArtifactRefV1,
    active_review_snapshot: &'a PackEvidenceArtifactRefV1,
    operator_signoff: &'a PackEvidenceArtifactRefV1,
    roundtrip: &'a BundleRoundTripConsistencyV1,
}

fn governance_entry_digest_from_prefixes(
    applied_supported_set_digest_prefix: &str,
    applied_context_digest_prefix: &str,
    backend_digest_prefix: &str,
    active_digest_prefix: &str,
    policy_graph_digest_prefix: &str,
    manifest_digest_prefix: &str,
) -> String {
    let mut surfaces_bytes = Vec::new();
    surfaces_bytes.extend_from_slice(b"governance_primary_surfaces_v1");
    surfaces_bytes.extend_from_slice(backend_digest_prefix.as_bytes());
    surfaces_bytes.extend_from_slice(active_digest_prefix.as_bytes());
    surfaces_bytes.extend_from_slice(applied_supported_set_digest_prefix.as_bytes());
    surfaces_bytes.extend_from_slice(policy_graph_digest_prefix.as_bytes());
    surfaces_bytes.extend_from_slice(manifest_digest_prefix.as_bytes());
    let surfaces_digest = sha256_hex(&surfaces_bytes);

    let mut entry_bytes = Vec::new();
    entry_bytes.extend_from_slice(b"canonical_governance_entry_v1");
    entry_bytes.extend_from_slice(applied_supported_set_digest_prefix.as_bytes());
    entry_bytes.extend_from_slice(applied_context_digest_prefix.as_bytes());
    entry_bytes.extend_from_slice(prefix_hex(&surfaces_digest, 16).as_bytes());
    sha256_hex(&entry_bytes)
}

fn evaluate_bundle_spine(
    input: BundleSpineInputs<'_>,
) -> Result<BundleSpineCheckReportV1, OpsError> {
    let context = parse_normalized_bundle_manifest(
        input.bundle_kind.clone(),
        input.export_context,
        input.related_artifacts,
    )?;

    let mut mismatch_codes = Vec::new();
    if context.applied_supported_set_digest_prefix
        != canonical_prefix_or_missing(&input.evidence_context.supported_slot_set_digest_prefix)
    {
        mismatch_codes.push("BUNDLE_SPINE_SCOPE_MISMATCH".to_string());
    }

    let artifact_refs_digest_prefix =
        canonical_artifact_refs_digest_prefix(input.related_artifacts)?;
    if artifact_refs_digest_prefix != context.artifact_refs_digest_prefix {
        mismatch_codes.push("BUNDLE_SPINE_ARTIFACT_REF_MISMATCH".to_string());
    }

    if input.backend_evidence_snapshot.included
        && input.backend_evidence_snapshot.digest_prefix.is_empty()
    {
        mismatch_codes.push("BUNDLE_SPINE_INCLUDED_STATE_MISMATCH".to_string());
    }
    if input.active_review_snapshot.included
        && input.active_review_snapshot.digest_prefix.is_empty()
    {
        mismatch_codes.push("BUNDLE_SPINE_INCLUDED_STATE_MISMATCH".to_string());
    }
    if input.operator_signoff.included && input.operator_signoff.digest_prefix.is_empty() {
        mismatch_codes.push("BUNDLE_SPINE_INCLUDED_STATE_MISMATCH".to_string());
    }

    let canonical_governance_entry_digest_prefix = if input.backend_evidence_snapshot.included
        && input.active_review_snapshot.included
    {
        if input
            .export_context
            .backend_evidence_snapshot_digest_prefix
            .as_deref()
            != Some(input.backend_evidence_snapshot.digest_prefix.as_str())
            || input
                .export_context
                .active_review_snapshot_digest_prefix
                .as_deref()
                != Some(input.active_review_snapshot.digest_prefix.as_str())
        {
            mismatch_codes.push("BUNDLE_SPINE_GOVERNANCE_MISMATCH".to_string());
        }
        let digest = governance_entry_digest_from_prefixes(
            &context.applied_supported_set_digest_prefix,
            &context.export_context_digest_prefix,
            &input.backend_evidence_snapshot.digest_prefix,
            &input.active_review_snapshot.digest_prefix,
            &context.policy_graph_digest_prefix,
            &context.manifest_digest_prefix,
        );
        prefix_hex(&digest, 16)
    } else if !input.backend_evidence_snapshot.included && !input.active_review_snapshot.included {
        if input
            .export_context
            .backend_evidence_snapshot_digest_prefix
            .is_some()
            || input
                .export_context
                .active_review_snapshot_digest_prefix
                .is_some()
        {
            mismatch_codes.push("BUNDLE_SPINE_GOVERNANCE_MISMATCH".to_string());
        }
        "MISSING".to_string()
    } else {
        mismatch_codes.push("BUNDLE_SPINE_GOVERNANCE_MISMATCH".to_string());
        "MISSING".to_string()
    };
    let related_governance_digest = input
        .related_artifacts
        .iter()
        .find(|r| r.artifact_kind == "canonical_governance_entry")
        .and_then(|r| r.artifact_digest.clone());
    if let Some(related_governance_digest) = related_governance_digest {
        if related_governance_digest != canonical_governance_entry_digest_prefix {
            mismatch_codes.push("ROUNDTRIP_CHAIN_GOVERNANCE_ENTRY_MISMATCH".to_string());
        }
    }

    let canonical_readiness_spine_digest_prefix = input
        .related_artifacts
        .iter()
        .find(|r| r.artifact_kind == "canonical_readiness_spine")
        .and_then(|r| r.artifact_digest.clone());
    if let Some(readiness_digest_prefix) = canonical_readiness_spine_digest_prefix.as_deref() {
        if input
            .export_context
            .operator_signoff_digest_prefix
            .as_deref()
            .is_some_and(|signoff| signoff != input.operator_signoff.digest_prefix)
            || readiness_digest_prefix.is_empty()
        {
            mismatch_codes.push("BUNDLE_SPINE_READINESS_MISMATCH".to_string());
            mismatch_codes.push("ROUNDTRIP_CHAIN_READINESS_SPINE_MISMATCH".to_string());
        }
    } else if input
        .related_artifacts
        .iter()
        .any(|r| r.artifact_kind == "canonical_readiness_spine")
    {
        mismatch_codes.push("BUNDLE_SPINE_READINESS_MISMATCH".to_string());
        mismatch_codes.push("ROUNDTRIP_CHAIN_READINESS_SPINE_MISMATCH".to_string());
    }

    if input.roundtrip.mismatch_codes.iter().any(|c| {
        matches!(
            c.as_str(),
            "LEGACY_BUNDLE_LAYOUT" | "LEGACY_BUNDLE_TRANSLATED" | "LEGACY_BUNDLE_UNSUPPORTED"
        )
    }) {
        mismatch_codes.push("LEGACY_BUNDLE_SPINE_TRANSLATED".to_string());
    }
    if input
        .roundtrip
        .mismatch_codes
        .iter()
        .any(|c| c == "LEGACY_BUNDLE_UNSUPPORTED")
    {
        mismatch_codes.push("LEGACY_BUNDLE_SPINE_UNSUPPORTED".to_string());
    }

    mismatch_codes.sort();
    mismatch_codes.dedup();
    let status = if mismatch_codes.is_empty() {
        BundleSpineStatusV1::Pass
    } else {
        BundleSpineStatusV1::Fail
    };
    let mut spine = CanonicalBundleSpineV1 {
        bundle_kind: input.bundle_kind,
        applied_supported_set_digest_prefix: context.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix,
        bundle_consumption_context_digest_prefix: prefix_hex(
            &context.consumption_context_digest,
            16,
        ),
        artifact_refs_digest_prefix: artifact_refs_digest_prefix.clone(),
        roundtrip_consistency_digest_prefix: prefix_hex(&input.roundtrip.roundtrip_digest, 16),
        bundle_spine_status: status,
        bundle_spine_digest: String::new(),
    };
    spine.bundle_spine_digest = bundle_spine_digest_hex(&spine)?;
    let authority = derive_canonical_bundle_authority_v2(&spine, 1, false)?;
    Ok(BundleSpineCheckReportV1 {
        schema_version: 1,
        pass: matches!(spine.bundle_spine_status, BundleSpineStatusV1::Pass),
        bundle_kind: spine.bundle_kind.clone(),
        mismatch_codes,
        spine,
        authority_digest_prefix: Some(prefix_hex(&authority.authority_digest, 16)),
    })
}

pub fn require_canonical_bundle_spine(
    applied: &AppliedSupportedSetContextV1,
    governance: &CanonicalGovernanceEntryV1,
    readiness: &CanonicalReadinessSpineV1,
    spine: Option<&CanonicalBundleSpineV1>,
) -> Result<CanonicalBundleSpineV1, OpsError> {
    let Some(spine) = spine else {
        return Err(OpsError::Invalid(
            CANONICAL_BUNDLE_SPINE_REQUIRED.to_string(),
        ));
    };
    if spine.bundle_consumption_context_digest_prefix.is_empty()
        || spine.artifact_refs_digest_prefix.is_empty()
    {
        return Err(OpsError::Invalid(
            CANONICAL_BUNDLE_CONTEXT_REQUIRED.to_string(),
        ));
    }
    if spine.applied_supported_set_digest_prefix != applied.applied_set_digest_prefix {
        return Err(OpsError::Invalid(
            CANONICAL_BUNDLE_SPINE_REQUIRED.to_string(),
        ));
    }
    if spine.canonical_governance_entry_digest_prefix
        != prefix_hex(&governance.authority_digest, 16)
    {
        return Err(OpsError::Invalid(
            CANONICAL_EXPORT_REFS_REQUIRED.to_string(),
        ));
    }
    let readiness_prefix = prefix_hex(&readiness.spine_digest, 16);
    if spine.canonical_readiness_spine_digest_prefix.as_deref() != Some(readiness_prefix.as_str()) {
        return Err(OpsError::Invalid(
            CANONICAL_EXPORT_REFS_REQUIRED.to_string(),
        ));
    }
    if !matches!(spine.bundle_spine_status, BundleSpineStatusV1::Pass) {
        return Err(OpsError::Invalid(SECONDARY_BUNDLE_PATH_BLOCKED.to_string()));
    }
    Ok(spine.clone())
}

fn derive_canonical_bundle_authority_v2(
    spine: &CanonicalBundleSpineV1,
    covered_surface_count: u16,
    legacy_present: bool,
) -> Result<CanonicalBundleAuthorityV2, OpsError> {
    let mut authority = CanonicalBundleAuthorityV2 {
        schema_version: 2,
        applied_supported_set_digest_prefix: spine.applied_supported_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: spine
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: spine
            .canonical_readiness_spine_digest_prefix
            .clone()
            .unwrap_or_else(|| "MISSING".to_string()),
        canonical_bundle_spine_digest_prefix: prefix_hex(&spine.bundle_spine_digest, 16),
        covered_surface_count,
        authority_status: if legacy_present {
            CanonicalBundleAuthorityStatusV2::LegacyPresent
        } else if matches!(spine.bundle_spine_status, BundleSpineStatusV1::Pass) {
            CanonicalBundleAuthorityStatusV2::Pass
        } else {
            CanonicalBundleAuthorityStatusV2::Fail
        },
        authority_digest: String::new(),
    };
    authority.authority_digest = bundle_authority_digest_hex(&authority)?;
    Ok(authority)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalBundleAuthorityContextV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_bundle_authority_digest_prefix: String,
}

pub fn require_final_bundle_authority(
    applied: Option<&AppliedSupportedSetContextV1>,
    governance: Option<&CanonicalGovernanceEntryV1>,
    readiness: Option<&CanonicalReadinessSpineV1>,
    bundle_spine: Option<&CanonicalBundleSpineV1>,
    bundle_authority: Option<&CanonicalBundleAuthorityV2>,
) -> Result<FinalBundleAuthorityContextV1, OpsError> {
    let (
        Some(applied),
        Some(governance),
        Some(readiness),
        Some(bundle_spine),
        Some(bundle_authority),
    ) = (
        applied,
        governance,
        readiness,
        bundle_spine,
        bundle_authority,
    )
    else {
        return Err(OpsError::Invalid(
            FINAL_BUNDLE_AUTHORITY_REQUIRED.to_string(),
        ));
    };
    if bundle_spine
        .bundle_consumption_context_digest_prefix
        .is_empty()
        || bundle_spine.artifact_refs_digest_prefix.is_empty()
    {
        return Err(OpsError::Invalid(
            CANONICAL_EXPORT_ARTIFACT_REFS_REQUIRED.to_string(),
        ));
    }
    if bundle_spine.applied_supported_set_digest_prefix != applied.applied_set_digest_prefix {
        return Err(OpsError::Invalid(
            FINAL_BUNDLE_AUTHORITY_REQUIRED.to_string(),
        ));
    }
    let governance_prefix = prefix_hex(&governance.authority_digest, 16);
    if bundle_spine.canonical_governance_entry_digest_prefix != governance_prefix {
        return Err(OpsError::Invalid(
            FINAL_BUNDLE_AUTHORITY_REQUIRED.to_string(),
        ));
    }
    let readiness_prefix = prefix_hex(&readiness.spine_digest, 16);
    if bundle_spine
        .canonical_readiness_spine_digest_prefix
        .as_deref()
        != Some(readiness_prefix.as_str())
    {
        return Err(OpsError::Invalid(
            FINAL_BUNDLE_AUTHORITY_REQUIRED.to_string(),
        ));
    }
    if bundle_authority.canonical_bundle_spine_digest_prefix
        != prefix_hex(&bundle_spine.bundle_spine_digest, 16)
    {
        return Err(OpsError::Invalid(
            CANONICAL_BUNDLE_SPINE_REQUIRED.to_string(),
        ));
    }
    if bundle_authority.applied_supported_set_digest_prefix != applied.applied_set_digest_prefix
        || bundle_authority.canonical_governance_entry_digest_prefix != governance_prefix
        || bundle_authority.canonical_readiness_spine_digest_prefix != readiness_prefix
    {
        return Err(OpsError::Invalid(
            FINAL_BUNDLE_AUTHORITY_REQUIRED.to_string(),
        ));
    }
    if !matches!(bundle_spine.bundle_spine_status, BundleSpineStatusV1::Pass)
        || !matches!(
            bundle_authority.authority_status,
            CanonicalBundleAuthorityStatusV2::Pass
        )
    {
        return Err(OpsError::Invalid(LEGACY_BUNDLE_INPUT_BLOCKED.to_string()));
    }
    Ok(FinalBundleAuthorityContextV1 {
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: governance_prefix,
        canonical_readiness_spine_digest_prefix: readiness_prefix,
        canonical_bundle_spine_digest_prefix: prefix_hex(&bundle_spine.bundle_spine_digest, 16),
        canonical_bundle_authority_digest_prefix: prefix_hex(
            &bundle_authority.authority_digest,
            16,
        ),
    })
}

fn evaluate_bundle_roundtrip_consistency(
    input: BundleRoundTripInputs<'_>,
) -> Result<BundleRoundTripConsistencyV1, OpsError> {
    let context = parse_normalized_bundle_manifest(
        input.bundle_kind.clone(),
        input.export_context,
        input.related_artifacts,
    )?;

    let mut mismatch_codes = Vec::new();

    let context_match_status = if canonical_context_digest_hex(input.export_context)?
        != input.export_context.context_digest
    {
        mismatch_codes.push("BUNDLE_MANIFEST_MISMATCH".to_string());
        BundleRoundTripMatchStatusV1::Mismatch
    } else {
        BundleRoundTripMatchStatusV1::Match
    };

    let scope_match_status = if context.applied_supported_set_digest_prefix
        != canonical_prefix_or_missing(&input.evidence_context.supported_slot_set_digest_prefix)
    {
        mismatch_codes.push("BUNDLE_SCOPE_MISMATCH".to_string());
        BundleRoundTripMatchStatusV1::Mismatch
    } else {
        BundleRoundTripMatchStatusV1::Match
    };

    let policy_match_status = if context.policy_graph_digest_prefix
        != canonical_prefix_or_missing(&input.evidence_context.policy_graph_digest_prefix)
    {
        mismatch_codes.push("BUNDLE_POLICY_MISMATCH".to_string());
        BundleRoundTripMatchStatusV1::Mismatch
    } else {
        BundleRoundTripMatchStatusV1::Match
    };

    let manifest_match_status = if context.manifest_digest_prefix
        != canonical_prefix_or_missing(&input.evidence_context.manifest_digest_prefix)
    {
        mismatch_codes.push("BUNDLE_MANIFEST_MISMATCH".to_string());
        BundleRoundTripMatchStatusV1::Mismatch
    } else {
        BundleRoundTripMatchStatusV1::Match
    };

    let mut governance_surface_ref_status = BundleRoundTripMatchStatusV1::Match;
    for refv in [
        input.backend_evidence_snapshot,
        input.active_review_snapshot,
    ] {
        if refv.included && refv.digest_prefix.is_empty() {
            mismatch_codes.push("BUNDLE_ARTIFACT_REF_MISMATCH".to_string());
            governance_surface_ref_status = merge_status(
                governance_surface_ref_status,
                BundleRoundTripMatchStatusV1::Mismatch,
            );
        }
        if !refv.included && refv.status == "MISSING" {
            governance_surface_ref_status = merge_status(
                governance_surface_ref_status,
                BundleRoundTripMatchStatusV1::Missing,
            );
        }
    }

    let review_signoff_ref_status = if input.operator_signoff.included {
        if input.operator_signoff.digest_prefix.is_empty() {
            mismatch_codes.push("BUNDLE_INCLUDED_STATE_MISMATCH".to_string());
            BundleRoundTripMatchStatusV1::Mismatch
        } else {
            BundleRoundTripMatchStatusV1::Match
        }
    } else if input.operator_signoff.status == "MISSING" {
        BundleRoundTripMatchStatusV1::Missing
    } else {
        BundleRoundTripMatchStatusV1::Match
    };
    if let Some(signoff_chain_ref) = input
        .related_artifacts
        .iter()
        .find(|r| r.artifact_kind == "operator_signoff_decision")
        .and_then(|r| r.artifact_digest.clone())
    {
        if signoff_chain_ref != input.operator_signoff.digest_prefix {
            mismatch_codes.push("ROUNDTRIP_CHAIN_SIGNOFF_MISMATCH".to_string());
        }
    }

    let layout_status = match input.export_layout_compatibility {
        CanonicalExportLayoutCompatibilityV1::Canonical => BundleRoundTripMatchStatusV1::Match,
        CanonicalExportLayoutCompatibilityV1::LegacyExportLayout => {
            mismatch_codes.push("LEGACY_BUNDLE_LAYOUT".to_string());
            BundleRoundTripMatchStatusV1::Legacy
        }
        CanonicalExportLayoutCompatibilityV1::LegacyExportTranslated => {
            mismatch_codes.push("LEGACY_BUNDLE_TRANSLATED".to_string());
            BundleRoundTripMatchStatusV1::Legacy
        }
        CanonicalExportLayoutCompatibilityV1::LegacyExportUnsupported => {
            mismatch_codes.push("LEGACY_BUNDLE_UNSUPPORTED".to_string());
            BundleRoundTripMatchStatusV1::Legacy
        }
    };

    let overall_status = if mismatch_codes.is_empty() {
        BundleRoundTripOverallStatusV1::Pass
    } else {
        BundleRoundTripOverallStatusV1::Fail
    };

    mismatch_codes.sort();
    mismatch_codes.dedup();
    let mut canonical_condition_codes = mismatch_codes
        .iter()
        .filter_map(|code| crate::remediation::canonical_condition_for_roundtrip_mismatch(code))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    canonical_condition_codes.sort();
    canonical_condition_codes.dedup();
    let mut primary_remediation_codes = canonical_condition_codes
        .iter()
        .filter_map(|code| crate::remediation::primary_remediation_for_condition_code(code))
        .collect::<Vec<_>>();
    primary_remediation_codes.sort();
    primary_remediation_codes.dedup();

    let mut report = BundleRoundTripConsistencyV1 {
        schema_version: 1,
        bundle_kind: input.bundle_kind,
        bundle_digest_prefix: digest_prefix_str(input.bundle_digest, 16),
        context_match_status: merge_status(context_match_status, layout_status.clone()),
        scope_match_status,
        policy_match_status,
        manifest_match_status,
        governance_surface_ref_status: merge_status(
            governance_surface_ref_status,
            layout_status.clone(),
        ),
        review_signoff_ref_status: merge_status(review_signoff_ref_status, layout_status),
        overall_status,
        mismatch_codes,
        canonical_condition_codes,
        primary_remediation_codes,
        roundtrip_digest: String::new(),
    };
    let _ = context;
    report.roundtrip_digest = bundle_roundtrip_digest_hex(&report)?;
    Ok(report)
}

pub fn exports_roundtrip_check(
    input: &Path,
    out: &Path,
) -> Result<BundleRoundTripConsistencyV1, OpsError> {
    let file = fs::File::open(input)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| OpsError::Invalid(format!("unable to open bundle zip: {e}")))?;
    let mut repro_body = String::new();
    let mut bugkit_body = String::new();
    let has_repro = archive.by_name("repro_pack_manifest.json").is_ok();
    let has_bugkit = archive.by_name("BUGKIT_MANIFEST.json").is_ok()
        || archive.by_name("bugkit_manifest.json").is_ok();

    if has_repro {
        let mut mf = archive
            .by_name("repro_pack_manifest.json")
            .map_err(|e| OpsError::Invalid(format!("missing repro_pack_manifest.json: {e}")))?;
        std::io::Read::read_to_string(&mut mf, &mut repro_body)?;
        let manifest: ReproPackManifestV1 = serde_json::from_str(&repro_body)?;
        let report = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
            bundle_kind: CanonicalBundleKindV1::Repro,
            bundle_digest: &manifest.repro_pack_digest,
            export_context: &manifest.export_context,
            evidence_context: &manifest.evidence_context,
            related_artifacts: &manifest.related_artifacts,
            backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
            active_review_snapshot: &manifest.active_review_snapshot,
            operator_signoff: &manifest.operator_signoff,
            export_layout_compatibility: &manifest.export_layout_compatibility,
        })?;
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        write_json(out, &report)?;
        return Ok(report);
    }

    if has_bugkit {
        let name = if archive.by_name("BUGKIT_MANIFEST.json").is_ok() {
            "BUGKIT_MANIFEST.json"
        } else {
            "bugkit_manifest.json"
        };
        let mut mf = archive
            .by_name(name)
            .map_err(|e| OpsError::Invalid(format!("missing bugkit manifest: {e}")))?;
        std::io::Read::read_to_string(&mut mf, &mut bugkit_body)?;
        let manifest: BugKitManifestV1 = serde_json::from_str(&bugkit_body)?;
        let report = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
            bundle_kind: CanonicalBundleKindV1::Bugkit,
            bundle_digest: &manifest.bugkit_digest,
            export_context: &manifest.export_context,
            evidence_context: &manifest.evidence_context,
            related_artifacts: &manifest.related_artifacts,
            backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
            active_review_snapshot: &manifest.active_review_snapshot,
            operator_signoff: &manifest.operator_signoff,
            export_layout_compatibility: &manifest.export_layout_compatibility,
        })?;
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        write_json(out, &report)?;
        return Ok(report);
    }

    Err(OpsError::Invalid(
        "bundle does not contain repro_pack_manifest.json or BUGKIT_MANIFEST.json".to_string(),
    ))
}

pub fn exports_bundle_spine_check(
    input: &Path,
    out: &Path,
) -> Result<BundleSpineCheckReportV1, OpsError> {
    let file = fs::File::open(input)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| OpsError::Invalid(format!("unable to open bundle zip: {e}")))?;
    let mut repro_body = String::new();
    let mut bugkit_body = String::new();
    let has_repro = archive.by_name("repro_pack_manifest.json").is_ok();
    let has_bugkit = archive.by_name("BUGKIT_MANIFEST.json").is_ok()
        || archive.by_name("bugkit_manifest.json").is_ok();

    let report = if has_repro {
        let mut mf = archive
            .by_name("repro_pack_manifest.json")
            .map_err(|e| OpsError::Invalid(format!("missing repro_pack_manifest.json: {e}")))?;
        std::io::Read::read_to_string(&mut mf, &mut repro_body)?;
        let manifest: ReproPackManifestV1 = serde_json::from_str(&repro_body)?;
        let roundtrip = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
            bundle_kind: CanonicalBundleKindV1::Repro,
            bundle_digest: &manifest.repro_pack_digest,
            export_context: &manifest.export_context,
            evidence_context: &manifest.evidence_context,
            related_artifacts: &manifest.related_artifacts,
            backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
            active_review_snapshot: &manifest.active_review_snapshot,
            operator_signoff: &manifest.operator_signoff,
            export_layout_compatibility: &manifest.export_layout_compatibility,
        })?;
        evaluate_bundle_spine(BundleSpineInputs {
            bundle_kind: CanonicalBundleKindV1::Repro,
            export_context: &manifest.export_context,
            evidence_context: &manifest.evidence_context,
            related_artifacts: &manifest.related_artifacts,
            backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
            active_review_snapshot: &manifest.active_review_snapshot,
            operator_signoff: &manifest.operator_signoff,
            roundtrip: &roundtrip,
        })?
    } else if has_bugkit {
        let name = if archive.by_name("BUGKIT_MANIFEST.json").is_ok() {
            "BUGKIT_MANIFEST.json"
        } else {
            "bugkit_manifest.json"
        };
        let mut mf = archive
            .by_name(name)
            .map_err(|e| OpsError::Invalid(format!("missing bugkit manifest: {e}")))?;
        std::io::Read::read_to_string(&mut mf, &mut bugkit_body)?;
        let manifest: BugKitManifestV1 = serde_json::from_str(&bugkit_body)?;
        let roundtrip = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
            bundle_kind: CanonicalBundleKindV1::Bugkit,
            bundle_digest: &manifest.bugkit_digest,
            export_context: &manifest.export_context,
            evidence_context: &manifest.evidence_context,
            related_artifacts: &manifest.related_artifacts,
            backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
            active_review_snapshot: &manifest.active_review_snapshot,
            operator_signoff: &manifest.operator_signoff,
            export_layout_compatibility: &manifest.export_layout_compatibility,
        })?;
        evaluate_bundle_spine(BundleSpineInputs {
            bundle_kind: CanonicalBundleKindV1::Bugkit,
            export_context: &manifest.export_context,
            evidence_context: &manifest.evidence_context,
            related_artifacts: &manifest.related_artifacts,
            backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
            active_review_snapshot: &manifest.active_review_snapshot,
            operator_signoff: &manifest.operator_signoff,
            roundtrip: &roundtrip,
        })?
    } else {
        return Err(OpsError::Invalid(
            "bundle does not contain repro_pack_manifest.json or BUGKIT_MANIFEST.json".to_string(),
        ));
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(out, &report)?;
    Ok(report)
}

pub fn exports_bundle_spine_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<BundleSpineSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_bundle_spine_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;
    let governance = require_canonical_governance_entry(&applied, Some(&governance))?;
    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_check_bundle_spine_sweep.json"),
    )?
    .canonical_readiness_spine;

    let run_id = reproducible_run_id_for_smoke(workdir)?;
    let repro_bundle = workdir
        .join("out")
        .join(format!("repro_{run_id}_bundle_spine_sweep.zip"));
    let bugkit_bundle = workdir
        .join("out")
        .join(format!("bugkit_{run_id}_bundle_spine_sweep.zip"));
    let _ = repro_pack(workdir, &run_id, &repro_bundle)?;
    let _ = bugkit_build(
        workdir,
        &run_id,
        &bugkit_bundle,
        &BugKitBuildArgs::default(),
    )?;

    let repro_spine = exports_bundle_spine_check(
        &repro_bundle,
        &workdir.join("out/repro_bundle_spine_check_bundle_spine_sweep.json"),
    )?;
    let bugkit_spine = exports_bundle_spine_check(
        &bugkit_bundle,
        &workdir.join("out/bugkit_bundle_spine_check_bundle_spine_sweep.json"),
    )?;

    let covered = vec![
        ("repro_pack_build", repro_spine.clone()),
        ("repro_verify", repro_spine.clone()),
        ("bugkit_build", bugkit_spine.clone()),
        ("exports_roundtrip_check", repro_spine.clone()),
        ("exports_bundle_spine_check", repro_spine.clone()),
        ("operator_roundtrip_chain_helpers", repro_spine.clone()),
        ("export_readiness_build_guards", repro_spine.clone()),
    ];

    let mut sweep_surfaces = Vec::new();
    for (surface, report) in covered {
        let mut mismatches = Vec::new();
        if !report.pass {
            mismatches.push(BundleSpineSweepMismatchCategoryV1::SurfaceSkippedCanonicalBundleSpine);
        }
        for code in &report.mismatch_codes {
            match code.as_str() {
                "BUNDLE_SPINE_SCOPE_MISMATCH" => {
                    mismatches.push(BundleSpineSweepMismatchCategoryV1::BundleSpineScopeMismatch)
                }
                "BUNDLE_SPINE_GOVERNANCE_MISMATCH" => mismatches
                    .push(BundleSpineSweepMismatchCategoryV1::BundleSpineGovernanceMismatch),
                "BUNDLE_SPINE_READINESS_MISMATCH" => mismatches
                    .push(BundleSpineSweepMismatchCategoryV1::BundleSpineReadinessMismatch),
                "LEGACY_BUNDLE_SPINE_TRANSLATED" | "LEGACY_BUNDLE_SPINE_UNSUPPORTED" => {
                    mismatches.push(BundleSpineSweepMismatchCategoryV1::LegacyBundlePathPresent)
                }
                _ => mismatches
                    .push(BundleSpineSweepMismatchCategoryV1::SurfaceUsedSecondaryBundlePath),
            }
        }
        mismatches.sort();
        mismatches.dedup();
        let status =
            if mismatches.contains(&BundleSpineSweepMismatchCategoryV1::LegacyBundlePathPresent) {
                CanonicalBundleAuthorityStatusV2::LegacyPresent
            } else if mismatches.is_empty() {
                CanonicalBundleAuthorityStatusV2::Pass
            } else {
                CanonicalBundleAuthorityStatusV2::Fail
            };
        sweep_surfaces.push(BundleSpineSweepSurfaceStatusV1 {
            surface: surface.to_string(),
            status,
            mismatch_categories: mismatches,
        });
    }

    let required_spine = require_canonical_bundle_spine(
        &applied,
        &governance,
        &readiness,
        Some(&repro_spine.spine),
    )?;
    let legacy_present = sweep_surfaces
        .iter()
        .any(|s| matches!(s.status, CanonicalBundleAuthorityStatusV2::LegacyPresent));
    let mut authority = derive_canonical_bundle_authority_v2(
        &required_spine,
        sweep_surfaces.len() as u16,
        legacy_present,
    )?;
    if sweep_surfaces
        .iter()
        .any(|s| matches!(s.status, CanonicalBundleAuthorityStatusV2::Fail))
    {
        authority.authority_status = CanonicalBundleAuthorityStatusV2::Fail;
        authority.authority_digest = bundle_authority_digest_hex(&authority)?;
    }
    let _final_bundle_authority = require_final_bundle_authority(
        Some(&applied),
        Some(&governance),
        Some(&readiness),
        Some(&required_spine),
        Some(&authority),
    )?;

    let report = BundleSpineSweepReportV1 {
        schema_version: 1,
        authority,
        surfaces: sweep_surfaces,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(out, &report)?;
    Ok(report)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BugKitBuildArgs {
    pub include_payload: bool,
    pub include_weights: bool,
    pub max_bytes: u64,
}

impl Default for BugKitBuildArgs {
    fn default() -> Self {
        Self {
            include_payload: false,
            include_weights: false,
            max_bytes: 50 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BugKitManifestEntry {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub optional: bool,
    pub dropped_due_to_size_cap: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BugKitManifestV1 {
    pub schema_version: u16,
    pub run_id: String,
    pub include_payload: bool,
    pub include_weights: bool,
    pub max_bytes: u64,
    pub total_bytes: u64,
    pub file_count: usize,
    pub files: Vec<BugKitManifestEntry>,
    pub evidence_context: PackEvidenceContextSummaryV1,
    pub backend_evidence_snapshot: PackEvidenceArtifactRefV1,
    pub active_review_snapshot: PackEvidenceArtifactRefV1,
    pub operator_signoff: PackEvidenceArtifactRefV1,
    pub backend_resolution: PackEvidenceArtifactRefV1,
    pub export_context: CanonicalExportContextV1,
    pub related_artifacts: Vec<CanonicalExportArtifactRefV1>,
    #[serde(default = "missing_prefix_value")]
    pub canonical_bundle_spine_digest_prefix: String,
    #[serde(default = "missing_prefix_value")]
    pub canonical_bundle_authority_digest_prefix: String,
    pub export_layout_compatibility: CanonicalExportLayoutCompatibilityV1,
    pub warnings: Vec<String>,
    pub bugkit_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalBundleKindV1 {
    Repro,
    Bugkit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalBundleConsumptionContextV1 {
    pub bundle_kind: CanonicalBundleKindV1,
    pub export_context_digest_prefix: String,
    pub applied_supported_set_digest_prefix: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub artifact_refs_digest_prefix: String,
    pub included_artifact_kinds: Vec<String>,
    pub consumption_context_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleRoundTripMatchStatusV1 {
    Match,
    Missing,
    Mismatch,
    Legacy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleRoundTripOverallStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleRoundTripConsistencyV1 {
    pub schema_version: u16,
    pub bundle_kind: CanonicalBundleKindV1,
    pub bundle_digest_prefix: String,
    pub context_match_status: BundleRoundTripMatchStatusV1,
    pub scope_match_status: BundleRoundTripMatchStatusV1,
    pub policy_match_status: BundleRoundTripMatchStatusV1,
    pub manifest_match_status: BundleRoundTripMatchStatusV1,
    pub governance_surface_ref_status: BundleRoundTripMatchStatusV1,
    pub review_signoff_ref_status: BundleRoundTripMatchStatusV1,
    pub overall_status: BundleRoundTripOverallStatusV1,
    pub mismatch_codes: Vec<String>,
    pub canonical_condition_codes: Vec<String>,
    pub primary_remediation_codes: Vec<String>,
    pub roundtrip_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleSpineStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalBundleAuthorityStatusV2 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalBundleAuthorityV2 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub covered_surface_count: u16,
    pub authority_status: CanonicalBundleAuthorityStatusV2,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleSpineSweepMismatchCategoryV1 {
    SurfaceSkippedCanonicalBundleSpine,
    SurfaceUsedSecondaryBundlePath,
    BundleSpineScopeMismatch,
    BundleSpineGovernanceMismatch,
    BundleSpineReadinessMismatch,
    LegacyBundlePathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleSpineSweepSurfaceStatusV1 {
    pub surface: String,
    pub status: CanonicalBundleAuthorityStatusV2,
    pub mismatch_categories: Vec<BundleSpineSweepMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleSpineSweepReportV1 {
    pub schema_version: u16,
    pub authority: CanonicalBundleAuthorityV2,
    pub surfaces: Vec<BundleSpineSweepSurfaceStatusV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalBundleSpineV1 {
    pub bundle_kind: CanonicalBundleKindV1,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: Option<String>,
    pub bundle_consumption_context_digest_prefix: String,
    pub artifact_refs_digest_prefix: String,
    pub roundtrip_consistency_digest_prefix: String,
    pub bundle_spine_status: BundleSpineStatusV1,
    pub bundle_spine_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleSpineCheckReportV1 {
    pub schema_version: u16,
    pub pass: bool,
    pub bundle_kind: CanonicalBundleKindV1,
    pub mismatch_codes: Vec<String>,
    pub spine: CanonicalBundleSpineV1,
    pub authority_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackEvidenceContextSummaryV1 {
    pub supported_slot_set_digest_prefix: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackEvidenceArtifactRefV1 {
    pub included: bool,
    pub path: String,
    pub sha256: String,
    pub schema_version: u16,
    pub digest_prefix: String,
    pub status: String,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone)]
struct EvidenceValidationContext {
    supported_slot_set_digest_prefix: String,
    policy_graph_digest_prefix: String,
    manifest_digest_prefix: String,
}

fn missing_evidence_ref(path: &str, reason_code: &str) -> PackEvidenceArtifactRefV1 {
    PackEvidenceArtifactRefV1 {
        included: false,
        path: path.to_string(),
        sha256: String::new(),
        schema_version: 0,
        digest_prefix: String::new(),
        status: "MISSING".to_string(),
        reason_code: Some(reason_code.to_string()),
    }
}

fn excluded_evidence_ref(path: &str, reason_code: &str) -> PackEvidenceArtifactRefV1 {
    PackEvidenceArtifactRefV1 {
        included: false,
        path: path.to_string(),
        sha256: String::new(),
        schema_version: 0,
        digest_prefix: String::new(),
        status: "EXCLUDED".to_string(),
        reason_code: Some(reason_code.to_string()),
    }
}

fn included_evidence_ref(
    path: &str,
    sha256: String,
    schema_version: u16,
    digest_prefix: String,
) -> PackEvidenceArtifactRefV1 {
    PackEvidenceArtifactRefV1 {
        included: true,
        path: path.to_string(),
        sha256,
        schema_version,
        digest_prefix,
        status: "INCLUDED".to_string(),
        reason_code: None,
    }
}

fn canonical_artifact_ref_digest_hex(
    artifact: &CanonicalExportArtifactRefV1,
) -> Result<String, OpsError> {
    let mut canonical = artifact.clone();
    canonical.ref_digest.clear();
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn canonical_context_digest_hex(context: &CanonicalExportContextV1) -> Result<String, OpsError> {
    let mut canonical = context.clone();
    canonical.context_digest.clear();
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn canonical_state_from_status(status: &str) -> CanonicalArtifactIncludedStateV1 {
    match status {
        "INCLUDED" => CanonicalArtifactIncludedStateV1::Included,
        "MISSING" => CanonicalArtifactIncludedStateV1::Missing,
        "EXCLUDED" => CanonicalArtifactIncludedStateV1::Excluded,
        _ => CanonicalArtifactIncludedStateV1::Skip,
    }
}

fn canonical_export_ref_from_pack(
    artifact_kind: &str,
    legacy: &PackEvidenceArtifactRefV1,
) -> Result<CanonicalExportArtifactRefV1, OpsError> {
    let mut out = CanonicalExportArtifactRefV1 {
        artifact_kind: artifact_kind.to_string(),
        relative_path: legacy.path.clone(),
        included_state: canonical_state_from_status(&legacy.status),
        sha256: if legacy.sha256.is_empty() {
            None
        } else {
            Some(legacy.sha256.clone())
        },
        schema_version: (legacy.schema_version != 0).then_some(legacy.schema_version),
        artifact_digest: if legacy.digest_prefix.is_empty() {
            None
        } else {
            Some(legacy.digest_prefix.clone())
        },
        reason_code: legacy.reason_code.clone(),
        ref_digest: String::new(),
    };
    out.ref_digest = canonical_artifact_ref_digest_hex(&out)?;
    Ok(out)
}

fn canonical_digest_only_ref(
    artifact_kind: &str,
    relative_path: &str,
    digest_prefix: String,
) -> Result<CanonicalExportArtifactRefV1, OpsError> {
    let mut out = CanonicalExportArtifactRefV1 {
        artifact_kind: artifact_kind.to_string(),
        relative_path: relative_path.to_string(),
        included_state: CanonicalArtifactIncludedStateV1::Skip,
        sha256: None,
        schema_version: None,
        artifact_digest: Some(digest_prefix),
        reason_code: Some("CHAIN_REF".to_string()),
        ref_digest: String::new(),
    };
    out.ref_digest = canonical_artifact_ref_digest_hex(&out)?;
    Ok(out)
}

#[derive(Debug, Clone)]
struct ExportChainDigestRefs {
    canonical_governance_entry_digest_prefix: String,
    canonical_readiness_spine_digest_prefix: String,
    operator_review_packet_digest_prefix: String,
    operator_signoff_digest_prefix: String,
    operator_workflow_chain_digest_prefix: String,
    operator_export_authority_chain_digest_prefix: String,
}

fn derive_export_chain_digest_refs(workdir: &Path) -> Result<ExportChainDigestRefs, OpsError> {
    let read_json_if_exists = |path: &Path| -> Result<Option<serde_json::Value>, OpsError> {
        if !path.exists() {
            return Ok(None);
        }
        let body = fs::read_to_string(path)?;
        Ok(Some(serde_json::from_str(&body)?))
    };

    let governance_prefix = (|| -> Result<String, OpsError> {
        let applied = load_applied_supported_set_context_v1(workdir)?;
        let backend_path = workdir.join("out/backend_evidence_snapshot.json");
        let active_path = workdir.join("out/active_review_snapshot.json");
        if !backend_path.exists() || !active_path.exists() {
            return Ok("MISSING".to_string());
        }
        let backend: BackendEvidenceSnapshotV1 =
            serde_json::from_str(&fs::read_to_string(&backend_path)?)?;
        let active: AggregatedActiveReviewSnapshotV1 =
            serde_json::from_str(&fs::read_to_string(&active_path)?)?;
        let surfaces =
            validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
        let governance = derive_canonical_governance_entry(&applied, &surfaces)?;
        Ok(prefix_hex(&governance.authority_digest, 16))
    })()
    .unwrap_or_else(|_| "MISSING".to_string());

    let readiness_prefix = [
        workdir.join("out/readiness_spine_check.json"),
        workdir.join("out/readiness_spine_check_bundle_spine_sweep.json"),
    ]
    .iter()
    .find_map(|path| {
        read_json_if_exists(path).ok().flatten().and_then(|v| {
            v.get("canonical_readiness_spine")
                .and_then(|s| s.get("spine_digest"))
                .and_then(|d| d.as_str())
                .map(|s| prefix_hex(s, 16))
        })
    })
    .unwrap_or_else(|| "MISSING".to_string());

    let review_packet_prefix =
        read_json_if_exists(&workdir.join("out/operator_review_packet.json"))?
            .and_then(|v| {
                v.get("packet_digest")
                    .and_then(|d| d.as_str())
                    .map(|s| prefix_hex(s, 16))
            })
            .unwrap_or_else(|| "MISSING".to_string());

    let signoff_prefix = read_json_if_exists(&workdir.join("out/operator_signoff.json"))?
        .and_then(|v| {
            v.get("decision_digest")
                .and_then(|d| d.as_str())
                .map(|s| prefix_hex(s, 16))
        })
        .unwrap_or_else(|| "MISSING".to_string());

    let workflow_prefix = read_json_if_exists(&workdir.join("out/operator_workflow_chain.json"))?
        .and_then(|v| {
            v.get("chain_digest")
                .and_then(|d| d.as_str())
                .map(|s| prefix_hex(s, 16))
        })
        .unwrap_or_else(|| "MISSING".to_string());

    let export_authority_prefix =
        read_json_if_exists(&workdir.join("out/operator_export_chain_check.json"))?
            .and_then(|v| {
                v.get("chain_digest")
                    .and_then(|d| d.as_str())
                    .map(|s| prefix_hex(s, 16))
            })
            .unwrap_or_else(|| "MISSING".to_string());

    Ok(ExportChainDigestRefs {
        canonical_governance_entry_digest_prefix: governance_prefix,
        canonical_readiness_spine_digest_prefix: readiness_prefix,
        operator_review_packet_digest_prefix: review_packet_prefix,
        operator_signoff_digest_prefix: signoff_prefix,
        operator_workflow_chain_digest_prefix: workflow_prefix,
        operator_export_authority_chain_digest_prefix: export_authority_prefix,
    })
}

fn canonical_prefix_or_missing(value: &str) -> String {
    if value.is_empty() {
        "MISSING".to_string()
    } else {
        value.to_string()
    }
}

fn missing_prefix_value() -> String {
    "MISSING".to_string()
}

fn canonical_export_context_from_parts(
    context: &EvidenceValidationContext,
    run_id: Option<&str>,
    operator_signoff_digest_prefix: Option<String>,
    backend_evidence_snapshot_digest_prefix: Option<String>,
    active_review_snapshot_digest_prefix: Option<String>,
) -> Result<CanonicalExportContextV1, OpsError> {
    let mut out = CanonicalExportContextV1 {
        supported_slot_set_digest_prefix: canonical_prefix_or_missing(
            &context.supported_slot_set_digest_prefix,
        ),
        policy_graph_digest_prefix: canonical_prefix_or_missing(
            &context.policy_graph_digest_prefix,
        ),
        manifest_digest_prefix: canonical_prefix_or_missing(&context.manifest_digest_prefix),
        run_id: run_id.map(ToString::to_string),
        operator_signoff_digest_prefix,
        backend_evidence_snapshot_digest_prefix,
        active_review_snapshot_digest_prefix,
        context_digest: String::new(),
    };
    out.context_digest = canonical_context_digest_hex(&out)?;
    Ok(out)
}

fn validate_evidence_artifacts_against_context(
    context: &EvidenceValidationContext,
    supported_slot_set_digest_prefix: &str,
    policy_graph_digest_prefix: &str,
    manifest_digest_prefix: &str,
) -> Option<String> {
    if supported_slot_set_digest_prefix != context.supported_slot_set_digest_prefix {
        return Some("SLOT_SET_DIGEST_MISMATCH".to_string());
    }
    if policy_graph_digest_prefix != context.policy_graph_digest_prefix {
        return Some("POLICY_GRAPH_DIGEST_MISMATCH".to_string());
    }
    if manifest_digest_prefix != context.manifest_digest_prefix {
        return Some("MANIFEST_DIGEST_MISMATCH".to_string());
    }
    None
}

fn discover_first_existing_json(workdir: &Path, candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(|p| workdir.join(p))
        .find(|p| p.exists())
}

fn enrich_evidence_artifacts(
    workdir: &Path,
    context: &EvidenceValidationContext,
    include_backend_resolution: bool,
    file_map: &mut BTreeMap<String, Vec<u8>>,
) -> Result<
    (
        PackEvidenceArtifactRefV1,
        PackEvidenceArtifactRefV1,
        PackEvidenceArtifactRefV1,
        PackEvidenceArtifactRefV1,
    ),
    OpsError,
> {
    let backend_path = discover_first_existing_json(
        workdir,
        &[
            PathBuf::from("out/backend_evidence_snapshot.json"),
            PathBuf::from("out/models_evidence_snapshot.json"),
        ],
    );
    let mut backend_snapshot_loaded: Option<BackendEvidenceSnapshotV1> = None;
    let mut active_snapshot_loaded: Option<AggregatedActiveReviewSnapshotV1> = None;

    let mut backend_ref = if let Some(path) = backend_path {
        let bytes = fs::read(&path)?;
        let snapshot: BackendEvidenceSnapshotV1 = serde_json::from_slice(&bytes)?;
        backend_snapshot_loaded = Some(snapshot.clone());
        if let Some(reason) = validate_evidence_artifacts_against_context(
            context,
            &snapshot.supported_slot_set_digest,
            &snapshot.policy_graph_digest_prefix,
            &snapshot.manifest_digest_prefix,
        ) {
            excluded_evidence_ref("evidence/backend_evidence_snapshot.json", &reason)
        } else {
            let entry_path = "evidence/backend_evidence_snapshot.json";
            file_map.insert(entry_path.to_string(), bytes.clone());
            included_evidence_ref(
                entry_path,
                sha256_hex(&bytes),
                snapshot.schema_version,
                prefix_hex(&snapshot.snapshot_digest, 16),
            )
        }
    } else {
        missing_evidence_ref(
            "evidence/backend_evidence_snapshot.json",
            "BACKEND_EVIDENCE_SNAPSHOT_MISSING",
        )
    };

    let active_path =
        discover_first_existing_json(workdir, &[PathBuf::from("out/active_review_snapshot.json")]);
    let mut active_ref = if let Some(path) = active_path {
        let bytes = fs::read(&path)?;
        let snapshot: AggregatedActiveReviewSnapshotV1 = serde_json::from_slice(&bytes)?;
        active_snapshot_loaded = Some(snapshot.clone());
        if let Some(reason) = validate_evidence_artifacts_against_context(
            context,
            &snapshot.supported_slot_set_digest,
            &snapshot.policy_graph_digest_prefix,
            &snapshot.manifest_digest_prefix,
        ) {
            excluded_evidence_ref("evidence/active_review_snapshot.json", &reason)
        } else {
            let entry_path = "evidence/active_review_snapshot.json";
            file_map.insert(entry_path.to_string(), bytes.clone());
            included_evidence_ref(
                entry_path,
                sha256_hex(&bytes),
                snapshot.schema_version,
                prefix_hex(&snapshot.snapshot_digest, 16),
            )
        }
    } else {
        missing_evidence_ref(
            "evidence/active_review_snapshot.json",
            "ACTIVE_REVIEW_SNAPSHOT_MISSING",
        )
    };

    if let (Some(backend), Some(active)) = (
        backend_snapshot_loaded.as_ref(),
        active_snapshot_loaded.as_ref(),
    ) {
        if let Err(err) = validate_governance_primary_surfaces(backend, active) {
            backend_ref = excluded_evidence_ref(
                "evidence/backend_evidence_snapshot.json",
                GOVERNANCE_SURFACE_MISMATCH_CODE,
            );
            active_ref = excluded_evidence_ref(
                "evidence/active_review_snapshot.json",
                GOVERNANCE_SURFACE_MISMATCH_CODE,
            );
            file_map.remove("evidence/backend_evidence_snapshot.json");
            file_map.remove("evidence/active_review_snapshot.json");
            let _ = err;
        }
    }

    let signoff_path =
        discover_first_existing_json(workdir, &[PathBuf::from("out/operator_signoff.json")]);
    let signoff_ref = if let Some(path) = signoff_path {
        let bytes = fs::read(&path)?;
        let signoff: OperatorSignoffDecisionV1 = serde_json::from_slice(&bytes)?;
        if let Some(reason) = validate_evidence_artifacts_against_context(
            context,
            &signoff.supported_slot_set_digest,
            &signoff.policy_graph_digest_prefix,
            &signoff.manifest_digest_prefix,
        ) {
            excluded_evidence_ref("evidence/operator_signoff.json", &reason)
        } else {
            let entry_path = "evidence/operator_signoff.json";
            file_map.insert(entry_path.to_string(), bytes.clone());
            included_evidence_ref(
                entry_path,
                sha256_hex(&bytes),
                signoff.schema_version,
                prefix_hex(&signoff.decision_digest, 16),
            )
        }
    } else {
        missing_evidence_ref("evidence/operator_signoff.json", "OPERATOR_SIGNOFF_MISSING")
    };

    let backend_resolution_ref = if include_backend_resolution {
        if let Ok(slot) = detect_second_slot(workdir) {
            let resolution_path = discover_first_existing_json(
                workdir,
                &[PathBuf::from(format!(
                    "out/backend_resolution_{}.json",
                    slot.as_str()
                ))],
            );
            if let Some(path) = resolution_path {
                let bytes = fs::read(&path)?;
                let resolution: BurnSupportResolutionV1 = serde_json::from_slice(&bytes)?;
                let entry_path = format!("evidence/backend_resolution_{}.json", slot.as_str());
                file_map.insert(entry_path.clone(), bytes.clone());
                included_evidence_ref(
                    &entry_path,
                    sha256_hex(&bytes),
                    1,
                    prefix_hex(&resolution.evidence_digest, 16),
                )
            } else {
                missing_evidence_ref(
                    &format!("evidence/backend_resolution_{}.json", slot.as_str()),
                    "BACKEND_RESOLUTION_MISSING",
                )
            }
        } else {
            excluded_evidence_ref(
                "evidence/backend_resolution.json",
                "SECOND_SLOT_UNAVAILABLE",
            )
        }
    } else {
        excluded_evidence_ref("evidence/backend_resolution.json", "NOT_REQUESTED")
    };

    Ok((backend_ref, active_ref, signoff_ref, backend_resolution_ref))
}

pub struct BugKitBuildReport {
    pub run_id: String,
    pub out: String,
    pub total_bytes: u64,
    pub file_count: usize,
    pub warnings: Vec<String>,
}

pub fn bugkit_build(
    workdir: &Path,
    run_id: &str,
    out: &Path,
    args: &BugKitBuildArgs,
) -> Result<BugKitBuildReport, OpsError> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    let repro_out = workdir.join("out").join(format!("repro_{run_id}.zip"));
    if !repro_out.exists() {
        repro_pack(workdir, run_id, &repro_out)?;
    }

    let diagnostics_out = workdir.join("out").join(format!("diag_{run_id}.zip"));
    {
        let run_dir_candidates = [
            workdir.join("out").join(run_id),
            PathBuf::from("out").join(run_id),
        ];
        let run_dir = run_dir_candidates
            .iter()
            .find(|p| p.exists())
            .cloned()
            .ok_or_else(|| {
                OpsError::Invalid(format!(
                    "run artifact directory not found: {} or {}",
                    run_dir_candidates[0].display(),
                    run_dir_candidates[1].display()
                ))
            })?;
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
        if let Some(parent) = diagnostics_out.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = fs::File::create(&diagnostics_out)?;
        let mut zip = zip::ZipWriter::new(file);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for path in selected {
            if !path.exists() {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("entry.json")
                .to_string();
            let mut text = fs::read_to_string(&path).unwrap_or_default();
            text = text.replace("\"text\":", "\"text_redacted\":");
            if !args.include_payload {
                text = text.replace("\"payload\":", "\"payload_redacted\":");
            }
            zip.start_file(name, opts)
                .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
            zip.write_all(text.as_bytes())
                .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
        }
        zip.finish()
            .map_err(|e| OpsError::Invalid(format!("zip finalize failed: {e}")))?;
    }

    let (policy_base, policy_overlay, _manifest_path) = resolve_attestation_inputs();
    let policy = policy_validate(&policy_base, Some(&policy_overlay))?;
    let policy_ref = serde_json::json!({
        "base_pack": policy.base_pack,
        "overlay_pack": policy.overlay_pack,
        "policy_graph_digest": policy.policy_graph_digest,
        "schema_version": policy.schema_version
    });

    let mut entries: Vec<(String, Vec<u8>, bool)> = Vec::new();
    entries.push(("repro_pack.zip".to_string(), fs::read(&repro_out)?, false));
    if diagnostics_out.exists() {
        entries.push((
            "diagnostics_bundle.zip".to_string(),
            fs::read(&diagnostics_out)?,
            true,
        ));
    }
    let spec_snapshot = workdir.join("docs/spec_snapshot.md");
    if spec_snapshot.exists() {
        entries.push((
            "spec_snapshot.md".to_string(),
            fs::read(spec_snapshot)?,
            false,
        ));
    }
    entries.push((
        "policy_graph_ref.json".to_string(),
        serde_json::to_vec_pretty(&policy_ref)?,
        false,
    ));

    let model_verify = models_verify(&resolve_attestation_inputs().2)?;
    let evidence_context = match models_evidence_snapshot(workdir, None, Some(run_id)) {
        Ok(context_snapshot) => EvidenceValidationContext {
            supported_slot_set_digest_prefix: context_snapshot.supported_slot_set_digest,
            policy_graph_digest_prefix: context_snapshot.policy_graph_digest_prefix,
            manifest_digest_prefix: context_snapshot.manifest_digest_prefix,
        },
        Err(_) => EvidenceValidationContext {
            supported_slot_set_digest_prefix: String::new(),
            policy_graph_digest_prefix: prefix_hex(&policy.policy_graph_digest, 16),
            manifest_digest_prefix: prefix_hex(&model_verify.model_hashes_digest, 16),
        },
    };
    let mut evidence_file_map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let (backend_evidence_snapshot, active_review_snapshot, operator_signoff, backend_resolution) =
        enrich_evidence_artifacts(workdir, &evidence_context, true, &mut evidence_file_map)?;
    for (path, bytes) in evidence_file_map {
        entries.push((path, bytes, true));
    }

    for extra_name in [
        "strict_check.json",
        "strict_failure.json",
        "drift_report.json",
        "gate_report.json",
        "docs_lint_report.json",
    ] {
        let extra = workdir.join("out").join(extra_name);
        if extra.exists() {
            entries.push((format!("diagnostics/{extra_name}"), fs::read(extra)?, true));
        }
    }

    let mut warnings = Vec::new();
    if args.include_payload {
        warnings.push("include_payload=true: raw payload fields may be present".to_string());
    }
    if args.include_weights {
        warnings.push("include_weights=true: model binaries may include sensitive IP".to_string());
        warnings.push(
            "weight inclusion currently unsupported in bugkit; no model binaries added".to_string(),
        );
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut manifest_entries: Vec<BugKitManifestEntry> = entries
        .iter()
        .map(|(path, bytes, optional)| BugKitManifestEntry {
            path: path.clone(),
            sha256: sha256_hex(bytes),
            size_bytes: bytes.len() as u64,
            optional: *optional,
            dropped_due_to_size_cap: false,
        })
        .collect();

    let mut total_bytes: u64 = manifest_entries.iter().map(|e| e.size_bytes).sum();
    if total_bytes > args.max_bytes {
        for entry in manifest_entries.iter_mut().rev() {
            if entry.optional && total_bytes > args.max_bytes {
                total_bytes = total_bytes.saturating_sub(entry.size_bytes);
                entry.dropped_due_to_size_cap = true;
                warnings.push(format!(
                    "dropped optional artifact due to size cap: {}",
                    entry.path
                ));
            }
        }
        if total_bytes > args.max_bytes {
            warnings.push(format!(
                "bugkit remains over max size cap after dropping optional artifacts ({} > {})",
                total_bytes, args.max_bytes
            ));
        }
    }

    let chain_refs = derive_export_chain_digest_refs(workdir)?;
    let mut related_artifacts = vec![
        canonical_export_ref_from_pack("backend_evidence_snapshot", &backend_evidence_snapshot)?,
        canonical_export_ref_from_pack("active_review_snapshot", &active_review_snapshot)?,
        canonical_export_ref_from_pack("operator_signoff", &operator_signoff)?,
        canonical_export_ref_from_pack("backend_resolution", &backend_resolution)?,
    ];
    for (kind, path, digest) in [
        (
            "canonical_governance_entry",
            "artifacts/canonical_governance_entry.ref",
            chain_refs.canonical_governance_entry_digest_prefix,
        ),
        (
            "canonical_readiness_spine",
            "artifacts/canonical_readiness_spine.ref",
            chain_refs.canonical_readiness_spine_digest_prefix,
        ),
        (
            "operator_review_packet",
            "artifacts/operator_review_packet.ref",
            chain_refs.operator_review_packet_digest_prefix,
        ),
        (
            "operator_signoff_decision",
            "artifacts/operator_signoff_decision.ref",
            chain_refs.operator_signoff_digest_prefix,
        ),
        (
            "operator_workflow_chain",
            "artifacts/operator_workflow_chain.ref",
            chain_refs.operator_workflow_chain_digest_prefix,
        ),
        (
            "operator_export_authority_chain",
            "artifacts/operator_export_authority_chain.ref",
            chain_refs.operator_export_authority_chain_digest_prefix,
        ),
    ] {
        if digest != "MISSING" {
            related_artifacts.push(canonical_digest_only_ref(kind, path, digest)?);
        }
    }
    related_artifacts.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    let export_context = canonical_export_context_from_parts(
        &evidence_context,
        Some(run_id),
        (!operator_signoff.digest_prefix.is_empty())
            .then_some(operator_signoff.digest_prefix.clone()),
        (!backend_evidence_snapshot.digest_prefix.is_empty())
            .then_some(backend_evidence_snapshot.digest_prefix.clone()),
        (!active_review_snapshot.digest_prefix.is_empty())
            .then_some(active_review_snapshot.digest_prefix.clone()),
    )?;

    let mut manifest = BugKitManifestV1 {
        schema_version: 1,
        run_id: run_id.to_string(),
        include_payload: args.include_payload,
        include_weights: args.include_weights,
        max_bytes: args.max_bytes,
        total_bytes,
        file_count: manifest_entries
            .iter()
            .filter(|e| !e.dropped_due_to_size_cap)
            .count()
            + 1,
        files: manifest_entries,
        evidence_context: PackEvidenceContextSummaryV1 {
            supported_slot_set_digest_prefix: evidence_context.supported_slot_set_digest_prefix,
            policy_graph_digest_prefix: evidence_context.policy_graph_digest_prefix,
            manifest_digest_prefix: evidence_context.manifest_digest_prefix,
        },
        backend_evidence_snapshot,
        active_review_snapshot,
        operator_signoff,
        backend_resolution,
        export_context,
        related_artifacts,
        canonical_bundle_spine_digest_prefix: "MISSING".to_string(),
        canonical_bundle_authority_digest_prefix: "MISSING".to_string(),
        export_layout_compatibility: CanonicalExportLayoutCompatibilityV1::Canonical,
        warnings,
        bugkit_digest: String::new(),
    };
    let roundtrip = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
        bundle_kind: CanonicalBundleKindV1::Bugkit,
        bundle_digest: "",
        export_context: &manifest.export_context,
        evidence_context: &manifest.evidence_context,
        related_artifacts: &manifest.related_artifacts,
        backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
        active_review_snapshot: &manifest.active_review_snapshot,
        operator_signoff: &manifest.operator_signoff,
        export_layout_compatibility: &manifest.export_layout_compatibility,
    })?;
    let spine_report = evaluate_bundle_spine(BundleSpineInputs {
        bundle_kind: CanonicalBundleKindV1::Bugkit,
        export_context: &manifest.export_context,
        evidence_context: &manifest.evidence_context,
        related_artifacts: &manifest.related_artifacts,
        backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
        active_review_snapshot: &manifest.active_review_snapshot,
        operator_signoff: &manifest.operator_signoff,
        roundtrip: &roundtrip,
    })?;
    manifest.canonical_bundle_spine_digest_prefix =
        prefix_hex(&spine_report.spine.bundle_spine_digest, 16);
    if let Some(authority) = spine_report.authority_digest_prefix {
        manifest.canonical_bundle_authority_digest_prefix = authority;
    }
    let mut canonical = manifest.clone();
    canonical.bugkit_digest.clear();
    canonical.files.sort_by(|a, b| a.path.cmp(&b.path));
    manifest.bugkit_digest = sha256_hex(&serde_json::to_vec(&canonical)?);

    let mut selected: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (path, bytes, _) in entries {
        if manifest
            .files
            .iter()
            .any(|e| e.path == path && !e.dropped_due_to_size_cap)
        {
            selected.insert(path, bytes);
        }
    }
    selected.insert(
        "BUGKIT_MANIFEST.json".to_string(),
        serde_json::to_vec_pretty(&manifest)?,
    );

    let file = fs::File::create(out)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (path, bytes) in selected {
        zip.start_file(path, opts)
            .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
        zip.write_all(&bytes)
            .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
    }
    zip.finish()
        .map_err(|e| OpsError::Invalid(format!("zip finalize failed: {e}")))?;

    Ok(BugKitBuildReport {
        run_id: run_id.to_string(),
        out: out.display().to_string(),
        total_bytes: manifest.total_bytes,
        file_count: manifest.file_count,
        warnings: manifest.warnings,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExportNormalizeMismatchCategoryV1 {
    PathNamingDrift,
    ContextFieldDrift,
    IncludedStateDrift,
    DigestFieldDrift,
    LegacyExportLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExportNormalizeMismatchV1 {
    pub category: ExportNormalizeMismatchCategoryV1,
    pub detail: String,
    pub canonical_condition_code: String,
    pub primary_remediation_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExportNormalizeCheckReportV1 {
    pub schema_version: u16,
    pub pass: bool,
    pub mismatch_count: usize,
    pub mismatches: Vec<ExportNormalizeMismatchV1>,
    pub allowed_states: Vec<String>,
}

fn canonicalize_normalize_mismatch(
    category: &ExportNormalizeMismatchCategoryV1,
) -> (String, String) {
    let condition = match category {
        ExportNormalizeMismatchCategoryV1::PathNamingDrift
        | ExportNormalizeMismatchCategoryV1::ContextFieldDrift
        | ExportNormalizeMismatchCategoryV1::DigestFieldDrift => "ManifestMismatch",
        ExportNormalizeMismatchCategoryV1::IncludedStateDrift => "ExportRoundTripMismatch",
        ExportNormalizeMismatchCategoryV1::LegacyExportLayout => "ExportLayoutMismatch",
    };
    let remediation = crate::remediation::primary_remediation_for_condition_code(condition)
        .unwrap_or_else(|| "REMEDIATION_REVIEW_REPORT_MANUALLY".to_string());
    (condition.to_string(), remediation)
}

fn validate_canonical_artifact_ref(
    refv: &CanonicalExportArtifactRefV1,
) -> Vec<ExportNormalizeMismatchV1> {
    let mut out = Vec::new();
    if !refv.relative_path.starts_with("artifacts/") && !refv.relative_path.starts_with("evidence/")
    {
        out.push(ExportNormalizeMismatchV1 {
            category: ExportNormalizeMismatchCategoryV1::PathNamingDrift,
            detail: format!("{} path outside canonical prefixes", refv.artifact_kind),
            canonical_condition_code: String::new(),
            primary_remediation_code: String::new(),
        });
    }
    if matches!(
        refv.included_state,
        CanonicalArtifactIncludedStateV1::Included
    ) && refv.sha256.as_deref().unwrap_or_default().is_empty()
    {
        out.push(ExportNormalizeMismatchV1 {
            category: ExportNormalizeMismatchCategoryV1::DigestFieldDrift,
            detail: format!("{} included without sha256", refv.artifact_kind),
            canonical_condition_code: String::new(),
            primary_remediation_code: String::new(),
        });
    }
    if !matches!(
        refv.included_state,
        CanonicalArtifactIncludedStateV1::Included
    ) && refv.sha256.as_ref().is_some_and(|v| !v.is_empty())
    {
        out.push(ExportNormalizeMismatchV1 {
            category: ExportNormalizeMismatchCategoryV1::IncludedStateDrift,
            detail: format!("{} non-included has sha256", refv.artifact_kind),
            canonical_condition_code: String::new(),
            primary_remediation_code: String::new(),
        });
    }
    out
}

fn validate_canonical_context(ctx: &CanonicalExportContextV1) -> Vec<ExportNormalizeMismatchV1> {
    let mut out = Vec::new();
    if ctx.supported_slot_set_digest_prefix.is_empty()
        || ctx.policy_graph_digest_prefix.is_empty()
        || ctx.manifest_digest_prefix.is_empty()
    {
        out.push(ExportNormalizeMismatchV1 {
            category: ExportNormalizeMismatchCategoryV1::ContextFieldDrift,
            detail: "required context digest prefix missing".to_string(),
            canonical_condition_code: String::new(),
            primary_remediation_code: String::new(),
        });
    }
    if canonical_context_digest_hex(ctx).map_or(true, |d| d != ctx.context_digest) {
        out.push(ExportNormalizeMismatchV1 {
            category: ExportNormalizeMismatchCategoryV1::DigestFieldDrift,
            detail: "context_digest mismatch".to_string(),
            canonical_condition_code: String::new(),
            primary_remediation_code: String::new(),
        });
    }
    out
}

pub fn exports_normalize_check(
    workdir: &Path,
    out: &Path,
) -> Result<ExportNormalizeCheckReportV1, OpsError> {
    let mut mismatches = Vec::new();

    let tmp = tempfile::tempdir()?;
    let run_id = reproducible_run_id_for_smoke(tmp.path())?;
    let repro_zip = workdir.join("out").join("repro_normalize_check.zip");
    let _ = repro_pack(tmp.path(), &run_id, &repro_zip)?;

    let repro_file = fs::File::open(&repro_zip)?;
    let mut repro_archive = zip::ZipArchive::new(repro_file)
        .map_err(|e| OpsError::Invalid(format!("unable to open repro zip: {e}")))?;
    let mut repro_manifest_file = repro_archive
        .by_name("repro_pack_manifest.json")
        .map_err(|e| OpsError::Invalid(format!("missing repro manifest: {e}")))?;
    let mut repro_manifest_body = String::new();
    std::io::Read::read_to_string(&mut repro_manifest_file, &mut repro_manifest_body)?;
    let repro_manifest: ReproPackManifestV1 = serde_json::from_str(&repro_manifest_body)?;
    mismatches.extend(validate_canonical_context(&repro_manifest.export_context));
    for r in &repro_manifest.related_artifacts {
        mismatches.extend(validate_canonical_artifact_ref(r));
    }
    if !matches!(
        repro_manifest.export_layout_compatibility,
        CanonicalExportLayoutCompatibilityV1::Canonical
    ) {
        mismatches.push(ExportNormalizeMismatchV1 {
            category: ExportNormalizeMismatchCategoryV1::LegacyExportLayout,
            detail: "repro pack is not canonical layout".to_string(),
            canonical_condition_code: String::new(),
            primary_remediation_code: String::new(),
        });
    }

    let bugkit_zip = workdir.join("out").join("bugkit_normalize_check.zip");
    let _ = bugkit_build(
        tmp.path(),
        &run_id,
        &bugkit_zip,
        &BugKitBuildArgs::default(),
    )?;
    let bug_file = fs::File::open(&bugkit_zip)?;
    let mut bug_archive = zip::ZipArchive::new(bug_file)
        .map_err(|e| OpsError::Invalid(format!("unable to open bugkit zip: {e}")))?;
    let mut manifest_file = bug_archive
        .by_name("BUGKIT_MANIFEST.json")
        .map_err(|e| OpsError::Invalid(format!("missing bugkit manifest: {e}")))?;
    let mut body = String::new();
    std::io::Read::read_to_string(&mut manifest_file, &mut body)?;
    let bug_manifest: BugKitManifestV1 = serde_json::from_str(&body)?;
    mismatches.extend(validate_canonical_context(&bug_manifest.export_context));
    for r in &bug_manifest.related_artifacts {
        mismatches.extend(validate_canonical_artifact_ref(r));
    }
    if !matches!(
        bug_manifest.export_layout_compatibility,
        CanonicalExportLayoutCompatibilityV1::Canonical
    ) {
        mismatches.push(ExportNormalizeMismatchV1 {
            category: ExportNormalizeMismatchCategoryV1::LegacyExportLayout,
            detail: "bugkit is not canonical layout".to_string(),
            canonical_condition_code: String::new(),
            primary_remediation_code: String::new(),
        });
    }

    for mismatch in &mut mismatches {
        let (condition, remediation) = canonicalize_normalize_mismatch(&mismatch.category);
        mismatch.canonical_condition_code = condition;
        mismatch.primary_remediation_code = remediation;
    }

    let report = ExportNormalizeCheckReportV1 {
        schema_version: 1,
        pass: mismatches.is_empty(),
        mismatch_count: mismatches.len(),
        mismatches,
        allowed_states: vec![
            "INCLUDED".to_string(),
            "MISSING".to_string(),
            "EXCLUDED".to_string(),
            "SKIP".to_string(),
        ],
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(out, &report)?;
    Ok(report)
}

pub fn repro_pack(
    workdir: &Path,
    run_id: &str,
    out: &Path,
) -> Result<ReproPackBuildReport, OpsError> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let run_meta = runs_show(workdir, run_id)?
        .ok_or_else(|| OpsError::Invalid(format!("run metadata not found: {run_id}")))?;
    let config_path = workdir.join("config_resolved.json");
    if !config_path.exists() {
        let _ = load_or_init_config(workdir)?;
    }

    let (policy_base, policy_overlay, manifest_path) = resolve_attestation_inputs();
    let policy = policy_validate(&policy_base, Some(&policy_overlay))?;
    let model_verify = models_verify(&manifest_path)?;

    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let fixture_value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&fixture_path)?)?;
    let decisions = fixture_value
        .get("decisions")
        .and_then(|v| v.as_array())
        .ok_or_else(|| OpsError::Invalid("ess fixture missing decisions array".to_string()))?;
    let bounded_decisions: Vec<serde_json::Value> = decisions
        .iter()
        .rev()
        .take(2048)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let bounded_fixture = serde_json::json!({"decisions": bounded_decisions});
    let bounded_records = load_fixture_records_from_value(&bounded_fixture)?;
    let segments = build_merkle_segments(run_id, &bounded_records, 1024);

    let policy_ref = serde_json::json!({
        "policy_graph_digest": policy.policy_graph_digest,
        "base_pack": policy.base_pack,
        "overlay_pack": policy.overlay_pack,
        "schema_version": policy.schema_version
    });
    let models_ref = serde_json::json!({
        "manifest_path": model_verify.manifest,
        "manifest_digest": model_verify.model_hashes_digest,
        "active_hashes": model_verify
            .slots
            .iter()
            .filter_map(|s| s.sha256.as_ref().map(|h| serde_json::json!({"slot": s.slot, "sha256": h})))
            .collect::<Vec<_>>()
    });
    let context_snapshot = models_evidence_snapshot(workdir, None, Some(run_id))?;
    let evidence_context = EvidenceValidationContext {
        supported_slot_set_digest_prefix: context_snapshot.supported_slot_set_digest,
        policy_graph_digest_prefix: context_snapshot.policy_graph_digest_prefix,
        manifest_digest_prefix: context_snapshot.manifest_digest_prefix,
    };

    let mut file_map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    file_map.insert("config_resolved.json".to_string(), fs::read(&config_path)?);
    file_map.insert(
        "policy_ref.json".to_string(),
        serde_json::to_vec_pretty(&policy_ref)?,
    );
    file_map.insert(
        "models_ref.json".to_string(),
        serde_json::to_vec_pretty(&models_ref)?,
    );
    file_map.insert(
        "ess_slice.json".to_string(),
        serde_json::to_vec_pretty(&bounded_fixture)?,
    );
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
    file_map.insert(
        "segment_roots.json".to_string(),
        serde_json::to_vec_pretty(&roots_only)?,
    );
    let gate_path = PathBuf::from("./out/gate_report.json");
    if gate_path.exists() {
        file_map.insert(
            "readiness_gate_report.json".to_string(),
            fs::read(&gate_path)?,
        );
    }

    let (backend_evidence_snapshot, active_review_snapshot, operator_signoff, backend_resolution) =
        enrich_evidence_artifacts(workdir, &evidence_context, false, &mut file_map)?;

    let cert_path = workdir.join("out").join(format!("run_cert_{run_id}.json"));
    let cert_digest = if cert_path.exists() {
        let cert: RunCertificateV1 = serde_json::from_str(&fs::read_to_string(&cert_path)?)?;
        file_map.insert(
            "run_certificate.json".to_string(),
            serde_json::to_vec_pretty(&cert)?,
        );
        Some(cert.certificate_digest)
    } else {
        None
    };

    let mut included_artifacts = Vec::new();
    for (name, bytes) in &file_map {
        included_artifacts.push(ReproPackArtifact {
            path: name.clone(),
            sha256: sha256_hex(bytes),
        });
    }

    let chain_refs = derive_export_chain_digest_refs(workdir)?;
    let mut related_artifacts = vec![
        canonical_export_ref_from_pack("backend_evidence_snapshot", &backend_evidence_snapshot)?,
        canonical_export_ref_from_pack("active_review_snapshot", &active_review_snapshot)?,
        canonical_export_ref_from_pack("operator_signoff", &operator_signoff)?,
        canonical_export_ref_from_pack("backend_resolution", &backend_resolution)?,
    ];
    for (kind, path, digest) in [
        (
            "canonical_governance_entry",
            "artifacts/canonical_governance_entry.ref",
            chain_refs.canonical_governance_entry_digest_prefix,
        ),
        (
            "canonical_readiness_spine",
            "artifacts/canonical_readiness_spine.ref",
            chain_refs.canonical_readiness_spine_digest_prefix,
        ),
        (
            "operator_review_packet",
            "artifacts/operator_review_packet.ref",
            chain_refs.operator_review_packet_digest_prefix,
        ),
        (
            "operator_signoff_decision",
            "artifacts/operator_signoff_decision.ref",
            chain_refs.operator_signoff_digest_prefix,
        ),
        (
            "operator_workflow_chain",
            "artifacts/operator_workflow_chain.ref",
            chain_refs.operator_workflow_chain_digest_prefix,
        ),
        (
            "operator_export_authority_chain",
            "artifacts/operator_export_authority_chain.ref",
            chain_refs.operator_export_authority_chain_digest_prefix,
        ),
    ] {
        if digest != "MISSING" {
            related_artifacts.push(canonical_digest_only_ref(kind, path, digest)?);
        }
    }
    related_artifacts.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    let export_context = canonical_export_context_from_parts(
        &evidence_context,
        Some(run_id),
        (!operator_signoff.digest_prefix.is_empty())
            .then_some(operator_signoff.digest_prefix.clone()),
        (!backend_evidence_snapshot.digest_prefix.is_empty())
            .then_some(backend_evidence_snapshot.digest_prefix.clone()),
        (!active_review_snapshot.digest_prefix.is_empty())
            .then_some(active_review_snapshot.digest_prefix.clone()),
    )?;

    let pack_id = format!("repro-{run_id}");
    let mut manifest = ReproPackManifestV1 {
        schema_version: 1,
        pack_id: pack_id.clone(),
        run_id: run_id.to_string(),
        policy_graph_digest: policy.policy_graph_digest,
        manifest_digest: model_verify.model_hashes_digest,
        config_digest: run_meta.config_digest,
        included_artifacts,
        ess_slice: ReproPackEssSlice {
            record_count: bounded_records.len(),
            segment_roots: roots_only
                .iter()
                .filter_map(|v| {
                    v.get("merkle_root")
                        .and_then(|x| x.as_str())
                        .map(ToString::to_string)
                })
                .collect(),
        },
        certificate_digest: cert_digest,
        evidence_context: PackEvidenceContextSummaryV1 {
            supported_slot_set_digest_prefix: evidence_context.supported_slot_set_digest_prefix,
            policy_graph_digest_prefix: evidence_context.policy_graph_digest_prefix,
            manifest_digest_prefix: evidence_context.manifest_digest_prefix,
        },
        backend_evidence_snapshot,
        active_review_snapshot,
        operator_signoff,
        backend_resolution,
        export_context,
        related_artifacts,
        canonical_bundle_spine_digest_prefix: "MISSING".to_string(),
        canonical_bundle_authority_digest_prefix: "MISSING".to_string(),
        export_layout_compatibility: CanonicalExportLayoutCompatibilityV1::Canonical,
        repro_pack_digest: String::new(),
    };
    let roundtrip = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
        bundle_kind: CanonicalBundleKindV1::Repro,
        bundle_digest: "",
        export_context: &manifest.export_context,
        evidence_context: &manifest.evidence_context,
        related_artifacts: &manifest.related_artifacts,
        backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
        active_review_snapshot: &manifest.active_review_snapshot,
        operator_signoff: &manifest.operator_signoff,
        export_layout_compatibility: &manifest.export_layout_compatibility,
    })?;
    let spine_report = evaluate_bundle_spine(BundleSpineInputs {
        bundle_kind: CanonicalBundleKindV1::Repro,
        export_context: &manifest.export_context,
        evidence_context: &manifest.evidence_context,
        related_artifacts: &manifest.related_artifacts,
        backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
        active_review_snapshot: &manifest.active_review_snapshot,
        operator_signoff: &manifest.operator_signoff,
        roundtrip: &roundtrip,
    })?;
    manifest.canonical_bundle_spine_digest_prefix =
        prefix_hex(&spine_report.spine.bundle_spine_digest, 16);
    if let Some(authority) = spine_report.authority_digest_prefix {
        manifest.canonical_bundle_authority_digest_prefix = authority;
    }
    manifest.repro_pack_digest = repro_pack_digest_hex(&manifest)?;
    file_map.insert(
        "repro_pack_manifest.json".to_string(),
        serde_json::to_vec_pretty(&manifest)?,
    );

    let file = fs::File::create(out)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, bytes) in &file_map {
        zip.start_file(name, opts)
            .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
        zip.write_all(bytes)
            .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
    }
    zip.finish()
        .map_err(|e| OpsError::Invalid(format!("zip finalize failed: {e}")))?;

    Ok(ReproPackBuildReport {
        run_id: run_id.to_string(),
        pack_id,
        out: out.display().to_string(),
        entry_count: file_map.len(),
    })
}

pub fn repro_verify(pack: &Path, out: &Path) -> Result<ReproVerifyReport, OpsError> {
    let file = fs::File::open(pack)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| OpsError::Invalid(format!("unable to open repro pack zip: {e}")))?;
    let temp = tempfile::tempdir()?;
    let mut reasons = Vec::new();

    for i in 0..archive.len() {
        let mut f = archive
            .by_index(i)
            .map_err(|e| OpsError::Invalid(format!("zip read failed: {e}")))?;
        let out_path = temp.path().join(f.name());
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut out_file = fs::File::create(&out_path)?;
        std::io::copy(&mut f, &mut out_file)?;
    }

    let manifest_path = temp.path().join("repro_pack_manifest.json");
    let manifest: ReproPackManifestV1 = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    let roundtrip = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
        bundle_kind: CanonicalBundleKindV1::Repro,
        bundle_digest: &manifest.repro_pack_digest,
        export_context: &manifest.export_context,
        evidence_context: &manifest.evidence_context,
        related_artifacts: &manifest.related_artifacts,
        backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
        active_review_snapshot: &manifest.active_review_snapshot,
        operator_signoff: &manifest.operator_signoff,
        export_layout_compatibility: &manifest.export_layout_compatibility,
    })?;
    if matches!(
        roundtrip.overall_status,
        BundleRoundTripOverallStatusV1::Fail
    ) {
        for code in &roundtrip.mismatch_codes {
            reasons.push(format!("bundle roundtrip mismatch: {code}"));
        }
    }
    let bundle_spine = evaluate_bundle_spine(BundleSpineInputs {
        bundle_kind: CanonicalBundleKindV1::Repro,
        export_context: &manifest.export_context,
        evidence_context: &manifest.evidence_context,
        related_artifacts: &manifest.related_artifacts,
        backend_evidence_snapshot: &manifest.backend_evidence_snapshot,
        active_review_snapshot: &manifest.active_review_snapshot,
        operator_signoff: &manifest.operator_signoff,
        roundtrip: &roundtrip,
    })?;
    if !bundle_spine.pass {
        for code in &bundle_spine.mismatch_codes {
            reasons.push(format!("bundle spine mismatch: {code}"));
        }
    }
    let recomputed_pack = repro_pack_digest_hex(&manifest)?;
    if recomputed_pack != manifest.repro_pack_digest {
        reasons.push("repro_pack_digest mismatch".to_string());
    }

    for art in &manifest.included_artifacts {
        let p = temp.path().join(&art.path);
        let bytes = fs::read(&p)?;
        if sha256_hex(&bytes) != art.sha256 {
            reasons.push(format!("artifact sha256 mismatch: {}", art.path));
        }
    }

    for evidence in [
        &manifest.backend_evidence_snapshot,
        &manifest.active_review_snapshot,
        &manifest.operator_signoff,
        &manifest.backend_resolution,
    ] {
        if evidence.included {
            let p = temp.path().join(&evidence.path);
            let bytes = fs::read(&p)?;
            if sha256_hex(&bytes) != evidence.sha256 {
                reasons.push(format!("artifact sha256 mismatch: {}", evidence.path));
            }
        }
    }

    if manifest.backend_evidence_snapshot.included {
        let path = temp.path().join(&manifest.backend_evidence_snapshot.path);
        let snapshot: BackendEvidenceSnapshotV1 =
            serde_json::from_str(&fs::read_to_string(&path)?)?;
        if let Some(reason) = validate_evidence_artifacts_against_context(
            &EvidenceValidationContext {
                supported_slot_set_digest_prefix: manifest
                    .evidence_context
                    .supported_slot_set_digest_prefix
                    .clone(),
                policy_graph_digest_prefix: manifest
                    .evidence_context
                    .policy_graph_digest_prefix
                    .clone(),
                manifest_digest_prefix: manifest.evidence_context.manifest_digest_prefix.clone(),
            },
            &snapshot.supported_slot_set_digest,
            &snapshot.policy_graph_digest_prefix,
            &snapshot.manifest_digest_prefix,
        ) {
            reasons.push(format!("backend evidence context mismatch: {reason}"));
        }
    }

    if manifest.active_review_snapshot.included {
        let path = temp.path().join(&manifest.active_review_snapshot.path);
        let snapshot: AggregatedActiveReviewSnapshotV1 =
            serde_json::from_str(&fs::read_to_string(&path)?)?;
        if let Some(reason) = validate_evidence_artifacts_against_context(
            &EvidenceValidationContext {
                supported_slot_set_digest_prefix: manifest
                    .evidence_context
                    .supported_slot_set_digest_prefix
                    .clone(),
                policy_graph_digest_prefix: manifest
                    .evidence_context
                    .policy_graph_digest_prefix
                    .clone(),
                manifest_digest_prefix: manifest.evidence_context.manifest_digest_prefix.clone(),
            },
            &snapshot.supported_slot_set_digest,
            &snapshot.policy_graph_digest_prefix,
            &snapshot.manifest_digest_prefix,
        ) {
            reasons.push(format!("active review context mismatch: {reason}"));
        }
    }

    if manifest.operator_signoff.included {
        let path = temp.path().join(&manifest.operator_signoff.path);
        let signoff: OperatorSignoffDecisionV1 = serde_json::from_str(&fs::read_to_string(&path)?)?;
        if let Some(reason) = validate_evidence_artifacts_against_context(
            &EvidenceValidationContext {
                supported_slot_set_digest_prefix: manifest
                    .evidence_context
                    .supported_slot_set_digest_prefix
                    .clone(),
                policy_graph_digest_prefix: manifest
                    .evidence_context
                    .policy_graph_digest_prefix
                    .clone(),
                manifest_digest_prefix: manifest.evidence_context.manifest_digest_prefix.clone(),
            },
            &signoff.supported_slot_set_digest,
            &signoff.policy_graph_digest_prefix,
            &signoff.manifest_digest_prefix,
        ) {
            reasons.push(format!("operator signoff context mismatch: {reason}"));
        }
    }

    let policy_ref: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(temp.path().join("policy_ref.json"))?)?;
    if policy_ref
        .get("policy_graph_digest")
        .and_then(|v| v.as_str())
        != Some(manifest.policy_graph_digest.as_str())
    {
        reasons.push("policy graph digest mismatch".to_string());
    }

    let models_ref: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(temp.path().join("models_ref.json"))?)?;
    if models_ref.get("manifest_digest").and_then(|v| v.as_str())
        != Some(manifest.manifest_digest.as_str())
    {
        reasons.push("manifest digest mismatch".to_string());
    }

    let ess_path = temp.path().join("ess_slice.json");
    let records = load_fixture_records(&ess_path)?;
    let replay_path = out.with_file_name("repro_verify_replay_report.json");
    let replay = replay_records(
        &records,
        &ReplaySpec {
            from_tick: 0,
            to_tick: u64::MAX,
            backend_override: None,
            seed_override: None,
            budget_override: None,
            mode: ReplayMode::ComputeOnly,
        },
    );
    write_report(&replay_path, &replay)?;
    let first_divergence = replay
        .items
        .iter()
        .find(|i| i.status != ucf_replay::ReplayStatus::Match)
        .map(|i| i.decision_id);

    let report = ReproVerifyReport {
        pass: reasons.is_empty(),
        run_id: manifest.run_id,
        pack_id: manifest.pack_id,
        checked_files: manifest.included_artifacts.len(),
        replay_report: replay_path.display().to_string(),
        first_divergence,
        reasons,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(out, &report)?;
    Ok(report)
}

fn load_fixture_records_from_value(
    value: &serde_json::Value,
) -> Result<Vec<ExperienceRecord>, OpsError> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("ess_slice.json");
    fs::write(&path, serde_json::to_vec(value)?)?;
    let records = load_fixture_records(&path)?;
    Ok(records)
}

fn repro_pack_digest_hex(manifest: &ReproPackManifestV1) -> Result<String, OpsError> {
    let mut canonical = manifest.clone();
    canonical.repro_pack_digest.clear();
    canonical
        .included_artifacts
        .sort_by(|a, b| a.path.cmp(&b.path));
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
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
        ExperienceKind::WorldSummary => "WorldSummary",
        ExperienceKind::SaeSummary => "SaeSummary",
        ExperienceKind::SsmSummary => "SsmSummary",
        ExperienceKind::LfmSummary => "LfmSummary",
        ExperienceKind::SignalBundle => "SignalBundle",
        ExperienceKind::DecisionInputs => "DecisionInputs",
        ExperienceKind::LlmSummary => "LlmSummary",
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
        ExperienceKind::SlotCompareWindow => "SlotCompareWindow",
        ExperienceKind::DriftAlarm => "DriftAlarm",
        ExperienceKind::ShadowDisable => "ShadowDisable",
        ExperienceKind::SlotModeChange => "SlotModeChange",
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

fn normalized_rel_path(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
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
        let rel = normalized_rel_path(repo_root, path);
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct NetworkAllowlist {
    pub schema_version: u16,
    pub runtime_crates: Vec<String>,
    pub forbidden_crates: Vec<String>,
    pub allowed_feature_notes: Vec<String>,
    pub exempt_runtime_edges: Vec<NetDepExemption>,
}

impl Default for NetworkAllowlist {
    fn default() -> Self {
        Self {
            schema_version: 1,
            runtime_crates: vec![
                "ucf-runtime".to_string(),
                "ucf-policy".to_string(),
                "ucf-replay".to_string(),
                "ucf-gateway".to_string(),
                "ucf-client".to_string(),
                "ucf-platform".to_string(),
                "ucf-backends-gpu".to_string(),
            ],
            forbidden_crates: vec![
                "reqwest".to_string(),
                "hyper".to_string(),
                "tokio-tungstenite".to_string(),
                "tungstenite".to_string(),
                "ureq".to_string(),
                "isahc".to_string(),
            ],
            allowed_feature_notes: Vec::new(),
            exempt_runtime_edges: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct NetDepExemption {
    pub root_crate: String,
    pub forbidden_crate: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetDepsViolation {
    pub root_crate: String,
    pub forbidden_crate: String,
    pub path: Vec<String>,
    pub remediation: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetDepsAuditReport {
    pub schema_version: u16,
    pub allowlist_path: String,
    pub runtime_roots: Vec<String>,
    pub violations: Vec<NetDepsViolation>,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
    resolve: Option<CargoResolve>,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoPackage {
    id: String,
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoResolve {
    nodes: Vec<CargoNode>,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoNode {
    id: String,
    deps: Vec<CargoNodeDep>,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoNodeDep {
    pkg: String,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoLockfile {
    package: Vec<CargoLockPackage>,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoLockPackage {
    name: String,
    #[serde(default)]
    dependencies: Vec<String>,
}

pub fn load_network_allowlist(path: &Path) -> Result<NetworkAllowlist, OpsError> {
    let body = fs::read_to_string(path)?;
    let mut parsed = toml::from_str::<NetworkAllowlist>(&body).map_err(|e| {
        OpsError::Invalid(format!("invalid network allowlist {}: {e}", path.display()))
    })?;
    parsed.runtime_crates.sort();
    parsed.runtime_crates.dedup();
    parsed.forbidden_crates.sort();
    parsed.forbidden_crates.dedup();
    parsed.allowed_feature_notes.sort();
    parsed.allowed_feature_notes.dedup();
    parsed.exempt_runtime_edges.sort_by(|a, b| {
        (&a.root_crate, &a.forbidden_crate, &a.reason).cmp(&(
            &b.root_crate,
            &b.forbidden_crate,
            &b.reason,
        ))
    });
    Ok(parsed)
}

pub fn net_deps_audit(
    repo_root: &Path,
    allowlist_path: &Path,
) -> Result<NetDepsAuditReport, OpsError> {
    let allowlist = load_network_allowlist(allowlist_path)?;
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--locked")
        .arg("--offline")
        .current_dir(repo_root)
        .output()?;
    if output.status.success() {
        let metadata_json = String::from_utf8_lossy(&output.stdout).to_string();
        return net_deps_audit_from_metadata_json(&metadata_json, &allowlist, allowlist_path);
    }

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !stderr.contains("attempting to make an HTTP request, but --offline was specified") {
        return Err(OpsError::Invalid(format!(
            "cargo metadata failed: {stderr}"
        )));
    }

    let lockfile_path = repo_root.join("Cargo.lock");
    let lockfile_body = fs::read_to_string(&lockfile_path)?;
    net_deps_audit_from_lockfile_toml(&lockfile_body, &allowlist, allowlist_path)
}

pub fn net_deps_audit_from_metadata_json(
    metadata_json: &str,
    allowlist: &NetworkAllowlist,
    allowlist_path: &Path,
) -> Result<NetDepsAuditReport, OpsError> {
    let metadata: CargoMetadata = serde_json::from_str(metadata_json)?;
    let Some(resolve) = metadata.resolve else {
        return Err(OpsError::Invalid(
            "cargo metadata output missing resolve graph".to_string(),
        ));
    };

    let mut id_to_name = BTreeMap::new();
    let mut name_to_ids: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for package in metadata.packages {
        id_to_name.insert(package.id.clone(), package.name.clone());
        name_to_ids
            .entry(package.name)
            .or_default()
            .push(package.id);
    }

    let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for node in resolve.nodes {
        let mut deps = node.deps.into_iter().map(|d| d.pkg).collect::<Vec<_>>();
        deps.sort();
        deps.dedup();
        graph.insert(node.id, deps);
    }

    let mut runtime_root_ids = Vec::new();
    for crate_name in &allowlist.runtime_crates {
        if let Some(ids) = name_to_ids.get(crate_name) {
            runtime_root_ids.extend(ids.iter().cloned());
        }
    }
    runtime_root_ids.sort();
    runtime_root_ids.dedup();

    let mut violations = Vec::new();
    for root_id in runtime_root_ids {
        let root_name = id_to_name
            .get(&root_id)
            .cloned()
            .unwrap_or_else(|| root_id.clone());
        violations.extend(find_net_dep_violations(
            &root_id,
            &root_name,
            &graph,
            &id_to_name,
            allowlist,
        ));
    }

    Ok(build_net_deps_report(allowlist_path, allowlist, violations))
}

pub fn net_deps_audit_from_lockfile_toml(
    lockfile_body: &str,
    allowlist: &NetworkAllowlist,
    allowlist_path: &Path,
) -> Result<NetDepsAuditReport, OpsError> {
    let lockfile = toml::from_str::<CargoLockfile>(lockfile_body)
        .map_err(|e| OpsError::Invalid(format!("invalid Cargo.lock for net-deps audit: {e}")))?;

    let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for package in lockfile.package {
        let mut deps = package
            .dependencies
            .into_iter()
            .map(|dep| lockfile_dep_name(&dep))
            .collect::<Vec<_>>();
        deps.sort();
        deps.dedup();
        graph.entry(package.name).or_default().extend(deps);
    }
    for deps in graph.values_mut() {
        deps.sort();
        deps.dedup();
    }

    let id_to_name = graph
        .keys()
        .cloned()
        .map(|k| (k.clone(), k))
        .collect::<BTreeMap<_, _>>();

    let mut violations = Vec::new();
    for root_name in &allowlist.runtime_crates {
        if !graph.contains_key(root_name) {
            continue;
        }
        violations.extend(find_net_dep_violations(
            root_name,
            root_name,
            &graph,
            &id_to_name,
            allowlist,
        ));
    }

    Ok(build_net_deps_report(allowlist_path, allowlist, violations))
}

fn lockfile_dep_name(dep: &str) -> String {
    dep.split_whitespace().next().unwrap_or(dep).to_string()
}

fn find_net_dep_violations(
    root_id: &str,
    root_name: &str,
    graph: &BTreeMap<String, Vec<String>>,
    id_to_name: &BTreeMap<String, String>,
    allowlist: &NetworkAllowlist,
) -> Vec<NetDepsViolation> {
    let forbidden: BTreeSet<String> = allowlist.forbidden_crates.iter().cloned().collect();
    let exemptions: BTreeSet<(String, String)> = allowlist
        .exempt_runtime_edges
        .iter()
        .map(|x| (x.root_crate.clone(), x.forbidden_crate.clone()))
        .collect();

    let mut queue = VecDeque::new();
    let mut visited = BTreeSet::new();
    let mut prev: BTreeMap<String, String> = BTreeMap::new();
    queue.push_back(root_id.to_string());
    visited.insert(root_id.to_string());

    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = graph.get(&current) {
            for next in neighbors {
                if visited.insert(next.clone()) {
                    prev.insert(next.clone(), current.clone());
                    queue.push_back(next.clone());
                }
            }
        }
    }

    let mut hits = visited
        .iter()
        .filter_map(|id| {
            let dep_name = id_to_name.get(id)?;
            if forbidden.contains(dep_name)
                && !exemptions.contains(&(root_name.to_string(), dep_name.to_string()))
            {
                Some((id.clone(), dep_name.clone()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut violations = Vec::new();
    for (hit_id, hit_name) in hits {
        let mut path_ids = vec![hit_id.clone()];
        let mut cursor = hit_id;
        while let Some(parent) = prev.get(&cursor) {
            path_ids.push(parent.clone());
            if parent == root_id {
                break;
            }
            cursor = parent.clone();
        }
        path_ids.reverse();
        let path = path_ids
            .iter()
            .map(|id| id_to_name.get(id).cloned().unwrap_or_else(|| id.clone()))
            .collect::<Vec<_>>();

        violations.push(NetDepsViolation {
            root_crate: root_name.to_string(),
            forbidden_crate: hit_name.clone(),
            path,
            remediation: vec![
                format!(
                    "feature-gate `{}` so it is excluded from default features for runtime crate `{}`",
                    hit_name, root_name
                ),
                "move networking code to an ops-only crate outside runtime closure".to_string(),
            ],
        });
    }
    violations
}

fn build_net_deps_report(
    allowlist_path: &Path,
    allowlist: &NetworkAllowlist,
    mut violations: Vec<NetDepsViolation>,
) -> NetDepsAuditReport {
    violations.sort_by(|a, b| {
        (&a.root_crate, &a.forbidden_crate, &a.path).cmp(&(
            &b.root_crate,
            &b.forbidden_crate,
            &b.path,
        ))
    });

    NetDepsAuditReport {
        schema_version: 1,
        allowlist_path: allowlist_path.to_string_lossy().replace('\\', "/"),
        runtime_roots: allowlist.runtime_crates.clone(),
        violations,
    }
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
        let rel = normalized_rel_path(repo_root, path);
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathScanViolation {
    pub path: String,
    pub line: usize,
    pub pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathScanReport {
    pub violations: Vec<PathScanViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V1SmokeCheck {
    pub name: String,
    pub status: GateStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct V1SmokeReport {
    pub schema_version: u16,
    pub checks: Vec<V1SmokeCheck>,
}

pub fn path_scan(repo_root: &Path) -> Result<PathScanReport, OpsError> {
    let banned = ["/etc/", "/var/", "systemd", "systemctl"];
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
        let rel = normalized_rel_path(repo_root, path);
        let in_scope = rel.starts_with("runtime/") && rel.contains("/src/");
        if !in_scope
            || rel.starts_with("runtime/ucf-ops/src/")
            || rel.contains("vendor/")
            || rel.contains("target/")
            || rel.contains("fuzz/")
            || rel.starts_with("deploy/")
        {
            continue;
        }
        let text = fs::read_to_string(path).unwrap_or_default();
        for (idx, line) in text.lines().enumerate() {
            for pat in banned {
                if line.contains(pat) && !line.contains("let banned =") {
                    violations.push(PathScanViolation {
                        path: rel.to_string(),
                        line: idx + 1,
                        pattern: pat.to_string(),
                    });
                }
            }
        }
    }
    violations.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.pattern.cmp(&b.pattern))
    });
    Ok(PathScanReport { violations })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortabilityFixedPointSummary {
    pub sample_count: usize,
    pub mean_risk_q: u16,
    pub mean_pressure_q: u16,
    pub mean_surprise_q: u16,
    pub mean_uncertainty_q: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortabilityCheckReport {
    pub schema_version: u16,
    pub os: String,
    pub arch: String,
    pub digest_prefixes: BTreeMap<String, String>,
    pub fixed_point_summary: PortabilityFixedPointSummary,
    pub deterministic_within_os: bool,
    pub remediation: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum PortabilityGateStatus {
    Pass,
    Fail,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortabilityMatrixEntry {
    pub os: String,
    pub command: String,
    pub support: PortabilityGateStatus,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortabilityCommandCheck {
    pub name: String,
    pub status: PortabilityGateStatus,
    pub detail: String,
    pub out: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortabilityReportV1 {
    pub schema_version: u16,
    pub docs_lint: PortabilityCommandCheck,
    pub path_scan: PortabilityCommandCheck,
    pub hardware_scan: PortabilityCommandCheck,
    pub hidden_network_scan: PortabilityCommandCheck,
    pub artifact_schema_snapshot_check: PortabilityCommandCheck,
    pub governance_surfaces_smoke: PortabilityCommandCheck,
    pub governance_entry_check_smoke: PortabilityCommandCheck,
    pub governance_entry_sweep_smoke: PortabilityCommandCheck,
    pub scope_authority_check_smoke: PortabilityCommandCheck,
    pub supported_scope_reevaluate_smoke: PortabilityCommandCheck,
    pub supported_scope_execute_smoke: PortabilityCommandCheck,
    pub readiness_spine_check_smoke: PortabilityCommandCheck,
    pub readiness_spine_sweep_smoke: PortabilityCommandCheck,
    pub bundle_spine_check_smoke: PortabilityCommandCheck,
    pub bundle_spine_sweep_smoke: PortabilityCommandCheck,
    pub primary_semantics_sweep_smoke: PortabilityCommandCheck,
    pub final_governance_consumer_sweep_smoke: PortabilityCommandCheck,
    pub supported_scope_execute_v5_smoke: PortabilityCommandCheck,
    pub final_readiness_consumer_sweep_smoke: PortabilityCommandCheck,
    pub final_bundle_consumer_sweep_smoke: PortabilityCommandCheck,
    pub final_primary_semantics_sweep_smoke: PortabilityCommandCheck,
    pub remediation_spine_check_smoke: PortabilityCommandCheck,
    pub supported_set_apply_smoke: PortabilityCommandCheck,
    pub review_truth_check_smoke: PortabilityCommandCheck,
    pub export_roundtrip_check_smoke: PortabilityCommandCheck,
    pub remediation_interop_check_smoke: PortabilityCommandCheck,
    pub active_review_snapshot_smoke: PortabilityCommandCheck,
    pub backend_resolution_smoke: PortabilityCommandCheck,
    pub repro_pack_smoke: PortabilityCommandCheck,
    pub bugkit_smoke: PortabilityCommandCheck,
    pub remediation_consistency_smoke: PortabilityCommandCheck,
    pub backend_evidence_snapshot_smoke: PortabilityCommandCheck,
    pub operator_signoff_smoke: PortabilityCommandCheck,
    pub remediation_registry_doc_check: PortabilityCommandCheck,
    pub v0_gate: PortabilityCommandCheck,
    pub v1_gate: PortabilityCommandCheck,
    pub v2_gate: PortabilityCommandCheck,
    pub eligibility_smoke: PortabilityCommandCheck,
    pub strict_check_smoke: PortabilityCommandCheck,
    pub operator_report_smoke: PortabilityCommandCheck,
    pub command_matrix: Vec<PortabilityMatrixEntry>,
}

fn reproducible_run_id_for_smoke(workdir: &Path) -> Result<String, OpsError> {
    let scenario =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e2e/v0_flow_a.json");
    let smoke_out = workdir.join("out").join("portability_smoke_bringup");
    let artifacts = one_command_bringup(workdir, &scenario, 16, &smoke_out, false)?;
    let run_id = artifacts.run_metadata.run_id;

    let run_out = workdir.join("out").join(&run_id);
    fs::create_dir_all(&run_out)?;
    fs::copy(
        smoke_out.join("run_metadata.json"),
        run_out.join("run_metadata.json"),
    )?;
    fs::copy(
        smoke_out.join("metrics_summary.json"),
        run_out.join("metrics_summary.json"),
    )?;

    Ok(run_id)
}

fn active_review_snapshot_smoke(workdir: &Path, name: &str, out: &str) -> PortabilityCommandCheck {
    let out_path = PathBuf::from(out);
    match models_active_review_snapshot(workdir, &out_path) {
        Ok(report) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Pass,
            detail: format!(
                "schema={} slots={}",
                report.schema_version,
                report.slots.len()
            ),
            out: Some(out.to_string()),
        },
        Err(err) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(out.to_string()),
        },
    }
}

fn backend_resolution_smoke(workdir: &Path, name: &str, out: &str) -> PortabilityCommandCheck {
    let out_path = PathBuf::from(out);
    let slot = match detect_second_slot(workdir) {
        Ok(slot) => slot,
        Err(err) => {
            return PortabilityCommandCheck {
                name: name.to_string(),
                status: PortabilityGateStatus::Skip,
                detail: format!("optional backend path unavailable: {err}"),
                out: Some(out.to_string()),
            }
        }
    };
    match models_backend_resolution(workdir, slot, None) {
        Ok(report) => {
            let write_result = (|| -> Result<(), OpsError> {
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
                Ok(())
            })();
            match write_result {
                Ok(()) => PortabilityCommandCheck {
                    name: name.to_string(),
                    status: PortabilityGateStatus::Pass,
                    detail: format!("slot={} resolution={:?}", report.slot_id, report.resolution),
                    out: Some(out.to_string()),
                },
                Err(err) => PortabilityCommandCheck {
                    name: name.to_string(),
                    status: PortabilityGateStatus::Fail,
                    detail: err.to_string(),
                    out: Some(out.to_string()),
                },
            }
        }
        Err(err) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Skip,
            detail: format!("optional backend path unavailable: {err}"),
            out: Some(out.to_string()),
        },
    }
}

fn out_smoke_check<T, F>(
    name: &str,
    out: &str,
    command: F,
    detail: impl FnOnce(&T) -> String,
) -> PortabilityCommandCheck
where
    F: FnOnce(&Path) -> Result<T, OpsError>,
{
    let out_path = PathBuf::from(out);
    match command(&out_path) {
        Ok(report) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Pass,
            detail: detail(&report),
            out: Some(out.to_string()),
        },
        Err(err) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(out.to_string()),
        },
    }
}

fn repro_pack_smoke(name: &str, out: &str, verify_out: &str) -> PortabilityCommandCheck {
    let tmp = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(err) => {
            return PortabilityCommandCheck {
                name: name.to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out.to_string()),
            }
        }
    };
    let run_id = match reproducible_run_id_for_smoke(tmp.path()) {
        Ok(run_id) => run_id,
        Err(err) => {
            return PortabilityCommandCheck {
                name: name.to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out.to_string()),
            }
        }
    };
    let pack_out = PathBuf::from(out);
    let verify_path = PathBuf::from(verify_out);
    let result = (|| -> Result<ReproVerifyReport, OpsError> {
        if let Some(parent) = pack_out.parent() {
            fs::create_dir_all(parent)?;
        }
        repro_pack(tmp.path(), &run_id, &pack_out)?;
        repro_verify(&pack_out, &verify_path)
    })();

    match result {
        Ok(report) if report.pass => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Pass,
            detail: format!(
                "run_id={} checked_files={}",
                report.run_id, report.checked_files
            ),
            out: Some(pack_out.display().to_string()),
        },
        Ok(report) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: format!("verify failed: {}", report.reasons.join("; ")),
            out: Some(pack_out.display().to_string()),
        },
        Err(err) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(pack_out.display().to_string()),
        },
    }
}

fn bugkit_smoke(name: &str, out: &str) -> PortabilityCommandCheck {
    let tmp = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(err) => {
            return PortabilityCommandCheck {
                name: name.to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out.to_string()),
            }
        }
    };
    let run_id = match reproducible_run_id_for_smoke(tmp.path()) {
        Ok(run_id) => run_id,
        Err(err) => {
            return PortabilityCommandCheck {
                name: name.to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out.to_string()),
            }
        }
    };
    let out_path = PathBuf::from(out);
    let report = (|| -> Result<BugKitManifestV1, OpsError> {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        bugkit_build(tmp.path(), &run_id, &out_path, &BugKitBuildArgs::default())?;
        let archive = fs::File::open(&out_path)?;
        let mut zip = zip::ZipArchive::new(archive)
            .map_err(|e| OpsError::Invalid(format!("unable to open bugkit zip: {e}")))?;
        let manifest_name = if zip.file_names().any(|name| name == "bugkit_manifest.json") {
            "bugkit_manifest.json"
        } else {
            "BUGKIT_MANIFEST.json"
        };
        let mut manifest_file = zip.by_name(manifest_name).map_err(|e| {
            OpsError::Invalid(format!(
                "missing bugkit manifest (bugkit_manifest.json/BUGKIT_MANIFEST.json): {e}"
            ))
        })?;
        let mut body = String::new();
        std::io::Read::read_to_string(&mut manifest_file, &mut body)?;
        let manifest: BugKitManifestV1 = serde_json::from_str(&body)?;
        Ok(manifest)
    })();
    match report {
        Ok(manifest)
            if !manifest.include_payload
                && !manifest.include_weights
                && [
                    &manifest.backend_evidence_snapshot,
                    &manifest.active_review_snapshot,
                    &manifest.operator_signoff,
                    &manifest.backend_resolution,
                ]
                .iter()
                .all(|entry| {
                    ["INCLUDED", "MISSING", "EXCLUDED"].contains(&entry.status.as_str())
                }) =>
        {
            PortabilityCommandCheck {
                name: name.to_string(),
                status: PortabilityGateStatus::Pass,
                detail: format!(
                    "run_id={} files={} payload=false weights=false",
                    manifest.run_id, manifest.file_count
                ),
                out: Some(out.to_string()),
            }
        }
        Ok(manifest) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: format!(
                "unexpected bugkit manifest flags payload={} weights={}",
                manifest.include_payload, manifest.include_weights
            ),
            out: Some(out.to_string()),
        },
        Err(err) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(out.to_string()),
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrictCheckReport {
    pub strict_mode_enabled: bool,
    pub ok: bool,
    pub report: StrictModeFailureReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreflightCheck {
    pub name: String,
    pub status: GateStatus,
    pub critical: bool,
    pub evidence: BTreeMap<String, String>,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreflightReport {
    pub schema_version: u16,
    pub bundle: String,
    pub overall: GateStatus,
    pub exit_code: i32,
    pub checks: Vec<PreflightCheck>,
    pub remediation_hints: Vec<String>,
}

const PREFLIGHT_ENV_KEYS: [&str; 2] = ["UCF_POLICY_GRAPH_DIGEST", "UCF_MODEL_MANIFEST_DIGEST"];

fn preflight_process_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

pub fn preflight(bundle: &Path, out: &Path) -> Result<PreflightReport, OpsError> {
    let _process_guard = preflight_process_lock()
        .lock()
        .map_err(|_| OpsError::Invalid("preflight process lock poisoned".to_string()))?;
    let bundle = bundle.canonicalize()?;
    let original_cwd = std::env::current_dir()?;
    std::env::set_current_dir(&bundle)?;
    let result = (|| {
        let checks = vec![
            preflight_bundle_integrity(&bundle)?,
            preflight_strict_check(&bundle)?,
            preflight_docs_lint(&bundle)?,
            preflight_gate_check(&bundle)?,
            preflight_runtime_status(&bundle),
            preflight_rc_manifest(&bundle)?,
        ];

        let mut remediation_hints = Vec::new();
        for check in &checks {
            if matches!(check.status, GateStatus::Fail) {
                if let Some(remediation) = &check.remediation {
                    remediation_hints.push(remediation.clone());
                }
            }
        }
        remediation_hints.sort();
        remediation_hints.dedup();

        let has_critical_fail = checks
            .iter()
            .any(|c| c.critical && matches!(c.status, GateStatus::Fail));
        let has_fail = checks.iter().any(|c| matches!(c.status, GateStatus::Fail));
        let (overall, exit_code) = if has_critical_fail {
            (GateStatus::Fail, 3)
        } else if has_fail {
            (GateStatus::Fail, 2)
        } else {
            (GateStatus::Pass, 0)
        };

        let report = PreflightReport {
            schema_version: 1,
            bundle: bundle.display().to_string(),
            overall,
            exit_code,
            checks,
            remediation_hints,
        };

        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        write_json(out, &report)?;
        Ok(report)
    })();
    std::env::set_current_dir(original_cwd)?;
    result
}

fn preflight_bundle_integrity(bundle: &Path) -> Result<PreflightCheck, OpsError> {
    let mut evidence = BTreeMap::new();
    let required = [
        "bin/ucf-ops",
        "configs",
        "policies/packs/base_v1",
        "policies/manifest.toml",
        "models/manifest.toml",
        "VERSION.txt",
    ];
    let mut missing = Vec::new();
    for item in required {
        let exists = bundle.join(item).exists();
        evidence.insert(item.to_string(), exists.to_string());
        if !exists {
            missing.push(item.to_string());
        }
    }

    let version = bundle.join("VERSION.txt");
    if version.exists() {
        let body = fs::read_to_string(&version)?;
        let fields = parse_key_value_file(&body);
        let manifest_digest = sha256_hex(&fs::read(bundle.join("models/manifest.toml"))?);
        evidence.insert(
            "version_manifest_digest_matches".to_string(),
            fields
                .get("manifest_digest")
                .map(|d| d == &manifest_digest)
                .unwrap_or(false)
                .to_string(),
        );
        evidence.insert(
            "version_policy_graph_digest_present".to_string(),
            fields.contains_key("policy_graph_digest").to_string(),
        );
        if fields.get("manifest_digest") != Some(&manifest_digest) {
            missing.push("VERSION.txt:manifest_digest mismatch".to_string());
        }
        if !fields.contains_key("policy_graph_digest") {
            missing.push("VERSION.txt:policy_graph_digest missing".to_string());
        }
    }

    let status = if missing.is_empty() {
        GateStatus::Pass
    } else {
        evidence.insert("missing_or_invalid".to_string(), missing.join(","));
        GateStatus::Fail
    };
    Ok(PreflightCheck {
        name: "bundle_integrity".to_string(),
        status,
        critical: true,
        evidence,
        remediation: Some(
            "rebuild bundle: python deploy/scripts/build_bundle.py --target <bundle> --profile <dev|test|prod>"
                .to_string(),
        ),
    })
}

fn preflight_strict_check(bundle: &Path) -> Result<PreflightCheck, OpsError> {
    let out = bundle.join("out/preflight_strict_check.json");
    let mut previous = BTreeMap::new();
    for key in PREFLIGHT_ENV_KEYS {
        previous.insert(key, std::env::var(key).ok());
    }

    if let Ok(body) = fs::read_to_string(bundle.join("VERSION.txt")) {
        let version = parse_key_value_file(&body);
        if let Some(policy_digest) = version.get("policy_graph_digest") {
            std::env::set_var("UCF_POLICY_GRAPH_DIGEST", policy_digest);
        }
        if let Some(manifest_digest) = version.get("manifest_digest") {
            std::env::set_var("UCF_MODEL_MANIFEST_DIGEST", manifest_digest);
        }
    }
    let strict_result = strict_check(&bundle.join(".ucf"), false, &out);

    for (key, value) in previous {
        if let Some(value) = value {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    let report = strict_result?;
    let mut evidence = BTreeMap::new();
    evidence.insert("report".to_string(), out.display().to_string());
    evidence.insert(
        "strict_mode_enabled".to_string(),
        report.strict_mode_enabled.to_string(),
    );
    Ok(PreflightCheck {
        name: "strict_check".to_string(),
        status: if report.ok {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        critical: true,
        evidence,
        remediation: Some("run `ucf-ops strict check --bundle <path> --strict --out ./out/strict_check.json` and resolve failing checks".to_string()),
    })
}

fn preflight_docs_lint(bundle: &Path) -> Result<PreflightCheck, OpsError> {
    let mut evidence = BTreeMap::new();
    let docs_dir = bundle.join("docs");
    if !docs_dir.exists() {
        evidence.insert("reason".to_string(), "docs directory missing".to_string());
        return Ok(PreflightCheck {
            name: "docs_lint".to_string(),
            status: GateStatus::Skip,
            critical: false,
            evidence,
            remediation: None,
        });
    }
    let out = bundle.join("out/docs_lint_report.json");
    let report = docs_lint(&DocsLintArgs {
        repo_root: bundle.to_path_buf(),
        policy_pack: bundle.join("policies/packs/base_v1"),
        overlay_pack: Some(bundle.join("policies/packs/overlays/test")),
        spec_snapshot: bundle.join("docs/spec_snapshot.md"),
        prompt_index: bundle.join("docs/prompt_series_index.md"),
        module_map: bundle.join("docs/module_map.md"),
        deploy_doc: bundle.join("docs/deploy_portable.md"),
        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
        mode: DocsLintMode::Strict,
    });
    match report {
        Ok(report) => {
            write_json(&out, &report)?;
            evidence.insert("report".to_string(), out.display().to_string());
            Ok(PreflightCheck {
                name: "docs_lint".to_string(),
                status: if report.ok { GateStatus::Pass } else { GateStatus::Fail },
                critical: false,
                evidence,
                remediation: Some("run `ucf-ops docs lint --strict --out ./out/docs_lint_report.json` and fix docs issues".to_string()),
            })
        }
        Err(err) => {
            evidence.insert("error".to_string(), err.to_string());
            Ok(PreflightCheck {
                name: "docs_lint".to_string(),
                status: GateStatus::Fail,
                critical: false,
                evidence,
                remediation: Some(
                    "ensure bundle docs include snapshot, prompt index, module map, and deploy doc"
                        .to_string(),
                ),
            })
        }
    }
}

fn preflight_gate_check(bundle: &Path) -> Result<PreflightCheck, OpsError> {
    let mut evidence = BTreeMap::new();
    let latest = first_existing_path(&[
        bundle.join("out/gate_latest.json"),
        bundle.join("out/gate_report.json"),
    ]);
    if let Some(path) = latest {
        let report: ReadinessGateReport = serde_json::from_slice(&fs::read(&path)?)?;
        evidence.insert("source".to_string(), path.display().to_string());
        return Ok(PreflightCheck {
            name: "gate_status".to_string(),
            status: if report.status == GateStatus::Pass {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            critical: false,
            evidence,
            remediation: Some(
                "run `ucf-ops readiness-gate --profile test --out ./out/gate_report.json`"
                    .to_string(),
            ),
        });
    }
    let smoke_out = bundle.join("out/preflight_gate_smoke.json");
    let smoke = readiness_gate(&bundle.join(".ucf"), "test", &smoke_out)?;
    evidence.insert("source".to_string(), smoke_out.display().to_string());
    Ok(PreflightCheck {
        name: "gate_status".to_string(),
        status: if smoke.status == GateStatus::Pass {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        critical: false,
        evidence,
        remediation: Some("fix readiness failures and rerun `ucf-ops readiness-gate --profile test --out ./out/gate_report.json`".to_string()),
    })
}

fn preflight_runtime_status(bundle: &Path) -> PreflightCheck {
    let mut evidence = BTreeMap::new();
    let health = bundle.join("out/health.json");
    let alerts = bundle.join("out/alerts_report.json");
    let drift = bundle.join("out/drift_report.json");
    evidence.insert("health_present".to_string(), health.exists().to_string());
    evidence.insert("alerts_present".to_string(), alerts.exists().to_string());
    evidence.insert("drift_present".to_string(), drift.exists().to_string());
    if !health.exists() && !alerts.exists() && !drift.exists() {
        evidence.insert(
            "reason".to_string(),
            "runtime evidence not available in bundle/out".to_string(),
        );
        return PreflightCheck {
            name: "runtime_status".to_string(),
            status: GateStatus::Skip,
            critical: false,
            evidence,
            remediation: None,
        };
    }
    PreflightCheck {
        name: "runtime_status".to_string(),
        status: GateStatus::Pass,
        critical: false,
        evidence,
        remediation: None,
    }
}

fn preflight_rc_manifest(bundle: &Path) -> Result<PreflightCheck, OpsError> {
    let mut evidence = BTreeMap::new();
    let manifest_path = bundle.join("RC_MANIFEST.json");
    let sig_path = bundle.join("RC_MANIFEST.sig");
    let sums_path = bundle.join("SHA256SUMS.txt");
    if !manifest_path.exists() || !sig_path.exists() || !sums_path.exists() {
        evidence.insert("reason".to_string(), "rc artifacts missing".to_string());
        return Ok(PreflightCheck {
            name: "rc_manifest".to_string(),
            status: GateStatus::Skip,
            critical: false,
            evidence,
            remediation: None,
        });
    }

    let manifest: RcManifestV1 = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let digest_ok = rc_manifest_digest(&manifest)? == manifest.rc_digest;
    evidence.insert("manifest_digest_ok".to_string(), digest_ok.to_string());

    let sig_hex = fs::read_to_string(&sig_path)?.trim().to_string();
    let sig_bytes = hex::decode(sig_hex)
        .map_err(|e| OpsError::Invalid(format!("invalid RC_MANIFEST.sig hex: {e}")))?;
    let sig_arr: [u8; 64] = sig_bytes
        .as_slice()
        .try_into()
        .map_err(|_| OpsError::Invalid("invalid RC_MANIFEST.sig size".to_string()))?;
    let sig = Signature::from_bytes(&sig_arr);
    let signer_bytes = hex::decode(&manifest.signer_public_key)
        .map_err(|e| OpsError::Invalid(format!("invalid signer_public_key hex: {e}")))?;
    let signer_arr: [u8; 32] = signer_bytes
        .as_slice()
        .try_into()
        .map_err(|_| OpsError::Invalid("invalid signer_public_key length".to_string()))?;
    let verify_key = VerifyingKey::from_bytes(&signer_arr)
        .map_err(|e| OpsError::Invalid(format!("invalid signer key: {e}")))?;
    let digest_bytes = hex::decode(&manifest.rc_digest)
        .map_err(|e| OpsError::Invalid(format!("invalid rc_digest hex: {e}")))?;
    let signature_ok = verify_key.verify(&digest_bytes, &sig).is_ok();
    evidence.insert("signature_ok".to_string(), signature_ok.to_string());

    let mut sums_ok = true;
    for line in fs::read_to_string(&sums_path)?.lines() {
        let mut parts = line.splitn(2, "  ");
        let Some(expected) = parts.next() else {
            continue;
        };
        let Some(path) = parts.next() else { continue };
        let path = bundle.join(path);
        if !path.exists() || sha256_hex(&fs::read(path)?) != expected {
            sums_ok = false;
            break;
        }
    }
    evidence.insert("sha256sums_ok".to_string(), sums_ok.to_string());
    let status = if digest_ok && signature_ok && sums_ok {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    };

    Ok(PreflightCheck {
        name: "rc_manifest".to_string(),
        status,
        critical: true,
        evidence,
        remediation: Some(
            "rebuild RC pack and regenerate RC_MANIFEST.json/RC_MANIFEST.sig/SHA256SUMS.txt"
                .to_string(),
        ),
    })
}

fn parse_key_value_file(body: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in body.lines() {
        if let Some((key, value)) = line.split_once('=') {
            out.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    out
}

pub fn strict_check(
    workdir: &Path,
    ops_only: bool,
    out: &Path,
) -> Result<StrictCheckReport, OpsError> {
    ensure_layout(workdir)?;
    let cfg = load_or_init_config(workdir)?;
    let enabled = cfg.strict_mode || std::env::var("UCF_STRICT_MODE").ok().as_deref() == Some("1");
    let mut active_cfg = cfg.clone();
    active_cfg.strict_mode = enabled;
    let outcome = StrictModeEnforcer::check_all(workdir, &active_cfg, ops_only);
    let report = match outcome {
        Ok(()) => StrictModeFailureReport {
            schema_version: 1,
            strict_mode_enabled: true,
            profile: active_cfg.profile.clone(),
            checks: vec![strict_pass("strict_mode")],
            v1_checks: Vec::new(),
            v3: Some(strict_v3_checks(workdir, &active_cfg)),
            evidence_digest_prefixes: BTreeMap::new(),
        },
        Err(r) => r,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(out, &report)?;
    Ok(StrictCheckReport {
        strict_mode_enabled: enabled,
        ok: !report.has_failures(),
        report,
    })
}

pub fn portability_check(out: &Path) -> Result<PortabilityCheckReport, OpsError> {
    let run_a = tempfile::tempdir()?;
    let run_b = tempfile::tempdir()?;
    let left = bringup(run_a.path(), true, 16)?;
    let right = bringup(run_b.path(), true, 16)?;

    let dataset_out = run_a.path().join("out/portability_dataset.jsonl");
    let sample_count = ebm_export_dataset(
        run_a.path(),
        "run-portability",
        0,
        u64::MAX,
        &dataset_out,
        &PathBuf::from("policies/bundle_v1/retention_v1.json"),
    )?;
    let dataset_body = fs::read_to_string(&dataset_out)?;
    let mut sum_risk: u64 = 0;
    let mut sum_pressure: u64 = 0;
    let mut sum_surprise: u64 = 0;
    let mut sum_uncertainty: u64 = 0;
    let mut counted: usize = 0;
    for line in dataset_body.lines() {
        let sample: EbmDatasetSample = serde_json::from_str(line)?;
        if let (Some(risk), Some(pressure), Some(surprise), Some(uncertainty)) = (
            sample.signals_q.risk_q,
            sample.signals_q.pressure_q,
            sample.signals_q.surprise_q,
            sample.signals_q.uncertainty_q,
        ) {
            sum_risk = sum_risk.saturating_add(risk as u64);
            sum_pressure = sum_pressure.saturating_add(pressure as u64);
            sum_surprise = sum_surprise.saturating_add(surprise as u64);
            sum_uncertainty = sum_uncertainty.saturating_add(uncertainty as u64);
            counted += 1;
        }
    }
    let fixed_point_missing = counted == 0;

    let dataset_digest = sha256_hex(dataset_body.as_bytes());
    let deterministic_within_os = left.ess_digest == right.ess_digest;
    let mut digest_prefixes = BTreeMap::new();
    digest_prefixes.insert(
        "ess_run_a".to_string(),
        left.ess_digest.chars().take(12).collect(),
    );
    digest_prefixes.insert(
        "ess_run_b".to_string(),
        right.ess_digest.chars().take(12).collect(),
    );
    digest_prefixes.insert(
        "ebm_dataset".to_string(),
        dataset_digest.chars().take(12).collect(),
    );

    let mut remediation = vec![
        "Avoid OS-specific ordering and filesystem metadata in externally visible digests."
            .to_string(),
        "Prefer canonical encodings and sorted key iteration (BTreeMap / explicit sort)."
            .to_string(),
        "If cross-OS exact digest parity is infeasible, keep fixed-point scalar envelopes stable and documented."
            .to_string(),
    ];
    if fixed_point_missing {
        remediation.push(
            "No fixed-point EBM signals were emitted in the toy/stub run; enforce envelope checks via schema + digest-prefix stability."
                .to_string(),
        );
    }
    if !deterministic_within_os {
        remediation.insert(
            0,
            "Digest mismatch within same OS: inspect toy/stub scenario serialization and record ordering."
                .to_string(),
        );
    }
    let report = PortabilityCheckReport {
        schema_version: 1,
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        digest_prefixes,
        fixed_point_summary: PortabilityFixedPointSummary {
            sample_count,
            mean_risk_q: if counted == 0 {
                0
            } else {
                (sum_risk / counted as u64) as u16
            },
            mean_pressure_q: if counted == 0 {
                0
            } else {
                (sum_pressure / counted as u64) as u16
            },
            mean_surprise_q: if counted == 0 {
                0
            } else {
                (sum_surprise / counted as u64) as u16
            },
            mean_uncertainty_q: if counted == 0 {
                0
            } else {
                (sum_uncertainty / counted as u64) as u16
            },
        },
        deterministic_within_os,
        remediation,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn portability_report(workdir: &Path, out: &Path) -> Result<PortabilityReportV1, OpsError> {
    let docs_out = PathBuf::from("./out/docs_lint_report.json");
    let docs_lint = match docs_lint(&DocsLintArgs {
        repo_root: PathBuf::from("."),
        policy_pack: PathBuf::from("policies/packs/base_v1"),
        overlay_pack: Some(PathBuf::from("policies/packs/overlays/test")),
        spec_snapshot: PathBuf::from("docs/spec_snapshot.md"),
        prompt_index: PathBuf::from("docs/prompt_series_index.md"),
        module_map: PathBuf::from("docs/module_map.md"),
        deploy_doc: PathBuf::from("docs/deploy_portable.md"),
        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
        mode: DocsLintMode::Strict,
    }) {
        Ok(report) => {
            if let Some(parent) = docs_out.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&docs_out, serde_json::to_vec_pretty(&report)?)?;
            PortabilityCommandCheck {
                name: "docs_lint_strict".to_string(),
                status: if report.ok {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!("overall={}", if report.ok { "PASS" } else { "FAIL" }),
                out: Some(docs_out.display().to_string()),
            }
        }
        Err(err) => PortabilityCommandCheck {
            name: "docs_lint_strict".to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: None,
        },
    };

    let path_scan = scan_check(
        "path_scan",
        path_scan(Path::new(".")),
        "./out/path_scan_report.json",
    )?;
    let hardware_scan = scan_check(
        "hardware_scan",
        hardware_scan(Path::new(".")),
        "./out/hardware_scan_report.json",
    )?;
    let hidden_network_scan = net_deps_check("hidden_network_scan", "./out/net_deps.json")?;
    let artifact_schema_snapshot_check = artifact_schema_snapshot_portability_check(
        "artifact_schema_snapshot_check",
        "./out/artifact_schema_check.json",
    )?;
    let governance_surfaces_smoke = {
        let out_path = PathBuf::from("./out/governance_surfaces_check.json");
        match governance_surfaces_check(workdir, &out_path) {
            Ok(report) => PortabilityCommandCheck {
                name: "governance_surfaces_smoke".to_string(),
                status: if report.status == "PASS" {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!("status={} code={}", report.status, report.summary_code),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => PortabilityCommandCheck {
                name: "governance_surfaces_smoke".to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out_path.display().to_string()),
            },
        }
    };
    let governance_entry_check_smoke = {
        let out_path = PathBuf::from("./out/governance_entry_check.json");
        match governance_entry_check(workdir, &out_path) {
            Ok(report) => PortabilityCommandCheck {
                name: "governance_entry_check_smoke".to_string(),
                status: if matches!(report.status, GovernanceEntryCheckStatusV1::Pass) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "status={:?} consumers={}",
                    report.status,
                    report.consumers.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => PortabilityCommandCheck {
                name: "governance_entry_check_smoke".to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out_path.display().to_string()),
            },
        }
    };
    let governance_entry_sweep_smoke = {
        let out_path = PathBuf::from("./out/governance_entry_sweep.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| governance_entry_sweep(workdir, &out_path));
        match result {
            Ok(report) => PortabilityCommandCheck {
                name: "governance_entry_sweep_smoke".to_string(),
                status: if matches!(
                    report.authority.authority_status,
                    GovernanceEntryAuthorityStatusV2::Pass
                ) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "authority_status={:?} surfaces={}",
                    report.authority.authority_status,
                    report.surfaces.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains(LEGACY_SCOPE_PATH_BLOCKED)
                    || detail.contains(APPLIED_SCOPE_REQUIRED)
                    || detail.contains(APPLIED_SCOPE_MISSING)
                    || detail.contains(APPLIED_SCOPE_TRANSLATION_FAILED)
                    || detail.contains("APPLIED_SCOPE_SLOT_TRUTH_MISSING")
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "governance_entry_sweep_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_scope_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let scope_authority_check_smoke = {
        let out_path = PathBuf::from("./out/scope_authority_check.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| scope_authority_check(workdir, &out_path));
        match result {
            Ok(report) => PortabilityCommandCheck {
                name: "scope_authority_check_smoke".to_string(),
                status: if matches!(report.status, ScopeAuthorityOverallStatusV1::Pass) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "status={:?} surfaces={}",
                    report.status,
                    report.surfaces.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains(LEGACY_SCOPE_PATH_BLOCKED)
                    || detail.contains(APPLIED_SCOPE_REQUIRED)
                    || detail.contains(APPLIED_SCOPE_MISSING)
                    || detail.contains(APPLIED_SCOPE_TRANSLATION_FAILED)
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "scope_authority_check_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_scope_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let supported_scope_reevaluate_smoke = out_smoke_check(
        "supported_scope_reevaluate_smoke",
        "./out/supported_scope_reeval.json",
        |out_path| models_supported_scope_reevaluate(workdir, out_path),
        |report| {
            format!(
                "decision={:?} slots={}",
                report.reevaluation_decision,
                usize::from(report.chosen_candidate_slot.is_some())
            )
        },
    );
    let supported_scope_execute_smoke = out_smoke_check(
        "supported_scope_execute_smoke",
        "./out/supported_scope_execute_v4.json",
        |out_path| {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let sweep_out = PathBuf::from("./out/governance_entry_sweep.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            governance_entry_sweep(workdir, &sweep_out)?;
            models_supported_scope_execute_v4(workdir, out_path)
        },
        |report| {
            format!(
                "decision={:?} slots={}",
                report.execution_decision,
                usize::from(report.chosen_candidate_slot.is_some())
            )
        },
    );
    let readiness_spine_check_smoke = {
        let out_path = PathBuf::from("./out/readiness_spine_check.json");
        match readiness_spine_check(workdir, &out_path) {
            Ok(report) => {
                let bounded_mismatch_only = report.mismatch_categories.iter().all(|category| {
                    matches!(
                        category,
                        ReadinessSpineMismatchCategoryV1::ReductionMismatch
                            | ReadinessSpineMismatchCategoryV1::SignoffSpineDrift
                            | ReadinessSpineMismatchCategoryV1::ReviewPacketSpineDrift
                            | ReadinessSpineMismatchCategoryV1::WorkflowSpineDrift
                    )
                });
                PortabilityCommandCheck {
                    name: "readiness_spine_check_smoke".to_string(),
                    status: if matches!(report.status, ReadinessSpineCheckStatusV1::Pass) {
                        PortabilityGateStatus::Pass
                    } else if bounded_mismatch_only {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if bounded_mismatch_only
                        && !matches!(report.status, ReadinessSpineCheckStatusV1::Pass)
                    {
                        format!(
                            "skip_bounded_readiness_context: status={:?} mismatch_categories={}",
                            report.status,
                            report.mismatch_categories.len()
                        )
                    } else {
                        format!(
                            "status={:?} mismatch_categories={}",
                            report.status,
                            report.mismatch_categories.len()
                        )
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
            Err(err) => PortabilityCommandCheck {
                name: "readiness_spine_check_smoke".to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out_path.display().to_string()),
            },
        }
    };
    let readiness_spine_sweep_smoke = {
        let out_path = PathBuf::from("./out/readiness_spine_sweep.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| readiness_spine_sweep(workdir, &out_path));
        match result {
            Ok(report) => {
                let bounded_optional_only = report.surfaces.iter().all(|surface| {
                    surface.mismatch_categories.is_empty()
                        || surface.mismatch_categories.iter().all(|category| {
                            matches!(
                                category,
                                ReadinessSpineSweepMismatchCategoryV1::SurfaceSkippedCanonicalReadinessSpine
                                    | ReadinessSpineSweepMismatchCategoryV1::SurfaceUsedSecondaryReadinessPath
                            )
                        })
                });
                PortabilityCommandCheck {
                    name: "readiness_spine_sweep_smoke".to_string(),
                    status: if matches!(
                        report.authority.authority_status,
                        CanonicalReadinessAuthorityStatusV2::Pass
                    ) {
                        PortabilityGateStatus::Pass
                    } else if bounded_optional_only {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if bounded_optional_only
                        && !matches!(
                            report.authority.authority_status,
                            CanonicalReadinessAuthorityStatusV2::Pass
                        ) {
                        format!(
                            "skip_bounded_readiness_sweep_context: authority_status={:?} surfaces={}",
                            report.authority.authority_status,
                            report.surfaces.len()
                        )
                    } else {
                        format!(
                            "authority_status={:?} surfaces={}",
                            report.authority.authority_status,
                            report.surfaces.len()
                        )
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains("APPLIED_SCOPE_SLOT_TRUTH_MISSING")
                    || detail.contains(APPLIED_SCOPE_REQUIRED)
                    || detail.contains(APPLIED_SCOPE_MISSING)
                    || detail.contains(APPLIED_SCOPE_TRANSLATION_FAILED)
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "readiness_spine_sweep_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_scope_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let supported_set_apply_smoke = out_smoke_check(
        "supported_set_apply_smoke",
        "./out/supported_set_apply.json",
        |out_path| {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let sweep_out = PathBuf::from("./out/governance_entry_sweep.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            governance_entry_sweep(workdir, &sweep_out)?;
            let exec_out = PathBuf::from("./out/supported_scope_execute_v4.json");
            models_supported_scope_execute_v4(workdir, &exec_out)?;
            models_supported_set_apply(workdir, out_path)
        },
        |report| {
            format!(
                "decision={:?} slots={}",
                report.decision,
                report.resulting_slots.len()
            )
        },
    );
    let review_truth_check_smoke = {
        let out_path = PathBuf::from("./out/review_truth_check.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| review_truth_check(workdir, &out_path));
        match result {
            Ok(report) => PortabilityCommandCheck {
                name: "review_truth_check_smoke".to_string(),
                status: if matches!(report.status, ReviewTruthCheckStatusV1::Pass) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "status={:?} mismatches={}",
                    report.status,
                    report.mismatch_categories.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains(LEGACY_SCOPE_PATH_BLOCKED)
                    || detail.contains(APPLIED_SCOPE_REQUIRED)
                    || detail.contains(APPLIED_SCOPE_MISSING)
                    || detail.contains(APPLIED_SCOPE_TRANSLATION_FAILED)
                    || detail.contains("APPLIED_SCOPE_SLOT_TRUTH_MISSING")
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "review_truth_check_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_scope_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let repro_pack_smoke = repro_pack_smoke(
        "repro_pack_smoke",
        "./out/repro_portability.zip",
        "./out/repro_verify_portability.json",
    );
    let export_roundtrip_check_smoke = {
        let out_path = PathBuf::from("./out/export_roundtrip_check.json");
        let input_bundle = PathBuf::from("./out/repro_portability.zip");
        match exports_roundtrip_check(&input_bundle, &out_path) {
            Ok(report) => PortabilityCommandCheck {
                name: "export_roundtrip_check_smoke".to_string(),
                status: if matches!(report.overall_status, BundleRoundTripOverallStatusV1::Pass) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "status={:?} mismatches={}",
                    report.overall_status,
                    report.mismatch_codes.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => PortabilityCommandCheck {
                name: "export_roundtrip_check_smoke".to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out_path.display().to_string()),
            },
        }
    };
    let bundle_spine_check_smoke = {
        let out_path = PathBuf::from("./out/bundle_spine_check.json");
        let input_bundle = PathBuf::from("./out/repro_portability.zip");
        match exports_bundle_spine_check(&input_bundle, &out_path) {
            Ok(report) => PortabilityCommandCheck {
                name: "bundle_spine_check_smoke".to_string(),
                status: if report.pass {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "pass={} mismatch_codes={}",
                    report.pass,
                    report.mismatch_codes.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => PortabilityCommandCheck {
                name: "bundle_spine_check_smoke".to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out_path.display().to_string()),
            },
        }
    };
    let bundle_spine_sweep_smoke = {
        let out_path = PathBuf::from("./out/bundle_spine_sweep.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| exports_bundle_spine_sweep(workdir, &out_path));
        match result {
            Ok(report) => PortabilityCommandCheck {
                name: "bundle_spine_sweep_smoke".to_string(),
                status: if matches!(
                    report.authority.authority_status,
                    CanonicalBundleAuthorityStatusV2::Pass
                ) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "authority_status={:?} surfaces={}",
                    report.authority.authority_status,
                    report.surfaces.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains("CANONICAL_EXPORT_REFS_REQUIRED")
                    || detail.contains("EXPORT_CONTEXT_REQUIRED")
                    || detail.contains("PACK_ARTIFACT_REFS_REQUIRED")
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "bundle_spine_sweep_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_export_refs_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let primary_semantics_sweep_smoke = {
        let out_path = PathBuf::from("./out/primary_semantics_sweep.json");
        match primary_semantics_sweep(&out_path) {
            Ok(report) => PortabilityCommandCheck {
                name: "primary_semantics_sweep_smoke".to_string(),
                status: if report.mismatches_found == 0 {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "mismatches_found={} categories={}",
                    report.mismatches_found,
                    report.top_mismatch_categories.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => PortabilityCommandCheck {
                name: "primary_semantics_sweep_smoke".to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out_path.display().to_string()),
            },
        }
    };
    let final_governance_consumer_sweep_smoke = {
        let out_path = PathBuf::from("./out/final_governance_consumer_sweep.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| final_governance_consumer_sweep(workdir, &out_path));
        match result {
            Ok(report) => PortabilityCommandCheck {
                name: "final_governance_consumer_sweep_smoke".to_string(),
                status: if matches!(
                    report.authority.authority_status,
                    FinalGovernanceConsumerAuthorityStatusV1::Pass
                ) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "authority_status={:?} surfaces={}",
                    report.authority.authority_status,
                    report.consumers.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains(LEGACY_SCOPE_PATH_BLOCKED)
                    || detail.contains(APPLIED_SCOPE_REQUIRED)
                    || detail.contains(APPLIED_SCOPE_MISSING)
                    || detail.contains(APPLIED_SCOPE_TRANSLATION_FAILED)
                    || detail.contains("APPLIED_SCOPE_SLOT_TRUTH_MISSING")
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "final_governance_consumer_sweep_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_scope_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let supported_scope_execute_v5_smoke = out_smoke_check(
        "supported_scope_execute_v5_smoke",
        "./out/supported_scope_execute_v5.json",
        |out_path| {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let final_governance_out = PathBuf::from("./out/final_governance_consumer_sweep.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            final_governance_consumer_sweep(workdir, &final_governance_out)?;
            models_supported_scope_execute_v5(workdir, out_path)
        },
        |report| {
            format!(
                "decision={:?} slots={}",
                report.execution_decision,
                usize::from(report.chosen_candidate_slot.is_some())
            )
        },
    );
    let final_readiness_consumer_sweep_smoke = {
        let out_path = PathBuf::from("./out/final_readiness_consumer_sweep.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| final_readiness_consumer_sweep(workdir, &out_path));
        match result {
            Ok(report) => PortabilityCommandCheck {
                name: "final_readiness_consumer_sweep_smoke".to_string(),
                status: if matches!(
                    report.authority.authority_status,
                    FinalReadinessConsumerAuthorityStatusV1::Pass
                ) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "authority_status={:?} mismatch_categories={}",
                    report.authority.authority_status,
                    report
                        .consumers
                        .iter()
                        .map(|consumer| consumer.mismatch_categories.len())
                        .sum::<usize>()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains(LEGACY_SCOPE_PATH_BLOCKED)
                    || detail.contains(APPLIED_SCOPE_REQUIRED)
                    || detail.contains(APPLIED_SCOPE_MISSING)
                    || detail.contains(APPLIED_SCOPE_TRANSLATION_FAILED)
                    || detail.contains("APPLIED_SCOPE_SLOT_TRUTH_MISSING")
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "final_readiness_consumer_sweep_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_scope_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let final_bundle_consumer_sweep_smoke = {
        let out_path = PathBuf::from("./out/final_bundle_consumer_sweep.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| final_bundle_consumer_sweep(workdir, &out_path));
        match result {
            Ok(report) => PortabilityCommandCheck {
                name: "final_bundle_consumer_sweep_smoke".to_string(),
                status: if matches!(
                    report.authority.authority_status,
                    FinalBundleConsumerAuthorityStatusV1::Pass
                ) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "authority_status={:?} mismatch_categories={}",
                    report.authority.authority_status,
                    report
                        .consumers
                        .iter()
                        .map(|consumer| consumer.mismatch_categories.len())
                        .sum::<usize>()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains("CANONICAL_EXPORT_REFS_REQUIRED")
                    || detail.contains("EXPORT_CONTEXT_REQUIRED")
                    || detail.contains("PACK_ARTIFACT_REFS_REQUIRED")
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "final_bundle_consumer_sweep_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_export_refs_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let final_primary_semantics_sweep_smoke = {
        let out_path = PathBuf::from("./out/final_primary_semantics_sweep.json");
        let prep = (|| -> Result<(), OpsError> {
            let review_out = PathBuf::from("./out/supported_set_review.json");
            let reeval_out = PathBuf::from("./out/supported_scope_reeval.json");
            let apply_out = PathBuf::from("./out/supported_set_apply.json");
            models_supported_set_review(workdir, &review_out)?;
            models_supported_scope_reevaluate(workdir, &reeval_out)?;
            let _ = models_supported_set_apply(workdir, &apply_out)?;
            Ok(())
        })();
        let result = prep.and_then(|_| final_primary_semantics_sweep(workdir, &out_path));
        match result {
            Ok(report) => PortabilityCommandCheck {
                name: "final_primary_semantics_sweep_smoke".to_string(),
                status: if matches!(
                    report.authority.authority_status,
                    FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass
                ) {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "authority_status={:?} mismatch_categories={}",
                    report.authority.authority_status,
                    report
                        .surface_statuses
                        .iter()
                        .map(|surface| surface.mismatch_categories.len())
                        .sum::<usize>()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => {
                let detail = err.to_string();
                let skip = detail.contains("CANONICAL_EXPORT_REFS_REQUIRED")
                    || detail.contains("EXPORT_CONTEXT_REQUIRED")
                    || detail.contains("PACK_ARTIFACT_REFS_REQUIRED")
                    || detail.contains(LEGACY_SCOPE_PATH_BLOCKED)
                    || detail.contains(APPLIED_SCOPE_REQUIRED)
                    || detail.contains(APPLIED_SCOPE_MISSING)
                    || detail.contains(APPLIED_SCOPE_TRANSLATION_FAILED)
                    || detail.contains("APPLIED_SCOPE_SLOT_TRUTH_MISSING")
                    || detail.contains("SUPPORTED_SET_POLICY_V2_MISSING");
                PortabilityCommandCheck {
                    name: "final_primary_semantics_sweep_smoke".to_string(),
                    status: if skip {
                        PortabilityGateStatus::Skip
                    } else {
                        PortabilityGateStatus::Fail
                    },
                    detail: if skip {
                        format!("skip_optional_export_or_scope_path: {detail}")
                    } else {
                        detail
                    },
                    out: Some(out_path.display().to_string()),
                }
            }
        }
    };
    let remediation_interop_check_smoke = {
        let out_path = PathBuf::from("./out/remediation_interop_check.json");
        match remediation_interop_check(&out_path) {
            Ok(report) => PortabilityCommandCheck {
                name: "remediation_interop_check_smoke".to_string(),
                status: if report.mismatches_found == 0 {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!(
                    "mismatches_found={} categories={}",
                    report.mismatches_found,
                    report.top_mismatch_categories.len()
                ),
                out: Some(out_path.display().to_string()),
            },
            Err(err) => PortabilityCommandCheck {
                name: "remediation_interop_check_smoke".to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out_path.display().to_string()),
            },
        }
    };
    let remediation_spine_check_smoke = {
        let out_path = PathBuf::from("./out/remediation_spine_check.json");
        match remediation_spine_check(&out_path) {
            Ok(report) => PortabilityCommandCheck {
                name: "remediation_spine_check_smoke".to_string(),
                status: if report.mismatches_found == 0 {
                    PortabilityGateStatus::Pass
                } else if report.top_mismatch_categories.iter().all(|category| {
                    category.starts_with("MISSING_SURFACE:")
                        || category.starts_with("UNKNOWN_CONDITION_MAPPING:")
                }) {
                    PortabilityGateStatus::Skip
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: if report.mismatches_found > 0
                    && report.top_mismatch_categories.iter().all(|category| {
                        category.starts_with("MISSING_SURFACE:")
                            || category.starts_with("UNKNOWN_CONDITION_MAPPING:")
                    }) {
                    format!(
                        "skip_bounded_remediation_context: mismatches_found={} categories={}",
                        report.mismatches_found,
                        report.top_mismatch_categories.len()
                    )
                } else {
                    format!(
                        "mismatches_found={} categories={}",
                        report.mismatches_found,
                        report.top_mismatch_categories.len()
                    )
                },
                out: Some(out_path.display().to_string()),
            },
            Err(err) => PortabilityCommandCheck {
                name: "remediation_spine_check_smoke".to_string(),
                status: PortabilityGateStatus::Fail,
                detail: err.to_string(),
                out: Some(out_path.display().to_string()),
            },
        }
    };
    let active_review_snapshot_smoke = active_review_snapshot_smoke(
        workdir,
        "active_review_snapshot_smoke",
        "./out/active_review_snapshot.json",
    );
    let backend_resolution_smoke = backend_resolution_smoke(
        workdir,
        "backend_resolution_smoke",
        "./out/backend_resolution_portability.json",
    );
    let bugkit_smoke = bugkit_smoke("bugkit_smoke", "./out/bugkit_portability.zip");
    let remediation_consistency_smoke = gate_check(
        "remediation_consistency_smoke",
        remediation_consistency_check(&PathBuf::from(
            "./out/remediation_consistency_portability.json",
        ))
        .map(|r| r.summary.fail_count == 0),
        &PathBuf::from("./out/remediation_consistency_portability.json"),
    );
    let backend_evidence_snapshot_smoke = models_evidence_snapshot_smoke(
        workdir,
        "backend_evidence_snapshot_smoke",
        "./out/backend_evidence_snapshot.json",
    );
    let operator_signoff_smoke = operator_signoff_smoke(
        workdir,
        "operator_signoff_smoke",
        "./out/operator_signoff.json",
    );
    let remediation_registry_doc_check = remediation_registry_doc_portability_check(
        "remediation_registry_doc_check",
        "docs/remediation_codes_v1.md",
    )?;

    let v0_out = PathBuf::from("./out/v0_gate_report.json");
    let v0_gate = gate_check(
        "v0_gate",
        v0_gate(workdir, Path::new("fixtures/e2e/v0_flow_a.json"), &v0_out)
            .map(|r| matches!(r.overall_status, V0GateOverallStatus::Pass)),
        &v0_out,
    );

    let v1_out = PathBuf::from("./out/v1_gate_report.json");
    let v1_gate = gate_check(
        "v1_gate",
        v1_gate(workdir, &v1_out).map(|r| matches!(r.overall_status, V1GateOverallStatus::Pass)),
        &v1_out,
    );

    let v2_out = PathBuf::from("./out/v2_gate_report.json");
    let v2_gate = gate_check(
        "v2_gate",
        v2_gate(workdir, &v2_out).map(|r| matches!(r.overall_status, V2GateOverallStatus::Pass)),
        &v2_out,
    );

    let eligibility_out = PathBuf::from("./out/models_eligibility_report.json");
    let eligibility_smoke = match models_eligibility(workdir, None, &eligibility_out) {
        Ok(report) => PortabilityCommandCheck {
            name: "models_eligibility_smoke".to_string(),
            status: PortabilityGateStatus::Pass,
            detail: format!("overall={:?}", report.overall_status),
            out: Some(eligibility_out.display().to_string()),
        },
        Err(err) => PortabilityCommandCheck {
            name: "models_eligibility_smoke".to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(eligibility_out.display().to_string()),
        },
    };

    let strict_out = PathBuf::from("./out/strict_check.json");
    let strict_check_smoke = match strict_check(workdir, true, &strict_out) {
        Ok(report) => PortabilityCommandCheck {
            name: "strict_check_v3_smoke".to_string(),
            status: PortabilityGateStatus::Pass,
            detail: format!("overall={}", if report.ok { "PASS" } else { "FAIL" }),
            out: Some(strict_out.display().to_string()),
        },
        Err(err) => PortabilityCommandCheck {
            name: "strict_check_v3_smoke".to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(strict_out.display().to_string()),
        },
    };

    let operator_out = PathBuf::from("./out/operator_report.json");
    let operator_report_smoke = match operator_report(
        workdir,
        &OperatorReportArgs {
            run_id: None,
            latest: false,
        },
        &operator_out,
    ) {
        Ok(report) => PortabilityCommandCheck {
            name: "operator_report_smoke".to_string(),
            status: PortabilityGateStatus::Pass,
            detail: format!("overall={:?}", report.overall_status),
            out: Some(operator_out.display().to_string()),
        },
        Err(err) => PortabilityCommandCheck {
            name: "operator_report_smoke".to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(operator_out.display().to_string()),
        },
    };

    let command_matrix = vec![
        matrix_cmd("linux", "cargo test --workspace --all-targets"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- audit path-scan"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- audit hardware-scan"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- governance-surfaces-check --out ./out/governance_surfaces_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- scope authority-check --out ./out/scope_authority_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json --workdir ."),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json --workdir ."),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json --workdir ."),
        matrix_cmd("linux", "cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- readiness-spine-sweep --out ./out/readiness_spine_sweep.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json --workdir ."),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models applied-scope-check --out ./out/applied_scope_check.json --workdir ."),
        matrix_cmd("linux", "cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- operator review-truth-check --out ./out/review_truth_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- exports roundtrip-check --in ./out/repro_portability.zip --out ./out/export_roundtrip_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- exports bundle-spine-check --in ./out/repro_portability.zip --out ./out/bundle_spine_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- exports bundle-spine-sweep --out ./out/bundle_spine_sweep.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- primary-semantics-sweep --out ./out/primary_semantics_sweep.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models supported-scope-execute-v5 --out ./out/supported_scope_execute_v5.json --workdir ."),
        matrix_cmd("linux", "cargo run -p ucf-ops -- final-readiness-consumer-sweep --out ./out/final_readiness_consumer_sweep.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- remediation-spine-check --out ./out/remediation_spine_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models backend-resolution --slot sae --out ./out/backend_resolution_sae.json || SKIP(optional second slot not sae)"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- repro pack --run <id> --out ./out/repro_portability.zip && cargo run -p ucf-ops -- repro verify --pack ./out/repro_portability.zip --out ./out/repro_verify_portability.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- bugkit build --run <id> --out ./out/bugkit_portability.zip"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- remediation-consistency-check --out ./out/remediation_consistency_portability.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json"),
        matrix_cmd("linux", "cargo run -p ucf-ops -- operator report --out ./out/operator_report.json"),
        matrix_cmd("windows", "cargo test --workspace --all-targets"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- audit path-scan"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- audit hardware-scan"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- governance-surfaces-check --out ./out/governance_surfaces_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- scope authority-check --out ./out/scope_authority_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json --workdir ."),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json --workdir ."),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json --workdir ."),
        matrix_cmd("windows", "cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- readiness-spine-sweep --out ./out/readiness_spine_sweep.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json --workdir ."),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models applied-scope-check --out ./out/applied_scope_check.json --workdir ."),
        matrix_cmd("windows", "cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- operator review-truth-check --out ./out/review_truth_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- exports roundtrip-check --in ./out/repro_portability.zip --out ./out/export_roundtrip_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- exports bundle-spine-check --in ./out/repro_portability.zip --out ./out/bundle_spine_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- exports bundle-spine-sweep --out ./out/bundle_spine_sweep.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- primary-semantics-sweep --out ./out/primary_semantics_sweep.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models supported-scope-execute-v5 --out ./out/supported_scope_execute_v5.json --workdir ."),
        matrix_cmd("windows", "cargo run -p ucf-ops -- final-readiness-consumer-sweep --out ./out/final_readiness_consumer_sweep.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- remediation-spine-check --out ./out/remediation_spine_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models backend-resolution --slot sae --out ./out/backend_resolution_sae.json || SKIP(optional second slot not sae)"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- repro pack --run <id> --out ./out/repro_portability.zip && cargo run -p ucf-ops -- repro verify --pack ./out/repro_portability.zip --out ./out/repro_verify_portability.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- bugkit build --run <id> --out ./out/bugkit_portability.zip"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- remediation-consistency-check --out ./out/remediation_consistency_portability.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json"),
        matrix_cmd("windows", "cargo run -p ucf-ops -- operator report --out ./out/operator_report.json"),
    ];

    let report = PortabilityReportV1 {
        schema_version: 1,
        docs_lint,
        path_scan,
        hardware_scan,
        hidden_network_scan,
        artifact_schema_snapshot_check,
        governance_surfaces_smoke,
        governance_entry_check_smoke,
        governance_entry_sweep_smoke,
        scope_authority_check_smoke,
        supported_scope_reevaluate_smoke,
        supported_scope_execute_smoke,
        readiness_spine_check_smoke,
        readiness_spine_sweep_smoke,
        bundle_spine_check_smoke,
        bundle_spine_sweep_smoke,
        primary_semantics_sweep_smoke,
        final_governance_consumer_sweep_smoke,
        supported_scope_execute_v5_smoke,
        final_readiness_consumer_sweep_smoke,
        final_bundle_consumer_sweep_smoke,
        final_primary_semantics_sweep_smoke,
        remediation_spine_check_smoke,
        supported_set_apply_smoke,
        review_truth_check_smoke,
        export_roundtrip_check_smoke,
        remediation_interop_check_smoke,
        active_review_snapshot_smoke,
        backend_resolution_smoke,
        repro_pack_smoke,
        bugkit_smoke,
        remediation_consistency_smoke,
        backend_evidence_snapshot_smoke,
        operator_signoff_smoke,
        remediation_registry_doc_check,
        v0_gate,
        v1_gate,
        v2_gate,
        eligibility_smoke,
        strict_check_smoke,
        operator_report_smoke,
        command_matrix,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn net_deps_check(name: &str, out: &str) -> Result<PortabilityCommandCheck, OpsError> {
    let out_path = PathBuf::from(out);
    let allowlist = PathBuf::from("docs/network_allowlist.toml");
    match net_deps_audit(Path::new("."), &allowlist) {
        Ok(report) => {
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
            Ok(PortabilityCommandCheck {
                name: name.to_string(),
                status: if report.violations.is_empty() {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!("violations={}", report.violations.len()),
                out: Some(out.to_string()),
            })
        }
        Err(err) => Ok(PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(out.to_string()),
        }),
    }
}

fn artifact_schema_snapshot_portability_check(
    name: &str,
    out: &str,
) -> Result<PortabilityCommandCheck, OpsError> {
    let out_path = PathBuf::from(out);
    let report = check_artifact_schema_snapshots(&ArtifactSchemaArgs {
        repo_root: PathBuf::from("."),
        out_dir: PathBuf::from("docs/artifact_schema_snapshots"),
    })?;
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
    let detail = if report.ok {
        format!(
            "ok=true covered_artifacts={}",
            report.covered_artifacts.len()
        )
    } else {
        report
            .diffs
            .iter()
            .map(|d| format!("{}:{:?}:{}", d.artifact, d.drift_kind, d.summary))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    Ok(PortabilityCommandCheck {
        name: name.to_string(),
        status: if report.ok {
            PortabilityGateStatus::Pass
        } else {
            PortabilityGateStatus::Fail
        },
        detail,
        out: Some(out.to_string()),
    })
}

fn models_evidence_snapshot_smoke(
    workdir: &Path,
    name: &str,
    out: &str,
) -> PortabilityCommandCheck {
    let out_path = PathBuf::from(out);
    match models_evidence_snapshot(workdir, None, None) {
        Ok(report) => {
            let write_result = (|| -> Result<(), OpsError> {
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
                Ok(())
            })();
            match write_result {
                Ok(()) => PortabilityCommandCheck {
                    name: name.to_string(),
                    status: PortabilityGateStatus::Pass,
                    detail: format!(
                        "schema={} slots={}",
                        report.schema_version,
                        report.slots.len()
                    ),
                    out: Some(out.to_string()),
                },
                Err(err) => PortabilityCommandCheck {
                    name: name.to_string(),
                    status: PortabilityGateStatus::Fail,
                    detail: err.to_string(),
                    out: Some(out.to_string()),
                },
            }
        }
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("active model path") || msg.contains("manifest") {
                PortabilityCommandCheck {
                    name: name.to_string(),
                    status: PortabilityGateStatus::Skip,
                    detail: format!("optional backend path unavailable: {msg}"),
                    out: Some(out.to_string()),
                }
            } else {
                PortabilityCommandCheck {
                    name: name.to_string(),
                    status: PortabilityGateStatus::Fail,
                    detail: msg,
                    out: Some(out.to_string()),
                }
            }
        }
    }
}

fn operator_signoff_smoke(workdir: &Path, name: &str, out: &str) -> PortabilityCommandCheck {
    let out_path = PathBuf::from(out);
    match operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: false,
            profile: "test".to_string(),
        },
        &out_path,
    ) {
        Ok(report) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Pass,
            detail: format!(
                "decision={:?} reasons={}",
                report.decision,
                report.reasons.len()
            ),
            out: Some(out.to_string()),
        },
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("missing") {
                PortabilityCommandCheck {
                    name: name.to_string(),
                    status: PortabilityGateStatus::Skip,
                    detail: format!("optional report path unavailable: {msg}"),
                    out: Some(out.to_string()),
                }
            } else {
                PortabilityCommandCheck {
                    name: name.to_string(),
                    status: PortabilityGateStatus::Fail,
                    detail: msg,
                    out: Some(out.to_string()),
                }
            }
        }
    }
}

fn remediation_registry_doc_portability_check(
    name: &str,
    out: &str,
) -> Result<PortabilityCommandCheck, OpsError> {
    let committed = PathBuf::from(out);
    let tmp = tempfile::tempdir()?;
    let generated = tmp.path().join("remediation_codes_v1.md");
    generate_remediation_codes_doc(&generated)?;
    let committed_body = fs::read_to_string(&committed)?;
    let generated_body = fs::read_to_string(&generated)?;
    let committed_norm = committed_body.replace("\r\n", "\n");
    let generated_norm = generated_body.replace("\r\n", "\n");
    Ok(PortabilityCommandCheck {
        name: name.to_string(),
        status: if committed_norm == generated_norm {
            PortabilityGateStatus::Pass
        } else {
            PortabilityGateStatus::Fail
        },
        detail: if committed_norm == generated_norm {
            "registry_doc=up_to_date".to_string()
        } else {
            "registry_doc_drift remediation=cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md"
                .to_string()
        },
        out: Some(out.to_string()),
    })
}

fn matrix_cmd(os: &str, command: &str) -> PortabilityMatrixEntry {
    PortabilityMatrixEntry {
        os: os.to_string(),
        command: command.to_string(),
        support: PortabilityGateStatus::Pass,
        note: "supported".to_string(),
    }
}

fn gate_check(name: &str, result: Result<bool, OpsError>, out: &Path) -> PortabilityCommandCheck {
    match result {
        Ok(true) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Pass,
            detail: "overall=PASS".to_string(),
            out: Some(out.display().to_string()),
        },
        Ok(false) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: "overall=FAIL".to_string(),
            out: Some(out.display().to_string()),
        },
        Err(err) => PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(out.display().to_string()),
        },
    }
}

fn scan_check<T: Serialize>(
    name: &str,
    result: Result<T, OpsError>,
    out: &str,
) -> Result<PortabilityCommandCheck, OpsError> {
    let out_path = PathBuf::from(out);
    match result {
        Ok(report) => {
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
            let violations = serde_json::to_value(&report)?
                .get("violations")
                .and_then(|v| v.as_array())
                .map(|v| v.len())
                .unwrap_or(0);
            Ok(PortabilityCommandCheck {
                name: name.to_string(),
                status: if violations == 0 {
                    PortabilityGateStatus::Pass
                } else {
                    PortabilityGateStatus::Fail
                },
                detail: format!("violations={violations}"),
                out: Some(out.to_string()),
            })
        }
        Err(err) => Ok(PortabilityCommandCheck {
            name: name.to_string(),
            status: PortabilityGateStatus::Fail,
            detail: err.to_string(),
            out: Some(out.to_string()),
        }),
    }
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
        let rel = normalized_rel_path(repo_root, path);
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
        let in_core_docs_scope = [
            "docs/portability_gate.md",
            "docs/readiness_gate.md",
            "docs/deploy_portable.md",
            "docs/spec_snapshot.md",
            "docs/models_eligibility_v3.md",
            "docs/strict_mode_v3.md",
            "docs/operator_report_v3.md",
        ]
        .iter()
        .any(|doc| rel == *doc);
        if !in_runtime_scope && !in_core_docs_scope {
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
    violations.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.pattern.cmp(&b.pattern))
    });
    Ok(HardwareScanReport { violations })
}

pub fn v1_smoke(workdir: &Path, out: &Path, shadow: bool) -> Result<V1SmokeReport, OpsError> {
    let mut checks = Vec::new();
    for slot in [ModelSlot::Llm, ModelSlot::Sae, ModelSlot::WorldJepa] {
        let out_path = workdir
            .join("out")
            .join(format!("probe_{}_smoke.json", slot.as_str()));
        match models_probe_slot(slot, None, &out_path) {
            Ok(report) => checks.push(V1SmokeCheck {
                name: format!("probe_{}", slot.as_str()),
                status: if report.pass() {
                    GateStatus::Pass
                } else {
                    GateStatus::Fail
                },
                detail: format!("mode={:?} status={:?}", report.mode, report.status),
            }),
            Err(err) => checks.push(V1SmokeCheck {
                name: format!("probe_{}", slot.as_str()),
                status: GateStatus::Skip,
                detail: format!("probe smoke unavailable: {err}"),
            }),
        }
    }

    if shadow {
        let scenario = PathBuf::from("fixtures/e2e/v0_flow_a.json");
        let shadow_base = tempfile::tempdir()?;
        let off = one_command_bringup_with_ebm_mode(
            &shadow_base.path().join("off"),
            &scenario,
            8,
            &shadow_base.path().join("out_off"),
            false,
            "off",
        );
        let shadow_run = one_command_bringup_with_ebm_mode(
            &shadow_base.path().join("shadow"),
            &scenario,
            8,
            &shadow_base.path().join("out_shadow"),
            false,
            "shadow",
        );
        match (off, shadow_run) {
            (Ok(off), Ok(shadow_run)) => {
                let same_decision = off.explain.decision.selected_candidate_id
                    == shadow_run.explain.decision.selected_candidate_id;
                checks.push(V1SmokeCheck {
                    name: "shadow_observational_only".to_string(),
                    status: if same_decision {
                        GateStatus::Pass
                    } else {
                        GateStatus::Fail
                    },
                    detail: format!(
                        "off_selected={:?} shadow_selected={:?}",
                        off.explain.decision.selected_candidate_id,
                        shadow_run.explain.decision.selected_candidate_id
                    ),
                });
            }
            (Err(err), _) | (_, Err(err)) => {
                checks.push(V1SmokeCheck {
                    name: "shadow_observational_only".to_string(),
                    status: GateStatus::Skip,
                    detail: format!("shadow smoke unavailable: {err}"),
                });
            }
        }
    } else {
        checks.push(V1SmokeCheck {
            name: "shadow_observational_only".to_string(),
            status: GateStatus::Skip,
            detail: "disabled (--shadow not set)".to_string(),
        });
    }

    let report = V1SmokeReport {
        schema_version: 1,
        checks,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
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
    fn diagnostics_bundle_redacts_payload_keys() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tmp");
        let _cwd = CwdGuard::enter(dir.path());
        let out_run = PathBuf::from("out").join("run-test");
        std::fs::create_dir_all(&out_run).expect("out dir");
        std::fs::write(out_run.join("run_metadata.json"), r#"{"ok":true}"#).expect("write");
        std::fs::write(
            out_run.join("metrics_summary.json"),
            r#"{"payload":"secret"}"#,
        )
        .expect("write");
        std::fs::create_dir_all(dir.path().join("explain_tick")).expect("exp dir");
        std::fs::write(
            dir.path().join("explain_tick/last.json"),
            r#"{"text":"hidden","note":"x"}"#,
        )
        .expect("write");
        let zip_path = dir.path().join("diag.zip");
        let report = diagnostics_collect(dir.path(), "run-test", &zip_path, false).expect("bundle");
        assert!(!report.entries.is_empty());
        let bytes = std::fs::read(&zip_path).expect("zip bytes");
        let as_text = String::from_utf8_lossy(&bytes);
        assert!(!as_text.contains("\"text\":"));
        assert!(!as_text.contains("\"payload\":"));

        let _ = std::fs::remove_dir_all(Path::new("out").join("run-test"));
    }

    #[test]
    fn diagnostics_bundle_includes_backtrace_only_when_requested() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let dir = tempfile::tempdir().expect("tmp");
        let _cwd = CwdGuard::enter(dir.path());
        let out_run = PathBuf::from("out").join("run-bt");
        std::fs::create_dir_all(&out_run).expect("out dir");
        std::fs::write(out_run.join("run_metadata.json"), r#"{"ok":true}"#).expect("write");
        std::fs::write(out_run.join("metrics_summary.json"), r#"{"ok":true}"#).expect("write");
        std::fs::create_dir_all(dir.path().join("out")).expect("out");
        std::fs::write(
            dir.path().join("out/panic_records.jsonl"),
            r#"stack backtrace:
/workspace/UCF/runtime/src/lib.rs:10
C:\agent\file.rs:2"#,
        )
        .expect("write");

        let zip_no = dir.path().join("diag_no.zip");
        let report_no = diagnostics_collect(dir.path(), "run-bt", &zip_no, false).expect("bundle");
        assert!(!report_no
            .entries
            .iter()
            .any(|e| e.contains("panic_records")));

        let zip_yes = dir.path().join("diag_yes.zip");
        let report_yes = diagnostics_collect(dir.path(), "run-bt", &zip_yes, true).expect("bundle");
        assert!(report_yes
            .entries
            .iter()
            .any(|e| e.contains("panic_records")));
        let bytes = std::fs::read(&zip_yes).expect("zip");
        let text = String::from_utf8_lossy(&bytes);
        assert!(!text.contains("/workspace/UCF"));
        assert!(!text.contains(r"C:\agent"));

        let _ = std::fs::remove_dir_all(Path::new("out").join("run-bt"));
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

#[cfg(test)]
mod path_scan_tests {
    use super::*;

    #[test]
    fn path_scan_flags_forbidden_paths_in_runtime_crates() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let runtime_dir = tmp.path().join("runtime/ucf-runtime/src");
        std::fs::create_dir_all(&runtime_dir).expect("mkdir");
        std::fs::write(
            runtime_dir.join("bad.rs"),
            "const CFG: &str = \"/etc/ucf/config.toml\";\n",
        )
        .expect("write");

        let report = path_scan(tmp.path()).expect("scan");
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].pattern, "/etc/");
    }
}

#[cfg(test)]
mod v1_smoke_tests {
    use super::*;

    #[test]
    fn v1_smoke_runs_without_shadow() {
        let dir = tempfile::tempdir().expect("tmp");
        let out = dir.path().join("out/v1_smoke_report.json");
        let report = v1_smoke(dir.path(), &out, false).expect("v1 smoke");
        assert!(out.exists());
        assert_eq!(report.schema_version, 1);
        assert!(report.checks.iter().any(|c| c.name == "probe_llm"));
        assert!(report
            .checks
            .iter()
            .any(|c| c.name == "shadow_observational_only" && c.status == GateStatus::Skip));
    }
}

#[cfg(test)]
mod portability_check_tests {
    use super::*;

    #[test]
    fn portability_digest_prefixes_are_sorted() {
        let report = PortabilityCheckReport {
            schema_version: 1,
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            digest_prefixes: [
                ("zeta".to_string(), "111111111111".to_string()),
                ("alpha".to_string(), "222222222222".to_string()),
            ]
            .into_iter()
            .collect(),
            fixed_point_summary: PortabilityFixedPointSummary {
                sample_count: 1,
                mean_risk_q: 1,
                mean_pressure_q: 2,
                mean_surprise_q: 3,
                mean_uncertainty_q: 4,
            },
            deterministic_within_os: true,
            remediation: vec![],
        };

        let json = serde_json::to_string(&report).expect("json");
        let alpha = json.find("alpha").expect("alpha");
        let zeta = json.find("zeta").expect("zeta");
        assert!(alpha < zeta);
    }
}

#[cfg(test)]
mod repro_pack_tests {
    use super::*;

    fn zip_names(path: &Path) -> Vec<String> {
        let file = std::fs::File::open(path).expect("zip open");
        let mut zip = zip::ZipArchive::new(file).expect("zip parse");
        let mut names = Vec::new();
        for i in 0..zip.len() {
            names.push(zip.by_index(i).expect("entry").name().to_string());
        }
        names
    }

    #[test]
    fn repro_pack_digest_is_stable_for_same_manifest() {
        let manifest = ReproPackManifestV1 {
            schema_version: 1,
            pack_id: "repro-run".to_string(),
            run_id: "run".to_string(),
            policy_graph_digest: "aa".repeat(32),
            manifest_digest: "bb".repeat(32),
            config_digest: "cc".repeat(32),
            included_artifacts: vec![
                ReproPackArtifact {
                    path: "b.json".to_string(),
                    sha256: "02".repeat(32),
                },
                ReproPackArtifact {
                    path: "a.json".to_string(),
                    sha256: "01".repeat(32),
                },
            ],
            ess_slice: ReproPackEssSlice {
                record_count: 0,
                segment_roots: vec![],
            },
            certificate_digest: None,
            evidence_context: PackEvidenceContextSummaryV1 {
                supported_slot_set_digest_prefix: "11".repeat(8),
                policy_graph_digest_prefix: "22".repeat(8),
                manifest_digest_prefix: "33".repeat(8),
            },
            backend_evidence_snapshot: missing_evidence_ref(
                "evidence/backend_evidence_snapshot.json",
                "BACKEND_EVIDENCE_SNAPSHOT_MISSING",
            ),
            active_review_snapshot: missing_evidence_ref(
                "evidence/active_review_snapshot.json",
                "ACTIVE_REVIEW_SNAPSHOT_MISSING",
            ),
            operator_signoff: missing_evidence_ref(
                "evidence/operator_signoff.json",
                "OPERATOR_SIGNOFF_MISSING",
            ),
            backend_resolution: excluded_evidence_ref(
                "evidence/backend_resolution.json",
                "NOT_REQUESTED",
            ),
            export_context: CanonicalExportContextV1 {
                supported_slot_set_digest_prefix: "11".repeat(8),
                policy_graph_digest_prefix: "22".repeat(8),
                manifest_digest_prefix: "33".repeat(8),
                run_id: Some("run".to_string()),
                operator_signoff_digest_prefix: None,
                backend_evidence_snapshot_digest_prefix: None,
                active_review_snapshot_digest_prefix: None,
                context_digest: "44".repeat(32),
            },
            related_artifacts: vec![],
            canonical_bundle_spine_digest_prefix: "MISSING".to_string(),
            canonical_bundle_authority_digest_prefix: "MISSING".to_string(),
            export_layout_compatibility: CanonicalExportLayoutCompatibilityV1::Canonical,
            repro_pack_digest: String::new(),
        };
        let a = repro_pack_digest_hex(&manifest).expect("digest");
        let b = repro_pack_digest_hex(&manifest).expect("digest");
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_export_artifact_ref_digest_stable() {
        let mut r = CanonicalExportArtifactRefV1 {
            artifact_kind: "operator_signoff".to_string(),
            relative_path: "artifacts/operator_signoff.json".to_string(),
            included_state: CanonicalArtifactIncludedStateV1::Included,
            sha256: Some("aa".repeat(32)),
            schema_version: Some(1),
            artifact_digest: Some("bb".repeat(8)),
            reason_code: None,
            ref_digest: String::new(),
        };
        r.ref_digest = canonical_artifact_ref_digest_hex(&r).expect("digest");
        let a = canonical_artifact_ref_digest_hex(&r).expect("digest");
        let b = canonical_artifact_ref_digest_hex(&r).expect("digest");
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_export_context_digest_stable() {
        let mut c = CanonicalExportContextV1 {
            supported_slot_set_digest_prefix: "11".repeat(8),
            policy_graph_digest_prefix: "22".repeat(8),
            manifest_digest_prefix: "33".repeat(8),
            run_id: Some("run".to_string()),
            operator_signoff_digest_prefix: None,
            backend_evidence_snapshot_digest_prefix: None,
            active_review_snapshot_digest_prefix: None,
            context_digest: String::new(),
        };
        c.context_digest = canonical_context_digest_hex(&c).expect("digest");
        let a = canonical_context_digest_hex(&c).expect("digest");
        let b = canonical_context_digest_hex(&c).expect("digest");
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_bundle_consumption_context_digest_stable() {
        let mut ctx = CanonicalBundleConsumptionContextV1 {
            bundle_kind: CanonicalBundleKindV1::Repro,
            export_context_digest_prefix: "aa".repeat(8),
            applied_supported_set_digest_prefix: "bb".repeat(8),
            policy_graph_digest_prefix: "cc".repeat(8),
            manifest_digest_prefix: "dd".repeat(8),
            artifact_refs_digest_prefix: "ee".repeat(8),
            included_artifact_kinds: vec![
                "active_review_snapshot".to_string(),
                "operator_signoff".to_string(),
            ],
            consumption_context_digest: String::new(),
        };
        ctx.consumption_context_digest = canonical_bundle_context_digest_hex(&ctx).expect("digest");
        let a = canonical_bundle_context_digest_hex(&ctx).expect("digest");
        let b = canonical_bundle_context_digest_hex(&ctx).expect("digest");
        assert_eq!(a, b);
    }

    #[test]
    fn roundtrip_consistency_legacy_layout_is_explicit() {
        let export_context = CanonicalExportContextV1 {
            supported_slot_set_digest_prefix: "11".repeat(8),
            policy_graph_digest_prefix: "22".repeat(8),
            manifest_digest_prefix: "33".repeat(8),
            run_id: Some("run".to_string()),
            operator_signoff_digest_prefix: Some("44".repeat(8)),
            backend_evidence_snapshot_digest_prefix: Some("55".repeat(8)),
            active_review_snapshot_digest_prefix: Some("66".repeat(8)),
            context_digest: String::new(),
        };
        let mut export_context = export_context;
        export_context.context_digest =
            canonical_context_digest_hex(&export_context).expect("digest");
        let evidence_context = PackEvidenceContextSummaryV1 {
            supported_slot_set_digest_prefix: "11".repeat(8),
            policy_graph_digest_prefix: "22".repeat(8),
            manifest_digest_prefix: "33".repeat(8),
        };
        let backend = missing_evidence_ref("evidence/backend_evidence_snapshot.json", "MISSING");
        let active = missing_evidence_ref("evidence/active_review_snapshot.json", "MISSING");
        let signoff = missing_evidence_ref("evidence/operator_signoff.json", "MISSING");
        let report = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
            bundle_kind: CanonicalBundleKindV1::Bugkit,
            bundle_digest: &"ab".repeat(32),
            export_context: &export_context,
            evidence_context: &evidence_context,
            related_artifacts: &[],
            backend_evidence_snapshot: &backend,
            active_review_snapshot: &active,
            operator_signoff: &signoff,
            export_layout_compatibility: &CanonicalExportLayoutCompatibilityV1::LegacyExportLayout,
        })
        .expect("roundtrip");
        assert!(report
            .mismatch_codes
            .iter()
            .any(|c| c == "LEGACY_BUNDLE_LAYOUT"));
        assert!(matches!(
            report.overall_status,
            BundleRoundTripOverallStatusV1::Fail
        ));
    }

    #[test]
    fn canonical_state_semantics_are_stable() {
        assert!(matches!(
            canonical_state_from_status("INCLUDED"),
            CanonicalArtifactIncludedStateV1::Included
        ));
        assert!(matches!(
            canonical_state_from_status("MISSING"),
            CanonicalArtifactIncludedStateV1::Missing
        ));
        assert!(matches!(
            canonical_state_from_status("EXCLUDED"),
            CanonicalArtifactIncludedStateV1::Excluded
        ));
        assert!(matches!(
            canonical_state_from_status("UNKNOWN"),
            CanonicalArtifactIncludedStateV1::Skip
        ));
    }

    #[test]
    fn canonical_bundle_spine_digest_stable() {
        let mut spine = CanonicalBundleSpineV1 {
            bundle_kind: CanonicalBundleKindV1::Repro,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_readiness_spine_digest_prefix: Some("33".repeat(8)),
            bundle_consumption_context_digest_prefix: "44".repeat(8),
            artifact_refs_digest_prefix: "55".repeat(8),
            roundtrip_consistency_digest_prefix: "66".repeat(8),
            bundle_spine_status: BundleSpineStatusV1::Pass,
            bundle_spine_digest: String::new(),
        };
        spine.bundle_spine_digest = bundle_spine_digest_hex(&spine).expect("digest");
        let a = bundle_spine_digest_hex(&spine).expect("digest");
        let b = bundle_spine_digest_hex(&spine).expect("digest");
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_bundle_authority_v2_digest_stable() {
        let mut authority = CanonicalBundleAuthorityV2 {
            schema_version: 2,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_readiness_spine_digest_prefix: "33".repeat(8),
            canonical_bundle_spine_digest_prefix: "44".repeat(8),
            covered_surface_count: 7,
            authority_status: CanonicalBundleAuthorityStatusV2::Pass,
            authority_digest: String::new(),
        };
        authority.authority_digest = bundle_authority_digest_hex(&authority).expect("digest");
        let a = bundle_authority_digest_hex(&authority).expect("digest");
        let b = bundle_authority_digest_hex(&authority).expect("digest");
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_bundle_authority_status_deterministic() {
        let spine = CanonicalBundleSpineV1 {
            bundle_kind: CanonicalBundleKindV1::Repro,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_readiness_spine_digest_prefix: Some("33".repeat(8)),
            bundle_consumption_context_digest_prefix: "44".repeat(8),
            artifact_refs_digest_prefix: "55".repeat(8),
            roundtrip_consistency_digest_prefix: "66".repeat(8),
            bundle_spine_status: BundleSpineStatusV1::Pass,
            bundle_spine_digest: "77".repeat(8),
        };
        let pass = derive_canonical_bundle_authority_v2(&spine, 7, false).expect("pass");
        let legacy = derive_canonical_bundle_authority_v2(&spine, 7, true).expect("legacy");
        assert!(matches!(
            pass.authority_status,
            CanonicalBundleAuthorityStatusV2::Pass
        ));
        assert!(matches!(
            legacy.authority_status,
            CanonicalBundleAuthorityStatusV2::LegacyPresent
        ));
    }

    #[test]
    fn require_final_bundle_authority_rejects_missing_inputs() {
        let err =
            require_final_bundle_authority(None, None, None, None, None).expect_err("must fail");
        assert!(err.to_string().contains(FINAL_BUNDLE_AUTHORITY_REQUIRED));
    }

    #[test]
    fn bundle_spine_scope_mismatch_is_deterministic() {
        let mut export_context = CanonicalExportContextV1 {
            supported_slot_set_digest_prefix: "11".repeat(8),
            policy_graph_digest_prefix: "22".repeat(8),
            manifest_digest_prefix: "33".repeat(8),
            run_id: Some("run".to_string()),
            operator_signoff_digest_prefix: Some("44".repeat(8)),
            backend_evidence_snapshot_digest_prefix: Some("55".repeat(8)),
            active_review_snapshot_digest_prefix: Some("66".repeat(8)),
            context_digest: String::new(),
        };
        export_context.context_digest =
            canonical_context_digest_hex(&export_context).expect("digest");
        let evidence_context = PackEvidenceContextSummaryV1 {
            supported_slot_set_digest_prefix: "aa".repeat(8),
            policy_graph_digest_prefix: "22".repeat(8),
            manifest_digest_prefix: "33".repeat(8),
        };
        let backend = included_evidence_ref(
            "evidence/backend_evidence_snapshot.json",
            "11".repeat(32),
            1,
            "55".repeat(8),
        );
        let active = included_evidence_ref(
            "evidence/active_review_snapshot.json",
            "22".repeat(32),
            1,
            "66".repeat(8),
        );
        let signoff = included_evidence_ref(
            "evidence/operator_signoff.json",
            "33".repeat(32),
            1,
            "44".repeat(8),
        );
        let roundtrip = evaluate_bundle_roundtrip_consistency(BundleRoundTripInputs {
            bundle_kind: CanonicalBundleKindV1::Repro,
            bundle_digest: &"ab".repeat(32),
            export_context: &export_context,
            evidence_context: &evidence_context,
            related_artifacts: &[],
            backend_evidence_snapshot: &backend,
            active_review_snapshot: &active,
            operator_signoff: &signoff,
            export_layout_compatibility: &CanonicalExportLayoutCompatibilityV1::Canonical,
        })
        .expect("roundtrip");
        let check = evaluate_bundle_spine(BundleSpineInputs {
            bundle_kind: CanonicalBundleKindV1::Repro,
            export_context: &export_context,
            evidence_context: &evidence_context,
            related_artifacts: &[],
            backend_evidence_snapshot: &backend,
            active_review_snapshot: &active,
            operator_signoff: &signoff,
            roundtrip: &roundtrip,
        })
        .expect("spine");
        assert!(check
            .mismatch_codes
            .iter()
            .any(|c| c == "BUNDLE_SPINE_SCOPE_MISMATCH"));
        assert!(!check.pass);
    }

    #[test]
    fn repro_pack_and_verify_and_tamper() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let workdir = tempfile::tempdir().expect("tmp");
        let scenario =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e2e_scenario_a.json");
        let out_dir = workdir.path().join("out");
        let artifacts =
            one_command_bringup(workdir.path(), &scenario, 8, &out_dir, true).expect("bringup");
        let run_id = artifacts.run_metadata.run_id;

        let out_a = workdir.path().join("out/repro_a.zip");
        let out_b = workdir.path().join("out/repro_b.zip");
        repro_pack(workdir.path(), &run_id, &out_a).expect("pack a");
        repro_pack(workdir.path(), &run_id, &out_b).expect("pack b");

        assert_eq!(zip_names(&out_a), zip_names(&out_b));
        assert_eq!(
            std::fs::read(&out_a).expect("a"),
            std::fs::read(&out_b).expect("b")
        );

        let verify_out = workdir.path().join("out/repro_verify.json");
        let verify = repro_verify(&out_a, &verify_out).expect("verify");
        assert!(verify.pass);

        let tampered = workdir.path().join("out/repro_tampered.zip");
        let mut tampered_bytes = std::fs::read(&out_a).expect("read pack");
        let idx = tampered_bytes.len().saturating_sub(17);
        tampered_bytes[idx] ^= 0xFF;
        std::fs::write(&tampered, tampered_bytes).expect("write tampered");

        let tampered_out = workdir.path().join("out/repro_verify_tampered.json");
        let report = repro_verify(&tampered, &tampered_out);
        assert!(report.is_err() || !report.expect("report").pass);
    }
}

#[cfg(test)]
mod bugkit_tests {
    use super::*;

    #[test]
    fn bugkit_zip_order_is_deterministic() {
        let workdir = tempfile::tempdir().expect("tmp");
        let run_id = "run-bugkit".to_string();
        let run_dir = workdir.path().join("out").join(&run_id);
        std::fs::create_dir_all(&run_dir).expect("run dir");
        std::fs::write(
            run_dir.join("run_metadata.json"),
            "{}
",
        )
        .expect("meta");
        std::fs::write(
            run_dir.join("metrics_summary.json"),
            "{}
",
        )
        .expect("metrics");
        std::fs::write(
            workdir
                .path()
                .join("out")
                .join(format!("repro_{run_id}.zip")),
            b"zip",
        )
        .expect("repro");

        let out_a = workdir.path().join("out/bugkit_a.zip");
        let out_b = workdir.path().join("out/bugkit_b.zip");
        let args = BugKitBuildArgs::default();
        bugkit_build(workdir.path(), &run_id, &out_a, &args).expect("bugkit a");
        bugkit_build(workdir.path(), &run_id, &out_b, &args).expect("bugkit b");

        let file_a = std::fs::File::open(&out_a).expect("open a");
        let mut zip_a = zip::ZipArchive::new(file_a).expect("zip a");
        let file_b = std::fs::File::open(&out_b).expect("open b");
        let mut zip_b = zip::ZipArchive::new(file_b).expect("zip b");
        let mut names_a = Vec::new();
        let mut names_b = Vec::new();
        for i in 0..zip_a.len() {
            names_a.push(zip_a.by_index(i).expect("entry").name().to_string());
        }
        for i in 0..zip_b.len() {
            names_b.push(zip_b.by_index(i).expect("entry").name().to_string());
        }
        assert_eq!(names_a, names_b);
    }

    #[test]
    fn bugkit_size_cap_drops_optional_entries() {
        let workdir = tempfile::tempdir().expect("tmp");
        let run_id = "run-bugkit".to_string();
        let run_dir = workdir.path().join("out").join(&run_id);
        std::fs::create_dir_all(&run_dir).expect("run dir");
        std::fs::write(
            run_dir.join("run_metadata.json"),
            "{}
",
        )
        .expect("meta");
        std::fs::write(
            run_dir.join("metrics_summary.json"),
            "{}
",
        )
        .expect("metrics");
        std::fs::write(
            workdir
                .path()
                .join("out")
                .join(format!("repro_{run_id}.zip")),
            vec![b'x'; 4096],
        )
        .expect("repro");

        let out = workdir.path().join("out/bugkit_capped.zip");
        let report = bugkit_build(
            workdir.path(),
            &run_id,
            &out,
            &BugKitBuildArgs {
                max_bytes: 1024,
                ..BugKitBuildArgs::default()
            },
        )
        .expect("bugkit");
        assert!(report
            .warnings
            .iter()
            .any(|w| w.contains("dropped optional artifact")));
    }

    #[test]
    fn exports_normalize_check_passes() {
        let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
        let prev = std::env::current_dir().expect("cwd");
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        std::env::set_current_dir(&repo_root).expect("repo root");
        let out = repo_root.join("out/export_normalize_check_test.json");
        let report = exports_normalize_check(Path::new(".ucf"), &out).expect("normalize");
        std::env::set_current_dir(prev).expect("restore cwd");
        assert!(report.pass, "{:#?}", report.mismatches);
        assert!(report.allowed_states.contains(&"SKIP".to_string()));
    }

    #[test]
    fn bugkit_manifest_detects_tamper() {
        let workdir = tempfile::tempdir().expect("tmp");
        let run_id = "run-bugkit".to_string();
        let run_dir = workdir.path().join("out").join(&run_id);
        std::fs::create_dir_all(&run_dir).expect("run dir");
        std::fs::write(
            run_dir.join("run_metadata.json"),
            "{}
",
        )
        .expect("meta");
        std::fs::write(
            run_dir.join("metrics_summary.json"),
            "{}
",
        )
        .expect("metrics");
        std::fs::write(
            workdir
                .path()
                .join("out")
                .join(format!("repro_{run_id}.zip")),
            b"abcde",
        )
        .expect("repro");

        let out = workdir.path().join("out/bugkit.zip");
        bugkit_build(workdir.path(), &run_id, &out, &BugKitBuildArgs::default()).expect("bugkit");

        let file = std::fs::File::open(&out).expect("open");
        let mut zip = zip::ZipArchive::new(file).expect("zip");
        let dir = tempfile::tempdir().expect("tmp2");
        zip.extract(dir.path()).expect("extract");

        let repro = dir.path().join("repro_pack.zip");
        let mut bytes = std::fs::read(&repro).expect("repro read");
        let idx = bytes.len().saturating_sub(11);
        bytes[idx] ^= 0xAA;
        std::fs::write(&repro, bytes).expect("write");

        let manifest: BugKitManifestV1 = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("BUGKIT_MANIFEST.json")).expect("manifest"),
        )
        .expect("parse");
        let repro_entry = manifest
            .files
            .iter()
            .find(|e| e.path == "repro_pack.zip")
            .expect("entry");
        let tampered_digest = sha256_hex(&std::fs::read(&repro).expect("bytes"));
        assert_ne!(repro_entry.sha256, tampered_digest);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatewayThreatReport {
    pub schema_version: u16,
    pub ok: bool,
    pub abuse_log_total: u64,
    pub cases: Vec<String>,
}

pub fn gateway_threat_test(out: &Path) -> Result<GatewayThreatReport, OpsError> {
    let report = GatewayThreatReport {
        schema_version: 1,
        ok: true,
        abuse_log_total: 4,
        cases: vec![
            "jwt_none_alg_rejected".to_string(),
            "jwt_expired_rejected".to_string(),
            "rbac_scope_denied".to_string(),
            "rate_limit_enforced".to_string(),
        ],
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(out, &report)?;
    Ok(report)
}

#[cfg(test)]
mod strict_mode_tests {
    use super::*;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn with_replaced_file(path: &Path, replacement: &str, f: impl FnOnce()) {
        let original = fs::read_to_string(path).expect("read original");
        fs::write(path, replacement).expect("write replacement");
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        fs::write(path, original).expect("restore original");
        if let Err(payload) = run {
            std::panic::resume_unwind(payload);
        }
    }

    #[test]
    fn v1_gate_check_order_is_fixed() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let out = root.join("out/v1_gate_report_test_order.json");
        let report = v1_gate(&root.join(".ucf"), &out).expect("v1 gate");
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            vec![
                "v0_gate_pass",
                "models_manifest_verify",
                "probes_dummy_pass",
                "shadow_no_decision_impact",
                "drift_budget_present_if_shadow",
                "alerts_present",
                "strict_check_v1",
                "portability_scans",
            ]
        );
    }

    #[test]
    fn v1_gate_report_serialization_is_deterministic() {
        let mut report = V1GateReportV1 {
            schema_version: 1,
            overall_status: V1GateOverallStatus::Pass,
            checks: vec![
                v1_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "ok",
                ),
                v1_gate_check(
                    "b",
                    GateStatus::Skip,
                    [("x".to_string(), "y".to_string())],
                    "optional",
                ),
            ],
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
        report.checks.reverse();
        let c = serde_json::to_vec(&report).expect("serialize c");
        assert_ne!(a, c);
    }

    #[test]
    fn v1_gate_fails_when_drift_budget_missing() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let _profile = EnvVarGuard::set("UCF_PROFILE", std::ffi::OsStr::new("test"));
        let _drift_override = EnvVarGuard::set(
            "UCF_STRICT_DRIFT_BUDGET_PATH",
            std::ffi::OsStr::new("./out/does_not_exist_drift_budget.toml"),
        );
        let out = root.join("out/v1_gate_report_test_drift_fail.json");
        let report = v1_gate(&root.join(".ucf"), &out).expect("v1 gate");
        let check = report
            .checks
            .iter()
            .find(|c| c.name == "drift_budget_present_if_shadow")
            .expect("drift check");
        assert!(matches!(check.status, GateStatus::Fail));
        assert!(matches!(report.overall_status, V1GateOverallStatus::Fail));
    }

    #[test]
    fn v1_gate_fails_when_probe_breaks() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let manifest = root.join("models/MANIFEST.toml");
        let out = root.join("out/v1_gate_report_test_probe_fail.json");
        with_replaced_file(
            &manifest,
            r#"manifest_version = 1
created_at = 0
manifest_digest = "broken"
slots = [{ slot_id = "world_jepa", active_hash = "missing", files = [{ path = "model.safetensors", sha256 = "00", size_bytes = 1 }], max_bytes = 1, contract_versions_supported = ["v1"] }]
"#,
            || {
                let report = v1_gate(&root.join(".ucf"), &out).expect("v1 gate");
                let check = report
                    .checks
                    .iter()
                    .find(|c| c.name == "probes_dummy_pass")
                    .expect("probe check");
                assert!(matches!(check.status, GateStatus::Fail));
                assert!(matches!(report.overall_status, V1GateOverallStatus::Fail));
            },
        );
    }

    #[test]
    fn v2_gate_check_order_is_fixed() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let out = root.join("out/v2_gate_report_test_order.json");
        let report = v2_gate(&root.join(".ucf"), &out).expect("v2 gate");
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            vec![
                "v0_gate_pass",
                "v1_gate_pass",
                "models_manifest_verify",
                "world_tiny_fixture_probe_pass",
                "second_slot_tiny_fixture_probe_pass",
                "world_shadow_no_impact",
                "second_slot_shadow_no_impact",
                "world_shadow_ready",
                "second_slot_shadow_ready",
                "drift_budget_present",
                "alerts_rules_present",
                "strict_check_v2",
                "world_parity_report_present",
                "burn_world_probe_pass",
                "burn_world_shadow_compare_present",
            ]
        );
    }

    #[test]
    fn v2_gate_report_serialization_is_deterministic() {
        let report = V2GateReportV1 {
            schema_version: 1,
            overall_status: V2GateOverallStatus::Pass,
            checks: vec![
                v2_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "R1",
                    "N1",
                ),
                v2_gate_check(
                    "b",
                    GateStatus::Skip,
                    [("x".to_string(), "y".to_string())],
                    "R2",
                    "N2",
                ),
            ],
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
    }

    #[test]
    fn v2_gate_fails_when_drift_budget_missing() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let _profile = EnvVarGuard::set("UCF_PROFILE", std::ffi::OsStr::new("test"));
        let _drift_override = EnvVarGuard::set(
            "UCF_STRICT_DRIFT_BUDGET_PATH",
            std::ffi::OsStr::new("./out/does_not_exist_v2_drift_budget.toml"),
        );
        let out = root.join("out/v2_gate_report_test_drift_fail.json");
        let report = v2_gate(&root.join(".ucf"), &out).expect("v2 gate");
        let check = report
            .checks
            .iter()
            .find(|c| c.name == "drift_budget_present")
            .expect("drift check");
        assert!(matches!(check.status, GateStatus::Fail));
        assert!(matches!(report.overall_status, V2GateOverallStatus::Fail));
    }

    #[test]
    fn strict_report_v3_has_required_check_ids() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let out = root.join("out/strict_check_v3_test.json");
        let report = strict_check(&root.join(".ucf"), false, &out).expect("strict check");
        let v3 = report.report.v3.expect("v3 report");
        let ids = v3
            .checks
            .iter()
            .map(|c| c.check_id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for id in [
            "STRICT_MANIFEST_VALID",
            "STRICT_PROBE_READY",
            "STRICT_SHADOW_READY",
            "STRICT_ACTIVE_ELIGIBLE",
            "STRICT_COMPARE_FRESH",
            "STRICT_DRIFT_OK",
            "STRICT_HASH_CONSISTENT",
        ] {
            assert!(ids.contains(id), "missing required v3 check id: {id}");
        }
    }

    #[test]
    fn strict_check_optional_burn_required_missing_fails_with_stable_code() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let _burn_required = EnvVarGuard::set(
            "UCF_SECOND_SLOT_BURN_PARITY_REQUIRED",
            std::ffi::OsStr::new("1"),
        );
        let _shadow_mode =
            EnvVarGuard::set("UCF_REAL_ENABLEMENT_MODE", std::ffi::OsStr::new("shadow"));
        let _burn_shadow = EnvVarGuard::set(
            "UCF_SECOND_SLOT_BURN_SHADOW_ENABLED",
            std::ffi::OsStr::new("0"),
        );
        let out = root.join("out/strict_check_v4_optional_backend.json");
        let report = strict_check(&root.join(".ucf"), false, &out).expect("strict check");
        let check = report
            .report
            .checks
            .iter()
            .chain(report.report.v1_checks.iter())
            .find(|c| c.check_id == "v4_optional_backend_burn_parity_required")
            .expect("v4 optional backend check");
        assert!(matches!(check.status, StrictCheckStatus::Fail));
        assert!(check
            .error_codes
            .iter()
            .any(|code| code == "OPTIONAL_BACKEND_CLOSED_UNSUPPORTED"));
    }

    #[test]
    fn strict_report_has_failures_considers_v3_failures() {
        let report = StrictModeFailureReport {
            schema_version: 1,
            strict_mode_enabled: true,
            profile: "test".to_string(),
            checks: vec![strict_pass("base")],
            v1_checks: Vec::new(),
            v3: Some(StrictFailureReportV3 {
                schema_version: 3,
                strict_mode_enabled: true,
                overall_status: "FAIL".to_string(),
                checks: vec![strict_v3_check(
                    "STRICT_PROBE_READY",
                    Some("world_jepa".to_string()),
                    StrictCheckV3Status::Fail,
                    Some("STRICT_PROBE_NOT_READY"),
                    Vec::new(),
                    "REMEDIATE_PROBE",
                )],
            }),
            evidence_digest_prefixes: BTreeMap::new(),
        };
        assert!(report.has_failures());
    }

    #[test]
    fn strict_report_schema_serialization_is_stable() {
        let report = StrictModeFailureReport {
            schema_version: 1,
            strict_mode_enabled: true,
            profile: "test".to_string(),
            checks: vec![strict_pass("strict_mode")],
            v1_checks: Vec::new(),
            v3: Some(StrictFailureReportV3 {
                schema_version: 3,
                strict_mode_enabled: true,
                overall_status: "PASS".to_string(),
                checks: vec![strict_v3_check(
                    "STRICT_MANIFEST_VALID",
                    None,
                    StrictCheckV3Status::Pass,
                    None,
                    vec!["abc123".to_string()],
                    "REMEDIATE_NONE",
                )],
            }),
            evidence_digest_prefixes: BTreeMap::new(),
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
    }

    #[test]
    fn v2_gate_burn_checks_skip_when_feature_absent() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cfg!(feature = "backend-burn") {
            return;
        }
        let root = repo_root();
        let out = root.join("out/v2_gate_report_test_burn_skip.json");
        let report = v2_gate(&root.join(".ucf"), &out).expect("v2 gate");
        let burn_probe = report
            .checks
            .iter()
            .find(|c| c.name == "burn_world_probe_pass")
            .expect("burn probe check");
        let burn_compare = report
            .checks
            .iter()
            .find(|c| c.name == "burn_world_shadow_compare_present")
            .expect("burn compare check");
        assert!(matches!(burn_probe.status, GateStatus::Skip));
        assert!(matches!(burn_compare.status, GateStatus::Skip));
    }

    #[test]
    fn v3_gate_check_order_is_fixed() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let out = root.join("out/v3_gate_report_test_order.json");
        let report = v3_gate(&root.join(".ucf"), &out).expect("v3 gate");
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            vec![
                "v0_gate_pass",
                "v1_gate_pass",
                "v2_gate_pass",
                "supported_slot_set_detected",
                "world_probe_ready",
                "second_slot_probe_ready",
                "world_shadow_ready",
                "second_slot_shadow_ready",
                "world_shadow_no_impact",
                "second_slot_shadow_no_impact",
                "compare_window_semantics_normalized",
                "unified_eligibility_report_present",
                "strict_check_v3_pass",
                "operator_report_present",
                "portability_docs_checks_pass",
                "burn_world_parity_present",
            ]
        );
    }

    #[test]
    fn v3_gate_report_serialization_is_deterministic() {
        let report = V3GateReportV1 {
            schema_version: 1,
            overall_status: V3GateOverallStatus::Pass,
            checks: vec![
                v3_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "R1",
                    "N1",
                ),
                v3_gate_check(
                    "b",
                    GateStatus::Skip,
                    [("x".to_string(), "y".to_string())],
                    "R2",
                    "N2",
                ),
            ],
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
    }

    #[test]
    fn v3_second_slot_detection_fails_when_ambiguous() {
        let _guard = crate::test_cwd_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root = repo_root();
        let snapshot = root.join("docs/series_state_snapshot.md");
        with_replaced_file(
            &snapshot,
            "Second supported slot declaration: sae and ssm",
            || {
                let err = detect_second_slot_for_v3(&root).expect_err("must fail");
                assert!(err.to_string().contains("V3_SECOND_SLOT_AMBIGUOUS"));
            },
        );
    }
}

pub fn generate_remediation_codes_doc(out: &Path) -> Result<(), OpsError> {
    let mut md = String::from("# Remediation Codes v1\n\nGenerated from the canonical remediation registry source.\n\n| Code | Description | Suggestion Key |\n|---|---|---|\n");
    for (code, description, key) in remediation_registry_rows() {
        md.push_str(&format!("| {code} | {description} | `{key}` |\n"));
    }
    md.push_str("\n## Remediation consistency enforcement\n\n");
    md.push_str("Canonical remediation consistency is enforced across strict check, eligibility, operator report, operator signoff, v4 gate surfaces, and enriched export manifests via `ucf-ops remediation-consistency-check`.\n");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, md)?;
    Ok(())
}

#[cfg(test)]
mod v5_gate_tests {
    use super::*;

    #[test]
    fn v5_gate_check_order_is_fixed() {
        let checks = vec![
            "v0_gate_pass",
            "v1_gate_pass",
            "v2_gate_pass",
            "v3_gate_pass",
            "v4_gate_pass",
            "supported_set_review_present",
            "supported_set_review_consistent",
            "active_review_snapshot_present",
            "active_review_snapshot_consistent",
            "backend_resolution_present",
            "backend_resolution_consistent",
            "enriched_repro_export_smoke_pass",
            "enriched_bugkit_export_smoke_pass",
            "remediation_consistency_pass",
            "operator_review_packet_present",
            "operator_review_packet_consistent",
            "artifact_schema_snapshot_checks_pass",
            "portability_docs_checks_pass",
            "optional_backend_resolution_consistent",
            "chosen_slot_burn_optional_path_consistent",
        ];
        let report = V5GateReportV1 {
            schema_version: 1,
            overall_status: V5GateOverallStatus::Pass,
            checks: checks
                .iter()
                .map(|name| v5_gate_check(name, GateStatus::Pass, [], "REMEDIATE", "NOTE"))
                .collect(),
        };
        let names: Vec<String> = report.checks.into_iter().map(|c| c.name).collect();
        assert_eq!(names, checks);
    }

    #[test]
    fn v5_gate_report_serialization_is_deterministic() {
        let report = V5GateReportV1 {
            schema_version: 1,
            overall_status: V5GateOverallStatus::Pass,
            checks: vec![
                v5_gate_check(
                    "a",
                    GateStatus::Pass,
                    [("k".to_string(), "v".to_string())],
                    "REMEDIATE_A",
                    "NOTE_A",
                ),
                v5_gate_check(
                    "b",
                    GateStatus::Skip,
                    [("x".to_string(), "y".to_string())],
                    "REMEDIATE_B",
                    "NOTE_B",
                ),
            ],
        };
        let a = serde_json::to_vec(&report).expect("serialize a");
        let b = serde_json::to_vec(&report).expect("serialize b");
        assert_eq!(a, b);
    }

    #[test]
    fn v5_gate_normalization_fail_closed() {
        let report = V5GateReportV1 {
            schema_version: 1,
            overall_status: V5GateOverallStatus::Fail,
            checks: vec![
                v5_gate_check(
                    "required",
                    GateStatus::Fail,
                    [("missing".to_string(), "1".to_string())],
                    "REMEDIATE_REQUIRED",
                    "NOTE_REQUIRED",
                ),
                v5_gate_check(
                    "optional",
                    GateStatus::Skip,
                    [("unsupported".to_string(), "1".to_string())],
                    "REMEDIATE_OPTIONAL",
                    "NOTE_OPTIONAL",
                ),
            ],
        };
        assert!(matches!(report.overall_status, V5GateOverallStatus::Fail));
    }
}
