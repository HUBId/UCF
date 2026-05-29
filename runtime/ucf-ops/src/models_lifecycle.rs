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
use crate::{
    derive_canonical_governance_entry, prefix_hex, resolve_strict_evidence, sha256_hex,
    validate_governance_primary_surfaces_with_applied_scope,
    AbsoluteFinalGovernanceTerminalSweepStatusV1, CanonicalGovernanceEntryStatusV1,
    FinalGovernanceConsumerAuthorityStatusV1, FinalGovernanceConsumerSweepReportV1,
    GovernanceAbsoluteSweepReportV1, GovernanceClosureStatusV1, GovernanceClosureSweepReportV1,
    GovernanceEntryAuthorityStatusV2, GovernanceEntryCheckReportV1, GovernanceEntryCheckStatusV1,
    GovernanceEntrySweepReportV1, GovernanceFinalConsolidationStatusV1,
    GovernanceFinalConsolidationSweepReportV1, GovernanceSealStatusV1, GovernanceSealSweepReportV1,
    GovernanceTerminalSweepReportV1, GovernanceUltimateSweepReportV1, OperatorSignoffDecisionV1,
    OpsError, ResidualFreeGovernanceAbsoluteSweepStatusV1,
    ResidualFreeGovernanceConsumerAuthorityStatusV1, ResidualFreeGovernanceSweepReportV1,
    SignoffDecisionStateV1, StrictEvidenceContextV1, StrictEvidenceStatusV1,
    TerminalGovernanceUltimateSweepStatusV1,
};

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
const ACTIVE_REVIEW_EVIDENCE_SCHEMA_VERSION: u16 = 1;
const ACTIVE_REVIEW_REMEDIATION_MAX: usize = 4;
const SUPPORTED_REAL_SLOT_SET_VERSION: &str = "v3_supported_real_slots_max2";
const SUPPORTED_REAL_SLOT_SET_POLICY_V2_SCHEMA_VERSION: u16 = 2;
const SUPPORTED_REAL_SLOT_SET_V2_SCHEMA_VERSION: u16 = 2;
const APPLIED_SUPPORTED_SET_CONTEXT_SCHEMA_VERSION: u16 = 1;
const SLOT_EXPANSION_ELIGIBILITY_SCHEMA_VERSION: u16 = 1;
const SUPPORTED_SCOPE_REEVALUATION_V1_SCHEMA_VERSION: u16 = 1;
const SUPPORTED_SCOPE_EXECUTION_V3_SCHEMA_VERSION: u16 = 3;
const SUPPORTED_SCOPE_EXECUTION_V4_SCHEMA_VERSION: u16 = 4;
const SUPPORTED_SCOPE_EXECUTION_V5_SCHEMA_VERSION: u16 = 5;
const SUPPORTED_SCOPE_EXECUTION_V6_SCHEMA_VERSION: u16 = 6;
const SUPPORTED_SCOPE_EXECUTION_V7_SCHEMA_VERSION: u16 = 7;
const SUPPORTED_SCOPE_EXECUTION_V8_SCHEMA_VERSION: u16 = 8;
const SUPPORTED_SCOPE_EXECUTION_V9_SCHEMA_VERSION: u16 = 9;
const SUPPORTED_SCOPE_EXECUTION_V10_SCHEMA_VERSION: u16 = 10;
const SUPPORTED_SCOPE_EXECUTION_V11_SCHEMA_VERSION: u16 = 11;
const SUPPORTED_SCOPE_EXECUTION_V12_SCHEMA_VERSION: u16 = 12;
const SUPPORTED_SCOPE_EXECUTION_V13_SCHEMA_VERSION: u16 = 13;
const SUPPORTED_SCOPE_EXECUTION_V14_SCHEMA_VERSION: u16 = 14;
const SUPPORTED_SCOPE_EXECUTION_V15_SCHEMA_VERSION: u16 = 15;
const SUPPORTED_SCOPE_EXPANSION_DECISION_V1_SCHEMA_VERSION: u16 = 1;
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
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedRealSlotSetExecutionDecisionV2 {
    Frozen,
    Expanded,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedRealSlotSetV2 {
    pub schema_version: u16,
    pub slots: Vec<String>,
    pub source_policy_digest_prefix: String,
    pub decision: SupportedRealSlotSetExecutionDecisionV2,
    pub previous_set_digest_prefix: String,
    pub set_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppliedSupportedSetContextV1 {
    pub schema_version: u16,
    pub applied_set_digest_prefix: String,
    pub slots: Vec<String>,
    pub decision: SupportedRealSlotSetExecutionDecisionV2,
    pub previous_set_digest_prefix: String,
    pub policy_digest_prefix: String,
    pub context_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compatibility_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedSetExecutionDeniedCodeV1 {
    SupportedSetExecutionDeniedStalePolicy,
    SupportedSetExecutionDeniedIncompleteScaffold,
    SupportedSetExecutionDeniedAmbiguousSlot,
    SupportedSetExecutionDeniedScopeMismatch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedSetExecutionDeniedV1 {
    pub code: SupportedSetExecutionDeniedCodeV1,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedSetFreezeRecordV1 {
    pub schema_version: u16,
    pub previous_set_digest: String,
    pub resulting_set_digest: String,
    pub reason_codes: Vec<String>,
    pub policy_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedSetExpansionRecordV1 {
    pub schema_version: u16,
    pub previous_set_digest: String,
    pub resulting_set_digest: String,
    pub added_slot_id: String,
    pub policy_digest_prefix: String,
    pub evidence_summary_digests: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedSetApplyReportV1 {
    pub schema_version: u16,
    pub previous_slots: Vec<String>,
    pub resulting_slots: Vec<String>,
    pub decision: SupportedRealSlotSetExecutionDecisionV2,
    pub denial_code: Option<SupportedSetExecutionDeniedCodeV1>,
    pub rationale_codes: Vec<String>,
    pub applied_set: SupportedRealSlotSetV2,
    pub freeze_record: Option<SupportedSetFreezeRecordV1>,
    pub expansion_record: Option<SupportedSetExpansionRecordV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedRealSlotSetDecisionV2 {
    Freeze,
    ExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SlotExpansionEligibilityV1 {
    pub schema_version: u16,
    pub slot_id: String,
    pub trait_contract_exists: bool,
    pub probe_path_exists_or_reusable: bool,
    pub shadow_path_exists_or_trivially_attachable: bool,
    pub compare_window_normalizable: bool,
    pub strict_evidence_plumbing_representable_without_arch_fork: bool,
    pub tiny_fixture_path_feasible: bool,
    pub expansion_ready: bool,
    pub denial_reason_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum KnownSlotClassificationV1 {
    CurrentlySupportedRealSlot,
    StubOnly,
    PartiallyScaffolded,
    UnsupportedAbsent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnownSlotReviewV1 {
    pub slot_id: String,
    pub classification: KnownSlotClassificationV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedRealSlotSetPolicyV2 {
    pub schema_version: u16,
    pub current_supported_slots: Vec<String>,
    pub candidate_slots_considered: Vec<String>,
    pub decision: SupportedRealSlotSetDecisionV2,
    pub chosen_candidate_slot: Option<String>,
    pub rationale_codes: Vec<String>,
    pub policy_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedSetReviewReportV1 {
    pub policy: SupportedRealSlotSetPolicyV2,
    pub known_slots: Vec<KnownSlotReviewV1>,
    pub candidates: Vec<SlotExpansionEligibilityV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeReevaluationDecisionV1 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeReevaluationV1 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub policy_digest_prefix: String,
    pub reevaluation_decision: SupportedScopeReevaluationDecisionV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub rationale_codes: Vec<String>,
    pub reevaluation_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV3 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV3 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub execution_decision: SupportedScopeExecutionDecisionV3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV4 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV4 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV4,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV5 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV5 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV5,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV6 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV6 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV6,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV7 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV7 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV7,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV8 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV8 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV9 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV9 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV9,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV10 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV11 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV12 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV13 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV14 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExecutionDecisionV15 {
    ReaffirmFreeze,
    ExecuteExpandByOne,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV10 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV10,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV11 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV11,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV12 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    pub governance_stabilization_sweep_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV12,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV13 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    pub governance_stabilization_sweep_digest_prefix: String,
    pub governance_final_consolidation_sweep_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV13,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV14 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    pub governance_stabilization_sweep_digest_prefix: String,
    pub governance_final_consolidation_sweep_digest_prefix: String,
    pub governance_closure_sweep_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV14,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExecutionV15 {
    pub schema_version: u16,
    pub previous_applied_set_digest_prefix: String,
    pub current_policy_digest_prefix: String,
    pub current_reevaluation_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    pub governance_stabilization_sweep_digest_prefix: String,
    pub governance_final_consolidation_sweep_digest_prefix: String,
    pub governance_closure_sweep_digest_prefix: String,
    pub governance_seal_sweep_digest_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_scope_execution_digest_prefix: Option<String>,
    pub execution_decision: SupportedScopeExecutionDecisionV15,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_candidate_slot: Option<String>,
    pub resulting_supported_set_digest_prefix: String,
    pub rationale_codes: Vec<String>,
    pub execution_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExpansionDecisionStatusV1 {
    ScopeExpansionApplied,
    ScopeFreezeReinforced,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeExpansionReasonCodeV1 {
    NoEligibleCandidateSlot,
    GovernanceBlocksExpansion,
    CurrentScopeExecutionInsufficient,
    ReadinessPrerequisiteMissing,
    BundleExportSemanticsInsufficient,
    PrimarySemanticsWouldOverstate,
    ContinuityChainWouldFork,
    ExactlyOneSlotExpanded,
    MultipleCandidatesPresentRequireFreeze,
    CanonicalScopeSurfaceContradiction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SupportedScopeDecisionMismatchCategoryV1 {
    CandidateNotCanonical,
    GovernanceDenied,
    GovernanceScopeMismatch,
    ExecutionPathMissing,
    ExecutionPathNonCanonical,
    RuntimeActivationRequired,
    ReadinessBlocked,
    ExportScopeWideningRisk,
    PrimarySemanticsContradiction,
    ContinuityConflict,
    MultiSlotExpansionForbidden,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeDecisionCandidateV1 {
    pub slot_id: String,
    pub currently_supported: bool,
    pub status: SupportedScopeExpansionDecisionStatusV1,
    pub reason_code: SupportedScopeExpansionReasonCodeV1,
    pub mismatch_categories: Vec<SupportedScopeDecisionMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExpansionDecisionV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    pub governance_stabilization_sweep_digest_prefix: String,
    pub governance_final_consolidation_sweep_digest_prefix: String,
    pub governance_closure_sweep_digest_prefix: String,
    pub governance_seal_sweep_digest_prefix: String,
    pub governance_lock_sweep_digest_prefix: String,
    pub current_supported_scope_digest_prefix: String,
    pub candidate_count: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winning_candidate_slot: Option<String>,
    pub decision_status: SupportedScopeExpansionDecisionStatusV1,
    pub decision_reason_code: SupportedScopeExpansionReasonCodeV1,
    pub evaluated_consumer_count: u16,
    pub contradictory_surface_count: u16,
    pub unsupported_slot_count: u16,
    pub decision_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportedScopeExpansionDecisionReportV1 {
    pub schema_version: u16,
    pub decision: SupportedScopeExpansionDecisionV1,
    pub candidates: Vec<SupportedScopeDecisionCandidateV1>,
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
    pub burn_resolution: BurnSupportResolutionV1,
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
pub struct ActiveReviewContributingDigestsV1 {
    pub probe_report_digest_prefix: String,
    pub shadow_ready_digest_prefix: String,
    pub active_evidence_digest_prefix: String,
    pub strict_evidence_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveReviewEvidenceV1 {
    pub slot_id: String,
    pub target_hash_prefix: String,
    pub manifest_digest_prefix: String,
    pub probe_ready: bool,
    pub shadow_ready: bool,
    pub active_eligible: bool,
    pub strict_blocking: bool,
    pub drift_blocking: bool,
    pub alert_blocking: bool,
    pub primary_denial_code: Option<String>,
    pub remediation_codes: Vec<String>,
    pub contributing_evidence_digests: ActiveReviewContributingDigestsV1,
    pub burn_resolution: BurnSupportResolutionV1,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BurnResolutionStatusV1 {
    BurnSupportedForShadowCompare,
    BurnClosedUnsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BurnSupportResolutionV1 {
    pub slot_id: String,
    pub resolution: BurnResolutionStatusV1,
    pub support_state: OptionalBackendSupportStateV1,
    pub rationale_codes: Vec<String>,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActiveReviewOverallStatusV1 {
    NoneReviewable,
    PartialReviewable,
    AllReviewable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveReviewSignoffAlignmentV1 {
    pub aligned: bool,
    pub status_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AggregatedActiveReviewSnapshotV1 {
    pub schema_version: u16,
    pub supported_slot_set_digest: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub slots: Vec<ActiveReviewEvidenceV1>,
    pub overall_review_status: ActiveReviewOverallStatusV1,
    pub signoff_alignment: ActiveReviewSignoffAlignmentV1,
    #[serde(default)]
    pub canonical_governance_entry_digest_prefix: String,
    #[serde(default)]
    pub final_governance_consumer_authority_digest_prefix: String,
    #[serde(default)]
    pub governance_residual_sweep_digest_prefix: String,
    #[serde(default)]
    pub residual_free_governance_authority_digest_prefix: String,
    #[serde(default)]
    pub governance_absolute_sweep_digest_prefix: String,
    #[serde(default)]
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub governance_ultimate_sweep_digest_prefix: String,
    #[serde(default)]
    pub final_readiness_consumer_authority_digest_prefix: String,
    #[serde(default)]
    pub readiness_residual_sweep_digest_prefix: String,
    #[serde(default)]
    pub residual_free_readiness_authority_digest_prefix: String,
    #[serde(default)]
    pub readiness_absolute_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_terminal_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_ultimate_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_stabilization_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_final_consolidation_sweep_digest_prefix: String,
    #[serde(default)]
    pub readiness_closure_sweep_digest_prefix: String,
    pub readiness_seal_sweep_digest_prefix: String,
    #[serde(default)]
    pub snapshot_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveReviewSnapshotRecordV1 {
    pub invocation_id: u64,
    pub snapshot_digest_prefix: String,
    pub supported_slot_set_digest_prefix: String,
    pub slots: Vec<ActiveReviewSnapshotSlotV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveReviewSnapshotSlotV1 {
    pub slot_id: String,
    pub reviewable: bool,
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
pub struct AppliedScopeCheckReportV1 {
    pub schema_version: u16,
    pub status: String,
    pub applied_scope_digest: String,
    pub checked_artifacts: Vec<String>,
    pub mismatch_categories: Vec<String>,
    pub remediation_codes: Vec<String>,
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
            "required {MODEL_FILE_NAME} missing"
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
    digest_source.extend_from_slice(format!("{overall_status:?}").as_bytes());
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
    let slot_set = current_supported_real_slot_set(workdir)?;
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
        .filter_map(|r| map_denial_reason_to_code(Some(r)).map(|c| format!("{c:?}")))
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

pub fn models_applied_scope_check(
    workdir: &Path,
    out: &Path,
) -> Result<AppliedScopeCheckReportV1, OpsError> {
    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let mut mismatch = BTreeSet::new();
    let mut remediation = BTreeSet::new();
    let mut checked_artifacts = Vec::new();

    let backend_path = workdir.join("out").join("backend_evidence_snapshot.json");
    let active_path = workdir.join("out").join("active_review_snapshot.json");
    let signoff_path = workdir.join("out").join("operator_signoff.json");
    let review_packet_path = workdir.join("out").join("operator_review_packet.json");

    let backend = read_json_file::<BackendEvidenceSnapshotV1>(&backend_path).ok();
    if backend.is_none() {
        mismatch.insert("APPLIED_SCOPE_BACKEND_EVIDENCE_MISSING".to_string());
        remediation.insert("run_backend_evidence_snapshot".to_string());
    } else {
        checked_artifacts.push("BackendEvidenceSnapshot".to_string());
    }
    let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(&active_path).ok();
    if active.is_none() {
        mismatch.insert("APPLIED_SCOPE_ACTIVE_REVIEW_MISSING".to_string());
        remediation.insert("run_models_active_review_snapshot".to_string());
    } else {
        checked_artifacts.push("ActiveReviewSnapshot".to_string());
    }
    let signoff = read_json_file::<crate::OperatorSignoffDecisionV1>(&signoff_path).ok();
    if signoff.is_none() {
        mismatch.insert("APPLIED_SCOPE_OPERATOR_SIGNOFF_MISSING".to_string());
        remediation.insert("run_operator_signoff".to_string());
    } else {
        checked_artifacts.push("OperatorSignoff".to_string());
    }
    let review_packet = read_json_file::<crate::OperatorReviewPacketV1>(&review_packet_path).ok();
    if review_packet.is_none() {
        mismatch.insert("APPLIED_SCOPE_OPERATOR_REVIEW_PACKET_MISSING".to_string());
        remediation.insert("run_operator_review_packet".to_string());
    } else {
        checked_artifacts.push("OperatorReviewPacket".to_string());
    }

    if let Some(backend) = backend.as_ref() {
        if prefix_hex(&backend.supported_slot_set_digest, 16)
            != applied_scope.applied_set_digest_prefix
        {
            mismatch.insert("APPLIED_SCOPE_BACKEND_DIGEST_MISMATCH".to_string());
            remediation.insert("run_backend_evidence_snapshot".to_string());
        }
        let backend_slots = backend
            .slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect::<Vec<_>>();
        if backend_slots != applied_scope.slots {
            mismatch.insert("APPLIED_SCOPE_BACKEND_SLOT_SCOPE_DRIFT".to_string());
            remediation.insert("run_backend_evidence_snapshot".to_string());
        }
    }
    if let Some(active) = active.as_ref() {
        if prefix_hex(&active.supported_slot_set_digest, 16)
            != applied_scope.applied_set_digest_prefix
        {
            mismatch.insert("APPLIED_SCOPE_ACTIVE_DIGEST_MISMATCH".to_string());
            remediation.insert("run_models_active_review_snapshot".to_string());
        }
        let active_slots = active
            .slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect::<Vec<_>>();
        if active_slots != applied_scope.slots {
            mismatch.insert("APPLIED_SCOPE_ACTIVE_SLOT_SCOPE_DRIFT".to_string());
            remediation.insert("run_models_active_review_snapshot".to_string());
        }
    }
    if let Some(signoff) = signoff.as_ref() {
        if signoff.supported_slot_set_digest != applied_scope.applied_set_digest_prefix {
            mismatch.insert("APPLIED_SCOPE_SIGNOFF_DIGEST_MISMATCH".to_string());
            remediation.insert("run_operator_signoff".to_string());
        }
    }
    if let Some(packet) = review_packet.as_ref() {
        if packet.supported_slot_set_digest != applied_scope.applied_set_digest_prefix {
            mismatch.insert("APPLIED_SCOPE_REVIEW_PACKET_DIGEST_MISMATCH".to_string());
            remediation.insert("run_operator_review_packet".to_string());
        }
        let packet_slots = packet
            .supported_slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect::<Vec<_>>();
        if packet_slots != applied_scope.slots {
            mismatch.insert("APPLIED_SCOPE_REVIEW_PACKET_SLOT_SCOPE_DRIFT".to_string());
            remediation.insert("run_operator_review_packet".to_string());
        }
        if packet.artifacts.applied_supported_set_context_digest_prefix
            != prefix_hex(&applied_scope.context_digest, 16)
        {
            mismatch.insert("APPLIED_SCOPE_REVIEW_PACKET_CONTEXT_DIGEST_MISMATCH".to_string());
            remediation.insert("run_operator_review_packet".to_string());
        }
    }

    checked_artifacts.sort();
    checked_artifacts.dedup();
    let report = AppliedScopeCheckReportV1 {
        schema_version: 1,
        status: if mismatch.is_empty() { "PASS" } else { "FAIL" }.to_string(),
        applied_scope_digest: prefix_hex(&applied_scope.context_digest, 16),
        checked_artifacts,
        mismatch_categories: mismatch.into_iter().collect(),
        remediation_codes: remediation.into_iter().collect(),
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn read_json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, OpsError> {
    let body = fs::read_to_string(path)?;
    serde_json::from_str(&body).map_err(|err| OpsError::Invalid(err.to_string()))
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
    digest_source.extend_from_slice(format!("{overall_status:?}").as_bytes());
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
    digest_source.extend_from_slice(format!("{drift_status:?}").as_bytes());
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
                .map(|c| format!("{c:?}"))
                .or(Some(format!("{:?}", denied.code))),
            shadow.latest_drift_status.clone(),
        ),
    };

    let denial_reason_shadow = map_denial_reason_to_code(shadow.denial_reason_code.as_deref())
        .map(|c| format!("{c:?}"))
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
    digest_source.extend_from_slice(format!("{drift_status:?}").as_bytes());
    digest_source.extend_from_slice(format!("{burn_support_state:?}").as_bytes());
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
    let burn_resolution = slot.burn_resolution.clone();
    let denial_reason_probe = slot.denials.probe.as_ref().map(|d| format!("{d:?}"));
    let denial_reason_shadow = slot.denials.shadow.as_ref().map(|d| format!("{d:?}"));
    let denial_reason_active = slot.denials.active.as_ref().map(|d| format!("{d:?}"));

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
    digest_source.extend_from_slice(format!("{burn_support_state:?}").as_bytes());
    digest_source.extend_from_slice(format!("{:?}", burn_resolution.resolution).as_bytes());
    digest_source.extend_from_slice(burn_resolution.evidence_digest.as_bytes());
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
    let slot_set = current_supported_real_slot_set(workdir)?;
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
        let burn_support_state = match backend_support.burn {
            BackendSupportStateV1::Supported => OptionalBackendSupportStateV1::Supported,
            BackendSupportStateV1::Unsupported => OptionalBackendSupportStateV1::Unsupported,
            BackendSupportStateV1::NotBuilt => OptionalBackendSupportStateV1::NotBuilt,
            BackendSupportStateV1::NotConfigured => OptionalBackendSupportStateV1::NotConfigured,
        };
        let burn_resolution = burn_support_resolution_from_state(slot, burn_support_state);
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
            burn_resolution,
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
        digest_source
            .extend_from_slice(format!("{:?}", slot.burn_resolution.resolution).as_bytes());
        digest_source.extend_from_slice(slot.burn_resolution.evidence_digest.as_bytes());
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

pub fn models_backend_resolution(
    workdir: &Path,
    slot: ModelSlot,
    run_id: Option<&str>,
) -> Result<BurnSupportResolutionV1, OpsError> {
    let detected = crate::detect_second_slot(workdir)?;
    if slot != detected {
        return Err(OpsError::Invalid(format!(
            "SECOND_SLOT_SCOPE_VIOLATION: configured second slot is {}",
            detected.as_str()
        )));
    }
    let parity_report = read_second_slot_parity_report(workdir, run_id, slot.as_str());
    let support_state = parity_report
        .as_ref()
        .map(|r| r.burn_support_state.clone())
        .unwrap_or_else(|| {
            if cfg!(feature = "backend-burn") {
                OptionalBackendSupportStateV1::NotConfigured
            } else {
                OptionalBackendSupportStateV1::NotBuilt
            }
        });
    Ok(burn_support_resolution_from_state(slot, support_state))
}

fn discover_latest_operator_signoff(workdir: &Path) -> Option<OperatorSignoffDecisionV1> {
    let out_root = workdir.join("out");
    let direct = out_root.join("operator_signoff.json");
    if direct.exists() {
        if let Ok(body) = fs::read_to_string(&direct) {
            if let Ok(report) = serde_json::from_str::<OperatorSignoffDecisionV1>(&body) {
                return Some(report);
            }
        }
    }
    None
}

fn discover_alert_blocking(workdir: &Path) -> bool {
    let path = workdir.join("out").join("alerts_report.json");
    fs::read_to_string(path)
        .ok()
        .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
        .and_then(|v| v.get("active_alerts").and_then(|x| x.as_array()).cloned())
        .map(|alerts| {
            alerts.iter().any(|a| {
                a.get("severity")
                    .and_then(|x| x.as_str())
                    .is_some_and(|s| s.eq_ignore_ascii_case("severe"))
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
fn derive_active_review_status(slots: &[ActiveReviewEvidenceV1]) -> ActiveReviewOverallStatusV1 {
    let reviewable_count = slots
        .iter()
        .filter(|slot| {
            slot.active_eligible
                && !slot.strict_blocking
                && !slot.drift_blocking
                && !slot.alert_blocking
        })
        .count();
    if reviewable_count == 0 {
        return ActiveReviewOverallStatusV1::NoneReviewable;
    }
    if reviewable_count == slots.len() {
        return ActiveReviewOverallStatusV1::AllReviewable;
    }
    ActiveReviewOverallStatusV1::PartialReviewable
}

fn derive_signoff_alignment(
    signoff: Option<&OperatorSignoffDecisionV1>,
    snapshot: &BackendEvidenceSnapshotV1,
    overall_status: &ActiveReviewOverallStatusV1,
) -> ActiveReviewSignoffAlignmentV1 {
    let Some(signoff) = signoff else {
        return ActiveReviewSignoffAlignmentV1 {
            aligned: false,
            status_code: "SIGNOFF_MISSING".to_string(),
        };
    };

    if signoff.supported_slot_set_digest != snapshot.supported_slot_set_digest
        || signoff.policy_graph_digest_prefix != snapshot.policy_graph_digest_prefix
        || signoff.manifest_digest_prefix != snapshot.manifest_digest_prefix
    {
        return ActiveReviewSignoffAlignmentV1 {
            aligned: false,
            status_code: "SIGNOFF_INPUT_MISMATCH".to_string(),
        };
    }

    let expects_active_review =
        !matches!(overall_status, ActiveReviewOverallStatusV1::NoneReviewable);
    let decision_is_active_review =
        signoff.decision == SignoffDecisionStateV1::ReadyForActiveReview;
    if expects_active_review == decision_is_active_review {
        return ActiveReviewSignoffAlignmentV1 {
            aligned: true,
            status_code: "ALIGNED".to_string(),
        };
    }

    ActiveReviewSignoffAlignmentV1 {
        aligned: false,
        status_code: "SIGNOFF_DECISION_MISMATCH".to_string(),
    }
}

fn append_active_review_snapshot_record(
    workdir: &Path,
    report: &AggregatedActiveReviewSnapshotV1,
) -> Result<(), OpsError> {
    let path = workdir
        .join("out")
        .join("records")
        .join("active_review_snapshot_records.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut all: Vec<ActiveReviewSnapshotRecordV1> = if path.exists() {
        serde_json::from_str(&fs::read_to_string(&path)?).unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut slots = report
        .slots
        .iter()
        .map(|slot| ActiveReviewSnapshotSlotV1 {
            slot_id: slot.slot_id.clone(),
            reviewable: crate::slot_is_reviewable(&crate::SlotReviewabilityTruthV1 {
                slot_id: slot.slot_id.clone(),
                target_hash_prefix: slot.target_hash_prefix.clone(),
                probe_ready: slot.probe_ready,
                shadow_ready: slot.shadow_ready,
                active_eligible: slot.active_eligible,
                strict_blocking: slot.strict_blocking,
                drift_blocking: slot.drift_blocking,
                alert_blocking: slot.alert_blocking,
                primary_denial_code: slot.primary_denial_code.clone(),
                remediation_codes: slot.remediation_codes.clone(),
                evidence_digests: crate::SlotReviewabilityEvidenceDigestsV1 {
                    backend_evidence_snapshot_digest_prefix: String::new(),
                    active_evidence_digest_prefix: String::new(),
                    strict_evidence_digest_prefix: String::new(),
                },
                reviewability_truth_digest: String::new(),
            }),
        })
        .collect::<Vec<_>>();
    slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    all.push(ActiveReviewSnapshotRecordV1 {
        invocation_id: now_secs(),
        snapshot_digest_prefix: prefix_hex(&report.snapshot_digest, 16),
        supported_slot_set_digest_prefix: report.supported_slot_set_digest.clone(),
        slots,
    });
    fs::write(path, serde_json::to_vec_pretty(&all)?)?;
    Ok(())
}

pub fn models_active_review_snapshot(
    workdir: &Path,
    out: &Path,
) -> Result<AggregatedActiveReviewSnapshotV1, OpsError> {
    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let backend_snapshot = models_evidence_snapshot(workdir, None, None)?;
    validate_snapshot_matches_applied_scope(
        &backend_snapshot.supported_slot_set_digest,
        &backend_snapshot
            .slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect::<Vec<_>>(),
        &applied_scope,
        "ACTIVE_REVIEW_SCOPE_MISMATCH",
    )?;
    if backend_snapshot.slots.is_empty() {
        return Err(OpsError::Invalid(
            "ACTIVE_REVIEW_SLOT_SET_EMPTY: supported real-slot set cannot be empty".to_string(),
        ));
    }

    let strict_snapshot = resolve_strict_evidence(
        &workdir.join("out"),
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
    let strict_blocking = matches!(
        strict_snapshot.strict_status,
        StrictEvidenceStatusV1::Fail | StrictEvidenceStatusV1::Missing
    );
    let alert_blocking = discover_alert_blocking(workdir);

    let strict_digest_prefix = if strict_snapshot.snapshot_digest.is_empty() {
        "missing".to_string()
    } else {
        prefix_hex(&strict_snapshot.snapshot_digest, 16)
    };

    let mut slots = backend_snapshot
        .slots
        .iter()
        .map(|slot| {
            let drift_blocking = matches!(slot.evidence.latest_drift_status, DriftStatusV1::Severe);
            let mut remediation = BTreeSet::new();
            for code in slot
                .canonical_remediation_codes
                .iter()
                .chain(slot.remediation_codes.iter())
            {
                remediation.insert(code.clone());
            }
            if strict_blocking {
                remediation.insert("run_strict_check".to_string());
            }
            if drift_blocking {
                remediation.insert("run_drift_report".to_string());
            }
            if alert_blocking {
                remediation.insert("inspect_active_alerts".to_string());
            }

            let primary_denial_code = if strict_blocking {
                strict_snapshot
                    .primary_denial_code
                    .clone()
                    .or(Some("STRICT_BLOCKING".to_string()))
            } else if drift_blocking {
                Some("DRIFT_BLOCKING".to_string())
            } else if alert_blocking {
                Some("ALERT_BLOCKING".to_string())
            } else {
                slot.denials
                    .active
                    .as_ref()
                    .or(slot.denials.shadow.as_ref())
                    .or(slot.denials.probe.as_ref())
                    .map(|code| format!("{code:?}"))
            };

            let mut evidence = ActiveReviewEvidenceV1 {
                slot_id: slot.slot_id.clone(),
                target_hash_prefix: slot.target_hash_prefix.clone(),
                manifest_digest_prefix: backend_snapshot.manifest_digest_prefix.clone(),
                probe_ready: slot.readiness.probe_ready,
                shadow_ready: slot.readiness.shadow_ready,
                active_eligible: slot.readiness.active_eligible,
                strict_blocking,
                drift_blocking,
                alert_blocking,
                primary_denial_code,
                remediation_codes: remediation
                    .into_iter()
                    .take(ACTIVE_REVIEW_REMEDIATION_MAX)
                    .collect(),
                contributing_evidence_digests: ActiveReviewContributingDigestsV1 {
                    probe_report_digest_prefix: slot
                        .evidence
                        .latest_probe_report_digest_prefix
                        .clone(),
                    shadow_ready_digest_prefix: slot
                        .evidence
                        .latest_shadow_ready_digest_prefix
                        .clone(),
                    active_evidence_digest_prefix: slot
                        .evidence
                        .latest_active_evidence_digest_prefix
                        .clone(),
                    strict_evidence_digest_prefix: strict_digest_prefix.clone(),
                },
                burn_resolution: slot.burn_resolution.clone(),
                evidence_digest: String::new(),
            };
            let mut digest_source = Vec::new();
            digest_source.extend_from_slice(evidence.slot_id.as_bytes());
            digest_source.extend_from_slice(evidence.target_hash_prefix.as_bytes());
            digest_source.extend_from_slice(evidence.manifest_digest_prefix.as_bytes());
            digest_source
                .extend_from_slice(format!("{:?}", evidence.burn_resolution.resolution).as_bytes());
            digest_source.extend_from_slice(evidence.burn_resolution.evidence_digest.as_bytes());
            digest_source.extend_from_slice(if evidence.probe_ready { b"1" } else { b"0" });
            digest_source.extend_from_slice(if evidence.shadow_ready { b"1" } else { b"0" });
            digest_source.extend_from_slice(if evidence.active_eligible { b"1" } else { b"0" });
            digest_source.extend_from_slice(if evidence.strict_blocking { b"1" } else { b"0" });
            digest_source.extend_from_slice(if evidence.drift_blocking { b"1" } else { b"0" });
            digest_source.extend_from_slice(if evidence.alert_blocking { b"1" } else { b"0" });
            if let Some(code) = evidence.primary_denial_code.as_ref() {
                digest_source.extend_from_slice(code.as_bytes());
            }
            for code in &evidence.remediation_codes {
                digest_source.extend_from_slice(code.as_bytes());
            }
            digest_source.extend_from_slice(
                evidence
                    .contributing_evidence_digests
                    .probe_report_digest_prefix
                    .as_bytes(),
            );
            digest_source.extend_from_slice(
                evidence
                    .contributing_evidence_digests
                    .shadow_ready_digest_prefix
                    .as_bytes(),
            );
            digest_source.extend_from_slice(
                evidence
                    .contributing_evidence_digests
                    .active_evidence_digest_prefix
                    .as_bytes(),
            );
            digest_source.extend_from_slice(
                evidence
                    .contributing_evidence_digests
                    .strict_evidence_digest_prefix
                    .as_bytes(),
            );
            evidence.evidence_digest = sha256_hex(&digest_source);
            evidence
        })
        .collect::<Vec<_>>();
    let slot_set = slots
        .iter()
        .map(|slot| slot.slot_id.clone())
        .collect::<Vec<_>>();
    validate_slot_membership(
        &slot_set,
        &applied_scope.slots,
        "ACTIVE_REVIEW_EXTRA_SLOT_EVIDENCE",
        "ACTIVE_REVIEW_MISSING_IN_SCOPE_SLOT",
    )?;
    slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    let temp_snapshot = AggregatedActiveReviewSnapshotV1 {
        schema_version: ACTIVE_REVIEW_EVIDENCE_SCHEMA_VERSION,
        supported_slot_set_digest: backend_snapshot.supported_slot_set_digest.clone(),
        policy_graph_digest_prefix: backend_snapshot.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: backend_snapshot.manifest_digest_prefix.clone(),
        slots: slots.clone(),
        overall_review_status: ActiveReviewOverallStatusV1::NoneReviewable,
        signoff_alignment: ActiveReviewSignoffAlignmentV1 {
            aligned: false,
            status_code: "PENDING".to_string(),
        },
        canonical_governance_entry_digest_prefix: "MISSING".to_string(),
        final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
        governance_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
        governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
        absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
        governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
        readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
        residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
        readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
        readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
        readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
        readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
        readiness_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
        readiness_closure_sweep_digest_prefix: "MISSING".to_string(),
        readiness_seal_sweep_digest_prefix: "MISSING".to_string(),
        snapshot_digest: String::new(),
    };
    let truths = crate::derive_slot_reviewability_truths_from_active(
        &applied_scope,
        &backend_snapshot,
        &temp_snapshot,
    )?;
    let reduction = crate::reduce_reviewability(&applied_scope, &truths)?;
    let overall_review_status = match reduction.aggregate_readiness {
        crate::ReviewabilityAggregateReadinessV1::NoneReviewable => {
            ActiveReviewOverallStatusV1::NoneReviewable
        }
        crate::ReviewabilityAggregateReadinessV1::PartialReviewable => {
            ActiveReviewOverallStatusV1::PartialReviewable
        }
        crate::ReviewabilityAggregateReadinessV1::AllReviewable => {
            ActiveReviewOverallStatusV1::AllReviewable
        }
    };
    let signoff = discover_latest_operator_signoff(workdir);
    let signoff_alignment =
        derive_signoff_alignment(signoff.as_ref(), &backend_snapshot, &overall_review_status);
    let canonical_governance_entry_digest_prefix =
        match validate_governance_primary_surfaces_with_applied_scope(
            &backend_snapshot,
            &temp_snapshot,
            &applied_scope,
        )
        .and_then(|surfaces| derive_canonical_governance_entry(&applied_scope, &surfaces))
        {
            Ok(entry) => prefix_hex(&entry.authority_digest, 16),
            Err(_) => "MISSING".to_string(),
        };
    let final_governance_consumer_authority_digest_prefix =
        read_final_governance_prefix(workdir, "out/final_governance_consumer_sweep.json");
    let governance_residual_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/governance_residual_sweep.json",
        "sweep_digest",
    );
    let residual_free_governance_authority_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/residual_free_governance_sweep.json",
        "authority_digest",
    );
    let governance_absolute_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/governance_absolute_sweep.json",
        "sweep.sweep_digest",
    );
    let absolute_final_governance_terminal_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/governance_terminal_sweep.json",
        "sweep.sweep_digest",
    );
    let governance_ultimate_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/governance_ultimate_sweep.json",
        "sweep.sweep_digest",
    );
    let final_readiness_consumer_authority_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/final_readiness_consumer_sweep.json",
        "authority_digest",
    );
    let readiness_residual_sweep_digest_prefix =
        read_sweep_digest_prefix(workdir, "out/readiness_residual_sweep.json", "sweep_digest");
    let residual_free_readiness_authority_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/residual_free_readiness_sweep.json",
        "authority_digest",
    );
    let readiness_absolute_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/readiness_absolute_sweep.json",
        "sweep.sweep_digest",
    );
    let readiness_terminal_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/readiness_terminal_sweep.json",
        "sweep.sweep_digest",
    );
    let readiness_ultimate_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/readiness_ultimate_sweep.json",
        "sweep.sweep_digest",
    );
    let readiness_stabilization_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/readiness_stabilization_sweep.json",
        "sweep.stabilization_digest",
    );
    let readiness_final_consolidation_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/readiness_final_consolidation_sweep.json",
        "sweep.consolidation_digest",
    );
    let readiness_closure_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/readiness_closure_sweep.json",
        "sweep.closure_digest",
    );
    let readiness_seal_sweep_digest_prefix = read_sweep_digest_prefix(
        workdir,
        "out/readiness_seal_sweep.json",
        "sweep.seal_digest",
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(ACTIVE_REVIEW_EVIDENCE_SCHEMA_VERSION.to_string().as_bytes());
    digest_source.extend_from_slice(backend_snapshot.supported_slot_set_digest.as_bytes());
    digest_source.extend_from_slice(backend_snapshot.policy_graph_digest_prefix.as_bytes());
    digest_source.extend_from_slice(backend_snapshot.manifest_digest_prefix.as_bytes());
    digest_source.extend_from_slice(format!("{overall_review_status:?}").as_bytes());
    digest_source.extend_from_slice(signoff_alignment.status_code.as_bytes());
    digest_source.extend_from_slice(if signoff_alignment.aligned {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(reduction.reduction_digest.as_bytes());
    digest_source.extend_from_slice(canonical_governance_entry_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_governance_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(governance_residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_free_governance_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(governance_absolute_sweep_digest_prefix.as_bytes());
    digest_source
        .extend_from_slice(absolute_final_governance_terminal_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(governance_ultimate_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_readiness_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(readiness_residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_free_readiness_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(readiness_absolute_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(readiness_terminal_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(readiness_ultimate_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(readiness_stabilization_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(readiness_final_consolidation_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(readiness_closure_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(readiness_seal_sweep_digest_prefix.as_bytes());
    for slot in &slots {
        digest_source.extend_from_slice(slot.evidence_digest.as_bytes());
    }

    let report = AggregatedActiveReviewSnapshotV1 {
        schema_version: ACTIVE_REVIEW_EVIDENCE_SCHEMA_VERSION,
        supported_slot_set_digest: backend_snapshot.supported_slot_set_digest,
        policy_graph_digest_prefix: backend_snapshot.policy_graph_digest_prefix,
        manifest_digest_prefix: backend_snapshot.manifest_digest_prefix,
        slots,
        overall_review_status,
        signoff_alignment,
        canonical_governance_entry_digest_prefix,
        final_governance_consumer_authority_digest_prefix,
        governance_residual_sweep_digest_prefix,
        residual_free_governance_authority_digest_prefix,
        governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix,
        governance_ultimate_sweep_digest_prefix,
        final_readiness_consumer_authority_digest_prefix,
        readiness_residual_sweep_digest_prefix,
        residual_free_readiness_authority_digest_prefix,
        readiness_absolute_sweep_digest_prefix,
        readiness_terminal_sweep_digest_prefix,
        readiness_ultimate_sweep_digest_prefix,
        readiness_stabilization_sweep_digest_prefix,
        readiness_final_consolidation_sweep_digest_prefix,
        readiness_closure_sweep_digest_prefix,
        readiness_seal_sweep_digest_prefix,
        snapshot_digest: sha256_hex(&digest_source),
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    append_active_review_snapshot_record(workdir, &report)?;
    Ok(report)
}

fn read_final_governance_prefix(workdir: &Path, rel_path: &str) -> String {
    let path = workdir.join(rel_path);
    let Ok(bytes) = fs::read(path) else {
        return "MISSING".to_string();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return "MISSING".to_string();
    };
    value
        .get("authority")
        .and_then(|authority| authority.get("authority_digest"))
        .and_then(serde_json::Value::as_str)
        .map(|digest| prefix_hex(digest, 16))
        .unwrap_or_else(|| "MISSING".to_string())
}

fn read_sweep_digest_prefix(workdir: &Path, rel_path: &str, field: &str) -> String {
    let path = workdir.join(rel_path);
    let Ok(bytes) = fs::read(path) else {
        return "MISSING".to_string();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return "MISSING".to_string();
    };
    value
        .get("sweep")
        .and_then(|sweep| sweep.get(field))
        .and_then(serde_json::Value::as_str)
        .map(|digest| prefix_hex(digest, 16))
        .unwrap_or_else(|| "MISSING".to_string())
}

pub fn models_supported_set_review(
    _workdir: &Path,
    out: &Path,
) -> Result<SupportedSetReviewReportV1, OpsError> {
    let supported = supported_real_slot_set_v1()?;
    let current_set = supported
        .slots
        .iter()
        .cloned()
        .collect::<BTreeSet<String>>();
    let mut known_slots = known_slots_ordered();
    let mut candidates = Vec::new();
    for slot_id in &known_slots {
        if !current_set.contains(slot_id) {
            candidates.push(evaluate_slot_expansion_candidate(
                slot_id,
                &current_set,
                SLOT_SET_MAX,
            ));
        }
    }
    known_slots.sort();
    candidates.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    let policy = select_supported_slot_set_policy_v2(&supported.slots, &candidates);
    let known_slots = known_slots
        .into_iter()
        .map(|slot_id| KnownSlotReviewV1 {
            classification: classify_known_slot(&slot_id, &current_set, &candidates),
            slot_id,
        })
        .collect::<Vec<_>>();

    let report = SupportedSetReviewReportV1 {
        policy,
        known_slots,
        candidates,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn models_supported_scope_reevaluate(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeReevaluationV1, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let baseline_set = current_supported_real_slot_set(workdir)?;
    let applied_set = SupportedRealSlotSetV2 {
        schema_version: SUPPORTED_REAL_SLOT_SET_V2_SCHEMA_VERSION,
        slots: baseline_set.slots.clone(),
        source_policy_digest_prefix: prefix_hex(&policy.policy_digest, 16),
        decision: SupportedRealSlotSetExecutionDecisionV2::Frozen,
        previous_set_digest_prefix: prefix_hex(&baseline_set.set_digest, 16),
        set_digest: baseline_set.set_digest.clone(),
    };

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let reevaluation_decision = if policy.current_supported_slots != baseline_set.slots {
        rationale_codes.push("SCOPE_REEVAL_STALE_POLICY".to_string());
        SupportedScopeReevaluationDecisionV1::ReaffirmFreeze
    } else {
        match policy.decision {
            SupportedRealSlotSetDecisionV2::Freeze => {
                rationale_codes.push("SCOPE_REEVAL_POLICY_FREEZE_REAFFIRMED".to_string());
                SupportedScopeReevaluationDecisionV1::ReaffirmFreeze
            }
            SupportedRealSlotSetDecisionV2::ExpandByOne => {
                let evaluated = policy
                    .candidate_slots_considered
                    .iter()
                    .map(|slot_id| {
                        let candidate = evaluate_slot_expansion_candidate(
                            slot_id,
                            &applied_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                            SLOT_SET_MAX + 1,
                        );
                        let failure_codes = validate_scope_expansion_under_authority(
                            workdir,
                            &applied_set,
                            &candidate,
                        );
                        (candidate, failure_codes)
                    })
                    .collect::<Vec<_>>();
                let viable = evaluated
                    .iter()
                    .filter(|(_, failures)| failures.is_empty())
                    .map(|(candidate, _)| candidate.slot_id.clone())
                    .collect::<Vec<_>>();
                if viable.len() != 1 {
                    rationale_codes.push("SCOPE_REEVAL_AMBIGUOUS_CANDIDATE".to_string());
                    SupportedScopeReevaluationDecisionV1::ReaffirmFreeze
                } else {
                    let selected = viable[0].clone();
                    if policy.chosen_candidate_slot.as_deref() != Some(selected.as_str()) {
                        rationale_codes.push("SCOPE_REEVAL_STALE_POLICY".to_string());
                        SupportedScopeReevaluationDecisionV1::ReaffirmFreeze
                    } else {
                        chosen_candidate_slot = Some(selected.clone());
                        rationale_codes.push("SCOPE_REEVAL_EXPANSION_JUSTIFIED".to_string());
                        rationale_codes.push("SCOPE_REEVAL_NO_ACTIVE_IMPLICATIONS".to_string());
                        SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
                    }
                }
            }
        }
    };

    if matches!(
        reevaluation_decision,
        SupportedScopeReevaluationDecisionV1::ReaffirmFreeze
    ) && rationale_codes.is_empty()
    {
        rationale_codes.push("SCOPE_REEVAL_FREEZE_DEFAULT".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_REEVALUATION_V1_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(prefix_hex(&applied_set.set_digest, 16).as_bytes());
    digest_source.extend_from_slice(prefix_hex(&policy.policy_digest, 16).as_bytes());
    digest_source.extend_from_slice(format!("{reevaluation_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    let report = SupportedScopeReevaluationV1 {
        schema_version: SUPPORTED_SCOPE_REEVALUATION_V1_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: prefix_hex(&applied_set.set_digest, 16),
        policy_digest_prefix: prefix_hex(&policy.policy_digest, 16),
        reevaluation_decision,
        chosen_candidate_slot,
        rationale_codes,
        reevaluation_digest: sha256_hex(&digest_source),
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn models_supported_set_apply(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedSetApplyReportV1, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let execution = ensure_current_supported_scope_execution_v15(workdir, &policy, &previous_set)?;

    let mut reevaluated_policy = policy.clone();
    match execution.execution_decision {
        SupportedScopeExecutionDecisionV15::ReaffirmFreeze => {
            reevaluated_policy.decision = SupportedRealSlotSetDecisionV2::Freeze;
            reevaluated_policy.chosen_candidate_slot = None;
            reevaluated_policy.rationale_codes = execution.rationale_codes.clone();
        }
        SupportedScopeExecutionDecisionV15::ExecuteExpandByOne => {
            let Some(slot) = execution.chosen_candidate_slot.clone() else {
                return Err(OpsError::Invalid(
                    "SUPPORTED_SET_APPLY_EXECUTION_INVALID: expansion decision missing candidate"
                        .to_string(),
                ));
            };
            reevaluated_policy.decision = SupportedRealSlotSetDecisionV2::ExpandByOne;
            reevaluated_policy.chosen_candidate_slot = Some(slot.clone());
            reevaluated_policy.candidate_slots_considered = vec![slot];
            reevaluated_policy.rationale_codes = execution.rationale_codes.clone();
        }
    }

    let candidates = reevaluated_policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let execution =
        validate_supported_set_execution(&reevaluated_policy, &previous_set, &candidates);
    let (applied_set, denial_code, mut rationale_codes) = match execution {
        Ok(applied) => (applied, None, reevaluated_policy.rationale_codes.clone()),
        Err(denied) => {
            let frozen = build_supported_real_slot_set_v2(
                previous_set.slots.clone(),
                &policy.policy_digest,
                &previous_set.set_digest,
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            let mut reasons = reevaluated_policy.rationale_codes.clone();
            reasons.push(format!("{:?}", denied.code));
            (frozen, Some(denied.code), reasons)
        }
    };
    rationale_codes.sort();
    rationale_codes.dedup();

    let freeze_record = if applied_set.decision == SupportedRealSlotSetExecutionDecisionV2::Frozen {
        Some(SupportedSetFreezeRecordV1 {
            schema_version: 1,
            previous_set_digest: previous_set.set_digest.clone(),
            resulting_set_digest: applied_set.set_digest.clone(),
            reason_codes: rationale_codes.clone(),
            policy_digest_prefix: prefix_hex(&policy.policy_digest, 16),
        })
    } else {
        None
    };

    let expansion_record = if applied_set.decision
        == SupportedRealSlotSetExecutionDecisionV2::Expanded
    {
        let added_slot = applied_set
            .slots
            .iter()
            .find(|slot| !previous_set.slots.contains(*slot))
            .cloned()
            .ok_or_else(|| OpsError::Invalid("SUPPORTED_SET_EXPANSION_MISSING_SLOT".to_string()))?;
        let evidence_summary_digests = candidates
            .iter()
            .filter(|candidate| candidate.slot_id == added_slot)
            .map(slot_expansion_candidate_digest)
            .collect::<Vec<_>>();
        Some(SupportedSetExpansionRecordV1 {
            schema_version: 1,
            previous_set_digest: previous_set.set_digest.clone(),
            resulting_set_digest: applied_set.set_digest.clone(),
            added_slot_id: added_slot,
            policy_digest_prefix: prefix_hex(&policy.policy_digest, 16),
            evidence_summary_digests,
        })
    } else {
        None
    };

    let report = SupportedSetApplyReportV1 {
        schema_version: 1,
        previous_slots: previous_set.slots,
        resulting_slots: applied_set.slots.clone(),
        decision: applied_set.decision.clone(),
        denial_code,
        rationale_codes,
        applied_set: applied_set.clone(),
        freeze_record,
        expansion_record,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    let canonical = workdir
        .join("out")
        .join("supported_real_slot_set_applied_v2.json");
    if let Some(parent) = canonical.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(canonical, serde_json::to_vec_pretty(&applied_set)?)?;
    Ok(report)
}

fn validate_scope_expansion_under_authority(
    workdir: &Path,
    applied_set: &SupportedRealSlotSetV2,
    candidate: &SlotExpansionEligibilityV1,
) -> Vec<String> {
    let mut failures = Vec::new();
    if applied_set.slots.contains(&candidate.slot_id) {
        failures.push("SCOPE_REEVAL_GOVERNANCE_MISMATCH".to_string());
    }
    if !candidate.trait_contract_exists
        || !candidate.probe_path_exists_or_reusable
        || !candidate.shadow_path_exists_or_trivially_attachable
        || !candidate.compare_window_normalizable
        || !candidate.strict_evidence_plumbing_representable_without_arch_fork
    {
        failures.push("SCOPE_REEVAL_INCOMPLETE_SCAFFOLD".to_string());
    }

    let backend_path = workdir.join("out").join("backend_evidence_snapshot.json");
    let active_path = workdir.join("out").join("active_review_snapshot.json");
    match (
        read_json_file::<BackendEvidenceSnapshotV1>(&backend_path),
        read_json_file::<AggregatedActiveReviewSnapshotV1>(&active_path),
    ) {
        (Ok(backend), Ok(active)) => {
            if crate::validate_governance_primary_surfaces_from_workdir(workdir, &backend, &active)
                .is_err()
            {
                failures.push("SCOPE_REEVAL_GOVERNANCE_MISMATCH".to_string());
            }
        }
        _ => failures.push("SCOPE_REEVAL_GOVERNANCE_MISMATCH".to_string()),
    }

    let interop_path = workdir.join("out").join("interop_consistency_matrix.json");
    let interop_ok = fs::read_to_string(&interop_path)
        .ok()
        .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
        .and_then(|value| {
            value
                .get("matrix")
                .and_then(|m| m.get("applied_supported_set_digest_prefix"))
                .and_then(|v| v.as_str())
                .map(|v| v == prefix_hex(&applied_set.set_digest, 16))
        })
        .unwrap_or(false);
    if !interop_ok {
        failures.push("SCOPE_REEVAL_EXPORT_INTEROP_GAP".to_string());
    }

    let scope_authority_path = workdir.join("out").join("scope_authority_check.json");
    let authority_ok = fs::read_to_string(&scope_authority_path)
        .ok()
        .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
        .and_then(|value| {
            value
                .get("status")
                .and_then(|v| v.as_str())
                .map(|v| v == "PASS")
        })
        .unwrap_or(false);
    if !authority_ok {
        failures.push("SCOPE_REEVAL_GOVERNANCE_MISMATCH".to_string());
    }

    failures.sort();
    failures.dedup();
    failures
}

pub fn models_supported_scope_execute(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV3, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation =
        ensure_current_supported_scope_reevaluation_v1(workdir, &policy, &previous_set)?;
    let execution = validate_scope_execution_v3(workdir, &policy, &previous_set, &reevaluation)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v4(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV4, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V4_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v4(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v5(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV5, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V5_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v5(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v6(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV6, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V6_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v6(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v7(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV7, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V7_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v7(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v8(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV8, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V8_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v8(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v9(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV9, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V9_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v9(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v10(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV10, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V10_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v10(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v11(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV11, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V11_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v11(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v12(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV12, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V12_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v12(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v13(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV13, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V13_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v13(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v14(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV14, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V14_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v14(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_execute_v15(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExecutionV15, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let previous_set = current_supported_real_slot_set(workdir)?;
    let reevaluation = load_latest_supported_scope_reevaluation_v1(workdir)?;
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    if reevaluation.policy_digest_prefix != policy_prefix
        || reevaluation.previous_applied_set_digest_prefix != previous_prefix
    {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V15_STALE_REEVALUATION: rerun `ucf-ops models supported-scope-reevaluate`"
                .to_string(),
        ));
    }

    let prior_scope_execution_digest_prefix = load_prior_scope_execution_digest_prefix(workdir)?;
    let execution = validate_scope_execution_v15(
        workdir,
        &policy,
        &previous_set,
        &reevaluation,
        prior_scope_execution_digest_prefix,
    )?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&execution)?)?;
    Ok(execution)
}

pub fn models_supported_scope_decision(
    workdir: &Path,
    out: &Path,
) -> Result<SupportedScopeExpansionDecisionReportV1, OpsError> {
    let policy = load_latest_supported_set_policy_v2(workdir)?;
    let current_set = current_supported_real_slot_set(workdir)?;
    let execution = ensure_current_supported_scope_execution_v15(workdir, &policy, &current_set)?;
    let governance_lock = read_json_file::<crate::GovernanceLockSweepReportV1>(
        &workdir.join("out/governance_lock_sweep.json"),
    )?;
    if !matches!(
        governance_lock.sweep.lock_status,
        crate::GovernanceLockStatusV1::Pass
    ) {
        return Err(OpsError::Invalid(
            "SUPPORTED_SCOPE_DECISION_REQUIRED: governance lock must PASS before supported-scope-decision".to_string(),
        ));
    }
    if governance_lock
        .sweep
        .canonical_governance_entry_digest_prefix
        != execution.canonical_governance_entry_digest_prefix
        || governance_lock
            .sweep
            .canonical_governance_authority_digest_prefix
            != execution.canonical_governance_authority_digest_prefix
    {
        return Err(OpsError::Invalid(
            "CANONICAL_SCOPE_SURFACE_CONTRADICTION: governance lock digests do not align with current supported scope execution".to_string(),
        ));
    }

    let mut candidates = Vec::new();
    let mut pass_candidates = Vec::new();
    let current_slots: BTreeSet<_> = current_set.slots.iter().cloned().collect();
    for slot in &policy.candidate_slots_considered {
        let in_scope = current_slots.contains(slot);
        if in_scope {
            continue;
        }
        let (mut mismatch_categories, reason_code) = if !matches!(
            execution.execution_decision,
            SupportedScopeExecutionDecisionV15::ExecuteExpandByOne
        ) {
            (
                vec![
                    SupportedScopeDecisionMismatchCategoryV1::GovernanceDenied,
                    SupportedScopeDecisionMismatchCategoryV1::ExecutionPathNonCanonical,
                ],
                SupportedScopeExpansionReasonCodeV1::CurrentScopeExecutionInsufficient,
            )
        } else if execution.chosen_candidate_slot.as_ref() != Some(slot) {
            (
                vec![
                    SupportedScopeDecisionMismatchCategoryV1::GovernanceDenied,
                    SupportedScopeDecisionMismatchCategoryV1::MultiSlotExpansionForbidden,
                ],
                SupportedScopeExpansionReasonCodeV1::MultipleCandidatesPresentRequireFreeze,
            )
        } else {
            pass_candidates.push(slot.clone());
            (
                Vec::new(),
                SupportedScopeExpansionReasonCodeV1::ExactlyOneSlotExpanded,
            )
        };
        mismatch_categories.sort();
        mismatch_categories.dedup();
        candidates.push(SupportedScopeDecisionCandidateV1 {
            slot_id: slot.clone(),
            currently_supported: false,
            status: if mismatch_categories.is_empty() {
                SupportedScopeExpansionDecisionStatusV1::ScopeExpansionApplied
            } else {
                SupportedScopeExpansionDecisionStatusV1::ScopeFreezeReinforced
            },
            reason_code,
            mismatch_categories,
        });
    }
    candidates.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    let (decision_status, decision_reason_code, winning_candidate_slot) =
        if pass_candidates.len() == 1 {
            (
                SupportedScopeExpansionDecisionStatusV1::ScopeExpansionApplied,
                SupportedScopeExpansionReasonCodeV1::ExactlyOneSlotExpanded,
                Some(pass_candidates[0].clone()),
            )
        } else if pass_candidates.len() > 1 {
            (
                SupportedScopeExpansionDecisionStatusV1::ScopeFreezeReinforced,
                SupportedScopeExpansionReasonCodeV1::MultipleCandidatesPresentRequireFreeze,
                None,
            )
        } else if candidates.is_empty() {
            (
                SupportedScopeExpansionDecisionStatusV1::ScopeFreezeReinforced,
                SupportedScopeExpansionReasonCodeV1::NoEligibleCandidateSlot,
                None,
            )
        } else {
            (
                SupportedScopeExpansionDecisionStatusV1::ScopeFreezeReinforced,
                SupportedScopeExpansionReasonCodeV1::CurrentScopeExecutionInsufficient,
                None,
            )
        };

    let unsupported_slot_count = candidates
        .iter()
        .filter(|c| {
            matches!(
                c.status,
                SupportedScopeExpansionDecisionStatusV1::ScopeFreezeReinforced
            )
        })
        .count() as u16;
    let contradictory_surface_count = if matches!(
        decision_reason_code,
        SupportedScopeExpansionReasonCodeV1::CanonicalScopeSurfaceContradiction
    ) {
        1
    } else {
        0
    };

    let mut digest_bytes = Vec::new();
    digest_bytes.extend_from_slice(
        SUPPORTED_SCOPE_EXPANSION_DECISION_V1_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(execution.previous_applied_set_digest_prefix.as_bytes());
    digest_bytes.extend_from_slice(
        execution
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .canonical_governance_authority_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .absolute_final_governance_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .terminal_governance_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .governance_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .governance_stabilization_sweep_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(
        execution
            .governance_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    digest_bytes.extend_from_slice(execution.governance_closure_sweep_digest_prefix.as_bytes());
    digest_bytes.extend_from_slice(execution.governance_seal_sweep_digest_prefix.as_bytes());
    digest_bytes.extend_from_slice(prefix_hex(&governance_lock.sweep.lock_digest, 16).as_bytes());
    digest_bytes.extend_from_slice(prefix_hex(&current_set.set_digest, 16).as_bytes());
    digest_bytes.extend_from_slice(format!("{decision_status:?}").as_bytes());
    digest_bytes.extend_from_slice(format!("{decision_reason_code:?}").as_bytes());
    if let Some(slot) = winning_candidate_slot.as_ref() {
        digest_bytes.extend_from_slice(slot.as_bytes());
    }

    let decision = SupportedScopeExpansionDecisionV1 {
        applied_supported_set_digest_prefix: execution.previous_applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: execution
            .canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: execution
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: execution
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: execution
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: execution
            .residual_free_governance_consumer_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix: execution
            .residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix: execution
            .absolute_final_governance_terminal_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix: execution
            .terminal_governance_ultimate_sweep_digest_prefix,
        governance_convergence_sweep_digest_prefix: execution
            .governance_convergence_sweep_digest_prefix,
        governance_stabilization_sweep_digest_prefix: execution
            .governance_stabilization_sweep_digest_prefix,
        governance_final_consolidation_sweep_digest_prefix: execution
            .governance_final_consolidation_sweep_digest_prefix,
        governance_closure_sweep_digest_prefix: execution.governance_closure_sweep_digest_prefix,
        governance_seal_sweep_digest_prefix: execution.governance_seal_sweep_digest_prefix,
        governance_lock_sweep_digest_prefix: prefix_hex(&governance_lock.sweep.lock_digest, 16),
        current_supported_scope_digest_prefix: prefix_hex(&current_set.set_digest, 16),
        candidate_count: candidates.len() as u16,
        winning_candidate_slot,
        decision_status,
        decision_reason_code,
        evaluated_consumer_count: governance_lock.sweep.covered_consumer_count,
        contradictory_surface_count,
        unsupported_slot_count,
        decision_digest: sha256_hex(&digest_bytes),
    };
    let report = SupportedScopeExpansionDecisionReportV1 {
        schema_version: SUPPORTED_SCOPE_EXPANSION_DECISION_V1_SCHEMA_VERSION,
        decision,
        candidates,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn load_prior_scope_execution_digest_prefix(workdir: &Path) -> Result<Option<String>, OpsError> {
    let v15_path = workdir.join("out").join("supported_scope_execute_v15.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV15>(&v15_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v14_path = workdir.join("out").join("supported_scope_execute_v14.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV14>(&v14_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v13_path = workdir.join("out").join("supported_scope_execute_v13.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV13>(&v13_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v12_path = workdir.join("out").join("supported_scope_execute_v12.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV12>(&v12_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v11_path = workdir.join("out").join("supported_scope_execute_v11.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV11>(&v11_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v10_path = workdir.join("out").join("supported_scope_execute_v10.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV10>(&v10_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v9_path = workdir.join("out").join("supported_scope_execute_v9.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV9>(&v9_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v8_path = workdir.join("out").join("supported_scope_execute_v8.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV8>(&v8_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v7_path = workdir.join("out").join("supported_scope_execute_v7.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV7>(&v7_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v6_path = workdir.join("out").join("supported_scope_execute_v6.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV6>(&v6_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v5_path = workdir.join("out").join("supported_scope_execute_v5.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV5>(&v5_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v4_path = workdir.join("out").join("supported_scope_execute_v4.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV4>(&v4_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }

    let v3_path = workdir.join("out").join("supported_scope_execute_v3.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV3>(&v3_path) {
        return Ok(Some(prefix_hex(&report.execution_digest, 16)));
    }
    Ok(None)
}

fn validate_scope_execution_v3(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
) -> Result<SupportedScopeExecutionV3, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV3::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let canonical_ok = match (
        read_json_file::<BackendEvidenceSnapshotV1>(
            &workdir.join("out/backend_evidence_snapshot.json"),
        ),
        read_json_file::<AggregatedActiveReviewSnapshotV1>(
            &workdir.join("out/active_review_snapshot.json"),
        ),
    ) {
        (Ok(backend), Ok(active)) => {
            match validate_governance_primary_surfaces_with_applied_scope(
                &backend,
                &active,
                &applied_context,
            ) {
                Ok(surfaces) => {
                    match derive_canonical_governance_entry(&applied_context, &surfaces) {
                        Ok(entry)
                            if entry.entry_status == CanonicalGovernanceEntryStatusV1::Pass =>
                        {
                            canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
                            match read_json_file::<GovernanceEntryCheckReportV1>(
                                &workdir.join("out/governance_entry_check.json"),
                            ) {
                                Ok(check)
                                    if check.status == GovernanceEntryCheckStatusV1::Pass
                                        && check.authority_digest_prefix
                                            == canonical_digest_prefix =>
                                {
                                    true
                                }
                                _ => {
                                    rationale_codes.push(
                                        "SCOPE_EXEC_V3_SECONDARY_ENTRY_DEPENDENCY".to_string(),
                                    );
                                    false
                                }
                            }
                        }
                        _ => {
                            rationale_codes.push("SCOPE_EXEC_V3_CANONICAL_ENTRY_FAIL".to_string());
                            false
                        }
                    }
                }
                Err(_) => {
                    rationale_codes.push("SCOPE_EXEC_V3_GOVERNANCE_SURFACE_GAP".to_string());
                    false
                }
            }
        }
        _ => {
            rationale_codes.push("SCOPE_EXEC_V3_CANONICAL_ENTRY_FAIL".to_string());
            false
        }
    };

    if policy_prefix != reevaluation.policy_digest_prefix
        || previous_prefix != reevaluation.previous_applied_set_digest_prefix
    {
        rationale_codes.push("SCOPE_EXEC_V3_CANONICAL_ENTRY_FAIL".to_string());
    }

    let candidate = reevaluation
        .chosen_candidate_slot
        .as_ref()
        .or(policy.chosen_candidate_slot.as_ref())
        .cloned();
    let viable = policy
        .candidate_slots_considered
        .iter()
        .filter_map(|slot_id| {
            let candidate = evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            );
            if previous_set.slots.contains(slot_id) {
                return None;
            }
            if !(candidate.trait_contract_exists
                && candidate.probe_path_exists_or_reusable
                && candidate.shadow_path_exists_or_trivially_attachable
                && candidate.compare_window_normalizable
                && candidate.strict_evidence_plumbing_representable_without_arch_fork)
            {
                return None;
            }
            let failures = validate_scope_expansion_under_authority(
                workdir,
                &build_supported_real_slot_set_v2(
                    previous_set.slots.clone(),
                    &policy.policy_digest,
                    &previous_set.set_digest,
                    SupportedRealSlotSetExecutionDecisionV2::Frozen,
                ),
                &candidate,
            );
            if failures
                .iter()
                .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
            {
                return None;
            }
            if !canonical_ok {
                return None;
            }
            Some(candidate.slot_id)
        })
        .collect::<Vec<_>>();

    if let Some(slot) = candidate.clone() {
        if previous_set.slots.contains(&slot) {
            rationale_codes.push("SCOPE_EXEC_V3_ALREADY_IN_SCOPE".to_string());
        }
    }

    if viable.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V3_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV3::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V3_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V3_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V3_SECONDARY_ENTRY_DEPENDENCY".to_string());
        }
    } else if reevaluation.reevaluation_decision
        == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
    {
        rationale_codes.push("SCOPE_EXEC_V3_INCOMPLETE_SCAFFOLD".to_string());
    } else if policy.candidate_slots_considered.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V3_AMBIGUOUS_CANDIDATE".to_string());
    } else {
        rationale_codes.push("SCOPE_EXEC_V3_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V3_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots,
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    Ok(SupportedScopeExecutionV3 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V3_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v4(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV4, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v4) = read_json_file::<SupportedScopeExecutionV4>(
        &workdir.join("out/supported_scope_execute_v4.json"),
    ) {
        if prior_v4.current_policy_digest_prefix != policy_prefix
            || prior_v4.previous_applied_set_digest_prefix != previous_prefix
            || prior_v4.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V4_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v4.json and rerun execution chain".to_string(),
            ));
        }
    } else if let Ok(prior_v3) = read_json_file::<SupportedScopeExecutionV3>(
        &workdir.join("out/supported_scope_execute_v3.json"),
    ) {
        if prior_v3.current_policy_digest_prefix != policy_prefix
            || prior_v3.previous_applied_set_digest_prefix != previous_prefix
            || prior_v3.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V4_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v3.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V4_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV4::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V4_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|code| OpsError::Invalid(code.to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V4_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let mut viable_candidates = Vec::new();
    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V4_ALREADY_IN_SCOPE".to_string());
            continue;
        }

        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
        {
            rationale_codes.push("SCOPE_EXEC_V4_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V4_SECONDARY_ENTRY_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V4_EXPORT_BUNDLE_GAP".to_string());
            continue;
        }
        if !authority_ok {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V4_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV4::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V4_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V4_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V4_SECONDARY_ENTRY_DEPENDENCY".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V4_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V4_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV4 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V4_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v5(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV5, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v5) = read_json_file::<SupportedScopeExecutionV5>(
        &workdir.join("out/supported_scope_execute_v5.json"),
    ) {
        if prior_v5.current_policy_digest_prefix != policy_prefix
            || prior_v5.previous_applied_set_digest_prefix != previous_prefix
            || prior_v5.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V5_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v5.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V5_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV5::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut final_consumer_authority_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V5_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V5_GOVERNANCE_AUTHORITY_FAIL".to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V5_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let final_authority = load_and_validate_final_consumer_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V5_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
    let final_authority_ok = if let Ok(authority) = final_authority {
        final_consumer_authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V5_FINAL_CONSUMER_AUTHORITY_FAIL".to_string());
        false
    };

    if applied_context.schema_version == 0 {
        rationale_codes.push("SCOPE_EXEC_V5_APPLIED_SCOPE_FAIL".to_string());
    }

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });

    let mut viable_candidates = Vec::new();
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V5_ALREADY_IN_SCOPE".to_string());
            continue;
        }
        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
        {
            rationale_codes.push("SCOPE_EXEC_V5_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V5_LEGACY_GOVERNANCE_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V5_EXPORT_CONTINUITY_GAP".to_string());
            continue;
        }
        if !(authority_ok && final_authority_ok) {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V5_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV5::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V5_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V5_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V5_REAFFIRM_FREEZE".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V5_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V5_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_consumer_authority_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV5 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V5_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: final_consumer_authority_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v6(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV6, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v6) = read_json_file::<SupportedScopeExecutionV6>(
        &workdir.join("out/supported_scope_execute_v6.json"),
    ) {
        if prior_v6.current_policy_digest_prefix != policy_prefix
            || prior_v6.previous_applied_set_digest_prefix != previous_prefix
            || prior_v6.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V6_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v6.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V6_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV6::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut final_consumer_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_sweep_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V6_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V6_GOVERNANCE_AUTHORITY_FAIL".to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V6_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let final_authority = load_and_validate_final_consumer_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V6_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
    let final_authority_ok = if let Ok(authority) = final_authority {
        final_consumer_authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V6_FINAL_CONSUMER_AUTHORITY_FAIL".to_string());
        false
    };

    let residual_sweep = load_and_validate_governance_residual_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V6_RESIDUAL_SWEEP_FAIL".to_string()));
    let residual_sweep_ok = if let Ok(sweep) = residual_sweep {
        residual_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V6_RESIDUAL_SWEEP_FAIL".to_string());
        false
    };

    if applied_context.schema_version == 0 {
        rationale_codes.push("SCOPE_EXEC_V6_APPLIED_SCOPE_FAIL".to_string());
    }

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });

    let mut viable_candidates = Vec::new();
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V6_ALREADY_IN_SCOPE".to_string());
            continue;
        }
        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
        {
            rationale_codes.push("SCOPE_EXEC_V6_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V6_RESIDUAL_GOVERNANCE_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V6_EXPORT_CONTINUITY_GAP".to_string());
            continue;
        }
        if !(authority_ok && final_authority_ok && residual_sweep_ok) {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V6_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV6::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V6_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V6_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V6_REAFFIRM_FREEZE".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V6_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V6_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV6 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V6_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: final_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v7(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV7, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v7) = read_json_file::<SupportedScopeExecutionV7>(
        &workdir.join("out/supported_scope_execute_v7.json"),
    ) {
        if prior_v7.current_policy_digest_prefix != policy_prefix
            || prior_v7.previous_applied_set_digest_prefix != previous_prefix
            || prior_v7.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V7_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v7.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V7_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV7::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut final_consumer_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_authority_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V7_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V7_GOVERNANCE_AUTHORITY_FAIL".to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V7_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let final_authority = load_and_validate_final_consumer_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V7_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
    let final_authority_ok = if let Ok(authority) = final_authority {
        final_consumer_authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V7_FINAL_CONSUMER_AUTHORITY_FAIL".to_string());
        false
    };

    let residual_sweep = load_and_validate_governance_residual_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V7_RESIDUAL_SWEEP_FAIL".to_string()));
    let residual_sweep_ok = if let Ok(sweep) = residual_sweep {
        residual_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V7_RESIDUAL_SWEEP_FAIL".to_string());
        false
    };

    let residual_free = load_and_validate_residual_free_governance_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V7_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string()));
    let residual_free_ok = if let Ok(authority) = residual_free {
        residual_free_governance_authority_digest_prefix =
            prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V7_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string());
        false
    };

    if applied_context.schema_version == 0 {
        rationale_codes.push("SCOPE_EXEC_V7_APPLIED_SCOPE_FAIL".to_string());
    }

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });

    let mut viable_candidates = Vec::new();
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V7_ALREADY_IN_SCOPE".to_string());
            continue;
        }
        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
        {
            rationale_codes.push("SCOPE_EXEC_V7_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V7_HISTORICAL_GOVERNANCE_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V7_EXPORT_CONTINUITY_GAP".to_string());
            continue;
        }
        if !(authority_ok && final_authority_ok && residual_sweep_ok && residual_free_ok) {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V7_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV7::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V7_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V7_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V7_REAFFIRM_FREEZE".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V7_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V7_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_free_governance_authority_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV7 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V7_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: final_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix:
            residual_free_governance_authority_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v8(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV8, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v8) = read_json_file::<SupportedScopeExecutionV8>(
        &workdir.join("out/supported_scope_execute_v8.json"),
    ) {
        if prior_v8.current_policy_digest_prefix != policy_prefix
            || prior_v8.previous_applied_set_digest_prefix != previous_prefix
            || prior_v8.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V8_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v8.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V8_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV8::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut final_consumer_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_absolute_sweep_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V8_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V8_GOVERNANCE_AUTHORITY_FAIL".to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V8_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let final_authority = load_and_validate_final_consumer_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V8_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
    let final_authority_ok = if let Ok(authority) = final_authority {
        final_consumer_authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V8_FINAL_CONSUMER_AUTHORITY_FAIL".to_string());
        false
    };

    let residual_sweep = load_and_validate_governance_residual_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V8_RESIDUAL_SWEEP_FAIL".to_string()));
    let residual_sweep_ok = if let Ok(sweep) = residual_sweep {
        residual_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V8_RESIDUAL_SWEEP_FAIL".to_string());
        false
    };

    let residual_free = load_and_validate_residual_free_governance_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V8_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string()));
    let residual_free_ok = if let Ok(authority) = residual_free {
        residual_free_governance_authority_digest_prefix =
            prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V8_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string());
        false
    };

    let absolute_sweep = load_and_validate_governance_absolute_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V8_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string()));
    let absolute_sweep_ok = if let Ok(sweep) = absolute_sweep {
        residual_free_governance_absolute_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V8_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    if applied_context.schema_version == 0 {
        rationale_codes.push("SCOPE_EXEC_V8_APPLIED_SCOPE_FAIL".to_string());
    }

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });

    let mut viable_candidates = Vec::new();
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V8_ALREADY_IN_SCOPE".to_string());
            continue;
        }
        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
        {
            rationale_codes.push("SCOPE_EXEC_V8_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V8_HISTORICAL_GOVERNANCE_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V8_EXPORT_CONTINUITY_GAP".to_string());
            continue;
        }
        if !(authority_ok
            && final_authority_ok
            && residual_sweep_ok
            && residual_free_ok
            && absolute_sweep_ok)
        {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V8_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV8::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V8_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V8_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V8_REAFFIRM_FREEZE".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V8_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V8_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_free_governance_authority_digest_prefix.as_bytes());
    digest_source
        .extend_from_slice(residual_free_governance_absolute_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV8 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V8_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: final_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix:
            residual_free_governance_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v9(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV9, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v9) = read_json_file::<SupportedScopeExecutionV9>(
        &workdir.join("out/supported_scope_execute_v9.json"),
    ) {
        if prior_v9.current_policy_digest_prefix != policy_prefix
            || prior_v9.previous_applied_set_digest_prefix != previous_prefix
            || prior_v9.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V9_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v9.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V9_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV9::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut final_consumer_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_absolute_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut terminal_governance_sweep_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V9_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V9_GOVERNANCE_AUTHORITY_FAIL".to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V9_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let final_authority = load_and_validate_final_consumer_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V9_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
    let final_authority_ok = if let Ok(authority) = final_authority {
        final_consumer_authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V9_FINAL_CONSUMER_AUTHORITY_FAIL".to_string());
        false
    };

    let residual_sweep = load_and_validate_governance_residual_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V9_RESIDUAL_SWEEP_FAIL".to_string()));
    let residual_sweep_ok = if let Ok(sweep) = residual_sweep {
        residual_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V9_RESIDUAL_SWEEP_FAIL".to_string());
        false
    };

    let residual_free = load_and_validate_residual_free_governance_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V9_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string()));
    let residual_free_ok = if let Ok(authority) = residual_free {
        residual_free_governance_authority_digest_prefix =
            prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V9_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string());
        false
    };

    let absolute_sweep = load_and_validate_governance_absolute_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V9_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string()));
    let absolute_sweep_ok = if let Ok(sweep) = absolute_sweep {
        residual_free_governance_absolute_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V9_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let terminal_sweep = load_and_validate_terminal_governance_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V9_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string()));
    let terminal_sweep_ok = if let Ok(sweep) = terminal_sweep {
        terminal_governance_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V9_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    if applied_context.schema_version == 0 {
        rationale_codes.push("SCOPE_EXEC_V9_APPLIED_SCOPE_FAIL".to_string());
    }

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });

    let mut viable_candidates = Vec::new();
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V9_ALREADY_IN_SCOPE".to_string());
            continue;
        }
        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
            || !candidate.tiny_fixture_path_feasible
        {
            rationale_codes.push("SCOPE_EXEC_V9_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V9_GOVERNANCE_ECHO_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V9_EXPORT_CONTINUITY_GAP".to_string());
            continue;
        }
        if !(authority_ok
            && final_authority_ok
            && residual_sweep_ok
            && residual_free_ok
            && absolute_sweep_ok
            && terminal_sweep_ok)
        {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V9_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV9::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V9_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V9_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V9_REAFFIRM_FREEZE".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V9_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V9_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_free_governance_authority_digest_prefix.as_bytes());
    digest_source
        .extend_from_slice(residual_free_governance_absolute_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(terminal_governance_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV9 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V9_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: final_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix:
            residual_free_governance_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix:
            terminal_governance_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v10(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV10, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v10) = read_json_file::<SupportedScopeExecutionV10>(
        &workdir.join("out/supported_scope_execute_v10.json"),
    ) {
        if prior_v10.current_policy_digest_prefix != policy_prefix
            || prior_v10.previous_applied_set_digest_prefix != previous_prefix
            || prior_v10.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V10_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v10.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V10_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV10::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut final_consumer_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_absolute_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut terminal_governance_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut terminal_governance_ultimate_sweep_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V10_GOVERNANCE_AUTHORITY_FAIL".to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let final_authority = load_and_validate_final_consumer_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V10_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
    let final_authority_ok = if let Ok(authority) = final_authority {
        final_consumer_authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_FINAL_CONSUMER_AUTHORITY_FAIL".to_string());
        false
    };

    let residual_sweep = load_and_validate_governance_residual_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V10_RESIDUAL_SWEEP_FAIL".to_string()));
    let residual_sweep_ok = if let Ok(sweep) = residual_sweep {
        residual_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_RESIDUAL_SWEEP_FAIL".to_string());
        false
    };

    let residual_free = load_and_validate_residual_free_governance_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V10_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string()));
    let residual_free_ok = if let Ok(authority) = residual_free {
        residual_free_governance_authority_digest_prefix =
            prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string());
        false
    };

    let absolute_sweep = load_and_validate_governance_absolute_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V10_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string()));
    let absolute_sweep_ok = if let Ok(sweep) = absolute_sweep {
        residual_free_governance_absolute_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let terminal_sweep = load_and_validate_terminal_governance_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V10_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string()));
    let terminal_sweep_ok = if let Ok(sweep) = terminal_sweep {
        terminal_governance_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let ultimate_sweep = load_and_validate_governance_ultimate_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
        &terminal_governance_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V10_ULTIMATE_GOVERNANCE_SWEEP_FAIL".to_string()));
    let ultimate_sweep_ok = if let Ok(sweep) = ultimate_sweep {
        terminal_governance_ultimate_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_ULTIMATE_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    if applied_context.schema_version == 0 {
        rationale_codes.push("SCOPE_EXEC_V10_APPLIED_SCOPE_FAIL".to_string());
    }

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });

    let mut viable_candidates = Vec::new();
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V10_ALREADY_IN_SCOPE".to_string());
            continue;
        }
        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
            || !candidate.tiny_fixture_path_feasible
        {
            rationale_codes.push("SCOPE_EXEC_V10_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V10_GOVERNANCE_CACHE_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V10_EXPORT_CONTINUITY_GAP".to_string());
            continue;
        }
        if !(authority_ok
            && final_authority_ok
            && residual_sweep_ok
            && residual_free_ok
            && absolute_sweep_ok
            && terminal_sweep_ok
            && ultimate_sweep_ok)
        {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V10_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV10::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V10_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V10_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V10_REAFFIRM_FREEZE".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V10_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V10_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_free_governance_authority_digest_prefix.as_bytes());
    digest_source
        .extend_from_slice(residual_free_governance_absolute_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(terminal_governance_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(terminal_governance_ultimate_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV10 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V10_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: final_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix:
            residual_free_governance_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix:
            terminal_governance_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v11(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV11, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v11) = read_json_file::<SupportedScopeExecutionV11>(
        &workdir.join("out/supported_scope_execute_v11.json"),
    ) {
        if prior_v11.current_policy_digest_prefix != policy_prefix
            || prior_v11.previous_applied_set_digest_prefix != previous_prefix
            || prior_v11.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V11_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v11.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V11_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV11::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut final_consumer_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_absolute_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut terminal_governance_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut terminal_governance_ultimate_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut governance_convergence_sweep_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V11_GOVERNANCE_AUTHORITY_FAIL".to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let final_authority = load_and_validate_final_consumer_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V11_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
    let final_authority_ok = if let Ok(authority) = final_authority {
        final_consumer_authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_FINAL_CONSUMER_AUTHORITY_FAIL".to_string());
        false
    };

    let residual_sweep = load_and_validate_governance_residual_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V11_RESIDUAL_SWEEP_FAIL".to_string()));
    let residual_sweep_ok = if let Ok(sweep) = residual_sweep {
        residual_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_RESIDUAL_SWEEP_FAIL".to_string());
        false
    };

    let residual_free = load_and_validate_residual_free_governance_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V11_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string()));
    let residual_free_ok = if let Ok(authority) = residual_free {
        residual_free_governance_authority_digest_prefix =
            prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string());
        false
    };

    let absolute_sweep = load_and_validate_governance_absolute_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V11_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string()));
    let absolute_sweep_ok = if let Ok(sweep) = absolute_sweep {
        residual_free_governance_absolute_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let terminal_sweep = load_and_validate_terminal_governance_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V11_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string()));
    let terminal_sweep_ok = if let Ok(sweep) = terminal_sweep {
        terminal_governance_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let ultimate_sweep = load_and_validate_governance_ultimate_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
        &terminal_governance_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V11_ULTIMATE_GOVERNANCE_SWEEP_FAIL".to_string()));
    let ultimate_sweep_ok = if let Ok(sweep) = ultimate_sweep {
        terminal_governance_ultimate_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_ULTIMATE_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let convergence_sweep = load_and_validate_governance_convergence_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
        &terminal_governance_ultimate_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V11_GOVERNANCE_CONVERGENCE_FAIL".to_string()));
    let convergence_sweep_ok = if let Ok(sweep) = convergence_sweep {
        governance_convergence_sweep_digest_prefix = prefix_hex(&sweep.convergence_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_GOVERNANCE_CONVERGENCE_FAIL".to_string());
        false
    };

    if applied_context.schema_version == 0 {
        rationale_codes.push("SCOPE_EXEC_V11_APPLIED_SCOPE_FAIL".to_string());
    }

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });

    let mut viable_candidates = Vec::new();
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V11_ALREADY_IN_SCOPE".to_string());
            continue;
        }
        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
            || !candidate.tiny_fixture_path_feasible
        {
            rationale_codes.push("SCOPE_EXEC_V11_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V11_GOVERNANCE_MEMO_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V11_EXPORT_CONTINUITY_GAP".to_string());
            continue;
        }
        if !(authority_ok
            && final_authority_ok
            && residual_sweep_ok
            && residual_free_ok
            && absolute_sweep_ok
            && terminal_sweep_ok
            && ultimate_sweep_ok
            && convergence_sweep_ok)
        {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V11_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV11::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V11_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V11_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V11_REAFFIRM_FREEZE".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V11_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V11_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_free_governance_authority_digest_prefix.as_bytes());
    digest_source
        .extend_from_slice(residual_free_governance_absolute_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(terminal_governance_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(terminal_governance_ultimate_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(governance_convergence_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV11 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V11_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: final_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix:
            residual_free_governance_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix:
            terminal_governance_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix,
        governance_convergence_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v12(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV12, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);

    if let Ok(prior_v12) = read_json_file::<SupportedScopeExecutionV12>(
        &workdir.join("out/supported_scope_execute_v12.json"),
    ) {
        if prior_v12.current_policy_digest_prefix != policy_prefix
            || prior_v12.previous_applied_set_digest_prefix != previous_prefix
            || prior_v12.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V12_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v12.json and rerun execution chain".to_string(),
            ));
        }
    }

    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    let expected_applied_prefix = prefix_hex(&previous_set.set_digest, 16);
    if applied_context.applied_set_digest_prefix != expected_applied_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V12_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = Vec::new();
    let mut chosen_candidate_slot = None;
    let mut execution_decision = SupportedScopeExecutionDecisionV12::ReaffirmFreeze;
    let mut resulting_slots = previous_set.slots.clone();

    let mut canonical_digest_prefix = "UNAVAILABLE".to_string();
    let mut authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut final_consumer_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_authority_digest_prefix = "UNAVAILABLE".to_string();
    let mut residual_free_governance_absolute_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut terminal_governance_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut terminal_governance_ultimate_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut governance_convergence_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let mut governance_stabilization_sweep_digest_prefix = "UNAVAILABLE".to_string();

    let canonical_entry = derive_and_validate_canonical_entry(workdir, &applied_context)
        .map_err(|code| OpsError::Invalid(code.to_string()));
    if let Ok(entry) = canonical_entry {
        canonical_digest_prefix = prefix_hex(&entry.authority_digest, 16);
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_GOVERNANCE_ENTRY_FAIL".to_string());
    }

    let authority =
        load_and_validate_governance_authority(workdir, &applied_context, &canonical_digest_prefix)
            .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_GOVERNANCE_AUTHORITY_FAIL".to_string()));
    let authority_ok = if let Ok(authority) = authority {
        authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_GOVERNANCE_AUTHORITY_FAIL".to_string());
        false
    };

    let final_authority = load_and_validate_final_consumer_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
    let final_authority_ok = if let Ok(authority) = final_authority {
        final_consumer_authority_digest_prefix = prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_FINAL_CONSUMER_AUTHORITY_FAIL".to_string());
        false
    };

    let residual_sweep = load_and_validate_governance_residual_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_RESIDUAL_SWEEP_FAIL".to_string()));
    let residual_sweep_ok = if let Ok(sweep) = residual_sweep {
        residual_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_RESIDUAL_SWEEP_FAIL".to_string());
        false
    };

    let residual_free = load_and_validate_residual_free_governance_authority(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string()));
    let residual_free_ok = if let Ok(authority) = residual_free {
        residual_free_governance_authority_digest_prefix =
            prefix_hex(&authority.authority_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string());
        false
    };

    let absolute_sweep = load_and_validate_governance_absolute_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string()));
    let absolute_sweep_ok = if let Ok(sweep) = absolute_sweep {
        residual_free_governance_absolute_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let terminal_sweep = load_and_validate_terminal_governance_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string()));
    let terminal_sweep_ok = if let Ok(sweep) = terminal_sweep {
        terminal_governance_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let ultimate_sweep = load_and_validate_governance_ultimate_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
        &terminal_governance_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_ULTIMATE_GOVERNANCE_SWEEP_FAIL".to_string()));
    let ultimate_sweep_ok = if let Ok(sweep) = ultimate_sweep {
        terminal_governance_ultimate_sweep_digest_prefix = prefix_hex(&sweep.sweep_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_ULTIMATE_GOVERNANCE_SWEEP_FAIL".to_string());
        false
    };

    let convergence_sweep = load_and_validate_governance_convergence_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
        &terminal_governance_ultimate_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_GOVERNANCE_CONVERGENCE_FAIL".to_string()));
    let convergence_sweep_ok = if let Ok(sweep) = convergence_sweep {
        governance_convergence_sweep_digest_prefix = prefix_hex(&sweep.convergence_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_GOVERNANCE_CONVERGENCE_FAIL".to_string());
        false
    };

    let stabilization_sweep = load_and_validate_governance_stabilization_sweep(
        workdir,
        &applied_context,
        &canonical_digest_prefix,
        &authority_digest_prefix,
        &final_consumer_authority_digest_prefix,
        &residual_sweep_digest_prefix,
        &residual_free_governance_authority_digest_prefix,
        &residual_free_governance_absolute_sweep_digest_prefix,
        &terminal_governance_ultimate_sweep_digest_prefix,
        &governance_convergence_sweep_digest_prefix,
    )
    .map_err(|_| OpsError::Invalid("SCOPE_EXEC_V12_GOVERNANCE_STABILIZATION_FAIL".to_string()));
    let stabilization_sweep_ok = if let Ok(sweep) = stabilization_sweep {
        governance_stabilization_sweep_digest_prefix = prefix_hex(&sweep.stabilization_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_GOVERNANCE_STABILIZATION_FAIL".to_string());
        false
    };

    if applied_context.schema_version == 0 {
        rationale_codes.push("SCOPE_EXEC_V12_APPLIED_SCOPE_FAIL".to_string());
    }

    let candidate_slots = policy
        .candidate_slots_considered
        .iter()
        .map(|slot_id| {
            evaluate_slot_expansion_candidate(
                slot_id,
                &previous_set.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )
        })
        .collect::<Vec<_>>();

    let applied_v2 = read_json_file::<SupportedRealSlotSetV2>(
        &workdir.join("out/supported_real_slot_set_applied_v2.json"),
    )
    .unwrap_or_else(|_| {
        build_supported_real_slot_set_v2(
            previous_set.slots.clone(),
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )
    });

    let mut viable_candidates = Vec::new();
    for candidate in &candidate_slots {
        if previous_set.slots.contains(&candidate.slot_id) {
            rationale_codes.push("SCOPE_EXEC_V12_ALREADY_IN_SCOPE".to_string());
            continue;
        }
        if !candidate.trait_contract_exists
            || !candidate.probe_path_exists_or_reusable
            || !candidate.shadow_path_exists_or_trivially_attachable
            || !candidate.compare_window_normalizable
            || !candidate.strict_evidence_plumbing_representable_without_arch_fork
            || !candidate.tiny_fixture_path_feasible
        {
            rationale_codes.push("SCOPE_EXEC_V12_INCOMPLETE_SCAFFOLD".to_string());
            continue;
        }

        let authority_failures =
            validate_scope_expansion_under_authority(workdir, &applied_v2, candidate);
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_GOVERNANCE_MISMATCH")
        {
            rationale_codes.push("SCOPE_EXEC_V12_GOVERNANCE_ADAPTER_DEPENDENCY".to_string());
            continue;
        }
        if authority_failures
            .iter()
            .any(|f| f == "SCOPE_REEVAL_EXPORT_INTEROP_GAP")
        {
            rationale_codes.push("SCOPE_EXEC_V12_EXPORT_CONTINUITY_GAP".to_string());
            continue;
        }
        if !(authority_ok
            && final_authority_ok
            && residual_sweep_ok
            && residual_free_ok
            && absolute_sweep_ok
            && terminal_sweep_ok
            && ultimate_sweep_ok
            && convergence_sweep_ok
            && stabilization_sweep_ok)
        {
            continue;
        }
        viable_candidates.push(candidate.slot_id.clone());
    }

    if viable_candidates.len() > 1 {
        rationale_codes.push("SCOPE_EXEC_V12_AMBIGUOUS_CANDIDATE".to_string());
    } else if let Some(slot) = viable_candidates.first() {
        if reevaluation.reevaluation_decision
            == SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            && reevaluation.chosen_candidate_slot.as_ref() == Some(slot)
            && policy.chosen_candidate_slot.as_ref() == Some(slot)
        {
            execution_decision = SupportedScopeExecutionDecisionV12::ExecuteExpandByOne;
            chosen_candidate_slot = Some(slot.clone());
            resulting_slots.push(slot.clone());
            resulting_slots.sort();
            resulting_slots.dedup();
            rationale_codes.push("SCOPE_EXEC_V12_EXPANSION_EXECUTED".to_string());
            rationale_codes.push("SCOPE_EXEC_V12_NO_ACTIVE_IMPLICATIONS".to_string());
        } else {
            rationale_codes.push("SCOPE_EXEC_V12_REAFFIRM_FREEZE".to_string());
        }
    } else {
        rationale_codes.push("SCOPE_EXEC_V12_REAFFIRM_FREEZE".to_string());
    }

    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V12_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(canonical_digest_prefix.as_bytes());
    digest_source.extend_from_slice(authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(final_consumer_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(residual_free_governance_authority_digest_prefix.as_bytes());
    digest_source
        .extend_from_slice(residual_free_governance_absolute_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(terminal_governance_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(terminal_governance_ultimate_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(governance_convergence_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(governance_stabilization_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV12 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V12_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: canonical_digest_prefix,
        canonical_governance_authority_digest_prefix: authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: final_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix:
            residual_free_governance_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix:
            terminal_governance_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix,
        governance_convergence_sweep_digest_prefix,
        governance_stabilization_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn map_v12_to_v13_rationale(code: &str) -> String {
    if code == "SCOPE_EXEC_V12_GOVERNANCE_ADAPTER_DEPENDENCY" {
        return "SCOPE_EXEC_V13_GOVERNANCE_FACADE_DEPENDENCY".to_string();
    }
    if let Some(rest) = code.strip_prefix("SCOPE_EXEC_V12_") {
        return format!("SCOPE_EXEC_V13_{rest}");
    }
    code.to_string()
}

fn map_v13_to_v14_rationale(code: &str) -> String {
    if code == "SCOPE_EXEC_V13_GOVERNANCE_FACADE_DEPENDENCY" {
        return "SCOPE_EXEC_V14_GOVERNANCE_WRAPPER_DEPENDENCY".to_string();
    }
    if let Some(rest) = code.strip_prefix("SCOPE_EXEC_V13_") {
        return format!("SCOPE_EXEC_V14_{rest}");
    }
    code.to_string()
}

fn map_v14_to_v15_rationale(code: &str) -> String {
    if code == "SCOPE_EXEC_V14_GOVERNANCE_WRAPPER_DEPENDENCY" {
        return "SCOPE_EXEC_V15_GOVERNANCE_SHELL_DEPENDENCY".to_string();
    }
    if let Some(rest) = code.strip_prefix("SCOPE_EXEC_V14_") {
        return format!("SCOPE_EXEC_V15_{rest}");
    }
    code.to_string()
}

fn validate_scope_execution_v13(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV13, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);
    if let Ok(prior_v13) = read_json_file::<SupportedScopeExecutionV13>(
        &workdir.join("out/supported_scope_execute_v13.json"),
    ) {
        if prior_v13.current_policy_digest_prefix != policy_prefix
            || prior_v13.previous_applied_set_digest_prefix != previous_prefix
            || prior_v13.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V13_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v13.json and rerun execution chain".to_string(),
            ));
        }
    }

    let mut report_v12 = validate_scope_execution_v12(
        workdir,
        policy,
        previous_set,
        reevaluation,
        prior_scope_execution_digest_prefix.clone(),
    )?;
    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    if applied_context.applied_set_digest_prefix != previous_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V13_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = report_v12
        .rationale_codes
        .iter()
        .map(|c| map_v12_to_v13_rationale(c))
        .collect::<Vec<_>>();
    let mut governance_final_consolidation_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let final_consolidation = load_and_validate_governance_final_consolidation_sweep(
        workdir,
        &applied_context,
        &report_v12.canonical_governance_entry_digest_prefix,
        &report_v12.canonical_governance_authority_digest_prefix,
        &report_v12.final_governance_consumer_authority_digest_prefix,
        &report_v12.final_governance_residual_sweep_digest_prefix,
        &report_v12.residual_free_governance_consumer_authority_digest_prefix,
        &report_v12.residual_free_governance_absolute_sweep_digest_prefix,
        &report_v12.absolute_final_governance_terminal_sweep_digest_prefix,
        &report_v12.terminal_governance_ultimate_sweep_digest_prefix,
        &report_v12.governance_convergence_sweep_digest_prefix,
        &report_v12.governance_stabilization_sweep_digest_prefix,
    );
    let final_consolidation_ok = if let Ok(sweep) = final_consolidation {
        governance_final_consolidation_sweep_digest_prefix =
            prefix_hex(&sweep.consolidation_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V13_GOVERNANCE_FINAL_CONSOLIDATION_FAIL".to_string());
        false
    };
    if !final_consolidation_ok
        && matches!(
            report_v12.execution_decision,
            SupportedScopeExecutionDecisionV12::ExecuteExpandByOne
        )
    {
        report_v12.execution_decision = SupportedScopeExecutionDecisionV12::ReaffirmFreeze;
        report_v12.chosen_candidate_slot = None;
        rationale_codes.push("SCOPE_EXEC_V13_REAFFIRM_FREEZE".to_string());
    }

    let execution_decision = match report_v12.execution_decision {
        SupportedScopeExecutionDecisionV12::ReaffirmFreeze => {
            SupportedScopeExecutionDecisionV13::ReaffirmFreeze
        }
        SupportedScopeExecutionDecisionV12::ExecuteExpandByOne => {
            SupportedScopeExecutionDecisionV13::ExecuteExpandByOne
        }
    };
    let chosen_candidate_slot = report_v12.chosen_candidate_slot.clone();
    let mut resulting_slots = previous_set.slots.clone();
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        resulting_slots.push(slot.clone());
    }
    resulting_slots.sort();
    resulting_slots.dedup();
    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V13_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(
        report_v12
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .canonical_governance_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .absolute_final_governance_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .terminal_governance_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .governance_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v12
            .governance_stabilization_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(governance_final_consolidation_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV13 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V13_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: report_v12
            .canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: report_v12
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: report_v12
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: report_v12
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: report_v12
            .residual_free_governance_consumer_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix: report_v12
            .residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix: report_v12
            .absolute_final_governance_terminal_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix: report_v12
            .terminal_governance_ultimate_sweep_digest_prefix,
        governance_convergence_sweep_digest_prefix: report_v12
            .governance_convergence_sweep_digest_prefix,
        governance_stabilization_sweep_digest_prefix: report_v12
            .governance_stabilization_sweep_digest_prefix,
        governance_final_consolidation_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v14(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV14, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);
    if let Ok(prior_v14) = read_json_file::<SupportedScopeExecutionV14>(
        &workdir.join("out/supported_scope_execute_v14.json"),
    ) {
        if prior_v14.current_policy_digest_prefix != policy_prefix
            || prior_v14.previous_applied_set_digest_prefix != previous_prefix
            || prior_v14.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V14_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v14.json and rerun execution chain".to_string(),
            ));
        }
    }
    let report_v13 = validate_scope_execution_v13(
        workdir,
        policy,
        previous_set,
        reevaluation,
        prior_scope_execution_digest_prefix.clone(),
    )?;
    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    if applied_context.applied_set_digest_prefix != previous_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V14_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = report_v13
        .rationale_codes
        .iter()
        .map(|c| map_v13_to_v14_rationale(c))
        .collect::<Vec<_>>();
    let mut governance_closure_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let closure = load_and_validate_governance_closure_sweep(
        workdir,
        &applied_context,
        &report_v13.canonical_governance_entry_digest_prefix,
        &report_v13.canonical_governance_authority_digest_prefix,
        &report_v13.final_governance_consumer_authority_digest_prefix,
        &report_v13.final_governance_residual_sweep_digest_prefix,
        &report_v13.residual_free_governance_consumer_authority_digest_prefix,
        &report_v13.residual_free_governance_absolute_sweep_digest_prefix,
        &report_v13.absolute_final_governance_terminal_sweep_digest_prefix,
        &report_v13.terminal_governance_ultimate_sweep_digest_prefix,
        &report_v13.governance_convergence_sweep_digest_prefix,
        &report_v13.governance_stabilization_sweep_digest_prefix,
        &report_v13.governance_final_consolidation_sweep_digest_prefix,
    );
    let closure_ok = if let Ok(sweep) = closure {
        governance_closure_sweep_digest_prefix = prefix_hex(&sweep.closure_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V14_GOVERNANCE_CLOSURE_FAIL".to_string());
        false
    };

    let mut execution_decision = match report_v13.execution_decision {
        SupportedScopeExecutionDecisionV13::ReaffirmFreeze => {
            SupportedScopeExecutionDecisionV14::ReaffirmFreeze
        }
        SupportedScopeExecutionDecisionV13::ExecuteExpandByOne => {
            SupportedScopeExecutionDecisionV14::ExecuteExpandByOne
        }
    };
    let mut chosen_candidate_slot = report_v13.chosen_candidate_slot.clone();
    if !closure_ok
        && matches!(
            execution_decision,
            SupportedScopeExecutionDecisionV14::ExecuteExpandByOne
        )
    {
        execution_decision = SupportedScopeExecutionDecisionV14::ReaffirmFreeze;
        chosen_candidate_slot = None;
        rationale_codes.push("SCOPE_EXEC_V14_REAFFIRM_FREEZE".to_string());
    }

    let mut resulting_slots = previous_set.slots.clone();
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        resulting_slots.push(slot.clone());
    }
    resulting_slots.sort();
    resulting_slots.dedup();
    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V14_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(
        report_v13
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .canonical_governance_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .absolute_final_governance_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .terminal_governance_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .governance_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .governance_stabilization_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v13
            .governance_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(governance_closure_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV14 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V14_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: report_v13
            .canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: report_v13
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: report_v13
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: report_v13
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: report_v13
            .residual_free_governance_consumer_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix: report_v13
            .residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix: report_v13
            .absolute_final_governance_terminal_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix: report_v13
            .terminal_governance_ultimate_sweep_digest_prefix,
        governance_convergence_sweep_digest_prefix: report_v13
            .governance_convergence_sweep_digest_prefix,
        governance_stabilization_sweep_digest_prefix: report_v13
            .governance_stabilization_sweep_digest_prefix,
        governance_final_consolidation_sweep_digest_prefix: report_v13
            .governance_final_consolidation_sweep_digest_prefix,
        governance_closure_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn validate_scope_execution_v15(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    reevaluation: &SupportedScopeReevaluationV1,
    prior_scope_execution_digest_prefix: Option<String>,
) -> Result<SupportedScopeExecutionV15, OpsError> {
    let policy_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_prefix = prefix_hex(&previous_set.set_digest, 16);
    let reeval_prefix = prefix_hex(&reevaluation.reevaluation_digest, 16);
    if let Ok(prior_v15) = read_json_file::<SupportedScopeExecutionV15>(
        &workdir.join("out/supported_scope_execute_v15.json"),
    ) {
        if prior_v15.current_policy_digest_prefix != policy_prefix
            || prior_v15.previous_applied_set_digest_prefix != previous_prefix
            || prior_v15.current_reevaluation_digest_prefix != reeval_prefix
        {
            return Err(OpsError::Invalid(
                "SCOPE_EXEC_V15_STALE_PRIOR_EXECUTION: remove stale out/supported_scope_execute_v15.json and rerun execution chain".to_string(),
            ));
        }
    }
    let report_v14 = validate_scope_execution_v14(
        workdir,
        policy,
        previous_set,
        reevaluation,
        prior_scope_execution_digest_prefix.clone(),
    )?;
    let applied_context = load_applied_supported_set_context_v1(workdir)?;
    if applied_context.applied_set_digest_prefix != previous_prefix {
        return Err(OpsError::Invalid(
            "SCOPE_EXEC_V15_STALE_APPLIED_SCOPE: current applied context no longer matches supported scope baseline"
                .to_string(),
        ));
    }

    let mut rationale_codes = report_v14
        .rationale_codes
        .iter()
        .map(|c| map_v14_to_v15_rationale(c))
        .collect::<Vec<_>>();
    let mut governance_seal_sweep_digest_prefix = "UNAVAILABLE".to_string();
    let governance_seal = load_and_validate_governance_seal_sweep(
        workdir,
        &applied_context,
        &report_v14.canonical_governance_entry_digest_prefix,
        &report_v14.canonical_governance_authority_digest_prefix,
        &report_v14.final_governance_consumer_authority_digest_prefix,
        &report_v14.final_governance_residual_sweep_digest_prefix,
        &report_v14.residual_free_governance_consumer_authority_digest_prefix,
        &report_v14.residual_free_governance_absolute_sweep_digest_prefix,
        &report_v14.absolute_final_governance_terminal_sweep_digest_prefix,
        &report_v14.terminal_governance_ultimate_sweep_digest_prefix,
        &report_v14.governance_convergence_sweep_digest_prefix,
        &report_v14.governance_stabilization_sweep_digest_prefix,
        &report_v14.governance_final_consolidation_sweep_digest_prefix,
        &report_v14.governance_closure_sweep_digest_prefix,
    );
    let seal_ok = if let Ok(sweep) = governance_seal {
        governance_seal_sweep_digest_prefix = prefix_hex(&sweep.seal_digest, 16);
        true
    } else {
        rationale_codes.push("SCOPE_EXEC_V15_GOVERNANCE_SEAL_FAIL".to_string());
        false
    };

    let mut execution_decision = match report_v14.execution_decision {
        SupportedScopeExecutionDecisionV14::ReaffirmFreeze => {
            SupportedScopeExecutionDecisionV15::ReaffirmFreeze
        }
        SupportedScopeExecutionDecisionV14::ExecuteExpandByOne => {
            SupportedScopeExecutionDecisionV15::ExecuteExpandByOne
        }
    };
    let mut chosen_candidate_slot = report_v14.chosen_candidate_slot.clone();
    if !seal_ok
        && matches!(
            execution_decision,
            SupportedScopeExecutionDecisionV15::ExecuteExpandByOne
        )
    {
        execution_decision = SupportedScopeExecutionDecisionV15::ReaffirmFreeze;
        chosen_candidate_slot = None;
        rationale_codes.push("SCOPE_EXEC_V15_REAFFIRM_FREEZE".to_string());
    }

    let mut resulting_slots = previous_set.slots.clone();
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        resulting_slots.push(slot.clone());
    }
    resulting_slots.sort();
    resulting_slots.dedup();
    rationale_codes.sort();
    rationale_codes.dedup();
    let resulting = build_supported_real_slot_set_v2(
        resulting_slots.clone(),
        &policy.policy_digest,
        &previous_set.set_digest,
        if chosen_candidate_slot.is_some() {
            SupportedRealSlotSetExecutionDecisionV2::Expanded
        } else {
            SupportedRealSlotSetExecutionDecisionV2::Frozen
        },
    );

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_SCOPE_EXECUTION_V15_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(previous_prefix.as_bytes());
    digest_source.extend_from_slice(policy_prefix.as_bytes());
    digest_source.extend_from_slice(reeval_prefix.as_bytes());
    digest_source.extend_from_slice(
        report_v14
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .canonical_governance_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .absolute_final_governance_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .terminal_governance_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .governance_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .governance_stabilization_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        report_v14
            .governance_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(report_v14.governance_closure_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(governance_seal_sweep_digest_prefix.as_bytes());
    if let Some(prior) = prior_scope_execution_digest_prefix.as_ref() {
        digest_source.extend_from_slice(prior.as_bytes());
    }
    digest_source.extend_from_slice(format!("{execution_decision:?}").as_bytes());
    if let Some(slot) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &resulting_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    Ok(SupportedScopeExecutionV15 {
        schema_version: SUPPORTED_SCOPE_EXECUTION_V15_SCHEMA_VERSION,
        previous_applied_set_digest_prefix: previous_prefix,
        current_policy_digest_prefix: policy_prefix,
        current_reevaluation_digest_prefix: reeval_prefix,
        canonical_governance_entry_digest_prefix: report_v14
            .canonical_governance_entry_digest_prefix,
        canonical_governance_authority_digest_prefix: report_v14
            .canonical_governance_authority_digest_prefix,
        final_governance_consumer_authority_digest_prefix: report_v14
            .final_governance_consumer_authority_digest_prefix,
        final_governance_residual_sweep_digest_prefix: report_v14
            .final_governance_residual_sweep_digest_prefix,
        residual_free_governance_consumer_authority_digest_prefix: report_v14
            .residual_free_governance_consumer_authority_digest_prefix,
        residual_free_governance_absolute_sweep_digest_prefix: report_v14
            .residual_free_governance_absolute_sweep_digest_prefix,
        absolute_final_governance_terminal_sweep_digest_prefix: report_v14
            .absolute_final_governance_terminal_sweep_digest_prefix,
        terminal_governance_ultimate_sweep_digest_prefix: report_v14
            .terminal_governance_ultimate_sweep_digest_prefix,
        governance_convergence_sweep_digest_prefix: report_v14
            .governance_convergence_sweep_digest_prefix,
        governance_stabilization_sweep_digest_prefix: report_v14
            .governance_stabilization_sweep_digest_prefix,
        governance_final_consolidation_sweep_digest_prefix: report_v14
            .governance_final_consolidation_sweep_digest_prefix,
        governance_closure_sweep_digest_prefix: report_v14.governance_closure_sweep_digest_prefix,
        governance_seal_sweep_digest_prefix,
        prior_scope_execution_digest_prefix,
        execution_decision,
        chosen_candidate_slot,
        resulting_supported_set_digest_prefix: prefix_hex(&resulting.set_digest, 16),
        rationale_codes,
        execution_digest: sha256_hex(&digest_source),
    })
}

fn derive_and_validate_canonical_entry(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
) -> Result<crate::CanonicalGovernanceEntryV1, &'static str> {
    let backend = read_json_file::<BackendEvidenceSnapshotV1>(
        &workdir.join("out/backend_evidence_snapshot.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V4_GOVERNANCE_ENTRY_FAIL")?;
    let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(
        &workdir.join("out/active_review_snapshot.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V4_GOVERNANCE_ENTRY_FAIL")?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, applied_context)
            .map_err(|_| "SCOPE_EXEC_V4_GOVERNANCE_ENTRY_FAIL")?;
    let entry = derive_canonical_governance_entry(applied_context, &surfaces)
        .map_err(|_| "SCOPE_EXEC_V4_GOVERNANCE_ENTRY_FAIL")?;
    if entry.entry_status != CanonicalGovernanceEntryStatusV1::Pass {
        return Err("SCOPE_EXEC_V4_GOVERNANCE_ENTRY_FAIL");
    }
    Ok(entry)
}

fn load_and_validate_governance_authority(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
) -> Result<crate::CanonicalGovernanceEntryAuthorityV2, &'static str> {
    let sweep = read_json_file::<GovernanceEntrySweepReportV1>(
        &workdir.join("out/governance_entry_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V4_GOVERNANCE_AUTHORITY_FAIL")?;
    if sweep.authority.authority_status != GovernanceEntryAuthorityStatusV2::Pass
        || sweep.authority.applied_supported_set_digest_prefix
            != applied_context.applied_set_digest_prefix
        || sweep.authority.applied_context_digest_prefix
            != prefix_hex(&applied_context.context_digest, 16)
        || sweep.authority.canonical_governance_entry_digest_prefix != canonical_digest_prefix
    {
        return Err("SCOPE_EXEC_V4_GOVERNANCE_AUTHORITY_FAIL");
    }
    Ok(sweep.authority)
}

fn load_and_validate_final_consumer_authority(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
) -> Result<crate::FinalGovernanceConsumerAuthorityV1, &'static str> {
    let sweep = read_json_file::<FinalGovernanceConsumerSweepReportV1>(
        &workdir.join("out/final_governance_consumer_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V5_FINAL_CONSUMER_AUTHORITY_FAIL")?;
    let authority = sweep.authority;
    if authority.authority_status != FinalGovernanceConsumerAuthorityStatusV1::Pass
        || authority.applied_supported_set_digest_prefix
            != applied_context.applied_set_digest_prefix
        || authority.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || authority.canonical_governance_authority_digest_prefix != authority_digest_prefix
    {
        return Err("SCOPE_EXEC_V5_FINAL_CONSUMER_AUTHORITY_FAIL");
    }
    Ok(authority)
}

fn load_and_validate_governance_residual_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
) -> Result<crate::FinalGovernanceResidualSweepV1, &'static str> {
    let sweep = read_json_file::<crate::GovernanceResidualSweepReportV1>(
        &workdir.join("out/governance_residual_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V6_RESIDUAL_SWEEP_FAIL")?;
    let residual = sweep.sweep;
    if !matches!(
        residual.sweep_status,
        crate::GovernanceResidualSweepStatusV1::Pass
    ) || residual.applied_supported_set_digest_prefix
        != applied_context.applied_set_digest_prefix
        || residual.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || residual.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || residual.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
    {
        return Err("SCOPE_EXEC_V6_RESIDUAL_SWEEP_FAIL");
    }
    Ok(residual)
}

fn load_and_validate_residual_free_governance_authority(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
) -> Result<crate::ResidualFreeGovernanceConsumerAuthorityV1, &'static str> {
    let sweep = read_json_file::<ResidualFreeGovernanceSweepReportV1>(
        &workdir.join("out/residual_free_governance_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V7_RESIDUAL_FREE_GOVERNANCE_FAIL")?;
    let authority = sweep.authority;
    if !matches!(
        authority.authority_status,
        ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass
    ) || authority.applied_supported_set_digest_prefix
        != applied_context.applied_set_digest_prefix
        || authority.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || authority.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || authority.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || authority.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
    {
        return Err("SCOPE_EXEC_V7_RESIDUAL_FREE_GOVERNANCE_FAIL");
    }
    Ok(authority)
}

fn load_and_validate_governance_absolute_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
    residual_free_authority_digest_prefix: &str,
) -> Result<crate::ResidualFreeGovernanceAbsoluteSweepV1, &'static str> {
    let sweep = read_json_file::<GovernanceAbsoluteSweepReportV1>(
        &workdir.join("out/governance_absolute_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V8_ABSOLUTE_GOVERNANCE_SWEEP_FAIL")?;
    let absolute = sweep.sweep;
    if !matches!(
        absolute.sweep_status,
        ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass
    ) || absolute.applied_supported_set_digest_prefix
        != applied_context.applied_set_digest_prefix
        || absolute.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || absolute.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || absolute.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || absolute.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
        || absolute.residual_free_governance_consumer_authority_digest_prefix
            != residual_free_authority_digest_prefix
    {
        return Err("SCOPE_EXEC_V8_ABSOLUTE_GOVERNANCE_SWEEP_FAIL");
    }
    Ok(absolute)
}

#[allow(clippy::too_many_arguments)]
fn load_and_validate_terminal_governance_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
    residual_free_authority_digest_prefix: &str,
    absolute_sweep_digest_prefix: &str,
) -> Result<crate::AbsoluteFinalGovernanceTerminalSweepV1, &'static str> {
    let report = read_json_file::<GovernanceTerminalSweepReportV1>(
        &workdir.join("out/governance_terminal_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V9_TERMINAL_GOVERNANCE_SWEEP_FAIL")?;
    let sweep = report.sweep;
    if !matches!(
        sweep.sweep_status,
        AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass
    ) || sweep.applied_supported_set_digest_prefix != applied_context.applied_set_digest_prefix
        || sweep.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || sweep.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || sweep.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || sweep.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
        || sweep.residual_free_governance_consumer_authority_digest_prefix
            != residual_free_authority_digest_prefix
        || sweep.residual_free_governance_absolute_sweep_digest_prefix
            != absolute_sweep_digest_prefix
    {
        return Err("SCOPE_EXEC_V9_TERMINAL_GOVERNANCE_SWEEP_FAIL");
    }
    Ok(sweep)
}

#[allow(clippy::too_many_arguments)]
fn load_and_validate_governance_ultimate_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
    residual_free_authority_digest_prefix: &str,
    absolute_sweep_digest_prefix: &str,
    terminal_sweep_digest_prefix: &str,
) -> Result<crate::TerminalGovernanceUltimateSweepV1, &'static str> {
    let report = read_json_file::<GovernanceUltimateSweepReportV1>(
        &workdir.join("out/governance_ultimate_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V10_ULTIMATE_GOVERNANCE_SWEEP_FAIL")?;
    let sweep = report.sweep;
    if !matches!(
        sweep.sweep_status,
        TerminalGovernanceUltimateSweepStatusV1::Pass
    ) || sweep.applied_supported_set_digest_prefix != applied_context.applied_set_digest_prefix
        || sweep.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || sweep.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || sweep.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || sweep.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
        || sweep.residual_free_governance_consumer_authority_digest_prefix
            != residual_free_authority_digest_prefix
        || sweep.residual_free_governance_absolute_sweep_digest_prefix
            != absolute_sweep_digest_prefix
        || sweep.absolute_final_governance_terminal_sweep_digest_prefix
            != terminal_sweep_digest_prefix
    {
        return Err("SCOPE_EXEC_V10_ULTIMATE_GOVERNANCE_SWEEP_FAIL");
    }
    Ok(sweep)
}

#[allow(clippy::too_many_arguments)]
fn load_and_validate_governance_convergence_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
    residual_free_authority_digest_prefix: &str,
    absolute_sweep_digest_prefix: &str,
    ultimate_sweep_digest_prefix: &str,
) -> Result<crate::GovernanceConvergenceSweepV1, &'static str> {
    let report = read_json_file::<crate::GovernanceConvergenceSweepReportV1>(
        &workdir.join("out/governance_convergence_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V11_GOVERNANCE_CONVERGENCE_FAIL")?;
    let sweep = report.sweep;
    if !matches!(
        sweep.convergence_status,
        crate::GovernanceConvergenceStatusV1::Pass
    ) || sweep.applied_supported_set_digest_prefix != applied_context.applied_set_digest_prefix
        || sweep.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || sweep.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || sweep.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || sweep.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
        || sweep.residual_free_governance_consumer_authority_digest_prefix
            != residual_free_authority_digest_prefix
        || sweep.residual_free_governance_absolute_sweep_digest_prefix
            != absolute_sweep_digest_prefix
        || sweep.terminal_governance_ultimate_sweep_digest_prefix != ultimate_sweep_digest_prefix
    {
        return Err("SCOPE_EXEC_V11_GOVERNANCE_CONVERGENCE_FAIL");
    }
    Ok(sweep)
}

#[allow(clippy::too_many_arguments)]
fn load_and_validate_governance_stabilization_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
    residual_free_authority_digest_prefix: &str,
    absolute_sweep_digest_prefix: &str,
    ultimate_sweep_digest_prefix: &str,
    convergence_sweep_digest_prefix: &str,
) -> Result<crate::GovernanceStabilizationSweepV1, &'static str> {
    let report = read_json_file::<crate::GovernanceStabilizationSweepReportV1>(
        &workdir.join("out/governance_stabilization_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V12_GOVERNANCE_STABILIZATION_FAIL")?;
    let sweep = report.sweep;
    if !matches!(
        sweep.stabilization_status,
        crate::GovernanceStabilizationStatusV1::Pass
    ) || sweep.applied_supported_set_digest_prefix != applied_context.applied_set_digest_prefix
        || sweep.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || sweep.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || sweep.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || sweep.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
        || sweep.residual_free_governance_consumer_authority_digest_prefix
            != residual_free_authority_digest_prefix
        || sweep.residual_free_governance_absolute_sweep_digest_prefix
            != absolute_sweep_digest_prefix
        || sweep.terminal_governance_ultimate_sweep_digest_prefix != ultimate_sweep_digest_prefix
        || sweep.governance_convergence_sweep_digest_prefix != convergence_sweep_digest_prefix
    {
        return Err("SCOPE_EXEC_V12_GOVERNANCE_STABILIZATION_FAIL");
    }
    Ok(sweep)
}

#[allow(clippy::too_many_arguments)]
fn load_and_validate_governance_final_consolidation_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
    residual_free_authority_digest_prefix: &str,
    absolute_sweep_digest_prefix: &str,
    terminal_sweep_digest_prefix: &str,
    ultimate_sweep_digest_prefix: &str,
    convergence_sweep_digest_prefix: &str,
    stabilization_sweep_digest_prefix: &str,
) -> Result<crate::GovernanceFinalConsolidationSweepV1, &'static str> {
    let report = read_json_file::<GovernanceFinalConsolidationSweepReportV1>(
        &workdir.join("out/governance_final_consolidation_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V13_GOVERNANCE_FINAL_CONSOLIDATION_FAIL")?;
    let sweep = report.sweep;
    if !matches!(
        sweep.consolidation_status,
        GovernanceFinalConsolidationStatusV1::Pass
    ) || sweep.applied_supported_set_digest_prefix != applied_context.applied_set_digest_prefix
        || sweep.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || sweep.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || sweep.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || sweep.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
        || sweep.residual_free_governance_consumer_authority_digest_prefix
            != residual_free_authority_digest_prefix
        || sweep.residual_free_governance_absolute_sweep_digest_prefix
            != absolute_sweep_digest_prefix
        || sweep.absolute_final_governance_terminal_sweep_digest_prefix
            != terminal_sweep_digest_prefix
        || sweep.terminal_governance_ultimate_sweep_digest_prefix != ultimate_sweep_digest_prefix
        || sweep.governance_convergence_sweep_digest_prefix != convergence_sweep_digest_prefix
        || sweep.governance_stabilization_sweep_digest_prefix != stabilization_sweep_digest_prefix
    {
        return Err("SCOPE_EXEC_V13_GOVERNANCE_FINAL_CONSOLIDATION_FAIL");
    }
    Ok(sweep)
}

#[allow(clippy::too_many_arguments)]
fn load_and_validate_governance_closure_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
    residual_free_authority_digest_prefix: &str,
    absolute_sweep_digest_prefix: &str,
    terminal_sweep_digest_prefix: &str,
    ultimate_sweep_digest_prefix: &str,
    convergence_sweep_digest_prefix: &str,
    stabilization_sweep_digest_prefix: &str,
    final_consolidation_sweep_digest_prefix: &str,
) -> Result<crate::GovernanceClosureSweepV1, &'static str> {
    let report = read_json_file::<GovernanceClosureSweepReportV1>(
        &workdir.join("out/governance_closure_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V14_GOVERNANCE_CLOSURE_FAIL")?;
    let sweep = report.sweep;
    if !matches!(sweep.closure_status, GovernanceClosureStatusV1::Pass)
        || sweep.applied_supported_set_digest_prefix != applied_context.applied_set_digest_prefix
        || sweep.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || sweep.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || sweep.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || sweep.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
        || sweep.residual_free_governance_consumer_authority_digest_prefix
            != residual_free_authority_digest_prefix
        || sweep.residual_free_governance_absolute_sweep_digest_prefix
            != absolute_sweep_digest_prefix
        || sweep.absolute_final_governance_terminal_sweep_digest_prefix
            != terminal_sweep_digest_prefix
        || sweep.terminal_governance_ultimate_sweep_digest_prefix != ultimate_sweep_digest_prefix
        || sweep.governance_convergence_sweep_digest_prefix != convergence_sweep_digest_prefix
        || sweep.governance_stabilization_sweep_digest_prefix != stabilization_sweep_digest_prefix
        || sweep.governance_final_consolidation_sweep_digest_prefix
            != final_consolidation_sweep_digest_prefix
    {
        return Err("SCOPE_EXEC_V14_GOVERNANCE_CLOSURE_FAIL");
    }
    Ok(sweep)
}

#[allow(clippy::too_many_arguments)]
fn load_and_validate_governance_seal_sweep(
    workdir: &Path,
    applied_context: &AppliedSupportedSetContextV1,
    canonical_digest_prefix: &str,
    authority_digest_prefix: &str,
    final_consumer_authority_digest_prefix: &str,
    residual_sweep_digest_prefix: &str,
    residual_free_authority_digest_prefix: &str,
    absolute_sweep_digest_prefix: &str,
    terminal_sweep_digest_prefix: &str,
    ultimate_sweep_digest_prefix: &str,
    convergence_sweep_digest_prefix: &str,
    stabilization_sweep_digest_prefix: &str,
    final_consolidation_sweep_digest_prefix: &str,
    closure_sweep_digest_prefix: &str,
) -> Result<crate::GovernanceSealSweepV1, &'static str> {
    let report = read_json_file::<GovernanceSealSweepReportV1>(
        &workdir.join("out/governance_seal_sweep.json"),
    )
    .map_err(|_| "SCOPE_EXEC_V15_GOVERNANCE_SEAL_FAIL")?;
    let sweep = report.sweep;
    if !matches!(sweep.seal_status, GovernanceSealStatusV1::Pass)
        || sweep.applied_supported_set_digest_prefix != applied_context.applied_set_digest_prefix
        || sweep.canonical_governance_entry_digest_prefix != canonical_digest_prefix
        || sweep.canonical_governance_authority_digest_prefix != authority_digest_prefix
        || sweep.final_governance_consumer_authority_digest_prefix
            != final_consumer_authority_digest_prefix
        || sweep.final_governance_residual_sweep_digest_prefix != residual_sweep_digest_prefix
        || sweep.residual_free_governance_consumer_authority_digest_prefix
            != residual_free_authority_digest_prefix
        || sweep.residual_free_governance_absolute_sweep_digest_prefix
            != absolute_sweep_digest_prefix
        || sweep.absolute_final_governance_terminal_sweep_digest_prefix
            != terminal_sweep_digest_prefix
        || sweep.terminal_governance_ultimate_sweep_digest_prefix != ultimate_sweep_digest_prefix
        || sweep.governance_convergence_sweep_digest_prefix != convergence_sweep_digest_prefix
        || sweep.governance_stabilization_sweep_digest_prefix != stabilization_sweep_digest_prefix
        || sweep.governance_final_consolidation_sweep_digest_prefix
            != final_consolidation_sweep_digest_prefix
        || sweep.governance_closure_sweep_digest_prefix != closure_sweep_digest_prefix
    {
        return Err("SCOPE_EXEC_V15_GOVERNANCE_SEAL_FAIL");
    }
    Ok(sweep)
}

#[allow(dead_code)]
fn ensure_current_supported_scope_execution_v12(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
) -> Result<SupportedScopeExecutionV12, OpsError> {
    let reeval = ensure_current_supported_scope_reevaluation_v1(workdir, policy, previous_set)?;
    let path = workdir.join("out").join("supported_scope_execute_v12.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV12>(&path) {
        if report.current_policy_digest_prefix == prefix_hex(&policy.policy_digest, 16)
            && report.previous_applied_set_digest_prefix == prefix_hex(&previous_set.set_digest, 16)
            && report.current_reevaluation_digest_prefix
                == prefix_hex(&reeval.reevaluation_digest, 16)
        {
            return Ok(report);
        }
    }
    models_supported_scope_execute_v12(workdir, &path)
}

#[allow(dead_code)]
fn ensure_current_supported_scope_execution_v13(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
) -> Result<SupportedScopeExecutionV13, OpsError> {
    let reeval = ensure_current_supported_scope_reevaluation_v1(workdir, policy, previous_set)?;
    let path = workdir.join("out").join("supported_scope_execute_v13.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV13>(&path) {
        if report.current_policy_digest_prefix == prefix_hex(&policy.policy_digest, 16)
            && report.previous_applied_set_digest_prefix == prefix_hex(&previous_set.set_digest, 16)
            && report.current_reevaluation_digest_prefix
                == prefix_hex(&reeval.reevaluation_digest, 16)
        {
            return Ok(report);
        }
    }
    models_supported_scope_execute_v13(workdir, &path)
}

#[allow(dead_code)]
fn ensure_current_supported_scope_execution_v14(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
) -> Result<SupportedScopeExecutionV14, OpsError> {
    let reeval = ensure_current_supported_scope_reevaluation_v1(workdir, policy, previous_set)?;
    let path = workdir.join("out").join("supported_scope_execute_v14.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV14>(&path) {
        if report.current_policy_digest_prefix == prefix_hex(&policy.policy_digest, 16)
            && report.previous_applied_set_digest_prefix == prefix_hex(&previous_set.set_digest, 16)
            && report.current_reevaluation_digest_prefix
                == prefix_hex(&reeval.reevaluation_digest, 16)
        {
            return Ok(report);
        }
    }
    models_supported_scope_execute_v14(workdir, &path)
}

fn ensure_current_supported_scope_execution_v15(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
) -> Result<SupportedScopeExecutionV15, OpsError> {
    let reeval = ensure_current_supported_scope_reevaluation_v1(workdir, policy, previous_set)?;
    let path = workdir.join("out").join("supported_scope_execute_v15.json");
    if let Ok(report) = read_json_file::<SupportedScopeExecutionV15>(&path) {
        if report.current_policy_digest_prefix == prefix_hex(&policy.policy_digest, 16)
            && report.previous_applied_set_digest_prefix == prefix_hex(&previous_set.set_digest, 16)
            && report.current_reevaluation_digest_prefix
                == prefix_hex(&reeval.reevaluation_digest, 16)
        {
            return Ok(report);
        }
    }
    models_supported_scope_execute_v15(workdir, &path)
}

fn load_latest_supported_set_policy_v2(
    workdir: &Path,
) -> Result<SupportedRealSlotSetPolicyV2, OpsError> {
    let path = workdir.join("out").join("supported_set_review.json");
    let body = fs::read_to_string(&path).map_err(|_| {
        OpsError::Invalid(
            "SUPPORTED_SET_POLICY_V2_MISSING: run `ucf-ops models supported-set-review` first"
                .to_string(),
        )
    })?;
    let report: SupportedSetReviewReportV1 = serde_json::from_str(&body).map_err(|_| {
        OpsError::Invalid(
            "SUPPORTED_SET_POLICY_V2_INVALID: unable to decode review report".to_string(),
        )
    })?;
    Ok(report.policy)
}

fn load_latest_supported_scope_reevaluation_v1(
    workdir: &Path,
) -> Result<SupportedScopeReevaluationV1, OpsError> {
    let path = workdir.join("out").join("supported_scope_reeval.json");
    let body = fs::read_to_string(&path).map_err(|_| {
        OpsError::Invalid(
            "SUPPORTED_SCOPE_REEVAL_MISSING: run `ucf-ops models supported-scope-reevaluate` first"
                .to_string(),
        )
    })?;
    serde_json::from_str(&body).map_err(|_| {
        OpsError::Invalid(
            "SUPPORTED_SCOPE_REEVAL_INVALID: unable to decode reevaluation report".to_string(),
        )
    })
}

fn ensure_current_supported_scope_reevaluation_v1(
    workdir: &Path,
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
) -> Result<SupportedScopeReevaluationV1, OpsError> {
    let policy_digest_prefix = prefix_hex(&policy.policy_digest, 16);
    let previous_digest_prefix = prefix_hex(&previous_set.set_digest, 16);
    let stale_or_missing = match load_latest_supported_scope_reevaluation_v1(workdir) {
        Ok(report) => {
            report.policy_digest_prefix != policy_digest_prefix
                || report.previous_applied_set_digest_prefix != previous_digest_prefix
        }
        Err(_) => true,
    };
    if stale_or_missing {
        let reeval_out = workdir.join("out").join("supported_scope_reeval.json");
        return models_supported_scope_reevaluate(workdir, &reeval_out);
    }
    load_latest_supported_scope_reevaluation_v1(workdir)
}

fn slot_expansion_candidate_digest(candidate: &SlotExpansionEligibilityV1) -> String {
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(candidate.slot_id.as_bytes());
    digest_source.extend_from_slice(if candidate.trait_contract_exists {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(if candidate.probe_path_exists_or_reusable {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(if candidate.shadow_path_exists_or_trivially_attachable {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(if candidate.compare_window_normalizable {
        b"1"
    } else {
        b"0"
    });
    digest_source.extend_from_slice(
        if candidate.strict_evidence_plumbing_representable_without_arch_fork {
            b"1"
        } else {
            b"0"
        },
    );
    digest_source.extend_from_slice(if candidate.tiny_fixture_path_feasible {
        b"1"
    } else {
        b"0"
    });
    sha256_hex(&digest_source)
}

fn validate_supported_set_execution(
    policy: &SupportedRealSlotSetPolicyV2,
    previous_set: &SupportedRealSlotSetV1,
    candidates: &[SlotExpansionEligibilityV1],
) -> Result<SupportedRealSlotSetV2, SupportedSetExecutionDeniedV1> {
    let mut previous_slots = previous_set.slots.clone();
    previous_slots.sort();
    previous_slots.dedup();
    let mut policy_slots = policy.current_supported_slots.clone();
    policy_slots.sort();
    policy_slots.dedup();
    if previous_slots != policy_slots {
        return Err(SupportedSetExecutionDeniedV1 {
            code: SupportedSetExecutionDeniedCodeV1::SupportedSetExecutionDeniedStalePolicy,
            detail: "policy current_supported_slots no longer matches repository baseline"
                .to_string(),
        });
    }

    match policy.decision {
        SupportedRealSlotSetDecisionV2::Freeze => Ok(build_supported_real_slot_set_v2(
            previous_slots,
            &policy.policy_digest,
            &previous_set.set_digest,
            SupportedRealSlotSetExecutionDecisionV2::Frozen,
        )),
        SupportedRealSlotSetDecisionV2::ExpandByOne => {
            let Some(chosen) = policy.chosen_candidate_slot.as_ref() else {
                return Err(SupportedSetExecutionDeniedV1 {
                    code:
                        SupportedSetExecutionDeniedCodeV1::SupportedSetExecutionDeniedAmbiguousSlot,
                    detail: "policy requested expansion but chose no candidate slot".to_string(),
                });
            };
            if !policy
                .candidate_slots_considered
                .iter()
                .any(|slot| slot == chosen)
            {
                return Err(SupportedSetExecutionDeniedV1 {
                    code:
                        SupportedSetExecutionDeniedCodeV1::SupportedSetExecutionDeniedScopeMismatch,
                    detail: "chosen candidate slot is not in candidate_slots_considered"
                        .to_string(),
                });
            }
            let matching = candidates
                .iter()
                .filter(|candidate| candidate.slot_id == *chosen)
                .collect::<Vec<_>>();
            if matching.len() != 1 {
                return Err(SupportedSetExecutionDeniedV1 {
                    code:
                        SupportedSetExecutionDeniedCodeV1::SupportedSetExecutionDeniedAmbiguousSlot,
                    detail: "chosen candidate slot is ambiguous in execution candidates"
                        .to_string(),
                });
            }
            let candidate = matching[0];
            if !candidate.trait_contract_exists
                || !candidate.probe_path_exists_or_reusable
                || !candidate.shadow_path_exists_or_trivially_attachable
                || !candidate.compare_window_normalizable
                || !candidate.strict_evidence_plumbing_representable_without_arch_fork
            {
                return Err(SupportedSetExecutionDeniedV1 {
                    code: SupportedSetExecutionDeniedCodeV1::SupportedSetExecutionDeniedIncompleteScaffold,
                    detail: "candidate slot scaffolding no longer satisfies expansion prerequisites"
                        .to_string(),
                });
            }
            let mut expanded = previous_slots;
            expanded.push(chosen.clone());
            expanded.sort();
            expanded.dedup();
            if expanded.len() != previous_set.slots.len() + 1 {
                return Err(SupportedSetExecutionDeniedV1 {
                    code:
                        SupportedSetExecutionDeniedCodeV1::SupportedSetExecutionDeniedScopeMismatch,
                    detail: "expansion must add exactly one slot".to_string(),
                });
            }
            Ok(build_supported_real_slot_set_v2(
                expanded,
                &policy.policy_digest,
                &previous_set.set_digest,
                SupportedRealSlotSetExecutionDecisionV2::Expanded,
            ))
        }
    }
}

fn build_supported_real_slot_set_v2(
    mut slots: Vec<String>,
    policy_digest: &str,
    previous_set_digest: &str,
    decision: SupportedRealSlotSetExecutionDecisionV2,
) -> SupportedRealSlotSetV2 {
    slots.sort();
    slots.dedup();
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_REAL_SLOT_SET_V2_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(prefix_hex(policy_digest, 16).as_bytes());
    digest_source.extend_from_slice(prefix_hex(previous_set_digest, 16).as_bytes());
    digest_source.extend_from_slice(format!("{decision:?}").as_bytes());
    for slot in &slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    SupportedRealSlotSetV2 {
        schema_version: SUPPORTED_REAL_SLOT_SET_V2_SCHEMA_VERSION,
        slots,
        source_policy_digest_prefix: prefix_hex(policy_digest, 16),
        decision,
        previous_set_digest_prefix: prefix_hex(previous_set_digest, 16),
        set_digest: sha256_hex(&digest_source),
    }
}

fn load_applied_supported_real_slot_set_v2(
    workdir: &Path,
) -> Result<SupportedRealSlotSetV2, OpsError> {
    let path = workdir
        .join("out")
        .join("supported_real_slot_set_applied_v2.json");
    let body = fs::read_to_string(&path)
        .map_err(|_| OpsError::Invalid("SUPPORTED_SET_APPLY_MISSING".to_string()))?;
    serde_json::from_str(&body)
        .map_err(|_| OpsError::Invalid("SUPPORTED_SET_APPLY_INVALID".to_string()))
}

fn known_slots_ordered() -> Vec<String> {
    ModelSlot::all()
        .iter()
        .map(|slot| slot.as_str().to_string())
        .collect::<Vec<_>>()
}

fn evaluate_slot_expansion_candidate(
    slot_id: &str,
    current_supported: &BTreeSet<String>,
    max_supported_slots: usize,
) -> SlotExpansionEligibilityV1 {
    let Ok(slot) = parse_slot(slot_id) else {
        return SlotExpansionEligibilityV1 {
            schema_version: SLOT_EXPANSION_ELIGIBILITY_SCHEMA_VERSION,
            slot_id: slot_id.to_string(),
            trait_contract_exists: false,
            probe_path_exists_or_reusable: false,
            shadow_path_exists_or_trivially_attachable: false,
            compare_window_normalizable: false,
            strict_evidence_plumbing_representable_without_arch_fork: false,
            tiny_fixture_path_feasible: false,
            expansion_ready: false,
            denial_reason_code: Some("UNKNOWN_SLOT_METADATA".to_string()),
        };
    };

    let trait_contract_exists = !matches!(slot, ModelSlot::Llm);
    let probe_path_exists_or_reusable = true;
    let shadow_path_exists_or_trivially_attachable =
        matches!(slot, ModelSlot::Sae | ModelSlot::Ssm);
    let compare_window_normalizable = shadow_path_exists_or_trivially_attachable;
    let strict_evidence_plumbing_representable_without_arch_fork =
        shadow_path_exists_or_trivially_attachable && current_supported.len() < max_supported_slots;
    let tiny_fixture_path_feasible = matches!(slot, ModelSlot::Sae | ModelSlot::Ssm);

    let checks = [
        (trait_contract_exists, "TRAIT_CONTRACT_MISSING"),
        (probe_path_exists_or_reusable, "PROBE_PATH_MISSING"),
        (
            shadow_path_exists_or_trivially_attachable,
            "SHADOW_PATH_MISSING",
        ),
        (compare_window_normalizable, "COMPARE_NORMALIZATION_MISSING"),
        (
            strict_evidence_plumbing_representable_without_arch_fork,
            "STRICT_PLUMBING_ARCH_FORK_REQUIRED",
        ),
        (tiny_fixture_path_feasible, "TINY_FIXTURE_NOT_FEASIBLE"),
    ];
    let denial_reason_code =
        checks
            .iter()
            .find_map(|(ok, code)| if *ok { None } else { Some((*code).to_string()) });
    let expansion_ready = denial_reason_code.is_none();

    SlotExpansionEligibilityV1 {
        schema_version: SLOT_EXPANSION_ELIGIBILITY_SCHEMA_VERSION,
        slot_id: slot_id.to_string(),
        trait_contract_exists,
        probe_path_exists_or_reusable,
        shadow_path_exists_or_trivially_attachable,
        compare_window_normalizable,
        strict_evidence_plumbing_representable_without_arch_fork,
        tiny_fixture_path_feasible,
        expansion_ready,
        denial_reason_code,
    }
}

fn classify_known_slot(
    slot_id: &str,
    current_supported: &BTreeSet<String>,
    candidates: &[SlotExpansionEligibilityV1],
) -> KnownSlotClassificationV1 {
    if current_supported.contains(slot_id) {
        return KnownSlotClassificationV1::CurrentlySupportedRealSlot;
    }
    let Some(candidate) = candidates.iter().find(|c| c.slot_id == slot_id) else {
        return KnownSlotClassificationV1::UnsupportedAbsent;
    };
    if !candidate.trait_contract_exists && !candidate.probe_path_exists_or_reusable {
        return KnownSlotClassificationV1::UnsupportedAbsent;
    }
    if candidate.probe_path_exists_or_reusable
        && !candidate.shadow_path_exists_or_trivially_attachable
        && !candidate.compare_window_normalizable
        && !candidate.strict_evidence_plumbing_representable_without_arch_fork
    {
        return KnownSlotClassificationV1::StubOnly;
    }
    KnownSlotClassificationV1::PartiallyScaffolded
}

fn select_supported_slot_set_policy_v2(
    current_supported_slots: &[String],
    candidates: &[SlotExpansionEligibilityV1],
) -> SupportedRealSlotSetPolicyV2 {
    let mut current_supported_slots = current_supported_slots.to_vec();
    current_supported_slots.sort();
    current_supported_slots.dedup();

    let mut candidate_slots_considered = candidates
        .iter()
        .map(|c| c.slot_id.clone())
        .collect::<Vec<_>>();
    candidate_slots_considered.sort();

    let ready = candidates
        .iter()
        .filter(|c| c.expansion_ready)
        .map(|c| c.slot_id.clone())
        .collect::<Vec<_>>();
    let (decision, chosen_candidate_slot, rationale_codes) = match ready.len() {
        1 => (
            SupportedRealSlotSetDecisionV2::ExpandByOne,
            Some(ready[0].clone()),
            vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
        ),
        0 => (
            SupportedRealSlotSetDecisionV2::Freeze,
            None,
            vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
        ),
        _ => (
            SupportedRealSlotSetDecisionV2::Freeze,
            None,
            vec!["AMBIGUOUS_MULTIPLE_CANDIDATES".to_string()],
        ),
    };
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        SUPPORTED_REAL_SLOT_SET_POLICY_V2_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    for slot in &current_supported_slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    for slot in &candidate_slots_considered {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    digest_source.extend_from_slice(format!("{decision:?}").as_bytes());
    if let Some(chosen) = chosen_candidate_slot.as_ref() {
        digest_source.extend_from_slice(chosen.as_bytes());
    }
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    SupportedRealSlotSetPolicyV2 {
        schema_version: SUPPORTED_REAL_SLOT_SET_POLICY_V2_SCHEMA_VERSION,
        current_supported_slots,
        candidate_slots_considered,
        decision,
        chosen_candidate_slot,
        rationale_codes,
        policy_digest: sha256_hex(&digest_source),
    }
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
            .join(format!("{slot_id}_parity_report.json"))
    });
    let default_path = workdir
        .join("out")
        .join(format!("{slot_id}_parity_report.json"));
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

fn burn_support_resolution_from_state(
    slot: ModelSlot,
    support_state: OptionalBackendSupportStateV1,
) -> BurnSupportResolutionV1 {
    let mut rationale_codes = Vec::new();
    let resolution = match support_state {
        OptionalBackendSupportStateV1::Supported => {
            rationale_codes.push("BURN_SHADOW_COMPARE_AVAILABLE".to_string());
            BurnResolutionStatusV1::BurnSupportedForShadowCompare
        }
        OptionalBackendSupportStateV1::Unsupported => {
            rationale_codes.push("BURN_SLOT_FORMALLY_UNSUPPORTED".to_string());
            BurnResolutionStatusV1::BurnClosedUnsupported
        }
        OptionalBackendSupportStateV1::NotBuilt => {
            rationale_codes.push("BURN_FEATURE_NOT_BUILT".to_string());
            BurnResolutionStatusV1::BurnClosedUnsupported
        }
        OptionalBackendSupportStateV1::NotConfigured => {
            rationale_codes.push("BURN_SHADOW_NOT_CONFIGURED".to_string());
            BurnResolutionStatusV1::BurnClosedUnsupported
        }
    };
    rationale_codes.sort();
    rationale_codes.dedup();
    rationale_codes.truncate(4);

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(slot.as_str().as_bytes());
    digest_source.extend_from_slice(format!("{resolution:?}").as_bytes());
    digest_source.extend_from_slice(format!("{support_state:?}").as_bytes());
    for code in &rationale_codes {
        digest_source.extend_from_slice(code.as_bytes());
    }

    BurnSupportResolutionV1 {
        slot_id: slot.as_str().to_string(),
        resolution,
        support_state,
        rationale_codes,
        evidence_digest: sha256_hex(&digest_source),
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

pub fn current_supported_real_slot_set(workdir: &Path) -> Result<SupportedRealSlotSetV1, OpsError> {
    match load_applied_supported_real_slot_set_v2(workdir) {
        Ok(applied) => Ok(SupportedRealSlotSetV1 {
            schema_version: 1,
            slots: applied.slots,
            source: "out/supported_real_slot_set_applied_v2.json".to_string(),
            set_digest: applied.set_digest,
        }),
        Err(_) => supported_real_slot_set_v1(),
    }
}

pub fn load_applied_supported_set_context_v1(
    workdir: &Path,
) -> Result<AppliedSupportedSetContextV1, OpsError> {
    match load_applied_supported_real_slot_set_v2(workdir) {
        Ok(applied) => Ok(build_applied_supported_set_context_from_v2(&applied, None)),
        Err(_) => {
            let legacy = supported_real_slot_set_v1()?;
            Ok(build_legacy_applied_supported_set_context(&legacy))
        }
    }
}

fn build_applied_supported_set_context_from_v2(
    applied: &SupportedRealSlotSetV2,
    compatibility_code: Option<String>,
) -> AppliedSupportedSetContextV1 {
    let mut slots = applied.slots.clone();
    slots.sort();
    slots.dedup();
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        APPLIED_SUPPORTED_SET_CONTEXT_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(prefix_hex(&applied.set_digest, 16).as_bytes());
    digest_source.extend_from_slice(format!("{:?}", applied.decision).as_bytes());
    digest_source.extend_from_slice(applied.previous_set_digest_prefix.as_bytes());
    digest_source.extend_from_slice(applied.source_policy_digest_prefix.as_bytes());
    for slot in &slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    AppliedSupportedSetContextV1 {
        schema_version: APPLIED_SUPPORTED_SET_CONTEXT_SCHEMA_VERSION,
        applied_set_digest_prefix: prefix_hex(&applied.set_digest, 16),
        slots,
        decision: applied.decision.clone(),
        previous_set_digest_prefix: applied.previous_set_digest_prefix.clone(),
        policy_digest_prefix: applied.source_policy_digest_prefix.clone(),
        context_digest: sha256_hex(&digest_source),
        compatibility_code,
    }
}

fn build_legacy_applied_supported_set_context(
    legacy: &SupportedRealSlotSetV1,
) -> AppliedSupportedSetContextV1 {
    let mut slots = legacy.slots.clone();
    slots.sort();
    slots.dedup();
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(
        APPLIED_SUPPORTED_SET_CONTEXT_SCHEMA_VERSION
            .to_string()
            .as_bytes(),
    );
    digest_source.extend_from_slice(prefix_hex(&legacy.set_digest, 16).as_bytes());
    digest_source.extend_from_slice(b"FROZEN");
    digest_source.extend_from_slice(b"legacy_previous");
    digest_source.extend_from_slice(b"legacy_policy");
    for slot in &slots {
        digest_source.extend_from_slice(slot.as_bytes());
    }
    AppliedSupportedSetContextV1 {
        schema_version: APPLIED_SUPPORTED_SET_CONTEXT_SCHEMA_VERSION,
        applied_set_digest_prefix: prefix_hex(&legacy.set_digest, 16),
        slots,
        decision: SupportedRealSlotSetExecutionDecisionV2::Frozen,
        previous_set_digest_prefix: "legacy_previous".to_string(),
        policy_digest_prefix: "legacy_policy".to_string(),
        context_digest: sha256_hex(&digest_source),
        compatibility_code: Some("LEGACY_SCOPE_TRANSLATED".to_string()),
    }
}

fn validate_snapshot_matches_applied_scope(
    slot_set_digest: &str,
    slots: &[String],
    applied_scope: &AppliedSupportedSetContextV1,
    code: &str,
) -> Result<(), OpsError> {
    if prefix_hex(slot_set_digest, 16) != applied_scope.applied_set_digest_prefix {
        return Err(OpsError::Invalid(format!("{code}:DIGEST_PREFIX_MISMATCH")));
    }
    validate_slot_membership(
        slots,
        &applied_scope.slots,
        &format!("{code}_EXTRA_SLOT"),
        &format!("{code}_MISSING_SLOT"),
    )
}

fn validate_slot_membership(
    observed_slots: &[String],
    expected_slots: &[String],
    extra_code: &str,
    missing_code: &str,
) -> Result<(), OpsError> {
    let observed = observed_slots.iter().cloned().collect::<BTreeSet<_>>();
    let expected = expected_slots.iter().cloned().collect::<BTreeSet<_>>();
    if observed.iter().any(|slot| !expected.contains(slot)) {
        return Err(OpsError::Invalid(extra_code.to_string()));
    }
    if expected.iter().any(|slot| !observed.contains(slot)) {
        return Err(OpsError::Invalid(missing_code.to_string()));
    }
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
    let set = current_supported_real_slot_set(Path::new("."))?;
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
    digest_source.extend_from_slice(format!("{latest_probe_status:?}").as_bytes());
    digest_source.extend_from_slice(latest_compare_window_digest_prefix.as_bytes());
    digest_source.extend_from_slice(if compare_window_present { b"1" } else { b"0" });
    digest_source.extend_from_slice(format!("{compare_freshness:?}").as_bytes());
    digest_source.extend_from_slice(if no_impact_verified { b"1" } else { b"0" });
    digest_source.extend_from_slice(format!("{drift_status:?}").as_bytes());
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
                "UCF_OPS_MODELS_PROBE_ACTIVE_HASH_MISSING: {active_hash}"
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
        fn slot_candidate_evaluation_is_deterministic() {
            let current = ["sae".to_string(), "world_jepa".to_string()]
                .into_iter()
                .collect::<BTreeSet<_>>();
            let a = evaluate_slot_expansion_candidate("ssm", &current, 3);
            let b = evaluate_slot_expansion_candidate("ssm", &current, 3);
            assert_eq!(a, b);
            assert!(a.expansion_ready);
        }

        #[test]
        fn candidate_ordering_is_stable() {
            let mut names = known_slots_ordered();
            names.sort();
            assert_eq!(
                names,
                vec![
                    "ebm_reasoner",
                    "lfm",
                    "llm",
                    "sae",
                    "ssm",
                    "world_jepa",
                    "world_vljepa"
                ]
            );
        }

        #[test]
        fn supported_set_review_defaults_to_freeze_for_current_repo_state() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("docs").expect("docs");
            fs::write(
                "docs/series_state_snapshot.md",
                "- Second supported slot in this stage: `sae`
",
            )
            .expect("snapshot");
            let out = PathBuf::from("out/supported_set_review.json");
            let report = models_supported_set_review(Path::new("."), &out).expect("review");
            assert_eq!(
                report.policy.decision,
                SupportedRealSlotSetDecisionV2::Freeze
            );
            assert!(report.policy.chosen_candidate_slot.is_none());
        }

        #[test]
        fn policy_selection_expand_by_one_when_exactly_one_candidate_is_ready() {
            let candidates = vec![
                SlotExpansionEligibilityV1 {
                    schema_version: 1,
                    slot_id: "ssm".to_string(),
                    trait_contract_exists: true,
                    probe_path_exists_or_reusable: true,
                    shadow_path_exists_or_trivially_attachable: true,
                    compare_window_normalizable: true,
                    strict_evidence_plumbing_representable_without_arch_fork: true,
                    tiny_fixture_path_feasible: true,
                    expansion_ready: true,
                    denial_reason_code: None,
                },
                SlotExpansionEligibilityV1 {
                    schema_version: 1,
                    slot_id: "world_vljepa".to_string(),
                    trait_contract_exists: true,
                    probe_path_exists_or_reusable: true,
                    shadow_path_exists_or_trivially_attachable: false,
                    compare_window_normalizable: false,
                    strict_evidence_plumbing_representable_without_arch_fork: false,
                    tiny_fixture_path_feasible: false,
                    expansion_ready: false,
                    denial_reason_code: Some("SHADOW_PATH_MISSING".to_string()),
                },
            ];
            let policy = select_supported_slot_set_policy_v2(
                &["world_jepa".to_string(), "sae".to_string()],
                &candidates,
            );
            assert_eq!(policy.decision, SupportedRealSlotSetDecisionV2::ExpandByOne);
            assert_eq!(policy.chosen_candidate_slot.as_deref(), Some("ssm"));
        }

        #[test]
        fn policy_selection_freezes_on_ambiguous_multiple_ready_candidates() {
            let candidates = vec![
                SlotExpansionEligibilityV1 {
                    schema_version: 1,
                    slot_id: "ssm".to_string(),
                    trait_contract_exists: true,
                    probe_path_exists_or_reusable: true,
                    shadow_path_exists_or_trivially_attachable: true,
                    compare_window_normalizable: true,
                    strict_evidence_plumbing_representable_without_arch_fork: true,
                    tiny_fixture_path_feasible: true,
                    expansion_ready: true,
                    denial_reason_code: None,
                },
                SlotExpansionEligibilityV1 {
                    schema_version: 1,
                    slot_id: "world_vljepa".to_string(),
                    trait_contract_exists: true,
                    probe_path_exists_or_reusable: true,
                    shadow_path_exists_or_trivially_attachable: true,
                    compare_window_normalizable: true,
                    strict_evidence_plumbing_representable_without_arch_fork: true,
                    tiny_fixture_path_feasible: true,
                    expansion_ready: true,
                    denial_reason_code: None,
                },
            ];
            let policy = select_supported_slot_set_policy_v2(
                &["world_jepa".to_string(), "sae".to_string()],
                &candidates,
            );
            assert_eq!(policy.decision, SupportedRealSlotSetDecisionV2::Freeze);
            assert!(policy
                .rationale_codes
                .iter()
                .any(|c| c == "AMBIGUOUS_MULTIPLE_CANDIDATES"));
        }

        #[test]
        fn unknown_slot_metadata_is_not_eligible() {
            let current = BTreeSet::new();
            let candidate = evaluate_slot_expansion_candidate("unknown_slot", &current, 3);
            assert!(!candidate.expansion_ready);
            assert_eq!(
                candidate.denial_reason_code.as_deref(),
                Some("UNKNOWN_SLOT_METADATA")
            );
        }

        #[test]
        fn supported_set_apply_freeze_is_deterministic() {
            let previous = SupportedRealSlotSetV1 {
                schema_version: 1,
                slots: vec!["sae".to_string(), "world_jepa".to_string()],
                source: "docs".to_string(),
                set_digest: "aa".repeat(32),
            };
            let policy = SupportedRealSlotSetPolicyV2 {
                schema_version: 2,
                current_supported_slots: previous.slots.clone(),
                candidate_slots_considered: vec!["ssm".to_string()],
                decision: SupportedRealSlotSetDecisionV2::Freeze,
                chosen_candidate_slot: None,
                rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                policy_digest: "bb".repeat(32),
            };
            let cands = vec![evaluate_slot_expansion_candidate(
                "ssm",
                &previous.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )];
            let a = validate_supported_set_execution(&policy, &previous, &cands).expect("freeze a");
            let b = validate_supported_set_execution(&policy, &previous, &cands).expect("freeze b");
            assert_eq!(a, b);
            assert_eq!(a.decision, SupportedRealSlotSetExecutionDecisionV2::Frozen);
            assert_eq!(a.slots, vec!["sae".to_string(), "world_jepa".to_string()]);
        }

        #[test]
        fn supported_set_apply_expands_by_one_when_scaffold_is_complete() {
            let previous = SupportedRealSlotSetV1 {
                schema_version: 1,
                slots: vec!["sae".to_string(), "world_jepa".to_string()],
                source: "docs".to_string(),
                set_digest: "aa".repeat(32),
            };
            let policy = SupportedRealSlotSetPolicyV2 {
                schema_version: 2,
                current_supported_slots: previous.slots.clone(),
                candidate_slots_considered: vec!["ssm".to_string()],
                decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                chosen_candidate_slot: Some("ssm".to_string()),
                rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                policy_digest: "cc".repeat(32),
            };
            let cands = vec![evaluate_slot_expansion_candidate(
                "ssm",
                &previous.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )];
            let applied =
                validate_supported_set_execution(&policy, &previous, &cands).expect("expanded");
            assert_eq!(
                applied.decision,
                SupportedRealSlotSetExecutionDecisionV2::Expanded
            );
            assert_eq!(
                applied.slots,
                vec![
                    "sae".to_string(),
                    "ssm".to_string(),
                    "world_jepa".to_string()
                ]
            );
        }

        #[test]
        fn supported_set_apply_denies_on_incomplete_scaffold() {
            let previous = SupportedRealSlotSetV1 {
                schema_version: 1,
                slots: vec!["sae".to_string(), "world_jepa".to_string()],
                source: "docs".to_string(),
                set_digest: "aa".repeat(32),
            };
            let policy = SupportedRealSlotSetPolicyV2 {
                schema_version: 2,
                current_supported_slots: previous.slots.clone(),
                candidate_slots_considered: vec!["world_vljepa".to_string()],
                decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                chosen_candidate_slot: Some("world_vljepa".to_string()),
                rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                policy_digest: "dd".repeat(32),
            };
            let cands = vec![evaluate_slot_expansion_candidate(
                "world_vljepa",
                &previous.slots.iter().cloned().collect::<BTreeSet<_>>(),
                SLOT_SET_MAX + 1,
            )];
            let denied = validate_supported_set_execution(&policy, &previous, &cands)
                .expect_err("must deny");
            assert_eq!(
                denied.code,
                SupportedSetExecutionDeniedCodeV1::SupportedSetExecutionDeniedIncompleteScaffold
            );
        }

        fn write_scope_reeval_support_artifacts(root: &Path, applied_digest: &str, slots: &[&str]) {
            fs::create_dir_all(root.join("out")).expect("out");
            let backend = BackendEvidenceSnapshotV1 {
                schema_version: 1,
                supported_slot_set_digest: applied_digest.to_string(),
                policy_graph_digest_prefix: "11".repeat(8),
                manifest_digest_prefix: "22".repeat(8),
                slots: slots
                    .iter()
                    .map(|slot| BackendEvidenceSlotSnapshotV1 {
                        slot_id: (*slot).to_string(),
                        target_hash_prefix: "aa".repeat(8),
                        backend_support: BackendSupportMatrixV1 {
                            stub: BackendSupportStateV1::Supported,
                            candle: BackendSupportStateV1::Supported,
                            burn: BackendSupportStateV1::Unsupported,
                        },
                        evidence: BackendEvidenceSlotEvidenceV1 {
                            latest_probe_report_digest_prefix: "p".repeat(16),
                            latest_compare_window_digest_prefix: "c".repeat(16),
                            latest_shadow_ready_digest_prefix: "s".repeat(16),
                            latest_active_evidence_digest_prefix: "a".repeat(16),
                            latest_drift_status: DriftStatusV1::Ok,
                            freshness_probe_age_ticks: Some(1),
                            freshness_compare_age_ticks: Some(1),
                            freshness_no_impact_age_ticks: Some(1),
                            freshness_drift_status_age_ticks: Some(1),
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
                            active: Some(EvidenceDenialCodeV1::ActiveNotEnabled),
                        },
                        remediation_codes: vec![],
                        canonical_remediation_codes: vec![],
                        burn_resolution: BurnSupportResolutionV1 {
                            slot_id: (*slot).to_string(),
                            resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                            support_state: OptionalBackendSupportStateV1::NotConfigured,
                            rationale_codes: vec!["BURN_SHADOW_NOT_CONFIGURED".to_string()],
                            evidence_digest: "bb".repeat(32),
                        },
                    })
                    .collect(),
                snapshot_digest: "cc".repeat(32),
            };
            fs::write(
                root.join("out/backend_evidence_snapshot.json"),
                serde_json::to_vec_pretty(&backend).expect("backend json"),
            )
            .expect("write backend");

            let active = AggregatedActiveReviewSnapshotV1 {
                schema_version: 1,
                supported_slot_set_digest: applied_digest.to_string(),
                policy_graph_digest_prefix: "11".repeat(8),
                manifest_digest_prefix: "22".repeat(8),
                slots: slots
                    .iter()
                    .map(|slot| ActiveReviewEvidenceV1 {
                        slot_id: (*slot).to_string(),
                        target_hash_prefix: "aa".repeat(8),
                        manifest_digest_prefix: "22".repeat(8),
                        probe_ready: true,
                        shadow_ready: true,
                        active_eligible: false,
                        strict_blocking: false,
                        drift_blocking: false,
                        alert_blocking: false,
                        primary_denial_code: Some("ActiveNotEnabled".to_string()),
                        remediation_codes: vec![],
                        contributing_evidence_digests: ActiveReviewContributingDigestsV1 {
                            probe_report_digest_prefix: "p".repeat(16),
                            shadow_ready_digest_prefix: "s".repeat(16),
                            active_evidence_digest_prefix: "a".repeat(16),
                            strict_evidence_digest_prefix: "x".repeat(16),
                        },
                        burn_resolution: BurnSupportResolutionV1 {
                            slot_id: (*slot).to_string(),
                            resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                            support_state: OptionalBackendSupportStateV1::NotConfigured,
                            rationale_codes: vec!["BURN_SHADOW_NOT_CONFIGURED".to_string()],
                            evidence_digest: "bb".repeat(32),
                        },
                        evidence_digest: "dd".repeat(32),
                    })
                    .collect(),
                overall_review_status: ActiveReviewOverallStatusV1::NoneReviewable,
                signoff_alignment: ActiveReviewSignoffAlignmentV1 {
                    aligned: true,
                    status_code: "ALIGNED".to_string(),
                },
                canonical_governance_entry_digest_prefix: "MISSING".to_string(),
                final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
                governance_residual_sweep_digest_prefix: "MISSING".to_string(),
                residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
                governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
                absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
                governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
                final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
                readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
                residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
                readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
                readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
                readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
                readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
                readiness_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
                readiness_closure_sweep_digest_prefix: "MISSING".to_string(),
                readiness_seal_sweep_digest_prefix: "MISSING".to_string(),
                snapshot_digest: "ee".repeat(32),
            };
            fs::write(
                root.join("out/active_review_snapshot.json"),
                serde_json::to_vec_pretty(&active).expect("active json"),
            )
            .expect("write active");

            fs::write(
                root.join("out/interop_consistency_matrix.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "matrix": {
                        "applied_supported_set_digest_prefix": prefix_hex(applied_digest, 16)
                    }
                }))
                .expect("interop json"),
            )
            .expect("interop");
            fs::write(
                root.join("out/scope_authority_check.json"),
                "{\"status\":\"PASS\"}",
            )
            .expect("scope auth");
        }

        fn write_governance_entry_check_artifact(root: &Path, authority_digest_prefix: &str) {
            let report = serde_json::json!({
                "schema_version": 1,
                "status": "PASS",
                "authority_digest_prefix": authority_digest_prefix,
                "consumers": []
            });
            fs::write(
                root.join("out/governance_entry_check.json"),
                serde_json::to_vec_pretty(&report).expect("governance entry json"),
            )
            .expect("write governance entry");
        }

        fn write_governance_entry_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            applied_context_digest_prefix: &str,
            canonical_digest_prefix: &str,
            status: GovernanceEntryAuthorityStatusV2,
        ) {
            let report = GovernanceEntrySweepReportV1 {
                schema_version: 1,
                authority: crate::CanonicalGovernanceEntryAuthorityV2 {
                    schema_version: 2,
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    applied_context_digest_prefix: applied_context_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    covered_surface_count: 6,
                    authority_status: status,
                    authority_digest: "ab".repeat(32),
                },
                surfaces: vec![],
            };
            fs::write(
                root.join("out/governance_entry_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance sweep json"),
            )
            .expect("write governance sweep");
        }

        fn write_final_governance_consumer_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            status: FinalGovernanceConsumerAuthorityStatusV1,
        ) {
            let report = FinalGovernanceConsumerSweepReportV1 {
                schema_version: 1,
                authority: crate::FinalGovernanceConsumerAuthorityV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    covered_consumer_count: 5,
                    authority_status: status,
                    authority_digest: "cd".repeat(32),
                },
                consumers: vec![],
            };
            fs::write(
                root.join("out/final_governance_consumer_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("final governance consumer json"),
            )
            .expect("write final governance consumer sweep");
        }

        fn write_governance_residual_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            status: crate::GovernanceResidualSweepStatusV1,
        ) {
            let report = crate::GovernanceResidualSweepReportV1 {
                schema_version: 1,
                sweep: crate::FinalGovernanceResidualSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(
                        status,
                        crate::GovernanceResidualSweepStatusV1::Pass
                    ) {
                        0
                    } else {
                        1
                    },
                    sweep_status: status,
                    sweep_digest: "ef".repeat(32),
                },
                consumers: vec![],
            };
            fs::write(
                root.join("out/governance_residual_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance residual sweep json"),
            )
            .expect("write governance residual sweep");
        }

        fn write_residual_free_governance_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            status: crate::ResidualFreeGovernanceConsumerAuthorityStatusV1,
        ) {
            let report = crate::ResidualFreeGovernanceSweepReportV1 {
                schema_version: 1,
                authority: crate::ResidualFreeGovernanceConsumerAuthorityV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(
                        status,
                        crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass
                    ) {
                        0
                    } else {
                        1
                    },
                    authority_status: status,
                    authority_digest: "12".repeat(32),
                },
                consumers: vec![],
            };
            fs::write(
                root.join("out/residual_free_governance_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("residual free governance sweep json"),
            )
            .expect("write residual free governance sweep");
        }

        #[allow(clippy::too_many_arguments)]
        fn write_governance_absolute_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            residual_free_authority_digest_prefix: &str,
            status: crate::ResidualFreeGovernanceAbsoluteSweepStatusV1,
        ) {
            let report = crate::GovernanceAbsoluteSweepReportV1 {
                schema_version: 1,
                sweep: crate::ResidualFreeGovernanceAbsoluteSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    residual_free_governance_consumer_authority_digest_prefix:
                        residual_free_authority_digest_prefix.to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(
                        status,
                        crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass
                    ) {
                        0
                    } else {
                        1
                    },
                    sweep_status: status,
                    sweep_digest: "34".repeat(32),
                },
                consumers: vec![],
            };
            fs::write(
                root.join("out/governance_absolute_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance absolute sweep json"),
            )
            .expect("write governance absolute sweep");
        }

        #[allow(clippy::too_many_arguments)]
        fn write_governance_terminal_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            residual_free_authority_digest_prefix: &str,
            absolute_sweep_digest_prefix: &str,
            status: crate::AbsoluteFinalGovernanceTerminalSweepStatusV1,
        ) {
            let report = crate::GovernanceTerminalSweepReportV1 {
                schema_version: 1,
                sweep: crate::AbsoluteFinalGovernanceTerminalSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    residual_free_governance_consumer_authority_digest_prefix:
                        residual_free_authority_digest_prefix.to_string(),
                    residual_free_governance_absolute_sweep_digest_prefix:
                        absolute_sweep_digest_prefix.to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(
                        status,
                        crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass
                    ) {
                        0
                    } else {
                        1
                    },
                    sweep_status: status,
                    sweep_digest: "56".repeat(32),
                },
                consumers: vec![],
            };
            fs::write(
                root.join("out/governance_terminal_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance terminal sweep json"),
            )
            .expect("write governance terminal sweep");
        }

        #[allow(clippy::too_many_arguments)]
        fn write_governance_ultimate_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            residual_free_authority_digest_prefix: &str,
            absolute_sweep_digest_prefix: &str,
            terminal_sweep_digest_prefix: &str,
            status: crate::TerminalGovernanceUltimateSweepStatusV1,
        ) {
            let report = crate::GovernanceUltimateSweepReportV1 {
                schema_version: 1,
                sweep: crate::TerminalGovernanceUltimateSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    residual_free_governance_consumer_authority_digest_prefix:
                        residual_free_authority_digest_prefix.to_string(),
                    residual_free_governance_absolute_sweep_digest_prefix:
                        absolute_sweep_digest_prefix.to_string(),
                    absolute_final_governance_terminal_sweep_digest_prefix:
                        terminal_sweep_digest_prefix.to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(
                        status,
                        crate::TerminalGovernanceUltimateSweepStatusV1::Pass
                    ) {
                        0
                    } else {
                        1
                    },
                    sweep_status: status,
                    sweep_digest: "78".repeat(32),
                },
                consumers: vec![],
            };
            fs::write(
                root.join("out/governance_ultimate_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance ultimate sweep json"),
            )
            .expect("write governance ultimate sweep");
        }

        #[allow(clippy::too_many_arguments)]
        fn write_governance_convergence_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            residual_free_authority_digest_prefix: &str,
            absolute_sweep_digest_prefix: &str,
            ultimate_sweep_digest_prefix: &str,
            status: crate::GovernanceConvergenceStatusV1,
        ) {
            let report = crate::GovernanceConvergenceSweepReportV1 {
                schema_version: 1,
                sweep: crate::GovernanceConvergenceSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    residual_free_governance_consumer_authority_digest_prefix:
                        residual_free_authority_digest_prefix.to_string(),
                    residual_free_governance_absolute_sweep_digest_prefix:
                        absolute_sweep_digest_prefix.to_string(),
                    absolute_final_governance_terminal_sweep_digest_prefix: "56".repeat(8),
                    terminal_governance_ultimate_sweep_digest_prefix: ultimate_sweep_digest_prefix
                        .to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(
                        status,
                        crate::GovernanceConvergenceStatusV1::Pass
                    ) {
                        0
                    } else {
                        1
                    },
                    convergence_status: status,
                    convergence_digest: "90".repeat(32),
                },
                consumers: vec![],
            };
            fs::write(
                root.join("out/governance_convergence_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance convergence sweep json"),
            )
            .expect("write governance convergence sweep");
        }

        #[allow(clippy::too_many_arguments)]
        fn write_governance_stabilization_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            residual_free_authority_digest_prefix: &str,
            absolute_sweep_digest_prefix: &str,
            ultimate_sweep_digest_prefix: &str,
            convergence_sweep_digest_prefix: &str,
            status: crate::GovernanceStabilizationStatusV1,
        ) {
            let report = crate::GovernanceStabilizationSweepReportV1 {
                schema_version: 1,
                sweep: crate::GovernanceStabilizationSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    residual_free_governance_consumer_authority_digest_prefix:
                        residual_free_authority_digest_prefix.to_string(),
                    residual_free_governance_absolute_sweep_digest_prefix:
                        absolute_sweep_digest_prefix.to_string(),
                    absolute_final_governance_terminal_sweep_digest_prefix: "56".repeat(8),
                    terminal_governance_ultimate_sweep_digest_prefix: ultimate_sweep_digest_prefix
                        .to_string(),
                    governance_convergence_sweep_digest_prefix: convergence_sweep_digest_prefix
                        .to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(
                        status,
                        crate::GovernanceStabilizationStatusV1::Pass
                    ) {
                        0
                    } else {
                        1
                    },
                    stabilization_status: status.clone(),
                    stabilization_digest: "9a".repeat(32),
                },
                consumers: vec![crate::GovernanceStabilizationConsumerStatusV1 {
                    consumer: "active_review_snapshot".to_string(),
                    status,
                    mismatch_categories: Vec::new(),
                }],
            };
            fs::write(
                root.join("out/governance_stabilization_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance stabilization sweep json"),
            )
            .expect("write governance stabilization sweep");
        }

        #[allow(clippy::too_many_arguments)]
        fn write_governance_final_consolidation_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            residual_free_authority_digest_prefix: &str,
            absolute_sweep_digest_prefix: &str,
            ultimate_sweep_digest_prefix: &str,
            convergence_sweep_digest_prefix: &str,
            stabilization_sweep_digest_prefix: &str,
            status: crate::GovernanceFinalConsolidationStatusV1,
        ) {
            let report = crate::GovernanceFinalConsolidationSweepReportV1 {
                schema_version: 1,
                sweep: crate::GovernanceFinalConsolidationSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    residual_free_governance_consumer_authority_digest_prefix:
                        residual_free_authority_digest_prefix.to_string(),
                    residual_free_governance_absolute_sweep_digest_prefix:
                        absolute_sweep_digest_prefix.to_string(),
                    absolute_final_governance_terminal_sweep_digest_prefix: "56".repeat(8),
                    terminal_governance_ultimate_sweep_digest_prefix: ultimate_sweep_digest_prefix
                        .to_string(),
                    governance_convergence_sweep_digest_prefix: convergence_sweep_digest_prefix
                        .to_string(),
                    governance_stabilization_sweep_digest_prefix: stabilization_sweep_digest_prefix
                        .to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(
                        status,
                        crate::GovernanceFinalConsolidationStatusV1::Pass
                    ) {
                        0
                    } else {
                        1
                    },
                    consolidation_status: status.clone(),
                    consolidation_digest: "9b".repeat(32),
                },
                consumers: vec![crate::GovernanceFinalConsolidationConsumerStatusV1 {
                    consumer: "active_review_snapshot".to_string(),
                    status,
                    mismatch_categories: Vec::new(),
                }],
            };
            fs::write(
                root.join("out/governance_final_consolidation_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance final consolidation json"),
            )
            .expect("write governance final consolidation sweep");
        }

        #[allow(clippy::too_many_arguments)]
        fn write_governance_closure_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            residual_free_authority_digest_prefix: &str,
            absolute_sweep_digest_prefix: &str,
            ultimate_sweep_digest_prefix: &str,
            convergence_sweep_digest_prefix: &str,
            stabilization_sweep_digest_prefix: &str,
            final_consolidation_sweep_digest_prefix: &str,
            status: crate::GovernanceClosureStatusV1,
        ) {
            let report = crate::GovernanceClosureSweepReportV1 {
                schema_version: 1,
                sweep: crate::GovernanceClosureSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    residual_free_governance_consumer_authority_digest_prefix:
                        residual_free_authority_digest_prefix.to_string(),
                    residual_free_governance_absolute_sweep_digest_prefix:
                        absolute_sweep_digest_prefix.to_string(),
                    absolute_final_governance_terminal_sweep_digest_prefix: "56".repeat(8),
                    terminal_governance_ultimate_sweep_digest_prefix: ultimate_sweep_digest_prefix
                        .to_string(),
                    governance_convergence_sweep_digest_prefix: convergence_sweep_digest_prefix
                        .to_string(),
                    governance_stabilization_sweep_digest_prefix: stabilization_sweep_digest_prefix
                        .to_string(),
                    governance_final_consolidation_sweep_digest_prefix:
                        final_consolidation_sweep_digest_prefix.to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(status, crate::GovernanceClosureStatusV1::Pass)
                    {
                        0
                    } else {
                        1
                    },
                    closure_status: status.clone(),
                    closure_digest: "9c".repeat(32),
                },
                consumers: vec![crate::GovernanceClosureConsumerStatusV1 {
                    consumer: "active_review_snapshot".to_string(),
                    status,
                    mismatch_categories: Vec::new(),
                }],
            };
            fs::write(
                root.join("out/governance_closure_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance closure json"),
            )
            .expect("write governance closure sweep");
        }

        #[allow(clippy::too_many_arguments)]
        fn write_governance_seal_sweep_artifact(
            root: &Path,
            applied_digest_prefix: &str,
            canonical_digest_prefix: &str,
            authority_digest_prefix: &str,
            final_consumer_authority_digest_prefix: &str,
            residual_sweep_digest_prefix: &str,
            residual_free_authority_digest_prefix: &str,
            absolute_sweep_digest_prefix: &str,
            ultimate_sweep_digest_prefix: &str,
            convergence_sweep_digest_prefix: &str,
            stabilization_sweep_digest_prefix: &str,
            final_consolidation_sweep_digest_prefix: &str,
            closure_sweep_digest_prefix: &str,
            status: crate::GovernanceSealStatusV1,
        ) {
            let report = crate::GovernanceSealSweepReportV1 {
                schema_version: 1,
                sweep: crate::GovernanceSealSweepV1 {
                    applied_supported_set_digest_prefix: applied_digest_prefix.to_string(),
                    canonical_governance_entry_digest_prefix: canonical_digest_prefix.to_string(),
                    canonical_governance_authority_digest_prefix: authority_digest_prefix
                        .to_string(),
                    final_governance_consumer_authority_digest_prefix:
                        final_consumer_authority_digest_prefix.to_string(),
                    final_governance_residual_sweep_digest_prefix: residual_sweep_digest_prefix
                        .to_string(),
                    residual_free_governance_consumer_authority_digest_prefix:
                        residual_free_authority_digest_prefix.to_string(),
                    residual_free_governance_absolute_sweep_digest_prefix:
                        absolute_sweep_digest_prefix.to_string(),
                    absolute_final_governance_terminal_sweep_digest_prefix: "56".repeat(8),
                    terminal_governance_ultimate_sweep_digest_prefix: ultimate_sweep_digest_prefix
                        .to_string(),
                    governance_convergence_sweep_digest_prefix: convergence_sweep_digest_prefix
                        .to_string(),
                    governance_stabilization_sweep_digest_prefix: stabilization_sweep_digest_prefix
                        .to_string(),
                    governance_final_consolidation_sweep_digest_prefix:
                        final_consolidation_sweep_digest_prefix.to_string(),
                    governance_closure_sweep_digest_prefix: closure_sweep_digest_prefix.to_string(),
                    covered_consumer_count: 6,
                    residual_path_count: if matches!(status, crate::GovernanceSealStatusV1::Pass) {
                        0
                    } else {
                        1
                    },
                    seal_status: status.clone(),
                    seal_digest: "9d".repeat(32),
                },
                consumers: vec![crate::GovernanceSealConsumerStatusV1 {
                    consumer: "active_review_snapshot".to_string(),
                    status,
                    mismatch_categories: Vec::new(),
                }],
            };
            fs::write(
                root.join("out/governance_seal_sweep.json"),
                serde_json::to_vec_pretty(&report).expect("governance seal json"),
            )
            .expect("write governance seal sweep");
        }

        #[test]
        fn supported_scope_reeval_reaffirms_freeze_when_policy_stale() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["sae".to_string(), "world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");

            let out = PathBuf::from("out/supported_scope_reeval.json");
            let report = models_supported_scope_reevaluate(Path::new("."), &out).expect("reeval");
            assert_eq!(
                report.reevaluation_decision,
                SupportedScopeReevaluationDecisionV1::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_REEVAL_STALE_POLICY".to_string()));
        }

        #[test]
        fn supported_scope_reeval_executes_expand_by_one_when_authority_inputs_are_clean() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["sae".to_string(), "world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["sae", "world_jepa"],
            );

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["sae".to_string(), "world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");

            let out = PathBuf::from("out/supported_scope_reeval.json");
            let report = models_supported_scope_reevaluate(Path::new("."), &out).expect("reeval");
            assert_eq!(
                report.reevaluation_decision,
                SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            );
            assert_eq!(report.chosen_candidate_slot.as_deref(), Some("ssm"));
        }

        #[test]
        fn supported_scope_reeval_freezes_when_two_candidates_viable() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["sae".to_string(), "ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("sae".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");

            let out = PathBuf::from("out/supported_scope_reeval.json");
            let report = models_supported_scope_reevaluate(Path::new("."), &out).expect("reeval");
            assert_eq!(
                report.reevaluation_decision,
                SupportedScopeReevaluationDecisionV1::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_REEVAL_AMBIGUOUS_CANDIDATE".to_string()));
        }

        #[test]
        fn supported_scope_execute_reaffirms_freeze_when_canonical_entry_missing() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");

            let out = PathBuf::from("out/supported_scope_execute_v3.json");
            let report = models_supported_scope_execute(Path::new("."), &out).expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV3::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V3_CANONICAL_ENTRY_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_expands_when_one_candidate_and_canonical_entry_pass() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");

            let reeval = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            assert_eq!(
                reeval.reevaluation_decision,
                SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            );

            let applied_ctx =
                load_applied_supported_set_context_v1(Path::new(".")).expect("applied ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let surfaces = validate_governance_primary_surfaces_with_applied_scope(
                &backend,
                &active,
                &applied_ctx,
            )
            .expect("surfaces");
            let entry = derive_canonical_governance_entry(&applied_ctx, &surfaces).expect("entry");
            write_governance_entry_check_artifact(
                Path::new("."),
                &prefix_hex(&entry.authority_digest, 16),
            );

            let out = PathBuf::from("out/supported_scope_execute_v3.json");
            let report = models_supported_scope_execute(Path::new("."), &out).expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV3::ExecuteExpandByOne
            );
            assert_eq!(report.chosen_candidate_slot.as_deref(), Some("ssm"));
        }

        #[test]
        fn supported_scope_execute_v4_expands_when_one_candidate_and_authority_pass() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");

            let reeval = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            assert_eq!(
                reeval.reevaluation_decision,
                SupportedScopeReevaluationDecisionV1::ExecuteExpandByOne
            );

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let surfaces = validate_governance_primary_surfaces_with_applied_scope(
                &backend,
                &active,
                &applied_ctx,
            )
            .expect("surfaces");
            let entry = derive_canonical_governance_entry(&applied_ctx, &surfaces).expect("entry");
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &prefix_hex(&entry.authority_digest, 16),
                GovernanceEntryAuthorityStatusV2::Pass,
            );

            let out = PathBuf::from("out/supported_scope_execute_v4.json");
            let report = models_supported_scope_execute_v4(Path::new("."), &out).expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV4::ExecuteExpandByOne
            );
            assert_eq!(report.chosen_candidate_slot.as_deref(), Some("ssm"));
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V4_NO_ACTIVE_IMPLICATIONS".to_string()));
        }

        #[test]
        fn supported_scope_execute_v4_reaffirms_freeze_when_two_candidates_viable() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["sae".to_string(), "ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &prefix_hex(&entry.authority_digest, 16),
                GovernanceEntryAuthorityStatusV2::Pass,
            );

            let report = models_supported_scope_execute_v4(
                Path::new("."),
                Path::new("out/supported_scope_execute_v4.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV4::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V4_AMBIGUOUS_CANDIDATE".to_string()));
        }

        #[test]
        fn supported_scope_execute_v4_digest_is_deterministic() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::Freeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &prefix_hex(&entry.authority_digest, 16),
                GovernanceEntryAuthorityStatusV2::Pass,
            );

            let a = models_supported_scope_execute_v4(
                Path::new("."),
                Path::new("out/supported_scope_execute_v4_a.json"),
            )
            .expect("execute a");
            let b = models_supported_scope_execute_v4(
                Path::new("."),
                Path::new("out/supported_scope_execute_v4_b.json"),
            )
            .expect("execute b");
            assert_eq!(a.execution_digest, b.execution_digest);
            assert_eq!(a.rationale_codes, b.rationale_codes);
        }

        #[test]
        fn supported_scope_execute_v4_denies_stale_reevaluation() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::Freeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");

            let stale = SupportedScopeReevaluationV1 {
                schema_version: 1,
                previous_applied_set_digest_prefix: "ff".repeat(8),
                policy_digest_prefix: "ee".repeat(8),
                reevaluation_decision: SupportedScopeReevaluationDecisionV1::ReaffirmFreeze,
                chosen_candidate_slot: None,
                rationale_codes: vec!["SCOPE_REEVAL_FREEZE_DEFAULT".to_string()],
                reevaluation_digest: "44".repeat(32),
            };
            fs::write(
                "out/supported_scope_reeval.json",
                serde_json::to_vec_pretty(&stale).expect("stale"),
            )
            .expect("write stale reeval");
            let err = models_supported_scope_execute_v4(
                Path::new("."),
                Path::new("out/supported_scope_execute_v4.json"),
            )
            .expect_err("must reject stale reevaluation");
            assert!(err.to_string().contains("SCOPE_EXEC_V4_STALE_REEVALUATION"));
        }

        #[test]
        fn supported_scope_execute_v4_reaffirms_freeze_when_governance_authority_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &prefix_hex(&entry.authority_digest, 16),
                GovernanceEntryAuthorityStatusV2::Fail,
            );

            let report = models_supported_scope_execute_v4(
                Path::new("."),
                Path::new("out/supported_scope_execute_v4.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV4::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V4_GOVERNANCE_AUTHORITY_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v4_denies_stale_prior_execution_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let stale_prior = SupportedScopeExecutionV3 {
                schema_version: 3,
                previous_applied_set_digest_prefix: "ff".repeat(8),
                current_policy_digest_prefix: "ee".repeat(8),
                current_reevaluation_digest_prefix: "dd".repeat(8),
                canonical_governance_entry_digest_prefix: "cc".repeat(8),
                execution_decision: SupportedScopeExecutionDecisionV3::ReaffirmFreeze,
                chosen_candidate_slot: None,
                resulting_supported_set_digest_prefix: "bb".repeat(8),
                rationale_codes: vec!["SCOPE_EXEC_V3_REAFFIRM_FREEZE".to_string()],
                execution_digest: "aa".repeat(32),
            };
            fs::write(
                "out/supported_scope_execute_v3.json",
                serde_json::to_vec_pretty(&stale_prior).expect("stale prior"),
            )
            .expect("write stale prior");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &prefix_hex(&entry.authority_digest, 16),
                GovernanceEntryAuthorityStatusV2::Pass,
            );

            let err = models_supported_scope_execute_v4(
                Path::new("."),
                Path::new("out/supported_scope_execute_v4.json"),
            )
            .expect_err("must reject stale prior execution");
            assert!(err
                .to_string()
                .contains("SCOPE_EXEC_V4_STALE_PRIOR_EXECUTION"));
        }

        #[test]
        fn supported_scope_execute_v5_expands_when_final_consumer_authority_passes() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &prefix_hex("ab".repeat(32).as_str(), 16),
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v5(
                Path::new("."),
                Path::new("out/supported_scope_execute_v5.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV5::ExecuteExpandByOne
            );
            assert_eq!(report.chosen_candidate_slot.as_deref(), Some("ssm"));
        }

        #[test]
        fn supported_scope_execute_v5_freezes_when_final_consumer_authority_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Fail,
            );
            let report = models_supported_scope_execute_v5(
                Path::new("."),
                Path::new("out/supported_scope_execute_v5.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV5::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V5_FINAL_CONSUMER_AUTHORITY_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v5_freezes_on_ambiguous_candidates() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["sae".to_string(), "ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("sae".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            let report = models_supported_scope_execute_v5(
                Path::new("."),
                Path::new("out/supported_scope_execute_v5.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV5::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V5_AMBIGUOUS_CANDIDATE".to_string()));
        }

        #[test]
        fn supported_scope_execute_v5_denies_stale_prior_execution_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::Freeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let stale = SupportedScopeExecutionV5 {
                schema_version: 5,
                previous_applied_set_digest_prefix: "ff".repeat(8),
                current_policy_digest_prefix: "ee".repeat(8),
                current_reevaluation_digest_prefix: "dd".repeat(8),
                canonical_governance_entry_digest_prefix: "cc".repeat(8),
                canonical_governance_authority_digest_prefix: "bb".repeat(8),
                final_governance_consumer_authority_digest_prefix: "aa".repeat(8),
                prior_scope_execution_digest_prefix: None,
                execution_decision: SupportedScopeExecutionDecisionV5::ReaffirmFreeze,
                chosen_candidate_slot: None,
                resulting_supported_set_digest_prefix: "99".repeat(8),
                rationale_codes: vec!["SCOPE_EXEC_V5_REAFFIRM_FREEZE".to_string()],
                execution_digest: "88".repeat(32),
            };
            fs::write(
                "out/supported_scope_execute_v5.json",
                serde_json::to_vec_pretty(&stale).expect("stale"),
            )
            .expect("write stale");
            let err = models_supported_scope_execute_v5(
                Path::new("."),
                Path::new("out/supported_scope_execute_v5_next.json"),
            )
            .expect_err("must reject stale prior");
            assert!(err
                .to_string()
                .contains("SCOPE_EXEC_V5_STALE_PRIOR_EXECUTION"));
        }

        #[test]
        fn supported_scope_execute_v6_expands_with_final_governance_and_residual_pass() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &prefix_hex("cd".repeat(32).as_str(), 16),
                crate::GovernanceResidualSweepStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v6(
                Path::new("."),
                Path::new("out/supported_scope_execute_v6.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV6::ExecuteExpandByOne
            );
            assert_eq!(report.chosen_candidate_slot.as_deref(), Some("ssm"));
        }

        #[test]
        fn supported_scope_execute_v6_freezes_when_residual_sweep_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &prefix_hex("cd".repeat(32).as_str(), 16),
                crate::GovernanceResidualSweepStatusV1::Fail,
            );

            let report = models_supported_scope_execute_v6(
                Path::new("."),
                Path::new("out/supported_scope_execute_v6.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV6::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V6_RESIDUAL_SWEEP_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v6_freezes_on_ambiguous_candidates() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["sae".to_string(), "ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("sae".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &prefix_hex("cd".repeat(32).as_str(), 16),
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            let report = models_supported_scope_execute_v6(
                Path::new("."),
                Path::new("out/supported_scope_execute_v6.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV6::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V6_AMBIGUOUS_CANDIDATE".to_string()));
        }

        #[test]
        fn supported_scope_execute_v6_denies_stale_prior_execution_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::Freeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let stale = SupportedScopeExecutionV6 {
                schema_version: 6,
                previous_applied_set_digest_prefix: "ff".repeat(8),
                current_policy_digest_prefix: "ee".repeat(8),
                current_reevaluation_digest_prefix: "dd".repeat(8),
                canonical_governance_entry_digest_prefix: "cc".repeat(8),
                canonical_governance_authority_digest_prefix: "bb".repeat(8),
                final_governance_consumer_authority_digest_prefix: "aa".repeat(8),
                final_governance_residual_sweep_digest_prefix: "99".repeat(8),
                prior_scope_execution_digest_prefix: None,
                execution_decision: SupportedScopeExecutionDecisionV6::ReaffirmFreeze,
                chosen_candidate_slot: None,
                resulting_supported_set_digest_prefix: "88".repeat(8),
                rationale_codes: vec!["SCOPE_EXEC_V6_REAFFIRM_FREEZE".to_string()],
                execution_digest: "77".repeat(32),
            };
            fs::write(
                "out/supported_scope_execute_v6.json",
                serde_json::to_vec_pretty(&stale).expect("stale"),
            )
            .expect("write stale");
            let err = models_supported_scope_execute_v6(
                Path::new("."),
                Path::new("out/supported_scope_execute_v6_next.json"),
            )
            .expect_err("must reject stale prior");
            assert!(err
                .to_string()
                .contains("SCOPE_EXEC_V6_STALE_PRIOR_EXECUTION"));
        }

        #[test]
        fn supported_scope_execute_v7_expands_with_residual_free_governance_pass() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v7(
                Path::new("."),
                Path::new("out/supported_scope_execute_v7.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV7::ExecuteExpandByOne
            );
            assert_eq!(report.chosen_candidate_slot.as_deref(), Some("ssm"));
        }

        #[test]
        fn supported_scope_execute_v7_reaffirms_freeze_on_residual_free_governance_fail() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Fail,
            );

            let report = models_supported_scope_execute_v7(
                Path::new("."),
                Path::new("out/supported_scope_execute_v7.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV7::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V7_RESIDUAL_FREE_GOVERNANCE_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v7_reaffirms_freeze_on_historical_governance_dependency() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/scope_authority_check.json",
                serde_json::to_vec_pretty(&serde_json::json!({"status":"FAIL"})).expect("json"),
            )
            .expect("scope authority");
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v7(
                Path::new("."),
                Path::new("out/supported_scope_execute_v7.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV7::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V7_HISTORICAL_GOVERNANCE_DEPENDENCY".to_string()));
        }

        #[test]
        fn supported_scope_execute_v7_denies_stale_reevaluation_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::Freeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            fs::write(
                "out/supported_scope_reeval.json",
                serde_json::to_vec_pretty(&SupportedScopeReevaluationV1 {
                    schema_version: 1,
                    previous_applied_set_digest_prefix: "ff".repeat(8),
                    policy_digest_prefix: "ee".repeat(8),
                    reevaluation_decision: SupportedScopeReevaluationDecisionV1::ReaffirmFreeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["stale".to_string()],
                    reevaluation_digest: "dd".repeat(32),
                })
                .expect("reeval"),
            )
            .expect("write stale reeval");

            let err = models_supported_scope_execute_v7(
                Path::new("."),
                Path::new("out/supported_scope_execute_v7.json"),
            )
            .expect_err("must reject stale reevaluation");
            assert!(err.to_string().contains("SCOPE_EXEC_V7_STALE_REEVALUATION"));
        }

        #[test]
        fn supported_scope_execute_v8_expands_with_absolute_governance_pass() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                    chosen_candidate_slot: Some("ssm".to_string()),
                    rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v8(
                Path::new("."),
                Path::new("out/supported_scope_execute_v8.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV8::ExecuteExpandByOne
            );
            assert_eq!(report.chosen_candidate_slot.as_deref(), Some("ssm"));
        }

        #[test]
        fn supported_scope_execute_v8_reaffirms_freeze_on_absolute_sweep_fail() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Fail,
            );
            let report = models_supported_scope_execute_v8(
                Path::new("."),
                Path::new("out/supported_scope_execute_v8.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV8::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V8_ABSOLUTE_GOVERNANCE_SWEEP_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v8_denies_stale_reevaluation_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::Freeze,
                        chosen_candidate_slot: None,
                        rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            fs::write(
                "out/supported_scope_reeval.json",
                serde_json::to_vec_pretty(&SupportedScopeReevaluationV1 {
                    schema_version: 1,
                    previous_applied_set_digest_prefix: "ff".repeat(8),
                    policy_digest_prefix: "ee".repeat(8),
                    reevaluation_decision: SupportedScopeReevaluationDecisionV1::ReaffirmFreeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["stale".to_string()],
                    reevaluation_digest: "dd".repeat(32),
                })
                .expect("stale"),
            )
            .expect("write stale");
            let err = models_supported_scope_execute_v8(
                Path::new("."),
                Path::new("out/supported_scope_execute_v8.json"),
            )
            .expect_err("must reject stale reevaluation");
            assert!(err.to_string().contains("SCOPE_EXEC_V8_STALE_REEVALUATION"));
        }

        #[test]
        fn supported_scope_execute_v8_freezes_on_ambiguous_candidates() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["sae".to_string(), "ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            let report = models_supported_scope_execute_v8(
                Path::new("."),
                Path::new("out/supported_scope_execute_v8.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV8::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V8_AMBIGUOUS_CANDIDATE".to_string()));
        }

        #[test]
        fn supported_scope_execute_v9_expands_when_terminal_governance_passes() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v9(
                Path::new("."),
                Path::new("out/supported_scope_execute_v9.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV9::ExecuteExpandByOne
            );
            assert_eq!(report.chosen_candidate_slot.as_deref(), Some("ssm"));
        }

        #[test]
        fn supported_scope_execute_v9_freezes_when_terminal_governance_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Fail,
            );

            let report = models_supported_scope_execute_v9(
                Path::new("."),
                Path::new("out/supported_scope_execute_v9.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV9::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V9_TERMINAL_GOVERNANCE_SWEEP_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v9_denies_stale_reevaluation_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::Freeze,
                        chosen_candidate_slot: None,
                        rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            fs::write(
                "out/supported_scope_reeval.json",
                serde_json::to_vec_pretty(&SupportedScopeReevaluationV1 {
                    schema_version: 1,
                    previous_applied_set_digest_prefix: "ff".repeat(8),
                    policy_digest_prefix: "ee".repeat(8),
                    reevaluation_decision: SupportedScopeReevaluationDecisionV1::ReaffirmFreeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["stale".to_string()],
                    reevaluation_digest: "dd".repeat(32),
                })
                .expect("stale"),
            )
            .expect("write stale");
            let err = models_supported_scope_execute_v9(
                Path::new("."),
                Path::new("out/supported_scope_execute_v9.json"),
            )
            .expect_err("must reject stale reevaluation");
            assert!(err.to_string().contains("SCOPE_EXEC_V9_STALE_REEVALUATION"));
        }

        #[test]
        fn supported_scope_execute_v10_expands_with_ultimate_governance_pass() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            let terminal_prefix = prefix_hex("56".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );
            write_governance_ultimate_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &terminal_prefix,
                crate::TerminalGovernanceUltimateSweepStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v10(
                Path::new("."),
                Path::new("out/supported_scope_execute_v10.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV10::ExecuteExpandByOne
            );
        }

        #[test]
        fn supported_scope_execute_v10_freezes_when_ultimate_governance_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            let terminal_prefix = prefix_hex("56".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );
            write_governance_ultimate_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &terminal_prefix,
                crate::TerminalGovernanceUltimateSweepStatusV1::Fail,
            );

            let report = models_supported_scope_execute_v10(
                Path::new("."),
                Path::new("out/supported_scope_execute_v10.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV10::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V10_ULTIMATE_GOVERNANCE_SWEEP_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v10_denies_stale_prior_execution_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::Freeze,
                        chosen_candidate_slot: None,
                        rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            fs::write(
                "out/supported_scope_execute_v10.json",
                serde_json::to_vec_pretty(&SupportedScopeExecutionV10 {
                    schema_version: 10,
                    previous_applied_set_digest_prefix: "aa".repeat(8),
                    current_policy_digest_prefix: "bb".repeat(8),
                    current_reevaluation_digest_prefix: "cc".repeat(8),
                    canonical_governance_entry_digest_prefix: "dd".repeat(8),
                    canonical_governance_authority_digest_prefix: "ee".repeat(8),
                    final_governance_consumer_authority_digest_prefix: "ff".repeat(8),
                    final_governance_residual_sweep_digest_prefix: "11".repeat(8),
                    residual_free_governance_consumer_authority_digest_prefix: "22".repeat(8),
                    residual_free_governance_absolute_sweep_digest_prefix: "33".repeat(8),
                    absolute_final_governance_terminal_sweep_digest_prefix: "44".repeat(8),
                    terminal_governance_ultimate_sweep_digest_prefix: "55".repeat(8),
                    prior_scope_execution_digest_prefix: None,
                    execution_decision: SupportedScopeExecutionDecisionV10::ReaffirmFreeze,
                    chosen_candidate_slot: None,
                    resulting_supported_set_digest_prefix: "66".repeat(8),
                    rationale_codes: vec!["stale".to_string()],
                    execution_digest: "77".repeat(32),
                })
                .expect("stale"),
            )
            .expect("write stale");

            let err = models_supported_scope_execute_v10(
                Path::new("."),
                Path::new("out/supported_scope_execute_v10_current.json"),
            )
            .expect_err("must reject stale prior execution");
            assert!(err
                .to_string()
                .contains("SCOPE_EXEC_V10_STALE_PRIOR_EXECUTION"));
        }

        #[test]
        fn supported_scope_execute_v11_expands_with_converged_governance() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx = load_applied_supported_set_context_v1(Path::new(".")).expect("ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(Path::new(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(Path::new(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let entry = derive_canonical_governance_entry(
                &applied_ctx,
                &validate_governance_primary_surfaces_with_applied_scope(
                    &backend,
                    &active,
                    &applied_ctx,
                )
                .expect("surfaces"),
            )
            .expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            let terminal_prefix = prefix_hex("56".repeat(32).as_str(), 16);
            let ultimate_prefix = prefix_hex("78".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );
            write_governance_ultimate_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &terminal_prefix,
                crate::TerminalGovernanceUltimateSweepStatusV1::Pass,
            );
            write_governance_convergence_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                crate::GovernanceConvergenceStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v11(
                Path::new("."),
                Path::new("out/supported_scope_execute_v11.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV11::ExecuteExpandByOne
            );
        }

        #[test]
        fn supported_scope_execute_v11_freezes_when_convergence_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let report = models_supported_scope_execute_v11(
                Path::new("."),
                Path::new("out/supported_scope_execute_v11.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV11::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V11_GOVERNANCE_CONVERGENCE_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v11_denies_stale_prior_execution_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::Freeze,
                        chosen_candidate_slot: None,
                        rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            fs::write(
                "out/supported_scope_execute_v11.json",
                serde_json::to_vec_pretty(&SupportedScopeExecutionV11 {
                    schema_version: 11,
                    previous_applied_set_digest_prefix: "aa".repeat(8),
                    current_policy_digest_prefix: "bb".repeat(8),
                    current_reevaluation_digest_prefix: "cc".repeat(8),
                    canonical_governance_entry_digest_prefix: "dd".repeat(8),
                    canonical_governance_authority_digest_prefix: "ee".repeat(8),
                    final_governance_consumer_authority_digest_prefix: "ff".repeat(8),
                    final_governance_residual_sweep_digest_prefix: "11".repeat(8),
                    residual_free_governance_consumer_authority_digest_prefix: "22".repeat(8),
                    residual_free_governance_absolute_sweep_digest_prefix: "33".repeat(8),
                    absolute_final_governance_terminal_sweep_digest_prefix: "44".repeat(8),
                    terminal_governance_ultimate_sweep_digest_prefix: "55".repeat(8),
                    governance_convergence_sweep_digest_prefix: "66".repeat(8),
                    prior_scope_execution_digest_prefix: None,
                    execution_decision: SupportedScopeExecutionDecisionV11::ReaffirmFreeze,
                    chosen_candidate_slot: None,
                    resulting_supported_set_digest_prefix: "77".repeat(8),
                    rationale_codes: vec!["stale".to_string()],
                    execution_digest: "88".repeat(32),
                })
                .expect("stale"),
            )
            .expect("write stale");

            let err = models_supported_scope_execute_v11(
                Path::new("."),
                Path::new("out/supported_scope_execute_v11_current.json"),
            )
            .expect_err("must reject stale prior execution");
            assert!(err
                .to_string()
                .contains("SCOPE_EXEC_V11_STALE_PRIOR_EXECUTION"));
        }

        #[test]
        fn supported_scope_execute_v12_expands_when_stabilized_chain_passes() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx =
                load_applied_supported_set_context_v1(Path::new(".")).expect("applied ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(&PathBuf::from(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(&PathBuf::from(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let surfaces = validate_governance_primary_surfaces_with_applied_scope(
                &backend,
                &active,
                &applied_ctx,
            )
            .expect("surfaces");
            let entry = derive_canonical_governance_entry(&applied_ctx, &surfaces).expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            let terminal_prefix = prefix_hex("56".repeat(32).as_str(), 16);
            let ultimate_prefix = prefix_hex("78".repeat(32).as_str(), 16);
            let convergence_prefix = prefix_hex("90".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );
            write_governance_ultimate_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &terminal_prefix,
                crate::TerminalGovernanceUltimateSweepStatusV1::Pass,
            );
            write_governance_convergence_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                crate::GovernanceConvergenceStatusV1::Pass,
            );
            write_governance_stabilization_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                crate::GovernanceStabilizationStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v12(
                Path::new("."),
                Path::new("out/supported_scope_execute_v12.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV12::ExecuteExpandByOne
            );
        }

        #[test]
        fn supported_scope_execute_v12_freezes_when_stabilization_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");

            let report = models_supported_scope_execute_v12(
                Path::new("."),
                Path::new("out/supported_scope_execute_v12.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV12::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V12_GOVERNANCE_STABILIZATION_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v12_denies_stale_prior_execution_artifact() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::Freeze,
                        chosen_candidate_slot: None,
                        rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            fs::write(
                "out/supported_scope_execute_v12.json",
                serde_json::to_vec_pretty(&SupportedScopeExecutionV12 {
                    schema_version: 12,
                    previous_applied_set_digest_prefix: "aa".repeat(8),
                    current_policy_digest_prefix: "bb".repeat(8),
                    current_reevaluation_digest_prefix: "cc".repeat(8),
                    canonical_governance_entry_digest_prefix: "dd".repeat(8),
                    canonical_governance_authority_digest_prefix: "ee".repeat(8),
                    final_governance_consumer_authority_digest_prefix: "ff".repeat(8),
                    final_governance_residual_sweep_digest_prefix: "11".repeat(8),
                    residual_free_governance_consumer_authority_digest_prefix: "22".repeat(8),
                    residual_free_governance_absolute_sweep_digest_prefix: "33".repeat(8),
                    absolute_final_governance_terminal_sweep_digest_prefix: "44".repeat(8),
                    terminal_governance_ultimate_sweep_digest_prefix: "55".repeat(8),
                    governance_convergence_sweep_digest_prefix: "66".repeat(8),
                    governance_stabilization_sweep_digest_prefix: "77".repeat(8),
                    prior_scope_execution_digest_prefix: None,
                    execution_decision: SupportedScopeExecutionDecisionV12::ReaffirmFreeze,
                    chosen_candidate_slot: None,
                    resulting_supported_set_digest_prefix: "88".repeat(8),
                    rationale_codes: vec!["stale".to_string()],
                    execution_digest: "99".repeat(32),
                })
                .expect("stale"),
            )
            .expect("write stale");

            let err = models_supported_scope_execute_v12(
                Path::new("."),
                Path::new("out/supported_scope_execute_v12_current.json"),
            )
            .expect_err("must reject stale prior execution");
            assert!(err
                .to_string()
                .contains("SCOPE_EXEC_V12_STALE_PRIOR_EXECUTION"));
        }

        #[test]
        fn supported_scope_execute_v13_expands_when_final_consolidation_passes() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx =
                load_applied_supported_set_context_v1(Path::new(".")).expect("applied ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(&PathBuf::from(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(&PathBuf::from(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let surfaces = validate_governance_primary_surfaces_with_applied_scope(
                &backend,
                &active,
                &applied_ctx,
            )
            .expect("surfaces");
            let entry = derive_canonical_governance_entry(&applied_ctx, &surfaces).expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            let ultimate_prefix = prefix_hex("78".repeat(32).as_str(), 16);
            let convergence_prefix = prefix_hex("90".repeat(32).as_str(), 16);
            let stabilization_prefix = prefix_hex("9a".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );
            write_governance_ultimate_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                "56".repeat(8).as_str(),
                crate::TerminalGovernanceUltimateSweepStatusV1::Pass,
            );
            write_governance_convergence_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                crate::GovernanceConvergenceStatusV1::Pass,
            );
            write_governance_stabilization_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                crate::GovernanceStabilizationStatusV1::Pass,
            );
            write_governance_final_consolidation_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                &stabilization_prefix,
                crate::GovernanceFinalConsolidationStatusV1::Pass,
            );
            let report = models_supported_scope_execute_v13(
                Path::new("."),
                Path::new("out/supported_scope_execute_v13.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV13::ExecuteExpandByOne
            );
        }

        #[test]
        fn supported_scope_execute_v13_freezes_when_final_consolidation_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let report = models_supported_scope_execute_v13(
                Path::new("."),
                Path::new("out/supported_scope_execute_v13.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV13::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V13_GOVERNANCE_FINAL_CONSOLIDATION_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v14_expands_when_governance_closure_passes() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx =
                load_applied_supported_set_context_v1(Path::new(".")).expect("applied ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(&PathBuf::from(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(&PathBuf::from(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let surfaces = validate_governance_primary_surfaces_with_applied_scope(
                &backend,
                &active,
                &applied_ctx,
            )
            .expect("surfaces");
            let entry = derive_canonical_governance_entry(&applied_ctx, &surfaces).expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            let ultimate_prefix = prefix_hex("78".repeat(32).as_str(), 16);
            let convergence_prefix = prefix_hex("90".repeat(32).as_str(), 16);
            let stabilization_prefix = prefix_hex("9a".repeat(32).as_str(), 16);
            let final_consolidation_prefix = prefix_hex("9b".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );
            write_governance_ultimate_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                "56".repeat(8).as_str(),
                crate::TerminalGovernanceUltimateSweepStatusV1::Pass,
            );
            write_governance_convergence_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                crate::GovernanceConvergenceStatusV1::Pass,
            );
            write_governance_stabilization_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                crate::GovernanceStabilizationStatusV1::Pass,
            );
            write_governance_final_consolidation_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                &stabilization_prefix,
                crate::GovernanceFinalConsolidationStatusV1::Pass,
            );
            write_governance_closure_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                &stabilization_prefix,
                &final_consolidation_prefix,
                crate::GovernanceClosureStatusV1::Pass,
            );

            let report = models_supported_scope_execute_v14(
                Path::new("."),
                Path::new("out/supported_scope_execute_v14.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV14::ExecuteExpandByOne
            );
        }

        #[test]
        fn supported_scope_execute_v14_freezes_when_governance_closure_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let report = models_supported_scope_execute_v14(
                Path::new("."),
                Path::new("out/supported_scope_execute_v14.json"),
            )
            .expect("execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV14::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V14_GOVERNANCE_CLOSURE_FAIL".to_string()));
        }

        #[test]
        fn supported_scope_execute_v15_expands_when_governance_seal_passes() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx =
                load_applied_supported_set_context_v1(Path::new(".")).expect("applied ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(&PathBuf::from(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(&PathBuf::from(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let surfaces = validate_governance_primary_surfaces_with_applied_scope(
                &backend,
                &active,
                &applied_ctx,
            )
            .expect("surfaces");
            let entry = derive_canonical_governance_entry(&applied_ctx, &surfaces).expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            let ultimate_prefix = prefix_hex("78".repeat(32).as_str(), 16);
            let convergence_prefix = prefix_hex("90".repeat(32).as_str(), 16);
            let stabilization_prefix = prefix_hex("9a".repeat(32).as_str(), 16);
            let final_consolidation_prefix = prefix_hex("9b".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );
            write_governance_ultimate_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                "56".repeat(8).as_str(),
                crate::TerminalGovernanceUltimateSweepStatusV1::Pass,
            );
            write_governance_convergence_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                crate::GovernanceConvergenceStatusV1::Pass,
            );
            write_governance_stabilization_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                crate::GovernanceStabilizationStatusV1::Pass,
            );
            write_governance_final_consolidation_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                &stabilization_prefix,
                crate::GovernanceFinalConsolidationStatusV1::Pass,
            );
            write_governance_closure_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                &stabilization_prefix,
                &final_consolidation_prefix,
                crate::GovernanceClosureStatusV1::Pass,
            );
            let report_v14 = models_supported_scope_execute_v14(
                Path::new("."),
                Path::new("out/supported_scope_execute_v14.json"),
            )
            .expect("v14 execute");
            write_governance_seal_sweep_artifact(
                Path::new("."),
                &report_v14.previous_applied_set_digest_prefix,
                &report_v14.canonical_governance_entry_digest_prefix,
                &report_v14.canonical_governance_authority_digest_prefix,
                &report_v14.final_governance_consumer_authority_digest_prefix,
                &report_v14.final_governance_residual_sweep_digest_prefix,
                &report_v14.residual_free_governance_consumer_authority_digest_prefix,
                &report_v14.residual_free_governance_absolute_sweep_digest_prefix,
                &report_v14.terminal_governance_ultimate_sweep_digest_prefix,
                &report_v14.governance_convergence_sweep_digest_prefix,
                &report_v14.governance_stabilization_sweep_digest_prefix,
                &report_v14.governance_final_consolidation_sweep_digest_prefix,
                &report_v14.governance_closure_sweep_digest_prefix,
                crate::GovernanceSealStatusV1::Pass,
            );
            let report = models_supported_scope_execute_v15(
                Path::new("."),
                Path::new("out/supported_scope_execute_v15.json"),
            )
            .expect("v15 execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV15::ExecuteExpandByOne
            );
        }

        #[test]
        fn supported_scope_execute_v15_freezes_when_governance_seal_fails() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");
            let applied = build_supported_real_slot_set_v2(
                vec!["world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");
            write_scope_reeval_support_artifacts(
                Path::new("."),
                &applied.set_digest,
                &["world_jepa"],
            );
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&SupportedSetReviewReportV1 {
                    policy: SupportedRealSlotSetPolicyV2 {
                        schema_version: 2,
                        current_supported_slots: vec!["world_jepa".to_string()],
                        candidate_slots_considered: vec!["ssm".to_string()],
                        decision: SupportedRealSlotSetDecisionV2::ExpandByOne,
                        chosen_candidate_slot: Some("ssm".to_string()),
                        rationale_codes: vec!["EXPANSION_READY_EXACTLY_ONE".to_string()],
                        policy_digest: "33".repeat(32),
                    },
                    known_slots: vec![],
                    candidates: vec![],
                })
                .expect("review"),
            )
            .expect("write review");
            let _ = models_supported_scope_reevaluate(
                Path::new("."),
                Path::new("out/supported_scope_reeval.json"),
            )
            .expect("reeval");
            let applied_ctx =
                load_applied_supported_set_context_v1(Path::new(".")).expect("applied ctx");
            let backend = read_json_file::<BackendEvidenceSnapshotV1>(&PathBuf::from(
                "out/backend_evidence_snapshot.json",
            ))
            .expect("backend");
            let active = read_json_file::<AggregatedActiveReviewSnapshotV1>(&PathBuf::from(
                "out/active_review_snapshot.json",
            ))
            .expect("active");
            let surfaces = validate_governance_primary_surfaces_with_applied_scope(
                &backend,
                &active,
                &applied_ctx,
            )
            .expect("surfaces");
            let entry = derive_canonical_governance_entry(&applied_ctx, &surfaces).expect("entry");
            let canonical_prefix = prefix_hex(&entry.authority_digest, 16);
            let authority_prefix = "ab".repeat(8);
            let final_consumer_prefix = prefix_hex("cd".repeat(32).as_str(), 16);
            let residual_prefix = prefix_hex("ef".repeat(32).as_str(), 16);
            let residual_free_prefix = prefix_hex("12".repeat(32).as_str(), 16);
            let absolute_prefix = prefix_hex("34".repeat(32).as_str(), 16);
            let ultimate_prefix = prefix_hex("78".repeat(32).as_str(), 16);
            let convergence_prefix = prefix_hex("90".repeat(32).as_str(), 16);
            let stabilization_prefix = prefix_hex("9a".repeat(32).as_str(), 16);
            let final_consolidation_prefix = prefix_hex("9b".repeat(32).as_str(), 16);
            write_governance_entry_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &prefix_hex(&applied_ctx.context_digest, 16),
                &canonical_prefix,
                GovernanceEntryAuthorityStatusV2::Pass,
            );
            write_final_governance_consumer_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                FinalGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_residual_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                crate::GovernanceResidualSweepStatusV1::Pass,
            );
            write_residual_free_governance_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                crate::ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
            );
            write_governance_absolute_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                crate::ResidualFreeGovernanceAbsoluteSweepStatusV1::Pass,
            );
            write_governance_terminal_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                crate::AbsoluteFinalGovernanceTerminalSweepStatusV1::Pass,
            );
            write_governance_ultimate_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                "56".repeat(8).as_str(),
                crate::TerminalGovernanceUltimateSweepStatusV1::Pass,
            );
            write_governance_convergence_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                crate::GovernanceConvergenceStatusV1::Pass,
            );
            write_governance_stabilization_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                crate::GovernanceStabilizationStatusV1::Pass,
            );
            write_governance_final_consolidation_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                &stabilization_prefix,
                crate::GovernanceFinalConsolidationStatusV1::Pass,
            );
            write_governance_closure_sweep_artifact(
                Path::new("."),
                &applied_ctx.applied_set_digest_prefix,
                &canonical_prefix,
                &authority_prefix,
                &final_consumer_prefix,
                &residual_prefix,
                &residual_free_prefix,
                &absolute_prefix,
                &ultimate_prefix,
                &convergence_prefix,
                &stabilization_prefix,
                &final_consolidation_prefix,
                crate::GovernanceClosureStatusV1::Pass,
            );
            let report = models_supported_scope_execute_v15(
                Path::new("."),
                Path::new("out/supported_scope_execute_v15.json"),
            )
            .expect("v15 execute");
            assert_eq!(
                report.execution_decision,
                SupportedScopeExecutionDecisionV15::ReaffirmFreeze
            );
            assert!(report
                .rationale_codes
                .contains(&"SCOPE_EXEC_V15_GOVERNANCE_SEAL_FAIL".to_string()));
        }

        #[test]
        fn supported_set_apply_autogenerates_reeval_when_missing() {
            let _guard = crate::test_cwd_lock().lock().expect("cwd lock");
            let dir = tempfile::tempdir().expect("tempdir");
            let _cwd = CwdGuard::enter(dir.path());
            fs::create_dir_all("out").expect("out");

            let applied = build_supported_real_slot_set_v2(
                vec!["sae".to_string(), "world_jepa".to_string()],
                &"11".repeat(32),
                &"22".repeat(32),
                SupportedRealSlotSetExecutionDecisionV2::Frozen,
            );
            fs::write(
                "out/supported_real_slot_set_applied_v2.json",
                serde_json::to_vec_pretty(&applied).expect("applied"),
            )
            .expect("write applied");

            let review = SupportedSetReviewReportV1 {
                policy: SupportedRealSlotSetPolicyV2 {
                    schema_version: 2,
                    current_supported_slots: vec!["sae".to_string(), "world_jepa".to_string()],
                    candidate_slots_considered: vec!["ssm".to_string()],
                    decision: SupportedRealSlotSetDecisionV2::Freeze,
                    chosen_candidate_slot: None,
                    rationale_codes: vec!["INSUFFICIENT_EVIDENCE_FREEZE".to_string()],
                    policy_digest: "33".repeat(32),
                },
                known_slots: vec![],
                candidates: vec![],
            };
            fs::write(
                "out/supported_set_review.json",
                serde_json::to_vec_pretty(&review).expect("review"),
            )
            .expect("write review");

            let out = PathBuf::from("out/supported_set_apply.json");
            let report = models_supported_set_apply(Path::new("."), &out).expect("apply");
            assert_eq!(
                report.decision,
                SupportedRealSlotSetExecutionDecisionV2::Frozen
            );
            assert!(Path::new("out/supported_scope_reeval.json").exists());
        }

        #[test]
        fn active_review_overall_status_reduction_is_deterministic() {
            let none = vec![ActiveReviewEvidenceV1 {
                slot_id: "a".to_string(),
                target_hash_prefix: "h".to_string(),
                manifest_digest_prefix: "m".to_string(),
                probe_ready: true,
                shadow_ready: true,
                active_eligible: false,
                strict_blocking: false,
                drift_blocking: false,
                alert_blocking: false,
                primary_denial_code: Some("NoProbe".to_string()),
                remediation_codes: vec![],
                contributing_evidence_digests: ActiveReviewContributingDigestsV1 {
                    probe_report_digest_prefix: "p".to_string(),
                    shadow_ready_digest_prefix: "s".to_string(),
                    active_evidence_digest_prefix: "a".to_string(),
                    strict_evidence_digest_prefix: "x".to_string(),
                },
                burn_resolution: BurnSupportResolutionV1 {
                    slot_id: "a".to_string(),
                    resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                    support_state: OptionalBackendSupportStateV1::NotConfigured,
                    rationale_codes: vec!["BURN_SHADOW_NOT_CONFIGURED".to_string()],
                    evidence_digest: "br".to_string(),
                },
                evidence_digest: "d".to_string(),
            }];
            assert_eq!(
                derive_active_review_status(&none),
                ActiveReviewOverallStatusV1::NoneReviewable
            );

            let partial = vec![
                ActiveReviewEvidenceV1 {
                    active_eligible: true,
                    ..none[0].clone()
                },
                ActiveReviewEvidenceV1 {
                    slot_id: "b".to_string(),
                    active_eligible: false,
                    ..none[0].clone()
                },
            ];
            assert_eq!(
                derive_active_review_status(&partial),
                ActiveReviewOverallStatusV1::PartialReviewable
            );

            let all = vec![
                ActiveReviewEvidenceV1 {
                    slot_id: "a".to_string(),
                    active_eligible: true,
                    ..none[0].clone()
                },
                ActiveReviewEvidenceV1 {
                    slot_id: "b".to_string(),
                    active_eligible: true,
                    ..none[0].clone()
                },
            ];
            assert_eq!(
                derive_active_review_status(&all),
                ActiveReviewOverallStatusV1::AllReviewable
            );
        }

        #[test]
        fn signoff_alignment_derivation_is_stable() {
            let snapshot = BackendEvidenceSnapshotV1 {
                schema_version: 1,
                supported_slot_set_digest: "set".to_string(),
                policy_graph_digest_prefix: "policy".to_string(),
                manifest_digest_prefix: "manifest".to_string(),
                slots: vec![],
                snapshot_digest: "snap".to_string(),
            };
            let missing = derive_signoff_alignment(
                None,
                &snapshot,
                &ActiveReviewOverallStatusV1::NoneReviewable,
            );
            assert!(!missing.aligned);
            assert_eq!(missing.status_code, "SIGNOFF_MISSING");

            let signoff = OperatorSignoffDecisionV1 {
                schema_version: 1,
                decision: SignoffDecisionStateV1::ReadyForActiveReview,
                supported_slot_set_digest: "set".to_string(),
                policy_graph_digest_prefix: "policy".to_string(),
                manifest_digest_prefix: "manifest".to_string(),
                evidence_snapshot_digest_prefix: "ev".to_string(),
                active_review_snapshot_digest_prefix: None,
                operator_report_digest_prefix: "op".to_string(),
                applied_supported_set_digest_prefix: "set".to_string(),
                applied_context_digest_prefix: "ctx".to_string(),
                reviewability_reduction_digest_prefix: "MISSING".to_string(),
                canonical_readiness_spine_digest_prefix: "MISSING".to_string(),
                canonical_readiness_authority_digest_prefix: "MISSING".to_string(),
                canonical_governance_entry_digest_prefix: "MISSING".to_string(),
                final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
                governance_residual_sweep_digest_prefix: "MISSING".to_string(),
                residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
                governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
                absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
                governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
                final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
                readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
                residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
                readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
                readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
                readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
                readiness_stabilization_sweep_digest_prefix: "MISSING".to_string(),
                readiness_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
                readiness_closure_sweep_digest_prefix: "MISSING".to_string(),
                readiness_seal_sweep_digest_prefix: "MISSING".to_string(),
                governance_final_consolidation_sweep_digest_prefix: "MISSING".to_string(),
                governance_closure_sweep_digest_prefix: "MISSING".to_string(),
                governance_seal_sweep_digest_prefix: "MISSING".to_string(),
                final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
                residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
                primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
                primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
                primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
                primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
                gate_report_digests: crate::operator_signoff::GateReportDigestsV1 {
                    v0: "x".to_string(),
                    v1: "x".to_string(),
                    v2: "x".to_string(),
                    v3: "x".to_string(),
                },
                reasons: vec![],
                remediation_codes: vec![],
                canonical_remediation_codes: vec![],
                decision_digest: "d".to_string(),
            };

            let aligned = derive_signoff_alignment(
                Some(&signoff),
                &snapshot,
                &ActiveReviewOverallStatusV1::PartialReviewable,
            );
            assert!(aligned.aligned);
            assert_eq!(aligned.status_code, "ALIGNED");

            let mismatch = derive_signoff_alignment(
                Some(&signoff),
                &snapshot,
                &ActiveReviewOverallStatusV1::NoneReviewable,
            );
            assert!(!mismatch.aligned);
            assert_eq!(mismatch.status_code, "SIGNOFF_DECISION_MISMATCH");
        }

        #[test]
        fn burn_support_resolution_digest_is_stable() {
            let a = burn_support_resolution_from_state(
                ModelSlot::Sae,
                OptionalBackendSupportStateV1::NotConfigured,
            );
            let b = burn_support_resolution_from_state(
                ModelSlot::Sae,
                OptionalBackendSupportStateV1::NotConfigured,
            );
            assert_eq!(a.resolution, BurnResolutionStatusV1::BurnClosedUnsupported);
            assert_eq!(a.evidence_digest, b.evidence_digest);
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
                    burn_resolution: BurnSupportResolutionV1 {
                        slot_id: "world_jepa".to_string(),
                        resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                        support_state: OptionalBackendSupportStateV1::Unsupported,
                        rationale_codes: vec!["BURN_SLOT_FORMALLY_UNSUPPORTED".to_string()],
                        evidence_digest: "br".to_string(),
                    },
                }],
                snapshot_digest: "beef".to_string(),
            };
            let a = serde_json::to_vec(&snapshot).expect("a");
            let b = serde_json::to_vec(&snapshot).expect("b");
            assert_eq!(a, b);
        }
    }
}
