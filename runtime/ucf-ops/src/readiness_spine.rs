use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, derive_slot_reviewability_truths,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_report, operator_review_packet, operator_signoff, operator_workflow_chain, prefix_hex,
    reduce_reviewability, require_canonical_governance_entry, resolve_strict_evidence,
    validate_governance_primary_surfaces_with_applied_scope, AppliedSupportedSetContextV1,
    CanonicalGovernanceEntryV1, OperatorReportArgs, OperatorReviewPacketArgs,
    OperatorReviewPacketV1, OperatorSignoffArgs, OperatorSignoffDecisionV1, OperatorWorkflowArgs,
    OperatorWorkflowChainV1, OpsError, ReviewabilityReductionV1, SlotReviewabilityTruthV1,
    StrictEvidenceContextV1,
};

pub const CANONICAL_READINESS_SPINE_REQUIRED: &str = "CANONICAL_READINESS_SPINE_REQUIRED";
pub const FINAL_READINESS_AUTHORITY_REQUIRED: &str = "FINAL_READINESS_AUTHORITY_REQUIRED";
pub const FINAL_READINESS_INPUTS_REQUIRED: &str = "FINAL_READINESS_INPUTS_REQUIRED";
pub const RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED: &str =
    "RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED";
pub const SLOT_REVIEWABILITY_TRUTH_REQUIRED: &str = "SLOT_REVIEWABILITY_TRUTH_REQUIRED";
pub const REVIEWABILITY_REDUCTION_REQUIRED: &str = "REVIEWABILITY_REDUCTION_REQUIRED";
pub const LEGACY_READINESS_INPUT_BLOCKED: &str = "LEGACY_READINESS_INPUT_BLOCKED";
pub const SECONDARY_READINESS_PATH_BLOCKED: &str = "SECONDARY_READINESS_PATH_BLOCKED";
pub const RESIDUAL_READINESS_PATH_BLOCKED: &str = "RESIDUAL_READINESS_PATH_BLOCKED";
pub const HISTORICAL_READINESS_PATH_BLOCKED: &str = "HISTORICAL_READINESS_PATH_BLOCKED";
pub const HISTORICAL_READINESS_PATH_TRANSLATED: &str = "HISTORICAL_READINESS_PATH_TRANSLATED";
pub const HISTORICAL_READINESS_PATH_REJECTED: &str = "HISTORICAL_READINESS_PATH_REJECTED";
pub const HISTORICAL_READINESS_LINEAGE_BLOCKED: &str = "HISTORICAL_READINESS_LINEAGE_BLOCKED";
pub const ABSOLUTE_RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED: &str =
    "ABSOLUTE_RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED";
pub const READINESS_ECHO_PATH_BLOCKED: &str = "READINESS_ECHO_PATH_BLOCKED";
pub const READINESS_ECHO_PATH_TRANSLATED: &str = "READINESS_ECHO_PATH_TRANSLATED";
pub const READINESS_ECHO_PATH_REJECTED: &str = "READINESS_ECHO_PATH_REJECTED";
pub const TERMINAL_ABSOLUTE_RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED: &str =
    "TERMINAL_ABSOLUTE_RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED";
pub const READINESS_CACHE_PATH_BLOCKED: &str = "READINESS_CACHE_PATH_BLOCKED";
pub const READINESS_CACHE_PATH_TRANSLATED: &str = "READINESS_CACHE_PATH_TRANSLATED";
pub const READINESS_CACHE_PATH_REJECTED: &str = "READINESS_CACHE_PATH_REJECTED";
pub const READINESS_MEMO_PATH_BLOCKED: &str = "READINESS_MEMO_PATH_BLOCKED";
pub const READINESS_MEMO_PATH_TRANSLATED: &str = "READINESS_MEMO_PATH_TRANSLATED";
pub const READINESS_MEMO_PATH_REJECTED: &str = "READINESS_MEMO_PATH_REJECTED";
pub const ULTIMATE_TERMINAL_ABSOLUTE_READINESS_INPUTS_REQUIRED: &str =
    "ULTIMATE_TERMINAL_ABSOLUTE_READINESS_INPUTS_REQUIRED";
pub const CONVERGED_CANONICAL_READINESS_INPUTS_REQUIRED: &str =
    "CONVERGED_CANONICAL_READINESS_INPUTS_REQUIRED";
pub const STABILIZED_CONVERGED_CANONICAL_READINESS_INPUTS_REQUIRED: &str =
    "STABILIZED_CONVERGED_CANONICAL_READINESS_INPUTS_REQUIRED";
pub const READINESS_ADAPTER_PATH_BLOCKED: &str = "READINESS_ADAPTER_PATH_BLOCKED";
pub const READINESS_ADAPTER_PATH_TRANSLATED: &str = "READINESS_ADAPTER_PATH_TRANSLATED";
pub const READINESS_ADAPTER_PATH_REJECTED: &str = "READINESS_ADAPTER_PATH_REJECTED";
pub const READINESS_FACADE_PATH_BLOCKED: &str = "READINESS_FACADE_PATH_BLOCKED";
pub const READINESS_FACADE_PATH_TRANSLATED: &str = "READINESS_FACADE_PATH_TRANSLATED";
pub const READINESS_FACADE_PATH_REJECTED: &str = "READINESS_FACADE_PATH_REJECTED";
pub const FINAL_CONSOLIDATED_STABILIZED_CANONICAL_READINESS_INPUTS_REQUIRED: &str =
    "FINAL_CONSOLIDATED_STABILIZED_CANONICAL_READINESS_INPUTS_REQUIRED";
pub const READINESS_WRAPPER_PATH_BLOCKED: &str = "READINESS_WRAPPER_PATH_BLOCKED";
pub const READINESS_WRAPPER_PATH_TRANSLATED: &str = "READINESS_WRAPPER_PATH_TRANSLATED";
pub const READINESS_WRAPPER_PATH_REJECTED: &str = "READINESS_WRAPPER_PATH_REJECTED";
pub const CLOSURE_COMPLETE_FINAL_CONSOLIDATED_STABILIZED_CANONICAL_READINESS_INPUTS_REQUIRED: &str =
    "CLOSURE_COMPLETE_FINAL_CONSOLIDATED_STABILIZED_CANONICAL_READINESS_INPUTS_REQUIRED";
pub const READINESS_SHELL_PATH_BLOCKED: &str = "READINESS_SHELL_PATH_BLOCKED";
pub const READINESS_SHELL_PATH_TRANSLATED: &str = "READINESS_SHELL_PATH_TRANSLATED";
pub const READINESS_SHELL_PATH_REJECTED: &str = "READINESS_SHELL_PATH_REJECTED";
pub const GOVERNANCE_LOCK_INPUTS_REQUIRED: &str = "GOVERNANCE_LOCK_INPUTS_REQUIRED";
pub const SUPPORTED_SCOPE_DECISION_REQUIRED: &str = "SUPPORTED_SCOPE_DECISION_REQUIRED";
pub const CANONICAL_EXECUTION_REALITY_REQUIRED: &str = "CANONICAL_EXECUTION_REALITY_REQUIRED";
pub const READINESS_INFLATION_PATH_BLOCKED: &str = "READINESS_INFLATION_PATH_BLOCKED";
pub const READINESS_SCOPE_OVERRUN: &str = "READINESS_SCOPE_OVERRUN";
pub const READINESS_AUXILIARY_PATH_REJECTED: &str = "READINESS_AUXILIARY_PATH_REJECTED";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalReadinessSpineStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalReadinessSpineV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub slot_truths_digest_prefix: String,
    pub reviewability_reduction_digest_prefix: String,
    pub active_review_snapshot_digest_prefix: Option<String>,
    pub signoff_digest_prefix: Option<String>,
    pub review_packet_digest_prefix: Option<String>,
    pub workflow_chain_digest_prefix: Option<String>,
    pub spine_status: CanonicalReadinessSpineStatusV1,
    pub spine_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessSpineMismatchCategoryV1 {
    SlotTruthMismatch,
    ReductionMismatch,
    SignoffSpineDrift,
    ReviewPacketSpineDrift,
    WorkflowSpineDrift,
    AppliedScopeSpineMismatch,
    LegacyReadinessField,
    LegacyReadinessTranslated,
    LegacyReadinessRejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessSpineCheckStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessSpineCheckReportV1 {
    pub schema_version: u16,
    pub status: ReadinessSpineCheckStatusV1,
    pub mismatch_categories: Vec<ReadinessSpineMismatchCategoryV1>,
    pub remediation_codes: Vec<String>,
    pub canonical_readiness_spine: CanonicalReadinessSpineV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalReadinessAuthorityStatusV2 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalReadinessAuthorityV2 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub covered_surface_count: u16,
    pub authority_status: CanonicalReadinessAuthorityStatusV2,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalReadinessAuthorityContextV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalReadinessInputsContextV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeFinalReadinessInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeReadinessAbsoluteInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AbsoluteFinalReadinessTerminalInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub residual_free_readiness_absolute_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminalReadinessUltimateInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub residual_free_readiness_absolute_sweep_digest_prefix: String,
    pub absolute_final_readiness_terminal_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessConvergenceInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub residual_free_readiness_absolute_sweep_digest_prefix: String,
    pub absolute_final_readiness_terminal_sweep_digest_prefix: String,
    pub terminal_readiness_ultimate_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessStabilizationInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub residual_free_readiness_absolute_sweep_digest_prefix: String,
    pub absolute_final_readiness_terminal_sweep_digest_prefix: String,
    pub terminal_readiness_ultimate_sweep_digest_prefix: String,
    pub readiness_convergence_sweep_digest_prefix: String,
    pub readiness_stabilization_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessFinalConsolidationInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub residual_free_readiness_absolute_sweep_digest_prefix: String,
    pub absolute_final_readiness_terminal_sweep_digest_prefix: String,
    pub terminal_readiness_ultimate_sweep_digest_prefix: String,
    pub readiness_convergence_sweep_digest_prefix: String,
    pub readiness_stabilization_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessClosureInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub residual_free_readiness_absolute_sweep_digest_prefix: String,
    pub absolute_final_readiness_terminal_sweep_digest_prefix: String,
    pub terminal_readiness_ultimate_sweep_digest_prefix: String,
    pub readiness_convergence_sweep_digest_prefix: String,
    pub readiness_stabilization_sweep_digest_prefix: String,
    pub readiness_final_consolidation_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessSealInputsV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub residual_free_readiness_absolute_sweep_digest_prefix: String,
    pub absolute_final_readiness_terminal_sweep_digest_prefix: String,
    pub terminal_readiness_ultimate_sweep_digest_prefix: String,
    pub readiness_convergence_sweep_digest_prefix: String,
    pub readiness_stabilization_sweep_digest_prefix: String,
    pub readiness_final_consolidation_sweep_digest_prefix: String,
    pub readiness_closure_sweep_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessLockInputsV1 {
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
    pub supported_scope_decision_digest_prefix: String,
    pub canonical_execution_reality_digest_prefix: String,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessSpineSweepMismatchCategoryV1 {
    SurfaceSkippedCanonicalReadinessSpine,
    SurfaceUsedSecondaryReadinessPath,
    ReadinessSpineScopeMismatch,
    ReadinessSpineGovernanceMismatch,
    LegacyReadinessPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessSpineSweepSurfaceStatusV1 {
    pub surface: String,
    pub status: CanonicalReadinessAuthorityStatusV2,
    pub mismatch_categories: Vec<ReadinessSpineSweepMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessSpineSweepReportV1 {
    pub schema_version: u16,
    pub authority: CanonicalReadinessAuthorityV2,
    pub surfaces: Vec<ReadinessSpineSweepSurfaceStatusV1>,
}

#[allow(clippy::too_many_arguments)]
pub fn derive_canonical_readiness_spine(
    applied_scope: &AppliedSupportedSetContextV1,
    entry: &CanonicalGovernanceEntryV1,
    truths: &[SlotReviewabilityTruthV1],
    reduction: &ReviewabilityReductionV1,
    active_digest: Option<&str>,
    signoff_digest: Option<&str>,
    review_packet_digest: Option<&str>,
    workflow_digest: Option<&str>,
) -> Result<CanonicalReadinessSpineV1, OpsError> {
    let truths_digest = sha_truths(truths);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(applied_scope.applied_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(entry.authority_digest.as_bytes());
    bytes.extend_from_slice(truths_digest.as_bytes());
    bytes.extend_from_slice(reduction.reduction_digest.as_bytes());
    bytes.extend_from_slice(active_digest.unwrap_or("MISSING").as_bytes());
    bytes.extend_from_slice(signoff_digest.unwrap_or("MISSING").as_bytes());
    bytes.extend_from_slice(review_packet_digest.unwrap_or("MISSING").as_bytes());
    bytes.extend_from_slice(workflow_digest.unwrap_or("MISSING").as_bytes());

    Ok(CanonicalReadinessSpineV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied_scope.applied_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: prefix_hex(&entry.authority_digest, 16),
        slot_truths_digest_prefix: prefix_hex(&truths_digest, 16),
        reviewability_reduction_digest_prefix: prefix_hex(&reduction.reduction_digest, 16),
        active_review_snapshot_digest_prefix: active_digest.map(|v| prefix_hex(v, 16)),
        signoff_digest_prefix: signoff_digest.map(|v| prefix_hex(v, 16)),
        review_packet_digest_prefix: review_packet_digest.map(|v| prefix_hex(v, 16)),
        workflow_chain_digest_prefix: workflow_digest.map(|v| prefix_hex(v, 16)),
        spine_status: CanonicalReadinessSpineStatusV1::Pass,
        spine_digest: crate::sha256_hex(&bytes),
    })
}

pub fn write_canonical_readiness_spine(
    path: &Path,
    spine: &CanonicalReadinessSpineV1,
) -> Result<(), OpsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(spine)?)?;
    Ok(())
}

pub fn require_canonical_readiness_spine(
    applied_scope: &AppliedSupportedSetContextV1,
    entry: &CanonicalGovernanceEntryV1,
    spine: Option<&CanonicalReadinessSpineV1>,
) -> Result<CanonicalReadinessSpineV1, OpsError> {
    let Some(spine) = spine else {
        return Err(OpsError::Invalid(
            CANONICAL_READINESS_SPINE_REQUIRED.to_string(),
        ));
    };
    if spine.slot_truths_digest_prefix == "MISSING" {
        return Err(OpsError::Invalid(
            SLOT_REVIEWABILITY_TRUTH_REQUIRED.to_string(),
        ));
    }
    if spine.reviewability_reduction_digest_prefix == "MISSING" {
        return Err(OpsError::Invalid(
            REVIEWABILITY_REDUCTION_REQUIRED.to_string(),
        ));
    }
    if spine.applied_supported_set_digest_prefix != applied_scope.applied_set_digest_prefix
        || spine.canonical_governance_entry_digest_prefix != prefix_hex(&entry.authority_digest, 16)
        || !matches!(spine.spine_status, CanonicalReadinessSpineStatusV1::Pass)
    {
        return Err(OpsError::Invalid(
            SECONDARY_READINESS_PATH_BLOCKED.to_string(),
        ));
    }
    Ok(spine.clone())
}

pub fn readiness_spine_check(
    workdir: &Path,
    out: &Path,
) -> Result<ReadinessSpineCheckReportV1, OpsError> {
    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_readiness_spine_check.json"),
    )?;
    let strict = resolve_strict_evidence(
        &workdir.join("out"),
        &StrictEvidenceContextV1 {
            run_id: None,
            latest: true,
            strict_required: false,
            expected_policy_graph_digest_prefix: Some(backend.policy_graph_digest_prefix.clone()),
            expected_manifest_digest_prefix: Some(backend.manifest_digest_prefix.clone()),
            expected_supported_slot_set_digest_prefix: Some(
                backend.supported_slot_set_digest.clone(),
            ),
        },
    );
    let truths = derive_slot_reviewability_truths(&applied_scope, &backend, &active, &strict)?;
    let reduction = reduce_reviewability(&applied_scope, &truths)?;
    let out_root = workdir.join("out");
    fs::create_dir_all(&out_root)?;
    fs::write(
        out_root.join("backend_evidence_snapshot.json"),
        serde_json::to_vec_pretty(&backend)?,
    )?;
    fs::write(
        out_root.join("active_review_snapshot.json"),
        serde_json::to_vec_pretty(&active)?,
    )?;
    let _operator_report = operator_report(
        workdir,
        &OperatorReportArgs {
            run_id: None,
            latest: false,
        },
        &out_root.join("operator_report.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied_scope)?;
    let entry = derive_canonical_governance_entry(&applied_scope, &surfaces)?;
    let entry = require_canonical_governance_entry(&applied_scope, Some(&entry))?;

    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: false,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_readiness_spine_check.json"),
    )?;
    fs::write(
        out_root.join("operator_signoff.json"),
        serde_json::to_vec_pretty(&signoff)?,
    )?;
    let packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: false,
        },
        &workdir.join("out/operator_review_packet_readiness_spine_check.json"),
    )?;
    fs::write(
        out_root.join("operator_review_packet.json"),
        serde_json::to_vec_pretty(&packet)?,
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: false,
        },
        &workdir.join("out/operator_workflow_chain_readiness_spine_check.json"),
    )?;

    let spine = derive_canonical_readiness_spine(
        &applied_scope,
        &entry,
        &truths,
        &reduction,
        Some(&active.snapshot_digest),
        Some(&signoff.decision_digest),
        Some(&packet.packet_digest),
        Some(&workflow.chain_digest),
    )?;

    let mut mismatches = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    if signoff.reviewability_reduction_digest_prefix != prefix_hex(&reduction.reduction_digest, 16)
    {
        mismatches.insert(ReadinessSpineMismatchCategoryV1::ReductionMismatch);
        mismatches.insert(ReadinessSpineMismatchCategoryV1::SignoffSpineDrift);
        remediation.insert("run_operator_signoff".to_string());
    }
    if packet.reviewability_reduction_digest_prefix != prefix_hex(&reduction.reduction_digest, 16) {
        mismatches.insert(ReadinessSpineMismatchCategoryV1::ReductionMismatch);
        mismatches.insert(ReadinessSpineMismatchCategoryV1::ReviewPacketSpineDrift);
        remediation.insert("run_operator_review_packet".to_string());
    }
    if workflow.reviewability_reduction_digest_prefix != prefix_hex(&reduction.reduction_digest, 16)
    {
        mismatches.insert(ReadinessSpineMismatchCategoryV1::ReductionMismatch);
        mismatches.insert(ReadinessSpineMismatchCategoryV1::WorkflowSpineDrift);
        remediation.insert("run_operator_workflow_chain".to_string());
    }
    if signoff.applied_supported_set_digest_prefix != applied_scope.applied_set_digest_prefix
        || packet.applied_supported_set_digest_prefix != applied_scope.applied_set_digest_prefix
        || workflow.applied_supported_set_digest_prefix != applied_scope.applied_set_digest_prefix
    {
        mismatches.insert(ReadinessSpineMismatchCategoryV1::AppliedScopeSpineMismatch);
        remediation.insert("run_models_applied_scope_check".to_string());
    }

    let report = ReadinessSpineCheckReportV1 {
        schema_version: 1,
        status: if mismatches.is_empty() {
            ReadinessSpineCheckStatusV1::Pass
        } else {
            ReadinessSpineCheckStatusV1::Fail
        },
        mismatch_categories: mismatches.into_iter().collect(),
        remediation_codes: remediation.into_iter().collect(),
        canonical_readiness_spine: spine,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn readiness_spine_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<ReadinessSpineSweepReportV1, OpsError> {
    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_readiness_spine_sweep.json"),
    )?;
    let strict = resolve_strict_evidence(
        &workdir.join("out"),
        &StrictEvidenceContextV1 {
            run_id: None,
            latest: true,
            strict_required: false,
            expected_policy_graph_digest_prefix: Some(backend.policy_graph_digest_prefix.clone()),
            expected_manifest_digest_prefix: Some(backend.manifest_digest_prefix.clone()),
            expected_supported_slot_set_digest_prefix: Some(
                backend.supported_slot_set_digest.clone(),
            ),
        },
    );
    let truths = derive_slot_reviewability_truths(&applied_scope, &backend, &active, &strict)?;
    let reduction = reduce_reviewability(&applied_scope, &truths)?;
    let governance =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied_scope)?;
    let entry = derive_canonical_governance_entry(&applied_scope, &governance)?;
    let entry = require_canonical_governance_entry(&applied_scope, Some(&entry))?;
    let spine = derive_canonical_readiness_spine(
        &applied_scope,
        &entry,
        &truths,
        &reduction,
        Some(&active.snapshot_digest),
        None,
        None,
        None,
    )?;
    let spine = require_canonical_readiness_spine(&applied_scope, &entry, Some(&spine))?;

    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff.json"),
    )?;
    let packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain.json"),
    )?;

    let expected_spine = signoff.canonical_readiness_spine_digest_prefix.clone();
    let expected_governance = prefix_hex(&entry.authority_digest, 16);
    let expected_scope = applied_scope.applied_set_digest_prefix.as_str();
    let surfaces = vec![
        check_surface_status(
            "OperatorSignoff",
            signoff.canonical_readiness_spine_digest_prefix != "MISSING",
            signoff.reviewability_reduction_digest_prefix == "MISSING",
            signoff.applied_supported_set_digest_prefix != expected_scope,
            false,
        ),
        check_surface_status(
            "OperatorReviewPacket",
            packet.canonical_readiness_spine_digest_prefix != "MISSING",
            packet.reviewability_reduction_digest_prefix == "MISSING",
            packet.applied_supported_set_digest_prefix != expected_scope,
            false,
        ),
        check_surface_status(
            "OperatorWorkflowChain",
            workflow.canonical_readiness_spine_digest_prefix != "MISSING",
            workflow.reviewability_reduction_digest_prefix == "MISSING",
            workflow.applied_supported_set_digest_prefix != expected_scope,
            false,
        ),
        check_surface_status(
            "CanonicalReadinessSpine",
            spine.canonical_governance_entry_digest_prefix == expected_governance,
            false,
            spine.applied_supported_set_digest_prefix != expected_scope,
            false,
        ),
    ];

    let authority_status = if surfaces
        .iter()
        .all(|s| matches!(s.status, CanonicalReadinessAuthorityStatusV2::Pass))
    {
        CanonicalReadinessAuthorityStatusV2::Pass
    } else {
        CanonicalReadinessAuthorityStatusV2::Fail
    };
    let authority = derive_canonical_readiness_authority_v2(
        expected_scope,
        &expected_governance,
        &expected_spine,
        surfaces.len() as u16,
        authority_status,
    );

    let report = ReadinessSpineSweepReportV1 {
        schema_version: 1,
        authority,
        surfaces,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn check_surface_status(
    surface: &str,
    used_spine: bool,
    secondary_path: bool,
    scope_mismatch: bool,
    governance_mismatch: bool,
) -> ReadinessSpineSweepSurfaceStatusV1 {
    let mut mismatch_categories = BTreeSet::new();
    if !used_spine {
        mismatch_categories
            .insert(ReadinessSpineSweepMismatchCategoryV1::SurfaceSkippedCanonicalReadinessSpine);
    }
    if secondary_path {
        mismatch_categories
            .insert(ReadinessSpineSweepMismatchCategoryV1::SurfaceUsedSecondaryReadinessPath);
    }
    if scope_mismatch {
        mismatch_categories
            .insert(ReadinessSpineSweepMismatchCategoryV1::ReadinessSpineScopeMismatch);
    }
    if governance_mismatch {
        mismatch_categories
            .insert(ReadinessSpineSweepMismatchCategoryV1::ReadinessSpineGovernanceMismatch);
    }
    let status = if mismatch_categories.is_empty() {
        CanonicalReadinessAuthorityStatusV2::Pass
    } else {
        CanonicalReadinessAuthorityStatusV2::Fail
    };
    ReadinessSpineSweepSurfaceStatusV1 {
        surface: surface.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    }
}

pub fn derive_canonical_readiness_authority_v2(
    applied_supported_set_digest_prefix: &str,
    canonical_governance_entry_digest_prefix: &str,
    canonical_readiness_spine_digest_prefix: &str,
    covered_surface_count: u16,
    authority_status: CanonicalReadinessAuthorityStatusV2,
) -> CanonicalReadinessAuthorityV2 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"canonical_readiness_authority_v2");
    bytes.extend_from_slice(applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_readiness_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_surface_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", authority_status).as_bytes());
    CanonicalReadinessAuthorityV2 {
        schema_version: 2,
        applied_supported_set_digest_prefix: applied_supported_set_digest_prefix.to_string(),
        canonical_governance_entry_digest_prefix: canonical_governance_entry_digest_prefix
            .to_string(),
        canonical_readiness_spine_digest_prefix: canonical_readiness_spine_digest_prefix
            .to_string(),
        covered_surface_count,
        authority_status,
        authority_digest: crate::sha256_hex(&bytes),
    }
}

pub fn require_final_readiness_authority(
    applied_scope: &AppliedSupportedSetContextV1,
    entry: &CanonicalGovernanceEntryV1,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
) -> Result<FinalReadinessAuthorityContextV1, OpsError> {
    let spine = require_canonical_readiness_spine(applied_scope, entry, spine)?;
    let Some(authority) = authority else {
        return Err(OpsError::Invalid(
            FINAL_READINESS_AUTHORITY_REQUIRED.to_string(),
        ));
    };
    if !matches!(
        authority.authority_status,
        CanonicalReadinessAuthorityStatusV2::Pass
    ) {
        return Err(OpsError::Invalid(
            LEGACY_READINESS_INPUT_BLOCKED.to_string(),
        ));
    }
    let governance_prefix = prefix_hex(&entry.authority_digest, 16);
    let spine_prefix = prefix_hex(&spine.spine_digest, 16);
    if authority.applied_supported_set_digest_prefix != applied_scope.applied_set_digest_prefix
        || authority.canonical_governance_entry_digest_prefix != governance_prefix
        || authority.canonical_readiness_spine_digest_prefix != spine_prefix
    {
        return Err(OpsError::Invalid(
            FINAL_READINESS_AUTHORITY_REQUIRED.to_string(),
        ));
    }
    Ok(FinalReadinessAuthorityContextV1 {
        applied_supported_set_digest_prefix: authority.applied_supported_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: authority
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: authority
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_readiness_authority_digest_prefix: prefix_hex(&authority.authority_digest, 16),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_final_readiness_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: &AppliedSupportedSetContextV1,
    entry: &CanonicalGovernanceEntryV1,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
) -> Result<FinalReadinessInputsContextV1, OpsError> {
    if truths.is_empty() {
        return Err(OpsError::Invalid(
            SLOT_REVIEWABILITY_TRUTH_REQUIRED.to_string(),
        ));
    }
    if reduction.is_none() {
        return Err(OpsError::Invalid(
            REVIEWABILITY_REDUCTION_REQUIRED.to_string(),
        ));
    }
    let final_authority =
        require_final_readiness_authority(applied_scope, entry, spine, authority)?;
    let Some(final_consumer_authority) = final_consumer_authority else {
        return Err(OpsError::Invalid(
            FINAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    if !matches!(
        final_consumer_authority.authority_status,
        crate::FinalReadinessConsumerAuthorityStatusV1::Pass
    ) {
        return Err(OpsError::Invalid(
            RESIDUAL_READINESS_PATH_BLOCKED.to_string(),
        ));
    }
    let expected_final_consumer_digest = prefix_hex(&final_consumer_authority.authority_digest, 16);
    if final_consumer_authority.applied_supported_set_digest_prefix
        != final_authority.applied_supported_set_digest_prefix
        || final_consumer_authority.canonical_governance_entry_digest_prefix
            != final_authority.canonical_governance_entry_digest_prefix
        || final_consumer_authority.canonical_readiness_spine_digest_prefix
            != final_authority.canonical_readiness_spine_digest_prefix
        || final_consumer_authority.canonical_readiness_authority_digest_prefix
            != final_authority.canonical_readiness_authority_digest_prefix
    {
        return Err(OpsError::Invalid(
            FINAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    }
    Ok(FinalReadinessInputsContextV1 {
        applied_supported_set_digest_prefix: final_authority.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: final_authority
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: final_authority
            .canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: final_authority
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: expected_final_consumer_digest,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_residual_free_final_readiness_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: &AppliedSupportedSetContextV1,
    entry: &CanonicalGovernanceEntryV1,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
) -> Result<ResidualFreeFinalReadinessInputsV1, OpsError> {
    let final_inputs = require_final_readiness_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
    )?;
    let Some(residual_sweep) = residual_sweep else {
        return Err(OpsError::Invalid(
            RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    if !matches!(
        residual_sweep.sweep_status,
        crate::FinalReadinessResidualSweepStatusV1::Pass
    ) {
        return Err(OpsError::Invalid(
            HISTORICAL_READINESS_PATH_BLOCKED.to_string(),
        ));
    }
    let residual_prefix = prefix_hex(&residual_sweep.sweep_digest, 16);
    if residual_sweep.applied_supported_set_digest_prefix
        != final_inputs.applied_supported_set_digest_prefix
        || residual_sweep.canonical_governance_entry_digest_prefix
            != final_inputs.canonical_governance_entry_digest_prefix
        || residual_sweep.canonical_readiness_spine_digest_prefix
            != final_inputs.canonical_readiness_spine_digest_prefix
        || residual_sweep.canonical_readiness_authority_digest_prefix
            != final_inputs.canonical_readiness_authority_digest_prefix
        || residual_sweep.final_readiness_consumer_authority_digest_prefix
            != final_inputs.final_readiness_consumer_authority_digest_prefix
    {
        return Err(OpsError::Invalid(
            RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    }
    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"residual_free_final_readiness_inputs_v1");
    digest_source.extend_from_slice(final_inputs.applied_supported_set_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        final_inputs
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        final_inputs
            .canonical_readiness_spine_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        final_inputs
            .canonical_readiness_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        final_inputs
            .final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(residual_prefix.as_bytes());

    Ok(ResidualFreeFinalReadinessInputsV1 {
        applied_supported_set_digest_prefix: final_inputs.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: final_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: final_inputs
            .canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: final_inputs
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: final_inputs
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: residual_prefix,
        authority_digest: crate::sha256_hex(&digest_source),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_residual_free_readiness_absolute_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: &AppliedSupportedSetContextV1,
    entry: &CanonicalGovernanceEntryV1,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
    residual_free_consumer: Option<&crate::ResidualFreeReadinessConsumerAuthorityV1>,
) -> Result<ResidualFreeReadinessAbsoluteInputsV1, OpsError> {
    let base = require_residual_free_final_readiness_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
        residual_sweep,
    )?;
    let Some(residual_free_consumer) = residual_free_consumer else {
        return Err(OpsError::Invalid(
            RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let expected_residual_free_prefix = prefix_hex(&residual_free_consumer.authority_digest, 16);
    if !matches!(
        residual_free_consumer.authority_status,
        crate::ResidualFreeReadinessConsumerAuthorityStatusV1::Pass
    ) || residual_free_consumer.applied_supported_set_digest_prefix
        != base.applied_supported_set_digest_prefix
        || residual_free_consumer.canonical_governance_entry_digest_prefix
            != base.canonical_governance_entry_digest_prefix
        || residual_free_consumer.canonical_readiness_spine_digest_prefix
            != base.canonical_readiness_spine_digest_prefix
        || residual_free_consumer.canonical_readiness_authority_digest_prefix
            != base.canonical_readiness_authority_digest_prefix
        || residual_free_consumer.final_readiness_consumer_authority_digest_prefix
            != base.final_readiness_consumer_authority_digest_prefix
        || residual_free_consumer.final_readiness_residual_sweep_digest_prefix
            != base.final_readiness_residual_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(
            HISTORICAL_READINESS_LINEAGE_BLOCKED.to_string(),
        ));
    }

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"residual_free_readiness_absolute_inputs_v1");
    digest_source.extend_from_slice(base.applied_supported_set_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_governance_entry_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_readiness_spine_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_readiness_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        base.final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(base.final_readiness_residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(expected_residual_free_prefix.as_bytes());

    Ok(ResidualFreeReadinessAbsoluteInputsV1 {
        applied_supported_set_digest_prefix: base.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: base.canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: base.canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: base
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: base
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: base
            .final_readiness_residual_sweep_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: expected_residual_free_prefix,
        authority_digest: crate::sha256_hex(&digest_source),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_absolute_final_readiness_terminal_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: &AppliedSupportedSetContextV1,
    entry: &CanonicalGovernanceEntryV1,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
    residual_free_consumer: Option<&crate::ResidualFreeReadinessConsumerAuthorityV1>,
    absolute_sweep: Option<&crate::ResidualFreeReadinessAbsoluteSweepV1>,
) -> Result<AbsoluteFinalReadinessTerminalInputsV1, OpsError> {
    let base = require_residual_free_readiness_absolute_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
        residual_sweep,
        residual_free_consumer,
    )?;
    let Some(absolute_sweep) = absolute_sweep else {
        return Err(OpsError::Invalid(
            ABSOLUTE_RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let expected_absolute_prefix = prefix_hex(&absolute_sweep.sweep_digest, 16);
    if !matches!(
        absolute_sweep.sweep_status,
        crate::ResidualFreeReadinessAbsoluteSweepStatusV1::Pass
    ) || absolute_sweep.applied_supported_set_digest_prefix
        != base.applied_supported_set_digest_prefix
        || absolute_sweep.canonical_governance_entry_digest_prefix
            != base.canonical_governance_entry_digest_prefix
        || absolute_sweep.canonical_readiness_spine_digest_prefix
            != base.canonical_readiness_spine_digest_prefix
        || absolute_sweep.canonical_readiness_authority_digest_prefix
            != base.canonical_readiness_authority_digest_prefix
        || absolute_sweep.final_readiness_consumer_authority_digest_prefix
            != base.final_readiness_consumer_authority_digest_prefix
        || absolute_sweep.final_readiness_residual_sweep_digest_prefix
            != base.final_readiness_residual_sweep_digest_prefix
        || absolute_sweep.residual_free_readiness_consumer_authority_digest_prefix
            != base.residual_free_readiness_consumer_authority_digest_prefix
    {
        return Err(OpsError::Invalid(READINESS_ECHO_PATH_BLOCKED.to_string()));
    }

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"absolute_final_readiness_terminal_inputs_v1");
    digest_source.extend_from_slice(base.applied_supported_set_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_governance_entry_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_readiness_spine_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_readiness_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        base.final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(base.final_readiness_residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        base.residual_free_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(expected_absolute_prefix.as_bytes());

    Ok(AbsoluteFinalReadinessTerminalInputsV1 {
        applied_supported_set_digest_prefix: base.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: base.canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: base.canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: base
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: base
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: base
            .final_readiness_residual_sweep_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: base
            .residual_free_readiness_consumer_authority_digest_prefix,
        residual_free_readiness_absolute_sweep_digest_prefix: expected_absolute_prefix,
        authority_digest: crate::sha256_hex(&digest_source),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_terminal_readiness_ultimate_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: &AppliedSupportedSetContextV1,
    entry: &CanonicalGovernanceEntryV1,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
    residual_free_consumer: Option<&crate::ResidualFreeReadinessConsumerAuthorityV1>,
    absolute_sweep: Option<&crate::ResidualFreeReadinessAbsoluteSweepV1>,
    terminal_sweep: Option<&crate::AbsoluteFinalReadinessTerminalSweepV1>,
) -> Result<TerminalReadinessUltimateInputsV1, OpsError> {
    let base = require_absolute_final_readiness_terminal_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
    )?;
    let Some(terminal_sweep) = terminal_sweep else {
        return Err(OpsError::Invalid(
            TERMINAL_ABSOLUTE_RESIDUAL_FREE_FINAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let terminal_prefix = prefix_hex(&terminal_sweep.sweep_digest, 16);
    if !matches!(
        terminal_sweep.sweep_status,
        crate::AbsoluteFinalReadinessTerminalSweepStatusV1::Pass
    ) || terminal_sweep.applied_supported_set_digest_prefix
        != base.applied_supported_set_digest_prefix
        || terminal_sweep.canonical_governance_entry_digest_prefix
            != base.canonical_governance_entry_digest_prefix
        || terminal_sweep.canonical_readiness_spine_digest_prefix
            != base.canonical_readiness_spine_digest_prefix
        || terminal_sweep.canonical_readiness_authority_digest_prefix
            != base.canonical_readiness_authority_digest_prefix
        || terminal_sweep.final_readiness_consumer_authority_digest_prefix
            != base.final_readiness_consumer_authority_digest_prefix
        || terminal_sweep.final_readiness_residual_sweep_digest_prefix
            != base.final_readiness_residual_sweep_digest_prefix
        || terminal_sweep.residual_free_readiness_consumer_authority_digest_prefix
            != base.residual_free_readiness_consumer_authority_digest_prefix
        || terminal_sweep.residual_free_readiness_absolute_sweep_digest_prefix
            != base.residual_free_readiness_absolute_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(READINESS_CACHE_PATH_BLOCKED.to_string()));
    }

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"terminal_readiness_ultimate_inputs_v1");
    digest_source.extend_from_slice(base.applied_supported_set_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_governance_entry_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_readiness_spine_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_readiness_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        base.final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(base.final_readiness_residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        base.residual_free_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        base.residual_free_readiness_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(terminal_prefix.as_bytes());

    Ok(TerminalReadinessUltimateInputsV1 {
        applied_supported_set_digest_prefix: base.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: base.canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: base.canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: base
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: base
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: base
            .final_readiness_residual_sweep_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: base
            .residual_free_readiness_consumer_authority_digest_prefix,
        residual_free_readiness_absolute_sweep_digest_prefix: base
            .residual_free_readiness_absolute_sweep_digest_prefix,
        absolute_final_readiness_terminal_sweep_digest_prefix: terminal_prefix,
        authority_digest: crate::sha256_hex(&digest_source),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_readiness_convergence_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
    residual_free_consumer: Option<&crate::ResidualFreeReadinessConsumerAuthorityV1>,
    absolute_sweep: Option<&crate::ResidualFreeReadinessAbsoluteSweepV1>,
    terminal_sweep: Option<&crate::AbsoluteFinalReadinessTerminalSweepV1>,
    ultimate_sweep: Option<&crate::TerminalReadinessUltimateSweepV1>,
) -> Result<ReadinessConvergenceInputsV1, OpsError> {
    let Some(applied_scope) = applied_scope else {
        return Err(OpsError::Invalid(
            ULTIMATE_TERMINAL_ABSOLUTE_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let Some(entry) = entry else {
        return Err(OpsError::Invalid(
            ULTIMATE_TERMINAL_ABSOLUTE_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let base = require_terminal_readiness_ultimate_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
        terminal_sweep,
    )?;
    let Some(ultimate_sweep) = ultimate_sweep else {
        return Err(OpsError::Invalid(
            ULTIMATE_TERMINAL_ABSOLUTE_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let ultimate_prefix = prefix_hex(&ultimate_sweep.sweep_digest, 16);
    if !matches!(
        ultimate_sweep.sweep_status,
        crate::TerminalReadinessUltimateSweepStatusV1::Pass
    ) || ultimate_sweep.applied_supported_set_digest_prefix
        != base.applied_supported_set_digest_prefix
        || ultimate_sweep.canonical_governance_entry_digest_prefix
            != base.canonical_governance_entry_digest_prefix
        || ultimate_sweep.canonical_readiness_spine_digest_prefix
            != base.canonical_readiness_spine_digest_prefix
        || ultimate_sweep.canonical_readiness_authority_digest_prefix
            != base.canonical_readiness_authority_digest_prefix
        || ultimate_sweep.final_readiness_consumer_authority_digest_prefix
            != base.final_readiness_consumer_authority_digest_prefix
        || ultimate_sweep.final_readiness_residual_sweep_digest_prefix
            != base.final_readiness_residual_sweep_digest_prefix
        || ultimate_sweep.residual_free_readiness_consumer_authority_digest_prefix
            != base.residual_free_readiness_consumer_authority_digest_prefix
        || ultimate_sweep.residual_free_readiness_absolute_sweep_digest_prefix
            != base.residual_free_readiness_absolute_sweep_digest_prefix
        || ultimate_sweep.absolute_final_readiness_terminal_sweep_digest_prefix
            != base.absolute_final_readiness_terminal_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(READINESS_MEMO_PATH_BLOCKED.to_string()));
    }

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"readiness_convergence_inputs_v1");
    digest_source.extend_from_slice(base.applied_supported_set_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_governance_entry_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_readiness_spine_digest_prefix.as_bytes());
    digest_source.extend_from_slice(base.canonical_readiness_authority_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        base.final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(base.final_readiness_residual_sweep_digest_prefix.as_bytes());
    digest_source.extend_from_slice(
        base.residual_free_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        base.residual_free_readiness_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        base.absolute_final_readiness_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(ultimate_prefix.as_bytes());

    Ok(ReadinessConvergenceInputsV1 {
        applied_supported_set_digest_prefix: base.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: base.canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: base.canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: base
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: base
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: base
            .final_readiness_residual_sweep_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: base
            .residual_free_readiness_consumer_authority_digest_prefix,
        residual_free_readiness_absolute_sweep_digest_prefix: base
            .residual_free_readiness_absolute_sweep_digest_prefix,
        absolute_final_readiness_terminal_sweep_digest_prefix: base
            .absolute_final_readiness_terminal_sweep_digest_prefix,
        terminal_readiness_ultimate_sweep_digest_prefix: ultimate_prefix,
        authority_digest: crate::sha256_hex(&digest_source),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_readiness_stabilization_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
    residual_free_consumer: Option<&crate::ResidualFreeReadinessConsumerAuthorityV1>,
    absolute_sweep: Option<&crate::ResidualFreeReadinessAbsoluteSweepV1>,
    terminal_sweep: Option<&crate::AbsoluteFinalReadinessTerminalSweepV1>,
    ultimate_sweep: Option<&crate::TerminalReadinessUltimateSweepV1>,
    convergence_sweep: Option<&crate::ReadinessConvergenceSweepV1>,
) -> Result<ReadinessStabilizationInputsV1, OpsError> {
    let convergence_inputs = require_readiness_convergence_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
        terminal_sweep,
        ultimate_sweep,
    )?;
    let Some(convergence_sweep) = convergence_sweep else {
        return Err(OpsError::Invalid(
            CONVERGED_CANONICAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let convergence_prefix = prefix_hex(&convergence_sweep.convergence_digest, 16);
    if !matches!(
        convergence_sweep.convergence_status,
        crate::ReadinessConvergenceStatusV1::Pass
    ) || convergence_sweep.applied_supported_set_digest_prefix
        != convergence_inputs.applied_supported_set_digest_prefix
        || convergence_sweep.canonical_governance_entry_digest_prefix
            != convergence_inputs.canonical_governance_entry_digest_prefix
        || convergence_sweep.canonical_readiness_spine_digest_prefix
            != convergence_inputs.canonical_readiness_spine_digest_prefix
        || convergence_sweep.canonical_readiness_authority_digest_prefix
            != convergence_inputs.canonical_readiness_authority_digest_prefix
        || convergence_sweep.final_readiness_consumer_authority_digest_prefix
            != convergence_inputs.final_readiness_consumer_authority_digest_prefix
        || convergence_sweep.final_readiness_residual_sweep_digest_prefix
            != convergence_inputs.final_readiness_residual_sweep_digest_prefix
        || convergence_sweep.residual_free_readiness_consumer_authority_digest_prefix
            != convergence_inputs.residual_free_readiness_consumer_authority_digest_prefix
        || convergence_sweep.residual_free_readiness_absolute_sweep_digest_prefix
            != convergence_inputs.residual_free_readiness_absolute_sweep_digest_prefix
        || convergence_sweep.absolute_final_readiness_terminal_sweep_digest_prefix
            != convergence_inputs.absolute_final_readiness_terminal_sweep_digest_prefix
        || convergence_sweep.terminal_readiness_ultimate_sweep_digest_prefix
            != convergence_inputs.terminal_readiness_ultimate_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(
            READINESS_ADAPTER_PATH_BLOCKED.to_string(),
        ));
    }

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"readiness_stabilization_inputs_v1");
    digest_source.extend_from_slice(
        convergence_inputs
            .applied_supported_set_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .canonical_readiness_spine_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .canonical_readiness_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .final_readiness_residual_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .residual_free_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .residual_free_readiness_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .absolute_final_readiness_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        convergence_inputs
            .terminal_readiness_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(convergence_prefix.as_bytes());

    Ok(ReadinessStabilizationInputsV1 {
        applied_supported_set_digest_prefix: convergence_inputs.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: convergence_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: convergence_inputs
            .canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: convergence_inputs
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: convergence_inputs
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: convergence_inputs
            .final_readiness_residual_sweep_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: convergence_inputs
            .residual_free_readiness_consumer_authority_digest_prefix,
        residual_free_readiness_absolute_sweep_digest_prefix: convergence_inputs
            .residual_free_readiness_absolute_sweep_digest_prefix,
        absolute_final_readiness_terminal_sweep_digest_prefix: convergence_inputs
            .absolute_final_readiness_terminal_sweep_digest_prefix,
        terminal_readiness_ultimate_sweep_digest_prefix: convergence_inputs
            .terminal_readiness_ultimate_sweep_digest_prefix,
        readiness_convergence_sweep_digest_prefix: convergence_prefix,
        readiness_stabilization_sweep_digest_prefix: String::new(),
        authority_digest: crate::sha256_hex(&digest_source),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_readiness_final_consolidation_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
    residual_free_consumer: Option<&crate::ResidualFreeReadinessConsumerAuthorityV1>,
    absolute_sweep: Option<&crate::ResidualFreeReadinessAbsoluteSweepV1>,
    terminal_sweep: Option<&crate::AbsoluteFinalReadinessTerminalSweepV1>,
    ultimate_sweep: Option<&crate::TerminalReadinessUltimateSweepV1>,
    convergence_sweep: Option<&crate::ReadinessConvergenceSweepV1>,
    stabilization_sweep: Option<&crate::ReadinessStabilizationSweepV1>,
) -> Result<ReadinessFinalConsolidationInputsV1, OpsError> {
    let stabilization_inputs = require_readiness_stabilization_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
        terminal_sweep,
        ultimate_sweep,
        convergence_sweep,
    )?;
    let Some(stabilization_sweep) = stabilization_sweep else {
        return Err(OpsError::Invalid(
            STABILIZED_CONVERGED_CANONICAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let stabilization_prefix = prefix_hex(&stabilization_sweep.stabilization_digest, 16);
    if !matches!(
        stabilization_sweep.stabilization_status,
        crate::ReadinessStabilizationStatusV1::Pass
    ) || stabilization_sweep.applied_supported_set_digest_prefix
        != stabilization_inputs.applied_supported_set_digest_prefix
        || stabilization_sweep.canonical_governance_entry_digest_prefix
            != stabilization_inputs.canonical_governance_entry_digest_prefix
        || stabilization_sweep.canonical_readiness_spine_digest_prefix
            != stabilization_inputs.canonical_readiness_spine_digest_prefix
        || stabilization_sweep.canonical_readiness_authority_digest_prefix
            != stabilization_inputs.canonical_readiness_authority_digest_prefix
        || stabilization_sweep.final_readiness_consumer_authority_digest_prefix
            != stabilization_inputs.final_readiness_consumer_authority_digest_prefix
        || stabilization_sweep.final_readiness_residual_sweep_digest_prefix
            != stabilization_inputs.final_readiness_residual_sweep_digest_prefix
        || stabilization_sweep.residual_free_readiness_consumer_authority_digest_prefix
            != stabilization_inputs.residual_free_readiness_consumer_authority_digest_prefix
        || stabilization_sweep.residual_free_readiness_absolute_sweep_digest_prefix
            != stabilization_inputs.residual_free_readiness_absolute_sweep_digest_prefix
        || stabilization_sweep.absolute_final_readiness_terminal_sweep_digest_prefix
            != stabilization_inputs.absolute_final_readiness_terminal_sweep_digest_prefix
        || stabilization_sweep.terminal_readiness_ultimate_sweep_digest_prefix
            != stabilization_inputs.terminal_readiness_ultimate_sweep_digest_prefix
        || stabilization_sweep.readiness_convergence_sweep_digest_prefix
            != stabilization_inputs.readiness_convergence_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(READINESS_FACADE_PATH_BLOCKED.to_string()));
    }

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"readiness_final_consolidation_inputs_v1");
    digest_source.extend_from_slice(
        stabilization_inputs
            .applied_supported_set_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .canonical_readiness_spine_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .canonical_readiness_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .final_readiness_residual_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .residual_free_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .residual_free_readiness_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .absolute_final_readiness_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .terminal_readiness_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(
        stabilization_inputs
            .readiness_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    digest_source.extend_from_slice(stabilization_prefix.as_bytes());

    Ok(ReadinessFinalConsolidationInputsV1 {
        applied_supported_set_digest_prefix: stabilization_inputs
            .applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: stabilization_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: stabilization_inputs
            .canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: stabilization_inputs
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: stabilization_inputs
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: stabilization_inputs
            .final_readiness_residual_sweep_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: stabilization_inputs
            .residual_free_readiness_consumer_authority_digest_prefix,
        residual_free_readiness_absolute_sweep_digest_prefix: stabilization_inputs
            .residual_free_readiness_absolute_sweep_digest_prefix,
        absolute_final_readiness_terminal_sweep_digest_prefix: stabilization_inputs
            .absolute_final_readiness_terminal_sweep_digest_prefix,
        terminal_readiness_ultimate_sweep_digest_prefix: stabilization_inputs
            .terminal_readiness_ultimate_sweep_digest_prefix,
        readiness_convergence_sweep_digest_prefix: stabilization_inputs
            .readiness_convergence_sweep_digest_prefix,
        readiness_stabilization_sweep_digest_prefix: stabilization_prefix,
        authority_digest: crate::sha256_hex(&digest_source),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_readiness_closure_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
    residual_free_consumer: Option<&crate::ResidualFreeReadinessConsumerAuthorityV1>,
    absolute_sweep: Option<&crate::ResidualFreeReadinessAbsoluteSweepV1>,
    terminal_sweep: Option<&crate::AbsoluteFinalReadinessTerminalSweepV1>,
    ultimate_sweep: Option<&crate::TerminalReadinessUltimateSweepV1>,
    convergence_sweep: Option<&crate::ReadinessConvergenceSweepV1>,
    stabilization_sweep: Option<&crate::ReadinessStabilizationSweepV1>,
    final_consolidation_sweep: Option<&crate::ReadinessFinalConsolidationSweepV1>,
) -> Result<ReadinessClosureInputsV1, OpsError> {
    let consolidation_inputs = require_readiness_final_consolidation_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
        terminal_sweep,
        ultimate_sweep,
        convergence_sweep,
        stabilization_sweep,
    )?;
    let Some(final_consolidation_sweep) = final_consolidation_sweep else {
        return Err(OpsError::Invalid(
            FINAL_CONSOLIDATED_STABILIZED_CANONICAL_READINESS_INPUTS_REQUIRED.to_string(),
        ));
    };
    let expected_final_consolidation_prefix =
        prefix_hex(&final_consolidation_sweep.consolidation_digest, 16);
    if !matches!(
        final_consolidation_sweep.consolidation_status,
        crate::ReadinessFinalConsolidationStatusV1::Pass
    ) || final_consolidation_sweep.applied_supported_set_digest_prefix
        != consolidation_inputs.applied_supported_set_digest_prefix
        || final_consolidation_sweep.canonical_governance_entry_digest_prefix
            != consolidation_inputs.canonical_governance_entry_digest_prefix
        || final_consolidation_sweep.canonical_readiness_spine_digest_prefix
            != consolidation_inputs.canonical_readiness_spine_digest_prefix
        || final_consolidation_sweep.canonical_readiness_authority_digest_prefix
            != consolidation_inputs.canonical_readiness_authority_digest_prefix
        || final_consolidation_sweep.final_readiness_consumer_authority_digest_prefix
            != consolidation_inputs.final_readiness_consumer_authority_digest_prefix
        || final_consolidation_sweep.final_readiness_residual_sweep_digest_prefix
            != consolidation_inputs.final_readiness_residual_sweep_digest_prefix
        || final_consolidation_sweep.residual_free_readiness_consumer_authority_digest_prefix
            != consolidation_inputs.residual_free_readiness_consumer_authority_digest_prefix
        || final_consolidation_sweep.residual_free_readiness_absolute_sweep_digest_prefix
            != consolidation_inputs.residual_free_readiness_absolute_sweep_digest_prefix
        || final_consolidation_sweep.absolute_final_readiness_terminal_sweep_digest_prefix
            != consolidation_inputs.absolute_final_readiness_terminal_sweep_digest_prefix
        || final_consolidation_sweep.terminal_readiness_ultimate_sweep_digest_prefix
            != consolidation_inputs.terminal_readiness_ultimate_sweep_digest_prefix
        || final_consolidation_sweep.readiness_convergence_sweep_digest_prefix
            != consolidation_inputs.readiness_convergence_sweep_digest_prefix
        || final_consolidation_sweep.readiness_stabilization_sweep_digest_prefix
            != consolidation_inputs.readiness_stabilization_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(
            READINESS_WRAPPER_PATH_BLOCKED.to_string(),
        ));
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"readiness_closure_inputs_v1");
    bytes.extend_from_slice(
        consolidation_inputs
            .applied_supported_set_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .canonical_readiness_spine_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .canonical_readiness_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .final_readiness_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .residual_free_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .residual_free_readiness_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .absolute_final_readiness_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .terminal_readiness_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .readiness_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        consolidation_inputs
            .readiness_stabilization_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(expected_final_consolidation_prefix.as_bytes());

    Ok(ReadinessClosureInputsV1 {
        applied_supported_set_digest_prefix: consolidation_inputs
            .applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: consolidation_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: consolidation_inputs
            .canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: consolidation_inputs
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: consolidation_inputs
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: consolidation_inputs
            .final_readiness_residual_sweep_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: consolidation_inputs
            .residual_free_readiness_consumer_authority_digest_prefix,
        residual_free_readiness_absolute_sweep_digest_prefix: consolidation_inputs
            .residual_free_readiness_absolute_sweep_digest_prefix,
        absolute_final_readiness_terminal_sweep_digest_prefix: consolidation_inputs
            .absolute_final_readiness_terminal_sweep_digest_prefix,
        terminal_readiness_ultimate_sweep_digest_prefix: consolidation_inputs
            .terminal_readiness_ultimate_sweep_digest_prefix,
        readiness_convergence_sweep_digest_prefix: consolidation_inputs
            .readiness_convergence_sweep_digest_prefix,
        readiness_stabilization_sweep_digest_prefix: consolidation_inputs
            .readiness_stabilization_sweep_digest_prefix,
        readiness_final_consolidation_sweep_digest_prefix: expected_final_consolidation_prefix,
        authority_digest: crate::sha256_hex(&bytes),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_readiness_seal_inputs(
    truths: &[SlotReviewabilityTruthV1],
    reduction: Option<&ReviewabilityReductionV1>,
    applied_scope: Option<&AppliedSupportedSetContextV1>,
    entry: Option<&CanonicalGovernanceEntryV1>,
    spine: Option<&CanonicalReadinessSpineV1>,
    authority: Option<&CanonicalReadinessAuthorityV2>,
    final_consumer_authority: Option<&crate::FinalReadinessConsumerAuthorityV1>,
    residual_sweep: Option<&crate::FinalReadinessResidualSweepV1>,
    residual_free_consumer: Option<&crate::ResidualFreeReadinessConsumerAuthorityV1>,
    absolute_sweep: Option<&crate::ResidualFreeReadinessAbsoluteSweepV1>,
    terminal_sweep: Option<&crate::AbsoluteFinalReadinessTerminalSweepV1>,
    ultimate_sweep: Option<&crate::TerminalReadinessUltimateSweepV1>,
    convergence_sweep: Option<&crate::ReadinessConvergenceSweepV1>,
    stabilization_sweep: Option<&crate::ReadinessStabilizationSweepV1>,
    final_consolidation_sweep: Option<&crate::ReadinessFinalConsolidationSweepV1>,
    closure_sweep: Option<&crate::ReadinessClosureSweepV1>,
) -> Result<ReadinessSealInputsV1, OpsError> {
    let closure_inputs = require_readiness_closure_inputs(
        truths,
        reduction,
        applied_scope,
        entry,
        spine,
        authority,
        final_consumer_authority,
        residual_sweep,
        residual_free_consumer,
        absolute_sweep,
        terminal_sweep,
        ultimate_sweep,
        convergence_sweep,
        stabilization_sweep,
        final_consolidation_sweep,
    )?;
    let Some(closure_sweep) = closure_sweep else {
        return Err(OpsError::Invalid(
            CLOSURE_COMPLETE_FINAL_CONSOLIDATED_STABILIZED_CANONICAL_READINESS_INPUTS_REQUIRED
                .to_string(),
        ));
    };
    let expected_closure_prefix = prefix_hex(&closure_sweep.closure_digest, 16);
    if !matches!(
        closure_sweep.closure_status,
        crate::ReadinessClosureStatusV1::Pass
    ) || closure_sweep.applied_supported_set_digest_prefix
        != closure_inputs.applied_supported_set_digest_prefix
        || closure_sweep.canonical_governance_entry_digest_prefix
            != closure_inputs.canonical_governance_entry_digest_prefix
        || closure_sweep.canonical_readiness_spine_digest_prefix
            != closure_inputs.canonical_readiness_spine_digest_prefix
        || closure_sweep.canonical_readiness_authority_digest_prefix
            != closure_inputs.canonical_readiness_authority_digest_prefix
        || closure_sweep.final_readiness_consumer_authority_digest_prefix
            != closure_inputs.final_readiness_consumer_authority_digest_prefix
        || closure_sweep.final_readiness_residual_sweep_digest_prefix
            != closure_inputs.final_readiness_residual_sweep_digest_prefix
        || closure_sweep.residual_free_readiness_consumer_authority_digest_prefix
            != closure_inputs.residual_free_readiness_consumer_authority_digest_prefix
        || closure_sweep.residual_free_readiness_absolute_sweep_digest_prefix
            != closure_inputs.residual_free_readiness_absolute_sweep_digest_prefix
        || closure_sweep.absolute_final_readiness_terminal_sweep_digest_prefix
            != closure_inputs.absolute_final_readiness_terminal_sweep_digest_prefix
        || closure_sweep.terminal_readiness_ultimate_sweep_digest_prefix
            != closure_inputs.terminal_readiness_ultimate_sweep_digest_prefix
        || closure_sweep.readiness_convergence_sweep_digest_prefix
            != closure_inputs.readiness_convergence_sweep_digest_prefix
        || closure_sweep.readiness_stabilization_sweep_digest_prefix
            != closure_inputs.readiness_stabilization_sweep_digest_prefix
        || closure_sweep.readiness_final_consolidation_sweep_digest_prefix
            != closure_inputs.readiness_final_consolidation_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(READINESS_SHELL_PATH_BLOCKED.to_string()));
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"readiness_seal_inputs_v1");
    bytes.extend_from_slice(
        closure_inputs
            .applied_supported_set_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .canonical_readiness_spine_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .canonical_readiness_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .final_readiness_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .residual_free_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .residual_free_readiness_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .absolute_final_readiness_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .terminal_readiness_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .readiness_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .readiness_stabilization_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        closure_inputs
            .readiness_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(expected_closure_prefix.as_bytes());

    Ok(ReadinessSealInputsV1 {
        applied_supported_set_digest_prefix: closure_inputs.applied_supported_set_digest_prefix,
        canonical_governance_entry_digest_prefix: closure_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: closure_inputs
            .canonical_readiness_spine_digest_prefix,
        canonical_readiness_authority_digest_prefix: closure_inputs
            .canonical_readiness_authority_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: closure_inputs
            .final_readiness_consumer_authority_digest_prefix,
        final_readiness_residual_sweep_digest_prefix: closure_inputs
            .final_readiness_residual_sweep_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: closure_inputs
            .residual_free_readiness_consumer_authority_digest_prefix,
        residual_free_readiness_absolute_sweep_digest_prefix: closure_inputs
            .residual_free_readiness_absolute_sweep_digest_prefix,
        absolute_final_readiness_terminal_sweep_digest_prefix: closure_inputs
            .absolute_final_readiness_terminal_sweep_digest_prefix,
        terminal_readiness_ultimate_sweep_digest_prefix: closure_inputs
            .terminal_readiness_ultimate_sweep_digest_prefix,
        readiness_convergence_sweep_digest_prefix: closure_inputs
            .readiness_convergence_sweep_digest_prefix,
        readiness_stabilization_sweep_digest_prefix: closure_inputs
            .readiness_stabilization_sweep_digest_prefix,
        readiness_final_consolidation_sweep_digest_prefix: closure_inputs
            .readiness_final_consolidation_sweep_digest_prefix,
        readiness_closure_sweep_digest_prefix: expected_closure_prefix,
        authority_digest: crate::sha256_hex(&bytes),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn require_readiness_lock_inputs(
    governance_lock_inputs: Option<&crate::GovernanceLockInputsV1>,
    governance_lock_sweep: Option<&crate::GovernanceLockSweepV1>,
    supported_scope_decision: Option<&crate::SupportedScopeExpansionDecisionV1>,
    canonical_execution_reality_digest_prefix: Option<&str>,
) -> Result<ReadinessLockInputsV1, OpsError> {
    let Some(governance_lock_inputs) = governance_lock_inputs else {
        return Err(OpsError::Invalid(
            GOVERNANCE_LOCK_INPUTS_REQUIRED.to_string(),
        ));
    };
    let Some(governance_lock_sweep) = governance_lock_sweep else {
        return Err(OpsError::Invalid(
            GOVERNANCE_LOCK_INPUTS_REQUIRED.to_string(),
        ));
    };
    let Some(supported_scope_decision) = supported_scope_decision else {
        return Err(OpsError::Invalid(
            SUPPORTED_SCOPE_DECISION_REQUIRED.to_string(),
        ));
    };
    let Some(canonical_execution_reality_digest_prefix) = canonical_execution_reality_digest_prefix
    else {
        return Err(OpsError::Invalid(
            CANONICAL_EXECUTION_REALITY_REQUIRED.to_string(),
        ));
    };
    if canonical_execution_reality_digest_prefix == "MISSING" {
        return Err(OpsError::Invalid(
            CANONICAL_EXECUTION_REALITY_REQUIRED.to_string(),
        ));
    }
    let expected_lock_prefix = prefix_hex(&governance_lock_sweep.lock_digest, 16);
    if !matches!(
        governance_lock_sweep.lock_status,
        crate::GovernanceLockStatusV1::Pass
    ) || governance_lock_sweep.applied_supported_set_digest_prefix
        != governance_lock_inputs.applied_supported_set_digest_prefix
        || governance_lock_sweep.canonical_governance_entry_digest_prefix
            != governance_lock_inputs.canonical_governance_entry_digest_prefix
        || governance_lock_sweep.governance_seal_sweep_digest_prefix
            != governance_lock_inputs.governance_seal_sweep_digest_prefix
    {
        return Err(OpsError::Invalid(
            GOVERNANCE_LOCK_INPUTS_REQUIRED.to_string(),
        ));
    }
    if supported_scope_decision.applied_supported_set_digest_prefix
        != governance_lock_inputs.applied_supported_set_digest_prefix
        || supported_scope_decision.governance_lock_sweep_digest_prefix != expected_lock_prefix
    {
        return Err(OpsError::Invalid(READINESS_SCOPE_OVERRUN.to_string()));
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"readiness_lock_inputs_v1");
    bytes.extend_from_slice(
        governance_lock_inputs
            .applied_supported_set_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .canonical_governance_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .final_governance_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .absolute_final_governance_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .terminal_governance_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .governance_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .governance_stabilization_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .governance_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .governance_closure_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        governance_lock_inputs
            .governance_seal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(expected_lock_prefix.as_bytes());
    bytes.extend_from_slice(prefix_hex(&supported_scope_decision.decision_digest, 16).as_bytes());
    bytes.extend_from_slice(canonical_execution_reality_digest_prefix.as_bytes());

    Ok(ReadinessLockInputsV1 {
        applied_supported_set_digest_prefix: governance_lock_inputs
            .applied_supported_set_digest_prefix
            .clone(),
        canonical_governance_entry_digest_prefix: governance_lock_inputs
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_governance_authority_digest_prefix: governance_lock_inputs
            .canonical_governance_authority_digest_prefix
            .clone(),
        final_governance_consumer_authority_digest_prefix: governance_lock_inputs
            .final_governance_consumer_authority_digest_prefix
            .clone(),
        final_governance_residual_sweep_digest_prefix: governance_lock_inputs
            .final_governance_residual_sweep_digest_prefix
            .clone(),
        residual_free_governance_consumer_authority_digest_prefix: governance_lock_inputs
            .residual_free_governance_consumer_authority_digest_prefix
            .clone(),
        residual_free_governance_absolute_sweep_digest_prefix: governance_lock_inputs
            .residual_free_governance_absolute_sweep_digest_prefix
            .clone(),
        absolute_final_governance_terminal_sweep_digest_prefix: governance_lock_inputs
            .absolute_final_governance_terminal_sweep_digest_prefix
            .clone(),
        terminal_governance_ultimate_sweep_digest_prefix: governance_lock_inputs
            .terminal_governance_ultimate_sweep_digest_prefix
            .clone(),
        governance_convergence_sweep_digest_prefix: governance_lock_inputs
            .governance_convergence_sweep_digest_prefix
            .clone(),
        governance_stabilization_sweep_digest_prefix: governance_lock_inputs
            .governance_stabilization_sweep_digest_prefix
            .clone(),
        governance_final_consolidation_sweep_digest_prefix: governance_lock_inputs
            .governance_final_consolidation_sweep_digest_prefix
            .clone(),
        governance_closure_sweep_digest_prefix: governance_lock_inputs
            .governance_closure_sweep_digest_prefix
            .clone(),
        governance_seal_sweep_digest_prefix: governance_lock_inputs
            .governance_seal_sweep_digest_prefix
            .clone(),
        governance_lock_sweep_digest_prefix: expected_lock_prefix,
        supported_scope_decision_digest_prefix: prefix_hex(
            &supported_scope_decision.decision_digest,
            16,
        ),
        canonical_execution_reality_digest_prefix: canonical_execution_reality_digest_prefix
            .to_string(),
        authority_digest: crate::sha256_hex(&bytes),
    })
}

fn sha_truths(truths: &[SlotReviewabilityTruthV1]) -> String {
    let mut bytes = Vec::new();
    for truth in truths {
        bytes.extend_from_slice(truth.reviewability_truth_digest.as_bytes());
    }
    crate::sha256_hex(&bytes)
}

pub fn attach_spine_prefix_to_signoff(
    signoff: &mut OperatorSignoffDecisionV1,
    spine: &CanonicalReadinessSpineV1,
) {
    signoff.canonical_readiness_spine_digest_prefix = prefix_hex(&spine.spine_digest, 16);
}

pub fn attach_spine_prefix_to_packet(
    packet: &mut OperatorReviewPacketV1,
    spine: &CanonicalReadinessSpineV1,
) {
    packet.canonical_readiness_spine_digest_prefix = prefix_hex(&spine.spine_digest, 16);
}

pub fn attach_spine_prefix_to_workflow(
    workflow: &mut OperatorWorkflowChainV1,
    spine: &CanonicalReadinessSpineV1,
) {
    workflow.canonical_readiness_spine_digest_prefix = prefix_hex(&spine.spine_digest, 16);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_authority_v2_digest_is_stable() {
        let a = derive_canonical_readiness_authority_v2(
            "scope123456789012",
            "governance1234567",
            "spine123456789012",
            4,
            CanonicalReadinessAuthorityStatusV2::Pass,
        );
        let b = derive_canonical_readiness_authority_v2(
            "scope123456789012",
            "governance1234567",
            "spine123456789012",
            4,
            CanonicalReadinessAuthorityStatusV2::Pass,
        );
        assert_eq!(a.authority_digest, b.authority_digest);
    }
}
