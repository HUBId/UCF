use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, derive_slot_reviewability_truths,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_signoff, operator_workflow_chain, prefix_hex,
    reduce_reviewability, resolve_strict_evidence,
    validate_governance_primary_surfaces_with_applied_scope, AppliedSupportedSetContextV1,
    CanonicalGovernanceEntryV1, OperatorReviewPacketArgs, OperatorReviewPacketV1,
    OperatorSignoffArgs, OperatorSignoffDecisionV1, OperatorWorkflowArgs, OperatorWorkflowChainV1,
    OpsError, ReviewabilityReductionV1, SlotReviewabilityTruthV1, StrictEvidenceContextV1,
};

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
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied_scope)?;
    let entry = derive_canonical_governance_entry(&applied_scope, &surfaces)?;

    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_readiness_spine_check.json"),
    )?;
    let packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_readiness_spine_check.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
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
