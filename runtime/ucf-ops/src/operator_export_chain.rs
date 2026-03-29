use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_signoff, operator_workflow_chain, prefix_hex,
    reduce_reviewability, sha256_hex, AppliedSupportedSetContextV1, BugKitManifestV1,
    OperatorReviewPacketArgs, OperatorReviewPacketV1, OperatorSignoffArgs,
    OperatorSignoffDecisionV1, OperatorWorkflowArgs, OperatorWorkflowChainV1, OpsError,
    ReproPackManifestV1,
};

const DIGEST_PREFIX_LEN: usize = 16;
const BLOCKING_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OperatorExportAuthorityChainStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OperatorExportAuthorityMismatchCategoryV1 {
    ReviewPacketScopeMismatch,
    SignoffScopeMismatch,
    WorkflowScopeMismatch,
    ExportContextScopeMismatch,
    ReviewabilityBasisMismatch,
    ReadinessSpineMismatch,
    AppliedScopeMissing,
    CanonicalConvergenceContinuityMissing,
    CanonicalConvergenceContinuityFail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalConvergenceContinuityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CanonicalConvergenceContinuityProbeV1 {
    continuity_status: CanonicalConvergenceContinuityStatusV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorExportAuthorityChainV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub applied_context_digest_prefix: String,
    pub reviewability_reduction_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub export_context_digest_prefix: Option<String>,
    pub authority_chain_status: OperatorExportAuthorityChainStatusV1,
    pub mismatch_categories: Vec<OperatorExportAuthorityMismatchCategoryV1>,
    pub blocking_codes: Vec<String>,
    pub chain_digest: String,
}

#[derive(Debug, Clone)]
pub struct OperatorExportAuthorityInputs<'a> {
    pub applied_scope: &'a AppliedSupportedSetContextV1,
    pub review_packet: &'a OperatorReviewPacketV1,
    pub signoff: &'a OperatorSignoffDecisionV1,
    pub workflow_chain: &'a OperatorWorkflowChainV1,
    pub export_context_digest_prefix: Option<String>,
    pub export_scope_digest_prefix: Option<String>,
}

pub fn derive_operator_export_authority_chain(
    inputs: OperatorExportAuthorityInputs<'_>,
) -> Result<OperatorExportAuthorityChainV1, OpsError> {
    let mut mismatches = BTreeSet::new();
    let mut blocking = BTreeSet::new();

    if inputs.applied_scope.applied_set_digest_prefix.is_empty()
        || inputs.applied_scope.context_digest.is_empty()
    {
        mismatches.insert(OperatorExportAuthorityMismatchCategoryV1::AppliedScopeMissing);
        blocking.insert("APPLIED_SCOPE_MISSING".to_string());
    }

    let applied_set = inputs.applied_scope.applied_set_digest_prefix.as_str();
    let applied_context = prefix_hex(&inputs.applied_scope.context_digest, DIGEST_PREFIX_LEN);

    if inputs.review_packet.applied_supported_set_digest_prefix != applied_set
        || inputs.review_packet.applied_context_digest_prefix != applied_context
    {
        mismatches.insert(OperatorExportAuthorityMismatchCategoryV1::ReviewPacketScopeMismatch);
        blocking.insert("REVIEW_PACKET_SCOPE_MISMATCH".to_string());
    }

    if inputs.signoff.applied_supported_set_digest_prefix != applied_set
        || inputs.signoff.applied_context_digest_prefix != applied_context
    {
        mismatches.insert(OperatorExportAuthorityMismatchCategoryV1::SignoffScopeMismatch);
        blocking.insert("SIGNOFF_SCOPE_MISMATCH".to_string());
    }

    if inputs.workflow_chain.applied_supported_set_digest_prefix != applied_set
        || inputs.workflow_chain.applied_context_digest_prefix != applied_context
    {
        mismatches.insert(OperatorExportAuthorityMismatchCategoryV1::WorkflowScopeMismatch);
        blocking.insert("WORKFLOW_SCOPE_MISMATCH".to_string());
    }

    let expected_reduction = inputs
        .review_packet
        .reviewability_reduction_digest_prefix
        .as_str();
    if expected_reduction == "MISSING"
        || inputs.signoff.reviewability_reduction_digest_prefix != expected_reduction
        || inputs.workflow_chain.reviewability_reduction_digest_prefix != expected_reduction
    {
        mismatches.insert(OperatorExportAuthorityMismatchCategoryV1::ReviewabilityBasisMismatch);
        blocking.insert("REVIEWABILITY_BASIS_MISMATCH".to_string());
    }
    let expected_spine = inputs
        .review_packet
        .canonical_readiness_spine_digest_prefix
        .as_str();
    if expected_spine == "MISSING"
        || inputs.signoff.canonical_readiness_spine_digest_prefix != expected_spine
        || inputs
            .workflow_chain
            .canonical_readiness_spine_digest_prefix
            != expected_spine
    {
        mismatches.insert(OperatorExportAuthorityMismatchCategoryV1::ReadinessSpineMismatch);
        blocking.insert("CANONICAL_READINESS_SPINE_REQUIRED".to_string());
    }

    if let Some(scope_digest_prefix) = inputs.export_scope_digest_prefix.as_ref() {
        if scope_digest_prefix != applied_set {
            mismatches
                .insert(OperatorExportAuthorityMismatchCategoryV1::ExportContextScopeMismatch);
            blocking.insert("EXPORT_CONTEXT_SCOPE_MISMATCH".to_string());
        }
    }

    let mut chain = OperatorExportAuthorityChainV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: inputs.applied_scope.applied_set_digest_prefix.clone(),
        applied_context_digest_prefix: applied_context,
        reviewability_reduction_digest_prefix: inputs
            .review_packet
            .reviewability_reduction_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: inputs
            .review_packet
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_readiness_authority_digest_prefix: inputs
            .review_packet
            .canonical_readiness_authority_digest_prefix
            .clone(),
        operator_review_packet_digest_prefix: prefix_hex(
            &inputs.review_packet.packet_digest,
            DIGEST_PREFIX_LEN,
        ),
        operator_signoff_digest_prefix: prefix_hex(
            &inputs.signoff.decision_digest,
            DIGEST_PREFIX_LEN,
        ),
        operator_workflow_chain_digest_prefix: prefix_hex(
            &inputs.workflow_chain.chain_digest,
            DIGEST_PREFIX_LEN,
        ),
        export_context_digest_prefix: inputs.export_context_digest_prefix,
        authority_chain_status: if mismatches.is_empty() {
            OperatorExportAuthorityChainStatusV1::Pass
        } else {
            OperatorExportAuthorityChainStatusV1::Fail
        },
        mismatch_categories: mismatches.into_iter().collect(),
        blocking_codes: blocking.into_iter().take(BLOCKING_CAP).collect(),
        chain_digest: String::new(),
    };
    chain.chain_digest = chain_digest(&chain)?;
    Ok(chain)
}

pub fn operator_export_chain_check(
    workdir: &Path,
    out: &Path,
) -> Result<OperatorExportAuthorityChainV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active =
        models_active_review_snapshot(workdir, &workdir.join("out/active_review_snapshot.json"))?;
    let truths = crate::derive_slot_reviewability_truths_from_active(&applied, &backend, &active)?;
    let reduction = reduce_reviewability(&applied, &truths)?;
    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_export_chain_check.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_export_chain_check.json"),
    )?;
    let workflow_chain = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_export_chain_check.json"),
    )?;

    let (export_context_digest_prefix, export_scope_digest_prefix) =
        discover_export_context(workdir)?;

    let mut chain = derive_operator_export_authority_chain(OperatorExportAuthorityInputs {
        applied_scope: &applied,
        review_packet: &review_packet,
        signoff: &signoff,
        workflow_chain: &workflow_chain,
        export_context_digest_prefix,
        export_scope_digest_prefix,
    })?;

    if chain.reviewability_reduction_digest_prefix
        != prefix_hex(&reduction.reduction_digest, DIGEST_PREFIX_LEN)
    {
        chain.authority_chain_status = OperatorExportAuthorityChainStatusV1::Fail;
        if !chain
            .mismatch_categories
            .contains(&OperatorExportAuthorityMismatchCategoryV1::ReviewabilityBasisMismatch)
        {
            chain
                .mismatch_categories
                .push(OperatorExportAuthorityMismatchCategoryV1::ReviewabilityBasisMismatch);
            chain.mismatch_categories.sort();
            chain.mismatch_categories.dedup();
        }
        if !chain
            .blocking_codes
            .iter()
            .any(|code| code == "REVIEWABILITY_BASIS_MISMATCH")
        {
            chain
                .blocking_codes
                .push("REVIEWABILITY_BASIS_MISMATCH".to_string());
            chain.blocking_codes.sort();
            chain.blocking_codes.dedup();
            chain.blocking_codes.truncate(BLOCKING_CAP);
        }
        chain.chain_digest = chain_digest(&chain)?;
    }

    let absolute_path = workdir.join("out/canonical_convergence_continuity_sweep.json");
    match fs::read_to_string(&absolute_path) {
        Ok(body) => {
            let probe: CanonicalConvergenceContinuityProbeV1 = serde_json::from_str(&body)?;
            if !matches!(
                probe.continuity_status,
                CanonicalConvergenceContinuityStatusV1::Pass
            ) {
                chain.authority_chain_status = OperatorExportAuthorityChainStatusV1::Fail;
                chain.mismatch_categories.push(
                    OperatorExportAuthorityMismatchCategoryV1::CanonicalConvergenceContinuityFail,
                );
                chain
                    .blocking_codes
                    .push("CANONICAL_CONVERGENCE_CONTINUITY_REQUIRED".to_string());
            }
        }
        Err(_) => {
            chain.authority_chain_status = OperatorExportAuthorityChainStatusV1::Fail;
            chain.mismatch_categories.push(
                OperatorExportAuthorityMismatchCategoryV1::CanonicalConvergenceContinuityMissing,
            );
            chain
                .blocking_codes
                .push("CANONICAL_CONVERGENCE_CONTINUITY_REQUIRED".to_string());
        }
    }
    chain.mismatch_categories.sort();
    chain.mismatch_categories.dedup();
    chain.blocking_codes.sort();
    chain.blocking_codes.dedup();
    chain.blocking_codes.truncate(BLOCKING_CAP);
    chain.chain_digest = chain_digest(&chain)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&chain)?)?;
    Ok(chain)
}

fn discover_export_context(workdir: &Path) -> Result<(Option<String>, Option<String>), OpsError> {
    let repro = workdir.join("out/repro_pack_manifest.json");
    if repro.exists() {
        let manifest: ReproPackManifestV1 = serde_json::from_str(&fs::read_to_string(repro)?)?;
        return Ok((
            Some(prefix_hex(
                &manifest.export_context.context_digest,
                DIGEST_PREFIX_LEN,
            )),
            Some(manifest.export_context.supported_slot_set_digest_prefix),
        ));
    }

    let bugkit = workdir.join("out/bugkit_manifest.json");
    if bugkit.exists() {
        let manifest: BugKitManifestV1 = serde_json::from_str(&fs::read_to_string(bugkit)?)?;
        return Ok((
            Some(prefix_hex(
                &manifest.export_context.context_digest,
                DIGEST_PREFIX_LEN,
            )),
            Some(manifest.export_context.supported_slot_set_digest_prefix),
        ));
    }

    Ok((None, None))
}

fn chain_digest(chain: &OperatorExportAuthorityChainV1) -> Result<String, OpsError> {
    let mut digestible = chain.clone();
    digestible.chain_digest.clear();
    Ok(sha256_hex(&serde_json::to_vec(&digestible)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_is_stable_for_same_inputs() {
        let applied = AppliedSupportedSetContextV1 {
            schema_version: 1,
            applied_set_digest_prefix: "aa".repeat(8),
            slots: vec!["slot_a".to_string()],
            decision: crate::SupportedRealSlotSetExecutionDecisionV2::Frozen,
            previous_set_digest_prefix: "bb".repeat(8),
            policy_digest_prefix: "cc".repeat(8),
            context_digest: "dd".repeat(32),
            compatibility_code: None,
        };
        let review = OperatorReviewPacketV1 {
            schema_version: 1,
            review_stage: crate::OperatorReviewStageV1::ReviewShadowReady,
            supported_slot_set_digest: applied.applied_set_digest_prefix.clone(),
            policy_graph_digest_prefix: "11".repeat(8),
            manifest_digest_prefix: "22".repeat(8),
            applied_supported_set_digest_prefix: applied.applied_set_digest_prefix.clone(),
            applied_context_digest_prefix: prefix_hex(&applied.context_digest, DIGEST_PREFIX_LEN),
            reviewability_reduction_digest_prefix: "33".repeat(8),
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
            final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
            primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
            artifacts: crate::operator_review_packet::OperatorReviewPacketArtifactsV1 {
                backend_evidence_snapshot_digest_prefix: "44".repeat(8),
                active_review_snapshot_digest_prefix: "55".repeat(8),
                operator_signoff_digest_prefix: "66".repeat(8),
                operator_report_digest_prefix: "77".repeat(8),
                gate_digests: crate::operator_review_packet::OperatorReviewPacketGateDigestsV1 {
                    v0: "1".repeat(16),
                    v1: "2".repeat(16),
                    v2: "3".repeat(16),
                    v3: "4".repeat(16),
                    v4: "5".repeat(16),
                },
                backend_resolution_digest_prefix: None,
                applied_supported_set_context_digest_prefix: "88".repeat(8),
            },
            supported_slots: vec![],
            blocking_codes: vec![],
            remediation_codes: vec![],
            packet_digest: "99".repeat(32),
        };
        let signoff = OperatorSignoffDecisionV1 {
            schema_version: 1,
            decision: crate::SignoffDecisionStateV1::ReadyForShadow,
            supported_slot_set_digest: applied.applied_set_digest_prefix.clone(),
            policy_graph_digest_prefix: "11".repeat(8),
            manifest_digest_prefix: "22".repeat(8),
            evidence_snapshot_digest_prefix: "44".repeat(8),
            active_review_snapshot_digest_prefix: None,
            operator_report_digest_prefix: "77".repeat(8),
            gate_report_digests: crate::operator_signoff::GateReportDigestsV1 {
                v0: "1".repeat(16),
                v1: "2".repeat(16),
                v2: "3".repeat(16),
                v3: "4".repeat(16),
            },
            applied_supported_set_digest_prefix: applied.applied_set_digest_prefix.clone(),
            applied_context_digest_prefix: prefix_hex(&applied.context_digest, DIGEST_PREFIX_LEN),
            reviewability_reduction_digest_prefix: review
                .reviewability_reduction_digest_prefix
                .clone(),
            canonical_readiness_spine_digest_prefix: review
                .canonical_readiness_spine_digest_prefix
                .clone(),
            canonical_readiness_authority_digest_prefix: review
                .canonical_readiness_authority_digest_prefix
                .clone(),
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
            final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
            primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
            reasons: vec![],
            remediation_codes: vec![],
            canonical_remediation_codes: vec![],
            decision_digest: "ab".repeat(32),
        };
        let workflow = OperatorWorkflowChainV1 {
            schema_version: 1,
            workflow_stage: crate::OperatorWorkflowStageV2::WorkflowReviewReady,
            governance_surfaces_digest_prefix: "cd".repeat(8),
            applied_supported_scope_digest_prefix: applied.applied_set_digest_prefix.clone(),
            applied_supported_set_digest_prefix: applied.applied_set_digest_prefix.clone(),
            applied_context_digest_prefix: prefix_hex(&applied.context_digest, DIGEST_PREFIX_LEN),
            reviewability_reduction_digest_prefix: review
                .reviewability_reduction_digest_prefix
                .clone(),
            canonical_readiness_spine_digest_prefix: review
                .canonical_readiness_spine_digest_prefix
                .clone(),
            canonical_readiness_authority_digest_prefix: review
                .canonical_readiness_authority_digest_prefix
                .clone(),
            canonical_governance_entry_digest_prefix: "MISSING".to_string(),
            final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
            governance_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_governance_authority_digest_prefix: "MISSING".to_string(),
            governance_absolute_sweep_digest_prefix: "MISSING".to_string(),
            governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
            absolute_final_governance_terminal_sweep_digest_prefix: "MISSING".to_string(),
            governance_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            final_readiness_consumer_authority_digest_prefix: "MISSING".to_string(),
            readiness_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_readiness_authority_digest_prefix: "MISSING".to_string(),
            readiness_absolute_sweep_digest_prefix: "MISSING".to_string(),
            readiness_terminal_sweep_digest_prefix: "MISSING".to_string(),
            readiness_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            final_bundle_residual_sweep_digest_prefix: "MISSING".to_string(),
            final_primary_semantics_residual_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_primary_semantics_authority_digest_prefix: "MISSING".to_string(),
            primary_semantics_absolute_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_terminal_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            primary_semantics_convergence_sweep_digest_prefix: "MISSING".to_string(),
            governance_convergence_sweep_digest_prefix: "MISSING".to_string(),
            readiness_convergence_sweep_digest_prefix: "MISSING".to_string(),
            bundle_convergence_sweep_digest_prefix: "MISSING".to_string(),
            canonical_convergence_continuity_authority_digest_prefix: "MISSING".to_string(),
            bundle_terminal_sweep_digest_prefix: "MISSING".to_string(),
            residual_free_continuity_authority_digest_prefix: "MISSING".to_string(),
            final_input_continuity_authority_digest_prefix: "MISSING".to_string(),
            terminal_absolute_final_input_continuity_authority_digest_prefix: "MISSING".to_string(),
            ultimate_terminal_absolute_final_input_continuity_authority_digest_prefix: "MISSING"
                .to_string(),
            operator_review_packet_digest_prefix: "ef".repeat(8),
            operator_signoff_digest_prefix: "01".repeat(8),
            interop_matrix_digest_prefix: "23".repeat(8),
            export_normalize_check_digest_prefix: "45".repeat(8),
            export_targets: crate::OperatorWorkflowExportTargetsV1 {
                repro_ready: true,
                bugkit_ready: true,
            },
            blocking_codes: vec![],
            remediation_codes: vec![],
            chain_digest: "67".repeat(32),
        };

        let c1 = derive_operator_export_authority_chain(OperatorExportAuthorityInputs {
            applied_scope: &applied,
            review_packet: &review,
            signoff: &signoff,
            workflow_chain: &workflow,
            export_context_digest_prefix: None,
            export_scope_digest_prefix: None,
        })
        .expect("chain 1");
        let c2 = derive_operator_export_authority_chain(OperatorExportAuthorityInputs {
            applied_scope: &applied,
            review_packet: &review,
            signoff: &signoff,
            workflow_chain: &workflow,
            export_context_digest_prefix: None,
            export_scope_digest_prefix: None,
        })
        .expect("chain 2");
        assert_eq!(c1, c2);
    }
}
