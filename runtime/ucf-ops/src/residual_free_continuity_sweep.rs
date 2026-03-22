use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    bundle_residual_sweep, continuity_authority_check, derive_canonical_governance_entry,
    final_bundle_consumer_sweep, final_continuity_sweep, final_governance_consumer_sweep,
    final_primary_semantics_sweep, final_readiness_consumer_sweep, governance_residual_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_roundtrip_chain_check, operator_signoff,
    operator_workflow_chain, prefix_hex, primary_semantics_residual_sweep,
    readiness_residual_sweep, CanonicalRoundTripChainStatusV1, ContinuityAuthorityStatusV1,
    FinalBundleConsumerAuthorityStatusV1, FinalBundleResidualSweepStatusV1,
    FinalContinuityStatusV2, FinalGovernanceConsumerAuthorityStatusV1,
    FinalPrimarySemanticsConsumerAuthorityStatusV1, FinalPrimarySemanticsResidualSweepStatusV1,
    FinalReadinessConsumerAuthorityStatusV1, FinalReadinessResidualSweepStatusV1,
    GovernanceResidualSweepStatusV1, OperatorReviewPacketArgs, OperatorSignoffArgs,
    OperatorWorkflowArgs, OpsError,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CODE_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreeContinuityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreeContinuityMismatchCategoryV1 {
    ResidualFreeGovernanceMismatch,
    ResidualFreeScopeMismatch,
    ResidualFreeReadinessMismatch,
    ResidualFreePrimarySemanticsMismatch,
    ResidualFreeWorkflowMismatch,
    ResidualFreeBundleMismatch,
    ResidualPathDependencyPresent,
    LegacyTopLevelContinuityPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeContinuityAuthorityV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,

    pub canonical_readiness_spine_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub final_bundle_consumer_authority_digest_prefix: String,
    pub final_bundle_residual_sweep_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
    pub final_primary_semantics_residual_sweep_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub canonical_roundtrip_chain_digest_prefix: String,
    pub final_continuity_authority_digest_prefix: String,
    pub continuity_status: ResidualFreeContinuityStatusV1,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub authority_digest: String,
}

pub fn residual_free_continuity_sweep(
    workdir: &Path,
    bundle: &Path,
    out: &Path,
) -> Result<ResidualFreeContinuityAuthorityV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_residual_free_continuity_sweep.json"),
    )?;
    let surfaces = crate::validate_governance_primary_surfaces_with_applied_scope(
        &backend, &active, &applied,
    )?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;

    let governance_sweep = final_governance_consumer_sweep(
        workdir,
        &workdir.join("out/final_governance_consumer_sweep_residual_free_continuity_sweep.json"),
    )?;
    let governance_residual = governance_residual_sweep(
        workdir,
        &workdir.join("out/governance_residual_sweep_residual_free_continuity_sweep.json"),
    )?;
    let readiness_sweep = final_readiness_consumer_sweep(
        workdir,
        &workdir.join("out/final_readiness_consumer_sweep_residual_free_continuity_sweep.json"),
    )?;
    let readiness_residual = readiness_residual_sweep(
        workdir,
        &workdir.join("out/readiness_residual_sweep_residual_free_continuity_sweep.json"),
    )?;
    let bundle_sweep = final_bundle_consumer_sweep(
        workdir,
        &workdir.join("out/final_bundle_consumer_sweep_residual_free_continuity_sweep.json"),
    )?;
    let bundle_residual = bundle_residual_sweep(
        workdir,
        &workdir.join("out/bundle_residual_sweep_residual_free_continuity_sweep.json"),
    )?;
    let primary_sweep = final_primary_semantics_sweep(
        workdir,
        &workdir.join("out/final_primary_semantics_sweep_residual_free_continuity_sweep.json"),
    )?;
    let primary_residual = primary_semantics_residual_sweep(
        workdir,
        &workdir.join("out/primary_semantics_residual_sweep_residual_free_continuity_sweep.json"),
    )?;

    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_residual_free_continuity_sweep.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_residual_free_continuity_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_residual_free_continuity_sweep.json"),
    )?;
    let roundtrip = operator_roundtrip_chain_check(
        workdir,
        bundle,
        &workdir.join("out/operator_roundtrip_chain_residual_free_continuity_sweep.json"),
    )?;
    let continuity = continuity_authority_check(
        workdir,
        bundle,
        &workdir.join("out/continuity_authority_check_residual_free_continuity_sweep.json"),
    )?;
    let legacy_final_continuity = final_continuity_sweep(
        workdir,
        bundle,
        &workdir.join("out/final_continuity_sweep_residual_free_continuity_sweep.json"),
    )?;

    let mut blocking = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    let expected_applied = &applied.applied_set_digest_prefix;
    if governance_sweep
        .authority
        .applied_supported_set_digest_prefix
        != *expected_applied
        || readiness_sweep
            .authority
            .applied_supported_set_digest_prefix
            != *expected_applied
        || bundle_sweep.authority.applied_supported_set_digest_prefix != *expected_applied
        || review_packet.applied_supported_set_digest_prefix != *expected_applied
        || signoff.applied_supported_set_digest_prefix != *expected_applied
        || workflow.applied_supported_set_digest_prefix != *expected_applied
    {
        blocking.insert(ResidualFreeContinuityMismatchCategoryV1::ResidualFreeScopeMismatch);
        remediation.insert("run_models_applied_scope_check".to_string());
    }

    let governance_entry_prefix = prefix_hex(&governance.authority_digest, DIGEST_PREFIX_LEN);
    if governance_sweep
        .authority
        .canonical_governance_entry_digest_prefix
        != governance_entry_prefix
    {
        blocking.insert(ResidualFreeContinuityMismatchCategoryV1::ResidualFreeGovernanceMismatch);
        remediation.insert("run_final_governance_consumer_sweep".to_string());
    }

    if !matches!(
        governance_sweep.authority.authority_status,
        FinalGovernanceConsumerAuthorityStatusV1::Pass
    ) || !matches!(
        governance_residual.sweep.sweep_status,
        GovernanceResidualSweepStatusV1::Pass
    ) {
        blocking.insert(ResidualFreeContinuityMismatchCategoryV1::ResidualFreeGovernanceMismatch);
        remediation.insert("run_governance_residual_sweep".to_string());
    }

    if !matches!(
        readiness_sweep.authority.authority_status,
        FinalReadinessConsumerAuthorityStatusV1::Pass
    ) || !matches!(
        readiness_residual.sweep.sweep_status,
        FinalReadinessResidualSweepStatusV1::Pass
    ) {
        blocking.insert(ResidualFreeContinuityMismatchCategoryV1::ResidualFreeReadinessMismatch);
        remediation.insert("run_readiness_residual_sweep".to_string());
    }

    if !matches!(
        primary_sweep.authority.authority_status,
        FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass
    ) || !matches!(
        primary_residual.sweep.sweep_status,
        FinalPrimarySemanticsResidualSweepStatusV1::Pass
    ) {
        blocking
            .insert(ResidualFreeContinuityMismatchCategoryV1::ResidualFreePrimarySemanticsMismatch);
        remediation.insert("run_primary_semantics_residual_sweep".to_string());
    }

    if !workflow.blocking_codes.is_empty() {
        blocking.insert(ResidualFreeContinuityMismatchCategoryV1::ResidualFreeWorkflowMismatch);
        remediation.insert("run_operator_workflow".to_string());
    }

    if !matches!(
        roundtrip.roundtrip_status,
        CanonicalRoundTripChainStatusV1::Pass
    ) || !matches!(
        continuity.continuity_status,
        ContinuityAuthorityStatusV1::Pass
    ) || !matches!(
        bundle_sweep.authority.authority_status,
        FinalBundleConsumerAuthorityStatusV1::Pass
    ) || !matches!(
        bundle_residual.sweep.sweep_status,
        FinalBundleResidualSweepStatusV1::Pass
    ) {
        blocking.insert(ResidualFreeContinuityMismatchCategoryV1::ResidualFreeBundleMismatch);
        remediation.insert("run_bundle_residual_sweep".to_string());
    }

    let residual_dependency_present = governance_residual.sweep.residual_path_count > 0
        || readiness_residual.sweep.residual_path_count > 0
        || bundle_residual.sweep.residual_path_count > 0
        || primary_residual.sweep.residual_path_count > 0;
    if residual_dependency_present {
        blocking.insert(ResidualFreeContinuityMismatchCategoryV1::ResidualPathDependencyPresent);
        remediation.insert("remove_residual_path_dependencies".to_string());
    }

    let legacy_present = matches!(
        continuity.continuity_status,
        ContinuityAuthorityStatusV1::LegacyPresent
    ) || matches!(
        legacy_final_continuity.continuity_status,
        FinalContinuityStatusV2::LegacyPresent
    );
    if legacy_present {
        blocking.insert(ResidualFreeContinuityMismatchCategoryV1::LegacyTopLevelContinuityPresent);
        remediation.insert("demote_legacy_top_level_continuity".to_string());
    }

    let mut report = ResidualFreeContinuityAuthorityV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: governance_entry_prefix,
        final_governance_consumer_authority_digest_prefix: prefix_hex(
            &governance_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        final_governance_residual_sweep_digest_prefix: prefix_hex(
            &governance_residual.sweep.sweep_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_readiness_spine_digest_prefix: readiness_sweep
            .authority
            .canonical_readiness_spine_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: prefix_hex(
            &readiness_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        final_readiness_residual_sweep_digest_prefix: prefix_hex(
            &readiness_residual.sweep.sweep_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_bundle_spine_digest_prefix: bundle_sweep
            .authority
            .canonical_bundle_spine_digest_prefix,
        final_bundle_consumer_authority_digest_prefix: prefix_hex(
            &bundle_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        final_bundle_residual_sweep_digest_prefix: prefix_hex(
            &bundle_residual.sweep.sweep_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_primary_semantics_authority_digest_prefix: primary_sweep
            .authority
            .canonical_primary_semantics_authority_digest_prefix,
        final_primary_semantics_consumer_authority_digest_prefix: prefix_hex(
            &primary_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        final_primary_semantics_residual_sweep_digest_prefix: prefix_hex(
            &primary_residual.sweep.sweep_digest,
            DIGEST_PREFIX_LEN,
        ),
        operator_review_packet_digest_prefix: prefix_hex(
            &review_packet.packet_digest,
            DIGEST_PREFIX_LEN,
        ),
        operator_signoff_digest_prefix: prefix_hex(&signoff.decision_digest, DIGEST_PREFIX_LEN),
        operator_workflow_chain_digest_prefix: prefix_hex(
            &workflow.chain_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_roundtrip_chain_digest_prefix: prefix_hex(
            &roundtrip.chain_digest,
            DIGEST_PREFIX_LEN,
        ),
        final_continuity_authority_digest_prefix: prefix_hex(
            &legacy_final_continuity.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        continuity_status: if legacy_present {
            ResidualFreeContinuityStatusV1::LegacyPresent
        } else if blocking.is_empty() {
            ResidualFreeContinuityStatusV1::Pass
        } else {
            ResidualFreeContinuityStatusV1::Fail
        },
        blocking_codes: blocking
            .into_iter()
            .map(|code| match code {
                ResidualFreeContinuityMismatchCategoryV1::ResidualFreeGovernanceMismatch => {
                    "RESIDUAL_FREE_GOVERNANCE_MISMATCH".to_string()
                }
                ResidualFreeContinuityMismatchCategoryV1::ResidualFreeScopeMismatch => {
                    "RESIDUAL_FREE_SCOPE_MISMATCH".to_string()
                }
                ResidualFreeContinuityMismatchCategoryV1::ResidualFreeReadinessMismatch => {
                    "RESIDUAL_FREE_READINESS_MISMATCH".to_string()
                }
                ResidualFreeContinuityMismatchCategoryV1::ResidualFreePrimarySemanticsMismatch => {
                    "RESIDUAL_FREE_PRIMARY_SEMANTICS_MISMATCH".to_string()
                }
                ResidualFreeContinuityMismatchCategoryV1::ResidualFreeWorkflowMismatch => {
                    "RESIDUAL_FREE_WORKFLOW_MISMATCH".to_string()
                }
                ResidualFreeContinuityMismatchCategoryV1::ResidualFreeBundleMismatch => {
                    "RESIDUAL_FREE_BUNDLE_MISMATCH".to_string()
                }
                ResidualFreeContinuityMismatchCategoryV1::ResidualPathDependencyPresent => {
                    "RESIDUAL_PATH_DEPENDENCY_PRESENT".to_string()
                }
                ResidualFreeContinuityMismatchCategoryV1::LegacyTopLevelContinuityPresent => {
                    "LEGACY_TOP_LEVEL_CONTINUITY_PRESENT".to_string()
                }
            })
            .take(CODE_CAP)
            .collect(),
        remediation_codes: remediation.into_iter().take(CODE_CAP).collect(),
        authority_digest: String::new(),
    };
    report.authority_digest = continuity_digest(&report)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn continuity_digest(report: &ResidualFreeContinuityAuthorityV1) -> Result<String, OpsError> {
    let mut canonical = report.clone();
    canonical.authority_digest.clear();
    canonical.blocking_codes.sort();
    canonical.remediation_codes.sort();
    Ok(crate::sha256_hex(&serde_json::to_vec(&canonical)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_free_continuity_digest_stable() {
        let mut report = ResidualFreeContinuityAuthorityV1 {
            schema_version: 1,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            final_governance_consumer_authority_digest_prefix: "33".repeat(8),
            final_governance_residual_sweep_digest_prefix: "44".repeat(8),
            canonical_readiness_spine_digest_prefix: "55".repeat(8),
            final_readiness_consumer_authority_digest_prefix: "66".repeat(8),
            final_readiness_residual_sweep_digest_prefix: "77".repeat(8),
            canonical_bundle_spine_digest_prefix: "88".repeat(8),
            final_bundle_consumer_authority_digest_prefix: "99".repeat(8),
            final_bundle_residual_sweep_digest_prefix: "aa".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "bb".repeat(8),
            final_primary_semantics_consumer_authority_digest_prefix: "cc".repeat(8),
            final_primary_semantics_residual_sweep_digest_prefix: "dd".repeat(8),
            operator_review_packet_digest_prefix: "ee".repeat(8),
            operator_signoff_digest_prefix: "ff".repeat(8),
            operator_workflow_chain_digest_prefix: "12".repeat(8),
            canonical_roundtrip_chain_digest_prefix: "34".repeat(8),
            final_continuity_authority_digest_prefix: "56".repeat(8),
            continuity_status: ResidualFreeContinuityStatusV1::Pass,
            blocking_codes: vec![],
            remediation_codes: vec!["x".to_string()],
            authority_digest: String::new(),
        };
        report.authority_digest = continuity_digest(&report).expect("digest");
        assert_eq!(
            report.authority_digest,
            continuity_digest(&report).expect("digest b")
        );
    }
}
