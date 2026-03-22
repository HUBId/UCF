use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, operator_review_packet,
    operator_roundtrip_chain_check, operator_signoff, operator_workflow_chain, prefix_hex,
    residual_free_bundle_sweep, residual_free_continuity_sweep, residual_free_governance_sweep,
    residual_free_primary_semantics_sweep, residual_free_readiness_sweep,
    validate_governance_primary_surfaces_with_applied_scope, CanonicalRoundTripChainStatusV1,
    OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
    ResidualFreeBundleConsumerAuthorityStatusV1, ResidualFreeContinuityStatusV1,
    ResidualFreeGovernanceConsumerAuthorityStatusV1, ResidualFreePrimarySemanticsAuthorityStatusV1,
    ResidualFreeReadinessConsumerAuthorityStatusV1,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CODE_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalInputContinuityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalInputContinuityMismatchCategoryV1 {
    FinalInputGovernanceMismatch,
    FinalInputScopeMismatch,
    FinalInputReadinessMismatch,
    FinalInputPrimarySemanticsMismatch,
    FinalInputWorkflowMismatch,
    FinalInputBundleMismatch,
    ResidualPathDependencyPresent,
    LegacyTopLevelContinuityPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalInputContinuityAuthorityV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub residual_free_bundle_consumer_authority_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub residual_free_primary_semantics_consumer_authority_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub canonical_roundtrip_chain_digest_prefix: String,
    pub residual_free_continuity_authority_digest_prefix: String,
    pub continuity_status: FinalInputContinuityStatusV1,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub authority_digest: String,
}

pub fn final_input_continuity_sweep(
    workdir: &Path,
    bundle: &Path,
    out: &Path,
) -> Result<FinalInputContinuityAuthorityV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_final_input_continuity_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;

    let governance_sweep = residual_free_governance_sweep(
        workdir,
        &workdir.join("out/residual_free_governance_sweep_final_input_continuity_sweep.json"),
    )?;
    let readiness_sweep = residual_free_readiness_sweep(
        workdir,
        &workdir.join("out/residual_free_readiness_sweep_final_input_continuity_sweep.json"),
    )?;
    let bundle_sweep = residual_free_bundle_sweep(
        workdir,
        &workdir.join("out/residual_free_bundle_sweep_final_input_continuity_sweep.json"),
    )?;
    let primary_sweep = residual_free_primary_semantics_sweep(
        workdir,
        &workdir
            .join("out/residual_free_primary_semantics_sweep_final_input_continuity_sweep.json"),
    )?;
    let residual_free_continuity = residual_free_continuity_sweep(
        workdir,
        bundle,
        &workdir.join("out/residual_free_continuity_sweep_final_input_continuity_sweep.json"),
    )?;

    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_final_input_continuity_sweep.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_final_input_continuity_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_final_input_continuity_sweep.json"),
    )?;
    let roundtrip = operator_roundtrip_chain_check(
        workdir,
        bundle,
        &workdir.join("out/operator_roundtrip_chain_final_input_continuity_sweep.json"),
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
        blocking.insert(FinalInputContinuityMismatchCategoryV1::FinalInputScopeMismatch);
        remediation.insert("run_models_applied_scope_check".to_string());
    }

    let governance_entry_prefix = prefix_hex(&governance.authority_digest, DIGEST_PREFIX_LEN);
    if governance_sweep
        .authority
        .canonical_governance_entry_digest_prefix
        != governance_entry_prefix
    {
        blocking.insert(FinalInputContinuityMismatchCategoryV1::FinalInputGovernanceMismatch);
        remediation.insert("run_residual_free_governance_sweep".to_string());
    }
    if !matches!(
        governance_sweep.authority.authority_status,
        ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass
    ) {
        blocking.insert(FinalInputContinuityMismatchCategoryV1::FinalInputGovernanceMismatch);
        remediation.insert("run_residual_free_governance_sweep".to_string());
    }
    if !matches!(
        readiness_sweep.authority.authority_status,
        ResidualFreeReadinessConsumerAuthorityStatusV1::Pass
    ) {
        blocking.insert(FinalInputContinuityMismatchCategoryV1::FinalInputReadinessMismatch);
        remediation.insert("run_residual_free_readiness_sweep".to_string());
    }
    if !matches!(
        primary_sweep.authority.authority_status,
        ResidualFreePrimarySemanticsAuthorityStatusV1::Pass
    ) {
        blocking.insert(FinalInputContinuityMismatchCategoryV1::FinalInputPrimarySemanticsMismatch);
        remediation.insert("run_residual_free_primary_semantics_sweep".to_string());
    }
    if !workflow.blocking_codes.is_empty() {
        blocking.insert(FinalInputContinuityMismatchCategoryV1::FinalInputWorkflowMismatch);
        remediation.insert("run_operator_workflow".to_string());
    }
    if !matches!(
        roundtrip.roundtrip_status,
        CanonicalRoundTripChainStatusV1::Pass
    ) || !matches!(
        bundle_sweep.authority.authority_status,
        ResidualFreeBundleConsumerAuthorityStatusV1::Pass
    ) {
        blocking.insert(FinalInputContinuityMismatchCategoryV1::FinalInputBundleMismatch);
        remediation.insert("run_operator_roundtrip_chain_check".to_string());
    }

    let residual_dependency_present = governance_sweep.authority.residual_path_count > 0
        || readiness_sweep.authority.residual_path_count > 0
        || bundle_sweep.authority.residual_path_count > 0
        || primary_sweep.authority.residual_path_count > 0;
    if residual_dependency_present {
        blocking.insert(FinalInputContinuityMismatchCategoryV1::ResidualPathDependencyPresent);
        remediation.insert("remove_residual_path_dependencies".to_string());
    }

    let legacy_present = !matches!(
        residual_free_continuity.continuity_status,
        ResidualFreeContinuityStatusV1::Pass
    );
    if legacy_present {
        blocking.insert(FinalInputContinuityMismatchCategoryV1::LegacyTopLevelContinuityPresent);
        remediation.insert("demote_legacy_top_level_continuity_surfaces".to_string());
    }

    let mut report = FinalInputContinuityAuthorityV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: governance_entry_prefix,
        residual_free_governance_consumer_authority_digest_prefix: prefix_hex(
            &governance_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_readiness_spine_digest_prefix: readiness_sweep
            .authority
            .canonical_readiness_spine_digest_prefix,
        residual_free_readiness_consumer_authority_digest_prefix: prefix_hex(
            &readiness_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_bundle_spine_digest_prefix: bundle_sweep
            .authority
            .canonical_bundle_spine_digest_prefix,
        residual_free_bundle_consumer_authority_digest_prefix: prefix_hex(
            &bundle_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_primary_semantics_authority_digest_prefix: primary_sweep
            .authority
            .canonical_primary_semantics_authority_digest_prefix,
        residual_free_primary_semantics_consumer_authority_digest_prefix: prefix_hex(
            &primary_sweep.authority.authority_digest,
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
        residual_free_continuity_authority_digest_prefix: prefix_hex(
            &residual_free_continuity.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        continuity_status: if legacy_present {
            FinalInputContinuityStatusV1::LegacyPresent
        } else if blocking.is_empty() {
            FinalInputContinuityStatusV1::Pass
        } else {
            FinalInputContinuityStatusV1::Fail
        },
        blocking_codes: blocking
            .iter()
            .map(|code| match code {
                FinalInputContinuityMismatchCategoryV1::FinalInputGovernanceMismatch => {
                    "FINAL_INPUT_GOVERNANCE_MISMATCH".to_string()
                }
                FinalInputContinuityMismatchCategoryV1::FinalInputScopeMismatch => {
                    "FINAL_INPUT_SCOPE_MISMATCH".to_string()
                }
                FinalInputContinuityMismatchCategoryV1::FinalInputReadinessMismatch => {
                    "FINAL_INPUT_READINESS_MISMATCH".to_string()
                }
                FinalInputContinuityMismatchCategoryV1::FinalInputPrimarySemanticsMismatch => {
                    "FINAL_INPUT_PRIMARY_SEMANTICS_MISMATCH".to_string()
                }
                FinalInputContinuityMismatchCategoryV1::FinalInputWorkflowMismatch => {
                    "FINAL_INPUT_WORKFLOW_MISMATCH".to_string()
                }
                FinalInputContinuityMismatchCategoryV1::FinalInputBundleMismatch => {
                    "FINAL_INPUT_BUNDLE_MISMATCH".to_string()
                }
                FinalInputContinuityMismatchCategoryV1::ResidualPathDependencyPresent => {
                    "RESIDUAL_PATH_DEPENDENCY_PRESENT".to_string()
                }
                FinalInputContinuityMismatchCategoryV1::LegacyTopLevelContinuityPresent => {
                    "LEGACY_TOP_LEVEL_CONTINUITY_PRESENT".to_string()
                }
            })
            .take(CODE_CAP)
            .collect(),
        remediation_codes: remediation.into_iter().take(CODE_CAP).collect(),
        authority_digest: String::new(),
    };
    report.authority_digest = final_input_continuity_digest(&report)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn final_input_continuity_digest(
    report: &FinalInputContinuityAuthorityV1,
) -> Result<String, OpsError> {
    let mut digestible = report.clone();
    digestible.authority_digest.clear();
    Ok(crate::sha256_hex(&serde_json::to_vec(&digestible)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_input_continuity_digest_stable() {
        let mut report = FinalInputContinuityAuthorityV1 {
            schema_version: 1,
            applied_supported_set_digest_prefix: "aa".repeat(8),
            canonical_governance_entry_digest_prefix: "bb".repeat(8),
            residual_free_governance_consumer_authority_digest_prefix: "cc".repeat(8),
            canonical_readiness_spine_digest_prefix: "dd".repeat(8),
            residual_free_readiness_consumer_authority_digest_prefix: "ee".repeat(8),
            canonical_bundle_spine_digest_prefix: "ff".repeat(8),
            residual_free_bundle_consumer_authority_digest_prefix: "11".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "22".repeat(8),
            residual_free_primary_semantics_consumer_authority_digest_prefix: "33".repeat(8),
            operator_review_packet_digest_prefix: "44".repeat(8),
            operator_signoff_digest_prefix: "55".repeat(8),
            operator_workflow_chain_digest_prefix: "66".repeat(8),
            canonical_roundtrip_chain_digest_prefix: "77".repeat(8),
            residual_free_continuity_authority_digest_prefix: "88".repeat(8),
            continuity_status: FinalInputContinuityStatusV1::Pass,
            blocking_codes: Vec::new(),
            remediation_codes: Vec::new(),
            authority_digest: String::new(),
        };
        report.authority_digest = final_input_continuity_digest(&report).expect("digest");
        let stable = final_input_continuity_digest(&report).expect("stable digest");
        assert_eq!(stable, report.authority_digest);
    }
}
