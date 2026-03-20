use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    continuity_authority_check, derive_canonical_governance_entry, final_bundle_consumer_sweep,
    final_governance_consumer_sweep, final_primary_semantics_sweep, final_readiness_consumer_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_roundtrip_chain_check, operator_signoff,
    operator_workflow_chain, prefix_hex, validate_governance_primary_surfaces_with_applied_scope,
    CanonicalRoundTripChainStatusV1, ContinuityAuthorityStatusV1,
    FinalBundleConsumerAuthorityStatusV1, FinalGovernanceConsumerAuthorityStatusV1,
    FinalPrimarySemanticsConsumerAuthorityStatusV1, FinalReadinessConsumerAuthorityStatusV1,
    OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CODE_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalContinuityStatusV2 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalContinuityMismatchCategoryV1 {
    FinalContinuityGovernanceMismatch,
    FinalContinuityScopeMismatch,
    FinalContinuityReadinessMismatch,
    FinalContinuityPrimarySemanticsMismatch,
    FinalContinuityWorkflowMismatch,
    FinalContinuityBundleMismatch,
    LegacyContinuityProofPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalContinuityAuthorityV2 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub final_bundle_consumer_authority_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub canonical_roundtrip_chain_digest_prefix: String,
    pub canonical_continuity_authority_digest_prefix: String,
    pub continuity_status: FinalContinuityStatusV2,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub authority_digest: String,
}

pub fn final_continuity_sweep(
    workdir: &Path,
    bundle: &Path,
    out: &Path,
) -> Result<FinalContinuityAuthorityV2, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_final_continuity_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;

    let governance_sweep = final_governance_consumer_sweep(
        workdir,
        &workdir.join("out/final_governance_consumer_sweep_final_continuity_sweep.json"),
    )?;
    let readiness_sweep = final_readiness_consumer_sweep(
        workdir,
        &workdir.join("out/final_readiness_consumer_sweep_final_continuity_sweep.json"),
    )?;
    let bundle_sweep = final_bundle_consumer_sweep(
        workdir,
        &workdir.join("out/final_bundle_consumer_sweep_final_continuity_sweep.json"),
    )?;
    let primary_sweep = final_primary_semantics_sweep(
        workdir,
        &workdir.join("out/final_primary_semantics_sweep_final_continuity_sweep.json"),
    )?;

    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_final_continuity_sweep.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_final_continuity_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_final_continuity_sweep.json"),
    )?;
    let roundtrip = operator_roundtrip_chain_check(
        workdir,
        bundle,
        &workdir.join("out/operator_roundtrip_chain_final_continuity_sweep.json"),
    )?;
    let continuity = continuity_authority_check(
        workdir,
        bundle,
        &workdir.join("out/continuity_authority_check_final_continuity_sweep.json"),
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
        blocking.insert(FinalContinuityMismatchCategoryV1::FinalContinuityScopeMismatch);
        remediation.insert("run_models_applied_scope_check".to_string());
    }

    let governance_entry_prefix = prefix_hex(&governance.authority_digest, DIGEST_PREFIX_LEN);
    if governance_sweep
        .authority
        .canonical_governance_entry_digest_prefix
        != governance_entry_prefix
    {
        blocking.insert(FinalContinuityMismatchCategoryV1::FinalContinuityGovernanceMismatch);
        remediation.insert("run_final_governance_consumer_sweep".to_string());
    }

    if !matches!(
        governance_sweep.authority.authority_status,
        FinalGovernanceConsumerAuthorityStatusV1::Pass
    ) {
        blocking.insert(FinalContinuityMismatchCategoryV1::FinalContinuityGovernanceMismatch);
        remediation.insert("run_final_governance_consumer_sweep".to_string());
    }

    if !matches!(
        readiness_sweep.authority.authority_status,
        FinalReadinessConsumerAuthorityStatusV1::Pass
    ) {
        blocking.insert(FinalContinuityMismatchCategoryV1::FinalContinuityReadinessMismatch);
        remediation.insert("run_final_readiness_consumer_sweep".to_string());
    }

    if review_packet.canonical_readiness_spine_digest_prefix
        != readiness_sweep
            .authority
            .canonical_readiness_spine_digest_prefix
        || signoff.canonical_readiness_spine_digest_prefix
            != readiness_sweep
                .authority
                .canonical_readiness_spine_digest_prefix
        || workflow.canonical_readiness_spine_digest_prefix
            != readiness_sweep
                .authority
                .canonical_readiness_spine_digest_prefix
    {
        blocking.insert(FinalContinuityMismatchCategoryV1::FinalContinuityReadinessMismatch);
        remediation.insert("run_readiness_spine_sweep".to_string());
    }

    if !matches!(
        primary_sweep.authority.authority_status,
        FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass
    ) {
        blocking.insert(FinalContinuityMismatchCategoryV1::FinalContinuityPrimarySemanticsMismatch);
        remediation.insert("run_final_primary_semantics_sweep".to_string());
    }

    if !workflow.blocking_codes.is_empty() {
        blocking.insert(FinalContinuityMismatchCategoryV1::FinalContinuityWorkflowMismatch);
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
    ) {
        blocking.insert(FinalContinuityMismatchCategoryV1::FinalContinuityBundleMismatch);
        remediation.insert("run_operator_roundtrip_chain_check".to_string());
    }

    let legacy_present = matches!(
        governance_sweep.authority.authority_status,
        FinalGovernanceConsumerAuthorityStatusV1::LegacyPresent
    ) || matches!(
        readiness_sweep.authority.authority_status,
        FinalReadinessConsumerAuthorityStatusV1::LegacyPresent
    ) || matches!(
        bundle_sweep.authority.authority_status,
        FinalBundleConsumerAuthorityStatusV1::LegacyPresent
    ) || matches!(
        primary_sweep.authority.authority_status,
        FinalPrimarySemanticsConsumerAuthorityStatusV1::LegacyPresent
    ) || matches!(
        continuity.continuity_status,
        ContinuityAuthorityStatusV1::LegacyPresent
    );

    if legacy_present {
        blocking.insert(FinalContinuityMismatchCategoryV1::LegacyContinuityProofPresent);
        remediation.insert("remove_legacy_continuity_proofs".to_string());
    }

    let mut report = FinalContinuityAuthorityV2 {
        schema_version: 2,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: governance_entry_prefix,
        final_governance_consumer_authority_digest_prefix: prefix_hex(
            &governance_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_readiness_spine_digest_prefix: readiness_sweep
            .authority
            .canonical_readiness_spine_digest_prefix,
        final_readiness_consumer_authority_digest_prefix: prefix_hex(
            &readiness_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_bundle_spine_digest_prefix: bundle_sweep
            .authority
            .canonical_bundle_spine_digest_prefix,
        final_bundle_consumer_authority_digest_prefix: prefix_hex(
            &bundle_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_primary_semantics_authority_digest_prefix: primary_sweep
            .authority
            .canonical_primary_semantics_authority_digest_prefix,
        final_primary_semantics_consumer_authority_digest_prefix: prefix_hex(
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
        canonical_continuity_authority_digest_prefix: prefix_hex(
            &continuity.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        continuity_status: if legacy_present {
            FinalContinuityStatusV2::LegacyPresent
        } else if blocking.is_empty() {
            FinalContinuityStatusV2::Pass
        } else {
            FinalContinuityStatusV2::Fail
        },
        blocking_codes: blocking
            .into_iter()
            .map(|code| match code {
                FinalContinuityMismatchCategoryV1::FinalContinuityGovernanceMismatch => {
                    "FINAL_CONTINUITY_GOVERNANCE_MISMATCH".to_string()
                }
                FinalContinuityMismatchCategoryV1::FinalContinuityScopeMismatch => {
                    "FINAL_CONTINUITY_SCOPE_MISMATCH".to_string()
                }
                FinalContinuityMismatchCategoryV1::FinalContinuityReadinessMismatch => {
                    "FINAL_CONTINUITY_READINESS_MISMATCH".to_string()
                }
                FinalContinuityMismatchCategoryV1::FinalContinuityPrimarySemanticsMismatch => {
                    "FINAL_CONTINUITY_PRIMARY_SEMANTICS_MISMATCH".to_string()
                }
                FinalContinuityMismatchCategoryV1::FinalContinuityWorkflowMismatch => {
                    "FINAL_CONTINUITY_WORKFLOW_MISMATCH".to_string()
                }
                FinalContinuityMismatchCategoryV1::FinalContinuityBundleMismatch => {
                    "FINAL_CONTINUITY_BUNDLE_MISMATCH".to_string()
                }
                FinalContinuityMismatchCategoryV1::LegacyContinuityProofPresent => {
                    "LEGACY_CONTINUITY_PROOF_PRESENT".to_string()
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

fn continuity_digest(report: &FinalContinuityAuthorityV2) -> Result<String, OpsError> {
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
    fn final_continuity_digest_stable() {
        let mut report = FinalContinuityAuthorityV2 {
            schema_version: 2,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            final_governance_consumer_authority_digest_prefix: "33".repeat(8),
            canonical_readiness_spine_digest_prefix: "44".repeat(8),
            final_readiness_consumer_authority_digest_prefix: "55".repeat(8),
            canonical_bundle_spine_digest_prefix: "66".repeat(8),
            final_bundle_consumer_authority_digest_prefix: "77".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "88".repeat(8),
            final_primary_semantics_consumer_authority_digest_prefix: "99".repeat(8),
            operator_review_packet_digest_prefix: "aa".repeat(8),
            operator_signoff_digest_prefix: "bb".repeat(8),
            operator_workflow_chain_digest_prefix: "cc".repeat(8),
            canonical_roundtrip_chain_digest_prefix: "dd".repeat(8),
            canonical_continuity_authority_digest_prefix: "ee".repeat(8),
            continuity_status: FinalContinuityStatusV2::Pass,
            blocking_codes: vec![],
            remediation_codes: vec!["x".to_string()],
            authority_digest: String::new(),
        };
        report.authority_digest = continuity_digest(&report).expect("digest");
        let a = report.authority_digest.clone();
        let b = continuity_digest(&report).expect("digest b");
        assert_eq!(a, b);
    }
}
