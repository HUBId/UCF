use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, derive_canonical_readiness_authority_v2,
    derive_canonical_readiness_spine, derive_slot_reviewability_truths, interop_consistency_matrix,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_signoff, operator_workflow_chain, reduce_reviewability,
    require_canonical_governance_entry, require_final_readiness_authority, resolve_strict_evidence,
    validate_governance_primary_surfaces_with_applied_scope, CanonicalReadinessAuthorityStatusV2,
    OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
    StrictEvidenceContextV1,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalReadinessConsumerAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalReadinessConsumerMismatchCategoryV1 {
    ConsumerSkippedFinalReadinessAuthority,
    ConsumerUsedLegacyReadinessInput,
    FinalReadinessScopeMismatch,
    FinalReadinessSpineMismatch,
    LegacyReadinessInputPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalReadinessConsumerStatusV1 {
    pub consumer: String,
    pub status: FinalReadinessConsumerAuthorityStatusV1,
    pub mismatch_categories: Vec<FinalReadinessConsumerMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalReadinessConsumerAuthorityV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub authority_status: FinalReadinessConsumerAuthorityStatusV1,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalReadinessConsumerSweepReportV1 {
    pub schema_version: u16,
    pub authority: FinalReadinessConsumerAuthorityV1,
    pub consumers: Vec<FinalReadinessConsumerStatusV1>,
}

pub fn final_readiness_consumer_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<FinalReadinessConsumerSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_final_readiness_consumer_sweep.json"),
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
    let truths = derive_slot_reviewability_truths(&applied, &backend, &active, &strict)?;
    let reduction = reduce_reviewability(&applied, &truths)?;
    let governance =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = require_canonical_governance_entry(
        &applied,
        Some(&derive_canonical_governance_entry(&applied, &governance)?),
    )?;
    let spine = derive_canonical_readiness_spine(
        &applied,
        &entry,
        &truths,
        &reduction,
        Some(&active.snapshot_digest),
        None,
        None,
        None,
    )?;
    let readiness_authority = derive_canonical_readiness_authority_v2(
        &applied.applied_set_digest_prefix,
        &crate::prefix_hex(&entry.authority_digest, 16),
        &crate::prefix_hex(&spine.spine_digest, 16),
        4,
        CanonicalReadinessAuthorityStatusV2::Pass,
    );
    let final_context = require_final_readiness_authority(
        &applied,
        &entry,
        Some(&spine),
        Some(&readiness_authority),
    )?;

    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_final_readiness_consumer_sweep.json"),
    )?;
    let review = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_final_readiness_consumer_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_final_readiness_consumer_sweep.json"),
    )?;
    let interop = interop_consistency_matrix(
        workdir,
        &workdir.join("out/interop_consistency_final_readiness_consumer_sweep.json"),
    )?;

    let consumers = vec![
        check_consumer(
            "ActiveReviewSnapshot",
            active
                .supported_slot_set_digest
                .starts_with(&final_context.applied_supported_set_digest_prefix),
            true,
            false,
        ),
        check_consumer(
            "OperatorSignoff",
            signoff.applied_supported_set_digest_prefix
                == final_context.applied_supported_set_digest_prefix,
            signoff.canonical_readiness_authority_digest_prefix != "MISSING",
            signoff.reviewability_reduction_digest_prefix == "MISSING",
        ),
        check_consumer(
            "OperatorReviewPacket",
            review.applied_supported_set_digest_prefix
                == final_context.applied_supported_set_digest_prefix,
            review.canonical_readiness_authority_digest_prefix != "MISSING",
            review.reviewability_reduction_digest_prefix == "MISSING",
        ),
        check_consumer(
            "OperatorWorkflowChain",
            workflow.applied_supported_set_digest_prefix
                == final_context.applied_supported_set_digest_prefix,
            workflow.canonical_readiness_authority_digest_prefix != "MISSING",
            workflow.reviewability_reduction_digest_prefix == "MISSING",
        ),
        check_consumer(
            "InteropConsistencyMatrix",
            interop.matrix.applied_supported_set_digest_prefix
                == final_context.applied_supported_set_digest_prefix,
            true,
            false,
        ),
    ];

    let authority_status = if consumers.iter().any(|c| {
        matches!(
            c.status,
            FinalReadinessConsumerAuthorityStatusV1::LegacyPresent
        )
    }) {
        FinalReadinessConsumerAuthorityStatusV1::LegacyPresent
    } else if consumers
        .iter()
        .all(|c| matches!(c.status, FinalReadinessConsumerAuthorityStatusV1::Pass))
    {
        FinalReadinessConsumerAuthorityStatusV1::Pass
    } else {
        FinalReadinessConsumerAuthorityStatusV1::Fail
    };

    let authority = derive_final_readiness_authority(
        &final_context.applied_supported_set_digest_prefix,
        &final_context.canonical_governance_entry_digest_prefix,
        &final_context.canonical_readiness_spine_digest_prefix,
        &final_context.canonical_readiness_authority_digest_prefix,
        consumers.len() as u16,
        authority_status,
    );
    let report = FinalReadinessConsumerSweepReportV1 {
        schema_version: 1,
        authority,
        consumers,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn check_consumer(
    consumer: &str,
    authority_match: bool,
    has_final_reference: bool,
    legacy_present: bool,
) -> FinalReadinessConsumerStatusV1 {
    let mut mismatch_categories = BTreeSet::new();
    if !authority_match {
        mismatch_categories.insert(
            FinalReadinessConsumerMismatchCategoryV1::ConsumerSkippedFinalReadinessAuthority,
        );
        mismatch_categories
            .insert(FinalReadinessConsumerMismatchCategoryV1::FinalReadinessScopeMismatch);
        mismatch_categories
            .insert(FinalReadinessConsumerMismatchCategoryV1::FinalReadinessSpineMismatch);
    }
    if !has_final_reference {
        mismatch_categories
            .insert(FinalReadinessConsumerMismatchCategoryV1::FinalReadinessSpineMismatch);
    }
    if legacy_present {
        mismatch_categories
            .insert(FinalReadinessConsumerMismatchCategoryV1::ConsumerUsedLegacyReadinessInput);
        mismatch_categories
            .insert(FinalReadinessConsumerMismatchCategoryV1::LegacyReadinessInputPresent);
    }
    let status = if legacy_present {
        FinalReadinessConsumerAuthorityStatusV1::LegacyPresent
    } else if mismatch_categories.is_empty() {
        FinalReadinessConsumerAuthorityStatusV1::Pass
    } else {
        FinalReadinessConsumerAuthorityStatusV1::Fail
    };
    FinalReadinessConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    }
}

fn derive_final_readiness_authority(
    scope_prefix: &str,
    entry_prefix: &str,
    spine_prefix: &str,
    readiness_authority_prefix: &str,
    covered_consumer_count: u16,
    authority_status: FinalReadinessConsumerAuthorityStatusV1,
) -> FinalReadinessConsumerAuthorityV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"final_readiness_consumer_authority_v1");
    bytes.extend_from_slice(scope_prefix.as_bytes());
    bytes.extend_from_slice(entry_prefix.as_bytes());
    bytes.extend_from_slice(spine_prefix.as_bytes());
    bytes.extend_from_slice(readiness_authority_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", authority_status).as_bytes());
    FinalReadinessConsumerAuthorityV1 {
        applied_supported_set_digest_prefix: scope_prefix.to_string(),
        canonical_governance_entry_digest_prefix: entry_prefix.to_string(),
        canonical_readiness_spine_digest_prefix: spine_prefix.to_string(),
        canonical_readiness_authority_digest_prefix: readiness_authority_prefix.to_string(),
        covered_consumer_count,
        authority_status,
        authority_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_readiness_authority_digest_is_stable() {
        let a = derive_final_readiness_authority(
            "scope123456789012",
            "entry123456789012",
            "spine123456789012",
            "authority12345678",
            5,
            FinalReadinessConsumerAuthorityStatusV1::Pass,
        );
        let b = derive_final_readiness_authority(
            "scope123456789012",
            "entry123456789012",
            "spine123456789012",
            "authority12345678",
            5,
            FinalReadinessConsumerAuthorityStatusV1::Pass,
        );
        assert_eq!(a.authority_digest, b.authority_digest);
    }

    #[test]
    fn consumer_status_reports_legacy_inputs() {
        let status = check_consumer("OperatorSignoff", true, true, true);
        assert!(matches!(
            status.status,
            FinalReadinessConsumerAuthorityStatusV1::LegacyPresent
        ));
        assert!(status
            .mismatch_categories
            .contains(&FinalReadinessConsumerMismatchCategoryV1::LegacyReadinessInputPresent));
    }
}
