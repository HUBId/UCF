use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, governance_entry_sweep, interop_consistency_matrix,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_signoff, operator_workflow_chain,
    require_final_governance_authority, validate_governance_primary_surfaces_with_applied_scope,
    OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalGovernanceConsumerAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalGovernanceConsumerMismatchCategoryV1 {
    ConsumerSkippedFinalGovernanceAuthority,
    ConsumerUsedLegacyGovernanceInput,
    FinalGovernanceScopeMismatch,
    FinalGovernanceEntryMismatch,
    LegacyGovernanceInputPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalGovernanceConsumerStatusV1 {
    pub consumer: String,
    pub status: FinalGovernanceConsumerAuthorityStatusV1,
    pub mismatch_categories: Vec<FinalGovernanceConsumerMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalGovernanceConsumerAuthorityV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub authority_status: FinalGovernanceConsumerAuthorityStatusV1,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalGovernanceConsumerSweepReportV1 {
    pub schema_version: u16,
    pub authority: FinalGovernanceConsumerAuthorityV1,
    pub consumers: Vec<FinalGovernanceConsumerStatusV1>,
}

pub fn final_governance_consumer_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<FinalGovernanceConsumerSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_final_governance_consumer_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = derive_canonical_governance_entry(&applied, &surfaces)?;
    let entry_sweep = governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_final_governance_consumer_sweep.json"),
    )?;
    let final_authority = require_final_governance_authority(
        Some(&applied),
        Some(&entry),
        Some(&entry_sweep.authority),
    )?;

    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_final_governance_consumer_sweep.json"),
    )?;
    let review = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_final_governance_consumer_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_final_governance_consumer_sweep.json"),
    )?;
    let interop = interop_consistency_matrix(
        workdir,
        &workdir.join("out/interop_consistency_final_governance_consumer_sweep.json"),
    )?;

    let expected_scope = final_authority.applied_supported_set_digest_prefix;
    let expected_authority = final_authority.canonical_governance_authority_digest_prefix;

    let consumers = vec![
        check_consumer(
            "ActiveReviewSnapshot",
            active
                .supported_slot_set_digest
                .starts_with(&expected_scope),
            true,
            false,
        ),
        check_consumer(
            "OperatorSignoff",
            signoff.applied_supported_set_digest_prefix == expected_scope
                && !signoff
                    .canonical_readiness_authority_digest_prefix
                    .is_empty(),
            true,
            false,
        ),
        check_consumer(
            "OperatorReviewPacket",
            review.applied_supported_set_digest_prefix == expected_scope
                && !review
                    .canonical_readiness_authority_digest_prefix
                    .is_empty(),
            true,
            false,
        ),
        check_consumer(
            "OperatorWorkflowChain",
            workflow.applied_supported_set_digest_prefix == expected_scope
                && !workflow
                    .canonical_readiness_authority_digest_prefix
                    .is_empty(),
            true,
            false,
        ),
        check_consumer(
            "InteropConsistencyMatrix",
            interop.matrix.applied_supported_set_digest_prefix == expected_scope
                && !interop.matrix.matrix_digest.is_empty(),
            true,
            false,
        ),
    ];

    let authority_status = if consumers.iter().any(|c| {
        matches!(
            c.status,
            FinalGovernanceConsumerAuthorityStatusV1::LegacyPresent
        )
    }) {
        FinalGovernanceConsumerAuthorityStatusV1::LegacyPresent
    } else if consumers
        .iter()
        .all(|c| matches!(c.status, FinalGovernanceConsumerAuthorityStatusV1::Pass))
    {
        FinalGovernanceConsumerAuthorityStatusV1::Pass
    } else {
        FinalGovernanceConsumerAuthorityStatusV1::Fail
    };

    let authority = derive_final_authority(
        &expected_scope,
        &entry_sweep
            .authority
            .canonical_governance_entry_digest_prefix,
        &expected_authority,
        consumers.len() as u16,
        authority_status,
    );

    let report = FinalGovernanceConsumerSweepReportV1 {
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
) -> FinalGovernanceConsumerStatusV1 {
    let mut mismatch_categories = BTreeSet::new();
    if !authority_match {
        mismatch_categories.insert(
            FinalGovernanceConsumerMismatchCategoryV1::ConsumerSkippedFinalGovernanceAuthority,
        );
        mismatch_categories
            .insert(FinalGovernanceConsumerMismatchCategoryV1::FinalGovernanceScopeMismatch);
    }
    if !has_final_reference {
        mismatch_categories
            .insert(FinalGovernanceConsumerMismatchCategoryV1::FinalGovernanceEntryMismatch);
    }
    if legacy_present {
        mismatch_categories
            .insert(FinalGovernanceConsumerMismatchCategoryV1::ConsumerUsedLegacyGovernanceInput);
        mismatch_categories
            .insert(FinalGovernanceConsumerMismatchCategoryV1::LegacyGovernanceInputPresent);
    }

    let status = if legacy_present {
        FinalGovernanceConsumerAuthorityStatusV1::LegacyPresent
    } else if mismatch_categories.is_empty() {
        FinalGovernanceConsumerAuthorityStatusV1::Pass
    } else {
        FinalGovernanceConsumerAuthorityStatusV1::Fail
    };

    FinalGovernanceConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    }
}

fn derive_final_authority(
    scope_prefix: &str,
    entry_prefix: &str,
    governance_authority_prefix: &str,
    covered_consumer_count: u16,
    authority_status: FinalGovernanceConsumerAuthorityStatusV1,
) -> FinalGovernanceConsumerAuthorityV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"final_governance_consumer_authority_v1");
    bytes.extend_from_slice(scope_prefix.as_bytes());
    bytes.extend_from_slice(entry_prefix.as_bytes());
    bytes.extend_from_slice(governance_authority_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", authority_status).as_bytes());

    FinalGovernanceConsumerAuthorityV1 {
        applied_supported_set_digest_prefix: scope_prefix.to_string(),
        canonical_governance_entry_digest_prefix: entry_prefix.to_string(),
        canonical_governance_authority_digest_prefix: governance_authority_prefix.to_string(),
        covered_consumer_count,
        authority_status,
        authority_digest: crate::sha256_hex(&bytes),
    }
}
