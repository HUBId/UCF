use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, governance_surfaces_check, interop_consistency_matrix,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_signoff, operator_workflow_chain, prefix_hex,
    validate_governance_primary_surfaces_with_applied_scope, OperatorReviewPacketArgs,
    OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceEntryCheckStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceEntryMismatchCategoryV1 {
    ConsumerSkippedCanonicalEntry,
    ConsumerUsedSecondaryEntry,
    GovernanceEntryScopeMismatch,
    GovernanceEntryPrimarySurfacesMismatch,
    LegacyEntryPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceEntryConsumerResultV1 {
    pub consumer: String,
    pub status: GovernanceEntryCheckStatusV1,
    pub mismatch_categories: Vec<GovernanceEntryMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceEntryCheckReportV1 {
    pub schema_version: u16,
    pub status: GovernanceEntryCheckStatusV1,
    pub authority_digest_prefix: String,
    pub consumers: Vec<GovernanceEntryConsumerResultV1>,
}

pub fn governance_entry_check(
    workdir: &Path,
    out: &Path,
) -> Result<GovernanceEntryCheckReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_governance_entry_check.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = derive_canonical_governance_entry(&applied, &surfaces)?;
    let expected_scope = entry.applied_supported_set_digest_prefix.clone();

    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_governance_entry_check.json"),
    )?;
    let packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_governance_entry_check.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_governance_entry_check.json"),
    )?;
    let interop = interop_consistency_matrix(
        workdir,
        &workdir.join("out/interop_consistency_matrix_governance_entry_check.json"),
    )?;
    let governance = governance_surfaces_check(
        workdir,
        &workdir.join("out/governance_surfaces_check_governance_entry_check.json"),
    )?;

    let consumers = vec![
        check(
            "ActiveReviewSnapshot",
            prefix_hex(&active.supported_slot_set_digest, 16) == expected_scope,
        ),
        check(
            "OperatorSignoff",
            signoff.applied_supported_set_digest_prefix == expected_scope,
        ),
        check(
            "OperatorReviewPacket",
            packet.applied_supported_set_digest_prefix == expected_scope,
        ),
        check(
            "OperatorWorkflowChain",
            workflow.applied_supported_set_digest_prefix == expected_scope,
        ),
        check(
            "InteropMatrix",
            interop.matrix.applied_supported_set_digest_prefix == expected_scope,
        ),
        check(
            "GovernanceSurfacesCheck",
            governance.governance_primary_surfaces.is_some(),
        ),
    ];

    let status = if consumers
        .iter()
        .all(|c| matches!(c.status, GovernanceEntryCheckStatusV1::Pass))
    {
        GovernanceEntryCheckStatusV1::Pass
    } else {
        GovernanceEntryCheckStatusV1::Fail
    };

    let report = GovernanceEntryCheckReportV1 {
        schema_version: 1,
        status,
        authority_digest_prefix: prefix_hex(&entry.authority_digest, 16),
        consumers,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn check(name: &str, pass: bool) -> GovernanceEntryConsumerResultV1 {
    GovernanceEntryConsumerResultV1 {
        consumer: name.to_string(),
        status: if pass {
            GovernanceEntryCheckStatusV1::Pass
        } else {
            GovernanceEntryCheckStatusV1::Fail
        },
        mismatch_categories: if pass {
            vec![]
        } else {
            vec![GovernanceEntryMismatchCategoryV1::ConsumerSkippedCanonicalEntry]
        },
    }
}
