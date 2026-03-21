use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, final_governance_consumer_sweep, governance_entry_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    prefix_hex, require_final_governance_inputs,
    validate_governance_primary_surfaces_with_applied_scope, FinalGovernanceConsumerAuthorityV1,
    OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceResidualSweepStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceResidualMismatchCategoryV1 {
    ConsumerSkippedFinalGovernanceInputs,
    ConsumerUsedResidualGovernancePath,
    GovernanceInputScopeMismatch,
    GovernanceInputEntryMismatch,
    ResidualGovernancePathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceResidualConsumerStatusV1 {
    pub consumer: String,
    pub status: GovernanceResidualSweepStatusV1,
    pub mismatch_categories: Vec<GovernanceResidualMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalGovernanceResidualSweepV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub sweep_status: GovernanceResidualSweepStatusV1,
    pub sweep_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceResidualSweepReportV1 {
    pub schema_version: u16,
    pub sweep: FinalGovernanceResidualSweepV1,
    pub consumers: Vec<GovernanceResidualConsumerStatusV1>,
}

pub fn governance_residual_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<GovernanceResidualSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_governance_residual_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = derive_canonical_governance_entry(&applied, &surfaces)?;
    let entry_sweep = governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_governance_residual_sweep.json"),
    )?;
    let final_governance = final_governance_consumer_sweep(
        workdir,
        &workdir.join("out/final_governance_consumer_sweep_governance_residual_sweep.json"),
    )?;
    let final_consumer = final_governance.authority;
    let final_inputs = require_final_governance_inputs(
        Some(&applied),
        Some(&entry),
        Some(&entry_sweep.authority),
        Some(&final_consumer),
    );
    if let Err(err) = final_inputs {
        let sweep = derive_residual_sweep(
            &applied.applied_set_digest_prefix,
            &prefix_hex(&entry.authority_digest, 16),
            &prefix_hex(&entry_sweep.authority.authority_digest, 16),
            &prefix_hex(&final_consumer.authority_digest, 16),
            0,
            1,
            GovernanceResidualSweepStatusV1::Fail,
        );
        let report = GovernanceResidualSweepReportV1 {
            schema_version: 1,
            sweep,
            consumers: vec![GovernanceResidualConsumerStatusV1 {
                consumer: "FinalGovernanceInputs".to_string(),
                status: GovernanceResidualSweepStatusV1::Fail,
                mismatch_categories: vec![
                    GovernanceResidualMismatchCategoryV1::ConsumerSkippedFinalGovernanceInputs,
                    GovernanceResidualMismatchCategoryV1::ResidualGovernancePathPresent,
                ],
            }],
        };
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out, serde_json::to_vec_pretty(&report)?)?;
        return Err(OpsError::Invalid(err.to_string()));
    }
    let final_inputs = final_inputs?;

    let mut consumers = vec![
        check_metadata_consumer(
            "ActiveReviewSnapshot",
            workdir,
            "out/active_review_snapshot.json",
            &final_consumer,
            false,
        )?,
        check_metadata_consumer(
            "OperatorSignoff",
            workdir,
            "out/operator_signoff.json",
            &final_consumer,
            false,
        )?,
        check_metadata_consumer(
            "OperatorReviewPacket",
            workdir,
            "out/operator_review_packet.json",
            &final_consumer,
            false,
        )?,
        check_metadata_consumer(
            "OperatorWorkflowChain",
            workdir,
            "out/operator_workflow_chain.json",
            &final_consumer,
            false,
        )?,
        check_metadata_consumer(
            "InteropConsistencyMatrix",
            workdir,
            "out/interop_consistency_matrix.json",
            &final_consumer,
            false,
        )?,
        check_metadata_consumer(
            "V11PrepGateHelper",
            workdir,
            "out/v10_gate_report.json",
            &final_consumer,
            true,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_path_count = consumers
        .iter()
        .filter(|consumer| {
            !matches!(consumer.status, GovernanceResidualSweepStatusV1::Pass)
                || consumer
                    .mismatch_categories
                    .contains(&GovernanceResidualMismatchCategoryV1::ResidualGovernancePathPresent)
        })
        .count() as u16;

    let sweep_status = if consumers
        .iter()
        .any(|c| matches!(c.status, GovernanceResidualSweepStatusV1::LegacyPresent))
    {
        GovernanceResidualSweepStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        GovernanceResidualSweepStatusV1::Pass
    } else {
        GovernanceResidualSweepStatusV1::Fail
    };

    let sweep = derive_residual_sweep(
        &final_inputs.applied_supported_set_digest_prefix,
        &final_inputs.canonical_governance_entry_digest_prefix,
        &final_inputs.canonical_governance_authority_digest_prefix,
        &final_inputs.final_governance_consumer_authority_digest_prefix,
        consumers.len() as u16,
        residual_path_count,
        sweep_status,
    );
    let report = GovernanceResidualSweepReportV1 {
        schema_version: 1,
        sweep,
        consumers,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn check_metadata_consumer(
    consumer_name: &str,
    workdir: &Path,
    rel_path: &str,
    final_consumer: &FinalGovernanceConsumerAuthorityV1,
    allow_absent: bool,
) -> Result<GovernanceResidualConsumerStatusV1, OpsError> {
    let mut categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        categories
            .insert(GovernanceResidualMismatchCategoryV1::ConsumerSkippedFinalGovernanceInputs);
        let status = if allow_absent {
            GovernanceResidualSweepStatusV1::Fail
        } else {
            GovernanceResidualSweepStatusV1::LegacyPresent
        };
        return Ok(GovernanceResidualConsumerStatusV1 {
            consumer: consumer_name.to_string(),
            status,
            mismatch_categories: categories.into_iter().collect(),
        });
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let expected_scope = &final_consumer.applied_supported_set_digest_prefix;
    let expected_entry = &final_consumer.canonical_governance_entry_digest_prefix;
    let expected_authority = prefix_hex(&final_consumer.authority_digest, 16);
    let scope_match = value
        .get("applied_supported_set_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|scope| scope == expected_scope)
        .unwrap_or(false);
    if !scope_match {
        categories.insert(GovernanceResidualMismatchCategoryV1::GovernanceInputScopeMismatch);
    }
    let final_authority_match = value
        .get("final_governance_consumer_authority_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|prefix| prefix == expected_authority)
        .unwrap_or(false);
    if !final_authority_match {
        categories
            .insert(GovernanceResidualMismatchCategoryV1::ConsumerSkippedFinalGovernanceInputs);
        categories.insert(GovernanceResidualMismatchCategoryV1::GovernanceInputEntryMismatch);
    }
    let final_entry_match = value
        .get("canonical_governance_entry_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|prefix| prefix == expected_entry)
        .unwrap_or(true);
    if !final_entry_match {
        categories.insert(GovernanceResidualMismatchCategoryV1::GovernanceInputEntryMismatch);
    }
    let residual_code_present = value
        .get("blocking_codes")
        .and_then(serde_json::Value::as_array)
        .map(|codes| {
            codes.iter().any(|code| {
                code.as_str()
                    .map(|code| code.contains("RESIDUAL_GOVERNANCE_PATH"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if residual_code_present {
        categories.insert(GovernanceResidualMismatchCategoryV1::ResidualGovernancePathPresent);
        categories.insert(GovernanceResidualMismatchCategoryV1::ConsumerUsedResidualGovernancePath);
    }

    let status = if categories.is_empty() {
        GovernanceResidualSweepStatusV1::Pass
    } else if categories
        .contains(&GovernanceResidualMismatchCategoryV1::ResidualGovernancePathPresent)
    {
        GovernanceResidualSweepStatusV1::LegacyPresent
    } else {
        GovernanceResidualSweepStatusV1::Fail
    };
    Ok(GovernanceResidualConsumerStatusV1 {
        consumer: consumer_name.to_string(),
        status,
        mismatch_categories: categories.into_iter().collect(),
    })
}

fn derive_residual_sweep(
    applied_scope_prefix: &str,
    entry_prefix: &str,
    authority_prefix: &str,
    final_consumer_prefix: &str,
    covered_consumer_count: u16,
    residual_path_count: u16,
    sweep_status: GovernanceResidualSweepStatusV1,
) -> FinalGovernanceResidualSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"final_governance_residual_sweep_v1");
    bytes.extend_from_slice(applied_scope_prefix.as_bytes());
    bytes.extend_from_slice(entry_prefix.as_bytes());
    bytes.extend_from_slice(authority_prefix.as_bytes());
    bytes.extend_from_slice(final_consumer_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", sweep_status).as_bytes());
    FinalGovernanceResidualSweepV1 {
        applied_supported_set_digest_prefix: applied_scope_prefix.to_string(),
        canonical_governance_entry_digest_prefix: entry_prefix.to_string(),
        canonical_governance_authority_digest_prefix: authority_prefix.to_string(),
        final_governance_consumer_authority_digest_prefix: final_consumer_prefix.to_string(),
        covered_consumer_count,
        residual_path_count,
        sweep_status,
        sweep_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_sweep_digest_stable() {
        let first = derive_residual_sweep(
            "scope123456789012",
            "entry123456789012",
            "auth123456789012",
            "final123456789012",
            5,
            0,
            GovernanceResidualSweepStatusV1::Pass,
        );
        let second = derive_residual_sweep(
            "scope123456789012",
            "entry123456789012",
            "auth123456789012",
            "final123456789012",
            5,
            0,
            GovernanceResidualSweepStatusV1::Pass,
        );
        assert_eq!(first.sweep_digest, second.sweep_digest);
    }
}
