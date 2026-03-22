use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, final_governance_consumer_sweep, governance_entry_sweep,
    governance_residual_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, prefix_hex,
    require_residual_free_final_governance_inputs,
    validate_governance_primary_surfaces_with_applied_scope, OpsError,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreeGovernanceConsumerAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreeGovernanceMismatchCategoryV1 {
    ConsumerSkippedResidualFreeFinalGovernanceInputs,
    ConsumerUsedHistoricalGovernancePath,
    GovernanceInputScopeMismatch,
    GovernanceInputEntryMismatch,
    HistoricalGovernancePathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeGovernanceConsumerStatusV1 {
    pub consumer: String,
    pub status: ResidualFreeGovernanceConsumerAuthorityStatusV1,
    pub mismatch_categories: Vec<ResidualFreeGovernanceMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeGovernanceConsumerAuthorityV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,

    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub authority_status: ResidualFreeGovernanceConsumerAuthorityStatusV1,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeGovernanceSweepReportV1 {
    pub schema_version: u16,
    pub authority: ResidualFreeGovernanceConsumerAuthorityV1,
    pub consumers: Vec<ResidualFreeGovernanceConsumerStatusV1>,
}

pub fn residual_free_governance_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<ResidualFreeGovernanceSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_residual_free_governance_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = derive_canonical_governance_entry(&applied, &surfaces)?;
    let entry_sweep = governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_residual_free_governance_sweep.json"),
    )?;
    let final_governance = final_governance_consumer_sweep(
        workdir,
        &workdir.join("out/final_governance_consumer_sweep_residual_free_governance_sweep.json"),
    )?;
    let residual = governance_residual_sweep(
        workdir,
        &workdir.join("out/governance_residual_sweep_residual_free_governance_sweep.json"),
    )?;

    let authority_ctx = require_residual_free_final_governance_inputs(
        Some(&applied),
        Some(&entry),
        Some(&entry_sweep.authority),
        Some(&final_governance.authority),
        Some(&residual.sweep),
    )?;

    let mut consumers = vec![
        check_consumer(
            "ActiveReviewSnapshot",
            workdir,
            "out/active_review_snapshot.json",
            &authority_ctx,
            false,
        )?,
        check_consumer(
            "OperatorSignoff",
            workdir,
            "out/operator_signoff.json",
            &authority_ctx,
            false,
        )?,
        check_consumer(
            "OperatorReviewPacket",
            workdir,
            "out/operator_review_packet.json",
            &authority_ctx,
            false,
        )?,
        check_consumer(
            "OperatorWorkflowChain",
            workdir,
            "out/operator_workflow_chain.json",
            &authority_ctx,
            false,
        )?,
        check_consumer(
            "InteropConsistencyMatrix",
            workdir,
            "out/interop_consistency_matrix.json",
            &authority_ctx,
            false,
        )?,
        check_consumer(
            "V12PrepGateHelper",
            workdir,
            "out/v11_gate_report.json",
            &authority_ctx,
            true,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_path_count = consumers
        .iter()
        .filter(|consumer| {
            !matches!(
                consumer.status,
                ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass
            )
        })
        .count() as u16;

    let authority_status = if consumers.iter().any(|c| {
        matches!(
            c.status,
            ResidualFreeGovernanceConsumerAuthorityStatusV1::LegacyPresent
        )
    }) {
        ResidualFreeGovernanceConsumerAuthorityStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass
    } else {
        ResidualFreeGovernanceConsumerAuthorityStatusV1::Fail
    };

    let authority = derive_authority(
        &authority_ctx.applied_supported_set_digest_prefix,
        &authority_ctx.canonical_governance_entry_digest_prefix,
        &authority_ctx.canonical_governance_authority_digest_prefix,
        &authority_ctx.final_governance_consumer_authority_digest_prefix,
        &authority_ctx.final_governance_residual_sweep_digest_prefix,
        consumers.len() as u16,
        residual_path_count,
        authority_status,
    );

    let report = ResidualFreeGovernanceSweepReportV1 {
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
    workdir: &Path,
    rel_path: &str,
    authority_ctx: &crate::ResidualFreeFinalGovernanceInputsV1,
    allow_absent: bool,
) -> Result<ResidualFreeGovernanceConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(
            ResidualFreeGovernanceMismatchCategoryV1::ConsumerSkippedResidualFreeFinalGovernanceInputs,
        );
        let status = if allow_absent {
            ResidualFreeGovernanceConsumerAuthorityStatusV1::Fail
        } else {
            ResidualFreeGovernanceConsumerAuthorityStatusV1::LegacyPresent
        };
        return Ok(ResidualFreeGovernanceConsumerStatusV1 {
            consumer: consumer.to_string(),
            status,
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let expected_scope = &authority_ctx.applied_supported_set_digest_prefix;
    let expected_entry = &authority_ctx.canonical_governance_entry_digest_prefix;
    let expected_final = &authority_ctx.final_governance_consumer_authority_digest_prefix;
    let expected_residual = &authority_ctx.final_governance_residual_sweep_digest_prefix;
    let expected_authority = prefix_hex(&authority_ctx.authority_digest, DIGEST_PREFIX_LEN);

    let scope_match = value
        .get("applied_supported_set_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_scope)
        .unwrap_or(false);
    if !scope_match {
        mismatch_categories
            .insert(ResidualFreeGovernanceMismatchCategoryV1::GovernanceInputScopeMismatch);
    }

    let entry_match = value
        .get("canonical_governance_entry_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_entry)
        .unwrap_or(false);
    if !entry_match {
        mismatch_categories
            .insert(ResidualFreeGovernanceMismatchCategoryV1::GovernanceInputEntryMismatch);
    }

    let final_match = value
        .get("final_governance_consumer_authority_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_final)
        .unwrap_or(false);
    if !final_match {
        mismatch_categories.insert(
            ResidualFreeGovernanceMismatchCategoryV1::ConsumerSkippedResidualFreeFinalGovernanceInputs,
        );
    }

    let residual_match = value
        .get("governance_residual_sweep_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_residual)
        .unwrap_or(false);
    if !residual_match {
        mismatch_categories
            .insert(ResidualFreeGovernanceMismatchCategoryV1::HistoricalGovernancePathPresent);
    }

    let authority_match = value
        .get("residual_free_governance_authority_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_authority)
        .unwrap_or(false);
    if !authority_match {
        mismatch_categories
            .insert(ResidualFreeGovernanceMismatchCategoryV1::HistoricalGovernancePathPresent);
    }

    let status = if mismatch_categories.is_empty() {
        ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass
    } else if mismatch_categories
        .contains(&ResidualFreeGovernanceMismatchCategoryV1::HistoricalGovernancePathPresent)
    {
        ResidualFreeGovernanceConsumerAuthorityStatusV1::LegacyPresent
    } else {
        ResidualFreeGovernanceConsumerAuthorityStatusV1::Fail
    };

    Ok(ResidualFreeGovernanceConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

#[allow(clippy::too_many_arguments)]
fn derive_authority(
    applied_supported_set_digest_prefix: &str,
    canonical_governance_entry_digest_prefix: &str,
    canonical_governance_authority_digest_prefix: &str,
    final_governance_consumer_authority_digest_prefix: &str,
    final_governance_residual_sweep_digest_prefix: &str,
    covered_consumer_count: u16,
    residual_path_count: u16,
    authority_status: ResidualFreeGovernanceConsumerAuthorityStatusV1,
) -> ResidualFreeGovernanceConsumerAuthorityV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"residual_free_governance_consumer_authority_v1");
    bytes.extend_from_slice(applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_governance_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(final_governance_consumer_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(final_governance_residual_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", authority_status).as_bytes());

    ResidualFreeGovernanceConsumerAuthorityV1 {
        applied_supported_set_digest_prefix: applied_supported_set_digest_prefix.to_string(),
        canonical_governance_entry_digest_prefix: canonical_governance_entry_digest_prefix
            .to_string(),
        canonical_governance_authority_digest_prefix: canonical_governance_authority_digest_prefix
            .to_string(),
        final_governance_consumer_authority_digest_prefix:
            final_governance_consumer_authority_digest_prefix.to_string(),
        final_governance_residual_sweep_digest_prefix:
            final_governance_residual_sweep_digest_prefix.to_string(),
        covered_consumer_count,
        residual_path_count,
        authority_status,
        authority_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_free_governance_authority_digest_is_stable() {
        let first = derive_authority(
            "11".repeat(8).as_str(),
            "22".repeat(8).as_str(),
            "33".repeat(8).as_str(),
            "44".repeat(8).as_str(),
            "55".repeat(8).as_str(),
            6,
            0,
            ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
        );
        let second = derive_authority(
            "11".repeat(8).as_str(),
            "22".repeat(8).as_str(),
            "33".repeat(8).as_str(),
            "44".repeat(8).as_str(),
            "55".repeat(8).as_str(),
            6,
            0,
            ResidualFreeGovernanceConsumerAuthorityStatusV1::Pass,
        );
        assert_eq!(first.authority_digest, second.authority_digest);
    }
}
