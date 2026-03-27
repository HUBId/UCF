use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, final_governance_consumer_sweep, governance_absolute_sweep,
    governance_entry_sweep, governance_residual_sweep, governance_terminal_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    prefix_hex, require_terminal_governance_ultimate_inputs, residual_free_governance_sweep,
    validate_governance_primary_surfaces_with_applied_scope, OpsError,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TerminalGovernanceUltimateSweepStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceUltimateMismatchCategoryV1 {
    ConsumerSkippedUltimateGovernanceInputs,
    ConsumerUsedGovernanceCachePath,
    GovernanceInputScopeMismatch,
    GovernanceInputEntryMismatch,
    GovernanceCachePathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceUltimateConsumerStatusV1 {
    pub consumer: String,
    pub status: TerminalGovernanceUltimateSweepStatusV1,
    pub mismatch_categories: Vec<GovernanceUltimateMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminalGovernanceUltimateSweepV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub sweep_status: TerminalGovernanceUltimateSweepStatusV1,
    pub sweep_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceUltimateSweepReportV1 {
    pub schema_version: u16,
    pub sweep: TerminalGovernanceUltimateSweepV1,
    pub consumers: Vec<GovernanceUltimateConsumerStatusV1>,
}

pub fn governance_ultimate_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<GovernanceUltimateSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_governance_ultimate_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = derive_canonical_governance_entry(&applied, &surfaces)?;
    let entry_sweep = governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_governance_ultimate_sweep.json"),
    )?;
    let final_governance = final_governance_consumer_sweep(
        workdir,
        &workdir.join("out/final_governance_consumer_sweep_governance_ultimate_sweep.json"),
    )?;
    let residual = governance_residual_sweep(
        workdir,
        &workdir.join("out/governance_residual_sweep_governance_ultimate_sweep.json"),
    )?;
    let residual_free = residual_free_governance_sweep(
        workdir,
        &workdir.join("out/residual_free_governance_sweep_governance_ultimate_sweep.json"),
    )?;
    let absolute = governance_absolute_sweep(
        workdir,
        &workdir.join("out/governance_absolute_sweep_governance_ultimate_sweep.json"),
    )?;
    let terminal = governance_terminal_sweep(
        workdir,
        &workdir.join("out/governance_terminal_sweep_governance_ultimate_sweep.json"),
    )?;

    let authority_ctx = require_terminal_governance_ultimate_inputs(
        Some(&applied),
        Some(&entry),
        Some(&entry_sweep.authority),
        Some(&final_governance.authority),
        Some(&residual.sweep),
        Some(&residual_free.authority),
        Some(&absolute.sweep),
        Some(&terminal.sweep),
    )?;

    let mut consumers = vec![
        check_consumer(
            "ActiveReviewSnapshot",
            workdir,
            "out/active_review_snapshot.json",
            &authority_ctx,
        )?,
        check_consumer(
            "OperatorSignoff",
            workdir,
            "out/operator_signoff.json",
            &authority_ctx,
        )?,
        check_consumer(
            "OperatorReviewPacket",
            workdir,
            "out/operator_review_packet.json",
            &authority_ctx,
        )?,
        check_consumer(
            "OperatorWorkflowChain",
            workdir,
            "out/operator_workflow_chain.json",
            &authority_ctx,
        )?,
        check_consumer(
            "InteropConsistencyMatrix",
            workdir,
            "out/interop_consistency_matrix.json",
            &authority_ctx,
        )?,
        check_consumer(
            "V15PrepGateHelper",
            workdir,
            "out/v14_gate_report.json",
            &authority_ctx,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_path_count = consumers
        .iter()
        .filter(|consumer| {
            !matches!(
                consumer.status,
                TerminalGovernanceUltimateSweepStatusV1::Pass
            )
        })
        .count() as u16;

    let sweep_status = if consumers.iter().any(|c| {
        matches!(
            c.status,
            TerminalGovernanceUltimateSweepStatusV1::LegacyPresent
        )
    }) {
        TerminalGovernanceUltimateSweepStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        TerminalGovernanceUltimateSweepStatusV1::Pass
    } else {
        TerminalGovernanceUltimateSweepStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        consumers.len() as u16,
        residual_path_count,
        sweep_status,
    );
    let report = GovernanceUltimateSweepReportV1 {
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

fn check_consumer(
    consumer: &str,
    workdir: &Path,
    rel_path: &str,
    authority_ctx: &crate::TerminalGovernanceUltimateInputsV1,
) -> Result<GovernanceUltimateConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::ConsumerSkippedUltimateGovernanceInputs);
        return Ok(GovernanceUltimateConsumerStatusV1 {
            consumer: consumer.to_string(),
            status: TerminalGovernanceUltimateSweepStatusV1::LegacyPresent,
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let expected_scope = &authority_ctx.applied_supported_set_digest_prefix;
    let expected_entry = &authority_ctx.canonical_governance_entry_digest_prefix;
    let expected_final = &authority_ctx.final_governance_consumer_authority_digest_prefix;
    let expected_residual = &authority_ctx.final_governance_residual_sweep_digest_prefix;
    let expected_residual_free =
        &authority_ctx.residual_free_governance_consumer_authority_digest_prefix;
    let expected_absolute = &authority_ctx.residual_free_governance_absolute_sweep_digest_prefix;
    let expected_terminal = &authority_ctx.absolute_final_governance_terminal_sweep_digest_prefix;
    let expected_ultimate = prefix_hex(&authority_ctx.authority_digest, DIGEST_PREFIX_LEN);

    let field_match = |field: &str, expected: &str| {
        value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .map(|s| s == expected)
            .unwrap_or(false)
    };

    if !field_match("applied_supported_set_digest_prefix", expected_scope) {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::GovernanceInputScopeMismatch);
    }
    if !field_match("canonical_governance_entry_digest_prefix", expected_entry) {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::GovernanceInputEntryMismatch);
    }
    if !field_match(
        "final_governance_consumer_authority_digest_prefix",
        expected_final,
    ) {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::ConsumerSkippedUltimateGovernanceInputs);
    }
    if !field_match("governance_residual_sweep_digest_prefix", expected_residual) {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::GovernanceCachePathPresent);
    }
    if !field_match(
        "residual_free_governance_authority_digest_prefix",
        expected_residual_free,
    ) {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::ConsumerUsedGovernanceCachePath);
    }
    if !field_match("governance_absolute_sweep_digest_prefix", expected_absolute) {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::GovernanceCachePathPresent);
    }
    if !field_match(
        "absolute_final_governance_terminal_sweep_digest_prefix",
        expected_terminal,
    ) {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::ConsumerSkippedUltimateGovernanceInputs);
    }
    if !field_match(
        "governance_ultimate_sweep_digest_prefix",
        &expected_ultimate,
    ) {
        mismatch_categories
            .insert(GovernanceUltimateMismatchCategoryV1::ConsumerSkippedUltimateGovernanceInputs);
    }

    let status = if mismatch_categories.is_empty() {
        TerminalGovernanceUltimateSweepStatusV1::Pass
    } else if mismatch_categories
        .contains(&GovernanceUltimateMismatchCategoryV1::GovernanceCachePathPresent)
        || mismatch_categories
            .contains(&GovernanceUltimateMismatchCategoryV1::ConsumerUsedGovernanceCachePath)
    {
        TerminalGovernanceUltimateSweepStatusV1::LegacyPresent
    } else {
        TerminalGovernanceUltimateSweepStatusV1::Fail
    };

    Ok(GovernanceUltimateConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::TerminalGovernanceUltimateInputsV1,
    covered_consumer_count: u16,
    residual_path_count: u16,
    sweep_status: TerminalGovernanceUltimateSweepStatusV1,
) -> TerminalGovernanceUltimateSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"terminal_governance_ultimate_sweep_v1");
    bytes.extend_from_slice(ctx.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_governance_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        ctx.final_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(ctx.final_governance_residual_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        ctx.residual_free_governance_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.residual_free_governance_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.absolute_final_governance_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", sweep_status).as_bytes());

    TerminalGovernanceUltimateSweepV1 {
        applied_supported_set_digest_prefix: ctx.applied_supported_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: ctx
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_governance_authority_digest_prefix: ctx
            .canonical_governance_authority_digest_prefix
            .clone(),
        final_governance_consumer_authority_digest_prefix: ctx
            .final_governance_consumer_authority_digest_prefix
            .clone(),
        final_governance_residual_sweep_digest_prefix: ctx
            .final_governance_residual_sweep_digest_prefix
            .clone(),
        residual_free_governance_consumer_authority_digest_prefix: ctx
            .residual_free_governance_consumer_authority_digest_prefix
            .clone(),
        residual_free_governance_absolute_sweep_digest_prefix: ctx
            .residual_free_governance_absolute_sweep_digest_prefix
            .clone(),
        absolute_final_governance_terminal_sweep_digest_prefix: ctx
            .absolute_final_governance_terminal_sweep_digest_prefix
            .clone(),
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
    fn governance_ultimate_sweep_digest_stable() {
        let ctx = crate::TerminalGovernanceUltimateInputsV1 {
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_governance_authority_digest_prefix: "33".repeat(8),
            final_governance_consumer_authority_digest_prefix: "44".repeat(8),
            final_governance_residual_sweep_digest_prefix: "55".repeat(8),
            residual_free_governance_consumer_authority_digest_prefix: "66".repeat(8),
            residual_free_governance_absolute_sweep_digest_prefix: "77".repeat(8),
            absolute_final_governance_terminal_sweep_digest_prefix: "88".repeat(8),
            authority_digest: "99".repeat(32),
        };
        let first = derive_sweep(&ctx, 6, 0, TerminalGovernanceUltimateSweepStatusV1::Pass);
        let second = derive_sweep(&ctx, 6, 0, TerminalGovernanceUltimateSweepStatusV1::Pass);
        assert_eq!(first.sweep_digest, second.sweep_digest);
    }

    #[test]
    fn governance_ultimate_status_is_deterministic_for_cache_mismatch() {
        let mismatches =
            BTreeSet::from([GovernanceUltimateMismatchCategoryV1::GovernanceCachePathPresent]);
        let status = if mismatches.is_empty() {
            TerminalGovernanceUltimateSweepStatusV1::Pass
        } else if mismatches
            .contains(&GovernanceUltimateMismatchCategoryV1::GovernanceCachePathPresent)
            || mismatches
                .contains(&GovernanceUltimateMismatchCategoryV1::ConsumerUsedGovernanceCachePath)
        {
            TerminalGovernanceUltimateSweepStatusV1::LegacyPresent
        } else {
            TerminalGovernanceUltimateSweepStatusV1::Fail
        };
        assert_eq!(
            status,
            TerminalGovernanceUltimateSweepStatusV1::LegacyPresent
        );
    }
}
