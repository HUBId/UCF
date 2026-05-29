use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, final_governance_consumer_sweep, governance_absolute_sweep,
    governance_convergence_sweep, governance_entry_sweep, governance_residual_sweep,
    governance_stabilization_sweep, governance_terminal_sweep, governance_ultimate_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    require_governance_final_consolidation_inputs, residual_free_governance_sweep,
    validate_governance_primary_surfaces_with_applied_scope, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceFinalConsolidationStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceFinalConsolidationMismatchCategoryV1 {
    ConsumerSkippedGovernanceFinalConsolidation,
    ConsumerUsedGovernanceFacadePath,
    GovernanceInputScopeMismatch,
    GovernanceInputEntryMismatch,
    GovernanceFacadePathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceFinalConsolidationConsumerStatusV1 {
    pub consumer: String,
    pub status: GovernanceFinalConsolidationStatusV1,
    pub mismatch_categories: Vec<GovernanceFinalConsolidationMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceFinalConsolidationSweepV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub final_governance_consumer_authority_digest_prefix: String,
    pub final_governance_residual_sweep_digest_prefix: String,
    pub residual_free_governance_consumer_authority_digest_prefix: String,
    pub residual_free_governance_absolute_sweep_digest_prefix: String,
    pub absolute_final_governance_terminal_sweep_digest_prefix: String,
    pub terminal_governance_ultimate_sweep_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    pub governance_stabilization_sweep_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub consolidation_status: GovernanceFinalConsolidationStatusV1,
    pub consolidation_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceFinalConsolidationSweepReportV1 {
    pub schema_version: u16,
    pub sweep: GovernanceFinalConsolidationSweepV1,
    pub consumers: Vec<GovernanceFinalConsolidationConsumerStatusV1>,
}

pub fn governance_final_consolidation_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<GovernanceFinalConsolidationSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_governance_final_consolidation_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = derive_canonical_governance_entry(&applied, &surfaces)?;
    let entry_sweep = governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_governance_final_consolidation_sweep.json"),
    )?;
    let final_governance = final_governance_consumer_sweep(
        workdir,
        &workdir
            .join("out/final_governance_consumer_sweep_governance_final_consolidation_sweep.json"),
    )?;
    let residual = governance_residual_sweep(
        workdir,
        &workdir.join("out/governance_residual_sweep_governance_final_consolidation_sweep.json"),
    )?;
    let residual_free = residual_free_governance_sweep(
        workdir,
        &workdir
            .join("out/residual_free_governance_sweep_governance_final_consolidation_sweep.json"),
    )?;
    let absolute = governance_absolute_sweep(
        workdir,
        &workdir.join("out/governance_absolute_sweep_governance_final_consolidation_sweep.json"),
    )?;
    let terminal = governance_terminal_sweep(
        workdir,
        &workdir.join("out/governance_terminal_sweep_governance_final_consolidation_sweep.json"),
    )?;
    let ultimate = governance_ultimate_sweep(
        workdir,
        &workdir.join("out/governance_ultimate_sweep_governance_final_consolidation_sweep.json"),
    )?;
    let convergence = governance_convergence_sweep(
        workdir,
        &workdir.join("out/governance_convergence_sweep_governance_final_consolidation_sweep.json"),
    )?;
    let stabilization = governance_stabilization_sweep(
        workdir,
        &workdir
            .join("out/governance_stabilization_sweep_governance_final_consolidation_sweep.json"),
    )?;

    let authority_ctx = require_governance_final_consolidation_inputs(
        Some(&applied),
        Some(&entry),
        Some(&entry_sweep.authority),
        Some(&final_governance.authority),
        Some(&residual.sweep),
        Some(&residual_free.authority),
        Some(&absolute.sweep),
        Some(&terminal.sweep),
        Some(&ultimate.sweep),
        Some(&convergence.sweep),
        Some(&stabilization.sweep),
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
            "ExportReadinessGuard",
            workdir,
            "out/operator_export_chain.json",
            &authority_ctx,
        )?,
        check_consumer(
            "BundleBuildVerifyOrchestration",
            workdir,
            "out/final_bundle_consumer_sweep.json",
            &authority_ctx,
        )?,
        check_consumer(
            "InteropConsistencyMatrix",
            workdir,
            "out/interop_consistency_matrix.json",
            &authority_ctx,
        )?,
        check_consumer(
            "V18PrepGateHelper",
            workdir,
            "out/v17_gate_report.json",
            &authority_ctx,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_path_count = consumers
        .iter()
        .filter(|consumer| !matches!(consumer.status, GovernanceFinalConsolidationStatusV1::Pass))
        .count() as u16;

    let consolidation_status = if consumers.iter().any(|c| {
        matches!(
            c.status,
            GovernanceFinalConsolidationStatusV1::LegacyPresent
        )
    }) {
        GovernanceFinalConsolidationStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        GovernanceFinalConsolidationStatusV1::Pass
    } else {
        GovernanceFinalConsolidationStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        consumers.len() as u16,
        residual_path_count,
        consolidation_status,
    );
    let report = GovernanceFinalConsolidationSweepReportV1 {
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
    authority_ctx: &crate::GovernanceFinalConsolidationInputsV1,
) -> Result<GovernanceFinalConsolidationConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories
            .insert(GovernanceFinalConsolidationMismatchCategoryV1::ConsumerSkippedGovernanceFinalConsolidation);
        return Ok(GovernanceFinalConsolidationConsumerStatusV1 {
            consumer: consumer.to_string(),
            status: GovernanceFinalConsolidationStatusV1::LegacyPresent,
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let field_match = |field: &str, expected: &str| {
        value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .map(|s| s == expected)
            .unwrap_or(false)
    };

    if !field_match(
        "applied_supported_set_digest_prefix",
        &authority_ctx.applied_supported_set_digest_prefix,
    ) {
        mismatch_categories
            .insert(GovernanceFinalConsolidationMismatchCategoryV1::GovernanceInputScopeMismatch);
    }
    if !field_match(
        "canonical_governance_entry_digest_prefix",
        &authority_ctx.canonical_governance_entry_digest_prefix,
    ) {
        mismatch_categories
            .insert(GovernanceFinalConsolidationMismatchCategoryV1::GovernanceInputEntryMismatch);
    }
    if !field_match(
        "governance_stabilization_sweep_digest_prefix",
        &authority_ctx.governance_stabilization_sweep_digest_prefix,
    ) {
        mismatch_categories.insert(
            GovernanceFinalConsolidationMismatchCategoryV1::ConsumerSkippedGovernanceFinalConsolidation,
        );
    }
    if !field_match(
        "final_governance_consumer_authority_digest_prefix",
        &authority_ctx.final_governance_consumer_authority_digest_prefix,
    ) {
        mismatch_categories.insert(
            GovernanceFinalConsolidationMismatchCategoryV1::ConsumerUsedGovernanceFacadePath,
        );
    }

    for forbidden in [
        "governance_facade_path",
        "governance_alias_path",
        "shadow_governance_view",
        "governance_shadow_view",
        "governance_notes",
        "export_governance_notes",
        "governance_history_entrypoint",
        "raw_evidence_governance_entrypoint",
    ] {
        if value.get(forbidden).is_some() {
            mismatch_categories.insert(
                GovernanceFinalConsolidationMismatchCategoryV1::GovernanceFacadePathPresent,
            );
        }
    }

    let status = if mismatch_categories.is_empty() {
        GovernanceFinalConsolidationStatusV1::Pass
    } else if mismatch_categories
        .contains(&GovernanceFinalConsolidationMismatchCategoryV1::GovernanceFacadePathPresent)
        || mismatch_categories.contains(
            &GovernanceFinalConsolidationMismatchCategoryV1::ConsumerUsedGovernanceFacadePath,
        )
    {
        GovernanceFinalConsolidationStatusV1::LegacyPresent
    } else {
        GovernanceFinalConsolidationStatusV1::Fail
    };

    Ok(GovernanceFinalConsolidationConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::GovernanceFinalConsolidationInputsV1,
    covered_consumer_count: u16,
    residual_path_count: u16,
    consolidation_status: GovernanceFinalConsolidationStatusV1,
) -> GovernanceFinalConsolidationSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"governance_final_consolidation_sweep_v1");
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
    bytes.extend_from_slice(
        ctx.terminal_governance_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(ctx.governance_convergence_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.governance_stabilization_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{consolidation_status:?}").as_bytes());

    GovernanceFinalConsolidationSweepV1 {
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
        terminal_governance_ultimate_sweep_digest_prefix: ctx
            .terminal_governance_ultimate_sweep_digest_prefix
            .clone(),
        governance_convergence_sweep_digest_prefix: ctx
            .governance_convergence_sweep_digest_prefix
            .clone(),
        governance_stabilization_sweep_digest_prefix: ctx
            .governance_stabilization_sweep_digest_prefix
            .clone(),
        covered_consumer_count,
        residual_path_count,
        consolidation_status,
        consolidation_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn governance_final_consolidation_sweep_digest_stable() {
        let ctx = crate::GovernanceFinalConsolidationInputsV1 {
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_governance_authority_digest_prefix: "33".repeat(8),
            final_governance_consumer_authority_digest_prefix: "44".repeat(8),
            final_governance_residual_sweep_digest_prefix: "55".repeat(8),
            residual_free_governance_consumer_authority_digest_prefix: "66".repeat(8),
            residual_free_governance_absolute_sweep_digest_prefix: "77".repeat(8),
            absolute_final_governance_terminal_sweep_digest_prefix: "88".repeat(8),
            terminal_governance_ultimate_sweep_digest_prefix: "99".repeat(8),
            governance_convergence_sweep_digest_prefix: "aa".repeat(8),
            governance_stabilization_sweep_digest_prefix: "bb".repeat(8),
            authority_digest: "cc".repeat(32),
        };
        let first = derive_sweep(&ctx, 8, 0, GovernanceFinalConsolidationStatusV1::Pass);
        let second = derive_sweep(&ctx, 8, 0, GovernanceFinalConsolidationStatusV1::Pass);
        assert_eq!(first.consolidation_digest, second.consolidation_digest);
    }

    #[test]
    fn governance_final_consolidation_status_deterministic_for_facade_paths() {
        let mismatches = BTreeSet::from([
            GovernanceFinalConsolidationMismatchCategoryV1::GovernanceFacadePathPresent,
        ]);
        let status = if mismatches.is_empty() {
            GovernanceFinalConsolidationStatusV1::Pass
        } else if mismatches
            .contains(&GovernanceFinalConsolidationMismatchCategoryV1::GovernanceFacadePathPresent)
            || mismatches.contains(
                &GovernanceFinalConsolidationMismatchCategoryV1::ConsumerUsedGovernanceFacadePath,
            )
        {
            GovernanceFinalConsolidationStatusV1::LegacyPresent
        } else {
            GovernanceFinalConsolidationStatusV1::Fail
        };
        assert_eq!(status, GovernanceFinalConsolidationStatusV1::LegacyPresent);
    }
}
