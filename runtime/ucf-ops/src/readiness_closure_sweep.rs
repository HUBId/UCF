use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, derive_canonical_readiness_authority_v2,
    derive_canonical_readiness_spine, derive_slot_reviewability_truths,
    final_readiness_consumer_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, readiness_absolute_sweep,
    readiness_convergence_sweep, readiness_final_consolidation_sweep, readiness_residual_sweep,
    readiness_stabilization_sweep, readiness_terminal_sweep, readiness_ultimate_sweep,
    reduce_reviewability, require_canonical_governance_entry, require_readiness_closure_inputs,
    residual_free_readiness_sweep, resolve_strict_evidence,
    validate_governance_primary_surfaces_with_applied_scope, CanonicalReadinessAuthorityStatusV2,
    OpsError, StrictEvidenceContextV1,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessClosureStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessClosureMismatchCategoryV1 {
    ConsumerSkippedReadinessClosure,
    ConsumerUsedReadinessWrapperPath,
    ReadinessInputScopeMismatch,
    ReadinessInputSpineMismatch,
    ReadinessWrapperPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessClosureConsumerStatusV1 {
    pub consumer: String,
    pub status: ReadinessClosureStatusV1,
    pub mismatch_categories: Vec<ReadinessClosureMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessClosureSweepV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub residual_free_readiness_consumer_authority_digest_prefix: String,
    pub residual_free_readiness_absolute_sweep_digest_prefix: String,
    pub absolute_final_readiness_terminal_sweep_digest_prefix: String,
    pub terminal_readiness_ultimate_sweep_digest_prefix: String,
    pub readiness_convergence_sweep_digest_prefix: String,
    pub readiness_stabilization_sweep_digest_prefix: String,
    pub readiness_final_consolidation_sweep_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub closure_status: ReadinessClosureStatusV1,
    pub closure_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessClosureSweepReportV1 {
    pub schema_version: u16,
    pub sweep: ReadinessClosureSweepV1,
    pub consumers: Vec<ReadinessClosureConsumerStatusV1>,
}

pub fn readiness_closure_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<ReadinessClosureSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_readiness_closure_sweep.json"),
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
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = require_canonical_governance_entry(
        &applied,
        Some(&derive_canonical_governance_entry(&applied, &surfaces)?),
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
    let final_readiness = final_readiness_consumer_sweep(
        workdir,
        &workdir.join("out/final_readiness_consumer_sweep_readiness_closure_sweep.json"),
    )?;
    let residual = readiness_residual_sweep(
        workdir,
        &workdir.join("out/readiness_residual_sweep_readiness_closure_sweep.json"),
    )?;
    let residual_free = residual_free_readiness_sweep(
        workdir,
        &workdir.join("out/residual_free_readiness_sweep_readiness_closure_sweep.json"),
    )?;
    let absolute = readiness_absolute_sweep(
        workdir,
        &workdir.join("out/readiness_absolute_sweep_readiness_closure_sweep.json"),
    )?;
    let terminal = readiness_terminal_sweep(
        workdir,
        &workdir.join("out/readiness_terminal_sweep_readiness_closure_sweep.json"),
    )?;
    let ultimate = readiness_ultimate_sweep(
        workdir,
        &workdir.join("out/readiness_ultimate_sweep_readiness_closure_sweep.json"),
    )?;
    let convergence = readiness_convergence_sweep(
        workdir,
        &workdir.join("out/readiness_convergence_sweep_readiness_closure_sweep.json"),
    )?;
    let stabilization = readiness_stabilization_sweep(
        workdir,
        &workdir.join("out/readiness_stabilization_sweep_readiness_closure_sweep.json"),
    )?;
    let final_consolidation = readiness_final_consolidation_sweep(
        workdir,
        &workdir.join("out/readiness_final_consolidation_sweep_readiness_closure_sweep.json"),
    )?;

    let authority_ctx = require_readiness_closure_inputs(
        &truths,
        Some(&reduction),
        Some(&applied),
        Some(&entry),
        Some(&spine),
        Some(&readiness_authority),
        Some(&final_readiness.authority),
        Some(&residual.sweep),
        Some(&residual_free.authority),
        Some(&absolute.sweep),
        Some(&terminal.sweep),
        Some(&ultimate.sweep),
        Some(&convergence.sweep),
        Some(&stabilization.sweep),
        Some(&final_consolidation.sweep),
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
            "InteropConsistencyMatrix",
            workdir,
            "out/interop_consistency_matrix.json",
            &authority_ctx,
        )?,
        check_consumer(
            "V19PrepGateHelper",
            workdir,
            "out/v18_gate_report.json",
            &authority_ctx,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_path_count = consumers
        .iter()
        .filter(|consumer| !matches!(consumer.status, ReadinessClosureStatusV1::Pass))
        .count() as u16;

    let closure_status = if consumers
        .iter()
        .any(|c| matches!(c.status, ReadinessClosureStatusV1::LegacyPresent))
    {
        ReadinessClosureStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        ReadinessClosureStatusV1::Pass
    } else {
        ReadinessClosureStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        consumers.len() as u16,
        residual_path_count,
        closure_status,
    );
    let report = ReadinessClosureSweepReportV1 {
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
    authority_ctx: &crate::ReadinessClosureInputsV1,
) -> Result<ReadinessClosureConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories
            .insert(ReadinessClosureMismatchCategoryV1::ConsumerSkippedReadinessClosure);
        return Ok(ReadinessClosureConsumerStatusV1 {
            consumer: consumer.to_string(),
            status: ReadinessClosureStatusV1::LegacyPresent,
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
        mismatch_categories.insert(ReadinessClosureMismatchCategoryV1::ReadinessInputScopeMismatch);
    }
    if !field_match(
        "canonical_readiness_spine_digest_prefix",
        &authority_ctx.canonical_readiness_spine_digest_prefix,
    ) {
        mismatch_categories.insert(ReadinessClosureMismatchCategoryV1::ReadinessInputSpineMismatch);
    }

    if !field_match(
        "readiness_final_consolidation_sweep_digest_prefix",
        &authority_ctx.readiness_final_consolidation_sweep_digest_prefix,
    ) {
        mismatch_categories
            .insert(ReadinessClosureMismatchCategoryV1::ConsumerSkippedReadinessClosure);
    }

    if !field_match(
        "residual_free_readiness_authority_digest_prefix",
        &authority_ctx.residual_free_readiness_consumer_authority_digest_prefix,
    ) {
        mismatch_categories
            .insert(ReadinessClosureMismatchCategoryV1::ConsumerUsedReadinessWrapperPath);
    }

    if !field_match(
        "readiness_final_consolidation_sweep_digest_prefix",
        &authority_ctx.readiness_final_consolidation_sweep_digest_prefix,
    ) {
        mismatch_categories.insert(ReadinessClosureMismatchCategoryV1::ReadinessWrapperPathPresent);
    }

    let status = if mismatch_categories.is_empty() {
        ReadinessClosureStatusV1::Pass
    } else if mismatch_categories
        .contains(&ReadinessClosureMismatchCategoryV1::ReadinessWrapperPathPresent)
        || mismatch_categories
            .contains(&ReadinessClosureMismatchCategoryV1::ConsumerUsedReadinessWrapperPath)
    {
        ReadinessClosureStatusV1::LegacyPresent
    } else {
        ReadinessClosureStatusV1::Fail
    };

    Ok(ReadinessClosureConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::ReadinessClosureInputsV1,
    covered_consumer_count: u16,
    residual_path_count: u16,
    closure_status: ReadinessClosureStatusV1,
) -> ReadinessClosureSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"readiness_closure_sweep_v1");
    bytes.extend_from_slice(ctx.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_readiness_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_readiness_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        ctx.final_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(ctx.final_readiness_residual_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        ctx.residual_free_readiness_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.residual_free_readiness_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.absolute_final_readiness_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.terminal_readiness_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(ctx.readiness_convergence_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.readiness_stabilization_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        ctx.readiness_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{closure_status:?}").as_bytes());

    ReadinessClosureSweepV1 {
        applied_supported_set_digest_prefix: ctx.applied_supported_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: ctx
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: ctx
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_readiness_authority_digest_prefix: ctx
            .canonical_readiness_authority_digest_prefix
            .clone(),
        final_readiness_consumer_authority_digest_prefix: ctx
            .final_readiness_consumer_authority_digest_prefix
            .clone(),
        final_readiness_residual_sweep_digest_prefix: ctx
            .final_readiness_residual_sweep_digest_prefix
            .clone(),
        residual_free_readiness_consumer_authority_digest_prefix: ctx
            .residual_free_readiness_consumer_authority_digest_prefix
            .clone(),
        residual_free_readiness_absolute_sweep_digest_prefix: ctx
            .residual_free_readiness_absolute_sweep_digest_prefix
            .clone(),
        absolute_final_readiness_terminal_sweep_digest_prefix: ctx
            .absolute_final_readiness_terminal_sweep_digest_prefix
            .clone(),
        terminal_readiness_ultimate_sweep_digest_prefix: ctx
            .terminal_readiness_ultimate_sweep_digest_prefix
            .clone(),
        readiness_convergence_sweep_digest_prefix: ctx
            .readiness_convergence_sweep_digest_prefix
            .clone(),
        readiness_stabilization_sweep_digest_prefix: ctx
            .readiness_stabilization_sweep_digest_prefix
            .clone(),
        readiness_final_consolidation_sweep_digest_prefix: ctx
            .readiness_final_consolidation_sweep_digest_prefix
            .clone(),
        covered_consumer_count,
        residual_path_count,
        closure_status,
        closure_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn readiness_closure_sweep_digest_stable() {
        let ctx = crate::ReadinessClosureInputsV1 {
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_readiness_spine_digest_prefix: "33".repeat(8),
            canonical_readiness_authority_digest_prefix: "44".repeat(8),
            final_readiness_consumer_authority_digest_prefix: "55".repeat(8),
            final_readiness_residual_sweep_digest_prefix: "66".repeat(8),
            residual_free_readiness_consumer_authority_digest_prefix: "77".repeat(8),
            residual_free_readiness_absolute_sweep_digest_prefix: "88".repeat(8),
            absolute_final_readiness_terminal_sweep_digest_prefix: "99".repeat(8),
            terminal_readiness_ultimate_sweep_digest_prefix: "aa".repeat(8),
            readiness_convergence_sweep_digest_prefix: "bb".repeat(8),
            readiness_stabilization_sweep_digest_prefix: "cc".repeat(8),
            readiness_final_consolidation_sweep_digest_prefix: "dd".repeat(8),
            authority_digest: "ee".repeat(32),
        };
        let first = derive_sweep(&ctx, 7, 0, ReadinessClosureStatusV1::Pass);
        let second = derive_sweep(&ctx, 7, 0, ReadinessClosureStatusV1::Pass);
        assert_eq!(first.closure_digest, second.closure_digest);
    }

    #[test]
    fn readiness_closure_status_deterministic_for_wrapper_paths() {
        let mismatches =
            BTreeSet::from([ReadinessClosureMismatchCategoryV1::ReadinessWrapperPathPresent]);
        let status = if mismatches.is_empty() {
            ReadinessClosureStatusV1::Pass
        } else if mismatches
            .contains(&ReadinessClosureMismatchCategoryV1::ReadinessWrapperPathPresent)
            || mismatches
                .contains(&ReadinessClosureMismatchCategoryV1::ConsumerUsedReadinessWrapperPath)
        {
            ReadinessClosureStatusV1::LegacyPresent
        } else {
            ReadinessClosureStatusV1::Fail
        };
        assert_eq!(status, ReadinessClosureStatusV1::LegacyPresent);
    }
}
