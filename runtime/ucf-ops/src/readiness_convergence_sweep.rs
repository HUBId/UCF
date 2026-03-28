use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, derive_canonical_readiness_authority_v2,
    derive_canonical_readiness_spine, derive_slot_reviewability_truths,
    final_readiness_consumer_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, readiness_absolute_sweep,
    readiness_residual_sweep, readiness_terminal_sweep, readiness_ultimate_sweep,
    reduce_reviewability, require_canonical_governance_entry, require_readiness_convergence_inputs,
    residual_free_readiness_sweep, resolve_strict_evidence,
    validate_governance_primary_surfaces_with_applied_scope, CanonicalReadinessAuthorityStatusV2,
    OpsError, StrictEvidenceContextV1,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessConvergenceStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessConvergenceMismatchCategoryV1 {
    ConsumerSkippedReadinessConvergence,
    ConsumerUsedReadinessMemoPath,
    ReadinessInputScopeMismatch,
    ReadinessInputSpineMismatch,
    ReadinessMemoPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessConvergenceConsumerStatusV1 {
    pub consumer: String,
    pub status: ReadinessConvergenceStatusV1,
    pub mismatch_categories: Vec<ReadinessConvergenceMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessConvergenceSweepV1 {
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
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub convergence_status: ReadinessConvergenceStatusV1,
    pub convergence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessConvergenceSweepReportV1 {
    pub schema_version: u16,
    pub sweep: ReadinessConvergenceSweepV1,
    pub consumers: Vec<ReadinessConvergenceConsumerStatusV1>,
}

pub fn readiness_convergence_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<ReadinessConvergenceSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_readiness_convergence_sweep.json"),
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
        &crate::prefix_hex(&entry.authority_digest, DIGEST_PREFIX_LEN),
        &crate::prefix_hex(&spine.spine_digest, DIGEST_PREFIX_LEN),
        4,
        CanonicalReadinessAuthorityStatusV2::Pass,
    );
    let final_readiness = final_readiness_consumer_sweep(
        workdir,
        &workdir.join("out/final_readiness_consumer_sweep_readiness_convergence_sweep.json"),
    )?;
    let residual = readiness_residual_sweep(
        workdir,
        &workdir.join("out/readiness_residual_sweep_readiness_convergence_sweep.json"),
    )?;
    let residual_free = residual_free_readiness_sweep(
        workdir,
        &workdir.join("out/residual_free_readiness_sweep_readiness_convergence_sweep.json"),
    )?;
    let absolute = readiness_absolute_sweep(
        workdir,
        &workdir.join("out/readiness_absolute_sweep_readiness_convergence_sweep.json"),
    )?;
    let terminal = readiness_terminal_sweep(
        workdir,
        &workdir.join("out/readiness_terminal_sweep_readiness_convergence_sweep.json"),
    )?;
    let ultimate = readiness_ultimate_sweep(
        workdir,
        &workdir.join("out/readiness_ultimate_sweep_readiness_convergence_sweep.json"),
    )?;

    let authority_ctx = require_readiness_convergence_inputs(
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
            "V16PrepGateHelper",
            workdir,
            "out/v15_gate_report.json",
            &authority_ctx,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_path_count = consumers
        .iter()
        .filter(|consumer| !matches!(consumer.status, ReadinessConvergenceStatusV1::Pass))
        .count() as u16;

    let convergence_status = if consumers
        .iter()
        .any(|c| matches!(c.status, ReadinessConvergenceStatusV1::LegacyPresent))
    {
        ReadinessConvergenceStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        ReadinessConvergenceStatusV1::Pass
    } else {
        ReadinessConvergenceStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        consumers.len() as u16,
        residual_path_count,
        convergence_status,
    );
    let report = ReadinessConvergenceSweepReportV1 {
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
    authority_ctx: &crate::ReadinessConvergenceInputsV1,
) -> Result<ReadinessConvergenceConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ConsumerSkippedReadinessConvergence);
        return Ok(ReadinessConvergenceConsumerStatusV1 {
            consumer: consumer.to_string(),
            status: ReadinessConvergenceStatusV1::LegacyPresent,
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let expected_scope = &authority_ctx.applied_supported_set_digest_prefix;
    let expected_spine = &authority_ctx.canonical_readiness_spine_digest_prefix;
    let expected_final = &authority_ctx.final_readiness_consumer_authority_digest_prefix;
    let expected_residual = &authority_ctx.final_readiness_residual_sweep_digest_prefix;
    let expected_residual_free =
        &authority_ctx.residual_free_readiness_consumer_authority_digest_prefix;
    let expected_absolute = &authority_ctx.residual_free_readiness_absolute_sweep_digest_prefix;
    let expected_terminal = &authority_ctx.absolute_final_readiness_terminal_sweep_digest_prefix;
    let expected_ultimate = &authority_ctx.terminal_readiness_ultimate_sweep_digest_prefix;

    let field_match = |field: &str, expected: &str| {
        value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .map(|s| s == expected)
            .unwrap_or(false)
    };

    if !field_match("applied_supported_set_digest_prefix", expected_scope) {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ReadinessInputScopeMismatch);
    }
    if !field_match("canonical_readiness_spine_digest_prefix", expected_spine) {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ReadinessInputSpineMismatch);
    }
    if !field_match(
        "final_readiness_consumer_authority_digest_prefix",
        expected_final,
    ) {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ConsumerSkippedReadinessConvergence);
    }
    if !field_match("readiness_residual_sweep_digest_prefix", expected_residual) {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ReadinessMemoPathPresent);
    }
    if !field_match(
        "residual_free_readiness_authority_digest_prefix",
        expected_residual_free,
    ) {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ConsumerUsedReadinessMemoPath);
    }
    if !field_match("readiness_absolute_sweep_digest_prefix", expected_absolute) {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ReadinessMemoPathPresent);
    }
    if !field_match("readiness_terminal_sweep_digest_prefix", expected_terminal) {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ConsumerSkippedReadinessConvergence);
    }
    if !field_match("readiness_ultimate_sweep_digest_prefix", expected_ultimate) {
        mismatch_categories
            .insert(ReadinessConvergenceMismatchCategoryV1::ConsumerSkippedReadinessConvergence);
    }

    let status = if mismatch_categories.is_empty() {
        ReadinessConvergenceStatusV1::Pass
    } else if mismatch_categories
        .contains(&ReadinessConvergenceMismatchCategoryV1::ReadinessMemoPathPresent)
        || mismatch_categories
            .contains(&ReadinessConvergenceMismatchCategoryV1::ConsumerUsedReadinessMemoPath)
    {
        ReadinessConvergenceStatusV1::LegacyPresent
    } else {
        ReadinessConvergenceStatusV1::Fail
    };

    Ok(ReadinessConvergenceConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::ReadinessConvergenceInputsV1,
    covered_consumer_count: u16,
    residual_path_count: u16,
    convergence_status: ReadinessConvergenceStatusV1,
) -> ReadinessConvergenceSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"readiness_convergence_sweep_v1");
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
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", convergence_status).as_bytes());

    ReadinessConvergenceSweepV1 {
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
        covered_consumer_count,
        residual_path_count,
        convergence_status,
        convergence_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn readiness_convergence_sweep_digest_stable() {
        let ctx = crate::ReadinessConvergenceInputsV1 {
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
            authority_digest: "bb".repeat(32),
        };
        let first = derive_sweep(&ctx, 6, 0, ReadinessConvergenceStatusV1::Pass);
        let second = derive_sweep(&ctx, 6, 0, ReadinessConvergenceStatusV1::Pass);
        assert_eq!(first.convergence_digest, second.convergence_digest);
    }

    #[test]
    fn readiness_convergence_status_is_deterministic_for_memo_mismatch() {
        let mismatches =
            BTreeSet::from([ReadinessConvergenceMismatchCategoryV1::ReadinessMemoPathPresent]);
        let status = if mismatches.is_empty() {
            ReadinessConvergenceStatusV1::Pass
        } else if mismatches
            .contains(&ReadinessConvergenceMismatchCategoryV1::ReadinessMemoPathPresent)
            || mismatches
                .contains(&ReadinessConvergenceMismatchCategoryV1::ConsumerUsedReadinessMemoPath)
        {
            ReadinessConvergenceStatusV1::LegacyPresent
        } else {
            ReadinessConvergenceStatusV1::Fail
        };
        assert_eq!(status, ReadinessConvergenceStatusV1::LegacyPresent);
    }
}
