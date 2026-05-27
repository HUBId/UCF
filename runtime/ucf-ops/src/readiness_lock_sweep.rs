use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, final_governance_consumer_sweep, governance_absolute_sweep,
    governance_closure_sweep, governance_convergence_sweep, governance_entry_sweep,
    governance_final_consolidation_sweep, governance_lock_sweep, governance_residual_sweep,
    governance_seal_sweep, governance_stabilization_sweep, governance_terminal_sweep,
    governance_ultimate_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, models_supported_scope_decision,
    require_governance_lock_inputs, require_readiness_lock_inputs, residual_free_governance_sweep,
    validate_governance_primary_surfaces_with_applied_scope, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessLockStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReadinessLockMismatchCategoryV1 {
    ConsumerSkippedReadinessLock,
    ConsumerUsedAuxiliaryReadinessPath,
    ReadinessScopeMismatch,
    ReadinessExecutionRealityMismatch,
    ReadinessInflationPathPresent,
    HandoffReadinessAsPrimaryTruth,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessLockConsumerStatusV1 {
    pub consumer: String,
    pub status: ReadinessLockStatusV1,
    pub mismatch_categories: Vec<ReadinessLockMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessLockSweepV1 {
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
    pub governance_final_consolidation_sweep_digest_prefix: String,
    pub governance_closure_sweep_digest_prefix: String,
    pub governance_seal_sweep_digest_prefix: String,
    pub governance_lock_sweep_digest_prefix: String,
    pub supported_scope_decision_digest_prefix: String,
    pub canonical_execution_reality_digest_prefix: String,
    pub covered_readiness_consumer_count: u16,
    pub residual_readiness_path_count: u16,
    pub readiness_lock_status: ReadinessLockStatusV1,
    pub readiness_lock_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadinessLockSweepReportV1 {
    pub schema_version: u16,
    pub sweep: ReadinessLockSweepV1,
    pub consumers: Vec<ReadinessLockConsumerStatusV1>,
}

pub fn readiness_lock_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<ReadinessLockSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_readiness_lock_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let entry = derive_canonical_governance_entry(&applied, &surfaces)?;

    let entry_sweep = governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_readiness_lock_sweep.json"),
    )?;
    let final_governance = final_governance_consumer_sweep(
        workdir,
        &workdir.join("out/final_governance_consumer_sweep_readiness_lock_sweep.json"),
    )?;
    let residual = governance_residual_sweep(
        workdir,
        &workdir.join("out/governance_residual_sweep_readiness_lock_sweep.json"),
    )?;
    let residual_free = residual_free_governance_sweep(
        workdir,
        &workdir.join("out/residual_free_governance_sweep_readiness_lock_sweep.json"),
    )?;
    let absolute = governance_absolute_sweep(
        workdir,
        &workdir.join("out/governance_absolute_sweep_readiness_lock_sweep.json"),
    )?;
    let terminal = governance_terminal_sweep(
        workdir,
        &workdir.join("out/governance_terminal_sweep_readiness_lock_sweep.json"),
    )?;
    let ultimate = governance_ultimate_sweep(
        workdir,
        &workdir.join("out/governance_ultimate_sweep_readiness_lock_sweep.json"),
    )?;
    let convergence = governance_convergence_sweep(
        workdir,
        &workdir.join("out/governance_convergence_sweep_readiness_lock_sweep.json"),
    )?;
    let stabilization = governance_stabilization_sweep(
        workdir,
        &workdir.join("out/governance_stabilization_sweep_readiness_lock_sweep.json"),
    )?;
    let final_consolidation = governance_final_consolidation_sweep(
        workdir,
        &workdir.join("out/governance_final_consolidation_sweep_readiness_lock_sweep.json"),
    )?;
    let closure = governance_closure_sweep(
        workdir,
        &workdir.join("out/governance_closure_sweep_readiness_lock_sweep.json"),
    )?;
    let seal = governance_seal_sweep(
        workdir,
        &workdir.join("out/governance_seal_sweep_readiness_lock_sweep.json"),
    )?;
    let lock = governance_lock_sweep(
        workdir,
        &workdir.join("out/governance_lock_sweep_readiness_lock_sweep.json"),
    )?;
    let scope_decision = models_supported_scope_decision(
        workdir,
        &workdir.join("out/supported_scope_decision_readiness_lock_sweep.json"),
    )?;

    let governance_lock_inputs = require_governance_lock_inputs(
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
        Some(&final_consolidation.sweep),
        Some(&closure.sweep),
        Some(&seal.sweep),
    )?;

    let mut reality_bytes = Vec::new();
    reality_bytes.extend_from_slice(backend.snapshot_digest.as_bytes());
    reality_bytes.extend_from_slice(active.snapshot_digest.as_bytes());
    let canonical_execution_reality_digest = crate::sha256_hex(&reality_bytes);

    let authority_ctx = require_readiness_lock_inputs(
        Some(&governance_lock_inputs),
        Some(&lock.sweep),
        Some(&scope_decision.decision),
        Some(&crate::prefix_hex(&canonical_execution_reality_digest, 16)),
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
            "V21ReadinessGate",
            workdir,
            "out/v20_gate_report.json",
            &authority_ctx,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_readiness_path_count = consumers
        .iter()
        .filter(|consumer| !matches!(consumer.status, ReadinessLockStatusV1::Pass))
        .count() as u16;
    let readiness_lock_status = if consumers
        .iter()
        .any(|c| matches!(c.status, ReadinessLockStatusV1::LegacyPresent))
    {
        ReadinessLockStatusV1::LegacyPresent
    } else if residual_readiness_path_count == 0 {
        ReadinessLockStatusV1::Pass
    } else {
        ReadinessLockStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        consumers.len() as u16,
        residual_readiness_path_count,
        readiness_lock_status,
    );
    let report = ReadinessLockSweepReportV1 {
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
    authority_ctx: &crate::ReadinessLockInputsV1,
) -> Result<ReadinessLockConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(ReadinessLockMismatchCategoryV1::ConsumerSkippedReadinessLock);
        return Ok(ReadinessLockConsumerStatusV1 {
            consumer: consumer.to_string(),
            status: ReadinessLockStatusV1::LegacyPresent,
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
        mismatch_categories.insert(ReadinessLockMismatchCategoryV1::ReadinessScopeMismatch);
    }
    if !field_match(
        "governance_lock_sweep_digest_prefix",
        &authority_ctx.governance_lock_sweep_digest_prefix,
    ) {
        mismatch_categories.insert(ReadinessLockMismatchCategoryV1::ConsumerSkippedReadinessLock);
    }
    if !field_match(
        "supported_scope_decision_digest_prefix",
        &authority_ctx.supported_scope_decision_digest_prefix,
    ) {
        mismatch_categories.insert(ReadinessLockMismatchCategoryV1::ReadinessScopeMismatch);
    }
    if !field_match(
        "canonical_execution_reality_digest_prefix",
        &authority_ctx.canonical_execution_reality_digest_prefix,
    ) {
        mismatch_categories
            .insert(ReadinessLockMismatchCategoryV1::ReadinessExecutionRealityMismatch);
    }

    for forbidden in [
        "readiness_handoff_summary",
        "readiness_auxiliary_projection",
        "readiness_implementation_intent",
        "readiness_compatibility_frame",
        "export_local_readiness_note",
        "historical_readiness_success",
        "dormant_adapter_readiness",
    ] {
        if value.get(forbidden).is_some() {
            mismatch_categories
                .insert(ReadinessLockMismatchCategoryV1::ReadinessInflationPathPresent);
        }
    }
    if value.get("operator_handoff_notes").is_some() || value.get("review_handoff_notes").is_some()
    {
        mismatch_categories.insert(ReadinessLockMismatchCategoryV1::HandoffReadinessAsPrimaryTruth);
    }
    if value.get("auxiliary_readiness_matrix").is_some() {
        mismatch_categories
            .insert(ReadinessLockMismatchCategoryV1::ConsumerUsedAuxiliaryReadinessPath);
    }

    let status = if mismatch_categories.is_empty() {
        ReadinessLockStatusV1::Pass
    } else if mismatch_categories
        .contains(&ReadinessLockMismatchCategoryV1::ReadinessInflationPathPresent)
        || mismatch_categories
            .contains(&ReadinessLockMismatchCategoryV1::ConsumerUsedAuxiliaryReadinessPath)
        || mismatch_categories
            .contains(&ReadinessLockMismatchCategoryV1::HandoffReadinessAsPrimaryTruth)
    {
        ReadinessLockStatusV1::LegacyPresent
    } else {
        ReadinessLockStatusV1::Fail
    };

    Ok(ReadinessLockConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::ReadinessLockInputsV1,
    covered_readiness_consumer_count: u16,
    residual_readiness_path_count: u16,
    readiness_lock_status: ReadinessLockStatusV1,
) -> ReadinessLockSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"readiness_lock_sweep_v1");
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
    bytes.extend_from_slice(
        ctx.governance_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(ctx.governance_closure_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.governance_seal_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.governance_lock_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.supported_scope_decision_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_execution_reality_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_readiness_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_readiness_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{readiness_lock_status:?}").as_bytes());

    ReadinessLockSweepV1 {
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
        governance_final_consolidation_sweep_digest_prefix: ctx
            .governance_final_consolidation_sweep_digest_prefix
            .clone(),
        governance_closure_sweep_digest_prefix: ctx.governance_closure_sweep_digest_prefix.clone(),
        governance_seal_sweep_digest_prefix: ctx.governance_seal_sweep_digest_prefix.clone(),
        governance_lock_sweep_digest_prefix: ctx.governance_lock_sweep_digest_prefix.clone(),
        supported_scope_decision_digest_prefix: ctx.supported_scope_decision_digest_prefix.clone(),
        canonical_execution_reality_digest_prefix: ctx
            .canonical_execution_reality_digest_prefix
            .clone(),
        covered_readiness_consumer_count,
        residual_readiness_path_count,
        readiness_lock_status,
        readiness_lock_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_lock_sweep_digest_stable() {
        let ctx = crate::ReadinessLockInputsV1 {
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
            governance_final_consolidation_sweep_digest_prefix: "cc".repeat(8),
            governance_closure_sweep_digest_prefix: "dd".repeat(8),
            governance_seal_sweep_digest_prefix: "ee".repeat(8),
            governance_lock_sweep_digest_prefix: "ff".repeat(8),
            supported_scope_decision_digest_prefix: "12".repeat(8),
            canonical_execution_reality_digest_prefix: "34".repeat(8),
            authority_digest: "56".repeat(32),
        };
        let first = derive_sweep(&ctx, 7, 0, ReadinessLockStatusV1::Pass);
        let second = derive_sweep(&ctx, 7, 0, ReadinessLockStatusV1::Pass);
        assert_eq!(first.readiness_lock_digest, second.readiness_lock_digest);
    }
}
