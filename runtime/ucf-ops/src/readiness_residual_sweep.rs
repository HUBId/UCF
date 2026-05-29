use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, derive_canonical_readiness_authority_v2,
    derive_canonical_readiness_spine, derive_slot_reviewability_truths,
    final_readiness_consumer_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, prefix_hex, reduce_reviewability,
    require_canonical_governance_entry, require_final_readiness_inputs, resolve_strict_evidence,
    validate_governance_primary_surfaces_with_applied_scope, CanonicalReadinessAuthorityStatusV2,
    FinalReadinessConsumerAuthorityV1, OpsError, StrictEvidenceContextV1,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalReadinessResidualSweepStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalReadinessResidualMismatchCategoryV1 {
    ConsumerSkippedFinalReadinessInputs,
    ConsumerUsedResidualReadinessPath,
    ReadinessInputScopeMismatch,
    ReadinessInputSpineMismatch,
    ResidualReadinessPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalReadinessResidualConsumerStatusV1 {
    pub consumer: String,
    pub status: FinalReadinessResidualSweepStatusV1,
    pub mismatch_categories: Vec<FinalReadinessResidualMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalReadinessResidualSweepV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub sweep_status: FinalReadinessResidualSweepStatusV1,
    pub sweep_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalReadinessResidualSweepReportV1 {
    pub schema_version: u16,
    pub sweep: FinalReadinessResidualSweepV1,
    pub consumers: Vec<FinalReadinessResidualConsumerStatusV1>,
}

pub fn readiness_residual_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<FinalReadinessResidualSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_readiness_residual_sweep.json"),
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
        &prefix_hex(&entry.authority_digest, 16),
        &prefix_hex(&spine.spine_digest, 16),
        4,
        CanonicalReadinessAuthorityStatusV2::Pass,
    );
    let final_readiness = final_readiness_consumer_sweep(
        workdir,
        &workdir.join("out/final_readiness_consumer_sweep_readiness_residual_sweep.json"),
    )?;
    let final_consumer = final_readiness.authority;

    let final_inputs = require_final_readiness_inputs(
        &truths,
        Some(&reduction),
        &applied,
        &entry,
        Some(&spine),
        Some(&readiness_authority),
        Some(&final_consumer),
    )?;

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
            !matches!(consumer.status, FinalReadinessResidualSweepStatusV1::Pass)
                || consumer.mismatch_categories.contains(
                    &FinalReadinessResidualMismatchCategoryV1::ResidualReadinessPathPresent,
                )
        })
        .count() as u16;

    let sweep_status = if consumers
        .iter()
        .any(|c| matches!(c.status, FinalReadinessResidualSweepStatusV1::LegacyPresent))
    {
        FinalReadinessResidualSweepStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        FinalReadinessResidualSweepStatusV1::Pass
    } else {
        FinalReadinessResidualSweepStatusV1::Fail
    };

    let sweep = derive_residual_sweep(
        &final_inputs.applied_supported_set_digest_prefix,
        &final_inputs.canonical_governance_entry_digest_prefix,
        &final_inputs.canonical_readiness_spine_digest_prefix,
        &final_inputs.canonical_readiness_authority_digest_prefix,
        &final_inputs.final_readiness_consumer_authority_digest_prefix,
        consumers.len() as u16,
        residual_path_count,
        sweep_status,
    );
    let report = FinalReadinessResidualSweepReportV1 {
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
    final_consumer: &FinalReadinessConsumerAuthorityV1,
    allow_absent: bool,
) -> Result<FinalReadinessResidualConsumerStatusV1, OpsError> {
    let mut categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        categories
            .insert(FinalReadinessResidualMismatchCategoryV1::ConsumerSkippedFinalReadinessInputs);
        let status = if allow_absent {
            FinalReadinessResidualSweepStatusV1::Fail
        } else {
            FinalReadinessResidualSweepStatusV1::LegacyPresent
        };
        return Ok(FinalReadinessResidualConsumerStatusV1 {
            consumer: consumer_name.to_string(),
            status,
            mismatch_categories: categories.into_iter().collect(),
        });
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let expected_scope = &final_consumer.applied_supported_set_digest_prefix;
    let expected_spine = &final_consumer.canonical_readiness_spine_digest_prefix;
    let expected_authority = prefix_hex(&final_consumer.authority_digest, 16);

    let scope_match = value
        .get("applied_supported_set_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|scope| scope == expected_scope)
        .unwrap_or(false);
    if !scope_match {
        categories.insert(FinalReadinessResidualMismatchCategoryV1::ReadinessInputScopeMismatch);
    }

    let spine_match = value
        .get("canonical_readiness_spine_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|prefix| prefix == expected_spine)
        .unwrap_or(false);
    if !spine_match {
        categories.insert(FinalReadinessResidualMismatchCategoryV1::ReadinessInputSpineMismatch);
    }

    let final_consumer_match = value
        .get("final_readiness_consumer_authority_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|prefix| prefix == expected_authority)
        .unwrap_or(false);
    if !final_consumer_match {
        categories
            .insert(FinalReadinessResidualMismatchCategoryV1::ConsumerSkippedFinalReadinessInputs);
    }

    let residual_code_present = value
        .get("blocking_codes")
        .and_then(serde_json::Value::as_array)
        .map(|codes| {
            codes.iter().any(|code| {
                code.as_str()
                    .map(|code| code.contains("RESIDUAL_READINESS_PATH"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if residual_code_present {
        categories.insert(FinalReadinessResidualMismatchCategoryV1::ResidualReadinessPathPresent);
        categories
            .insert(FinalReadinessResidualMismatchCategoryV1::ConsumerUsedResidualReadinessPath);
    }

    let status = if categories.is_empty() {
        FinalReadinessResidualSweepStatusV1::Pass
    } else if categories
        .contains(&FinalReadinessResidualMismatchCategoryV1::ResidualReadinessPathPresent)
    {
        FinalReadinessResidualSweepStatusV1::LegacyPresent
    } else {
        FinalReadinessResidualSweepStatusV1::Fail
    };

    Ok(FinalReadinessResidualConsumerStatusV1 {
        consumer: consumer_name.to_string(),
        status,
        mismatch_categories: categories.into_iter().collect(),
    })
}

#[allow(clippy::too_many_arguments)]
fn derive_residual_sweep(
    applied_scope_prefix: &str,
    entry_prefix: &str,
    spine_prefix: &str,
    authority_prefix: &str,
    final_consumer_prefix: &str,
    covered_consumer_count: u16,
    residual_path_count: u16,
    sweep_status: FinalReadinessResidualSweepStatusV1,
) -> FinalReadinessResidualSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"final_readiness_residual_sweep_v1");
    bytes.extend_from_slice(applied_scope_prefix.as_bytes());
    bytes.extend_from_slice(entry_prefix.as_bytes());
    bytes.extend_from_slice(spine_prefix.as_bytes());
    bytes.extend_from_slice(authority_prefix.as_bytes());
    bytes.extend_from_slice(final_consumer_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{sweep_status:?}").as_bytes());
    FinalReadinessResidualSweepV1 {
        applied_supported_set_digest_prefix: applied_scope_prefix.to_string(),
        canonical_governance_entry_digest_prefix: entry_prefix.to_string(),
        canonical_readiness_spine_digest_prefix: spine_prefix.to_string(),
        canonical_readiness_authority_digest_prefix: authority_prefix.to_string(),
        final_readiness_consumer_authority_digest_prefix: final_consumer_prefix.to_string(),
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
            "spine123456789012",
            "auth123456789012",
            "final123456789012",
            5,
            0,
            FinalReadinessResidualSweepStatusV1::Pass,
        );
        let second = derive_residual_sweep(
            "scope123456789012",
            "entry123456789012",
            "spine123456789012",
            "auth123456789012",
            "final123456789012",
            5,
            0,
            FinalReadinessResidualSweepStatusV1::Pass,
        );
        assert_eq!(first.sweep_digest, second.sweep_digest);
    }
}
