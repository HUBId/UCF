use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, derive_canonical_readiness_authority_v2,
    derive_canonical_readiness_spine, derive_slot_reviewability_truths,
    final_readiness_consumer_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, prefix_hex, readiness_residual_sweep,
    reduce_reviewability, require_canonical_governance_entry,
    require_residual_free_final_readiness_inputs, resolve_strict_evidence,
    validate_governance_primary_surfaces_with_applied_scope, CanonicalReadinessAuthorityStatusV2,
    OpsError, StrictEvidenceContextV1,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreeReadinessConsumerAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreeReadinessMismatchCategoryV1 {
    ConsumerSkippedResidualFreeFinalReadinessInputs,
    ConsumerUsedHistoricalReadinessPath,
    ReadinessInputScopeMismatch,
    ReadinessInputSpineMismatch,
    HistoricalReadinessPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeReadinessConsumerStatusV1 {
    pub consumer: String,
    pub status: ResidualFreeReadinessConsumerAuthorityStatusV1,
    pub mismatch_categories: Vec<ResidualFreeReadinessMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeReadinessConsumerAuthorityV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub final_readiness_consumer_authority_digest_prefix: String,
    pub final_readiness_residual_sweep_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub authority_status: ResidualFreeReadinessConsumerAuthorityStatusV1,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeReadinessSweepReportV1 {
    pub schema_version: u16,
    pub authority: ResidualFreeReadinessConsumerAuthorityV1,
    pub consumers: Vec<ResidualFreeReadinessConsumerStatusV1>,
}

pub fn residual_free_readiness_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<ResidualFreeReadinessSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_residual_free_readiness_sweep.json"),
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
        &workdir.join("out/final_readiness_consumer_sweep_residual_free_readiness_sweep.json"),
    )?;
    let residual = readiness_residual_sweep(
        workdir,
        &workdir.join("out/readiness_residual_sweep_residual_free_readiness_sweep.json"),
    )?;

    let authority_ctx = require_residual_free_final_readiness_inputs(
        &truths,
        Some(&reduction),
        &applied,
        &entry,
        Some(&spine),
        Some(&readiness_authority),
        Some(&final_readiness.authority),
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
                ResidualFreeReadinessConsumerAuthorityStatusV1::Pass
            )
        })
        .count() as u16;

    let authority_status = if consumers.iter().any(|c| {
        matches!(
            c.status,
            ResidualFreeReadinessConsumerAuthorityStatusV1::LegacyPresent
        )
    }) {
        ResidualFreeReadinessConsumerAuthorityStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        ResidualFreeReadinessConsumerAuthorityStatusV1::Pass
    } else {
        ResidualFreeReadinessConsumerAuthorityStatusV1::Fail
    };

    let authority = derive_authority(
        &authority_ctx.applied_supported_set_digest_prefix,
        &authority_ctx.canonical_governance_entry_digest_prefix,
        &authority_ctx.canonical_readiness_spine_digest_prefix,
        &authority_ctx.canonical_readiness_authority_digest_prefix,
        &authority_ctx.final_readiness_consumer_authority_digest_prefix,
        &authority_ctx.final_readiness_residual_sweep_digest_prefix,
        consumers.len() as u16,
        residual_path_count,
        authority_status,
    );

    let report = ResidualFreeReadinessSweepReportV1 {
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
    authority_ctx: &crate::ResidualFreeFinalReadinessInputsV1,
    allow_absent: bool,
) -> Result<ResidualFreeReadinessConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(
            ResidualFreeReadinessMismatchCategoryV1::ConsumerSkippedResidualFreeFinalReadinessInputs,
        );
        let status = if allow_absent {
            ResidualFreeReadinessConsumerAuthorityStatusV1::Fail
        } else {
            ResidualFreeReadinessConsumerAuthorityStatusV1::LegacyPresent
        };
        return Ok(ResidualFreeReadinessConsumerStatusV1 {
            consumer: consumer.to_string(),
            status,
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let expected_scope = &authority_ctx.applied_supported_set_digest_prefix;
    let expected_spine = &authority_ctx.canonical_readiness_spine_digest_prefix;
    let expected_final = &authority_ctx.final_readiness_consumer_authority_digest_prefix;
    let expected_residual = &authority_ctx.final_readiness_residual_sweep_digest_prefix;
    let expected_authority = prefix_hex(&authority_ctx.authority_digest, DIGEST_PREFIX_LEN);

    let scope_match = value
        .get("applied_supported_set_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_scope)
        .unwrap_or(false);
    if !scope_match {
        mismatch_categories
            .insert(ResidualFreeReadinessMismatchCategoryV1::ReadinessInputScopeMismatch);
    }

    let spine_match = value
        .get("canonical_readiness_spine_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_spine)
        .unwrap_or(false);
    if !spine_match {
        mismatch_categories
            .insert(ResidualFreeReadinessMismatchCategoryV1::ReadinessInputSpineMismatch);
    }

    let final_match = value
        .get("final_readiness_consumer_authority_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_final)
        .unwrap_or(false);
    if !final_match {
        mismatch_categories.insert(
            ResidualFreeReadinessMismatchCategoryV1::ConsumerSkippedResidualFreeFinalReadinessInputs,
        );
    }

    let residual_match = value
        .get("readiness_residual_sweep_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_residual)
        .unwrap_or(false);
    if !residual_match {
        mismatch_categories
            .insert(ResidualFreeReadinessMismatchCategoryV1::HistoricalReadinessPathPresent);
    }

    let authority_match = value
        .get("residual_free_readiness_authority_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_authority)
        .unwrap_or(false);
    if !authority_match {
        mismatch_categories
            .insert(ResidualFreeReadinessMismatchCategoryV1::HistoricalReadinessPathPresent);
    }

    let status = if mismatch_categories.is_empty() {
        ResidualFreeReadinessConsumerAuthorityStatusV1::Pass
    } else if mismatch_categories
        .contains(&ResidualFreeReadinessMismatchCategoryV1::HistoricalReadinessPathPresent)
    {
        ResidualFreeReadinessConsumerAuthorityStatusV1::LegacyPresent
    } else {
        ResidualFreeReadinessConsumerAuthorityStatusV1::Fail
    };

    Ok(ResidualFreeReadinessConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

#[allow(clippy::too_many_arguments)]
fn derive_authority(
    applied_supported_set_digest_prefix: &str,
    canonical_governance_entry_digest_prefix: &str,
    canonical_readiness_spine_digest_prefix: &str,
    canonical_readiness_authority_digest_prefix: &str,
    final_readiness_consumer_authority_digest_prefix: &str,
    final_readiness_residual_sweep_digest_prefix: &str,
    covered_consumer_count: u16,
    residual_path_count: u16,
    authority_status: ResidualFreeReadinessConsumerAuthorityStatusV1,
) -> ResidualFreeReadinessConsumerAuthorityV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"residual_free_readiness_consumer_authority_v1");
    bytes.extend_from_slice(applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_readiness_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_readiness_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(final_readiness_consumer_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(final_readiness_residual_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", authority_status).as_bytes());

    ResidualFreeReadinessConsumerAuthorityV1 {
        applied_supported_set_digest_prefix: applied_supported_set_digest_prefix.to_string(),
        canonical_governance_entry_digest_prefix: canonical_governance_entry_digest_prefix
            .to_string(),
        canonical_readiness_spine_digest_prefix: canonical_readiness_spine_digest_prefix
            .to_string(),
        canonical_readiness_authority_digest_prefix: canonical_readiness_authority_digest_prefix
            .to_string(),
        final_readiness_consumer_authority_digest_prefix:
            final_readiness_consumer_authority_digest_prefix.to_string(),
        final_readiness_residual_sweep_digest_prefix: final_readiness_residual_sweep_digest_prefix
            .to_string(),
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
    fn residual_free_readiness_authority_digest_is_stable() {
        let first = derive_authority(
            "11".repeat(8).as_str(),
            "22".repeat(8).as_str(),
            "33".repeat(8).as_str(),
            "44".repeat(8).as_str(),
            "55".repeat(8).as_str(),
            "66".repeat(8).as_str(),
            6,
            0,
            ResidualFreeReadinessConsumerAuthorityStatusV1::Pass,
        );
        let second = derive_authority(
            "11".repeat(8).as_str(),
            "22".repeat(8).as_str(),
            "33".repeat(8).as_str(),
            "44".repeat(8).as_str(),
            "55".repeat(8).as_str(),
            "66".repeat(8).as_str(),
            6,
            0,
            ResidualFreeReadinessConsumerAuthorityStatusV1::Pass,
        );
        assert_eq!(first.authority_digest, second.authority_digest);
    }
}
