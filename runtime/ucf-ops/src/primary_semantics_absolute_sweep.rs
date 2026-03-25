use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    final_primary_semantics_sweep, prefix_hex, primary_semantics_residual_sweep,
    primary_semantics_sweep, require_residual_free_primary_semantics_absolute_inputs,
    residual_free_primary_semantics_sweep, OpsError,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreePrimarySemanticsAbsoluteSweepStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PrimarySemanticsAbsoluteMismatchCategoryV1 {
    SurfaceSkippedAbsolutePrimarySemanticsInputs,
    SurfaceUsedHistoricalPrimarySemanticsLineage,
    PrimaryBlockingOrderMismatch,
    PrimaryRemediationOrderMismatch,
    CanonicalConditionMappingMismatch,
    HistoricalPrimarySemanticsLineagePresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsAbsoluteSurfaceStatusV1 {
    pub surface: String,
    pub status: ResidualFreePrimarySemanticsAbsoluteSweepStatusV1,
    pub mismatch_categories: Vec<PrimarySemanticsAbsoluteMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreePrimarySemanticsAbsoluteSweepV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
    pub final_primary_semantics_residual_sweep_digest_prefix: String,
    pub residual_free_primary_semantics_consumer_authority_digest_prefix: String,
    pub covered_surface_count: u16,
    pub residual_path_count: u16,
    pub sweep_status: ResidualFreePrimarySemanticsAbsoluteSweepStatusV1,
    pub sweep_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsAbsoluteSweepReportV1 {
    pub schema_version: u16,
    pub sweep: ResidualFreePrimarySemanticsAbsoluteSweepV1,
    pub surfaces: Vec<PrimarySemanticsAbsoluteSurfaceStatusV1>,
}

pub fn primary_semantics_absolute_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<PrimarySemanticsAbsoluteSweepReportV1, OpsError> {
    let primary = primary_semantics_sweep(
        &workdir.join("out/primary_semantics_sweep_v13_absolute_sweep.json"),
    )?;
    let final_sweep = final_primary_semantics_sweep(
        workdir,
        &workdir.join("out/final_primary_semantics_sweep_v13_absolute_sweep.json"),
    )?;
    let residual = primary_semantics_residual_sweep(
        workdir,
        &workdir.join("out/primary_semantics_residual_sweep_v13_absolute_sweep.json"),
    )?;
    let residual_free = residual_free_primary_semantics_sweep(
        workdir,
        &workdir.join("out/residual_free_primary_semantics_sweep_v13_absolute_sweep.json"),
    )?;
    let authority_ctx = require_residual_free_primary_semantics_absolute_inputs(
        None,
        None,
        Some(&primary.authority),
        Some(&final_sweep.authority),
        Some(&residual.sweep),
        Some(&residual_free.authority),
    )?;

    let mut surfaces = vec![
        check_surface(
            "GovernanceAbsoluteSweep",
            workdir,
            "out/governance_absolute_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "ReadinessAbsoluteSweep",
            workdir,
            "out/readiness_absolute_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "BundleAbsoluteSweep",
            workdir,
            "out/bundle_absolute_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "OperatorSignoff",
            workdir,
            "out/operator_signoff.json",
            &authority_ctx,
        )?,
        check_surface(
            "OperatorReviewPacket",
            workdir,
            "out/operator_review_packet.json",
            &authority_ctx,
        )?,
        check_surface(
            "OperatorWorkflowChain",
            workdir,
            "out/operator_workflow_chain.json",
            &authority_ctx,
        )?,
        check_surface(
            "InteropConsistencyMatrix",
            workdir,
            "out/interop_consistency_matrix.json",
            &authority_ctx,
        )?,
        check_surface(
            "V13PrepGateHelper",
            workdir,
            "out/v12_gate_report.json",
            &authority_ctx,
        )?,
    ];
    surfaces.sort_by(|a, b| a.surface.cmp(&b.surface));

    let residual_path_count = surfaces
        .iter()
        .filter(|surface| {
            !matches!(
                surface.status,
                ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::Pass
            )
        })
        .count() as u16;

    let sweep_status = if surfaces.iter().any(|surface| {
        matches!(
            surface.status,
            ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::LegacyPresent
        )
    }) {
        ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::Pass
    } else {
        ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        surfaces.len() as u16,
        residual_path_count,
        sweep_status,
    );
    let report = PrimarySemanticsAbsoluteSweepReportV1 {
        schema_version: 1,
        sweep,
        surfaces,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn check_surface(
    surface: &str,
    workdir: &Path,
    rel_path: &str,
    authority_ctx: &crate::ResidualFreePrimarySemanticsAbsoluteInputsV1,
) -> Result<PrimarySemanticsAbsoluteSurfaceStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(
            PrimarySemanticsAbsoluteMismatchCategoryV1::SurfaceSkippedAbsolutePrimarySemanticsInputs,
        );
        mismatch_categories.insert(
            PrimarySemanticsAbsoluteMismatchCategoryV1::HistoricalPrimarySemanticsLineagePresent,
        );
        return Ok(PrimarySemanticsAbsoluteSurfaceStatusV1 {
            surface: surface.to_string(),
            status: ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::LegacyPresent,
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let expected_final = &authority_ctx.final_primary_semantics_consumer_authority_digest_prefix;
    let expected_residual = &authority_ctx.final_primary_semantics_residual_sweep_digest_prefix;
    let expected_residual_free =
        &authority_ctx.residual_free_primary_semantics_consumer_authority_digest_prefix;
    let expected_absolute = prefix_hex(&authority_ctx.authority_digest, DIGEST_PREFIX_LEN);

    let final_match = value
        .get("final_primary_semantics_consumer_authority_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_final)
        .unwrap_or(false);
    if !final_match {
        mismatch_categories.insert(
            PrimarySemanticsAbsoluteMismatchCategoryV1::SurfaceSkippedAbsolutePrimarySemanticsInputs,
        );
    }

    let residual_match = value
        .get("final_primary_semantics_residual_sweep_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_residual)
        .unwrap_or(false);
    if !residual_match {
        mismatch_categories.insert(
            PrimarySemanticsAbsoluteMismatchCategoryV1::HistoricalPrimarySemanticsLineagePresent,
        );
    }

    let residual_free_match = value
        .get("residual_free_primary_semantics_authority_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_residual_free)
        .unwrap_or(false);
    if !residual_free_match {
        mismatch_categories.insert(
            PrimarySemanticsAbsoluteMismatchCategoryV1::SurfaceUsedHistoricalPrimarySemanticsLineage,
        );
    }

    let absolute_match = value
        .get("primary_semantics_absolute_sweep_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|s| s == expected_absolute)
        .unwrap_or(false);
    if !absolute_match {
        mismatch_categories.insert(
            PrimarySemanticsAbsoluteMismatchCategoryV1::SurfaceSkippedAbsolutePrimarySemanticsInputs,
        );
    }

    let status = if mismatch_categories.contains(
        &PrimarySemanticsAbsoluteMismatchCategoryV1::HistoricalPrimarySemanticsLineagePresent,
    ) {
        ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::LegacyPresent
    } else if mismatch_categories.is_empty() {
        ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::Pass
    } else {
        ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::Fail
    };

    Ok(PrimarySemanticsAbsoluteSurfaceStatusV1 {
        surface: surface.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::ResidualFreePrimarySemanticsAbsoluteInputsV1,
    covered_surface_count: u16,
    residual_path_count: u16,
    sweep_status: ResidualFreePrimarySemanticsAbsoluteSweepStatusV1,
) -> ResidualFreePrimarySemanticsAbsoluteSweepV1 {
    let payload = serde_json::to_vec(&(
        &ctx.canonical_governance_entry_digest_prefix,
        &ctx.canonical_readiness_spine_digest_prefix,
        &ctx.canonical_bundle_spine_digest_prefix,
        &ctx.canonical_primary_semantics_authority_digest_prefix,
        &ctx.final_primary_semantics_consumer_authority_digest_prefix,
        &ctx.final_primary_semantics_residual_sweep_digest_prefix,
        &ctx.residual_free_primary_semantics_consumer_authority_digest_prefix,
        covered_surface_count,
        residual_path_count,
        &sweep_status,
    ))
    .expect("serializing primary semantics absolute sweep");

    ResidualFreePrimarySemanticsAbsoluteSweepV1 {
        canonical_governance_entry_digest_prefix: ctx
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: ctx
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_bundle_spine_digest_prefix: ctx.canonical_bundle_spine_digest_prefix.clone(),
        canonical_primary_semantics_authority_digest_prefix: ctx
            .canonical_primary_semantics_authority_digest_prefix
            .clone(),
        final_primary_semantics_consumer_authority_digest_prefix: ctx
            .final_primary_semantics_consumer_authority_digest_prefix
            .clone(),
        final_primary_semantics_residual_sweep_digest_prefix: ctx
            .final_primary_semantics_residual_sweep_digest_prefix
            .clone(),
        residual_free_primary_semantics_consumer_authority_digest_prefix: ctx
            .residual_free_primary_semantics_consumer_authority_digest_prefix
            .clone(),
        covered_surface_count,
        residual_path_count,
        sweep_status,
        sweep_digest: crate::sha256_hex(&payload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_digest_is_stable() {
        let ctx = crate::ResidualFreePrimarySemanticsAbsoluteInputsV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "44".repeat(8),
            final_primary_semantics_consumer_authority_digest_prefix: "55".repeat(8),
            final_primary_semantics_residual_sweep_digest_prefix: "66".repeat(8),
            residual_free_primary_semantics_consumer_authority_digest_prefix: "77".repeat(8),
            authority_digest: "88".repeat(32),
        };
        let a = derive_sweep(
            &ctx,
            7,
            0,
            ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::Pass,
        );
        let b = derive_sweep(
            &ctx,
            7,
            0,
            ResidualFreePrimarySemanticsAbsoluteSweepStatusV1::Pass,
        );
        assert_eq!(a.sweep_digest, b.sweep_digest);
    }
}
