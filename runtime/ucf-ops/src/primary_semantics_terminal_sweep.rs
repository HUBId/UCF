use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    final_primary_semantics_sweep, prefix_hex, primary_semantics_absolute_sweep,
    primary_semantics_residual_sweep, primary_semantics_sweep,
    require_absolute_final_primary_semantics_terminal_inputs,
    residual_free_primary_semantics_sweep, OpsError,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PrimarySemanticsTerminalMismatchCategoryV1 {
    SurfaceSkippedTerminalPrimarySemanticsInputs,
    SurfaceUsedPrimarySemanticsEchoPath,
    PrimaryBlockingOrderMismatch,
    PrimaryRemediationOrderMismatch,
    CanonicalConditionMappingMismatch,
    PrimarySemanticsEchoPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsTerminalSurfaceStatusV1 {
    pub surface: String,
    pub status: AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1,
    pub mismatch_categories: Vec<PrimarySemanticsTerminalMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AbsoluteFinalPrimarySemanticsTerminalSweepV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
    pub final_primary_semantics_residual_sweep_digest_prefix: String,
    pub residual_free_primary_semantics_consumer_authority_digest_prefix: String,
    pub residual_free_primary_semantics_absolute_sweep_digest_prefix: String,
    pub covered_surface_count: u16,
    pub residual_path_count: u16,
    pub sweep_status: AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1,
    pub sweep_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsTerminalSweepReportV1 {
    pub schema_version: u16,
    pub sweep: AbsoluteFinalPrimarySemanticsTerminalSweepV1,
    pub surfaces: Vec<PrimarySemanticsTerminalSurfaceStatusV1>,
}

pub fn primary_semantics_terminal_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<PrimarySemanticsTerminalSweepReportV1, OpsError> {
    let primary =
        primary_semantics_sweep(&workdir.join("out/primary_semantics_sweep_v14_terminal.json"))?;
    let final_sweep = final_primary_semantics_sweep(
        workdir,
        &workdir.join("out/final_primary_semantics_sweep_v14_terminal.json"),
    )?;
    let residual = primary_semantics_residual_sweep(
        workdir,
        &workdir.join("out/primary_semantics_residual_sweep_v14_terminal.json"),
    )?;
    let residual_free = residual_free_primary_semantics_sweep(
        workdir,
        &workdir.join("out/residual_free_primary_semantics_sweep_v14_terminal.json"),
    )?;
    let absolute = primary_semantics_absolute_sweep(
        workdir,
        &workdir.join("out/primary_semantics_absolute_sweep_v14_terminal.json"),
    )?;

    let authority_ctx = require_absolute_final_primary_semantics_terminal_inputs(
        None,
        None,
        Some(&primary.authority),
        Some(&final_sweep.authority),
        Some(&residual.sweep),
        Some(&residual_free.authority),
        Some(&absolute.sweep),
    )?;

    let mut surfaces = vec![
        check_surface(
            "GovernanceTerminalSweep",
            workdir,
            "out/governance_terminal_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "ReadinessTerminalSweep",
            workdir,
            "out/readiness_terminal_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "BundleTerminalSweep",
            workdir,
            "out/bundle_terminal_sweep.json",
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
            "V14PrepGateHelper",
            workdir,
            "out/v13_gate_report.json",
            &authority_ctx,
        )?,
    ];
    surfaces.sort_by(|a, b| a.surface.cmp(&b.surface));

    let residual_path_count = surfaces
        .iter()
        .filter(|surface| {
            !matches!(
                surface.status,
                AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::Pass
            )
        })
        .count() as u16;

    let sweep_status = if surfaces.iter().any(|surface| {
        matches!(
            surface.status,
            AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::LegacyPresent
        )
    }) {
        AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::Pass
    } else {
        AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        surfaces.len() as u16,
        residual_path_count,
        sweep_status,
    );
    let report = PrimarySemanticsTerminalSweepReportV1 {
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
    authority_ctx: &crate::AbsoluteFinalPrimarySemanticsTerminalInputsV1,
) -> Result<PrimarySemanticsTerminalSurfaceStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(
            PrimarySemanticsTerminalMismatchCategoryV1::SurfaceSkippedTerminalPrimarySemanticsInputs,
        );
        mismatch_categories
            .insert(PrimarySemanticsTerminalMismatchCategoryV1::PrimarySemanticsEchoPathPresent);
        return Ok(PrimarySemanticsTerminalSurfaceStatusV1 {
            surface: surface.to_string(),
            status: AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::LegacyPresent,
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let expected_final = &authority_ctx.final_primary_semantics_consumer_authority_digest_prefix;
    let expected_residual = &authority_ctx.final_primary_semantics_residual_sweep_digest_prefix;
    let expected_residual_free =
        &authority_ctx.residual_free_primary_semantics_consumer_authority_digest_prefix;
    let expected_absolute =
        &authority_ctx.residual_free_primary_semantics_absolute_sweep_digest_prefix;
    let expected_terminal = prefix_hex(&authority_ctx.authority_digest, DIGEST_PREFIX_LEN);

    let field_match = |field: &str, expected: &str| {
        value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .map(|s| s == expected)
            .unwrap_or(false)
    };

    if !field_match(
        "final_primary_semantics_consumer_authority_digest_prefix",
        expected_final,
    ) {
        mismatch_categories.insert(
            PrimarySemanticsTerminalMismatchCategoryV1::SurfaceSkippedTerminalPrimarySemanticsInputs,
        );
    }
    if !field_match(
        "final_primary_semantics_residual_sweep_digest_prefix",
        expected_residual,
    ) {
        mismatch_categories
            .insert(PrimarySemanticsTerminalMismatchCategoryV1::PrimarySemanticsEchoPathPresent);
    }
    if !field_match(
        "residual_free_primary_semantics_authority_digest_prefix",
        expected_residual_free,
    ) {
        mismatch_categories.insert(
            PrimarySemanticsTerminalMismatchCategoryV1::SurfaceUsedPrimarySemanticsEchoPath,
        );
    }
    if !field_match(
        "primary_semantics_absolute_sweep_digest_prefix",
        expected_absolute,
    ) {
        mismatch_categories
            .insert(PrimarySemanticsTerminalMismatchCategoryV1::PrimarySemanticsEchoPathPresent);
    }
    if surface != "V14PrepGateHelper"
        && !field_match(
            "primary_semantics_terminal_sweep_digest_prefix",
            &expected_terminal,
        )
    {
        mismatch_categories.insert(
            PrimarySemanticsTerminalMismatchCategoryV1::SurfaceSkippedTerminalPrimarySemanticsInputs,
        );
    }

    let status = if mismatch_categories
        .contains(&PrimarySemanticsTerminalMismatchCategoryV1::PrimarySemanticsEchoPathPresent)
    {
        AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::LegacyPresent
    } else if mismatch_categories.is_empty() {
        AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::Pass
    } else {
        AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::Fail
    };

    Ok(PrimarySemanticsTerminalSurfaceStatusV1 {
        surface: surface.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::AbsoluteFinalPrimarySemanticsTerminalInputsV1,
    covered_surface_count: u16,
    residual_path_count: u16,
    sweep_status: AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1,
) -> AbsoluteFinalPrimarySemanticsTerminalSweepV1 {
    let payload = serde_json::to_vec(&(
        &ctx.canonical_governance_entry_digest_prefix,
        &ctx.canonical_readiness_spine_digest_prefix,
        &ctx.canonical_bundle_spine_digest_prefix,
        &ctx.canonical_primary_semantics_authority_digest_prefix,
        &ctx.final_primary_semantics_consumer_authority_digest_prefix,
        &ctx.final_primary_semantics_residual_sweep_digest_prefix,
        &ctx.residual_free_primary_semantics_consumer_authority_digest_prefix,
        &ctx.residual_free_primary_semantics_absolute_sweep_digest_prefix,
        covered_surface_count,
        residual_path_count,
        &sweep_status,
    ))
    .expect("serializing primary semantics terminal sweep");

    AbsoluteFinalPrimarySemanticsTerminalSweepV1 {
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
        residual_free_primary_semantics_absolute_sweep_digest_prefix: ctx
            .residual_free_primary_semantics_absolute_sweep_digest_prefix
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
        let ctx = crate::AbsoluteFinalPrimarySemanticsTerminalInputsV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "44".repeat(8),
            final_primary_semantics_consumer_authority_digest_prefix: "55".repeat(8),
            final_primary_semantics_residual_sweep_digest_prefix: "66".repeat(8),
            residual_free_primary_semantics_consumer_authority_digest_prefix: "77".repeat(8),
            residual_free_primary_semantics_absolute_sweep_digest_prefix: "88".repeat(8),
            authority_digest: "99".repeat(32),
        };
        let a = derive_sweep(
            &ctx,
            8,
            0,
            AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::Pass,
        );
        let b = derive_sweep(
            &ctx,
            8,
            0,
            AbsoluteFinalPrimarySemanticsTerminalSweepStatusV1::Pass,
        );
        assert_eq!(a.sweep_digest, b.sweep_digest);
    }
}
