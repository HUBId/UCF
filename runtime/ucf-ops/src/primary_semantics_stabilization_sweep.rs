use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    final_primary_semantics_sweep, primary_semantics_absolute_sweep,
    primary_semantics_convergence_sweep, primary_semantics_residual_sweep, primary_semantics_sweep,
    primary_semantics_terminal_sweep, primary_semantics_ultimate_sweep,
    require_primary_semantics_stabilization_inputs, residual_free_primary_semantics_sweep,
    OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PrimarySemanticsStabilizationStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PrimarySemanticsStabilizationMismatchCategoryV1 {
    SurfaceSkippedPrimarySemanticsStabilization,
    SurfaceUsedPrimarySemanticsAdapterPath,
    PrimaryBlockingOrderMismatch,
    PrimaryRemediationOrderMismatch,
    CanonicalConditionMappingMismatch,
    PrimarySemanticsAdapterPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsStabilizationSurfaceStatusV1 {
    pub surface: String,
    pub status: PrimarySemanticsStabilizationStatusV1,
    pub mismatch_categories: Vec<PrimarySemanticsStabilizationMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsStabilizationSweepV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
    pub final_primary_semantics_residual_sweep_digest_prefix: String,
    pub residual_free_primary_semantics_consumer_authority_digest_prefix: String,
    pub residual_free_primary_semantics_absolute_sweep_digest_prefix: String,
    pub absolute_final_primary_semantics_terminal_sweep_digest_prefix: String,
    pub terminal_primary_semantics_ultimate_sweep_digest_prefix: String,
    pub primary_semantics_convergence_sweep_digest_prefix: String,
    pub covered_surface_count: u16,
    pub residual_path_count: u16,
    pub stabilization_status: PrimarySemanticsStabilizationStatusV1,
    pub stabilization_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsStabilizationSweepReportV1 {
    pub schema_version: u16,
    pub sweep: PrimarySemanticsStabilizationSweepV1,
    pub surfaces: Vec<PrimarySemanticsStabilizationSurfaceStatusV1>,
}

pub fn primary_semantics_stabilization_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<PrimarySemanticsStabilizationSweepReportV1, OpsError> {
    let primary = primary_semantics_sweep(
        &workdir.join("out/primary_semantics_sweep_v17_stabilization.json"),
    )?;
    let final_sweep = final_primary_semantics_sweep(
        workdir,
        &workdir.join("out/final_primary_semantics_sweep_v17_stabilization.json"),
    )?;
    let residual = primary_semantics_residual_sweep(
        workdir,
        &workdir.join("out/primary_semantics_residual_sweep_v17_stabilization.json"),
    )?;
    let residual_free = residual_free_primary_semantics_sweep(
        workdir,
        &workdir.join("out/residual_free_primary_semantics_sweep_v17_stabilization.json"),
    )?;
    let absolute = primary_semantics_absolute_sweep(
        workdir,
        &workdir.join("out/primary_semantics_absolute_sweep_v17_stabilization.json"),
    )?;
    let terminal = primary_semantics_terminal_sweep(
        workdir,
        &workdir.join("out/primary_semantics_terminal_sweep_v17_stabilization.json"),
    )?;
    let ultimate = primary_semantics_ultimate_sweep(
        workdir,
        &workdir.join("out/primary_semantics_ultimate_sweep_v17_stabilization.json"),
    )?;
    let convergence = primary_semantics_convergence_sweep(
        workdir,
        &workdir.join("out/primary_semantics_convergence_sweep_v17_stabilization.json"),
    )?;

    let authority_ctx = require_primary_semantics_stabilization_inputs(
        None,
        None,
        Some(&primary.authority),
        Some(&final_sweep.authority),
        Some(&residual.sweep),
        Some(&residual_free.authority),
        Some(&absolute.sweep),
        Some(&terminal.sweep),
        Some(&ultimate.sweep),
        Some(&convergence.sweep),
    )?;

    let mut surfaces = vec![
        check_surface(
            "GovernanceUltimateSweep",
            workdir,
            "out/governance_ultimate_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "ReadinessUltimateSweep",
            workdir,
            "out/readiness_ultimate_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "BundleUltimateSweep",
            workdir,
            "out/bundle_ultimate_sweep.json",
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
            "ExportNormalize",
            workdir,
            "out/export_normalize_check.json",
            &authority_ctx,
        )?,
        check_surface(
            "ExportRoundTrip",
            workdir,
            "out/export_roundtrip_check.json",
            &authority_ctx,
        )?,
        check_surface(
            "V17PrepGateHelper",
            workdir,
            "out/v16_gate_report.json",
            &authority_ctx,
        )?,
    ];
    surfaces.sort_by(|a, b| a.surface.cmp(&b.surface));

    let residual_path_count = surfaces
        .iter()
        .filter(|surface| !matches!(surface.status, PrimarySemanticsStabilizationStatusV1::Pass))
        .count() as u16;

    let stabilization_status = if surfaces.iter().any(|s| {
        matches!(
            s.status,
            PrimarySemanticsStabilizationStatusV1::LegacyPresent
        )
    }) {
        PrimarySemanticsStabilizationStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        PrimarySemanticsStabilizationStatusV1::Pass
    } else {
        PrimarySemanticsStabilizationStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        surfaces.len() as u16,
        residual_path_count,
        stabilization_status,
    );
    let report = PrimarySemanticsStabilizationSweepReportV1 {
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
    authority_ctx: &crate::PrimarySemanticsStabilizationInputsV1,
) -> Result<PrimarySemanticsStabilizationSurfaceStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(
            PrimarySemanticsStabilizationMismatchCategoryV1::SurfaceSkippedPrimarySemanticsStabilization,
        );
        return Ok(PrimarySemanticsStabilizationSurfaceStatusV1 {
            surface: surface.to_string(),
            status: PrimarySemanticsStabilizationStatusV1::LegacyPresent,
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
        "final_primary_semantics_consumer_authority_digest_prefix",
        &authority_ctx.final_primary_semantics_consumer_authority_digest_prefix,
    ) {
        mismatch_categories.insert(
            PrimarySemanticsStabilizationMismatchCategoryV1::SurfaceSkippedPrimarySemanticsStabilization,
        );
    }
    if !field_match(
        "final_primary_semantics_residual_sweep_digest_prefix",
        &authority_ctx.final_primary_semantics_residual_sweep_digest_prefix,
    ) {
        mismatch_categories.insert(
            PrimarySemanticsStabilizationMismatchCategoryV1::PrimarySemanticsAdapterPathPresent,
        );
    }
    if !field_match(
        "residual_free_primary_semantics_authority_digest_prefix",
        &authority_ctx.residual_free_primary_semantics_consumer_authority_digest_prefix,
    ) {
        mismatch_categories.insert(
            PrimarySemanticsStabilizationMismatchCategoryV1::SurfaceUsedPrimarySemanticsAdapterPath,
        );
    }
    if !field_match(
        "primary_semantics_absolute_sweep_digest_prefix",
        &authority_ctx.residual_free_primary_semantics_absolute_sweep_digest_prefix,
    ) {
        mismatch_categories.insert(
            PrimarySemanticsStabilizationMismatchCategoryV1::PrimaryRemediationOrderMismatch,
        );
    }
    if !field_match(
        "primary_semantics_terminal_sweep_digest_prefix",
        &authority_ctx.absolute_final_primary_semantics_terminal_sweep_digest_prefix,
    ) {
        mismatch_categories
            .insert(PrimarySemanticsStabilizationMismatchCategoryV1::PrimaryBlockingOrderMismatch);
    }
    if surface != "V17PrepGateHelper"
        && !field_match(
            "primary_semantics_ultimate_sweep_digest_prefix",
            &authority_ctx.terminal_primary_semantics_ultimate_sweep_digest_prefix,
        )
    {
        mismatch_categories.insert(
            PrimarySemanticsStabilizationMismatchCategoryV1::CanonicalConditionMappingMismatch,
        );
    }

    let convergence_present = value
        .get("primary_semantics_convergence_sweep_digest_prefix")
        .and_then(serde_json::Value::as_str)
        .map(|v| v == authority_ctx.primary_semantics_convergence_sweep_digest_prefix)
        .unwrap_or(false);
    if !convergence_present {
        mismatch_categories.insert(
            PrimarySemanticsStabilizationMismatchCategoryV1::PrimarySemanticsAdapterPathPresent,
        );
    }

    let status = if mismatch_categories.is_empty() {
        PrimarySemanticsStabilizationStatusV1::Pass
    } else if mismatch_categories.contains(
        &PrimarySemanticsStabilizationMismatchCategoryV1::PrimarySemanticsAdapterPathPresent,
    ) || mismatch_categories.contains(
        &PrimarySemanticsStabilizationMismatchCategoryV1::SurfaceUsedPrimarySemanticsAdapterPath,
    ) {
        PrimarySemanticsStabilizationStatusV1::LegacyPresent
    } else {
        PrimarySemanticsStabilizationStatusV1::Fail
    };

    Ok(PrimarySemanticsStabilizationSurfaceStatusV1 {
        surface: surface.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::PrimarySemanticsStabilizationInputsV1,
    covered_surface_count: u16,
    residual_path_count: u16,
    stabilization_status: PrimarySemanticsStabilizationStatusV1,
) -> PrimarySemanticsStabilizationSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"primary_semantics_stabilization_sweep_v1");
    bytes.extend_from_slice(ctx.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_readiness_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_bundle_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        ctx.canonical_primary_semantics_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.final_primary_semantics_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.final_primary_semantics_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.residual_free_primary_semantics_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.residual_free_primary_semantics_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.absolute_final_primary_semantics_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.terminal_primary_semantics_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.primary_semantics_convergence_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(covered_surface_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", stabilization_status).as_bytes());

    PrimarySemanticsStabilizationSweepV1 {
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
        absolute_final_primary_semantics_terminal_sweep_digest_prefix: ctx
            .absolute_final_primary_semantics_terminal_sweep_digest_prefix
            .clone(),
        terminal_primary_semantics_ultimate_sweep_digest_prefix: ctx
            .terminal_primary_semantics_ultimate_sweep_digest_prefix
            .clone(),
        primary_semantics_convergence_sweep_digest_prefix: ctx
            .primary_semantics_convergence_sweep_digest_prefix
            .clone(),
        covered_surface_count,
        residual_path_count,
        stabilization_status,
        stabilization_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_semantics_stabilization_digest_stable() {
        let ctx = crate::PrimarySemanticsStabilizationInputsV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "44".repeat(8),
            final_primary_semantics_consumer_authority_digest_prefix: "55".repeat(8),
            final_primary_semantics_residual_sweep_digest_prefix: "66".repeat(8),
            residual_free_primary_semantics_consumer_authority_digest_prefix: "77".repeat(8),
            residual_free_primary_semantics_absolute_sweep_digest_prefix: "88".repeat(8),
            absolute_final_primary_semantics_terminal_sweep_digest_prefix: "99".repeat(8),
            terminal_primary_semantics_ultimate_sweep_digest_prefix: "aa".repeat(8),
            primary_semantics_convergence_sweep_digest_prefix: "bb".repeat(8),
            authority_digest: "cc".repeat(32),
        };
        let first = derive_sweep(&ctx, 10, 0, PrimarySemanticsStabilizationStatusV1::Pass);
        let second = derive_sweep(&ctx, 10, 0, PrimarySemanticsStabilizationStatusV1::Pass);
        assert_eq!(first.stabilization_digest, second.stabilization_digest);
    }
}
