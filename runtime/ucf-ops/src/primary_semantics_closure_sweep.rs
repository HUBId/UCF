use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    final_primary_semantics_sweep, primary_semantics_absolute_sweep,
    primary_semantics_convergence_sweep, primary_semantics_final_consolidation_sweep,
    primary_semantics_residual_sweep, primary_semantics_stabilization_sweep,
    primary_semantics_sweep, primary_semantics_terminal_sweep, primary_semantics_ultimate_sweep,
    require_primary_semantics_closure_inputs, residual_free_primary_semantics_sweep, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PrimarySemanticsClosureStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PrimarySemanticsClosureMismatchCategoryV1 {
    SurfaceSkippedPrimarySemanticsClosure,
    SurfaceUsedPrimarySemanticsWrapperPath,
    PrimaryBlockingOrderMismatch,
    PrimaryRemediationOrderMismatch,
    CanonicalConditionMappingMismatch,
    PrimarySemanticsWrapperPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsClosureSurfaceStatusV1 {
    pub surface: String,
    pub status: PrimarySemanticsClosureStatusV1,
    pub mismatch_categories: Vec<PrimarySemanticsClosureMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsClosureSweepV1 {
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
    pub primary_semantics_stabilization_sweep_digest_prefix: String,
    pub primary_semantics_final_consolidation_sweep_digest_prefix: String,
    pub covered_surface_count: u16,
    pub residual_path_count: u16,
    pub closure_status: PrimarySemanticsClosureStatusV1,
    pub closure_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrimarySemanticsClosureSweepReportV1 {
    pub schema_version: u16,
    pub sweep: PrimarySemanticsClosureSweepV1,
    pub surfaces: Vec<PrimarySemanticsClosureSurfaceStatusV1>,
}

pub fn primary_semantics_closure_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<PrimarySemanticsClosureSweepReportV1, OpsError> {
    let primary =
        primary_semantics_sweep(&workdir.join("out/primary_semantics_sweep_v19_closure.json"))?;
    let final_sweep = final_primary_semantics_sweep(
        workdir,
        &workdir.join("out/final_primary_semantics_sweep_v19_closure.json"),
    )?;
    let residual = primary_semantics_residual_sweep(
        workdir,
        &workdir.join("out/primary_semantics_residual_sweep_v19_closure.json"),
    )?;
    let residual_free = residual_free_primary_semantics_sweep(
        workdir,
        &workdir.join("out/residual_free_primary_semantics_sweep_v19_closure.json"),
    )?;
    let absolute = primary_semantics_absolute_sweep(
        workdir,
        &workdir.join("out/primary_semantics_absolute_sweep_v19_closure.json"),
    )?;
    let terminal = primary_semantics_terminal_sweep(
        workdir,
        &workdir.join("out/primary_semantics_terminal_sweep_v19_closure.json"),
    )?;
    let ultimate = primary_semantics_ultimate_sweep(
        workdir,
        &workdir.join("out/primary_semantics_ultimate_sweep_v19_closure.json"),
    )?;
    let convergence = primary_semantics_convergence_sweep(
        workdir,
        &workdir.join("out/primary_semantics_convergence_sweep_v19_closure.json"),
    )?;
    let stabilization = primary_semantics_stabilization_sweep(
        workdir,
        &workdir.join("out/primary_semantics_stabilization_sweep_v19_closure.json"),
    )?;
    let final_consolidation = primary_semantics_final_consolidation_sweep(
        workdir,
        &workdir.join("out/primary_semantics_final_consolidation_sweep_v19_closure.json"),
    )?;

    let authority_ctx = require_primary_semantics_closure_inputs(
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
        Some(&stabilization.sweep),
        Some(&final_consolidation.sweep),
    )?;

    let mut surfaces = vec![
        check_surface(
            "GovernanceClosureSweep",
            workdir,
            "out/governance_closure_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "ReadinessClosureSweep",
            workdir,
            "out/readiness_closure_sweep.json",
            &authority_ctx,
        )?,
        check_surface(
            "BundleClosureSweep",
            workdir,
            "out/bundle_closure_sweep.json",
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
            "V19PrepGateHelper",
            workdir,
            "out/v18_gate_report.json",
            &authority_ctx,
        )?,
    ];
    surfaces.sort_by(|a, b| a.surface.cmp(&b.surface));

    let residual_path_count = surfaces
        .iter()
        .filter(|surface| !matches!(surface.status, PrimarySemanticsClosureStatusV1::Pass))
        .count() as u16;

    let closure_status = if surfaces
        .iter()
        .any(|s| matches!(s.status, PrimarySemanticsClosureStatusV1::LegacyPresent))
    {
        PrimarySemanticsClosureStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        PrimarySemanticsClosureStatusV1::Pass
    } else {
        PrimarySemanticsClosureStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        surfaces.len() as u16,
        residual_path_count,
        closure_status,
    );
    let report = PrimarySemanticsClosureSweepReportV1 {
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
    authority_ctx: &crate::PrimarySemanticsClosureInputsV1,
) -> Result<PrimarySemanticsClosureSurfaceStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(
            PrimarySemanticsClosureMismatchCategoryV1::SurfaceSkippedPrimarySemanticsClosure,
        );
        return Ok(PrimarySemanticsClosureSurfaceStatusV1 {
            surface: surface.to_string(),
            status: PrimarySemanticsClosureStatusV1::LegacyPresent,
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
        "primary_semantics_final_consolidation_sweep_digest_prefix",
        &authority_ctx.primary_semantics_final_consolidation_sweep_digest_prefix,
    ) {
        mismatch_categories.insert(
            PrimarySemanticsClosureMismatchCategoryV1::SurfaceSkippedPrimarySemanticsClosure,
        );
    }
    if !field_match(
        "primary_semantics_stabilization_sweep_digest_prefix",
        &authority_ctx.primary_semantics_stabilization_sweep_digest_prefix,
    ) {
        mismatch_categories
            .insert(PrimarySemanticsClosureMismatchCategoryV1::PrimarySemanticsWrapperPathPresent);
    }
    if !field_match(
        "primary_semantics_convergence_sweep_digest_prefix",
        &authority_ctx.primary_semantics_convergence_sweep_digest_prefix,
    ) {
        mismatch_categories.insert(
            PrimarySemanticsClosureMismatchCategoryV1::SurfaceUsedPrimarySemanticsWrapperPath,
        );
    }
    if !field_match(
        "primary_semantics_ultimate_sweep_digest_prefix",
        &authority_ctx.terminal_primary_semantics_ultimate_sweep_digest_prefix,
    ) {
        mismatch_categories
            .insert(PrimarySemanticsClosureMismatchCategoryV1::PrimaryBlockingOrderMismatch);
    }
    if !field_match(
        "primary_semantics_absolute_sweep_digest_prefix",
        &authority_ctx.residual_free_primary_semantics_absolute_sweep_digest_prefix,
    ) {
        mismatch_categories
            .insert(PrimarySemanticsClosureMismatchCategoryV1::PrimaryRemediationOrderMismatch);
    }
    if !field_match(
        "final_primary_semantics_residual_sweep_digest_prefix",
        &authority_ctx.final_primary_semantics_residual_sweep_digest_prefix,
    ) {
        mismatch_categories
            .insert(PrimarySemanticsClosureMismatchCategoryV1::CanonicalConditionMappingMismatch);
    }

    let status = if mismatch_categories.is_empty() {
        PrimarySemanticsClosureStatusV1::Pass
    } else if mismatch_categories
        .contains(&PrimarySemanticsClosureMismatchCategoryV1::PrimarySemanticsWrapperPathPresent)
        || mismatch_categories.contains(
            &PrimarySemanticsClosureMismatchCategoryV1::SurfaceUsedPrimarySemanticsWrapperPath,
        )
    {
        PrimarySemanticsClosureStatusV1::LegacyPresent
    } else {
        PrimarySemanticsClosureStatusV1::Fail
    };

    Ok(PrimarySemanticsClosureSurfaceStatusV1 {
        surface: surface.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::PrimarySemanticsClosureInputsV1,
    covered_surface_count: u16,
    residual_path_count: u16,
    closure_status: PrimarySemanticsClosureStatusV1,
) -> PrimarySemanticsClosureSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"primary_semantics_closure_sweep_v1");
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
    bytes.extend_from_slice(
        ctx.primary_semantics_stabilization_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.primary_semantics_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(covered_surface_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", closure_status).as_bytes());

    PrimarySemanticsClosureSweepV1 {
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
        primary_semantics_stabilization_sweep_digest_prefix: ctx
            .primary_semantics_stabilization_sweep_digest_prefix
            .clone(),
        primary_semantics_final_consolidation_sweep_digest_prefix: ctx
            .primary_semantics_final_consolidation_sweep_digest_prefix
            .clone(),
        covered_surface_count,
        residual_path_count,
        closure_status,
        closure_digest: crate::sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_semantics_closure_digest_stable() {
        let ctx = crate::PrimarySemanticsClosureInputsV1 {
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
            primary_semantics_stabilization_sweep_digest_prefix: "cc".repeat(8),
            primary_semantics_final_consolidation_sweep_digest_prefix: "dd".repeat(8),
            authority_digest: "ee".repeat(32),
        };

        let first = derive_sweep(&ctx, 8, 0, PrimarySemanticsClosureStatusV1::Pass);
        let second = derive_sweep(&ctx, 8, 0, PrimarySemanticsClosureStatusV1::Pass);
        assert_eq!(first.closure_digest, second.closure_digest);
    }
}
