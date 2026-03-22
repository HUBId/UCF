use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    final_primary_semantics_sweep, prefix_hex, primary_semantics_residual_sweep,
    primary_semantics_sweep, require_residual_free_final_primary_semantics_inputs,
    CrossSurfaceObservationStatusV1, FinalPrimarySemanticsConsumerAuthorityStatusV1,
    FinalPrimarySemanticsResidualSweepStatusV1, OpsError,
};

const SCHEMA_VERSION: u16 = 1;
const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreePrimarySemanticsAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreePrimarySemanticsMismatchCategoryV1 {
    SurfaceSkippedResidualFreeFinalPrimarySemanticsInputs,
    SurfaceUsedHistoricalPrimarySemanticsPath,
    PrimaryBlockingOrderMismatch,
    PrimaryRemediationOrderMismatch,
    CanonicalConditionMappingMismatch,
    HistoricalPrimarySemanticsPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreePrimarySemanticsSurfaceStatusV1 {
    pub surface_kind: String,
    pub status: CrossSurfaceObservationStatusV1,
    pub mismatch_categories: Vec<ResidualFreePrimarySemanticsMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreePrimarySemanticsConsumerAuthorityV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
    pub final_primary_semantics_residual_sweep_digest_prefix: String,
    pub covered_surface_count: u16,
    pub residual_path_count: u16,
    pub authority_status: ResidualFreePrimarySemanticsAuthorityStatusV1,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreePrimarySemanticsSweepReportV1 {
    pub schema_version: u16,
    pub authority: ResidualFreePrimarySemanticsConsumerAuthorityV1,
    pub surfaces: Vec<ResidualFreePrimarySemanticsSurfaceStatusV1>,
}

pub fn residual_free_primary_semantics_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<ResidualFreePrimarySemanticsSweepReportV1, OpsError> {
    let primary = primary_semantics_sweep(
        &workdir.join("out/primary_semantics_sweep_v12_residual_free.json"),
    )?;
    let final_sweep = final_primary_semantics_sweep(
        workdir,
        &workdir.join("out/final_primary_semantics_sweep_v12_residual_free.json"),
    )?;
    let residual_sweep = primary_semantics_residual_sweep(
        workdir,
        &workdir.join("out/primary_semantics_residual_sweep_v12_residual_free.json"),
    )?;
    let final_inputs = require_residual_free_final_primary_semantics_inputs(
        None,
        None,
        Some(&primary.authority),
        Some(&final_sweep.authority),
        Some(&residual_sweep.sweep),
    )?;

    let mut surfaces = final_sweep
        .surface_statuses
        .iter()
        .map(|surface| {
            let mut categories = BTreeSet::new();
            for mismatch in &surface.mismatch_categories {
                match mismatch.as_str() {
                    "SURFACE_SKIPPED_FINAL_PRIMARY_SEMANTICS_AUTHORITY" => {
                        categories.insert(
                            ResidualFreePrimarySemanticsMismatchCategoryV1::SurfaceSkippedResidualFreeFinalPrimarySemanticsInputs,
                        );
                    }
                    "SURFACE_USED_LEGACY_PRIMARY_SEMANTICS_INPUT" => {
                        categories.insert(
                            ResidualFreePrimarySemanticsMismatchCategoryV1::SurfaceUsedHistoricalPrimarySemanticsPath,
                        );
                        categories.insert(
                            ResidualFreePrimarySemanticsMismatchCategoryV1::HistoricalPrimarySemanticsPathPresent,
                        );
                    }
                    "PRIMARY_BLOCKING_MISMATCH" => {
                        categories.insert(
                            ResidualFreePrimarySemanticsMismatchCategoryV1::PrimaryBlockingOrderMismatch,
                        );
                    }
                    "PRIMARY_REMEDIATION_MISMATCH" => {
                        categories.insert(
                            ResidualFreePrimarySemanticsMismatchCategoryV1::PrimaryRemediationOrderMismatch,
                        );
                    }
                    "CANONICAL_CONDITION_MISMATCH" => {
                        categories.insert(
                            ResidualFreePrimarySemanticsMismatchCategoryV1::CanonicalConditionMappingMismatch,
                        );
                    }
                    _ => {}
                }
            }
            let status = if categories.is_empty() {
                CrossSurfaceObservationStatusV1::Pass
            } else {
                CrossSurfaceObservationStatusV1::Fail
            };
            ResidualFreePrimarySemanticsSurfaceStatusV1 {
                surface_kind: surface.surface_kind.clone(),
                status,
                mismatch_categories: categories.into_iter().collect(),
            }
        })
        .collect::<Vec<_>>();
    surfaces.sort_by(|a, b| a.surface_kind.cmp(&b.surface_kind));

    if !matches!(
        residual_sweep.sweep.sweep_status,
        FinalPrimarySemanticsResidualSweepStatusV1::Pass
    ) || !matches!(
        final_sweep.authority.authority_status,
        FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass
    ) {
        for surface in &mut surfaces {
            if !surface.mismatch_categories.contains(
                &ResidualFreePrimarySemanticsMismatchCategoryV1::HistoricalPrimarySemanticsPathPresent,
            ) {
                surface
                    .mismatch_categories
                    .push(ResidualFreePrimarySemanticsMismatchCategoryV1::HistoricalPrimarySemanticsPathPresent);
                surface.status = CrossSurfaceObservationStatusV1::Fail;
            }
        }
    }

    let residual_path_count = surfaces
        .iter()
        .filter(|surface| {
            surface
                .mismatch_categories
                .contains(&ResidualFreePrimarySemanticsMismatchCategoryV1::HistoricalPrimarySemanticsPathPresent)
                || !matches!(surface.status, CrossSurfaceObservationStatusV1::Pass)
        })
        .count() as u16;
    let authority_status = if residual_path_count == 0 {
        ResidualFreePrimarySemanticsAuthorityStatusV1::Pass
    } else if surfaces.iter().any(|surface| {
        surface.mismatch_categories.contains(
            &ResidualFreePrimarySemanticsMismatchCategoryV1::HistoricalPrimarySemanticsPathPresent,
        )
    }) {
        ResidualFreePrimarySemanticsAuthorityStatusV1::LegacyPresent
    } else {
        ResidualFreePrimarySemanticsAuthorityStatusV1::Fail
    };

    let digest_source = serde_json::to_vec(&(
        &final_inputs.canonical_governance_entry_digest_prefix,
        &final_inputs.canonical_readiness_spine_digest_prefix,
        &final_inputs.canonical_bundle_spine_digest_prefix,
        &final_inputs.canonical_primary_semantics_authority_digest_prefix,
        &final_inputs.final_primary_semantics_consumer_authority_digest_prefix,
        &final_inputs.final_primary_semantics_residual_sweep_digest_prefix,
        surfaces.len() as u16,
        residual_path_count,
        &authority_status,
        &surfaces,
    ))?;

    let authority = ResidualFreePrimarySemanticsConsumerAuthorityV1 {
        canonical_governance_entry_digest_prefix: final_inputs
            .canonical_governance_entry_digest_prefix,
        canonical_readiness_spine_digest_prefix: final_inputs
            .canonical_readiness_spine_digest_prefix,
        canonical_bundle_spine_digest_prefix: final_inputs.canonical_bundle_spine_digest_prefix,
        canonical_primary_semantics_authority_digest_prefix: final_inputs
            .canonical_primary_semantics_authority_digest_prefix,
        final_primary_semantics_consumer_authority_digest_prefix: final_inputs
            .final_primary_semantics_consumer_authority_digest_prefix,
        final_primary_semantics_residual_sweep_digest_prefix: final_inputs
            .final_primary_semantics_residual_sweep_digest_prefix,
        covered_surface_count: surfaces.len() as u16,
        residual_path_count,
        authority_status,
        authority_digest: crate::sha256_hex(&digest_source),
    };

    let report = ResidualFreePrimarySemanticsSweepReportV1 {
        schema_version: SCHEMA_VERSION,
        authority,
        surfaces,
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn residual_free_primary_semantics_authority_digest_prefix(workdir: &Path) -> String {
    let path = workdir.join("out/residual_free_primary_semantics_sweep.json");
    let Ok(bytes) = fs::read(path) else {
        return "MISSING".to_string();
    };
    let Ok(report) = serde_json::from_slice::<ResidualFreePrimarySemanticsSweepReportV1>(&bytes)
    else {
        return "MISSING".to_string();
    };
    prefix_hex(&report.authority.authority_digest, DIGEST_PREFIX_LEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_digest_is_stable() {
        let a = ResidualFreePrimarySemanticsConsumerAuthorityV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "44".repeat(8),
            final_primary_semantics_consumer_authority_digest_prefix: "55".repeat(8),
            final_primary_semantics_residual_sweep_digest_prefix: "66".repeat(8),
            covered_surface_count: 9,
            residual_path_count: 0,
            authority_status: ResidualFreePrimarySemanticsAuthorityStatusV1::Pass,
            authority_digest: "aa".repeat(32),
        };
        let left = serde_json::to_vec(&a).expect("serialize");
        let right = serde_json::to_vec(&a).expect("serialize");
        assert_eq!(crate::sha256_hex(&left), crate::sha256_hex(&right));
    }
}
