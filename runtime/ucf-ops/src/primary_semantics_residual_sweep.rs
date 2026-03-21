use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::remediation_consistency::FinalPrimarySemanticsConsumerSurfaceStatusV1;
use crate::{
    final_primary_semantics_sweep, prefix_hex, require_final_primary_semantics_inputs,
    FinalPrimarySemanticsConsumerAuthorityStatusV1, FinalPrimarySemanticsConsumerAuthorityV1,
    FinalPrimarySemanticsInputsContextV1, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalPrimarySemanticsResidualSweepStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalPrimarySemanticsResidualMismatchCategoryV1 {
    SurfaceSkippedFinalPrimarySemanticsInputs,
    SurfaceUsedResidualPrimarySemanticsPath,
    PrimaryBlockingOrderMismatch,
    PrimaryRemediationOrderMismatch,
    CanonicalConditionMappingMismatch,
    ResidualPrimarySemanticsPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalPrimarySemanticsResidualSurfaceStatusV1 {
    pub surface_kind: String,
    pub status: FinalPrimarySemanticsResidualSweepStatusV1,
    pub mismatch_categories: Vec<FinalPrimarySemanticsResidualMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalPrimarySemanticsResidualSweepV1 {
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub final_primary_semantics_consumer_authority_digest_prefix: String,
    pub covered_surface_count: u16,
    pub residual_path_count: u16,
    pub sweep_status: FinalPrimarySemanticsResidualSweepStatusV1,
    pub sweep_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalPrimarySemanticsResidualSweepReportV1 {
    pub schema_version: u16,
    pub sweep: FinalPrimarySemanticsResidualSweepV1,
    pub surfaces: Vec<FinalPrimarySemanticsResidualSurfaceStatusV1>,
}

pub fn primary_semantics_residual_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<FinalPrimarySemanticsResidualSweepReportV1, OpsError> {
    let final_sweep = final_primary_semantics_sweep(
        workdir,
        &workdir.join("out/final_primary_semantics_sweep_v11_residual_sweep.json"),
    )?;
    let context =
        require_final_primary_semantics_inputs(None, None, None, Some(&final_sweep.authority))?;
    let mut surfaces = final_sweep
        .surface_statuses
        .iter()
        .map(|status| map_surface_status(status, &final_sweep.authority))
        .collect::<Vec<_>>();
    surfaces.sort_by(|a, b| a.surface_kind.cmp(&b.surface_kind));

    let residual_path_count = surfaces
        .iter()
        .filter(|surface| {
            !matches!(surface.status, FinalPrimarySemanticsResidualSweepStatusV1::Pass)
                || surface.mismatch_categories.contains(
                    &FinalPrimarySemanticsResidualMismatchCategoryV1::ResidualPrimarySemanticsPathPresent,
                )
        })
        .count() as u16;

    let sweep_status = if surfaces.iter().any(|surface| {
        matches!(
            surface.status,
            FinalPrimarySemanticsResidualSweepStatusV1::LegacyPresent
        )
    }) {
        FinalPrimarySemanticsResidualSweepStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        FinalPrimarySemanticsResidualSweepStatusV1::Pass
    } else {
        FinalPrimarySemanticsResidualSweepStatusV1::Fail
    };

    let sweep = derive_sweep(
        &context,
        &final_sweep.authority,
        surfaces.len() as u16,
        residual_path_count,
        sweep_status,
    );
    let report = FinalPrimarySemanticsResidualSweepReportV1 {
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

fn map_surface_status(
    status: &FinalPrimarySemanticsConsumerSurfaceStatusV1,
    authority: &FinalPrimarySemanticsConsumerAuthorityV1,
) -> FinalPrimarySemanticsResidualSurfaceStatusV1 {
    let mut categories = BTreeSet::new();
    for mismatch in &status.mismatch_categories {
        match mismatch.as_str() {
            "SURFACE_SKIPPED_FINAL_PRIMARY_SEMANTICS_AUTHORITY" => {
                categories.insert(
                    FinalPrimarySemanticsResidualMismatchCategoryV1::SurfaceSkippedFinalPrimarySemanticsInputs,
                );
            }
            "PRIMARY_BLOCKING_MISMATCH" => {
                categories.insert(
                    FinalPrimarySemanticsResidualMismatchCategoryV1::PrimaryBlockingOrderMismatch,
                );
            }
            "PRIMARY_REMEDIATION_MISMATCH" => {
                categories.insert(
                    FinalPrimarySemanticsResidualMismatchCategoryV1::PrimaryRemediationOrderMismatch,
                );
            }
            "CANONICAL_CONDITION_MISMATCH" => {
                categories.insert(
                    FinalPrimarySemanticsResidualMismatchCategoryV1::CanonicalConditionMappingMismatch,
                );
            }
            other if other.contains("LEGACY") => {
                categories.insert(
                    FinalPrimarySemanticsResidualMismatchCategoryV1::SurfaceUsedResidualPrimarySemanticsPath,
                );
                categories.insert(
                    FinalPrimarySemanticsResidualMismatchCategoryV1::ResidualPrimarySemanticsPathPresent,
                );
            }
            _ => {}
        }
    }
    if !matches!(
        authority.authority_status,
        FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass
    ) {
        categories.insert(
            FinalPrimarySemanticsResidualMismatchCategoryV1::ResidualPrimarySemanticsPathPresent,
        );
    }
    let residual = categories.contains(
        &FinalPrimarySemanticsResidualMismatchCategoryV1::ResidualPrimarySemanticsPathPresent,
    );
    let mapped_status = if residual {
        FinalPrimarySemanticsResidualSweepStatusV1::LegacyPresent
    } else if categories.is_empty() {
        FinalPrimarySemanticsResidualSweepStatusV1::Pass
    } else {
        FinalPrimarySemanticsResidualSweepStatusV1::Fail
    };
    FinalPrimarySemanticsResidualSurfaceStatusV1 {
        surface_kind: status.surface_kind.clone(),
        status: mapped_status,
        mismatch_categories: categories.into_iter().collect(),
    }
}

fn derive_sweep(
    inputs: &FinalPrimarySemanticsInputsContextV1,
    final_consumer: &FinalPrimarySemanticsConsumerAuthorityV1,
    covered_surface_count: u16,
    residual_path_count: u16,
    sweep_status: FinalPrimarySemanticsResidualSweepStatusV1,
) -> FinalPrimarySemanticsResidualSweepV1 {
    let payload = serde_json::to_vec(&(
        &inputs.canonical_governance_entry_digest_prefix,
        &inputs.canonical_readiness_spine_digest_prefix,
        &inputs.canonical_bundle_spine_digest_prefix,
        &inputs.canonical_primary_semantics_authority_digest_prefix,
        &inputs.final_primary_semantics_consumer_authority_digest_prefix,
        covered_surface_count,
        residual_path_count,
        &sweep_status,
    ))
    .expect("serializing primary semantics residual sweep");
    FinalPrimarySemanticsResidualSweepV1 {
        canonical_governance_entry_digest_prefix: inputs
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: inputs
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_bundle_spine_digest_prefix: inputs.canonical_bundle_spine_digest_prefix.clone(),
        canonical_primary_semantics_authority_digest_prefix: inputs
            .canonical_primary_semantics_authority_digest_prefix
            .clone(),
        final_primary_semantics_consumer_authority_digest_prefix: prefix_hex(
            &final_consumer.authority_digest,
            16,
        ),
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
        let inputs = FinalPrimarySemanticsInputsContextV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "44".repeat(8),
            final_primary_semantics_consumer_authority_digest_prefix: "55".repeat(8),
        };
        let consumer = FinalPrimarySemanticsConsumerAuthorityV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "44".repeat(8),
            covered_consumer_count: 1,
            authority_status: FinalPrimarySemanticsConsumerAuthorityStatusV1::Pass,
            authority_digest: "aa".repeat(32),
        };
        let a = derive_sweep(
            &inputs,
            &consumer,
            3,
            0,
            FinalPrimarySemanticsResidualSweepStatusV1::Pass,
        );
        let b = derive_sweep(
            &inputs,
            &consumer,
            3,
            0,
            FinalPrimarySemanticsResidualSweepStatusV1::Pass,
        );
        assert_eq!(a.sweep_digest, b.sweep_digest);
    }

    #[test]
    fn residual_status_is_deterministic_for_legacy_mismatch() {
        let surface = FinalPrimarySemanticsConsumerSurfaceStatusV1 {
            surface_kind: "OperatorSignoff".to_string(),
            status: crate::CrossSurfaceObservationStatusV1::Fail,
            mismatch_categories: vec!["LEGACY_PRIMARY_SEMANTICS_PRESENT".to_string()],
        };
        let authority = FinalPrimarySemanticsConsumerAuthorityV1 {
            canonical_governance_entry_digest_prefix: "11".repeat(8),
            canonical_readiness_spine_digest_prefix: "22".repeat(8),
            canonical_bundle_spine_digest_prefix: "33".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "44".repeat(8),
            covered_consumer_count: 1,
            authority_status: FinalPrimarySemanticsConsumerAuthorityStatusV1::LegacyPresent,
            authority_digest: "aa".repeat(32),
        };
        let mapped = map_surface_status(&surface, &authority);
        assert!(matches!(
            mapped.status,
            FinalPrimarySemanticsResidualSweepStatusV1::LegacyPresent
        ));
        assert!(mapped.mismatch_categories.contains(
            &FinalPrimarySemanticsResidualMismatchCategoryV1::ResidualPrimarySemanticsPathPresent
        ));
    }
}
