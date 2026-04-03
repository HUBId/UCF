use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    bundle_absolute_sweep, bundle_closure_sweep, bundle_convergence_sweep,
    bundle_final_consolidation_sweep, bundle_residual_sweep, bundle_stabilization_sweep,
    bundle_terminal_sweep, bundle_ultimate_sweep, derive_canonical_bundle_authority_v2,
    exports_bundle_spine_check, final_bundle_consumer_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, readiness_spine_check, repro_pack,
    require_bundle_seal_inputs, require_canonical_governance_entry, residual_free_bundle_sweep,
    validate_governance_primary_surfaces_with_applied_scope, BugKitBuildArgs,
    CanonicalBundleKindV1, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleSealStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleSealMismatchCategoryV1 {
    ConsumerSkippedBundleSeal,
    ConsumerUsedBundleShellPath,
    BundleInputScopeMismatch,
    BundleInputSpineMismatch,
    BundleInputExportContextMismatch,
    BundleShellPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleSealConsumerStatusV1 {
    pub consumer: String,
    pub status: BundleSealStatusV1,
    pub mismatch_categories: Vec<BundleSealMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleSealSweepV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_bundle_authority_digest_prefix: String,
    pub final_bundle_consumer_authority_digest_prefix: String,
    pub final_bundle_residual_sweep_digest_prefix: String,
    pub residual_free_bundle_consumer_authority_digest_prefix: String,
    pub residual_free_bundle_absolute_sweep_digest_prefix: String,
    pub absolute_final_bundle_terminal_sweep_digest_prefix: String,
    pub terminal_bundle_ultimate_sweep_digest_prefix: String,
    pub bundle_convergence_sweep_digest_prefix: String,
    pub bundle_stabilization_sweep_digest_prefix: String,
    pub bundle_final_consolidation_sweep_digest_prefix: String,
    pub bundle_closure_sweep_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub seal_status: BundleSealStatusV1,
    pub seal_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleSealSweepReportV1 {
    pub schema_version: u16,
    pub sweep: BundleSealSweepV1,
    pub consumers: Vec<BundleSealConsumerStatusV1>,
}

pub fn bundle_seal_sweep(workdir: &Path, out: &Path) -> Result<BundleSealSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_bundle_seal_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = require_canonical_governance_entry(
        &applied,
        Some(&crate::derive_canonical_governance_entry(
            &applied, &surfaces,
        )?),
    )?;
    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_check_bundle_seal_sweep.json"),
    )?
    .canonical_readiness_spine;
    let final_bundle_consumer = final_bundle_consumer_sweep(
        workdir,
        &workdir.join("out/final_bundle_consumer_sweep_bundle_seal_sweep.json"),
    )?;
    let final_bundle_residual = bundle_residual_sweep(
        workdir,
        &workdir.join("out/bundle_residual_sweep_bundle_seal_sweep.json"),
    )?;
    let residual_free = residual_free_bundle_sweep(
        workdir,
        &workdir.join("out/residual_free_bundle_sweep_bundle_seal_sweep.json"),
    )?;
    let absolute = bundle_absolute_sweep(
        workdir,
        &workdir.join("out/bundle_absolute_sweep_bundle_seal_sweep.json"),
    )?;
    let terminal = bundle_terminal_sweep(
        workdir,
        &workdir.join("out/bundle_terminal_sweep_bundle_seal_sweep.json"),
    )?;
    let ultimate = bundle_ultimate_sweep(
        workdir,
        &workdir.join("out/bundle_ultimate_sweep_bundle_seal_sweep.json"),
    )?;
    let convergence = bundle_convergence_sweep(
        workdir,
        &workdir.join("out/bundle_convergence_sweep_bundle_seal_sweep.json"),
    )?;
    let stabilization = bundle_stabilization_sweep(
        workdir,
        &workdir.join("out/bundle_stabilization_sweep_bundle_seal_sweep.json"),
    )?;
    let final_consolidation = bundle_final_consolidation_sweep(
        workdir,
        &workdir.join("out/bundle_final_consolidation_sweep_bundle_seal_sweep.json"),
    )?;
    let closure = bundle_closure_sweep(
        workdir,
        &workdir.join("out/bundle_closure_sweep_bundle_seal_sweep.json"),
    )?;

    let run_id = crate::reproducible_run_id_for_smoke(workdir)?;
    let repro_bundle = workdir
        .join("out")
        .join(format!("repro_{run_id}_bundle_seal_sweep.zip"));
    let _ = repro_pack(workdir, &run_id, &repro_bundle)?;
    let repro_spine = exports_bundle_spine_check(
        &repro_bundle,
        &workdir.join("out/repro_bundle_spine_check_bundle_seal_sweep.json"),
    )?;
    let repro_manifest = read_repro_manifest(&repro_bundle)?;
    let repro_context = crate::parse_normalized_bundle_manifest(
        CanonicalBundleKindV1::Repro,
        &repro_manifest.export_context,
        &repro_manifest.related_artifacts,
    )?;
    let repro_authority = derive_canonical_bundle_authority_v2(&repro_spine.spine, 1, false)?;

    let authority_ctx = require_bundle_seal_inputs(
        CanonicalBundleKindV1::Repro,
        Some(&repro_manifest.export_context),
        Some(&repro_manifest.related_artifacts),
        Some(&repro_context),
        Some(&repro_spine.spine),
        Some(&repro_authority),
        Some(&final_bundle_consumer.authority),
        Some(&final_bundle_residual.sweep),
        Some(&residual_free.authority),
        Some(&absolute.sweep),
        Some(&terminal.sweep),
        Some(&ultimate.sweep),
        Some(&convergence.sweep),
        Some(&stabilization.sweep),
        Some(&final_consolidation.sweep),
        Some(&closure.sweep),
        Some(&applied),
        Some(&governance),
        Some(&readiness),
    )?;

    let bugkit_bundle = workdir
        .join("out")
        .join(format!("bugkit_{run_id}_bundle_seal_sweep.zip"));
    let _ = crate::bugkit_build(
        workdir,
        &run_id,
        &bugkit_bundle,
        &BugKitBuildArgs::default(),
    )?;

    let mut consumers = vec![
        check_consumer(
            "ReproManifest",
            workdir,
            "out/repro_pack_manifest.json",
            &authority_ctx,
        )?,
        check_consumer(
            "BugKitManifest",
            workdir,
            "out/bugkit_manifest.json",
            &authority_ctx,
        )?,
        check_consumer(
            "ReproVerify",
            workdir,
            "out/repro_verify.json",
            &authority_ctx,
        )?,
        check_consumer(
            "RoundTripCheck",
            workdir,
            "out/export_roundtrip_check.json",
            &authority_ctx,
        )?,
        check_consumer(
            "InspectSummary",
            workdir,
            "out/export_inspect_report.json",
            &authority_ctx,
        )?,
        check_consumer(
            "Continuity",
            workdir,
            "out/canonical_closure_continuity_sweep.json",
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
            "V20PrepGateHelper",
            workdir,
            "out/v19_gate_report.json",
            &authority_ctx,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_path_count = consumers
        .iter()
        .filter(|consumer| !matches!(consumer.status, BundleSealStatusV1::Pass))
        .count() as u16;
    let seal_status = if consumers
        .iter()
        .any(|c| matches!(c.status, BundleSealStatusV1::LegacyPresent))
    {
        BundleSealStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        BundleSealStatusV1::Pass
    } else {
        BundleSealStatusV1::Fail
    };

    let report = BundleSealSweepReportV1 {
        schema_version: 1,
        sweep: derive_sweep(
            &authority_ctx,
            consumers.len() as u16,
            residual_path_count,
            seal_status,
        ),
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
    authority_ctx: &crate::BundleSealInputsV1,
) -> Result<BundleSealConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(BundleSealMismatchCategoryV1::ConsumerSkippedBundleSeal);
        return Ok(BundleSealConsumerStatusV1 {
            consumer: consumer.to_string(),
            status: BundleSealStatusV1::LegacyPresent,
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
        mismatch_categories.insert(BundleSealMismatchCategoryV1::BundleInputScopeMismatch);
    }
    if !field_match(
        "canonical_bundle_spine_digest_prefix",
        &authority_ctx.canonical_bundle_spine_digest_prefix,
    ) {
        mismatch_categories.insert(BundleSealMismatchCategoryV1::BundleInputSpineMismatch);
    }
    if !field_match(
        "bundle_closure_sweep_digest_prefix",
        &authority_ctx.bundle_closure_sweep_digest_prefix,
    ) {
        mismatch_categories.insert(BundleSealMismatchCategoryV1::ConsumerSkippedBundleSeal);
    }
    if value
        .get("export_context")
        .is_some_and(|ctx| ctx.get("context_digest").is_none())
    {
        mismatch_categories.insert(BundleSealMismatchCategoryV1::BundleInputExportContextMismatch);
    }

    for forbidden in [
        "bundle_shell_path",
        "bundle_compatibility_shell",
        "bundle_bridge_layer",
        "bundle_auxiliary_export_view",
        "bundle_lineage",
        "bundle_history",
        "bundle_roundtrip_primary",
        "bundle_check_primary",
        "inspect_summary_primary",
    ] {
        if value.get(forbidden).is_some() {
            mismatch_categories.insert(BundleSealMismatchCategoryV1::BundleShellPathPresent);
        }
    }

    if value.get("bundle_adapter_path").is_some()
        || value.get("bundle_translation_path").is_some()
        || value.get("bundle_facade_path").is_some()
        || value.get("bundle_alias_layer").is_some()
        || value.get("shadow_bundle_view").is_some()
    {
        mismatch_categories.insert(BundleSealMismatchCategoryV1::ConsumerUsedBundleShellPath);
    }

    let status = if mismatch_categories.is_empty() {
        BundleSealStatusV1::Pass
    } else if mismatch_categories.contains(&BundleSealMismatchCategoryV1::BundleShellPathPresent)
        || mismatch_categories.contains(&BundleSealMismatchCategoryV1::ConsumerUsedBundleShellPath)
    {
        BundleSealStatusV1::LegacyPresent
    } else {
        BundleSealStatusV1::Fail
    };

    Ok(BundleSealConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    ctx: &crate::BundleSealInputsV1,
    covered_consumer_count: u16,
    residual_path_count: u16,
    seal_status: BundleSealStatusV1,
) -> BundleSealSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"bundle_seal_sweep_v1");
    bytes.extend_from_slice(ctx.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_readiness_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_bundle_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.canonical_bundle_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.final_bundle_consumer_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.final_bundle_residual_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        ctx.residual_free_bundle_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.residual_free_bundle_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        ctx.absolute_final_bundle_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(ctx.terminal_bundle_ultimate_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.bundle_convergence_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(ctx.bundle_stabilization_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        ctx.bundle_final_consolidation_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(ctx.bundle_closure_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", seal_status).as_bytes());

    BundleSealSweepV1 {
        applied_supported_set_digest_prefix: ctx.applied_supported_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: ctx
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: ctx
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_bundle_spine_digest_prefix: ctx.canonical_bundle_spine_digest_prefix.clone(),
        canonical_bundle_authority_digest_prefix: ctx
            .canonical_bundle_authority_digest_prefix
            .clone(),
        final_bundle_consumer_authority_digest_prefix: ctx
            .final_bundle_consumer_authority_digest_prefix
            .clone(),
        final_bundle_residual_sweep_digest_prefix: ctx
            .final_bundle_residual_sweep_digest_prefix
            .clone(),
        residual_free_bundle_consumer_authority_digest_prefix: ctx
            .residual_free_bundle_consumer_authority_digest_prefix
            .clone(),
        residual_free_bundle_absolute_sweep_digest_prefix: ctx
            .residual_free_bundle_absolute_sweep_digest_prefix
            .clone(),
        absolute_final_bundle_terminal_sweep_digest_prefix: ctx
            .absolute_final_bundle_terminal_sweep_digest_prefix
            .clone(),
        terminal_bundle_ultimate_sweep_digest_prefix: ctx
            .terminal_bundle_ultimate_sweep_digest_prefix
            .clone(),
        bundle_convergence_sweep_digest_prefix: ctx.bundle_convergence_sweep_digest_prefix.clone(),
        bundle_stabilization_sweep_digest_prefix: ctx
            .bundle_stabilization_sweep_digest_prefix
            .clone(),
        bundle_final_consolidation_sweep_digest_prefix: ctx
            .bundle_final_consolidation_sweep_digest_prefix
            .clone(),
        bundle_closure_sweep_digest_prefix: ctx.bundle_closure_sweep_digest_prefix.clone(),
        covered_consumer_count,
        residual_path_count,
        seal_status,
        seal_digest: crate::sha256_hex(&bytes),
    }
}

fn read_repro_manifest(bundle: &Path) -> Result<crate::ReproPackManifestV1, OpsError> {
    let file = fs::File::open(bundle)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| OpsError::Invalid(format!("unable to open repro zip: {e}")))?;
    let mut manifest = archive
        .by_name("repro_pack_manifest.json")
        .map_err(|e| OpsError::Invalid(format!("missing repro_pack_manifest.json: {e}")))?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut manifest, &mut buf)?;
    Ok(serde_json::from_slice(&buf)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_seal_digest_stable() {
        let ctx = crate::BundleSealInputsV1 {
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_readiness_spine_digest_prefix: "33".repeat(8),
            canonical_bundle_spine_digest_prefix: "44".repeat(8),
            canonical_bundle_authority_digest_prefix: "55".repeat(8),
            final_bundle_consumer_authority_digest_prefix: "66".repeat(8),
            final_bundle_residual_sweep_digest_prefix: "77".repeat(8),
            residual_free_bundle_consumer_authority_digest_prefix: "88".repeat(8),
            residual_free_bundle_absolute_sweep_digest_prefix: "99".repeat(8),
            absolute_final_bundle_terminal_sweep_digest_prefix: "aa".repeat(8),
            terminal_bundle_ultimate_sweep_digest_prefix: "bb".repeat(8),
            bundle_convergence_sweep_digest_prefix: "cc".repeat(8),
            bundle_stabilization_sweep_digest_prefix: "dd".repeat(8),
            bundle_final_consolidation_sweep_digest_prefix: "ee".repeat(8),
            bundle_closure_sweep_digest_prefix: "ff".repeat(8),
            authority_digest: "01".repeat(32),
        };
        let first = derive_sweep(&ctx, 9, 0, BundleSealStatusV1::Pass);
        let second = derive_sweep(&ctx, 9, 0, BundleSealStatusV1::Pass);
        assert_eq!(first.seal_digest, second.seal_digest);
    }

    #[test]
    fn bundle_seal_status_deterministic_for_shell_paths() {
        let mismatches = BTreeSet::from([BundleSealMismatchCategoryV1::BundleShellPathPresent]);
        let status = if mismatches.is_empty() {
            BundleSealStatusV1::Pass
        } else if mismatches.contains(&BundleSealMismatchCategoryV1::BundleShellPathPresent)
            || mismatches.contains(&BundleSealMismatchCategoryV1::ConsumerUsedBundleShellPath)
        {
            BundleSealStatusV1::LegacyPresent
        } else {
            BundleSealStatusV1::Fail
        };
        assert_eq!(status, BundleSealStatusV1::LegacyPresent);
    }
}
