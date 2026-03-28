use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    bundle_absolute_sweep, bundle_residual_sweep, bundle_terminal_sweep, bundle_ultimate_sweep,
    derive_canonical_bundle_authority_v2, derive_canonical_governance_entry,
    exports_bundle_spine_check, final_bundle_consumer_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, prefix_hex, readiness_spine_check,
    repro_pack, require_bundle_convergence_inputs, require_canonical_governance_entry,
    residual_free_bundle_sweep, validate_governance_primary_surfaces_with_applied_scope,
    BugKitBuildArgs, CanonicalBundleKindV1, OpsError,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleConvergenceStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleConvergenceMismatchCategoryV1 {
    ConsumerSkippedBundleConvergence,
    ConsumerUsedBundleMemoPath,
    BundleInputScopeMismatch,
    BundleInputSpineMismatch,
    BundleInputExportContextMismatch,
    BundleMemoPathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleConvergenceConsumerStatusV1 {
    pub consumer: String,
    pub status: BundleConvergenceStatusV1,
    pub mismatch_categories: Vec<BundleConvergenceMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleConvergenceSweepV1 {
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
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub convergence_status: BundleConvergenceStatusV1,
    pub convergence_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleConvergenceSweepReportV1 {
    pub schema_version: u16,
    pub sweep: BundleConvergenceSweepV1,
    pub consumers: Vec<BundleConvergenceConsumerStatusV1>,
}

pub fn bundle_convergence_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<BundleConvergenceSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_bundle_convergence_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = require_canonical_governance_entry(
        &applied,
        Some(&derive_canonical_governance_entry(&applied, &surfaces)?),
    )?;
    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_check_bundle_convergence_sweep.json"),
    )?
    .canonical_readiness_spine;
    let final_bundle_consumer = final_bundle_consumer_sweep(
        workdir,
        &workdir.join("out/final_bundle_consumer_sweep_bundle_convergence.json"),
    )?;
    let final_bundle_residual = bundle_residual_sweep(
        workdir,
        &workdir.join("out/bundle_residual_sweep_bundle_convergence.json"),
    )?;
    let residual_free = residual_free_bundle_sweep(
        workdir,
        &workdir.join("out/residual_free_bundle_sweep_bundle_convergence.json"),
    )?;
    let absolute = bundle_absolute_sweep(
        workdir,
        &workdir.join("out/bundle_absolute_sweep_bundle_convergence.json"),
    )?;
    let terminal = bundle_terminal_sweep(
        workdir,
        &workdir.join("out/bundle_terminal_sweep_bundle_convergence.json"),
    )?;
    let ultimate = bundle_ultimate_sweep(
        workdir,
        &workdir.join("out/bundle_ultimate_sweep_bundle_convergence.json"),
    )?;

    let run_id = crate::reproducible_run_id_for_smoke(workdir)?;
    let repro_bundle = workdir
        .join("out")
        .join(format!("repro_{run_id}_bundle_convergence_sweep.zip"));
    let _ = repro_pack(workdir, &run_id, &repro_bundle)?;
    let repro_spine = exports_bundle_spine_check(
        &repro_bundle,
        &workdir.join("out/repro_bundle_spine_check_bundle_convergence_sweep.json"),
    )?;
    let repro_manifest = read_repro_manifest(&repro_bundle)?;
    let repro_context = crate::parse_normalized_bundle_manifest(
        CanonicalBundleKindV1::Repro,
        &repro_manifest.export_context,
        &repro_manifest.related_artifacts,
    )?;
    let repro_authority = derive_canonical_bundle_authority_v2(&repro_spine.spine, 1, false)?;

    let authority_ctx = require_bundle_convergence_inputs(
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
        Some(&applied),
        Some(&governance),
        Some(&readiness),
    )?;

    let bugkit_bundle = workdir
        .join("out")
        .join(format!("bugkit_{run_id}_bundle_convergence_sweep.zip"));
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
            "BundleSpineCheck",
            workdir,
            "out/bundle_spine_check.json",
            &authority_ctx,
        )?,
        check_consumer(
            "ContinuityArtifacts",
            workdir,
            "out/ultimate_terminal_absolute_final_input_continuity_sweep.json",
            &authority_ctx,
        )?,
        check_consumer(
            "InspectSummary",
            workdir,
            "out/export_inspect_report.json",
            &authority_ctx,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));

    let residual_path_count = consumers
        .iter()
        .filter(|consumer| !matches!(consumer.status, BundleConvergenceStatusV1::Pass))
        .count() as u16;

    let convergence_status = if consumers
        .iter()
        .any(|c| matches!(c.status, BundleConvergenceStatusV1::LegacyPresent))
    {
        BundleConvergenceStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        BundleConvergenceStatusV1::Pass
    } else {
        BundleConvergenceStatusV1::Fail
    };

    let sweep = derive_sweep(
        &authority_ctx,
        consumers.len() as u16,
        residual_path_count,
        convergence_status,
    );
    let report = BundleConvergenceSweepReportV1 {
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
    authority_ctx: &crate::BundleConvergenceInputsV1,
) -> Result<BundleConvergenceConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories
            .insert(BundleConvergenceMismatchCategoryV1::ConsumerSkippedBundleConvergence);
        return Ok(BundleConvergenceConsumerStatusV1 {
            consumer: consumer.to_string(),
            status: BundleConvergenceStatusV1::LegacyPresent,
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }

    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;

    let field_match = |field: &str, expected: &str| {
        value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|v| v == expected)
    };

    if !field_match(
        "bundle_ultimate_sweep_digest_prefix",
        &authority_ctx.terminal_bundle_ultimate_sweep_digest_prefix,
    ) {
        mismatch_categories
            .insert(BundleConvergenceMismatchCategoryV1::BundleInputExportContextMismatch);
    }
    if value
        .get("bundle_convergence_sweep_digest_prefix")
        .is_some()
        && !field_match(
            "bundle_convergence_sweep_digest_prefix",
            &prefix_hex(&authority_ctx.authority_digest, DIGEST_PREFIX_LEN),
        )
    {
        mismatch_categories.insert(BundleConvergenceMismatchCategoryV1::BundleMemoPathPresent);
    }
    if !field_match(
        "canonical_bundle_spine_digest_prefix",
        &authority_ctx.canonical_bundle_spine_digest_prefix,
    ) {
        mismatch_categories.insert(BundleConvergenceMismatchCategoryV1::BundleInputSpineMismatch);
    }

    if let Some(context) = value.get("export_context") {
        if context
            .get("supported_slot_set_digest_prefix")
            .and_then(serde_json::Value::as_str)
            != Some(authority_ctx.applied_supported_set_digest_prefix.as_str())
        {
            mismatch_categories
                .insert(BundleConvergenceMismatchCategoryV1::BundleInputScopeMismatch);
        }
    }

    for forbidden_key in [
        "bundle_echo_cache",
        "manifest_echo",
        "manifest_mirror",
        "embedded_export_snapshot",
        "bundle_lineage",
        "bundle_history",
        "inspect_summary",
        "report_summary",
    ] {
        if value.get(forbidden_key).is_some() {
            mismatch_categories
                .insert(BundleConvergenceMismatchCategoryV1::ConsumerUsedBundleMemoPath);
        }
    }

    let status = if mismatch_categories.is_empty() {
        BundleConvergenceStatusV1::Pass
    } else if mismatch_categories
        .contains(&BundleConvergenceMismatchCategoryV1::BundleMemoPathPresent)
        || mismatch_categories
            .contains(&BundleConvergenceMismatchCategoryV1::ConsumerUsedBundleMemoPath)
    {
        BundleConvergenceStatusV1::LegacyPresent
    } else {
        BundleConvergenceStatusV1::Fail
    };

    Ok(BundleConvergenceConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    })
}

fn derive_sweep(
    authority_ctx: &crate::BundleConvergenceInputsV1,
    covered_consumer_count: u16,
    residual_path_count: u16,
    convergence_status: BundleConvergenceStatusV1,
) -> BundleConvergenceSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"bundle_convergence_sweep_v1");
    bytes.extend_from_slice(authority_ctx.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(
        authority_ctx
            .canonical_governance_entry_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .canonical_readiness_spine_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .canonical_bundle_spine_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .canonical_bundle_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .final_bundle_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .final_bundle_residual_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .residual_free_bundle_consumer_authority_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .residual_free_bundle_absolute_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .absolute_final_bundle_terminal_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(
        authority_ctx
            .terminal_bundle_ultimate_sweep_digest_prefix
            .as_bytes(),
    );
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{convergence_status:?}").as_bytes());

    BundleConvergenceSweepV1 {
        applied_supported_set_digest_prefix: authority_ctx
            .applied_supported_set_digest_prefix
            .clone(),
        canonical_governance_entry_digest_prefix: authority_ctx
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: authority_ctx
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_bundle_spine_digest_prefix: authority_ctx
            .canonical_bundle_spine_digest_prefix
            .clone(),
        canonical_bundle_authority_digest_prefix: authority_ctx
            .canonical_bundle_authority_digest_prefix
            .clone(),
        final_bundle_consumer_authority_digest_prefix: authority_ctx
            .final_bundle_consumer_authority_digest_prefix
            .clone(),
        final_bundle_residual_sweep_digest_prefix: authority_ctx
            .final_bundle_residual_sweep_digest_prefix
            .clone(),
        residual_free_bundle_consumer_authority_digest_prefix: authority_ctx
            .residual_free_bundle_consumer_authority_digest_prefix
            .clone(),
        residual_free_bundle_absolute_sweep_digest_prefix: authority_ctx
            .residual_free_bundle_absolute_sweep_digest_prefix
            .clone(),
        absolute_final_bundle_terminal_sweep_digest_prefix: authority_ctx
            .absolute_final_bundle_terminal_sweep_digest_prefix
            .clone(),
        terminal_bundle_ultimate_sweep_digest_prefix: authority_ctx
            .terminal_bundle_ultimate_sweep_digest_prefix
            .clone(),
        covered_consumer_count,
        residual_path_count,
        convergence_status,
        convergence_digest: crate::sha256_hex(&bytes),
    }
}

fn read_repro_manifest(bundle: &Path) -> Result<crate::ReproPackManifestV1, OpsError> {
    let file = fs::File::open(bundle)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| OpsError::Invalid(format!("unable to open repro zip: {e}")))?;
    let mut body = String::new();
    let mut mf = archive
        .by_name("repro_pack_manifest.json")
        .map_err(|e| OpsError::Invalid(format!("missing repro_pack_manifest.json: {e}")))?;
    std::io::Read::read_to_string(&mut mf, &mut body)?;
    Ok(serde_json::from_str(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_convergence_sweep_digest_stable() {
        let input = crate::BundleConvergenceInputsV1 {
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
            authority_digest: "cc".repeat(8),
        };
        let a = derive_sweep(&input, 8, 0, BundleConvergenceStatusV1::Pass);
        let b = derive_sweep(&input, 8, 0, BundleConvergenceStatusV1::Pass);
        assert_eq!(a.convergence_digest, b.convergence_digest);
    }
}
