use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    bundle_residual_sweep, derive_canonical_bundle_authority_v2, derive_canonical_governance_entry,
    exports_bundle_spine_check, final_bundle_consumer_sweep, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, prefix_hex, readiness_spine_check,
    repro_pack, require_canonical_governance_entry, require_residual_free_final_bundle_inputs,
    validate_governance_primary_surfaces_with_applied_scope, BugKitBuildArgs,
    CanonicalBundleConsumptionContextV1, OpsError, ReproPackManifestV1,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreeBundleConsumerAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidualFreeBundleMismatchCategoryV1 {
    ConsumerSkippedResidualFreeFinalBundleInputs,
    ConsumerUsedHistoricalBundlePath,
    BundleInputScopeMismatch,
    BundleInputSpineMismatch,
    BundleInputExportContextMismatch,
    HistoricalBundlePathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeBundleConsumerStatusV1 {
    pub consumer: String,
    pub status: ResidualFreeBundleConsumerAuthorityStatusV1,
    pub mismatch_categories: Vec<ResidualFreeBundleMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeBundleConsumerAuthorityV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_bundle_authority_digest_prefix: String,
    pub final_bundle_consumer_authority_digest_prefix: String,
    pub final_bundle_residual_sweep_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub authority_status: ResidualFreeBundleConsumerAuthorityStatusV1,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidualFreeBundleSweepReportV1 {
    pub schema_version: u16,
    pub authority: ResidualFreeBundleConsumerAuthorityV1,
    pub consumers: Vec<ResidualFreeBundleConsumerStatusV1>,
}

pub fn residual_free_bundle_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<ResidualFreeBundleSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_residual_free_bundle_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = require_canonical_governance_entry(
        &applied,
        Some(&derive_canonical_governance_entry(&applied, &surfaces)?),
    )?;
    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_check_residual_free_bundle_sweep.json"),
    )?
    .canonical_readiness_spine;
    let final_bundle_consumer = final_bundle_consumer_sweep(
        workdir,
        &workdir.join("out/final_bundle_consumer_sweep_residual_free_bundle_sweep.json"),
    )?;
    let final_bundle_residual = bundle_residual_sweep(
        workdir,
        &workdir.join("out/bundle_residual_sweep_residual_free_bundle_sweep.json"),
    )?;

    let run_id = crate::reproducible_run_id_for_smoke(workdir)?;
    let repro_bundle = workdir
        .join("out")
        .join(format!("repro_{run_id}_residual_free_bundle_sweep.zip"));
    let bugkit_bundle = workdir
        .join("out")
        .join(format!("bugkit_{run_id}_residual_free_bundle_sweep.zip"));
    let _ = repro_pack(workdir, &run_id, &repro_bundle)?;
    let _ = crate::bugkit_build(
        workdir,
        &run_id,
        &bugkit_bundle,
        &BugKitBuildArgs::default(),
    )?;
    let repro_spine = exports_bundle_spine_check(
        &repro_bundle,
        &workdir.join("out/repro_bundle_spine_check_residual_free_bundle_sweep.json"),
    )?;
    let repro_manifest = read_repro_manifest(&repro_bundle)?;
    let repro_context = derive_context(
        crate::CanonicalBundleKindV1::Repro,
        &repro_manifest.export_context,
        &repro_manifest.related_artifacts,
    )?;
    let repro_authority = derive_canonical_bundle_authority_v2(&repro_spine.spine, 1, false)?;

    let authority_ctx = require_residual_free_final_bundle_inputs(
        crate::CanonicalBundleKindV1::Repro,
        Some(&repro_manifest.export_context),
        Some(&repro_manifest.related_artifacts),
        Some(&repro_context),
        Some(&repro_spine.spine),
        Some(&repro_authority),
        Some(&final_bundle_consumer.authority),
        Some(&final_bundle_residual.sweep),
        Some(&applied),
        Some(&governance),
        Some(&readiness),
    )?;

    let mut consumers = vec![
        check_consumer(
            "ReproManifest",
            workdir,
            "out/repro_pack_manifest.json",
            &authority_ctx,
            false,
        )?,
        check_consumer(
            "BugKitManifest",
            workdir,
            "out/bugkit_manifest.json",
            &authority_ctx,
            false,
        )?,
        check_consumer(
            "BundleVerifyReport",
            workdir,
            "out/export_roundtrip_check.json",
            &authority_ctx,
            true,
        )?,
        check_consumer(
            "BundleInspectSummary",
            workdir,
            "out/export_inspect_report.json",
            &authority_ctx,
            true,
        )?,
        check_consumer(
            "ContinuityArtifacts",
            workdir,
            "out/operator_roundtrip_chain.json",
            &authority_ctx,
            true,
        )?,
    ];
    consumers.sort_by(|a, b| a.consumer.cmp(&b.consumer));
    let residual_path_count = consumers
        .iter()
        .filter(|c| !matches!(c.status, ResidualFreeBundleConsumerAuthorityStatusV1::Pass))
        .count() as u16;

    let authority_status = if consumers.iter().any(|c| {
        matches!(
            c.status,
            ResidualFreeBundleConsumerAuthorityStatusV1::LegacyPresent
        )
    }) {
        ResidualFreeBundleConsumerAuthorityStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        ResidualFreeBundleConsumerAuthorityStatusV1::Pass
    } else {
        ResidualFreeBundleConsumerAuthorityStatusV1::Fail
    };
    let authority = derive_authority(
        &authority_ctx.applied_supported_set_digest_prefix,
        &authority_ctx.canonical_governance_entry_digest_prefix,
        &authority_ctx.canonical_readiness_spine_digest_prefix,
        &authority_ctx.canonical_bundle_spine_digest_prefix,
        &authority_ctx.canonical_bundle_authority_digest_prefix,
        &authority_ctx.final_bundle_consumer_authority_digest_prefix,
        &authority_ctx.final_bundle_residual_sweep_digest_prefix,
        consumers.len() as u16,
        residual_path_count,
        authority_status,
    );
    let report = ResidualFreeBundleSweepReportV1 {
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
    authority_ctx: &crate::ResidualFreeFinalBundleInputsV1,
    allow_absent: bool,
) -> Result<ResidualFreeBundleConsumerStatusV1, OpsError> {
    let mut mismatch_categories = BTreeSet::new();
    let path = workdir.join(rel_path);
    if !path.exists() {
        mismatch_categories.insert(
            ResidualFreeBundleMismatchCategoryV1::ConsumerSkippedResidualFreeFinalBundleInputs,
        );
        return Ok(ResidualFreeBundleConsumerStatusV1 {
            consumer: consumer.to_string(),
            status: if allow_absent {
                ResidualFreeBundleConsumerAuthorityStatusV1::Fail
            } else {
                ResidualFreeBundleConsumerAuthorityStatusV1::LegacyPresent
            },
            mismatch_categories: mismatch_categories.into_iter().collect(),
        });
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let match_prefix = |key: &str, expected: &String| {
        value.get(key).and_then(|v| v.as_str()) == Some(expected.as_str())
    };

    if !match_prefix(
        "final_bundle_consumer_authority_digest_prefix",
        &authority_ctx.final_bundle_consumer_authority_digest_prefix,
    ) || !match_prefix(
        "final_bundle_residual_sweep_digest_prefix",
        &authority_ctx.final_bundle_residual_sweep_digest_prefix,
    ) {
        mismatch_categories
            .insert(ResidualFreeBundleMismatchCategoryV1::HistoricalBundlePathPresent);
    }
    if let Some(context) = value.get("export_context") {
        if context
            .get("supported_slot_set_digest_prefix")
            .and_then(|v| v.as_str())
            != Some(authority_ctx.applied_supported_set_digest_prefix.as_str())
        {
            mismatch_categories
                .insert(ResidualFreeBundleMismatchCategoryV1::BundleInputScopeMismatch);
        }
    }

    let status = if mismatch_categories
        .contains(&ResidualFreeBundleMismatchCategoryV1::HistoricalBundlePathPresent)
    {
        ResidualFreeBundleConsumerAuthorityStatusV1::LegacyPresent
    } else if mismatch_categories.is_empty() {
        ResidualFreeBundleConsumerAuthorityStatusV1::Pass
    } else {
        ResidualFreeBundleConsumerAuthorityStatusV1::Fail
    };
    Ok(ResidualFreeBundleConsumerStatusV1 {
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
    canonical_bundle_spine_digest_prefix: &str,
    canonical_bundle_authority_digest_prefix: &str,
    final_bundle_consumer_authority_digest_prefix: &str,
    final_bundle_residual_sweep_digest_prefix: &str,
    covered_consumer_count: u16,
    residual_path_count: u16,
    authority_status: ResidualFreeBundleConsumerAuthorityStatusV1,
) -> ResidualFreeBundleConsumerAuthorityV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"residual_free_bundle_consumer_authority_v1");
    bytes.extend_from_slice(applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_readiness_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_bundle_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_bundle_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(final_bundle_consumer_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(final_bundle_residual_sweep_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{authority_status:?}").as_bytes());
    ResidualFreeBundleConsumerAuthorityV1 {
        applied_supported_set_digest_prefix: applied_supported_set_digest_prefix.to_string(),
        canonical_governance_entry_digest_prefix: canonical_governance_entry_digest_prefix
            .to_string(),
        canonical_readiness_spine_digest_prefix: canonical_readiness_spine_digest_prefix
            .to_string(),
        canonical_bundle_spine_digest_prefix: canonical_bundle_spine_digest_prefix.to_string(),
        canonical_bundle_authority_digest_prefix: canonical_bundle_authority_digest_prefix
            .to_string(),
        final_bundle_consumer_authority_digest_prefix:
            final_bundle_consumer_authority_digest_prefix.to_string(),
        final_bundle_residual_sweep_digest_prefix: final_bundle_residual_sweep_digest_prefix
            .to_string(),
        covered_consumer_count,
        residual_path_count,
        authority_status,
        authority_digest: crate::sha256_hex(&bytes),
    }
}

fn derive_context(
    bundle_kind: crate::CanonicalBundleKindV1,
    export_context: &crate::CanonicalExportContextV1,
    related_artifacts: &[crate::CanonicalExportArtifactRefV1],
) -> Result<CanonicalBundleConsumptionContextV1, OpsError> {
    let mut included_artifact_kinds = related_artifacts
        .iter()
        .filter(|item| {
            matches!(
                item.included_state,
                crate::CanonicalArtifactIncludedStateV1::Included
            )
        })
        .map(|item| item.artifact_kind.clone())
        .collect::<Vec<_>>();
    included_artifact_kinds.sort();
    included_artifact_kinds.dedup();
    let mut canonical_refs = related_artifacts.to_vec();
    canonical_refs.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    let mut context = CanonicalBundleConsumptionContextV1 {
        bundle_kind,
        export_context_digest_prefix: prefix_hex(&export_context.context_digest, DIGEST_PREFIX_LEN),
        applied_supported_set_digest_prefix: export_context
            .supported_slot_set_digest_prefix
            .clone(),
        policy_graph_digest_prefix: export_context.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: export_context.manifest_digest_prefix.clone(),
        artifact_refs_digest_prefix: prefix_hex(
            &crate::sha256_hex(&serde_json::to_vec(&canonical_refs)?),
            DIGEST_PREFIX_LEN,
        ),
        included_artifact_kinds,
        consumption_context_digest: String::new(),
    };
    context.consumption_context_digest = crate::sha256_hex(&serde_json::to_vec(&context)?);
    Ok(context)
}

fn read_repro_manifest(bundle: &Path) -> Result<ReproPackManifestV1, OpsError> {
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
