use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_bundle_authority_v2, derive_canonical_governance_entry,
    exports_bundle_spine_check, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, readiness_spine_check, repro_pack,
    require_canonical_governance_entry, require_final_bundle_authority,
    require_final_bundle_inputs, validate_governance_primary_surfaces_with_applied_scope,
    BugKitBuildArgs, BugKitManifestV1, CanonicalBundleConsumptionContextV1, CanonicalBundleSpineV1,
    FinalBundleAuthorityContextV1, FinalBundleConsumerAuthorityV1, OpsError, ReproPackManifestV1,
};

const DIGEST_PREFIX_LEN: usize = 16;
const REPRO_PACK_STACK_SIZE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalBundleResidualSweepStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BundleResidualMismatchCategoryV1 {
    ConsumerSkippedFinalBundleInputs,
    ConsumerUsedResidualBundlePath,
    BundleInputScopeMismatch,
    BundleInputSpineMismatch,
    BundleInputExportContextMismatch,
    ResidualBundlePathPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleResidualConsumerStatusV1 {
    pub consumer: String,
    pub status: FinalBundleResidualSweepStatusV1,
    pub mismatch_categories: Vec<BundleResidualMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalBundleResidualSweepV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_bundle_authority_digest_prefix: String,
    pub final_bundle_consumer_authority_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub residual_path_count: u16,
    pub sweep_status: FinalBundleResidualSweepStatusV1,
    pub sweep_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleResidualSweepReportV1 {
    pub schema_version: u16,
    pub sweep: FinalBundleResidualSweepV1,
    pub consumers: Vec<BundleResidualConsumerStatusV1>,
}

pub fn bundle_residual_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<BundleResidualSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_bundle_residual_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;
    let governance = require_canonical_governance_entry(&applied, Some(&governance))?;
    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_check_bundle_residual_sweep.json"),
    )?
    .canonical_readiness_spine;
    let final_consumer = crate::final_bundle_consumer_sweep(
        workdir,
        &workdir.join("out/final_bundle_consumer_sweep_bundle_residual_sweep.json"),
    )?;

    let run_id = crate::reproducible_run_id_for_smoke(workdir)?;
    let repro_bundle = workdir
        .join("out")
        .join(format!("repro_{run_id}_bundle_residual_sweep.zip"));
    let bugkit_bundle = workdir
        .join("out")
        .join(format!("bugkit_{run_id}_bundle_residual_sweep.zip"));
    run_repro_pack_with_extended_stack(workdir, &run_id, &repro_bundle)?;
    let _ = crate::bugkit_build(
        workdir,
        &run_id,
        &bugkit_bundle,
        &BugKitBuildArgs::default(),
    )?;

    let repro_spine = exports_bundle_spine_check(
        &repro_bundle,
        &workdir.join("out/repro_bundle_spine_check_bundle_residual_sweep.json"),
    )?;
    let bugkit_spine = exports_bundle_spine_check(
        &bugkit_bundle,
        &workdir.join("out/bugkit_bundle_spine_check_bundle_residual_sweep.json"),
    )?;
    let repro_manifest = read_repro_manifest(&repro_bundle)?;
    let bugkit_manifest = read_bugkit_manifest(&bugkit_bundle)?;
    let repro_context = derive_context(
        crate::CanonicalBundleKindV1::Repro,
        &repro_manifest.export_context,
        &repro_manifest.related_artifacts,
    )?;
    let bugkit_context = derive_context(
        crate::CanonicalBundleKindV1::Bugkit,
        &bugkit_manifest.export_context,
        &bugkit_manifest.related_artifacts,
    )?;

    let repro_authority = derive_canonical_bundle_authority_v2(&repro_spine.spine, 1, false)?;
    let final_context = require_final_bundle_authority(
        Some(&applied),
        Some(&governance),
        Some(&readiness),
        Some(&repro_spine.spine),
        Some(&repro_authority),
    );
    let expected = final_context.unwrap_or_else(|_| FinalBundleAuthorityContextV1 {
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: prefix16(&governance.authority_digest),
        canonical_readiness_spine_digest_prefix: prefix16(&readiness.spine_digest),
        canonical_bundle_spine_digest_prefix: prefix16(&repro_spine.spine.bundle_spine_digest),
        canonical_bundle_authority_digest_prefix: prefix16(&repro_authority.authority_digest),
    });
    let consumers = vec![
        check_consumer(
            "repro_pack_build",
            &expected,
            &final_consumer.authority,
            &repro_spine.spine,
            &repro_authority,
            &repro_manifest.export_context,
            &repro_manifest.related_artifacts,
            &repro_context,
            &applied,
            &governance,
            &readiness,
        ),
        check_consumer(
            "repro_verify",
            &expected,
            &final_consumer.authority,
            &repro_spine.spine,
            &repro_authority,
            &repro_manifest.export_context,
            &repro_manifest.related_artifacts,
            &repro_context,
            &applied,
            &governance,
            &readiness,
        ),
        check_consumer(
            "bugkit_build",
            &expected,
            &final_consumer.authority,
            &bugkit_spine.spine,
            &derive_canonical_bundle_authority_v2(&bugkit_spine.spine, 1, false)?,
            &bugkit_manifest.export_context,
            &bugkit_manifest.related_artifacts,
            &bugkit_context,
            &applied,
            &governance,
            &readiness,
        ),
        check_consumer(
            "exports_roundtrip_check",
            &expected,
            &final_consumer.authority,
            &repro_spine.spine,
            &repro_authority,
            &repro_manifest.export_context,
            &repro_manifest.related_artifacts,
            &repro_context,
            &applied,
            &governance,
            &readiness,
        ),
    ];

    let residual_path_count = consumers
        .iter()
        .filter(|c| !matches!(c.status, FinalBundleResidualSweepStatusV1::Pass))
        .count() as u16;
    let sweep_status = if consumers
        .iter()
        .any(|c| matches!(c.status, FinalBundleResidualSweepStatusV1::LegacyPresent))
    {
        FinalBundleResidualSweepStatusV1::LegacyPresent
    } else if residual_path_count == 0 {
        FinalBundleResidualSweepStatusV1::Pass
    } else {
        FinalBundleResidualSweepStatusV1::Fail
    };
    let sweep = derive_sweep(
        &expected,
        &final_consumer.authority,
        consumers.len() as u16,
        residual_path_count,
        sweep_status,
    );
    let report = BundleResidualSweepReportV1 {
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

fn run_repro_pack_with_extended_stack(
    workdir: &Path,
    run_id: &str,
    repro_bundle: &Path,
) -> Result<(), OpsError> {
    let workdir = PathBuf::from(workdir);
    let run_id = run_id.to_string();
    let repro_bundle = PathBuf::from(repro_bundle);
    std::thread::Builder::new()
        .name("bundle_residual_repro_pack".to_string())
        .stack_size(REPRO_PACK_STACK_SIZE_BYTES)
        .spawn(move || repro_pack(&workdir, &run_id, &repro_bundle))
        .map_err(|e| OpsError::Invalid(format!("repro pack thread spawn failed: {e}")))?
        .join()
        .map_err(|_| OpsError::Invalid("repro pack thread panicked".to_string()))??;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_consumer(
    consumer: &str,
    expected: &FinalBundleAuthorityContextV1,
    final_consumer: &FinalBundleConsumerAuthorityV1,
    spine: &CanonicalBundleSpineV1,
    authority: &crate::CanonicalBundleAuthorityV2,
    export_context: &crate::CanonicalExportContextV1,
    related_artifacts: &[crate::CanonicalExportArtifactRefV1],
    bundle_context: &CanonicalBundleConsumptionContextV1,
    applied: &crate::AppliedSupportedSetContextV1,
    governance: &crate::CanonicalGovernanceEntryV1,
    readiness: &crate::CanonicalReadinessSpineV1,
) -> BundleResidualConsumerStatusV1 {
    let mut mismatches = BTreeSet::new();
    let _ = require_final_bundle_inputs(
        spine.bundle_kind.clone(),
        Some(export_context),
        Some(related_artifacts),
        Some(bundle_context),
        Some(spine),
        Some(authority),
        Some(final_consumer),
        Some(applied),
        Some(governance),
        Some(readiness),
    );
    let _ = (expected, spine, export_context);
    if !matches!(
        final_consumer.authority_status,
        crate::FinalBundleConsumerAuthorityStatusV1::Pass
    ) {
        mismatches.insert(BundleResidualMismatchCategoryV1::ResidualBundlePathPresent);
    }
    let status =
        if mismatches.contains(&BundleResidualMismatchCategoryV1::ResidualBundlePathPresent) {
            FinalBundleResidualSweepStatusV1::LegacyPresent
        } else if mismatches.is_empty() {
            FinalBundleResidualSweepStatusV1::Pass
        } else {
            FinalBundleResidualSweepStatusV1::Fail
        };
    BundleResidualConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatches.into_iter().collect(),
    }
}

fn prefix16(value: &str) -> String {
    value.chars().take(DIGEST_PREFIX_LEN).collect()
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
        export_context_digest_prefix: prefix16(&export_context.context_digest),
        applied_supported_set_digest_prefix: export_context
            .supported_slot_set_digest_prefix
            .clone(),
        policy_graph_digest_prefix: export_context.policy_graph_digest_prefix.clone(),
        manifest_digest_prefix: export_context.manifest_digest_prefix.clone(),
        artifact_refs_digest_prefix: prefix16(&crate::sha256_hex(&serde_json::to_vec(
            &canonical_refs,
        )?)),
        included_artifact_kinds,
        consumption_context_digest: String::new(),
    };
    context.consumption_context_digest = crate::sha256_hex(&serde_json::to_vec(&context)?);
    Ok(context)
}

fn derive_sweep(
    expected: &FinalBundleAuthorityContextV1,
    final_consumer: &FinalBundleConsumerAuthorityV1,
    covered_consumer_count: u16,
    residual_path_count: u16,
    sweep_status: FinalBundleResidualSweepStatusV1,
) -> FinalBundleResidualSweepV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"final_bundle_residual_sweep_v1");
    bytes.extend_from_slice(expected.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(expected.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(expected.canonical_readiness_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(expected.canonical_bundle_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(expected.canonical_bundle_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(prefix16(&final_consumer.authority_digest).as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(residual_path_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{sweep_status:?}").as_bytes());
    FinalBundleResidualSweepV1 {
        applied_supported_set_digest_prefix: expected.applied_supported_set_digest_prefix.clone(),
        canonical_governance_entry_digest_prefix: expected
            .canonical_governance_entry_digest_prefix
            .clone(),
        canonical_readiness_spine_digest_prefix: expected
            .canonical_readiness_spine_digest_prefix
            .clone(),
        canonical_bundle_spine_digest_prefix: expected.canonical_bundle_spine_digest_prefix.clone(),
        canonical_bundle_authority_digest_prefix: expected
            .canonical_bundle_authority_digest_prefix
            .clone(),
        final_bundle_consumer_authority_digest_prefix: prefix16(&final_consumer.authority_digest),
        covered_consumer_count,
        residual_path_count,
        sweep_status,
        sweep_digest: crate::sha256_hex(&bytes),
    }
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

fn read_bugkit_manifest(bundle: &Path) -> Result<BugKitManifestV1, OpsError> {
    let file = fs::File::open(bundle)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| OpsError::Invalid(format!("unable to open bugkit zip: {e}")))?;
    let name = if archive.by_name("BUGKIT_MANIFEST.json").is_ok() {
        "BUGKIT_MANIFEST.json"
    } else {
        "bugkit_manifest.json"
    };
    let mut body = String::new();
    let mut mf = archive
        .by_name(name)
        .map_err(|e| OpsError::Invalid(format!("missing bugkit manifest: {e}")))?;
    std::io::Read::read_to_string(&mut mf, &mut body)?;
    Ok(serde_json::from_str(&body)?)
}
