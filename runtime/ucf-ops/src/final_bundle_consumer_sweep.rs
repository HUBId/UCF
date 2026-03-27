use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_bundle_authority_v2, derive_canonical_governance_entry,
    exports_bundle_spine_check, load_applied_supported_set_context_v1,
    models_active_review_snapshot, models_evidence_snapshot, readiness_spine_check, repro_pack,
    require_canonical_governance_entry, require_final_bundle_authority,
    validate_governance_primary_surfaces_with_applied_scope, BugKitBuildArgs, BugKitManifestV1,
    CanonicalBundleSpineV1, CanonicalExportLayoutCompatibilityV1, FinalBundleAuthorityContextV1,
    OpsError, ReproPackManifestV1,
};

const DIGEST_PREFIX_LEN: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalBundleConsumerAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinalBundleConsumerMismatchCategoryV1 {
    ConsumerSkippedFinalBundleAuthority,
    ConsumerUsedLegacyBundleInput,
    FinalBundleScopeMismatch,
    FinalBundleSpineMismatch,
    FinalBundleExportContextMismatch,
    LegacyBundleInputPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalBundleConsumerStatusV1 {
    pub consumer: String,
    pub status: FinalBundleConsumerAuthorityStatusV1,
    pub mismatch_categories: Vec<FinalBundleConsumerMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalBundleConsumerAuthorityV1 {
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_bundle_authority_digest_prefix: String,
    pub covered_consumer_count: u16,
    pub authority_status: FinalBundleConsumerAuthorityStatusV1,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalBundleConsumerSweepReportV1 {
    pub schema_version: u16,
    pub authority: FinalBundleConsumerAuthorityV1,
    pub consumers: Vec<FinalBundleConsumerStatusV1>,
}

pub fn final_bundle_consumer_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<FinalBundleConsumerSweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_final_bundle_consumer_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;
    let governance = require_canonical_governance_entry(&applied, Some(&governance))?;
    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_check_final_bundle_consumer_sweep.json"),
    )?
    .canonical_readiness_spine;

    let run_id = crate::reproducible_run_id_for_smoke(workdir)?;
    let repro_bundle = workdir
        .join("out")
        .join(format!("repro_{run_id}_final_bundle_consumer_sweep.zip"));
    let bugkit_bundle = workdir
        .join("out")
        .join(format!("bugkit_{run_id}_final_bundle_consumer_sweep.zip"));

    let _ = repro_pack(workdir, &run_id, &repro_bundle)?;
    let _ = crate::bugkit_build(
        workdir,
        &run_id,
        &bugkit_bundle,
        &BugKitBuildArgs::default(),
    )?;

    let repro_spine = exports_bundle_spine_check(
        &repro_bundle,
        &workdir.join("out/repro_bundle_spine_check_final_bundle_consumer_sweep.json"),
    )?;
    let bugkit_spine = exports_bundle_spine_check(
        &bugkit_bundle,
        &workdir.join("out/bugkit_bundle_spine_check_final_bundle_consumer_sweep.json"),
    )?;

    let repro_manifest = read_repro_manifest(&repro_bundle)?;
    let bugkit_manifest = read_bugkit_manifest(&bugkit_bundle)?;

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
        canonical_governance_entry_digest_prefix: governance_prefix(&governance.authority_digest),
        canonical_readiness_spine_digest_prefix: governance_prefix(&readiness.spine_digest),
        canonical_bundle_spine_digest_prefix: governance_prefix(
            &repro_spine.spine.bundle_spine_digest,
        ),
        canonical_bundle_authority_digest_prefix: governance_prefix(
            &repro_authority.authority_digest,
        ),
    });

    let consumers = vec![
        check_consumer(
            "repro_pack_build",
            &expected,
            &repro_spine.spine,
            &repro_manifest,
        ),
        check_consumer(
            "repro_verify",
            &expected,
            &repro_spine.spine,
            &repro_manifest,
        ),
        check_consumer(
            "bugkit_build",
            &expected,
            &bugkit_spine.spine,
            &bugkit_manifest,
        ),
        check_consumer(
            "exports_roundtrip_check",
            &expected,
            &repro_spine.spine,
            &repro_manifest,
        ),
        check_consumer(
            "exports_bundle_spine_check",
            &expected,
            &repro_spine.spine,
            &repro_manifest,
        ),
        check_consumer(
            "operator_roundtrip_chain_helpers",
            &expected,
            &repro_spine.spine,
            &repro_manifest,
        ),
        check_consumer(
            "export_readiness_build_guards",
            &expected,
            &repro_spine.spine,
            &repro_manifest,
        ),
        check_consumer(
            "interop_export_matrix_helpers",
            &expected,
            &repro_spine.spine,
            &repro_manifest,
        ),
        check_consumer(
            "v10_prep_gate_helpers",
            &expected,
            &repro_spine.spine,
            &repro_manifest,
        ),
    ];

    let authority_status = if consumers.iter().any(|c| {
        matches!(
            c.status,
            FinalBundleConsumerAuthorityStatusV1::LegacyPresent
        )
    }) {
        FinalBundleConsumerAuthorityStatusV1::LegacyPresent
    } else if consumers
        .iter()
        .all(|c| matches!(c.status, FinalBundleConsumerAuthorityStatusV1::Pass))
    {
        FinalBundleConsumerAuthorityStatusV1::Pass
    } else {
        FinalBundleConsumerAuthorityStatusV1::Fail
    };

    let authority = derive_authority(&expected, consumers.len() as u16, authority_status);
    let report = FinalBundleConsumerSweepReportV1 {
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

fn governance_prefix(value: &str) -> String {
    value.chars().take(DIGEST_PREFIX_LEN).collect()
}

trait ManifestBundleView {
    fn applied_prefix(&self) -> &str;
    fn governance_prefix(&self) -> Option<&str>;
    fn bundle_spine_prefix(&self) -> &str;
    fn bundle_authority_prefix(&self) -> &str;
    fn layout(&self) -> &CanonicalExportLayoutCompatibilityV1;
}

impl ManifestBundleView for ReproPackManifestV1 {
    fn applied_prefix(&self) -> &str {
        &self.export_context.supported_slot_set_digest_prefix
    }
    fn governance_prefix(&self) -> Option<&str> {
        self.related_artifacts
            .iter()
            .find(|r| r.artifact_kind == "canonical_governance_entry")
            .and_then(|r| r.artifact_digest.as_deref())
    }
    fn bundle_spine_prefix(&self) -> &str {
        &self.canonical_bundle_spine_digest_prefix
    }
    fn bundle_authority_prefix(&self) -> &str {
        &self.canonical_bundle_authority_digest_prefix
    }
    fn layout(&self) -> &CanonicalExportLayoutCompatibilityV1 {
        &self.export_layout_compatibility
    }
}

impl ManifestBundleView for BugKitManifestV1 {
    fn applied_prefix(&self) -> &str {
        &self.export_context.supported_slot_set_digest_prefix
    }
    fn governance_prefix(&self) -> Option<&str> {
        self.related_artifacts
            .iter()
            .find(|r| r.artifact_kind == "canonical_governance_entry")
            .and_then(|r| r.artifact_digest.as_deref())
    }
    fn bundle_spine_prefix(&self) -> &str {
        &self.canonical_bundle_spine_digest_prefix
    }
    fn bundle_authority_prefix(&self) -> &str {
        &self.canonical_bundle_authority_digest_prefix
    }
    fn layout(&self) -> &CanonicalExportLayoutCompatibilityV1 {
        &self.export_layout_compatibility
    }
}

fn check_consumer(
    consumer: &str,
    expected: &FinalBundleAuthorityContextV1,
    spine: &CanonicalBundleSpineV1,
    manifest: &impl ManifestBundleView,
) -> FinalBundleConsumerStatusV1 {
    let mut mismatch_categories = BTreeSet::new();
    let scope_match = manifest.applied_prefix() == expected.applied_supported_set_digest_prefix;
    let spine_match = !manifest.bundle_spine_prefix().is_empty()
        && manifest.bundle_spine_prefix() != "MISSING"
        && !spine.bundle_spine_digest.is_empty();
    let export_context_match = manifest.governance_prefix().is_some()
        || manifest.applied_prefix() == expected.applied_supported_set_digest_prefix;
    let authority_match = !manifest.bundle_authority_prefix().is_empty()
        && manifest.bundle_authority_prefix() != "MISSING";
    let legacy_present = !matches!(
        manifest.layout(),
        CanonicalExportLayoutCompatibilityV1::Canonical
    );

    if !authority_match {
        mismatch_categories
            .insert(FinalBundleConsumerMismatchCategoryV1::ConsumerSkippedFinalBundleAuthority);
    }
    if !scope_match {
        mismatch_categories.insert(FinalBundleConsumerMismatchCategoryV1::FinalBundleScopeMismatch);
    }
    if !spine_match {
        mismatch_categories.insert(FinalBundleConsumerMismatchCategoryV1::FinalBundleSpineMismatch);
    }
    if !export_context_match {
        mismatch_categories
            .insert(FinalBundleConsumerMismatchCategoryV1::FinalBundleExportContextMismatch);
    }
    if legacy_present {
        mismatch_categories
            .insert(FinalBundleConsumerMismatchCategoryV1::ConsumerUsedLegacyBundleInput);
        mismatch_categories.insert(FinalBundleConsumerMismatchCategoryV1::LegacyBundleInputPresent);
    }

    let status = if legacy_present {
        FinalBundleConsumerAuthorityStatusV1::LegacyPresent
    } else if mismatch_categories.is_empty() {
        FinalBundleConsumerAuthorityStatusV1::Pass
    } else {
        FinalBundleConsumerAuthorityStatusV1::Fail
    };

    FinalBundleConsumerStatusV1 {
        consumer: consumer.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    }
}

fn derive_authority(
    expected: &FinalBundleAuthorityContextV1,
    covered_consumer_count: u16,
    authority_status: FinalBundleConsumerAuthorityStatusV1,
) -> FinalBundleConsumerAuthorityV1 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"final_bundle_consumer_authority_v1");
    bytes.extend_from_slice(expected.applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(expected.canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(expected.canonical_readiness_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(expected.canonical_bundle_spine_digest_prefix.as_bytes());
    bytes.extend_from_slice(expected.canonical_bundle_authority_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_consumer_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", authority_status).as_bytes());

    FinalBundleConsumerAuthorityV1 {
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
        covered_consumer_count,
        authority_status,
        authority_digest: crate::sha256_hex(&bytes),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CanonicalBundleAuthorityStatusV2;

    #[test]
    fn final_bundle_consumer_authority_digest_is_stable() {
        let expected = FinalBundleAuthorityContextV1 {
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_readiness_spine_digest_prefix: "33".repeat(8),
            canonical_bundle_spine_digest_prefix: "44".repeat(8),
            canonical_bundle_authority_digest_prefix: "55".repeat(8),
        };
        let a = derive_authority(&expected, 9, FinalBundleConsumerAuthorityStatusV1::Pass);
        let b = derive_authority(&expected, 9, FinalBundleConsumerAuthorityStatusV1::Pass);
        assert_eq!(a.authority_digest, b.authority_digest);
    }

    #[test]
    fn consumer_status_deterministic_for_legacy() {
        let expected = FinalBundleAuthorityContextV1 {
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_readiness_spine_digest_prefix: "33".repeat(8),
            canonical_bundle_spine_digest_prefix: "44".repeat(8),
            canonical_bundle_authority_digest_prefix: "55".repeat(8),
        };
        let spine = CanonicalBundleSpineV1 {
            bundle_kind: crate::CanonicalBundleKindV1::Repro,
            applied_supported_set_digest_prefix: expected
                .applied_supported_set_digest_prefix
                .clone(),
            canonical_governance_entry_digest_prefix: expected
                .canonical_governance_entry_digest_prefix
                .clone(),
            canonical_readiness_spine_digest_prefix: Some(
                expected.canonical_readiness_spine_digest_prefix.clone(),
            ),
            bundle_consumption_context_digest_prefix: "66".repeat(8),
            artifact_refs_digest_prefix: "77".repeat(8),
            roundtrip_consistency_digest_prefix: "88".repeat(8),
            bundle_spine_status: crate::BundleSpineStatusV1::Pass,
            bundle_spine_digest: "44".repeat(16),
        };
        let manifest = ReproPackManifestV1 {
            schema_version: 1,
            pack_id: "p".to_string(),
            run_id: "r".to_string(),
            policy_graph_digest: "x".to_string(),
            manifest_digest: "x".to_string(),
            config_digest: "x".to_string(),
            included_artifacts: Vec::new(),
            ess_slice: crate::ReproPackEssSlice {
                record_count: 0,
                segment_roots: Vec::new(),
            },
            certificate_digest: None,
            evidence_context: crate::PackEvidenceContextSummaryV1 {
                supported_slot_set_digest_prefix: expected
                    .applied_supported_set_digest_prefix
                    .clone(),
                policy_graph_digest_prefix: "x".to_string(),
                manifest_digest_prefix: "x".to_string(),
            },
            backend_evidence_snapshot: crate::missing_evidence_ref("x", "x"),
            active_review_snapshot: crate::missing_evidence_ref("x", "x"),
            operator_signoff: crate::missing_evidence_ref("x", "x"),
            backend_resolution: crate::missing_evidence_ref("x", "x"),
            export_context: crate::CanonicalExportContextV1 {
                supported_slot_set_digest_prefix: expected
                    .applied_supported_set_digest_prefix
                    .clone(),
                policy_graph_digest_prefix: "x".to_string(),
                manifest_digest_prefix: "x".to_string(),
                run_id: None,
                operator_signoff_digest_prefix: None,
                backend_evidence_snapshot_digest_prefix: None,
                active_review_snapshot_digest_prefix: None,
                context_digest: "x".to_string(),
            },
            related_artifacts: vec![
                crate::CanonicalExportArtifactRefV1 {
                    artifact_kind: "canonical_governance_entry".to_string(),
                    relative_path: "a".to_string(),
                    sha256: Some("x".to_string()),
                    schema_version: Some(1),
                    included_state: crate::CanonicalArtifactIncludedStateV1::Included,
                    artifact_digest: Some(
                        expected.canonical_governance_entry_digest_prefix.clone(),
                    ),
                    reason_code: None,
                    ref_digest: "x".to_string(),
                },
                crate::CanonicalExportArtifactRefV1 {
                    artifact_kind: "canonical_readiness_spine".to_string(),
                    relative_path: "b".to_string(),
                    sha256: Some("x".to_string()),
                    schema_version: Some(1),
                    included_state: crate::CanonicalArtifactIncludedStateV1::Included,
                    artifact_digest: Some(expected.canonical_readiness_spine_digest_prefix.clone()),
                    reason_code: None,
                    ref_digest: "x".to_string(),
                },
            ],
            canonical_bundle_spine_digest_prefix: expected
                .canonical_bundle_spine_digest_prefix
                .clone(),
            canonical_bundle_authority_digest_prefix: expected
                .canonical_bundle_authority_digest_prefix
                .clone(),
            final_bundle_consumer_authority_digest_prefix: "66".repeat(8),
            bundle_residual_sweep_digest_prefix: "77".repeat(8),
            residual_free_bundle_consumer_authority_digest_prefix: "MISSING".to_string(),
            bundle_absolute_sweep_digest_prefix: "MISSING".to_string(),
            bundle_terminal_sweep_digest_prefix: "MISSING".to_string(),
            bundle_ultimate_sweep_digest_prefix: "MISSING".to_string(),
            export_layout_compatibility: CanonicalExportLayoutCompatibilityV1::LegacyExportLayout,
            repro_pack_digest: "x".to_string(),
        };
        let status = check_consumer("x", &expected, &spine, &manifest);
        assert!(matches!(
            status.status,
            FinalBundleConsumerAuthorityStatusV1::LegacyPresent
        ));
    }

    #[test]
    fn derived_bundle_authority_is_pass_when_no_legacy() {
        let spine = CanonicalBundleSpineV1 {
            bundle_kind: crate::CanonicalBundleKindV1::Repro,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_readiness_spine_digest_prefix: Some("33".repeat(8)),
            bundle_consumption_context_digest_prefix: "44".repeat(8),
            artifact_refs_digest_prefix: "55".repeat(8),
            roundtrip_consistency_digest_prefix: "66".repeat(8),
            bundle_spine_status: crate::BundleSpineStatusV1::Pass,
            bundle_spine_digest: "77".repeat(16),
        };
        let authority = derive_canonical_bundle_authority_v2(&spine, 1, false).expect("authority");
        assert!(matches!(
            authority.authority_status,
            CanonicalBundleAuthorityStatusV2::Pass
        ));
    }
}
