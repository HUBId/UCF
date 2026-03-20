use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, exports_bundle_spine_check, exports_roundtrip_check,
    governance_entry_sweep, load_applied_supported_set_context_v1, models_active_review_snapshot,
    models_evidence_snapshot, operator_export_chain_check, operator_review_packet,
    operator_signoff, operator_workflow_chain, prefix_hex, readiness_spine_check,
    readiness_spine_sweep, require_canonical_governance_entry,
    validate_governance_primary_surfaces_with_applied_scope, BugKitManifestV1, BundleSpineStatusV1,
    OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
    ReproPackManifestV1, SignoffDecisionStateV1,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CODE_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalRoundTripChainStatusV1 {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalRoundTripChainV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub operator_export_authority_chain_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: Option<String>,
    pub canonical_bundle_authority_digest_prefix: Option<String>,
    pub roundtrip_status: CanonicalRoundTripChainStatusV1,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub chain_digest: String,
}

pub fn operator_roundtrip_chain_check(
    workdir: &Path,
    bundle: &Path,
    out: &Path,
) -> Result<CanonicalRoundTripChainV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_roundtrip_chain_check.json"),
    )?;
    let governance_surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance_entry = derive_canonical_governance_entry(&applied, &governance_surfaces)?;
    let governance_entry = require_canonical_governance_entry(&applied, Some(&governance_entry))?;
    let governance_sweep = governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_roundtrip_chain_check.json"),
    )?;

    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_roundtrip_chain_check.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_roundtrip_chain_check.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_roundtrip_chain_check.json"),
    )?;
    let export_authority = operator_export_chain_check(
        workdir,
        &workdir.join("out/operator_export_chain_roundtrip_chain_check.json"),
    )?;
    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_roundtrip_chain_check.json"),
    )?;
    let readiness_sweep = readiness_spine_sweep(
        workdir,
        &workdir.join("out/readiness_spine_sweep_roundtrip_chain_check.json"),
    )?;

    let roundtrip = exports_roundtrip_check(
        bundle,
        &workdir.join("out/export_roundtrip_chain_check.json"),
    )?;
    let bundle_spine_report = exports_bundle_spine_check(
        bundle,
        &workdir.join("out/bundle_spine_roundtrip_chain_check.json"),
    )?;
    let bundle_refs = extract_bundle_chain_refs(bundle)?;

    let mut blocking = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    if bundle_refs.canonical_governance_entry_digest_prefix
        != prefix_hex(&governance_entry.authority_digest, DIGEST_PREFIX_LEN)
    {
        blocking.insert("ROUNDTRIP_CHAIN_GOVERNANCE_ENTRY_MISMATCH".to_string());
        remediation.insert("run_governance_entry_check".to_string());
    }

    let readiness_expected = prefix_hex(
        &readiness.canonical_readiness_spine.spine_digest,
        DIGEST_PREFIX_LEN,
    );
    if bundle_refs.canonical_readiness_spine_digest_prefix != readiness_expected {
        blocking.insert("ROUNDTRIP_CHAIN_READINESS_SPINE_MISMATCH".to_string());
        remediation.insert("run_readiness_spine_check".to_string());
    }

    let packet_expected = prefix_hex(&review_packet.packet_digest, DIGEST_PREFIX_LEN);
    if bundle_refs.operator_review_packet_digest_prefix != packet_expected {
        blocking.insert("ROUNDTRIP_CHAIN_REVIEW_PACKET_MISMATCH".to_string());
        remediation.insert("run_operator_review_packet".to_string());
    }

    let signoff_expected = prefix_hex(&signoff.decision_digest, DIGEST_PREFIX_LEN);
    if bundle_refs.operator_signoff_digest_prefix != signoff_expected {
        blocking.insert("ROUNDTRIP_CHAIN_SIGNOFF_MISMATCH".to_string());
        remediation.insert("run_operator_signoff".to_string());
    }

    let workflow_expected = prefix_hex(&workflow.chain_digest, DIGEST_PREFIX_LEN);
    if bundle_refs.operator_workflow_chain_digest_prefix != workflow_expected {
        blocking.insert("ROUNDTRIP_CHAIN_WORKFLOW_MISMATCH".to_string());
        remediation.insert("run_operator_workflow".to_string());
    }

    let export_expected = prefix_hex(&export_authority.chain_digest, DIGEST_PREFIX_LEN);
    if bundle_refs.operator_export_authority_chain_digest_prefix != export_expected {
        blocking.insert("ROUNDTRIP_CHAIN_EXPORT_AUTHORITY_MISMATCH".to_string());
        remediation.insert("run_operator_export_chain_check".to_string());
    }

    if !matches!(
        roundtrip.overall_status,
        crate::BundleRoundTripOverallStatusV1::Pass
    ) {
        blocking.insert("ROUNDTRIP_CHAIN_EXPORT_ROUNDTRIP_FAIL".to_string());
        remediation.insert("run_exports_roundtrip_check".to_string());
    }

    if !bundle_spine_report.pass
        || !matches!(
            bundle_spine_report.spine.bundle_spine_status,
            BundleSpineStatusV1::Pass
        )
    {
        blocking.insert("ROUNDTRIP_CHAIN_BUNDLE_SPINE_FAIL".to_string());
        remediation.insert("run_exports_bundle_spine_check".to_string());
    }
    if bundle_refs.canonical_bundle_authority_digest_prefix
        != bundle_spine_report
            .authority_digest_prefix
            .clone()
            .unwrap_or_else(|| "MISSING".to_string())
    {
        blocking.insert("ROUNDTRIP_CHAIN_BUNDLE_AUTHORITY_MISMATCH".to_string());
        remediation.insert("run_exports_bundle_spine_check".to_string());
    }

    if matches!(signoff.decision, SignoffDecisionStateV1::NotReady) {
        blocking.insert("ROUNDTRIP_CHAIN_OPERATOR_NOT_READY".to_string());
        remediation.insert("run_operator_signoff".to_string());
    }

    let mut chain = CanonicalRoundTripChainV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: prefix_hex(
            &governance_entry.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_governance_authority_digest_prefix: prefix_hex(
            &governance_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_readiness_spine_digest_prefix: readiness_expected,
        canonical_readiness_authority_digest_prefix: prefix_hex(
            &readiness_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        operator_review_packet_digest_prefix: packet_expected,
        operator_signoff_digest_prefix: signoff_expected,
        operator_workflow_chain_digest_prefix: workflow_expected,
        operator_export_authority_chain_digest_prefix: export_expected,
        canonical_bundle_spine_digest_prefix: Some(prefix_hex(
            &bundle_spine_report.spine.bundle_spine_digest,
            DIGEST_PREFIX_LEN,
        )),
        canonical_bundle_authority_digest_prefix: bundle_spine_report.authority_digest_prefix,
        roundtrip_status: if blocking.is_empty() {
            CanonicalRoundTripChainStatusV1::Pass
        } else {
            CanonicalRoundTripChainStatusV1::Fail
        },
        blocking_codes: blocking.into_iter().take(CODE_CAP).collect(),
        remediation_codes: remediation.into_iter().take(CODE_CAP).collect(),
        chain_digest: String::new(),
    };
    chain.chain_digest = chain_digest(&chain)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&chain)?)?;
    Ok(chain)
}

#[derive(Debug, Clone)]
struct BundleChainRefs {
    canonical_governance_entry_digest_prefix: String,
    canonical_readiness_spine_digest_prefix: String,
    operator_review_packet_digest_prefix: String,
    operator_signoff_digest_prefix: String,
    operator_workflow_chain_digest_prefix: String,
    operator_export_authority_chain_digest_prefix: String,
    canonical_bundle_authority_digest_prefix: String,
}

fn extract_bundle_chain_refs(bundle: &Path) -> Result<BundleChainRefs, OpsError> {
    let file = fs::File::open(bundle)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| OpsError::Invalid(format!("unable to open bundle zip: {e}")))?;

    if archive.by_name("repro_pack_manifest.json").is_ok() {
        let mut body = String::new();
        let mut mf = archive
            .by_name("repro_pack_manifest.json")
            .map_err(|e| OpsError::Invalid(format!("missing repro_pack_manifest.json: {e}")))?;
        std::io::Read::read_to_string(&mut mf, &mut body)?;
        let manifest: ReproPackManifestV1 = serde_json::from_str(&body)?;
        return Ok(refs_from_related(
            &manifest.related_artifacts,
            &manifest.operator_signoff.digest_prefix,
            &manifest.canonical_bundle_authority_digest_prefix,
        ));
    }

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
    let manifest: BugKitManifestV1 = serde_json::from_str(&body)?;
    Ok(refs_from_related(
        &manifest.related_artifacts,
        &manifest.operator_signoff.digest_prefix,
        &manifest.canonical_bundle_authority_digest_prefix,
    ))
}

fn refs_from_related(
    related: &[crate::CanonicalExportArtifactRefV1],
    signoff_fallback: &str,
    bundle_authority_fallback: &str,
) -> BundleChainRefs {
    let find = |kind: &str| {
        related
            .iter()
            .find(|r| r.artifact_kind == kind)
            .and_then(|r| r.artifact_digest.clone())
            .unwrap_or_else(|| "MISSING".to_string())
    };

    BundleChainRefs {
        canonical_governance_entry_digest_prefix: find("canonical_governance_entry"),
        canonical_readiness_spine_digest_prefix: find("canonical_readiness_spine"),
        operator_review_packet_digest_prefix: find("operator_review_packet"),
        operator_signoff_digest_prefix: {
            let from_chain = find("operator_signoff_decision");
            if from_chain == "MISSING" {
                if signoff_fallback.is_empty() {
                    "MISSING".to_string()
                } else {
                    signoff_fallback.to_string()
                }
            } else {
                from_chain
            }
        },
        operator_workflow_chain_digest_prefix: find("operator_workflow_chain"),
        operator_export_authority_chain_digest_prefix: find("operator_export_authority_chain"),
        canonical_bundle_authority_digest_prefix: if bundle_authority_fallback.is_empty() {
            "MISSING".to_string()
        } else {
            bundle_authority_fallback.to_string()
        },
    }
}

fn chain_digest(chain: &CanonicalRoundTripChainV1) -> Result<String, OpsError> {
    let mut canonical = chain.clone();
    canonical.chain_digest.clear();
    canonical.blocking_codes.sort();
    canonical.remediation_codes.sort();
    Ok(crate::sha256_hex(&serde_json::to_vec(&canonical)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_chain_digest_stable_for_same_inputs() {
        let mut chain = CanonicalRoundTripChainV1 {
            schema_version: 1,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_governance_authority_digest_prefix: "23".repeat(8),
            canonical_readiness_spine_digest_prefix: "33".repeat(8),
            canonical_readiness_authority_digest_prefix: "34".repeat(8),
            operator_review_packet_digest_prefix: "44".repeat(8),
            operator_signoff_digest_prefix: "55".repeat(8),
            operator_workflow_chain_digest_prefix: "66".repeat(8),
            operator_export_authority_chain_digest_prefix: "77".repeat(8),
            canonical_bundle_spine_digest_prefix: Some("88".repeat(8)),
            canonical_bundle_authority_digest_prefix: Some("89".repeat(8)),
            roundtrip_status: CanonicalRoundTripChainStatusV1::Pass,
            blocking_codes: vec![],
            remediation_codes: vec!["run_x".to_string()],
            chain_digest: String::new(),
        };
        chain.chain_digest = chain_digest(&chain).expect("digest");
        let digest_a = chain.chain_digest.clone();
        let digest_b = chain_digest(&chain).expect("digest b");
        assert_eq!(digest_a, digest_b);
    }
}
