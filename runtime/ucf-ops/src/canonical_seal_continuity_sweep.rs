use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    bundle_seal_sweep, derive_canonical_governance_entry, governance_seal_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_roundtrip_chain_check, operator_signoff,
    operator_workflow_chain, prefix_hex, primary_semantics_seal_sweep, readiness_seal_sweep,
    validate_governance_primary_surfaces_with_applied_scope, BundleSealStatusV1,
    CanonicalClosureContinuityStatusV1, CanonicalRoundTripChainStatusV1, GovernanceSealStatusV1,
    OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
    PrimarySemanticsSealStatusV1, ReadinessSealStatusV1,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CODE_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalSealContinuityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalSealContinuityAuthorityV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub governance_seal_sweep_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub readiness_seal_sweep_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub bundle_seal_sweep_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub primary_semantics_seal_sweep_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub canonical_roundtrip_chain_digest_prefix: String,
    pub canonical_closure_continuity_authority_digest_prefix: String,
    pub continuity_status: CanonicalSealContinuityStatusV1,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub authority_digest: String,
}

pub fn canonical_seal_continuity_sweep(
    workdir: &Path,
    bundle: &Path,
    out: &Path,
) -> Result<CanonicalSealContinuityAuthorityV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_canonical_seal_continuity_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;

    let governance_seal = governance_seal_sweep(
        workdir,
        &workdir.join("out/governance_seal_sweep_canonical_seal_continuity_sweep.json"),
    )?;
    let readiness_seal = readiness_seal_sweep(
        workdir,
        &workdir.join("out/readiness_seal_sweep_canonical_seal_continuity_sweep.json"),
    )?;
    let bundle_seal = bundle_seal_sweep(
        workdir,
        &workdir.join("out/bundle_seal_sweep_canonical_seal_continuity_sweep.json"),
    )?;
    let primary_seal = primary_semantics_seal_sweep(
        workdir,
        &workdir.join("out/primary_semantics_seal_sweep_canonical_seal_continuity_sweep.json"),
    )?;
    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_canonical_seal_continuity_sweep.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_canonical_seal_continuity_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_canonical_seal_continuity_sweep.json"),
    )?;
    let roundtrip = operator_roundtrip_chain_check(
        workdir,
        bundle,
        &workdir.join("out/operator_roundtrip_chain_canonical_seal_continuity_sweep.json"),
    )?;

    let closure: crate::CanonicalClosureContinuityAuthorityV1 = serde_json::from_slice(&fs::read(
        workdir.join("out/canonical_closure_continuity_sweep.json"),
    )?)?;

    let mut blocking = BTreeSet::new();
    let mut remediation = BTreeSet::new();
    let expected_applied = &applied.applied_set_digest_prefix;

    if governance_seal.sweep.applied_supported_set_digest_prefix != *expected_applied
        || readiness_seal.sweep.applied_supported_set_digest_prefix != *expected_applied
        || bundle_seal.sweep.applied_supported_set_digest_prefix != *expected_applied
        || review_packet.applied_supported_set_digest_prefix != *expected_applied
        || signoff.applied_supported_set_digest_prefix != *expected_applied
        || workflow.applied_supported_set_digest_prefix != *expected_applied
        || roundtrip.applied_supported_set_digest_prefix != *expected_applied
    {
        blocking.insert("SEAL_SCOPE_MISMATCH");
        remediation.insert("run_models_applied_scope_check");
    }

    let governance_entry_prefix = prefix_hex(&governance.authority_digest, DIGEST_PREFIX_LEN);
    if governance_seal
        .sweep
        .canonical_governance_entry_digest_prefix
        != governance_entry_prefix
        || !matches!(
            governance_seal.sweep.seal_status,
            GovernanceSealStatusV1::Pass
        )
    {
        blocking.insert("SEAL_GOVERNANCE_MISMATCH");
        remediation.insert("run_governance_seal_sweep");
    }

    if !matches!(
        readiness_seal.sweep.seal_status,
        ReadinessSealStatusV1::Pass
    ) {
        blocking.insert("SEAL_READINESS_MISMATCH");
        remediation.insert("run_readiness_seal_sweep");
    }
    if !matches!(
        primary_seal.sweep.seal_status,
        PrimarySemanticsSealStatusV1::Pass
    ) {
        blocking.insert("SEAL_PRIMARY_SEMANTICS_MISMATCH");
        remediation.insert("run_primary_semantics_seal_sweep");
    }
    if workflow.canonical_seal_continuity_authority_digest_prefix == "MISSING"
        || !workflow.blocking_codes.is_empty()
    {
        blocking.insert("SEAL_WORKFLOW_MISMATCH");
        remediation.insert("run_operator_workflow_chain");
    }
    if !matches!(
        roundtrip.roundtrip_status,
        CanonicalRoundTripChainStatusV1::Pass
    ) || !matches!(bundle_seal.sweep.seal_status, BundleSealStatusV1::Pass)
    {
        blocking.insert("SEAL_BUNDLE_MISMATCH");
        remediation.insert("run_bundle_seal_sweep");
    }

    if governance_seal.sweep.residual_path_count > 0
        || readiness_seal.sweep.residual_path_count > 0
        || bundle_seal.sweep.residual_path_count > 0
        || primary_seal.sweep.residual_path_count > 0
    {
        blocking.insert("RESIDUAL_PATH_DEPENDENCY_PRESENT");
        remediation.insert("remove_residual_path_dependencies");
    }

    let legacy_present = matches!(
        governance_seal.sweep.seal_status,
        GovernanceSealStatusV1::LegacyPresent
    ) || matches!(
        readiness_seal.sweep.seal_status,
        ReadinessSealStatusV1::LegacyPresent
    ) || matches!(
        bundle_seal.sweep.seal_status,
        BundleSealStatusV1::LegacyPresent
    ) || matches!(
        primary_seal.sweep.seal_status,
        PrimarySemanticsSealStatusV1::LegacyPresent
    ) || !matches!(
        closure.continuity_status,
        CanonicalClosureContinuityStatusV1::Pass
    );
    if legacy_present {
        blocking.insert("LEGACY_TOP_LEVEL_CONTINUITY_PRESENT");
        remediation.insert("demote_legacy_top_level_continuity_surfaces");
    }

    let mut report = CanonicalSealContinuityAuthorityV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: governance_entry_prefix,
        governance_seal_sweep_digest_prefix: prefix_hex(
            &governance_seal.sweep.seal_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_readiness_spine_digest_prefix: readiness_seal
            .sweep
            .canonical_readiness_spine_digest_prefix,
        readiness_seal_sweep_digest_prefix: prefix_hex(
            &readiness_seal.sweep.seal_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_bundle_spine_digest_prefix: bundle_seal
            .sweep
            .canonical_bundle_spine_digest_prefix,
        bundle_seal_sweep_digest_prefix: prefix_hex(
            &bundle_seal.sweep.seal_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_primary_semantics_authority_digest_prefix: primary_seal
            .sweep
            .canonical_primary_semantics_authority_digest_prefix,
        primary_semantics_seal_sweep_digest_prefix: prefix_hex(
            &primary_seal.sweep.seal_digest,
            DIGEST_PREFIX_LEN,
        ),
        operator_review_packet_digest_prefix: prefix_hex(
            &review_packet.packet_digest,
            DIGEST_PREFIX_LEN,
        ),
        operator_signoff_digest_prefix: prefix_hex(&signoff.decision_digest, DIGEST_PREFIX_LEN),
        operator_workflow_chain_digest_prefix: prefix_hex(
            &workflow.chain_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_roundtrip_chain_digest_prefix: prefix_hex(
            &roundtrip.chain_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_closure_continuity_authority_digest_prefix: prefix_hex(
            &closure.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        continuity_status: if legacy_present {
            CanonicalSealContinuityStatusV1::LegacyPresent
        } else if blocking.is_empty() {
            CanonicalSealContinuityStatusV1::Pass
        } else {
            CanonicalSealContinuityStatusV1::Fail
        },
        blocking_codes: blocking
            .into_iter()
            .take(CODE_CAP)
            .map(String::from)
            .collect(),
        remediation_codes: remediation
            .into_iter()
            .take(CODE_CAP)
            .map(String::from)
            .collect(),
        authority_digest: String::new(),
    };
    report.authority_digest = crate::sha256_hex(&serde_json::to_vec(&{
        let mut digestible = report.clone();
        digestible.authority_digest.clear();
        digestible
    })?);

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
