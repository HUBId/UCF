use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    bundle_closure_sweep, derive_canonical_governance_entry, governance_closure_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_roundtrip_chain_check, operator_signoff,
    operator_workflow_chain, prefix_hex, primary_semantics_closure_sweep, readiness_closure_sweep,
    validate_governance_primary_surfaces_with_applied_scope,
    CanonicalFinalConsolidationContinuityStatusV1, CanonicalRoundTripChainStatusV1,
    OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CODE_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalClosureContinuityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalClosureContinuityAuthorityV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub governance_closure_sweep_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub readiness_closure_sweep_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub bundle_closure_sweep_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub primary_semantics_closure_sweep_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub canonical_roundtrip_chain_digest_prefix: String,
    pub canonical_final_consolidation_continuity_authority_digest_prefix: String,
    pub continuity_status: CanonicalClosureContinuityStatusV1,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CanonicalFinalConsolidationProbeV1 {
    continuity_status: CanonicalFinalConsolidationContinuityStatusV1,
    authority_digest: String,
}

pub fn canonical_closure_continuity_sweep(
    workdir: &Path,
    bundle: &Path,
    out: &Path,
) -> Result<CanonicalClosureContinuityAuthorityV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_canonical_closure_continuity_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;

    let governance_closure = governance_closure_sweep(
        workdir,
        &workdir.join("out/governance_closure_sweep_canonical_closure_continuity_sweep.json"),
    )?;
    let readiness_closure = readiness_closure_sweep(
        workdir,
        &workdir.join("out/readiness_closure_sweep_canonical_closure_continuity_sweep.json"),
    )?;
    let bundle_closure = bundle_closure_sweep(
        workdir,
        &workdir.join("out/bundle_closure_sweep_canonical_closure_continuity_sweep.json"),
    )?;
    let primary_closure = primary_semantics_closure_sweep(
        workdir,
        &workdir
            .join("out/primary_semantics_closure_sweep_canonical_closure_continuity_sweep.json"),
    )?;

    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_canonical_closure_continuity_sweep.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_canonical_closure_continuity_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_canonical_closure_continuity_sweep.json"),
    )?;
    let roundtrip = operator_roundtrip_chain_check(
        workdir,
        bundle,
        &workdir.join("out/operator_roundtrip_chain_canonical_closure_continuity_sweep.json"),
    )?;

    let final_consolidation_path =
        workdir.join("out/canonical_final_consolidation_continuity_sweep.json");
    let final_probe: CanonicalFinalConsolidationProbeV1 =
        serde_json::from_slice(&fs::read(final_consolidation_path)?)?;

    let mut blocking = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    let expected_applied = &applied.applied_set_digest_prefix;
    if governance_closure.sweep.applied_supported_set_digest_prefix != *expected_applied
        || readiness_closure.sweep.applied_supported_set_digest_prefix != *expected_applied
        || bundle_closure.sweep.applied_supported_set_digest_prefix != *expected_applied
        || review_packet.applied_supported_set_digest_prefix != *expected_applied
        || signoff.applied_supported_set_digest_prefix != *expected_applied
        || workflow.applied_supported_set_digest_prefix != *expected_applied
        || roundtrip.applied_supported_set_digest_prefix != *expected_applied
    {
        blocking.insert("CLOSURE_SCOPE_MISMATCH");
        remediation.insert("run_models_applied_scope_check");
    }

    let governance_entry_prefix = prefix_hex(&governance.authority_digest, DIGEST_PREFIX_LEN);
    if governance_closure
        .sweep
        .canonical_governance_entry_digest_prefix
        != governance_entry_prefix
        || !matches!(
            governance_closure.sweep.closure_status,
            crate::GovernanceClosureStatusV1::Pass
        )
    {
        blocking.insert("CLOSURE_GOVERNANCE_MISMATCH");
        remediation.insert("run_governance_closure_sweep");
    }

    if !matches!(
        readiness_closure.sweep.closure_status,
        crate::ReadinessClosureStatusV1::Pass
    ) {
        blocking.insert("CLOSURE_READINESS_MISMATCH");
        remediation.insert("run_readiness_closure_sweep");
    }

    if !matches!(
        primary_closure.sweep.closure_status,
        crate::PrimarySemanticsClosureStatusV1::Pass
    ) {
        blocking.insert("CLOSURE_PRIMARY_SEMANTICS_MISMATCH");
        remediation.insert("run_primary_semantics_closure_sweep");
    }

    if !workflow.blocking_codes.is_empty()
        || workflow.canonical_closure_continuity_authority_digest_prefix == "MISSING"
    {
        blocking.insert("CLOSURE_WORKFLOW_MISMATCH");
        remediation.insert("run_operator_workflow_chain");
    }

    if !matches!(
        roundtrip.roundtrip_status,
        CanonicalRoundTripChainStatusV1::Pass
    ) || !matches!(
        bundle_closure.sweep.closure_status,
        crate::BundleClosureStatusV1::Pass
    ) {
        blocking.insert("CLOSURE_BUNDLE_MISMATCH");
        remediation.insert("run_bundle_closure_sweep");
    }

    if governance_closure.sweep.residual_path_count > 0
        || readiness_closure.sweep.residual_path_count > 0
        || bundle_closure.sweep.residual_path_count > 0
        || primary_closure.sweep.residual_path_count > 0
    {
        blocking.insert("RESIDUAL_PATH_DEPENDENCY_PRESENT");
        remediation.insert("remove_residual_path_dependencies");
    }

    let legacy_present = matches!(
        governance_closure.sweep.closure_status,
        crate::GovernanceClosureStatusV1::LegacyPresent
    ) || matches!(
        readiness_closure.sweep.closure_status,
        crate::ReadinessClosureStatusV1::LegacyPresent
    ) || matches!(
        bundle_closure.sweep.closure_status,
        crate::BundleClosureStatusV1::LegacyPresent
    ) || matches!(
        primary_closure.sweep.closure_status,
        crate::PrimarySemanticsClosureStatusV1::LegacyPresent
    ) || !matches!(
        final_probe.continuity_status,
        CanonicalFinalConsolidationContinuityStatusV1::Pass
    );

    if legacy_present {
        blocking.insert("LEGACY_TOP_LEVEL_CONTINUITY_PRESENT");
        remediation.insert("demote_legacy_top_level_continuity_surfaces");
    }

    let mut report = CanonicalClosureContinuityAuthorityV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: governance_entry_prefix,
        governance_closure_sweep_digest_prefix: prefix_hex(
            &governance_closure.sweep.closure_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_readiness_spine_digest_prefix: readiness_closure
            .sweep
            .canonical_readiness_spine_digest_prefix,
        readiness_closure_sweep_digest_prefix: prefix_hex(
            &readiness_closure.sweep.closure_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_bundle_spine_digest_prefix: bundle_closure
            .sweep
            .canonical_bundle_spine_digest_prefix,
        bundle_closure_sweep_digest_prefix: prefix_hex(
            &bundle_closure.sweep.closure_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_primary_semantics_authority_digest_prefix: primary_closure
            .sweep
            .canonical_primary_semantics_authority_digest_prefix,
        primary_semantics_closure_sweep_digest_prefix: prefix_hex(
            &primary_closure.sweep.closure_digest,
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
        canonical_final_consolidation_continuity_authority_digest_prefix: prefix_hex(
            &final_probe.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        continuity_status: if legacy_present {
            CanonicalClosureContinuityStatusV1::LegacyPresent
        } else if blocking.is_empty() {
            CanonicalClosureContinuityStatusV1::Pass
        } else {
            CanonicalClosureContinuityStatusV1::Fail
        },
        blocking_codes: blocking
            .into_iter()
            .map(str::to_string)
            .take(CODE_CAP)
            .collect(),
        remediation_codes: remediation
            .into_iter()
            .map(str::to_string)
            .take(CODE_CAP)
            .collect(),
        authority_digest: String::new(),
    };

    report.authority_digest = canonical_closure_continuity_digest(&report)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn canonical_closure_continuity_digest(
    report: &CanonicalClosureContinuityAuthorityV1,
) -> Result<String, OpsError> {
    let mut clone = report.clone();
    clone.authority_digest.clear();
    Ok(crate::sha256_hex(&serde_json::to_vec(&clone)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_closure_continuity_digest_stable() {
        let mut report = CanonicalClosureContinuityAuthorityV1 {
            schema_version: 1,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            governance_closure_sweep_digest_prefix: "33".repeat(8),
            canonical_readiness_spine_digest_prefix: "44".repeat(8),
            readiness_closure_sweep_digest_prefix: "55".repeat(8),
            canonical_bundle_spine_digest_prefix: "66".repeat(8),
            bundle_closure_sweep_digest_prefix: "77".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "88".repeat(8),
            primary_semantics_closure_sweep_digest_prefix: "99".repeat(8),
            operator_review_packet_digest_prefix: "aa".repeat(8),
            operator_signoff_digest_prefix: "bb".repeat(8),
            operator_workflow_chain_digest_prefix: "cc".repeat(8),
            canonical_roundtrip_chain_digest_prefix: "dd".repeat(8),
            canonical_final_consolidation_continuity_authority_digest_prefix: "ee".repeat(8),
            continuity_status: CanonicalClosureContinuityStatusV1::Pass,
            blocking_codes: vec!["A".to_string()],
            remediation_codes: vec!["B".to_string()],
            authority_digest: String::new(),
        };

        report.authority_digest = canonical_closure_continuity_digest(&report).expect("digest");
        let stable = canonical_closure_continuity_digest(&report).expect("stable");
        assert_eq!(stable, report.authority_digest);
    }
}
