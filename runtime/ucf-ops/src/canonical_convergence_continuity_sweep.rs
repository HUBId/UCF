use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    bundle_convergence_sweep, derive_canonical_governance_entry, governance_convergence_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_roundtrip_chain_check, operator_signoff,
    operator_workflow_chain, prefix_hex, primary_semantics_convergence_sweep,
    readiness_convergence_sweep, ultimate_terminal_absolute_final_input_continuity_sweep,
    validate_governance_primary_surfaces_with_applied_scope, BundleConvergenceStatusV1,
    CanonicalRoundTripChainStatusV1, GovernanceConvergenceStatusV1, OperatorReviewPacketArgs,
    OperatorSignoffArgs, OperatorWorkflowArgs, OpsError, PrimarySemanticsConvergenceStatusV1,
    ReadinessConvergenceStatusV1, UltimateTerminalAbsoluteFinalInputContinuityStatusV1,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CODE_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalConvergenceContinuityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalConvergenceContinuityAuthorityV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub governance_convergence_sweep_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub readiness_convergence_sweep_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub bundle_convergence_sweep_digest_prefix: String,
    pub canonical_primary_semantics_authority_digest_prefix: String,
    pub primary_semantics_convergence_sweep_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub canonical_roundtrip_chain_digest_prefix: String,
    pub ultimate_terminal_absolute_final_input_continuity_authority_digest_prefix: String,
    pub continuity_status: CanonicalConvergenceContinuityStatusV1,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub authority_digest: String,
}

pub fn canonical_convergence_continuity_sweep(
    workdir: &Path,
    bundle: &Path,
    out: &Path,
) -> Result<CanonicalConvergenceContinuityAuthorityV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_canonical_convergence_continuity_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;

    let governance_convergence = governance_convergence_sweep(
        workdir,
        &workdir
            .join("out/governance_convergence_sweep_canonical_convergence_continuity_sweep.json"),
    )?;
    let readiness_convergence = readiness_convergence_sweep(
        workdir,
        &workdir
            .join("out/readiness_convergence_sweep_canonical_convergence_continuity_sweep.json"),
    )?;
    let bundle_convergence = bundle_convergence_sweep(
        workdir,
        &workdir.join("out/bundle_convergence_sweep_canonical_convergence_continuity_sweep.json"),
    )?;
    let primary_convergence = primary_semantics_convergence_sweep(
        workdir,
        &workdir.join(
            "out/primary_semantics_convergence_sweep_canonical_convergence_continuity_sweep.json",
        ),
    )?;

    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_canonical_convergence_continuity_sweep.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_canonical_convergence_continuity_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_canonical_convergence_continuity_sweep.json"),
    )?;
    let roundtrip = operator_roundtrip_chain_check(
        workdir,
        bundle,
        &workdir.join("out/operator_roundtrip_chain_canonical_convergence_continuity_sweep.json"),
    )?;
    let ultimate = ultimate_terminal_absolute_final_input_continuity_sweep(
        workdir,
        bundle,
        &workdir.join(
            "out/ultimate_terminal_absolute_final_input_continuity_sweep_canonical_convergence_continuity_sweep.json",
        ),
    )?;

    let mut blocking = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    let expected_applied = &applied.applied_set_digest_prefix;
    if governance_convergence
        .sweep
        .applied_supported_set_digest_prefix
        != *expected_applied
        || readiness_convergence
            .sweep
            .applied_supported_set_digest_prefix
            != *expected_applied
        || bundle_convergence.sweep.applied_supported_set_digest_prefix != *expected_applied
        || review_packet.applied_supported_set_digest_prefix != *expected_applied
        || signoff.applied_supported_set_digest_prefix != *expected_applied
        || workflow.applied_supported_set_digest_prefix != *expected_applied
    {
        blocking.insert("CONVERGENCE_SCOPE_MISMATCH");
        remediation.insert("run_models_applied_scope_check");
    }

    let governance_entry_prefix = prefix_hex(&governance.authority_digest, DIGEST_PREFIX_LEN);
    if governance_convergence
        .sweep
        .canonical_governance_entry_digest_prefix
        != governance_entry_prefix
        || !matches!(
            governance_convergence.sweep.convergence_status,
            GovernanceConvergenceStatusV1::Pass
        )
    {
        blocking.insert("CONVERGENCE_GOVERNANCE_MISMATCH");
        remediation.insert("run_governance_convergence_sweep");
    }

    if !matches!(
        readiness_convergence.sweep.convergence_status,
        ReadinessConvergenceStatusV1::Pass
    ) {
        blocking.insert("CONVERGENCE_READINESS_MISMATCH");
        remediation.insert("run_readiness_convergence_sweep");
    }

    if !matches!(
        primary_convergence.sweep.convergence_status,
        PrimarySemanticsConvergenceStatusV1::Pass
    ) {
        blocking.insert("CONVERGENCE_PRIMARY_SEMANTICS_MISMATCH");
        remediation.insert("run_primary_semantics_convergence_sweep");
    }

    if !workflow.blocking_codes.is_empty() {
        blocking.insert("CONVERGENCE_WORKFLOW_MISMATCH");
        remediation.insert("run_operator_workflow_chain");
    }

    if !matches!(
        roundtrip.roundtrip_status,
        CanonicalRoundTripChainStatusV1::Pass
    ) || !matches!(
        bundle_convergence.sweep.convergence_status,
        BundleConvergenceStatusV1::Pass
    ) {
        blocking.insert("CONVERGENCE_BUNDLE_MISMATCH");
        remediation.insert("run_bundle_convergence_sweep");
    }

    if governance_convergence.sweep.residual_path_count > 0
        || readiness_convergence.sweep.residual_path_count > 0
        || bundle_convergence.sweep.residual_path_count > 0
        || primary_convergence.sweep.residual_path_count > 0
    {
        blocking.insert("RESIDUAL_PATH_DEPENDENCY_PRESENT");
        remediation.insert("remove_residual_path_dependencies");
    }

    let legacy_present = matches!(
        governance_convergence.sweep.convergence_status,
        GovernanceConvergenceStatusV1::LegacyPresent
    ) || matches!(
        readiness_convergence.sweep.convergence_status,
        ReadinessConvergenceStatusV1::LegacyPresent
    ) || matches!(
        bundle_convergence.sweep.convergence_status,
        BundleConvergenceStatusV1::LegacyPresent
    ) || matches!(
        primary_convergence.sweep.convergence_status,
        PrimarySemanticsConvergenceStatusV1::LegacyPresent
    ) || matches!(
        ultimate.continuity_status,
        UltimateTerminalAbsoluteFinalInputContinuityStatusV1::LegacyPresent
    );
    if legacy_present {
        blocking.insert("LEGACY_TOP_LEVEL_CONTINUITY_PRESENT");
        remediation.insert("demote_legacy_top_level_continuity_surfaces");
    }

    let mut report = CanonicalConvergenceContinuityAuthorityV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: governance_entry_prefix,
        governance_convergence_sweep_digest_prefix: prefix_hex(
            &governance_convergence.sweep.convergence_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_readiness_spine_digest_prefix: readiness_convergence
            .sweep
            .canonical_readiness_spine_digest_prefix,
        readiness_convergence_sweep_digest_prefix: prefix_hex(
            &readiness_convergence.sweep.convergence_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_bundle_spine_digest_prefix: bundle_convergence
            .sweep
            .canonical_bundle_spine_digest_prefix,
        bundle_convergence_sweep_digest_prefix: prefix_hex(
            &bundle_convergence.sweep.convergence_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_primary_semantics_authority_digest_prefix: primary_convergence
            .sweep
            .canonical_primary_semantics_authority_digest_prefix,
        primary_semantics_convergence_sweep_digest_prefix: prefix_hex(
            &primary_convergence.sweep.convergence_digest,
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
        ultimate_terminal_absolute_final_input_continuity_authority_digest_prefix: prefix_hex(
            &ultimate.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        continuity_status: if legacy_present {
            CanonicalConvergenceContinuityStatusV1::LegacyPresent
        } else if blocking.is_empty() {
            CanonicalConvergenceContinuityStatusV1::Pass
        } else {
            CanonicalConvergenceContinuityStatusV1::Fail
        },
        blocking_codes: blocking
            .into_iter()
            .map(ToString::to_string)
            .take(CODE_CAP)
            .collect(),
        remediation_codes: remediation
            .into_iter()
            .map(ToString::to_string)
            .take(CODE_CAP)
            .collect(),
        authority_digest: String::new(),
    };
    report.authority_digest = canonical_convergence_continuity_digest(&report)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn canonical_convergence_continuity_digest(
    report: &CanonicalConvergenceContinuityAuthorityV1,
) -> Result<String, OpsError> {
    let mut digestible = report.clone();
    digestible.authority_digest.clear();
    Ok(crate::sha256_hex(&serde_json::to_vec(&digestible)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_convergence_continuity_digest_stable() {
        let mut report = CanonicalConvergenceContinuityAuthorityV1 {
            schema_version: 1,
            applied_supported_set_digest_prefix: "aa".repeat(8),
            canonical_governance_entry_digest_prefix: "bb".repeat(8),
            governance_convergence_sweep_digest_prefix: "cc".repeat(8),
            canonical_readiness_spine_digest_prefix: "dd".repeat(8),
            readiness_convergence_sweep_digest_prefix: "ee".repeat(8),
            canonical_bundle_spine_digest_prefix: "ff".repeat(8),
            bundle_convergence_sweep_digest_prefix: "11".repeat(8),
            canonical_primary_semantics_authority_digest_prefix: "22".repeat(8),
            primary_semantics_convergence_sweep_digest_prefix: "33".repeat(8),
            operator_review_packet_digest_prefix: "44".repeat(8),
            operator_signoff_digest_prefix: "55".repeat(8),
            operator_workflow_chain_digest_prefix: "66".repeat(8),
            canonical_roundtrip_chain_digest_prefix: "77".repeat(8),
            ultimate_terminal_absolute_final_input_continuity_authority_digest_prefix: "88"
                .repeat(8),
            continuity_status: CanonicalConvergenceContinuityStatusV1::Pass,
            blocking_codes: Vec::new(),
            remediation_codes: Vec::new(),
            authority_digest: String::new(),
        };
        report.authority_digest = canonical_convergence_continuity_digest(&report).expect("digest");
        let stable = canonical_convergence_continuity_digest(&report).expect("stable");
        assert_eq!(stable, report.authority_digest);
    }
}
