use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, exports_bundle_spine_check, governance_entry_sweep,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_export_chain_check, operator_review_packet, operator_roundtrip_chain_check,
    operator_signoff, operator_workflow_chain, prefix_hex, readiness_spine_check,
    readiness_spine_sweep, validate_governance_primary_surfaces_with_applied_scope,
    GovernanceEntryAuthorityStatusV2, OperatorReviewPacketArgs, OperatorSignoffArgs,
    OperatorWorkflowArgs, OpsError, ReadinessSpineCheckStatusV1,
};

const DIGEST_PREFIX_LEN: usize = 16;
const CODE_CAP: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ContinuityAuthorityStatusV1 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalContinuityAuthorityV1 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub canonical_governance_authority_digest_prefix: String,
    pub canonical_readiness_spine_digest_prefix: String,
    pub canonical_readiness_authority_digest_prefix: String,
    pub operator_review_packet_digest_prefix: String,
    pub operator_signoff_digest_prefix: String,
    pub operator_workflow_chain_digest_prefix: String,
    pub canonical_bundle_spine_digest_prefix: String,
    pub canonical_bundle_authority_digest_prefix: String,
    pub canonical_roundtrip_chain_digest_prefix: String,
    pub continuity_status: ContinuityAuthorityStatusV1,
    pub blocking_codes: Vec<String>,
    pub remediation_codes: Vec<String>,
    pub authority_digest: String,
}

pub fn continuity_authority_check(
    workdir: &Path,
    bundle: &Path,
    out: &Path,
) -> Result<CanonicalContinuityAuthorityV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_continuity_authority_check.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let governance = derive_canonical_governance_entry(&applied, &surfaces)?;
    let governance_entry_prefix = prefix_hex(&governance.authority_digest, DIGEST_PREFIX_LEN);

    let governance_sweep = governance_entry_sweep(
        workdir,
        &workdir.join("out/governance_entry_sweep_continuity_authority_check.json"),
    )?;

    let readiness = readiness_spine_check(
        workdir,
        &workdir.join("out/readiness_spine_check_continuity_authority_check.json"),
    )?;
    let readiness_sweep = readiness_spine_sweep(
        workdir,
        &workdir.join("out/readiness_spine_sweep_continuity_authority_check.json"),
    )?;

    let review_packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_continuity_authority_check.json"),
    )?;
    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_continuity_authority_check.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_continuity_authority_check.json"),
    )?;
    let export_chain = operator_export_chain_check(
        workdir,
        &workdir.join("out/operator_export_chain_continuity_authority_check.json"),
    )?;

    let bundle_spine = exports_bundle_spine_check(
        bundle,
        &workdir.join("out/bundle_spine_continuity_authority_check.json"),
    )?;
    let roundtrip = operator_roundtrip_chain_check(
        workdir,
        bundle,
        &workdir.join("out/operator_roundtrip_chain_continuity_authority_check.json"),
    )?;

    let mut blocking = BTreeSet::new();
    let mut remediation = BTreeSet::new();

    let applied_set = applied.applied_set_digest_prefix.as_str();
    if review_packet.applied_supported_set_digest_prefix != applied_set
        || signoff.applied_supported_set_digest_prefix != applied_set
        || workflow.applied_supported_set_digest_prefix != applied_set
    {
        blocking.insert("CONTINUITY_SCOPE_MISMATCH".to_string());
        remediation.insert("run_models_applied_scope_check".to_string());
    }

    if governance_sweep
        .authority
        .applied_supported_set_digest_prefix
        != applied_set
        || governance_sweep
            .authority
            .canonical_governance_entry_digest_prefix
            != governance_entry_prefix
    {
        blocking.insert("CONTINUITY_GOVERNANCE_MISMATCH".to_string());
        remediation.insert("run_governance_entry_sweep".to_string());
    }

    if !matches!(
        governance_sweep.authority.authority_status,
        GovernanceEntryAuthorityStatusV2::Pass
    ) {
        blocking.insert("CONTINUITY_GOVERNANCE_MISMATCH".to_string());
        remediation.insert("run_governance_entry_sweep".to_string());
    }

    let readiness_spine_prefix = prefix_hex(
        &readiness.canonical_readiness_spine.spine_digest,
        DIGEST_PREFIX_LEN,
    );
    if readiness
        .canonical_readiness_spine
        .applied_supported_set_digest_prefix
        != applied_set
        || review_packet.canonical_readiness_spine_digest_prefix != readiness_spine_prefix
        || signoff.canonical_readiness_spine_digest_prefix != readiness_spine_prefix
        || workflow.canonical_readiness_spine_digest_prefix != readiness_spine_prefix
        || !matches!(readiness.status, ReadinessSpineCheckStatusV1::Pass)
    {
        blocking.insert("CONTINUITY_READINESS_MISMATCH".to_string());
        remediation.insert("run_readiness_spine_sweep".to_string());
    }

    if !matches!(
        readiness_sweep.authority.authority_status,
        crate::CanonicalReadinessAuthorityStatusV2::Pass
    ) {
        blocking.insert("CONTINUITY_READINESS_MISMATCH".to_string());
        remediation.insert("run_readiness_spine_sweep".to_string());
    }

    if !workflow.blocking_codes.is_empty() {
        blocking.insert("CONTINUITY_WORKFLOW_MISMATCH".to_string());
        remediation.insert("run_operator_workflow".to_string());
    }

    if !workflow.export_targets.repro_ready
        || !workflow.export_targets.bugkit_ready
        || !matches!(
            export_chain.authority_chain_status,
            crate::OperatorExportAuthorityChainStatusV1::Pass
        )
    {
        blocking.insert("CONTINUITY_EXPORT_READY_MISMATCH".to_string());
        remediation.insert("run_operator_export_chain_check".to_string());
    }

    if !matches!(
        roundtrip.roundtrip_status,
        crate::CanonicalRoundTripChainStatusV1::Pass
    ) || !bundle_spine.pass
    {
        blocking.insert("CONTINUITY_BUNDLE_MISMATCH".to_string());
        remediation.insert("run_operator_roundtrip_chain_check".to_string());
    }

    let legacy_present = matches!(
        governance_sweep.authority.authority_status,
        GovernanceEntryAuthorityStatusV2::LegacyPresent
    ) || matches!(
        readiness_sweep.authority.authority_status,
        crate::CanonicalReadinessAuthorityStatusV2::LegacyPresent
    ) || bundle_spine
        .mismatch_codes
        .iter()
        .any(|code| code.contains("LEGACY"));
    if legacy_present {
        blocking.insert("LEGACY_CONTINUITY_PATH_PRESENT".to_string());
        remediation.insert("remove_legacy_continuity_paths".to_string());
    }

    let mut report = CanonicalContinuityAuthorityV1 {
        schema_version: 1,
        applied_supported_set_digest_prefix: applied.applied_set_digest_prefix,
        canonical_governance_entry_digest_prefix: governance_entry_prefix,
        canonical_governance_authority_digest_prefix: prefix_hex(
            &governance_sweep.authority.authority_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_readiness_spine_digest_prefix: readiness_spine_prefix,
        canonical_readiness_authority_digest_prefix: prefix_hex(
            &readiness_sweep.authority.authority_digest,
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
        canonical_bundle_spine_digest_prefix: prefix_hex(
            &bundle_spine.spine.bundle_spine_digest,
            DIGEST_PREFIX_LEN,
        ),
        canonical_bundle_authority_digest_prefix: bundle_spine
            .authority_digest_prefix
            .unwrap_or_else(|| "MISSING".to_string()),
        canonical_roundtrip_chain_digest_prefix: prefix_hex(
            &roundtrip.chain_digest,
            DIGEST_PREFIX_LEN,
        ),
        continuity_status: if legacy_present {
            ContinuityAuthorityStatusV1::LegacyPresent
        } else if blocking.is_empty() {
            ContinuityAuthorityStatusV1::Pass
        } else {
            ContinuityAuthorityStatusV1::Fail
        },
        blocking_codes: blocking.into_iter().take(CODE_CAP).collect(),
        remediation_codes: remediation.into_iter().take(CODE_CAP).collect(),
        authority_digest: String::new(),
    };

    report.authority_digest = continuity_digest(&report)?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;

    Ok(report)
}

fn continuity_digest(report: &CanonicalContinuityAuthorityV1) -> Result<String, OpsError> {
    let mut canonical = report.clone();
    canonical.authority_digest.clear();
    canonical.blocking_codes.sort();
    canonical.remediation_codes.sort();
    Ok(crate::sha256_hex(&serde_json::to_vec(&canonical)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuity_digest_is_stable() {
        let mut report = CanonicalContinuityAuthorityV1 {
            schema_version: 1,
            applied_supported_set_digest_prefix: "11".repeat(8),
            canonical_governance_entry_digest_prefix: "22".repeat(8),
            canonical_governance_authority_digest_prefix: "33".repeat(8),
            canonical_readiness_spine_digest_prefix: "44".repeat(8),
            canonical_readiness_authority_digest_prefix: "55".repeat(8),
            operator_review_packet_digest_prefix: "66".repeat(8),
            operator_signoff_digest_prefix: "77".repeat(8),
            operator_workflow_chain_digest_prefix: "88".repeat(8),
            canonical_bundle_spine_digest_prefix: "99".repeat(8),
            canonical_bundle_authority_digest_prefix: "aa".repeat(8),
            canonical_roundtrip_chain_digest_prefix: "bb".repeat(8),
            continuity_status: ContinuityAuthorityStatusV1::Pass,
            blocking_codes: vec![],
            remediation_codes: vec!["x".to_string()],
            authority_digest: String::new(),
        };
        report.authority_digest = continuity_digest(&report).expect("digest");
        let a = report.authority_digest.clone();
        let b = continuity_digest(&report).expect("digest b");
        assert_eq!(a, b);
    }
}
