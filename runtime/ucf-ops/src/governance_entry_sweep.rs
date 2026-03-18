use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    derive_canonical_governance_entry, governance_surfaces_check, interop_consistency_matrix,
    load_applied_supported_set_context_v1, models_active_review_snapshot, models_evidence_snapshot,
    operator_review_packet, operator_signoff, operator_workflow_chain, prefix_hex,
    require_canonical_governance_entry, validate_governance_primary_surfaces_with_applied_scope,
    OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs, OpsError,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceEntryAuthorityStatusV2 {
    Pass,
    Fail,
    LegacyPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GovernanceEntrySweepMismatchCategoryV1 {
    SurfaceSkippedCanonicalGovernanceEntry,
    SurfaceUsedSecondaryGovernanceEntry,
    GovernanceEntryScopeMismatch,
    GovernanceEntryPolicyMismatch,
    LegacyGovernanceEntryPresent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalGovernanceEntryAuthorityV2 {
    pub schema_version: u16,
    pub applied_supported_set_digest_prefix: String,
    pub applied_context_digest_prefix: String,
    pub canonical_governance_entry_digest_prefix: String,
    pub covered_surface_count: u16,
    pub authority_status: GovernanceEntryAuthorityStatusV2,
    pub authority_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceEntrySweepSurfaceStatusV1 {
    pub surface: String,
    pub status: GovernanceEntryAuthorityStatusV2,
    pub mismatch_categories: Vec<GovernanceEntrySweepMismatchCategoryV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceEntrySweepReportV1 {
    pub schema_version: u16,
    pub authority: CanonicalGovernanceEntryAuthorityV2,
    pub surfaces: Vec<GovernanceEntrySweepSurfaceStatusV1>,
}

pub fn governance_entry_sweep(
    workdir: &Path,
    out: &Path,
) -> Result<GovernanceEntrySweepReportV1, OpsError> {
    let applied = load_applied_supported_set_context_v1(workdir)?;
    let backend = models_evidence_snapshot(workdir, None, None)?;
    let active = models_active_review_snapshot(
        workdir,
        &workdir.join("out/active_review_snapshot_governance_entry_sweep.json"),
    )?;
    let surfaces =
        validate_governance_primary_surfaces_with_applied_scope(&backend, &active, &applied)?;
    let derived_entry = derive_canonical_governance_entry(&applied, &surfaces)?;
    let canonical_entry = require_canonical_governance_entry(&applied, Some(&derived_entry))?;

    let signoff = operator_signoff(
        workdir,
        &OperatorSignoffArgs {
            run_id: None,
            latest: true,
            profile: std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string()),
        },
        &workdir.join("out/operator_signoff_governance_entry_sweep.json"),
    )?;
    let packet = operator_review_packet(
        workdir,
        &OperatorReviewPacketArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_review_packet_governance_entry_sweep.json"),
    )?;
    let workflow = operator_workflow_chain(
        workdir,
        &OperatorWorkflowArgs {
            run_id: None,
            latest: true,
        },
        &workdir.join("out/operator_workflow_chain_governance_entry_sweep.json"),
    )?;
    let interop = interop_consistency_matrix(
        workdir,
        &workdir.join("out/interop_consistency_matrix_governance_entry_sweep.json"),
    )?;
    let governance = governance_surfaces_check(
        workdir,
        &workdir.join("out/governance_surfaces_check_governance_entry_sweep.json"),
    )?;

    let expected_scope = canonical_entry.applied_supported_set_digest_prefix.clone();
    let expected_ctx = canonical_entry.applied_context_digest_prefix.clone();
    let mut surfaces = vec![
        check_surface(
            "ActiveReviewSnapshot",
            prefix_hex(&active.supported_slot_set_digest, 16) == expected_scope,
            false,
            false,
        ),
        check_surface(
            "OperatorSignoff",
            signoff.applied_supported_set_digest_prefix == expected_scope
                && signoff.applied_context_digest_prefix == expected_ctx,
            false,
            false,
        ),
        check_surface(
            "OperatorReviewPacket",
            packet.applied_supported_set_digest_prefix == expected_scope
                && packet.applied_context_digest_prefix == expected_ctx,
            false,
            false,
        ),
        check_surface(
            "OperatorWorkflowChain",
            workflow.applied_supported_set_digest_prefix == expected_scope
                && workflow.applied_context_digest_prefix == expected_ctx,
            false,
            false,
        ),
        check_surface(
            "InteropConsistencyMatrix",
            interop.matrix.applied_supported_set_digest_prefix == expected_scope,
            false,
            false,
        ),
        check_surface(
            "GovernanceSurfacesCheck",
            governance.governance_primary_surfaces.is_some(),
            false,
            false,
        ),
    ];

    for surface in &mut surfaces {
        if surface.surface == "InteropConsistencyMatrix"
            && interop.matrix.policy_graph_digest_prefix != backend.policy_graph_digest_prefix
        {
            surface
                .mismatch_categories
                .push(GovernanceEntrySweepMismatchCategoryV1::GovernanceEntryPolicyMismatch);
            surface.status = GovernanceEntryAuthorityStatusV2::Fail;
        }
    }

    let authority_status = if surfaces
        .iter()
        .all(|s| matches!(s.status, GovernanceEntryAuthorityStatusV2::Pass))
    {
        GovernanceEntryAuthorityStatusV2::Pass
    } else {
        GovernanceEntryAuthorityStatusV2::Fail
    };

    let authority = derive_authority_v2(
        &canonical_entry.applied_supported_set_digest_prefix,
        &canonical_entry.applied_context_digest_prefix,
        &canonical_entry.authority_digest,
        surfaces.len() as u16,
        authority_status,
    )?;

    let report = GovernanceEntrySweepReportV1 {
        schema_version: 1,
        authority,
        surfaces,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn check_surface(
    surface: &str,
    pass: bool,
    legacy_present: bool,
    secondary_path: bool,
) -> GovernanceEntrySweepSurfaceStatusV1 {
    let mut mismatch_categories = BTreeSet::new();
    if !pass {
        mismatch_categories
            .insert(GovernanceEntrySweepMismatchCategoryV1::SurfaceSkippedCanonicalGovernanceEntry);
        mismatch_categories
            .insert(GovernanceEntrySweepMismatchCategoryV1::GovernanceEntryScopeMismatch);
    }
    if secondary_path {
        mismatch_categories
            .insert(GovernanceEntrySweepMismatchCategoryV1::SurfaceUsedSecondaryGovernanceEntry);
    }
    if legacy_present {
        mismatch_categories
            .insert(GovernanceEntrySweepMismatchCategoryV1::LegacyGovernanceEntryPresent);
    }
    let status = if legacy_present {
        GovernanceEntryAuthorityStatusV2::LegacyPresent
    } else if mismatch_categories.is_empty() {
        GovernanceEntryAuthorityStatusV2::Pass
    } else {
        GovernanceEntryAuthorityStatusV2::Fail
    };

    GovernanceEntrySweepSurfaceStatusV1 {
        surface: surface.to_string(),
        status,
        mismatch_categories: mismatch_categories.into_iter().collect(),
    }
}

fn derive_authority_v2(
    applied_supported_set_digest_prefix: &str,
    applied_context_digest_prefix: &str,
    canonical_governance_entry_digest: &str,
    covered_surface_count: u16,
    authority_status: GovernanceEntryAuthorityStatusV2,
) -> Result<CanonicalGovernanceEntryAuthorityV2, OpsError> {
    let canonical_governance_entry_digest_prefix =
        prefix_hex(canonical_governance_entry_digest, 16);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"canonical_governance_entry_authority_v2");
    bytes.extend_from_slice(applied_supported_set_digest_prefix.as_bytes());
    bytes.extend_from_slice(applied_context_digest_prefix.as_bytes());
    bytes.extend_from_slice(canonical_governance_entry_digest_prefix.as_bytes());
    bytes.extend_from_slice(covered_surface_count.to_string().as_bytes());
    bytes.extend_from_slice(format!("{:?}", authority_status).as_bytes());

    Ok(CanonicalGovernanceEntryAuthorityV2 {
        schema_version: 2,
        applied_supported_set_digest_prefix: applied_supported_set_digest_prefix.to_string(),
        applied_context_digest_prefix: applied_context_digest_prefix.to_string(),
        canonical_governance_entry_digest_prefix,
        covered_surface_count,
        authority_status,
        authority_digest: crate::sha256_hex(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_v2_digest_is_stable() {
        let a = derive_authority_v2(
            "scope123456789012",
            "ctx1234567890123",
            &"a".repeat(64),
            6,
            GovernanceEntryAuthorityStatusV2::Pass,
        )
        .expect("authority");
        let b = derive_authority_v2(
            "scope123456789012",
            "ctx1234567890123",
            &"a".repeat(64),
            6,
            GovernanceEntryAuthorityStatusV2::Pass,
        )
        .expect("authority");
        assert_eq!(a.authority_digest, b.authority_digest);
        assert_eq!(a.schema_version, 2);
    }

    #[test]
    fn check_surface_marks_legacy() {
        let status = check_surface("OperatorSignoff", true, true, false);
        assert!(matches!(
            status.status,
            GovernanceEntryAuthorityStatusV2::LegacyPresent
        ));
        assert!(status
            .mismatch_categories
            .contains(&GovernanceEntrySweepMismatchCategoryV1::LegacyGovernanceEntryPresent));
    }
}
