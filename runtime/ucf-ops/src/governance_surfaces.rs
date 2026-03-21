use serde::{Deserialize, Serialize};

use crate::{
    load_applied_supported_set_context_v1, prefix_hex, sha256_hex,
    AggregatedActiveReviewSnapshotV1, AppliedSupportedSetContextV1, BackendEvidenceSnapshotV1,
    OpsError,
};

pub const GOVERNANCE_SURFACE_MISMATCH_CODE: &str = "GOVERNANCE_SURFACE_MISMATCH";
pub const GOVERNANCE_SURFACE_MISSING_CODE: &str = "GOVERNANCE_PRIMARY_SURFACE_MISSING";
pub const GOVERNANCE_APPLIED_SET_MISMATCH_CODE: &str = "GOVERNANCE_APPLIED_SET_MISMATCH";
pub const GOVERNANCE_PRIMARY_SURFACE_SCOPE_DRIFT_CODE: &str =
    "GOVERNANCE_PRIMARY_SURFACE_SCOPE_DRIFT";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernancePrimarySurfacesV1 {
    pub schema_version: u16,
    pub backend_evidence_snapshot_digest_prefix: String,
    pub active_review_snapshot_digest_prefix: String,
    pub supported_slot_set_digest_prefix: String,
    pub policy_graph_digest_prefix: String,
    pub manifest_digest_prefix: String,
    pub consistency_ok: bool,
    pub governance_surfaces_digest: String,
}

pub fn validate_governance_primary_surfaces(
    backend_snapshot: &BackendEvidenceSnapshotV1,
    active_review_snapshot: &AggregatedActiveReviewSnapshotV1,
) -> Result<GovernancePrimarySurfacesV1, OpsError> {
    if backend_snapshot.supported_slot_set_digest
        != active_review_snapshot.supported_slot_set_digest
    {
        return mismatch("SUPPORTED_SLOT_SET_DIGEST_MISMATCH");
    }

    if backend_snapshot.policy_graph_digest_prefix
        != active_review_snapshot.policy_graph_digest_prefix
    {
        return mismatch("POLICY_GRAPH_DIGEST_MISMATCH");
    }

    if backend_snapshot.manifest_digest_prefix != active_review_snapshot.manifest_digest_prefix {
        return mismatch("MANIFEST_DIGEST_MISMATCH");
    }

    let backend_slot_order = backend_snapshot
        .slots
        .iter()
        .map(|slot| slot.slot_id.as_str())
        .collect::<Vec<_>>();
    let active_slot_order = active_review_snapshot
        .slots
        .iter()
        .map(|slot| slot.slot_id.as_str())
        .collect::<Vec<_>>();
    if backend_slot_order != active_slot_order {
        return mismatch("SLOT_ORDER_MISMATCH");
    }

    for (backend_slot, active_slot) in backend_snapshot
        .slots
        .iter()
        .zip(&active_review_snapshot.slots)
    {
        if backend_slot.slot_id != active_slot.slot_id {
            return mismatch("SLOT_ID_MISMATCH");
        }
        if backend_slot.target_hash_prefix != active_slot.target_hash_prefix {
            return mismatch("TARGET_HASH_CONTRADICTION");
        }
    }

    let backend_digest_prefix = prefix_hex(&backend_snapshot.snapshot_digest, 16);
    let active_digest_prefix = prefix_hex(&active_review_snapshot.snapshot_digest, 16);
    let supported_slot_set_digest_prefix =
        prefix_hex(&backend_snapshot.supported_slot_set_digest, 16);
    let policy_graph_digest_prefix = backend_snapshot.policy_graph_digest_prefix.clone();
    let manifest_digest_prefix = backend_snapshot.manifest_digest_prefix.clone();

    let mut digest_source = Vec::new();
    digest_source.extend_from_slice(b"governance_primary_surfaces_v1");
    digest_source.extend_from_slice(backend_digest_prefix.as_bytes());
    digest_source.extend_from_slice(active_digest_prefix.as_bytes());
    digest_source.extend_from_slice(supported_slot_set_digest_prefix.as_bytes());
    digest_source.extend_from_slice(policy_graph_digest_prefix.as_bytes());
    digest_source.extend_from_slice(manifest_digest_prefix.as_bytes());

    Ok(GovernancePrimarySurfacesV1 {
        schema_version: 1,
        backend_evidence_snapshot_digest_prefix: backend_digest_prefix,
        active_review_snapshot_digest_prefix: active_digest_prefix,
        supported_slot_set_digest_prefix,
        policy_graph_digest_prefix,
        manifest_digest_prefix,
        consistency_ok: true,
        governance_surfaces_digest: sha256_hex(&digest_source),
    })
}

pub fn validate_governance_primary_surfaces_with_applied_scope(
    backend_snapshot: &BackendEvidenceSnapshotV1,
    active_review_snapshot: &AggregatedActiveReviewSnapshotV1,
    applied_scope: &AppliedSupportedSetContextV1,
) -> Result<GovernancePrimarySurfacesV1, OpsError> {
    if prefix_hex(&backend_snapshot.supported_slot_set_digest, 16)
        != applied_scope.applied_set_digest_prefix
        || prefix_hex(&active_review_snapshot.supported_slot_set_digest, 16)
            != applied_scope.applied_set_digest_prefix
    {
        return Err(OpsError::Invalid(format!(
            "{GOVERNANCE_APPLIED_SET_MISMATCH_CODE}:DIGEST_PREFIX"
        )));
    }

    let backend_slots = backend_snapshot
        .slots
        .iter()
        .map(|slot| slot.slot_id.clone())
        .collect::<Vec<_>>();
    let active_slots = active_review_snapshot
        .slots
        .iter()
        .map(|slot| slot.slot_id.clone())
        .collect::<Vec<_>>();
    let expected_slots = applied_scope.slots.clone();

    if backend_slots != expected_slots || active_slots != expected_slots {
        return Err(OpsError::Invalid(format!(
            "{GOVERNANCE_PRIMARY_SURFACE_SCOPE_DRIFT_CODE}:SLOT_MEMBERSHIP_OR_ORDER"
        )));
    }

    validate_governance_primary_surfaces(backend_snapshot, active_review_snapshot)
}

pub fn validate_governance_primary_surfaces_from_workdir(
    workdir: &std::path::Path,
    backend_snapshot: &BackendEvidenceSnapshotV1,
    active_review_snapshot: &AggregatedActiveReviewSnapshotV1,
) -> Result<GovernancePrimarySurfacesV1, OpsError> {
    let applied_scope = load_applied_supported_set_context_v1(workdir)?;
    validate_governance_primary_surfaces_with_applied_scope(
        backend_snapshot,
        active_review_snapshot,
        &applied_scope,
    )
}

pub fn validate_governance_primary_surfaces_optional(
    backend_snapshot: Option<&BackendEvidenceSnapshotV1>,
    active_review_snapshot: Option<&AggregatedActiveReviewSnapshotV1>,
) -> Result<GovernancePrimarySurfacesV1, OpsError> {
    let Some(backend_snapshot) = backend_snapshot else {
        return Err(OpsError::Invalid(format!(
            "{GOVERNANCE_SURFACE_MISSING_CODE}:BACKEND_EVIDENCE_SNAPSHOT_MISSING"
        )));
    };
    let Some(active_review_snapshot) = active_review_snapshot else {
        return Err(OpsError::Invalid(format!(
            "{GOVERNANCE_SURFACE_MISSING_CODE}:ACTIVE_REVIEW_SNAPSHOT_MISSING"
        )));
    };
    validate_governance_primary_surfaces(backend_snapshot, active_review_snapshot)
}

fn mismatch(detail: &str) -> Result<GovernancePrimarySurfacesV1, OpsError> {
    Err(OpsError::Invalid(format!(
        "{GOVERNANCE_SURFACE_MISMATCH_CODE}:{detail}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        models_lifecycle::{
            ActiveReviewContributingDigestsV1, ActiveReviewEvidenceV1, ActiveReviewOverallStatusV1,
            ActiveReviewSignoffAlignmentV1, BackendEvidenceSlotDenialsV1,
            BackendEvidenceSlotEvidenceV1, BackendEvidenceSlotReadinessV1,
            BackendEvidenceSlotSnapshotV1, BackendSupportMatrixV1, BurnResolutionStatusV1,
            BurnSupportResolutionV1, DriftStatusV1,
        },
        BackendSupportStateV1, OptionalBackendSupportStateV1,
    };

    fn backend_snapshot() -> BackendEvidenceSnapshotV1 {
        BackendEvidenceSnapshotV1 {
            schema_version: 1,
            supported_slot_set_digest: "11".repeat(32),
            policy_graph_digest_prefix: "22".repeat(8),
            manifest_digest_prefix: "33".repeat(8),
            slots: vec![slot_backend("sae", "bbbb"), slot_backend("world", "aaaa")],
            snapshot_digest: "44".repeat(32),
        }
    }

    fn active_snapshot() -> AggregatedActiveReviewSnapshotV1 {
        AggregatedActiveReviewSnapshotV1 {
            schema_version: 1,
            supported_slot_set_digest: "11".repeat(32),
            policy_graph_digest_prefix: "22".repeat(8),
            manifest_digest_prefix: "33".repeat(8),
            slots: vec![slot_active("sae", "bbbb"), slot_active("world", "aaaa")],
            overall_review_status: ActiveReviewOverallStatusV1::AllReviewable,
            signoff_alignment: ActiveReviewSignoffAlignmentV1 {
                aligned: true,
                status_code: "ALIGNED".to_string(),
            },
            canonical_governance_entry_digest_prefix: "MISSING".to_string(),
            final_governance_consumer_authority_digest_prefix: "MISSING".to_string(),
            governance_residual_sweep_digest_prefix: "MISSING".to_string(),
            snapshot_digest: "55".repeat(32),
        }
    }

    fn slot_backend(slot_id: &str, target_hash_prefix: &str) -> BackendEvidenceSlotSnapshotV1 {
        BackendEvidenceSlotSnapshotV1 {
            slot_id: slot_id.to_string(),
            target_hash_prefix: target_hash_prefix.to_string(),
            backend_support: BackendSupportMatrixV1 {
                stub: BackendSupportStateV1::Supported,
                candle: BackendSupportStateV1::Supported,
                burn: BackendSupportStateV1::Unsupported,
            },
            evidence: BackendEvidenceSlotEvidenceV1 {
                latest_probe_report_digest_prefix: "p".to_string(),
                latest_compare_window_digest_prefix: "c".to_string(),
                latest_shadow_ready_digest_prefix: "s".to_string(),
                latest_active_evidence_digest_prefix: "a".to_string(),
                latest_drift_status: DriftStatusV1::Ok,
                freshness_probe_age_ticks: Some(1),
                freshness_compare_age_ticks: Some(1),
                freshness_no_impact_age_ticks: Some(1),
                freshness_drift_status_age_ticks: Some(1),
                hash_consistency_ok: true,
            },
            readiness: BackendEvidenceSlotReadinessV1 {
                probe_ready: true,
                shadow_ready: true,
                active_eligible: true,
            },
            denials: BackendEvidenceSlotDenialsV1 {
                probe: None,
                shadow: None,
                active: None,
            },
            remediation_codes: Vec::new(),
            canonical_remediation_codes: Vec::new(),
            burn_resolution: BurnSupportResolutionV1 {
                slot_id: slot_id.to_string(),
                resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                support_state: OptionalBackendSupportStateV1::Unsupported,
                rationale_codes: vec!["X".to_string()],
                evidence_digest: "66".repeat(32),
            },
        }
    }

    fn slot_active(slot_id: &str, target_hash_prefix: &str) -> ActiveReviewEvidenceV1 {
        ActiveReviewEvidenceV1 {
            slot_id: slot_id.to_string(),
            target_hash_prefix: target_hash_prefix.to_string(),
            manifest_digest_prefix: "33".repeat(8),
            probe_ready: true,
            shadow_ready: true,
            active_eligible: true,
            strict_blocking: false,
            drift_blocking: false,
            alert_blocking: false,
            primary_denial_code: None,
            remediation_codes: Vec::new(),
            contributing_evidence_digests: ActiveReviewContributingDigestsV1 {
                probe_report_digest_prefix: "p".to_string(),
                shadow_ready_digest_prefix: "s".to_string(),
                active_evidence_digest_prefix: "a".to_string(),
                strict_evidence_digest_prefix: "t".to_string(),
            },
            burn_resolution: BurnSupportResolutionV1 {
                slot_id: slot_id.to_string(),
                resolution: BurnResolutionStatusV1::BurnClosedUnsupported,
                support_state: OptionalBackendSupportStateV1::Unsupported,
                rationale_codes: vec!["X".to_string()],
                evidence_digest: "66".repeat(32),
            },
            evidence_digest: "77".repeat(32),
        }
    }

    #[test]
    fn governance_surface_digest_is_stable() {
        let one = validate_governance_primary_surfaces(&backend_snapshot(), &active_snapshot())
            .expect("surfaces");
        let two = validate_governance_primary_surfaces(&backend_snapshot(), &active_snapshot())
            .expect("surfaces");
        assert_eq!(one, two);
    }

    #[test]
    fn mismatch_is_deterministic() {
        let mut active = active_snapshot();
        active.slots[0].target_hash_prefix = "different".to_string();
        let err = validate_governance_primary_surfaces(&backend_snapshot(), &active)
            .expect_err("must fail");
        assert!(err
            .to_string()
            .contains("GOVERNANCE_SURFACE_MISMATCH:TARGET_HASH_CONTRADICTION"));
    }
}
