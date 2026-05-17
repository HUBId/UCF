use ucf_geist::{
    build_geist_projection_candidate_from_sleep_audit,
    verify_minimal_spine_geist_projection_candidate, MinimalSpineGeistProjectionAuditFailure,
    MinimalSpineGeistProjectionAuditStatus, MINIMAL_SPINE_GEIST_PROJECTION_AUDIT_SOURCE,
};
use ucf_sleep_coordinator::{
    build_sleep_applied_boundary_from_audit, build_sleep_plan_candidate_from_replay_boundary,
    verify_minimal_spine_sleep_plan_candidate, MinimalSpineSleepAppliedBoundary,
    MinimalSpineSleepPlanAudit, MinimalSpineSleepPlanAuditStatus, MinimalSpineSleepPlanInput,
};
use ucf_types::Digest32;

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn sleep_input() -> MinimalSpineSleepPlanInput {
    MinimalSpineSleepPlanInput {
        replay_audit_digest: digest(51),
        replay_schedule_digest: digest(52),
        replay_applied_boundary_digest: Some(digest(53)),
        token_count: 11,
        source: "replay-audit-fixture",
    }
}

fn pass_sleep_audit() -> MinimalSpineSleepPlanAudit {
    let candidate = build_sleep_plan_candidate_from_replay_boundary(&sleep_input())
        .expect("valid sleep candidate");
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    audit
}

fn pass_sleep_boundary(audit: &MinimalSpineSleepPlanAudit) -> MinimalSpineSleepAppliedBoundary {
    build_sleep_applied_boundary_from_audit(audit).expect("valid sleep boundary")
}

#[test]
fn geist_projection_audit_passes_for_valid_candidate() {
    let audit = pass_sleep_audit();
    let boundary = pass_sleep_boundary(&audit);
    let candidate = build_geist_projection_candidate_from_sleep_audit(&audit, Some(&boundary))
        .expect("valid projection candidate");

    let projection_audit = verify_minimal_spine_geist_projection_candidate(&candidate);

    assert_eq!(
        projection_audit.status,
        MinimalSpineGeistProjectionAuditStatus::Pass
    );
    assert!(projection_audit.failure_reasons.is_empty());
    assert_eq!(
        projection_audit.projection_digest,
        candidate.projection_digest
    );
    assert_eq!(
        projection_audit.recomputed_projection_digest,
        candidate.digest()
    );
    assert_eq!(projection_audit.audit_digest, projection_audit.digest());
    assert_eq!(
        projection_audit.source,
        MINIMAL_SPINE_GEIST_PROJECTION_AUDIT_SOURCE
    );
}

#[test]
fn geist_projection_audit_is_deterministic() {
    let audit = pass_sleep_audit();
    let boundary = pass_sleep_boundary(&audit);
    let candidate = build_geist_projection_candidate_from_sleep_audit(&audit, Some(&boundary))
        .expect("valid projection candidate");

    let first = verify_minimal_spine_geist_projection_candidate(&candidate);
    let second = verify_minimal_spine_geist_projection_candidate(&candidate);

    assert_eq!(first, second);
    assert_eq!(first.audit_digest, second.audit_digest);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
}

#[test]
fn geist_projection_audit_detects_tampered_candidate_metadata() {
    let audit = pass_sleep_audit();
    let mut candidate = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");
    candidate.token_count = 0;
    candidate.candidate_only = false;
    candidate.geist_applied = true;

    let projection_audit = verify_minimal_spine_geist_projection_candidate(&candidate);

    assert_eq!(
        projection_audit.status,
        MinimalSpineGeistProjectionAuditStatus::Fail
    );
    assert_eq!(projection_audit.audit_digest, projection_audit.digest());
    assert!(projection_audit
        .failure_reasons
        .contains(&MinimalSpineGeistProjectionAuditFailure::ProjectionDigestMismatch));
    assert!(projection_audit
        .failure_reasons
        .contains(&MinimalSpineGeistProjectionAuditFailure::InvalidTokenCount));
    assert!(projection_audit
        .failure_reasons
        .contains(&MinimalSpineGeistProjectionAuditFailure::NotCandidateOnly));
    assert!(projection_audit
        .failure_reasons
        .contains(&MinimalSpineGeistProjectionAuditFailure::GeistAppliedFlagSet));
}

#[test]
fn geist_projection_audit_preserves_sleep_and_replay_provenance() {
    let audit = pass_sleep_audit();
    let boundary = pass_sleep_boundary(&audit);
    let candidate = build_geist_projection_candidate_from_sleep_audit(&audit, Some(&boundary))
        .expect("valid projection candidate");

    let projection_audit = verify_minimal_spine_geist_projection_candidate(&candidate);

    assert_eq!(projection_audit.sleep_plan_audit_digest, audit.audit_digest);
    assert_eq!(
        projection_audit.sleep_plan_candidate_digest,
        audit.sleep_plan_candidate_digest
    );
    assert_eq!(
        projection_audit.sleep_applied_boundary_digest,
        Some(boundary.applied_boundary_digest)
    );
    assert_eq!(
        projection_audit.replay_audit_digest,
        audit.replay_audit_digest
    );
    assert_eq!(
        projection_audit.replay_schedule_digest,
        audit.replay_schedule_digest
    );
    assert_eq!(projection_audit.token_count, audit.token_count);
    assert_eq!(projection_audit.candidate_source, candidate.source);
    assert_eq!(projection_audit.sleep_source, audit.source);
}

#[test]
fn geist_projection_audit_is_verify_only_not_applied() {
    let audit = pass_sleep_audit();
    let candidate = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");
    let before = candidate.clone();

    let projection_audit = verify_minimal_spine_geist_projection_candidate(&candidate);

    assert_eq!(candidate, before);
    assert_eq!(
        projection_audit.status,
        MinimalSpineGeistProjectionAuditStatus::Pass
    );
    assert!(projection_audit.candidate_only);
    assert!(!projection_audit.geist_applied);
}

#[test]
fn geist_projection_audit_has_no_ism_identity_side_effects() {
    let audit = pass_sleep_audit();
    let candidate = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");

    let projection_audit = verify_minimal_spine_geist_projection_candidate(&candidate);

    assert!(!projection_audit.ism_written);
    assert!(!projection_audit.identity_anchor);
    assert!(!projection_audit.identity_finalized);
}

#[test]
fn geist_projection_audit_does_not_mutate_policy_or_append() {
    let audit = pass_sleep_audit();
    let candidate = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");

    let projection_audit = verify_minimal_spine_geist_projection_candidate(&candidate);

    assert!(!projection_audit.policy_mutated);
    assert!(!projection_audit.evidence_archive_appended);
}

#[test]
fn geist_projection_audit_does_not_expose_gateway_or_runtime() {
    let audit = pass_sleep_audit();
    let candidate = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");

    let projection_audit = verify_minimal_spine_geist_projection_candidate(&candidate);

    assert!(!projection_audit.gateway_visible);
    assert!(!projection_audit.geist_applied);
    assert_eq!(
        projection_audit.status,
        MinimalSpineGeistProjectionAuditStatus::Pass
    );
}
