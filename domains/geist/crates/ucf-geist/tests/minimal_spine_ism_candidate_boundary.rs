use ucf_geist::{
    build_geist_projection_candidate_from_sleep_audit,
    build_ism_candidate_boundary_from_geist_audit, verify_minimal_spine_geist_projection_candidate,
    IsmCandidateBoundaryError, MinimalSpineGeistProjectionAudit,
    MinimalSpineGeistProjectionAuditStatus, MinimalSpineIsmCandidateBoundary,
    MINIMAL_SPINE_ISM_CANDIDATE_BOUNDARY_SOURCE,
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
        replay_audit_digest: digest(61),
        replay_schedule_digest: digest(62),
        replay_applied_boundary_digest: Some(digest(63)),
        token_count: 13,
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

fn pass_geist_audit() -> MinimalSpineGeistProjectionAudit {
    let audit = pass_sleep_audit();
    let boundary = pass_sleep_boundary(&audit);
    let candidate = build_geist_projection_candidate_from_sleep_audit(&audit, Some(&boundary))
        .expect("valid projection candidate");
    let projection_audit = verify_minimal_spine_geist_projection_candidate(&candidate);
    assert_eq!(
        projection_audit.status,
        MinimalSpineGeistProjectionAuditStatus::Pass
    );
    projection_audit
}

fn pass_boundary() -> MinimalSpineIsmCandidateBoundary {
    build_ism_candidate_boundary_from_geist_audit(&pass_geist_audit())
        .expect("valid ism candidate boundary")
}

#[test]
fn ism_candidate_boundary_from_pass_audit_is_deterministic() {
    let audit = pass_geist_audit();

    let first = build_ism_candidate_boundary_from_geist_audit(&audit)
        .expect("first boundary from pass audit");
    let second = build_ism_candidate_boundary_from_geist_audit(&audit)
        .expect("second boundary from pass audit");

    assert_eq!(first, second);
    assert_eq!(first.ism_candidate_digest, second.ism_candidate_digest);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.ism_candidate_digest, first.digest());
    assert_eq!(first.source, MINIMAL_SPINE_ISM_CANDIDATE_BOUNDARY_SOURCE);
}

#[test]
fn ism_candidate_boundary_rejects_failed_audit() {
    let sleep_audit = pass_sleep_audit();
    let mut candidate = build_geist_projection_candidate_from_sleep_audit(&sleep_audit, None)
        .expect("valid projection candidate");
    candidate.token_count = 0;
    let failed_audit = verify_minimal_spine_geist_projection_candidate(&candidate);
    assert_eq!(
        failed_audit.status,
        MinimalSpineGeistProjectionAuditStatus::Fail
    );

    let error = build_ism_candidate_boundary_from_geist_audit(&failed_audit)
        .expect_err("failed audit cannot create boundary");

    assert_eq!(error, IsmCandidateBoundaryError::AuditStatusNotPass);
}

#[test]
fn ism_candidate_boundary_preserves_geist_sleep_replay_provenance() {
    let audit = pass_geist_audit();

    let boundary =
        build_ism_candidate_boundary_from_geist_audit(&audit).expect("boundary from pass audit");

    assert_eq!(boundary.geist_projection_audit_digest, audit.audit_digest);
    assert_eq!(boundary.geist_projection_digest, audit.projection_digest);
    assert_eq!(
        boundary.sleep_plan_audit_digest,
        audit.sleep_plan_audit_digest
    );
    assert_eq!(
        boundary.sleep_plan_candidate_digest,
        audit.sleep_plan_candidate_digest
    );
    assert_eq!(
        boundary.sleep_applied_boundary_digest,
        audit.sleep_applied_boundary_digest
    );
    assert_eq!(boundary.replay_audit_digest, audit.replay_audit_digest);
    assert_eq!(
        boundary.replay_schedule_digest,
        audit.replay_schedule_digest
    );
    assert_eq!(boundary.token_count, audit.token_count);
    assert_eq!(boundary.audit_source, audit.source);
    assert_eq!(boundary.candidate_source, audit.candidate_source);
    assert_eq!(boundary.sleep_source, audit.sleep_source);
}

#[test]
fn ism_candidate_boundary_is_not_persistent_ism_write() {
    let boundary = pass_boundary();

    assert!(boundary.ism_candidate_only);
    assert!(!boundary.ism_written);
    assert!(!boundary.ism_upserted);
}

#[test]
fn ism_candidate_boundary_is_not_identity_anchor_or_finalization() {
    let boundary = pass_boundary();

    assert!(!boundary.identity_anchor);
    assert!(!boundary.identity_finalized);
}

#[test]
fn ism_candidate_boundary_is_not_memory_stabilization() {
    let boundary = pass_boundary();

    assert!(!boundary.memory_stabilized);
}

#[test]
fn ism_candidate_boundary_does_not_mutate_policy_or_append() {
    let boundary = pass_boundary();

    assert!(!boundary.policy_mutated);
    assert!(!boundary.evidence_archive_appended);
}

#[test]
fn ism_candidate_boundary_does_not_expose_gateway() {
    let boundary = pass_boundary();

    assert!(!boundary.gateway_visible);
}

#[test]
fn ism_candidate_boundary_does_not_mutate_audit() {
    let audit = pass_geist_audit();
    let before = audit.clone();

    let boundary =
        build_ism_candidate_boundary_from_geist_audit(&audit).expect("boundary from pass audit");

    assert_eq!(audit, before);
    assert_eq!(boundary.geist_projection_audit_digest, audit.audit_digest);
}

#[test]
fn ism_candidate_boundary_rejects_invalid_audit_links() {
    let mut zero_projection_digest = pass_geist_audit();
    zero_projection_digest.projection_digest = Digest32::new([0u8; Digest32::LEN]);
    zero_projection_digest.recomputed_projection_digest = Digest32::new([0u8; Digest32::LEN]);
    zero_projection_digest.audit_digest = zero_projection_digest.digest();
    assert_eq!(
        build_ism_candidate_boundary_from_geist_audit(&zero_projection_digest)
            .expect_err("zero projection digest rejected"),
        IsmCandidateBoundaryError::ZeroGeistProjectionDigest
    );

    let mut zero_token_count = pass_geist_audit();
    zero_token_count.token_count = 0;
    zero_token_count.audit_digest = zero_token_count.digest();
    assert_eq!(
        build_ism_candidate_boundary_from_geist_audit(&zero_token_count)
            .expect_err("zero token count rejected"),
        IsmCandidateBoundaryError::ZeroTokenCount
    );

    let mut empty_source = pass_geist_audit();
    empty_source.source = "";
    empty_source.audit_digest = empty_source.digest();
    assert_eq!(
        build_ism_candidate_boundary_from_geist_audit(&empty_source)
            .expect_err("empty audit source rejected"),
        IsmCandidateBoundaryError::EmptySource
    );
}
