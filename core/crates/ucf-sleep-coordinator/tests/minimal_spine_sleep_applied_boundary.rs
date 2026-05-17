use ucf_sleep_coordinator::{
    build_sleep_applied_boundary_from_audit, build_sleep_plan_candidate_from_replay_boundary,
    verify_minimal_spine_sleep_plan_candidate, MinimalSpineSleepPlanAuditStatus,
    MinimalSpineSleepPlanInput, MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE,
    MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE, MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE,
};
use ucf_types::Digest32;

const REPLAY_SOURCE: &str = "minimal_spine_replay_audit_fixture_for_sleep_applied_boundary";

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn zero_digest() -> Digest32 {
    Digest32::new([0u8; Digest32::LEN])
}

fn valid_input() -> MinimalSpineSleepPlanInput {
    MinimalSpineSleepPlanInput {
        replay_audit_digest: digest(21),
        replay_schedule_digest: digest(22),
        replay_applied_boundary_digest: Some(digest(23)),
        token_count: 5,
        source: REPLAY_SOURCE,
    }
}

fn pass_audit() -> ucf_sleep_coordinator::MinimalSpineSleepPlanAudit {
    let candidate =
        build_sleep_plan_candidate_from_replay_boundary(&valid_input()).expect("valid candidate");
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    audit
}

#[test]
fn sleep_applied_boundary_from_pass_audit_is_deterministic() {
    let audit = pass_audit();

    let first = build_sleep_applied_boundary_from_audit(&audit).expect("first boundary");
    let second = build_sleep_applied_boundary_from_audit(&audit).expect("second boundary");

    assert_eq!(first, second);
    assert_eq!(
        first.applied_boundary_digest,
        second.applied_boundary_digest
    );
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.applied_boundary_digest, first.digest());
    assert_eq!(first.source, MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE);
}

#[test]
fn sleep_applied_boundary_rejects_failed_audit() {
    let mut candidate =
        build_sleep_plan_candidate_from_replay_boundary(&valid_input()).expect("valid candidate");
    candidate.sleep_completed = true;
    let failed_audit = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert_eq!(failed_audit.status, MinimalSpineSleepPlanAuditStatus::Fail);
    assert!(build_sleep_applied_boundary_from_audit(&failed_audit).is_err());
}

#[test]
fn sleep_applied_boundary_preserves_audit_and_replay_provenance() {
    let audit = pass_audit();
    let boundary = build_sleep_applied_boundary_from_audit(&audit).expect("boundary");

    assert_eq!(boundary.sleep_plan_audit_digest, audit.audit_digest);
    assert_eq!(
        boundary.sleep_plan_candidate_digest,
        audit.sleep_plan_candidate_digest
    );
    assert_eq!(boundary.replay_audit_digest, audit.replay_audit_digest);
    assert_eq!(
        boundary.replay_schedule_digest,
        audit.replay_schedule_digest
    );
    assert_eq!(
        boundary.replay_applied_boundary_digest,
        audit.replay_applied_boundary_digest
    );
    assert_eq!(boundary.token_count, audit.token_count);
    assert_eq!(
        boundary.sleep_plan_audit_source,
        MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE
    );
    assert_eq!(
        boundary.candidate_source,
        MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE
    );
    assert_eq!(boundary.replay_source, REPLAY_SOURCE);
}

#[test]
fn sleep_applied_boundary_is_not_sleep_completion() {
    let boundary = build_sleep_applied_boundary_from_audit(&pass_audit()).expect("boundary");

    assert!(boundary.sleep_subsystem_applied);
    assert!(!boundary.sleep_completed);
    assert!(!boundary.memory_stabilized);
}

#[test]
fn sleep_applied_boundary_does_not_trigger_coordinator_runtime() {
    let boundary = build_sleep_applied_boundary_from_audit(&pass_audit()).expect("boundary");
    assert!(!boundary.coordinator_runtime_triggered);
    assert_eq!(boundary.source, MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE);
    assert!(!boundary.sleep_completed);
}

#[test]
fn sleep_applied_boundary_is_not_geist_or_ism() {
    let boundary = build_sleep_applied_boundary_from_audit(&pass_audit()).expect("boundary");

    assert!(!boundary.geist_ingested);
    assert!(!boundary.ism_written);
    assert!(!boundary.identity_anchor);
}

#[test]
fn sleep_applied_boundary_does_not_append_or_expose_gateway() {
    let boundary = build_sleep_applied_boundary_from_audit(&pass_audit()).expect("boundary");

    assert!(!boundary.evidence_archive_appended);
    assert!(!boundary.gateway_visible);
}

#[test]
fn sleep_applied_boundary_does_not_mutate_audit() {
    let audit = pass_audit();
    let before = audit.clone();

    let _boundary = build_sleep_applied_boundary_from_audit(&audit).expect("boundary");

    assert_eq!(audit, before);
}

#[test]
fn sleep_applied_boundary_rejects_invalid_audit_links() {
    let audit = pass_audit();

    let mut zero_audit_digest = audit.clone();
    zero_audit_digest.audit_digest = zero_digest();
    assert!(build_sleep_applied_boundary_from_audit(&zero_audit_digest).is_err());

    let mut zero_candidate_digest = audit.clone();
    zero_candidate_digest.sleep_plan_candidate_digest = zero_digest();
    zero_candidate_digest.recomputed_sleep_plan_candidate_digest = zero_digest();
    zero_candidate_digest.audit_digest = zero_candidate_digest.digest();
    assert!(build_sleep_applied_boundary_from_audit(&zero_candidate_digest).is_err());

    let mut zero_replay_audit_digest = audit.clone();
    zero_replay_audit_digest.replay_audit_digest = zero_digest();
    zero_replay_audit_digest.audit_digest = zero_replay_audit_digest.digest();
    assert!(build_sleep_applied_boundary_from_audit(&zero_replay_audit_digest).is_err());

    let mut zero_replay_schedule_digest = audit.clone();
    zero_replay_schedule_digest.replay_schedule_digest = zero_digest();
    zero_replay_schedule_digest.audit_digest = zero_replay_schedule_digest.digest();
    assert!(build_sleep_applied_boundary_from_audit(&zero_replay_schedule_digest).is_err());
}
