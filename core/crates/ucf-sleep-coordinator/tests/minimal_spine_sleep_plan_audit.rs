use ucf_sleep_coordinator::{
    build_sleep_plan_candidate_from_replay_boundary, verify_minimal_spine_sleep_plan_candidate,
    MinimalSpineSleepPlanAuditFailure, MinimalSpineSleepPlanAuditStatus,
    MinimalSpineSleepPlanInput, MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE,
    MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE,
};
use ucf_types::Digest32;

const REPLAY_SOURCE: &str = "minimal_spine_replay_audit_fixture_for_sleep_plan_audit";

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn valid_input() -> MinimalSpineSleepPlanInput {
    MinimalSpineSleepPlanInput {
        replay_audit_digest: digest(11),
        replay_schedule_digest: digest(12),
        replay_applied_boundary_digest: Some(digest(13)),
        token_count: 3,
        source: REPLAY_SOURCE,
    }
}

fn valid_candidate() -> ucf_sleep_coordinator::MinimalSpineSleepPlanCandidate {
    build_sleep_plan_candidate_from_replay_boundary(&valid_input()).expect("valid candidate")
}

#[test]
fn sleep_plan_audit_passes_for_valid_candidate() {
    let candidate = valid_candidate();
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    assert!(audit.failure_reasons.is_empty());
    assert_eq!(audit.audit_digest, audit.digest());
    assert_eq!(audit.source, MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE);
    assert_eq!(
        audit.candidate_source,
        MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE
    );
}

#[test]
fn sleep_plan_audit_is_deterministic() {
    let candidate = valid_candidate();

    let first = verify_minimal_spine_sleep_plan_candidate(&candidate);
    let second = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert_eq!(first, second);
    assert_eq!(first.audit_digest, second.audit_digest);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
}

#[test]
fn sleep_plan_audit_detects_tampered_candidate_metadata() {
    let mut candidate = valid_candidate();
    candidate.token_count = 0;
    candidate.sleep_applied = true;

    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Fail);
    assert_eq!(audit.audit_digest, audit.digest());
    assert_eq!(
        audit.failure_reasons,
        vec![
            MinimalSpineSleepPlanAuditFailure::CandidateDigestMismatch,
            MinimalSpineSleepPlanAuditFailure::InvalidTokenCount,
            MinimalSpineSleepPlanAuditFailure::SleepAppliedFlagSet,
        ]
    );
}

#[test]
fn sleep_plan_audit_preserves_replay_provenance() {
    let input = valid_input();
    let candidate =
        build_sleep_plan_candidate_from_replay_boundary(&input).expect("candidate from input");
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert_eq!(audit.replay_audit_digest, input.replay_audit_digest);
    assert_eq!(audit.replay_schedule_digest, input.replay_schedule_digest);
    assert_eq!(
        audit.replay_applied_boundary_digest,
        input.replay_applied_boundary_digest
    );
    assert_eq!(audit.token_count, input.token_count);
    assert_eq!(audit.replay_source, input.source);
    assert_eq!(
        audit.sleep_plan_candidate_digest,
        candidate.sleep_plan_digest
    );
    assert_eq!(
        audit.recomputed_sleep_plan_candidate_digest,
        candidate.digest()
    );
}

#[test]
fn sleep_plan_audit_is_verify_only_not_applied() {
    let candidate = valid_candidate();
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert!(audit.candidate_only);
    assert!(!audit.sleep_applied);
    assert!(!audit.sleep_completed);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
}

#[test]
fn sleep_plan_audit_has_no_geist_ism_identity_side_effects() {
    let candidate = valid_candidate();
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert!(!audit.geist_ingested);
    assert!(!audit.ism_written);
    assert!(!audit.identity_anchor);
}

#[test]
fn sleep_plan_audit_does_not_append_or_expose_gateway() {
    let candidate = valid_candidate();
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert!(!audit.evidence_archive_appended);
    assert!(!audit.gateway_visible);
}

#[test]
fn sleep_plan_audit_does_not_trigger_coordinator_runtime() {
    let candidate = valid_candidate();
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);

    assert_eq!(audit.source, MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    assert!(!audit.sleep_applied);
    assert!(!audit.sleep_completed);
    assert!(!audit.geist_ingested);
    assert!(!audit.ism_written);
    assert!(!audit.evidence_archive_appended);
    assert!(!audit.gateway_visible);
}
