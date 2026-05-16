use ucf_replay::{
    build_replay_applied_boundary_from_audit, build_replay_schedule_from_minimal_spine_tokens,
    build_replay_token_from_minimal_spine_input, verify_minimal_spine_replay_schedule,
    MinimalSpineReplayAppliedBoundary, MinimalSpineReplayAuditStatus,
    MinimalSpineReplayScheduleAudit, MinimalSpineReplayScheduleConfig,
    MinimalSpineReplayTokenInput, MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_sleep_coordinator::{
    build_sleep_plan_candidate_from_replay_audit, build_sleep_plan_candidate_from_replay_boundary,
    MinimalSpineSleepPlanInput, SleepPlanCandidateError,
};
use ucf_types::Digest32;

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn replay_input(seed: u8) -> MinimalSpineReplayTokenInput {
    MinimalSpineReplayTokenInput {
        macro_candidate_digest: digest(seed),
        macro_milestone_digest: digest(seed.saturating_add(1)),
        meso_aggregation_digest: digest(seed.saturating_add(2)),
        macro_finalization_digest: digest(seed.saturating_add(3)),
        meso_count: u32::from(seed),
        source: MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
    }
}

fn build_replay_audit_and_boundary() -> (
    MinimalSpineReplayScheduleAudit,
    MinimalSpineReplayAppliedBoundary,
) {
    let tokens = [replay_input(10), replay_input(40), replay_input(70)]
        .iter()
        .map(|input| build_replay_token_from_minimal_spine_input(input).expect("replay token"))
        .collect::<Vec<_>>();
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &tokens,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("replay schedule");
    let audit = verify_minimal_spine_replay_schedule(&schedule);
    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Pass);
    let boundary = build_replay_applied_boundary_from_audit(&audit).expect("replay boundary");
    (audit, boundary)
}

fn digest_input(byte: u8) -> MinimalSpineSleepPlanInput {
    MinimalSpineSleepPlanInput {
        replay_audit_digest: digest(byte),
        replay_schedule_digest: digest(byte.saturating_add(1)),
        replay_applied_boundary_digest: Some(digest(byte.saturating_add(2))),
        token_count: 3,
        source: "test_replay_boundary",
    }
}

#[test]
fn sleep_plan_candidate_from_replay_boundary_is_deterministic() {
    let input = digest_input(20);

    let first = build_sleep_plan_candidate_from_replay_boundary(&input).expect("first candidate");
    let second = build_sleep_plan_candidate_from_replay_boundary(&input).expect("second candidate");

    assert_eq!(first, second);
    assert_eq!(first.sleep_plan_digest, second.sleep_plan_digest);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.sleep_plan_digest, first.digest());
}

#[test]
fn sleep_plan_candidate_changes_when_replay_digest_changes() {
    let baseline = build_sleep_plan_candidate_from_replay_boundary(&digest_input(30))
        .expect("baseline candidate");

    let mut changed_audit = digest_input(30);
    changed_audit.replay_audit_digest = digest(90);
    let changed_audit = build_sleep_plan_candidate_from_replay_boundary(&changed_audit)
        .expect("changed audit candidate");

    let mut changed_schedule = digest_input(30);
    changed_schedule.replay_schedule_digest = digest(91);
    let changed_schedule = build_sleep_plan_candidate_from_replay_boundary(&changed_schedule)
        .expect("changed schedule candidate");

    let mut changed_boundary = digest_input(30);
    changed_boundary.replay_applied_boundary_digest = Some(digest(92));
    let changed_boundary = build_sleep_plan_candidate_from_replay_boundary(&changed_boundary)
        .expect("changed boundary candidate");

    assert_ne!(baseline.sleep_plan_digest, changed_audit.sleep_plan_digest);
    assert_ne!(
        baseline.sleep_plan_digest,
        changed_schedule.sleep_plan_digest
    );
    assert_ne!(
        baseline.sleep_plan_digest,
        changed_boundary.sleep_plan_digest
    );
}

#[test]
fn sleep_plan_candidate_rejects_failed_or_invalid_replay_audit() {
    let (audit, _) = build_replay_audit_and_boundary();
    let mut failed = audit.clone();
    failed.status = MinimalSpineReplayAuditStatus::Fail;
    assert_eq!(
        build_sleep_plan_candidate_from_replay_audit(&failed, None),
        Err(SleepPlanCandidateError::AuditStatusNotPass)
    );

    let mut invalid = digest_input(40);
    invalid.replay_audit_digest = Digest32::new([0u8; Digest32::LEN]);
    assert_eq!(
        build_sleep_plan_candidate_from_replay_boundary(&invalid),
        Err(SleepPlanCandidateError::ZeroReplayAuditDigest)
    );
}

#[test]
fn sleep_plan_candidate_validates_optional_applied_boundary_match() {
    let (audit, boundary) = build_replay_audit_and_boundary();
    build_sleep_plan_candidate_from_replay_audit(&audit, Some(&boundary))
        .expect("matching boundary is accepted");

    let mut wrong_audit_digest = boundary.clone();
    wrong_audit_digest.audit_digest = digest(100);
    wrong_audit_digest.applied_boundary_digest = wrong_audit_digest.digest();
    assert_eq!(
        build_sleep_plan_candidate_from_replay_audit(&audit, Some(&wrong_audit_digest)),
        Err(SleepPlanCandidateError::BoundaryAuditDigestMismatch)
    );

    let mut wrong_schedule_digest = boundary.clone();
    wrong_schedule_digest.schedule_digest = digest(101);
    wrong_schedule_digest.applied_boundary_digest = wrong_schedule_digest.digest();
    assert_eq!(
        build_sleep_plan_candidate_from_replay_audit(&audit, Some(&wrong_schedule_digest)),
        Err(SleepPlanCandidateError::BoundaryScheduleDigestMismatch)
    );

    let mut wrong_token_count = boundary.clone();
    wrong_token_count.token_count = wrong_token_count.token_count.saturating_add(1);
    wrong_token_count.applied_boundary_digest = wrong_token_count.digest();
    assert_eq!(
        build_sleep_plan_candidate_from_replay_audit(&audit, Some(&wrong_token_count)),
        Err(SleepPlanCandidateError::BoundaryTokenCountMismatch)
    );
}

#[test]
fn sleep_plan_candidate_preserves_replay_provenance() {
    let (audit, boundary) = build_replay_audit_and_boundary();
    let candidate = build_sleep_plan_candidate_from_replay_audit(&audit, Some(&boundary))
        .expect("sleep plan candidate");

    assert_eq!(candidate.replay_audit_digest, audit.audit_digest);
    assert_eq!(candidate.replay_schedule_digest, audit.schedule_digest);
    assert_eq!(
        candidate.replay_applied_boundary_digest,
        Some(boundary.applied_boundary_digest)
    );
    assert_eq!(candidate.token_count, audit.token_count);
    assert_eq!(candidate.replay_source, audit.source);
}

#[test]
fn sleep_plan_candidate_is_candidate_only() {
    let (audit, boundary) = build_replay_audit_and_boundary();
    let candidate = build_sleep_plan_candidate_from_replay_audit(&audit, Some(&boundary))
        .expect("sleep plan candidate");

    assert!(candidate.candidate_only);
    assert!(!candidate.sleep_applied);
    assert!(!candidate.sleep_completed);
}

#[test]
fn sleep_plan_candidate_has_no_geist_ism_identity_side_effects() {
    let (audit, boundary) = build_replay_audit_and_boundary();
    let candidate = build_sleep_plan_candidate_from_replay_audit(&audit, Some(&boundary))
        .expect("sleep plan candidate");

    assert!(!candidate.geist_ingested);
    assert!(!candidate.ism_written);
    assert!(!candidate.identity_anchor);
}

#[test]
fn sleep_plan_candidate_does_not_append_or_expose_gateway() {
    let (audit, boundary) = build_replay_audit_and_boundary();
    let candidate = build_sleep_plan_candidate_from_replay_audit(&audit, Some(&boundary))
        .expect("sleep plan candidate");

    assert!(!candidate.evidence_archive_appended);
    assert!(!candidate.gateway_visible);
}

#[test]
fn sleep_plan_candidate_does_not_activate_coordinator_runtime() {
    let (audit, boundary) = build_replay_audit_and_boundary();
    let candidate = build_sleep_plan_candidate_from_replay_audit(&audit, Some(&boundary))
        .expect("sleep plan candidate");

    assert_eq!(
        candidate.source,
        ucf_sleep_coordinator::MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE
    );
    assert!(candidate.candidate_only);
    assert!(!candidate.sleep_applied);
    assert!(!candidate.sleep_completed);
}
