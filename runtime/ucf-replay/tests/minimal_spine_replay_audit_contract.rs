use ucf_replay::{
    build_replay_schedule_from_minimal_spine_tokens, build_replay_token_from_minimal_spine_input,
    verify_minimal_spine_replay_schedule, MinimalSpineReplayAuditFailureReason,
    MinimalSpineReplayAuditStatus, MinimalSpineReplayScheduleConfig,
    MinimalSpineReplayTokenBuildOutput, MinimalSpineReplayTokenInput,
    MINIMAL_SPINE_REPLAY_SCHEDULE_AUDIT_SOURCE, MINIMAL_SPINE_REPLAY_SCHEDULE_SOURCE,
    MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_types::Digest32;

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn input(seed: u8) -> MinimalSpineReplayTokenInput {
    MinimalSpineReplayTokenInput {
        macro_candidate_digest: digest(seed),
        macro_milestone_digest: digest(seed.saturating_add(1)),
        meso_aggregation_digest: digest(seed.saturating_add(2)),
        macro_finalization_digest: digest(seed.saturating_add(3)),
        meso_count: u32::from(seed),
        source: MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
    }
}

fn token(seed: u8) -> MinimalSpineReplayTokenBuildOutput {
    build_replay_token_from_minimal_spine_input(&input(seed)).expect("token output")
}

fn three_tokens() -> Vec<MinimalSpineReplayTokenBuildOutput> {
    vec![token(10), token(20), token(30)]
}

#[test]
fn replay_audit_passes_for_valid_schedule() {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");

    let audit = verify_minimal_spine_replay_schedule(&schedule);

    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Pass);
    assert!(audit.failure_reasons.is_empty());
    assert_eq!(audit.schedule_digest, schedule.schedule_digest);
    assert_eq!(audit.recomputed_schedule_digest, schedule.digest());
    assert_eq!(audit.audit_digest, audit.digest());
    assert_eq!(audit.source, MINIMAL_SPINE_REPLAY_SCHEDULE_AUDIT_SOURCE);
}

#[test]
fn replay_audit_is_deterministic() {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");

    let first = verify_minimal_spine_replay_schedule(&schedule);
    let second = verify_minimal_spine_replay_schedule(&schedule);

    assert_eq!(first, second);
    assert_eq!(first.audit_digest, second.audit_digest);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
}

#[test]
fn replay_audit_detects_invalid_schedule_metadata() {
    let mut schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");
    schedule.token_count = schedule.token_count.saturating_add(1);
    schedule.applied = true;

    let audit = verify_minimal_spine_replay_schedule(&schedule);

    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Fail);
    assert!(audit
        .failure_reasons
        .contains(&MinimalSpineReplayAuditFailureReason::TokenCountMismatch));
    assert!(audit
        .failure_reasons
        .contains(&MinimalSpineReplayAuditFailureReason::ScheduleDigestMismatch));
    assert!(audit
        .failure_reasons
        .contains(&MinimalSpineReplayAuditFailureReason::AppliedFlagSet));
    assert!(!audit.applied);
}

#[test]
fn replay_audit_reports_token_order_and_count() {
    let tokens = three_tokens();
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &tokens,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");

    let audit = verify_minimal_spine_replay_schedule(&schedule);

    assert_eq!(audit.token_count, schedule.token_count);
    assert_eq!(audit.token_digests, schedule.replay_token_digests);
    assert_eq!(
        audit.token_digests.len(),
        schedule.replay_token_digests.len()
    );
    assert!(audit.duplicate_free);
}

#[test]
fn replay_audit_reports_cap_and_truncation_metadata() {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig {
            max_tokens: Some(1),
            source: MINIMAL_SPINE_REPLAY_SCHEDULE_SOURCE,
        },
    )
    .expect("capped schedule");

    let audit = verify_minimal_spine_replay_schedule(&schedule);

    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Pass);
    assert_eq!(audit.token_count, 1);
    assert!(audit.truncated);
    assert_eq!(audit.token_digests, schedule.replay_token_digests);
}

#[test]
fn replay_audit_detects_duplicate_token_digests_if_schedule_is_tampered() {
    let mut schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");
    schedule.replay_token_digests[1] = schedule.replay_token_digests[0];

    let audit = verify_minimal_spine_replay_schedule(&schedule);

    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Fail);
    assert!(!audit.duplicate_free);
    assert!(audit
        .failure_reasons
        .contains(&MinimalSpineReplayAuditFailureReason::DuplicateReplayTokenDigest));
    assert!(audit
        .failure_reasons
        .contains(&MinimalSpineReplayAuditFailureReason::ProvenanceDigestOrderMismatch));
}

#[test]
fn replay_audit_is_verify_only_not_applied() {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");
    let before = schedule.clone();

    let audit = verify_minimal_spine_replay_schedule(&schedule);

    assert_eq!(schedule, before);
    assert!(!audit.applied);
    assert!(!audit.replay_completed);
    let replay_source = include_str!("../src/lib.rs");
    assert!(!replay_source.contains("ReplayApplied {"));
}

#[test]
fn replay_audit_has_no_sleep_geist_identity_side_effects() {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");

    let audit = verify_minimal_spine_replay_schedule(&schedule);

    assert!(!audit.sleep_cycle);
    assert!(!audit.geist_ingested);
    assert!(!audit.identity_anchor);
}

#[test]
fn replay_audit_does_not_append_to_evidence_archive() {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");

    let audit = verify_minimal_spine_replay_schedule(&schedule);

    assert!(!audit.evidence_archive_appended);
    assert!(!audit.deterministic_bytes().is_empty());
}
