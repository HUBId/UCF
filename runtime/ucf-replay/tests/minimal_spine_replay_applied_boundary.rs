use ucf_replay::{
    build_replay_applied_boundary_from_audit, build_replay_schedule_from_minimal_spine_tokens,
    build_replay_token_from_minimal_spine_input, verify_minimal_spine_replay_schedule,
    MinimalSpineReplayAuditStatus, MinimalSpineReplayScheduleConfig,
    MinimalSpineReplayTokenBuildOutput, MinimalSpineReplayTokenInput,
    MINIMAL_SPINE_REPLAY_APPLIED_BOUNDARY_SOURCE, MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
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

fn pass_audit() -> ucf_replay::MinimalSpineReplayScheduleAudit {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &[token(10), token(20), token(30)],
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");
    let audit = verify_minimal_spine_replay_schedule(&schedule);
    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Pass);
    audit
}

fn fail_audit() -> ucf_replay::MinimalSpineReplayScheduleAudit {
    let mut schedule = build_replay_schedule_from_minimal_spine_tokens(
        &[token(10), token(20), token(30)],
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");
    schedule.applied = true;
    let audit = verify_minimal_spine_replay_schedule(&schedule);
    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Fail);
    audit
}

#[test]
fn replay_applied_boundary_from_pass_audit_is_deterministic() {
    let audit = pass_audit();

    let first = build_replay_applied_boundary_from_audit(&audit).expect("applied boundary");
    let second = build_replay_applied_boundary_from_audit(&audit).expect("applied boundary");

    assert_eq!(first, second);
    assert_eq!(
        first.applied_boundary_digest,
        second.applied_boundary_digest
    );
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.applied_boundary_digest, first.digest());
}

#[test]
fn replay_applied_boundary_rejects_failed_audit() {
    let audit = fail_audit();

    let result = build_replay_applied_boundary_from_audit(&audit);

    assert!(result.is_err());
}

#[test]
fn replay_applied_boundary_preserves_audit_and_schedule_provenance() {
    let audit = pass_audit();

    let boundary = build_replay_applied_boundary_from_audit(&audit).expect("applied boundary");

    assert_eq!(boundary.audit_digest, audit.audit_digest);
    assert_eq!(boundary.schedule_digest, audit.schedule_digest);
    assert_eq!(boundary.token_count, audit.token_count);
    assert_eq!(
        boundary.source,
        MINIMAL_SPINE_REPLAY_APPLIED_BOUNDARY_SOURCE
    );
    assert!(boundary.replay_subsystem_applied);
}

#[test]
fn replay_applied_boundary_is_not_geist_or_ism() {
    let boundary = build_replay_applied_boundary_from_audit(&pass_audit()).expect("boundary");

    assert!(!boundary.geist_ingested);
    assert!(!boundary.ism_written);
    assert!(!boundary.identity_anchor);
}

#[test]
fn replay_applied_boundary_is_not_sleep_completion() {
    let boundary = build_replay_applied_boundary_from_audit(&pass_audit()).expect("boundary");

    assert!(!boundary.sleep_completed);
}

#[test]
fn replay_applied_boundary_does_not_append_or_expose_gateway() {
    let boundary = build_replay_applied_boundary_from_audit(&pass_audit()).expect("boundary");

    assert!(!boundary.evidence_archive_appended);
    assert!(!boundary.gateway_visible);
}

#[test]
fn replay_applied_boundary_does_not_mutate_audit() {
    let audit = pass_audit();
    let before = audit.clone();

    let _boundary = build_replay_applied_boundary_from_audit(&audit).expect("boundary");

    assert_eq!(audit, before);
}

#[test]
fn replay_applied_boundary_does_not_emit_broad_replay_applied_runtime() {
    let boundary = build_replay_applied_boundary_from_audit(&pass_audit()).expect("boundary");

    assert!(boundary.replay_subsystem_applied);
    let replay_source = include_str!("../src/lib.rs");
    assert!(!replay_source.contains("ReplayApplied {"));
    assert!(!replay_source.contains("apply_replay_effects"));
}
