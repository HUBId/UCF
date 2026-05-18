use ucf_replay::{
    build_replay_applied_boundary_from_audit, build_replay_schedule_from_minimal_spine_tokens,
    build_replay_token_from_minimal_spine_input, verify_minimal_spine_replay_schedule,
    MinimalSpineReplayAppliedBoundary, MinimalSpineReplayAuditStatus,
    MinimalSpineReplayScheduleAudit, MinimalSpineReplayScheduleBuildOutput,
    MinimalSpineReplayScheduleConfig, MinimalSpineReplayTokenBuildOutput,
    MinimalSpineReplayTokenInput, MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_types::Digest32;

#[derive(Clone, Debug)]
struct ReplayPipelineRun {
    tokens: Vec<MinimalSpineReplayTokenBuildOutput>,
    schedule: MinimalSpineReplayScheduleBuildOutput,
    audit: MinimalSpineReplayScheduleAudit,
    applied_boundary: MinimalSpineReplayAppliedBoundary,
}

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn zero_digest() -> Digest32 {
    Digest32::new([0u8; Digest32::LEN])
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

fn replay_inputs() -> [MinimalSpineReplayTokenInput; 3] {
    [replay_input(10), replay_input(40), replay_input(70)]
}

fn build_tokens() -> Vec<MinimalSpineReplayTokenBuildOutput> {
    replay_inputs()
        .iter()
        .map(|input| build_replay_token_from_minimal_spine_input(input).expect("replay token"))
        .collect()
}

fn sorted_replay_token_digests(tokens: &[MinimalSpineReplayTokenBuildOutput]) -> Vec<Digest32> {
    let mut digests: Vec<Digest32> = tokens
        .iter()
        .map(|token| token.replay_token_digest)
        .collect();
    digests.sort_by_key(|digest| *digest.as_bytes());
    digests
}

fn run_pipeline() -> ReplayPipelineRun {
    let tokens = build_tokens();
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &tokens,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("replay schedule");
    assert!(!schedule.applied);
    assert!(!schedule.sleep_cycle);
    assert!(!schedule.geist_ingested);
    assert!(!schedule.identity_anchor);

    let audit = verify_minimal_spine_replay_schedule(&schedule);
    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Pass);

    let applied_boundary =
        build_replay_applied_boundary_from_audit(&audit).expect("replay applied boundary");

    ReplayPipelineRun {
        tokens,
        schedule,
        audit,
        applied_boundary,
    }
}

#[test]
fn replay_pipeline_e2e_is_deterministic_across_fresh_runs() {
    let first = run_pipeline();
    let second = run_pipeline();

    let first_token_digests: Vec<Digest32> = first
        .tokens
        .iter()
        .map(|token| token.replay_token_digest)
        .collect();
    let second_token_digests: Vec<Digest32> = second
        .tokens
        .iter()
        .map(|token| token.replay_token_digest)
        .collect();

    assert_eq!(first_token_digests, second_token_digests);
    assert_eq!(
        first.schedule.schedule_digest,
        second.schedule.schedule_digest
    );
    assert_eq!(first.audit.audit_digest, second.audit.audit_digest);
    assert_eq!(
        first.applied_boundary.applied_boundary_digest,
        second.applied_boundary.applied_boundary_digest
    );
    assert_eq!(
        first.schedule.deterministic_bytes(),
        second.schedule.deterministic_bytes()
    );
    assert_eq!(
        first.audit.deterministic_bytes(),
        second.audit.deterministic_bytes()
    );
    assert_eq!(
        first.applied_boundary.deterministic_bytes(),
        second.applied_boundary.deterministic_bytes()
    );
}

#[test]
fn replay_pipeline_preserves_token_to_applied_provenance() {
    let run = run_pipeline();
    let expected_order = sorted_replay_token_digests(&run.tokens);

    assert_eq!(run.schedule.replay_token_digests, expected_order);
    assert_eq!(run.audit.token_digests, run.schedule.replay_token_digests);
    assert_eq!(run.audit.schedule_digest, run.schedule.schedule_digest);
    assert_eq!(run.applied_boundary.audit_digest, run.audit.audit_digest);
    assert_eq!(
        run.applied_boundary.schedule_digest,
        run.schedule.schedule_digest
    );
    assert_eq!(run.schedule.token_count, run.audit.token_count);
    assert_eq!(run.audit.token_count, run.applied_boundary.token_count);
    assert_eq!(
        usize::try_from(run.applied_boundary.token_count).expect("token count fits usize"),
        run.tokens.len()
    );

    for (index, provenance) in run.schedule.scheduled_token_provenance.iter().enumerate() {
        let token = run
            .tokens
            .iter()
            .find(|token| token.replay_token_digest == provenance.replay_token_digest)
            .expect("scheduled provenance maps to a built token");
        assert_eq!(
            provenance.order,
            u32::try_from(index).expect("order fits u32")
        );
        assert_eq!(provenance.token_build_output_digest, token.digest());
        assert_eq!(
            provenance.macro_candidate_digest,
            token.macro_candidate_digest
        );
        assert_eq!(
            provenance.macro_milestone_digest,
            token.macro_milestone_digest
        );
        assert_eq!(
            provenance.meso_aggregation_digest,
            token.meso_aggregation_digest
        );
        assert_eq!(
            provenance.macro_finalization_digest,
            token.macro_finalization_digest
        );
        assert_eq!(provenance.meso_count, token.meso_count);
    }
}

#[test]
fn replay_pipeline_requires_pass_audit_before_applied_boundary() {
    let run = run_pipeline();

    assert_eq!(run.audit.status, MinimalSpineReplayAuditStatus::Pass);
    assert!(build_replay_applied_boundary_from_audit(&run.audit).is_ok());

    let mut tampered_schedule = run.schedule.clone();
    tampered_schedule.schedule_digest = zero_digest();
    let fail_audit = verify_minimal_spine_replay_schedule(&tampered_schedule);

    assert_eq!(fail_audit.status, MinimalSpineReplayAuditStatus::Fail);
    assert!(build_replay_applied_boundary_from_audit(&fail_audit).is_err());
}

#[test]
fn replay_pipeline_has_no_sleep_geist_identity_side_effects() {
    let run = run_pipeline();

    assert!(!run.schedule.applied);
    assert!(!run.schedule.sleep_cycle);
    assert!(!run.schedule.geist_ingested);
    assert!(!run.schedule.identity_anchor);
    assert!(!run.schedule.evidence_archive_appended);

    assert_eq!(run.audit.status, MinimalSpineReplayAuditStatus::Pass);
    assert!(!run.audit.applied);
    assert!(!run.audit.replay_completed);
    assert!(!run.audit.sleep_cycle);
    assert!(!run.audit.geist_ingested);
    assert!(!run.audit.identity_anchor);
    assert!(!run.audit.evidence_archive_appended);

    assert!(run.applied_boundary.replay_subsystem_applied);
    assert!(!run.applied_boundary.geist_ingested);
    assert!(!run.applied_boundary.ism_written);
    assert!(!run.applied_boundary.identity_anchor);
    assert!(!run.applied_boundary.sleep_completed);
    assert!(!run.applied_boundary.evidence_archive_appended);
    assert!(!run.applied_boundary.gateway_visible);

    let replay_source = include_str!("../src/lib.rs");
    assert!(!replay_source.contains("ReplayApplied {"));
    assert!(!replay_source.contains("apply_replay_effects"));
}

#[test]
fn replay_pipeline_rejects_duplicate_or_invalid_inputs() {
    let tokens = build_tokens();
    let duplicate_tokens = vec![tokens[0], tokens[0]];
    assert!(build_replay_schedule_from_minimal_spine_tokens(
        &duplicate_tokens,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .is_err());

    let invalid_input = MinimalSpineReplayTokenInput {
        macro_candidate_digest: zero_digest(),
        ..replay_input(90)
    };
    assert!(build_replay_token_from_minimal_spine_input(&invalid_input).is_err());

    let mut tampered_schedule = build_replay_schedule_from_minimal_spine_tokens(
        &tokens,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");
    tampered_schedule.scheduled_token_provenance.swap(0, 1);
    let audit = verify_minimal_spine_replay_schedule(&tampered_schedule);
    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Fail);
}

#[test]
fn replay_pipeline_builders_do_not_schedule_runtime_queue() {
    let run = run_pipeline();

    assert!(!run.audit.evidence_archive_appended);
    assert!(!run.applied_boundary.evidence_archive_appended);
    assert!(!run.applied_boundary.gateway_visible);

    let replay_source = include_str!("../src/lib.rs");
    assert!(!replay_source.contains("RuntimeReplayApply"));
    assert!(!replay_source.contains("ReplayWorker"));
    assert!(!replay_source.contains("RuntimeScheduler"));
    assert!(!replay_source.contains("BackgroundQueue"));
    assert!(!replay_source.contains("spawn_replay_worker"));
}
