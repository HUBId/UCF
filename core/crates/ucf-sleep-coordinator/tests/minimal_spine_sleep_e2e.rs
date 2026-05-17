use ucf_replay::{
    build_replay_applied_boundary_from_audit, build_replay_schedule_from_minimal_spine_tokens,
    build_replay_token_from_minimal_spine_input, verify_minimal_spine_replay_schedule,
    MinimalSpineReplayAppliedBoundary, MinimalSpineReplayAuditStatus,
    MinimalSpineReplayScheduleAudit, MinimalSpineReplayScheduleBuildOutput,
    MinimalSpineReplayScheduleConfig, MinimalSpineReplayTokenBuildOutput,
    MinimalSpineReplayTokenInput, MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_sleep_coordinator::{
    build_sleep_applied_boundary_from_audit, build_sleep_plan_candidate_from_replay_audit,
    build_sleep_plan_candidate_from_replay_boundary, verify_minimal_spine_sleep_plan_candidate,
    MinimalSpineSleepAppliedBoundary, MinimalSpineSleepPlanAudit, MinimalSpineSleepPlanAuditStatus,
    MinimalSpineSleepPlanCandidate, MinimalSpineSleepPlanInput,
    MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE, MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE,
    MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE,
};
use ucf_types::Digest32;

#[derive(Clone, Debug)]
struct ReplayPipelineRun {
    tokens: Vec<MinimalSpineReplayTokenBuildOutput>,
    schedule: MinimalSpineReplayScheduleBuildOutput,
    audit: MinimalSpineReplayScheduleAudit,
    applied_boundary: MinimalSpineReplayAppliedBoundary,
}

#[derive(Clone, Debug)]
struct SleepPipelineRun {
    replay: ReplayPipelineRun,
    candidate: MinimalSpineSleepPlanCandidate,
    audit: MinimalSpineSleepPlanAudit,
    boundary: MinimalSpineSleepAppliedBoundary,
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
    [replay_input(11), replay_input(41), replay_input(71)]
}

fn build_tokens() -> Vec<MinimalSpineReplayTokenBuildOutput> {
    replay_inputs()
        .iter()
        .map(|input| build_replay_token_from_minimal_spine_input(input).expect("replay token"))
        .collect()
}

fn run_replay_pipeline() -> ReplayPipelineRun {
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
    assert!(!schedule.evidence_archive_appended);

    let audit = verify_minimal_spine_replay_schedule(&schedule);
    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Pass);
    assert!(!audit.applied);
    assert!(!audit.replay_completed);
    assert!(!audit.sleep_cycle);
    assert!(!audit.geist_ingested);
    assert!(!audit.identity_anchor);
    assert!(!audit.evidence_archive_appended);

    let applied_boundary =
        build_replay_applied_boundary_from_audit(&audit).expect("replay applied boundary");
    assert!(applied_boundary.replay_subsystem_applied);
    assert!(!applied_boundary.sleep_completed);
    assert!(!applied_boundary.geist_ingested);
    assert!(!applied_boundary.ism_written);
    assert!(!applied_boundary.identity_anchor);
    assert!(!applied_boundary.evidence_archive_appended);
    assert!(!applied_boundary.gateway_visible);

    ReplayPipelineRun {
        tokens,
        schedule,
        audit,
        applied_boundary,
    }
}

fn run_sleep_pipeline() -> SleepPipelineRun {
    let replay = run_replay_pipeline();
    let candidate =
        build_sleep_plan_candidate_from_replay_audit(&replay.audit, Some(&replay.applied_boundary))
            .expect("sleep plan candidate");
    assert_candidate_boundaries(&candidate);

    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    assert_audit_boundaries(&audit);

    let boundary = build_sleep_applied_boundary_from_audit(&audit).expect("sleep applied boundary");
    assert_boundary_boundaries(&boundary);

    SleepPipelineRun {
        replay,
        candidate,
        audit,
        boundary,
    }
}

fn assert_candidate_boundaries(candidate: &MinimalSpineSleepPlanCandidate) {
    assert_eq!(candidate.source, MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE);
    assert_eq!(candidate.sleep_plan_digest, candidate.digest());
    assert!(candidate.candidate_only);
    assert!(!candidate.sleep_applied);
    assert!(!candidate.sleep_completed);
    assert!(!candidate.geist_ingested);
    assert!(!candidate.ism_written);
    assert!(!candidate.identity_anchor);
    assert!(!candidate.evidence_archive_appended);
    assert!(!candidate.gateway_visible);
}

fn assert_audit_boundaries(audit: &MinimalSpineSleepPlanAudit) {
    assert_eq!(audit.source, MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE);
    assert_eq!(audit.audit_digest, audit.digest());
    assert!(audit.failure_reasons.is_empty());
    assert!(audit.candidate_only);
    assert!(!audit.sleep_applied);
    assert!(!audit.sleep_completed);
    assert!(!audit.geist_ingested);
    assert!(!audit.ism_written);
    assert!(!audit.identity_anchor);
    assert!(!audit.evidence_archive_appended);
    assert!(!audit.gateway_visible);
}

fn assert_boundary_boundaries(boundary: &MinimalSpineSleepAppliedBoundary) {
    assert_eq!(boundary.source, MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE);
    assert_eq!(boundary.applied_boundary_digest, boundary.digest());
    assert!(boundary.sleep_subsystem_applied);
    assert!(!boundary.sleep_completed);
    assert!(!boundary.coordinator_runtime_triggered);
    assert!(!boundary.geist_ingested);
    assert!(!boundary.ism_written);
    assert!(!boundary.identity_anchor);
    assert!(!boundary.memory_stabilized);
    assert!(!boundary.evidence_archive_appended);
    assert!(!boundary.gateway_visible);
}

fn assert_no_bounded_path_marker(value: &str) {
    for marker in [
        "SleepCompleted",
        "Geist",
        "ISM",
        "identity_anchor=true",
        "memory_stabilized=true",
        "EvidenceStore",
        "ArchiveStore",
        "append_evidence",
        "append_archive",
        "Gateway",
        "coordinator_runtime_triggered=true",
        "RuntimeScheduler",
        "WAL",
        "journal",
        "triggered=true",
        "report_ready",
    ] {
        assert!(
            !value.contains(marker),
            "bounded Sleep E2E path introduced forbidden marker {marker}"
        );
    }
}

fn assert_no_bounded_path_byte_marker(bytes: &[u8]) {
    let rendered = String::from_utf8_lossy(bytes);
    assert_no_bounded_path_marker(&rendered);
}

#[test]
fn sleep_pipeline_e2e_is_deterministic_across_fresh_runs() {
    let first = run_sleep_pipeline();
    let second = run_sleep_pipeline();

    assert_eq!(first.candidate, second.candidate);
    assert_eq!(first.audit, second.audit);
    assert_eq!(first.boundary, second.boundary);
    assert_eq!(
        first.candidate.sleep_plan_digest,
        second.candidate.sleep_plan_digest
    );
    assert_eq!(
        first.candidate.deterministic_bytes(),
        second.candidate.deterministic_bytes()
    );
    assert_eq!(first.audit.audit_digest, second.audit.audit_digest);
    assert_eq!(
        first.audit.deterministic_bytes(),
        second.audit.deterministic_bytes()
    );
    assert_eq!(
        first.boundary.applied_boundary_digest,
        second.boundary.applied_boundary_digest
    );
    assert_eq!(
        first.boundary.deterministic_bytes(),
        second.boundary.deterministic_bytes()
    );
}

#[test]
fn sleep_pipeline_preserves_replay_to_applied_provenance() {
    let run = run_sleep_pipeline();

    assert_eq!(
        run.candidate.replay_audit_digest,
        run.replay.audit.audit_digest
    );
    assert_eq!(
        run.candidate.replay_schedule_digest,
        run.replay.schedule.schedule_digest
    );
    assert_eq!(
        run.candidate.replay_applied_boundary_digest,
        Some(run.replay.applied_boundary.applied_boundary_digest)
    );
    assert_eq!(run.candidate.token_count, run.replay.audit.token_count);
    assert_eq!(
        usize::try_from(run.candidate.token_count).expect("token count fits usize"),
        run.replay.tokens.len()
    );

    assert_eq!(
        run.audit.sleep_plan_candidate_digest,
        run.candidate.sleep_plan_digest
    );
    assert_eq!(
        run.audit.recomputed_sleep_plan_candidate_digest,
        run.candidate.digest()
    );
    assert_eq!(
        run.audit.replay_audit_digest,
        run.candidate.replay_audit_digest
    );
    assert_eq!(
        run.audit.replay_schedule_digest,
        run.candidate.replay_schedule_digest
    );
    assert_eq!(
        run.audit.replay_applied_boundary_digest,
        run.candidate.replay_applied_boundary_digest
    );
    assert_eq!(run.audit.token_count, run.candidate.token_count);

    assert_eq!(run.boundary.sleep_plan_audit_digest, run.audit.audit_digest);
    assert_eq!(
        run.boundary.sleep_plan_candidate_digest,
        run.audit.sleep_plan_candidate_digest
    );
    assert_eq!(
        run.boundary.replay_audit_digest,
        run.audit.replay_audit_digest
    );
    assert_eq!(
        run.boundary.replay_schedule_digest,
        run.audit.replay_schedule_digest
    );
    assert_eq!(
        run.boundary.replay_applied_boundary_digest,
        run.audit.replay_applied_boundary_digest
    );
    assert_eq!(run.boundary.token_count, run.audit.token_count);
}

#[test]
fn sleep_pipeline_requires_pass_audit_before_applied_boundary() {
    let run = run_sleep_pipeline();

    assert_eq!(run.audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    assert!(build_sleep_applied_boundary_from_audit(&run.audit).is_ok());

    let mut tampered_candidate = run.candidate.clone();
    tampered_candidate.sleep_completed = true;
    let fail_audit = verify_minimal_spine_sleep_plan_candidate(&tampered_candidate);

    assert_eq!(fail_audit.status, MinimalSpineSleepPlanAuditStatus::Fail);
    assert!(build_sleep_applied_boundary_from_audit(&fail_audit).is_err());
}

#[test]
fn sleep_pipeline_has_no_runtime_geist_identity_side_effects() {
    let run = run_sleep_pipeline();

    assert_candidate_boundaries(&run.candidate);
    assert_audit_boundaries(&run.audit);
    assert_boundary_boundaries(&run.boundary);

    for source in [
        run.candidate.source,
        run.candidate.replay_source,
        run.audit.source,
        run.audit.candidate_source,
        run.audit.replay_source,
        run.boundary.source,
        run.boundary.sleep_plan_audit_source,
        run.boundary.candidate_source,
        run.boundary.replay_source,
    ] {
        assert_no_bounded_path_marker(source);
    }

    assert_no_bounded_path_byte_marker(&run.candidate.deterministic_bytes());
    assert_no_bounded_path_byte_marker(&run.audit.deterministic_bytes());
    assert_no_bounded_path_byte_marker(&run.boundary.deterministic_bytes());
}

#[test]
fn sleep_pipeline_rejects_invalid_inputs() {
    let replay = run_replay_pipeline();
    let invalid_input = MinimalSpineSleepPlanInput {
        replay_audit_digest: zero_digest(),
        replay_schedule_digest: replay.schedule.schedule_digest,
        replay_applied_boundary_digest: Some(replay.applied_boundary.applied_boundary_digest),
        token_count: replay.audit.token_count,
        source: replay.audit.source,
    };
    assert!(build_sleep_plan_candidate_from_replay_boundary(&invalid_input).is_err());

    let mut mismatched_boundary = replay.applied_boundary.clone();
    mismatched_boundary.schedule_digest = digest(222);
    mismatched_boundary.applied_boundary_digest = mismatched_boundary.digest();
    assert!(build_sleep_plan_candidate_from_replay_audit(
        &replay.audit,
        Some(&mismatched_boundary)
    )
    .is_err());

    let sleep = run_sleep_pipeline();
    let mut tampered_audit = sleep.audit.clone();
    tampered_audit.sleep_plan_candidate_digest = zero_digest();
    assert_eq!(
        tampered_audit.status,
        MinimalSpineSleepPlanAuditStatus::Pass
    );
    assert!(build_sleep_applied_boundary_from_audit(&tampered_audit).is_err());
}

#[test]
fn sleep_pipeline_does_not_append_or_activate_coordinator_runtime() {
    let run = run_sleep_pipeline();

    assert!(!run.candidate.evidence_archive_appended);
    assert!(!run.candidate.gateway_visible);
    assert!(!run.audit.evidence_archive_appended);
    assert!(!run.audit.gateway_visible);
    assert!(!run.boundary.evidence_archive_appended);
    assert!(!run.boundary.gateway_visible);
    assert!(!run.boundary.coordinator_runtime_triggered);

    assert_no_bounded_path_byte_marker(&run.candidate.deterministic_bytes());
    assert_no_bounded_path_byte_marker(&run.audit.deterministic_bytes());
    assert_no_bounded_path_byte_marker(&run.boundary.deterministic_bytes());
}
