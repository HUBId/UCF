use ucf_archive_store::{ArchiveAppender, ArchiveStore, InMemoryArchiveStore};
use ucf_evidence::{EvidenceStore, InMemoryEvidenceStore};
use ucf_replay::{
    build_replay_applied_boundary_from_audit, build_replay_schedule_from_minimal_spine_tokens,
    build_replay_token_from_minimal_spine_input, verify_minimal_spine_replay_schedule,
    MinimalSpineReplayAppliedBoundary, MinimalSpineReplayAuditStatus,
    MinimalSpineReplayScheduleAudit, MinimalSpineReplayScheduleBuildOutput,
    MinimalSpineReplayScheduleConfig, MinimalSpineReplayTokenBuildOutput,
    MinimalSpineReplayTokenInput, MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_sleep_coordinator::{
    append_minimal_spine_sleep_record, build_sleep_applied_boundary_from_audit,
    build_sleep_plan_candidate_from_replay_audit, verify_minimal_spine_sleep_plan_candidate,
    MinimalSpineSleepAppendPayload, MinimalSpineSleepAppliedBoundary, MinimalSpineSleepPlanAudit,
    MinimalSpineSleepPlanAuditStatus, MinimalSpineSleepPlanCandidate,
    MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, MINIMAL_SPINE_SLEEP_APPEND_CONTRACT,
    MINIMAL_SPINE_SLEEP_APPEND_MEANING, MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE,
    MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE, MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE,
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

fn build_tokens() -> Vec<MinimalSpineReplayTokenBuildOutput> {
    [replay_input(13), replay_input(43), replay_input(73)]
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

fn run_sleep_pipeline() -> SleepPipelineRun {
    let replay = run_replay_pipeline();
    let candidate =
        build_sleep_plan_candidate_from_replay_audit(&replay.audit, Some(&replay.applied_boundary))
            .expect("sleep plan candidate");
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    let boundary = build_sleep_applied_boundary_from_audit(&audit).expect("sleep applied boundary");
    SleepPipelineRun {
        replay,
        candidate,
        audit,
        boundary,
    }
}

#[test]
fn sleep_append_contract_is_explicit_and_readbackable() {
    let run = run_sleep_pipeline();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    assert!(evidence_store.is_empty());
    assert!(archive_store.root_commit().is_none());

    let result = append_minimal_spine_sleep_record(
        &run.candidate,
        &run.audit,
        &run.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("sleep append");

    let payload =
        MinimalSpineSleepAppendPayload::from_artifacts(&run.candidate, &run.audit, &run.boundary)
            .expect("payload");
    assert_eq!(result.payload_digest, payload.digest());
    assert_eq!(evidence_store.len(), 1);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, None)
            .count(),
        1
    );

    let evidence = evidence_store
        .get(result.appended_evidence_id.clone())
        .expect("evidence readback");
    let proof = evidence.proof.expect("proof envelope");
    assert_eq!(
        proof.envelope_id,
        format!(
            "{}:{}",
            MINIMAL_SPINE_SLEEP_APPEND_CONTRACT,
            hex::encode(result.payload_digest.as_bytes())
        )
    );
    assert_eq!(proof.payload, payload.deterministic_bytes());
    assert_eq!(
        proof
            .payload_digest
            .expect("payload digest")
            .value_32
            .expect("digest32"),
        result.payload_digest.as_bytes().to_vec()
    );

    let archive_record = archive_store
        .get(result.archive_key)
        .expect("archive readback");
    assert_eq!(archive_record.kind, MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND);
    assert_eq!(archive_record.payload_commit, result.payload_digest);
    assert_eq!(
        archive_record.meta.boundary_commit,
        run.boundary.applied_boundary_digest
    );
    assert_eq!(archive_record.meta.flags, 0);
    assert_eq!(archive_record.meta.cycle_id, 0);
    assert_eq!(archive_record.meta.tier, 3);
    assert!(!result
        .readback_digest
        .as_bytes()
        .iter()
        .all(|byte| *byte == 0));
    assert!(!result
        .archive_record_digest
        .as_bytes()
        .iter()
        .all(|byte| *byte == 0));
}

#[test]
fn sleep_append_preserves_candidate_audit_boundary_replay_provenance() {
    let run = run_sleep_pipeline();
    assert_eq!(
        run.replay.tokens.len(),
        run.replay.schedule.token_count as usize
    );

    let payload =
        MinimalSpineSleepAppendPayload::from_artifacts(&run.candidate, &run.audit, &run.boundary)
            .expect("payload");

    assert_eq!(
        payload.sleep_plan_candidate_digest,
        run.candidate.sleep_plan_digest
    );
    assert_eq!(payload.sleep_plan_audit_digest, run.audit.audit_digest);
    assert_eq!(
        payload.sleep_applied_boundary_digest,
        run.boundary.applied_boundary_digest
    );
    assert_eq!(payload.replay_audit_digest, run.replay.audit.audit_digest);
    assert_eq!(
        payload.replay_schedule_digest,
        run.replay.schedule.schedule_digest
    );
    assert_eq!(
        payload.replay_applied_boundary_digest,
        Some(run.replay.applied_boundary.applied_boundary_digest)
    );
    assert_eq!(payload.token_count, run.replay.schedule.token_count);
    assert_eq!(
        payload.candidate_source,
        MINIMAL_SPINE_SLEEP_PLAN_CANDIDATE_SOURCE
    );
    assert_eq!(payload.audit_source, MINIMAL_SPINE_SLEEP_PLAN_AUDIT_SOURCE);
    assert_eq!(
        payload.applied_boundary_source,
        MINIMAL_SPINE_SLEEP_APPLIED_BOUNDARY_SOURCE
    );
    assert_eq!(payload.replay_source, run.candidate.replay_source);
    assert_eq!(payload.source, MINIMAL_SPINE_SLEEP_APPEND_CONTRACT);
    assert_eq!(
        payload.evidence_archive_appended_meaning,
        MINIMAL_SPINE_SLEEP_APPEND_MEANING
    );
}

#[test]
fn sleep_append_is_deterministic_for_fresh_stores() {
    let run = run_sleep_pipeline();
    let first_evidence = InMemoryEvidenceStore::new();
    let first_archive = InMemoryArchiveStore::new();
    let mut first_appender = ArchiveAppender::new();
    let second_evidence = InMemoryEvidenceStore::new();
    let second_archive = InMemoryArchiveStore::new();
    let mut second_appender = ArchiveAppender::new();

    let first = append_minimal_spine_sleep_record(
        &run.candidate,
        &run.audit,
        &run.boundary,
        &first_evidence,
        &first_archive,
        &mut first_appender,
    )
    .expect("first append");
    let second = append_minimal_spine_sleep_record(
        &run.candidate,
        &run.audit,
        &run.boundary,
        &second_evidence,
        &second_archive,
        &mut second_appender,
    )
    .expect("second append");

    assert_eq!(first.payload_digest, second.payload_digest);
    assert_eq!(first.appended_evidence_id, second.appended_evidence_id);
    assert_eq!(first.archive_key, second.archive_key);
    assert_eq!(first.archive_record_digest, second.archive_record_digest);
    assert_eq!(first.readback_digest, second.readback_digest);
}

#[test]
fn sleep_builders_remain_append_free() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();

    let run = run_sleep_pipeline();
    assert_eq!(run.candidate.sleep_plan_digest, run.candidate.digest());
    assert_eq!(run.audit.audit_digest, run.audit.digest());
    assert_eq!(run.boundary.applied_boundary_digest, run.boundary.digest());

    assert!(evidence_store.is_empty());
    assert!(archive_store.root_commit().is_none());
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, None)
            .count(),
        0
    );
}

#[test]
fn sleep_append_rejects_invalid_or_mismatched_inputs() {
    let run = run_sleep_pipeline();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();

    let mut mismatched_boundary = run.boundary.clone();
    mismatched_boundary.token_count = mismatched_boundary.token_count.saturating_add(1);
    mismatched_boundary.applied_boundary_digest = mismatched_boundary.digest();
    assert!(append_minimal_spine_sleep_record(
        &run.candidate,
        &run.audit,
        &mismatched_boundary,
        &evidence_store,
        &archive_store,
        &mut ArchiveAppender::new(),
    )
    .is_err());

    let mut failed_candidate = run.candidate.clone();
    failed_candidate.sleep_completed = true;
    failed_candidate.sleep_plan_digest = failed_candidate.digest();
    let failed_audit = verify_minimal_spine_sleep_plan_candidate(&failed_candidate);
    assert_eq!(failed_audit.status, MinimalSpineSleepPlanAuditStatus::Fail);
    assert!(append_minimal_spine_sleep_record(
        &failed_candidate,
        &failed_audit,
        &run.boundary,
        &evidence_store,
        &archive_store,
        &mut ArchiveAppender::new(),
    )
    .is_err());

    let mut zero_candidate = run.candidate.clone();
    zero_candidate.replay_audit_digest = zero_digest();
    zero_candidate.sleep_plan_digest = zero_candidate.digest();
    let zero_audit = verify_minimal_spine_sleep_plan_candidate(&zero_candidate);
    assert!(append_minimal_spine_sleep_record(
        &zero_candidate,
        &zero_audit,
        &run.boundary,
        &evidence_store,
        &archive_store,
        &mut ArchiveAppender::new(),
    )
    .is_err());

    let mut zero_token_candidate = run.candidate.clone();
    zero_token_candidate.token_count = 0;
    zero_token_candidate.sleep_plan_digest = zero_token_candidate.digest();
    let zero_token_audit = verify_minimal_spine_sleep_plan_candidate(&zero_token_candidate);
    assert!(append_minimal_spine_sleep_record(
        &zero_token_candidate,
        &zero_token_audit,
        &run.boundary,
        &evidence_store,
        &archive_store,
        &mut ArchiveAppender::new(),
    )
    .is_err());
}

#[test]
fn sleep_append_does_not_trigger_runtime_coordinator_geist_identity() {
    let run = run_sleep_pipeline();
    let payload =
        MinimalSpineSleepAppendPayload::from_artifacts(&run.candidate, &run.audit, &run.boundary)
            .expect("payload");

    assert!(!payload.runtime_executed);
    assert!(!payload.coordinator_triggered);
    assert!(!payload.coordinator_report_written);
    assert!(!payload.coordinator_wal_written);
    assert!(!payload.coordinator_journal_written);
    assert!(!payload.sleep_completed);
    assert!(!payload.geist_ingested);
    assert!(!payload.ism_written);
    assert!(!payload.identity_anchor);
    assert!(!payload.memory_stabilized);
    assert!(!payload.gateway_visible);
    assert_eq!(
        payload.evidence_archive_appended_meaning,
        MINIMAL_SPINE_SLEEP_APPEND_MEANING
    );
}

#[test]
fn sleep_append_does_not_create_second_event_log() {
    let run = run_sleep_pipeline();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    append_minimal_spine_sleep_record(
        &run.candidate,
        &run.audit,
        &run.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("sleep append");

    assert_eq!(evidence_store.len(), 1);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, None)
            .count(),
        1
    );
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, Some(1))
            .count(),
        1
    );
}
