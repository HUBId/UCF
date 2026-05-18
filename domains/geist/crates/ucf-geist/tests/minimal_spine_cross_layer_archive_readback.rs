use ucf_archive_store::{ArchiveAppender, ArchiveStore, InMemoryArchiveStore, RecordKind};
use ucf_evidence::{EvidenceStore, InMemoryEvidenceStore};
use ucf_geist::{
    append_minimal_spine_geist_ism_record, build_geist_projection_candidate_from_sleep_audit,
    build_ism_candidate_boundary_from_geist_audit, verify_minimal_spine_geist_projection_candidate,
    MinimalSpineGeistIsmAppendPayload, MinimalSpineGeistIsmAppendResult,
    MinimalSpineGeistProjectionAudit, MinimalSpineGeistProjectionAuditStatus,
    MinimalSpineGeistProjectionCandidate, MinimalSpineIsmCandidateBoundary,
    MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, MINIMAL_SPINE_GEIST_ISM_APPEND_MEANING,
};
use ucf_replay::{
    append_minimal_spine_replay_record, build_replay_applied_boundary_from_audit,
    build_replay_schedule_from_minimal_spine_tokens, build_replay_token_from_minimal_spine_input,
    verify_minimal_spine_replay_schedule, MinimalSpineReplayAppendPayload,
    MinimalSpineReplayAppendResult, MinimalSpineReplayAppliedBoundary,
    MinimalSpineReplayAuditStatus, MinimalSpineReplayScheduleAudit,
    MinimalSpineReplayScheduleBuildOutput, MinimalSpineReplayScheduleConfig,
    MinimalSpineReplayTokenBuildOutput, MinimalSpineReplayTokenInput,
    MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND, MINIMAL_SPINE_REPLAY_APPEND_MEANING,
    MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_sleep_coordinator::{
    append_minimal_spine_sleep_record, build_sleep_applied_boundary_from_audit,
    build_sleep_plan_candidate_from_replay_audit, verify_minimal_spine_sleep_plan_candidate,
    MinimalSpineSleepAppendPayload, MinimalSpineSleepAppendResult,
    MinimalSpineSleepAppliedBoundary, MinimalSpineSleepPlanAudit, MinimalSpineSleepPlanAuditStatus,
    MinimalSpineSleepPlanCandidate, MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND,
    MINIMAL_SPINE_SLEEP_APPEND_MEANING,
};
use ucf_types::Digest32;

#[derive(Clone, Debug)]
struct ReplayArtifacts {
    tokens: Vec<MinimalSpineReplayTokenBuildOutput>,
    schedule: MinimalSpineReplayScheduleBuildOutput,
    audit: MinimalSpineReplayScheduleAudit,
    boundary: MinimalSpineReplayAppliedBoundary,
    payload: MinimalSpineReplayAppendPayload,
}

#[derive(Clone, Debug)]
struct SleepArtifacts {
    candidate: MinimalSpineSleepPlanCandidate,
    audit: MinimalSpineSleepPlanAudit,
    boundary: MinimalSpineSleepAppliedBoundary,
    payload: MinimalSpineSleepAppendPayload,
}

#[derive(Clone, Debug)]
struct GeistArtifacts {
    candidate: MinimalSpineGeistProjectionCandidate,
    audit: MinimalSpineGeistProjectionAudit,
    boundary: MinimalSpineIsmCandidateBoundary,
    payload: MinimalSpineGeistIsmAppendPayload,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CrossLayerReadbackSummary {
    replay_payload_digest: Digest32,
    replay_readback_digest: Digest32,
    replay_archive_record_digest: Digest32,
    replay_archive_key: Digest32,
    sleep_payload_digest: Digest32,
    sleep_readback_digest: Digest32,
    sleep_archive_record_digest: Digest32,
    sleep_archive_key: Digest32,
    geist_payload_digest: Digest32,
    geist_readback_digest: Digest32,
    geist_archive_record_digest: Digest32,
    geist_archive_key: Digest32,
    archive_root_commit: Option<Digest32>,
    evidence_count: usize,
    replay_archive_count: usize,
    sleep_archive_count: usize,
    geist_archive_count: usize,
}

#[derive(Clone, Debug)]
struct CrossLayerRun {
    replay: ReplayArtifacts,
    sleep: SleepArtifacts,
    geist: GeistArtifacts,
    replay_result: MinimalSpineReplayAppendResult,
    sleep_result: MinimalSpineSleepAppendResult,
    geist_result: MinimalSpineGeistIsmAppendResult,
    summary: CrossLayerReadbackSummary,
}

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

fn build_replay_artifacts() -> ReplayArtifacts {
    let tokens: Vec<_> = [replay_input(11), replay_input(41), replay_input(71)]
        .iter()
        .map(|input| build_replay_token_from_minimal_spine_input(input).expect("replay token"))
        .collect();
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &tokens,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("replay schedule");
    let audit = verify_minimal_spine_replay_schedule(&schedule);
    assert_eq!(audit.status, MinimalSpineReplayAuditStatus::Pass);
    let boundary = build_replay_applied_boundary_from_audit(&audit).expect("replay boundary");
    let payload = MinimalSpineReplayAppendPayload::from_artifacts(&schedule, &audit, &boundary)
        .expect("replay payload");

    ReplayArtifacts {
        tokens,
        schedule,
        audit,
        boundary,
        payload,
    }
}

fn build_sleep_artifacts(replay: &ReplayArtifacts) -> SleepArtifacts {
    let candidate =
        build_sleep_plan_candidate_from_replay_audit(&replay.audit, Some(&replay.boundary))
            .expect("sleep candidate from replay provenance");
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    let boundary = build_sleep_applied_boundary_from_audit(&audit).expect("sleep boundary");
    let payload = MinimalSpineSleepAppendPayload::from_artifacts(&candidate, &audit, &boundary)
        .expect("sleep payload");

    SleepArtifacts {
        candidate,
        audit,
        boundary,
        payload,
    }
}

fn build_geist_artifacts(sleep: &SleepArtifacts) -> GeistArtifacts {
    let candidate =
        build_geist_projection_candidate_from_sleep_audit(&sleep.audit, Some(&sleep.boundary))
            .expect("geist candidate from sleep provenance");
    let audit = verify_minimal_spine_geist_projection_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineGeistProjectionAuditStatus::Pass);
    let boundary = build_ism_candidate_boundary_from_geist_audit(&audit).expect("ism boundary");
    let payload = MinimalSpineGeistIsmAppendPayload::from_artifacts(&candidate, &audit, &boundary)
        .expect("geist payload");

    GeistArtifacts {
        candidate,
        audit,
        boundary,
        payload,
    }
}

fn build_cross_layer_artifacts() -> (ReplayArtifacts, SleepArtifacts, GeistArtifacts) {
    let replay = build_replay_artifacts();
    let sleep = build_sleep_artifacts(&replay);
    let geist = build_geist_artifacts(&sleep);
    (replay, sleep, geist)
}

fn run_cross_layer_pipeline() -> CrossLayerRun {
    let (replay, sleep, geist) = build_cross_layer_artifacts();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    let replay_result = append_minimal_spine_replay_record(
        &replay.schedule,
        &replay.audit,
        &replay.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append replay");
    assert_eq!(evidence_store.len(), 1);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND, None)
            .count(),
        1
    );

    let sleep_result = append_minimal_spine_sleep_record(
        &sleep.candidate,
        &sleep.audit,
        &sleep.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append sleep");
    assert_eq!(evidence_store.len(), 2);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, None)
            .count(),
        1
    );

    let geist_result = append_minimal_spine_geist_ism_record(
        &geist.candidate,
        &geist.audit,
        &geist.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append geist/ism");

    assert_evidence_payload_readback(
        &evidence_store,
        &replay_result.appended_evidence_id,
        replay.payload.deterministic_bytes(),
    );
    assert_evidence_payload_readback(
        &evidence_store,
        &sleep_result.appended_evidence_id,
        sleep.payload.deterministic_bytes(),
    );
    assert_evidence_payload_readback(
        &evidence_store,
        &geist_result.appended_evidence_id,
        geist.payload.deterministic_bytes(),
    );

    assert_eq!(
        archive_store
            .get(replay_result.archive_key)
            .expect("replay archive readback")
            .payload_commit,
        replay.payload.digest()
    );
    assert_eq!(
        archive_store
            .get(sleep_result.archive_key)
            .expect("sleep archive readback")
            .payload_commit,
        sleep.payload.digest()
    );
    assert_eq!(
        archive_store
            .get(geist_result.archive_key)
            .expect("geist archive readback")
            .payload_commit,
        geist.payload.digest()
    );

    let summary = CrossLayerReadbackSummary {
        replay_payload_digest: replay_result.payload_digest,
        replay_readback_digest: replay_result.readback_digest,
        replay_archive_record_digest: replay_result.archive_record_digest,
        replay_archive_key: replay_result.archive_key,
        sleep_payload_digest: sleep_result.payload_digest,
        sleep_readback_digest: sleep_result.readback_digest,
        sleep_archive_record_digest: sleep_result.archive_record_digest,
        sleep_archive_key: sleep_result.archive_key,
        geist_payload_digest: geist_result.payload_digest,
        geist_readback_digest: geist_result.readback_digest,
        geist_archive_record_digest: geist_result.archive_record_digest,
        geist_archive_key: geist_result.archive_key,
        archive_root_commit: archive_store.root_commit(),
        evidence_count: evidence_store.len(),
        replay_archive_count: archive_store
            .iter_kind(MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND, None)
            .count(),
        sleep_archive_count: archive_store
            .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, None)
            .count(),
        geist_archive_count: archive_store
            .iter_kind(MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, None)
            .count(),
    };

    CrossLayerRun {
        replay,
        sleep,
        geist,
        replay_result,
        sleep_result,
        geist_result,
        summary,
    }
}

fn assert_evidence_payload_readback(
    evidence_store: &InMemoryEvidenceStore,
    evidence_id: &ucf_types::EvidenceId,
    expected_payload: Vec<u8>,
) {
    let evidence = evidence_store
        .get(evidence_id.clone())
        .expect("evidence readback");
    assert_eq!(
        evidence.proof.expect("proof envelope").payload,
        expected_payload
    );
}

#[test]
fn cross_layer_append_readback_is_deterministic_across_fresh_runs() {
    let first = run_cross_layer_pipeline();
    let second = run_cross_layer_pipeline();

    assert_eq!(first.summary, second.summary);
    assert_eq!(first.replay_result, second.replay_result);
    assert_eq!(first.sleep_result, second.sleep_result);
    assert_eq!(first.geist_result, second.geist_result);
    assert_eq!(first.summary.evidence_count, 3);
    assert_eq!(first.summary.replay_archive_count, 1);
    assert_eq!(first.summary.sleep_archive_count, 1);
    assert_eq!(first.summary.geist_archive_count, 1);
    assert!(first.summary.archive_root_commit.is_some());
}

#[test]
fn cross_layer_append_readback_preserves_replay_sleep_geist_provenance() {
    let run = run_cross_layer_pipeline();

    assert_eq!(
        run.sleep.payload.replay_audit_digest,
        run.replay.audit.audit_digest
    );
    assert_eq!(
        run.sleep.payload.replay_schedule_digest,
        run.replay.schedule.schedule_digest
    );
    assert_eq!(
        run.sleep.payload.replay_applied_boundary_digest,
        Some(run.replay.boundary.applied_boundary_digest)
    );
    assert_eq!(
        run.sleep.payload.token_count,
        run.replay.schedule.token_count
    );

    assert_eq!(
        run.geist.payload.sleep_plan_candidate_digest,
        run.sleep.candidate.sleep_plan_digest
    );
    assert_eq!(
        run.geist.payload.sleep_plan_audit_digest,
        run.sleep.audit.audit_digest
    );
    assert_eq!(
        run.geist.payload.sleep_applied_boundary_digest,
        Some(run.sleep.boundary.applied_boundary_digest)
    );
    assert_eq!(
        run.geist.payload.replay_audit_digest,
        run.replay.audit.audit_digest
    );
    assert_eq!(
        run.geist.payload.replay_schedule_digest,
        run.replay.schedule.schedule_digest
    );
    assert_eq!(
        run.geist.payload.token_count,
        run.replay.schedule.token_count
    );
    assert_eq!(
        run.geist.payload.token_count as usize,
        run.replay.tokens.len()
    );
}

#[test]
fn cross_layer_archive_records_use_expected_kinds() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();
    let (replay, sleep, geist) = build_cross_layer_artifacts();

    append_minimal_spine_replay_record(
        &replay.schedule,
        &replay.audit,
        &replay.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append replay");
    append_minimal_spine_sleep_record(
        &sleep.candidate,
        &sleep.audit,
        &sleep.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append sleep");
    append_minimal_spine_geist_ism_record(
        &geist.candidate,
        &geist.audit,
        &geist.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append geist/ism");

    assert_eq!(
        archive_store.iter_kind(RecordKind::Other(65), None).count(),
        1
    );
    assert_eq!(
        archive_store.iter_kind(RecordKind::Other(66), None).count(),
        1
    );
    assert_eq!(
        archive_store.iter_kind(RecordKind::Other(67), None).count(),
        1
    );
    assert_eq!(
        archive_store
            .iter_kind(RecordKind::ReplayApplied, None)
            .count(),
        0
    );
    assert_eq!(
        archive_store.iter_kind(RecordKind::IsmAnchor, None).count(),
        0
    );
}

#[test]
fn cross_layer_builders_remain_append_free_until_explicit_helpers() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();
    let (replay, sleep, geist) = build_cross_layer_artifacts();

    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND, None)
            .count(),
        0
    );
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, None)
            .count(),
        0
    );
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, None)
            .count(),
        0
    );

    append_minimal_spine_replay_record(
        &replay.schedule,
        &replay.audit,
        &replay.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append replay");
    assert_eq!(evidence_store.len(), 1);
    append_minimal_spine_sleep_record(
        &sleep.candidate,
        &sleep.audit,
        &sleep.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append sleep");
    assert_eq!(evidence_store.len(), 2);
    append_minimal_spine_geist_ism_record(
        &geist.candidate,
        &geist.audit,
        &geist.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append geist/ism");
    assert_eq!(evidence_store.len(), 3);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND, None)
            .count()
            + archive_store
                .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, None)
                .count()
            + archive_store
                .iter_kind(MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, None)
                .count(),
        3
    );
}

#[test]
fn cross_layer_append_rejects_mismatched_provenance() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();
    let (replay, sleep, geist) = build_cross_layer_artifacts();

    append_minimal_spine_replay_record(
        &replay.schedule,
        &replay.audit,
        &replay.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append replay");

    let mut mismatched_sleep_boundary = sleep.boundary.clone();
    mismatched_sleep_boundary.replay_schedule_digest = digest(240);
    mismatched_sleep_boundary.applied_boundary_digest = mismatched_sleep_boundary.digest();
    assert!(append_minimal_spine_sleep_record(
        &sleep.candidate,
        &sleep.audit,
        &mismatched_sleep_boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .is_err());
    assert_eq!(evidence_store.len(), 1);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND, None)
            .count(),
        0
    );

    append_minimal_spine_sleep_record(
        &sleep.candidate,
        &sleep.audit,
        &sleep.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append sleep");

    let mut mismatched_geist_boundary = geist.boundary.clone();
    mismatched_geist_boundary.token_count = mismatched_geist_boundary.token_count.saturating_add(1);
    mismatched_geist_boundary.ism_candidate_digest = mismatched_geist_boundary.digest();
    assert!(append_minimal_spine_geist_ism_record(
        &geist.candidate,
        &geist.audit,
        &mismatched_geist_boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .is_err());
    assert_eq!(evidence_store.len(), 2);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, None)
            .count(),
        0
    );
}

#[test]
fn cross_layer_append_has_no_runtime_identity_gateway_semantics() {
    let run = run_cross_layer_pipeline();

    assert!(!run.replay.payload.runtime_executed);
    assert!(!run.replay.payload.scheduler_activated);
    assert!(!run.replay.payload.sleep_triggered);
    assert!(!run.replay.payload.geist_ingested);
    assert!(!run.replay.payload.ism_written);
    assert!(!run.replay.payload.identity_anchor);
    assert!(!run.replay.payload.gateway_visible);
    assert_eq!(
        run.replay.payload.evidence_archive_appended_meaning,
        MINIMAL_SPINE_REPLAY_APPEND_MEANING
    );

    assert!(!run.sleep.payload.runtime_executed);
    assert!(!run.sleep.payload.coordinator_triggered);
    assert!(!run.sleep.payload.coordinator_report_written);
    assert!(!run.sleep.payload.coordinator_wal_written);
    assert!(!run.sleep.payload.coordinator_journal_written);
    assert!(!run.sleep.payload.sleep_completed);
    assert!(!run.sleep.payload.geist_ingested);
    assert!(!run.sleep.payload.ism_written);
    assert!(!run.sleep.payload.identity_anchor);
    assert!(!run.sleep.payload.memory_stabilized);
    assert!(!run.sleep.payload.gateway_visible);
    assert_eq!(
        run.sleep.payload.evidence_archive_appended_meaning,
        MINIMAL_SPINE_SLEEP_APPEND_MEANING
    );

    assert!(!run.geist.payload.geist_runtime_applied);
    assert!(!run.geist.payload.ism_written);
    assert!(!run.geist.payload.ism_upserted);
    assert!(!run.geist.payload.identity_anchor);
    assert!(!run.geist.payload.identity_finalized);
    assert!(!run.geist.payload.memory_stabilized);
    assert!(!run.geist.payload.policy_mutated);
    assert!(!run.geist.payload.gateway_visible);
    assert_eq!(
        run.geist.payload.evidence_archive_appended_meaning,
        MINIMAL_SPINE_GEIST_ISM_APPEND_MEANING
    );
}

#[test]
fn cross_layer_append_does_not_create_second_event_log() {
    let run = run_cross_layer_pipeline();

    assert_eq!(run.summary.evidence_count, 3);
    assert_eq!(
        run.summary.replay_archive_count
            + run.summary.sleep_archive_count
            + run.summary.geist_archive_count,
        3
    );
    assert_eq!(run.summary.replay_archive_count, 1);
    assert_eq!(run.summary.sleep_archive_count, 1);
    assert_eq!(run.summary.geist_archive_count, 1);
}
