use ucf_archive_store::{ArchiveAppender, ArchiveStore, InMemoryArchiveStore};
use ucf_evidence::{EvidenceStore, InMemoryEvidenceStore};
use ucf_replay::{
    append_minimal_spine_replay_record, build_replay_applied_boundary_from_audit,
    build_replay_schedule_from_minimal_spine_tokens, build_replay_token_from_minimal_spine_input,
    verify_minimal_spine_replay_schedule, MinimalSpineReplayAppendPayload,
    MinimalSpineReplayAppliedBoundary, MinimalSpineReplayAuditStatus,
    MinimalSpineReplayScheduleAudit, MinimalSpineReplayScheduleBuildOutput,
    MinimalSpineReplayScheduleConfig, MinimalSpineReplayTokenBuildOutput,
    MinimalSpineReplayTokenInput, MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND,
    MINIMAL_SPINE_REPLAY_APPEND_CONTRACT, MINIMAL_SPINE_REPLAY_APPEND_MEANING,
    MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_types::Digest32;

#[derive(Clone, Debug)]
struct ReplayAppendFixture {
    tokens: Vec<MinimalSpineReplayTokenBuildOutput>,
    schedule: MinimalSpineReplayScheduleBuildOutput,
    audit: MinimalSpineReplayScheduleAudit,
    boundary: MinimalSpineReplayAppliedBoundary,
}

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn zero_digest() -> Digest32 {
    Digest32::new([0; Digest32::LEN])
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

fn build_fixture() -> ReplayAppendFixture {
    let tokens: Vec<_> = [replay_input(10), replay_input(40), replay_input(70)]
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
    let boundary = build_replay_applied_boundary_from_audit(&audit).expect("applied boundary");
    ReplayAppendFixture {
        tokens,
        schedule,
        audit,
        boundary,
    }
}

#[test]
fn replay_append_contract_is_explicit_and_readbackable() {
    let fixture = build_fixture();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);

    let result = append_minimal_spine_replay_record(
        &fixture.schedule,
        &fixture.audit,
        &fixture.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append replay provenance");
    let payload = MinimalSpineReplayAppendPayload::from_artifacts(
        &fixture.schedule,
        &fixture.audit,
        &fixture.boundary,
    )
    .expect("payload");
    let evidence_readback = evidence_store
        .get(result.appended_evidence_id.clone())
        .expect("evidence readback");
    let archive_readback = archive_store
        .get(result.archive_key)
        .expect("archive readback");

    assert_eq!(evidence_store.len(), 1);
    assert!(archive_store.root_commit().is_some());
    assert_eq!(result.payload_digest, payload.digest());
    assert_eq!(
        result.replay_schedule_digest,
        fixture.schedule.schedule_digest
    );
    assert_eq!(result.replay_audit_digest, fixture.audit.audit_digest);
    assert_eq!(
        result.replay_applied_boundary_digest,
        fixture.boundary.applied_boundary_digest
    );
    assert_eq!(
        evidence_readback.proof.expect("proof").payload,
        payload.deterministic_bytes()
    );
    assert_eq!(
        archive_readback.kind,
        MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND
    );
    assert_eq!(archive_readback.payload_commit, payload.digest());
    assert_eq!(
        archive_readback.meta.boundary_commit,
        fixture.boundary.applied_boundary_digest
    );
    assert_eq!(archive_readback.meta.cycle_id, 0);
    assert_eq!(archive_readback.meta.tier, 3);
    assert_eq!(archive_readback.meta.flags, 0);
}

#[test]
fn replay_append_preserves_token_schedule_audit_boundary_provenance() {
    let fixture = build_fixture();
    let payload = MinimalSpineReplayAppendPayload::from_artifacts(
        &fixture.schedule,
        &fixture.audit,
        &fixture.boundary,
    )
    .expect("payload");

    assert_eq!(payload.version, 1);
    assert_eq!(
        payload.append_contract,
        MINIMAL_SPINE_REPLAY_APPEND_CONTRACT
    );
    assert_eq!(
        payload.replay_token_digests,
        fixture.schedule.replay_token_digests
    );
    assert_eq!(
        payload.replay_schedule_digest,
        fixture.schedule.schedule_digest
    );
    assert_eq!(payload.replay_audit_digest, fixture.audit.audit_digest);
    assert_eq!(
        payload.replay_applied_boundary_digest,
        fixture.boundary.applied_boundary_digest
    );
    assert_eq!(payload.token_count, fixture.schedule.token_count);
    assert_eq!(payload.token_count as usize, fixture.tokens.len());
    assert_eq!(payload.schedule_source, fixture.schedule.source);
    assert_eq!(payload.audit_source, fixture.audit.source);
    assert_eq!(payload.applied_boundary_source, fixture.boundary.source);
    assert_eq!(
        payload.evidence_archive_appended_meaning,
        MINIMAL_SPINE_REPLAY_APPEND_MEANING
    );
    assert!(payload.validate_links_nonzero());
}

#[test]
fn replay_append_is_deterministic_for_fresh_stores() {
    let fixture = build_fixture();
    let first_evidence_store = InMemoryEvidenceStore::new();
    let first_archive_store = InMemoryArchiveStore::new();
    let mut first_archive_appender = ArchiveAppender::new();
    let second_evidence_store = InMemoryEvidenceStore::new();
    let second_archive_store = InMemoryArchiveStore::new();
    let mut second_archive_appender = ArchiveAppender::new();

    let first = append_minimal_spine_replay_record(
        &fixture.schedule,
        &fixture.audit,
        &fixture.boundary,
        &first_evidence_store,
        &first_archive_store,
        &mut first_archive_appender,
    )
    .expect("first append");
    let second = append_minimal_spine_replay_record(
        &fixture.schedule,
        &fixture.audit,
        &fixture.boundary,
        &second_evidence_store,
        &second_archive_store,
        &mut second_archive_appender,
    )
    .expect("second append");

    assert_eq!(first, second);
}

#[test]
fn replay_builders_remain_append_free() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();

    let fixture = build_fixture();

    assert_eq!(fixture.audit.status, MinimalSpineReplayAuditStatus::Pass);
    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);
}

#[test]
fn replay_append_rejects_invalid_or_mismatched_inputs() {
    let fixture = build_fixture();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    let mut mismatched_boundary = fixture.boundary.clone();
    mismatched_boundary.schedule_digest = digest(200);
    mismatched_boundary.applied_boundary_digest = mismatched_boundary.digest();
    assert!(append_minimal_spine_replay_record(
        &fixture.schedule,
        &fixture.audit,
        &mismatched_boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .is_err());

    let mut fail_schedule = fixture.schedule.clone();
    fail_schedule.schedule_digest = zero_digest();
    let fail_audit = verify_minimal_spine_replay_schedule(&fail_schedule);
    assert_eq!(fail_audit.status, MinimalSpineReplayAuditStatus::Fail);
    assert!(append_minimal_spine_replay_record(
        &fail_schedule,
        &fail_audit,
        &fixture.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .is_err());

    let mut empty_schedule = fixture.schedule.clone();
    empty_schedule.scheduled_tokens.clear();
    empty_schedule.scheduled_token_provenance.clear();
    empty_schedule.replay_token_digests.clear();
    empty_schedule.token_build_output_digests.clear();
    empty_schedule.token_count = 0;
    empty_schedule.schedule_digest = empty_schedule.digest();
    let empty_audit = verify_minimal_spine_replay_schedule(&empty_schedule);
    assert!(append_minimal_spine_replay_record(
        &empty_schedule,
        &empty_audit,
        &fixture.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .is_err());

    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);
}

#[test]
fn replay_append_does_not_trigger_runtime_scheduler_sleep_geist_or_identity() {
    let fixture = build_fixture();
    let payload = MinimalSpineReplayAppendPayload::from_artifacts(
        &fixture.schedule,
        &fixture.audit,
        &fixture.boundary,
    )
    .expect("payload");

    assert!(!payload.runtime_executed);
    assert!(!payload.scheduler_activated);
    assert!(!payload.sleep_triggered);
    assert!(!payload.geist_ingested);
    assert!(!payload.ism_written);
    assert!(!payload.identity_anchor);
    assert!(!payload.gateway_visible);
    assert!(!fixture.schedule.applied);
    assert!(!fixture.audit.replay_completed);
    assert!(!fixture.boundary.geist_ingested);
    assert!(!fixture.boundary.ism_written);
    assert!(!fixture.boundary.identity_anchor);
    assert!(!fixture.boundary.gateway_visible);
}

#[test]
fn replay_append_does_not_create_second_event_log() {
    let fixture = build_fixture();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    append_minimal_spine_replay_record(
        &fixture.schedule,
        &fixture.audit,
        &fixture.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("append replay provenance");

    assert_eq!(evidence_store.len(), 1);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND, None)
            .count(),
        1
    );
    assert_eq!(
        archive_store
            .iter_kind(ucf_archive_store::RecordKind::ReplayApplied, None)
            .count(),
        0
    );
    assert_eq!(
        archive_store
            .iter_kind(ucf_archive_store::RecordKind::ReplayToken, None)
            .count(),
        0
    );
}
