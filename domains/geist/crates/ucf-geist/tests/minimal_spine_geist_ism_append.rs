use ucf_archive_store::{ArchiveAppender, ArchiveStore, InMemoryArchiveStore};
use ucf_evidence::{EvidenceStore, InMemoryEvidenceStore};
use ucf_geist::{
    append_minimal_spine_geist_ism_record, build_geist_projection_candidate_from_sleep_audit,
    build_ism_candidate_boundary_from_geist_audit, verify_minimal_spine_geist_projection_candidate,
    MinimalSpineGeistIsmAppendPayload, MinimalSpineGeistProjectionAudit,
    MinimalSpineGeistProjectionAuditStatus, MinimalSpineGeistProjectionCandidate,
    MinimalSpineIsmCandidateBoundary, MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND,
    MINIMAL_SPINE_GEIST_ISM_APPEND_CONTRACT, MINIMAL_SPINE_GEIST_ISM_APPEND_MEANING,
};
use ucf_sleep_coordinator::{
    build_sleep_applied_boundary_from_audit, build_sleep_plan_candidate_from_replay_boundary,
    verify_minimal_spine_sleep_plan_candidate, MinimalSpineSleepAppliedBoundary,
    MinimalSpineSleepPlanAudit, MinimalSpineSleepPlanAuditStatus, MinimalSpineSleepPlanInput,
};
use ucf_types::Digest32;

const REPLAY_SOURCE: &str = "minimal_spine_replay_audit_fixture_for_geist_ism_append";

#[derive(Clone, Debug)]
struct GeistIsmAppendRun {
    sleep_audit: MinimalSpineSleepPlanAudit,
    sleep_boundary: MinimalSpineSleepAppliedBoundary,
    candidate: MinimalSpineGeistProjectionCandidate,
    audit: MinimalSpineGeistProjectionAudit,
    boundary: MinimalSpineIsmCandidateBoundary,
}

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn zero_digest() -> Digest32 {
    Digest32::new([0; Digest32::LEN])
}

fn run_pipeline() -> GeistIsmAppendRun {
    let sleep_input = MinimalSpineSleepPlanInput {
        replay_audit_digest: digest(81),
        replay_schedule_digest: digest(82),
        replay_applied_boundary_digest: Some(digest(83)),
        token_count: 19,
        source: REPLAY_SOURCE,
    };
    let sleep_candidate = build_sleep_plan_candidate_from_replay_boundary(&sleep_input)
        .expect("valid sleep candidate fixture");
    let sleep_audit = verify_minimal_spine_sleep_plan_candidate(&sleep_candidate);
    assert_eq!(sleep_audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    let sleep_boundary =
        build_sleep_applied_boundary_from_audit(&sleep_audit).expect("valid sleep boundary");
    let candidate =
        build_geist_projection_candidate_from_sleep_audit(&sleep_audit, Some(&sleep_boundary))
            .expect("valid Geist projection candidate");
    let audit = verify_minimal_spine_geist_projection_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineGeistProjectionAuditStatus::Pass);
    let boundary = build_ism_candidate_boundary_from_geist_audit(&audit)
        .expect("valid local ISM candidate boundary");

    GeistIsmAppendRun {
        sleep_audit,
        sleep_boundary,
        candidate,
        audit,
        boundary,
    }
}

#[test]
fn geist_ism_append_contract_is_explicit_and_readbackable() {
    let run = run_pipeline();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    assert!(evidence_store.is_empty());
    assert!(archive_store.root_commit().is_none());

    let result = append_minimal_spine_geist_ism_record(
        &run.candidate,
        &run.audit,
        &run.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("explicit Geist/ISM append succeeds");
    let payload = MinimalSpineGeistIsmAppendPayload::from_artifacts(
        &run.candidate,
        &run.audit,
        &run.boundary,
    )
    .expect("payload");

    assert_eq!(result.payload_digest, payload.digest());
    assert_eq!(evidence_store.len(), 1);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, None)
            .count(),
        1
    );

    let evidence = evidence_store
        .get(result.appended_evidence_id.clone())
        .expect("evidence readback exists");
    let proof = evidence.proof.expect("proof envelope exists");
    assert_eq!(
        proof.envelope_id,
        format!(
            "{}:{}",
            MINIMAL_SPINE_GEIST_ISM_APPEND_CONTRACT,
            hex(result.payload_digest)
        )
    );
    assert_eq!(proof.payload, payload.deterministic_bytes());
    assert_eq!(
        proof
            .payload_digest
            .expect("payload digest")
            .value_32
            .expect("value_32"),
        result.payload_digest.as_bytes().to_vec()
    );

    let archive_record = archive_store
        .get(result.archive_key)
        .expect("archive readback exists");
    assert_eq!(
        archive_record.kind,
        MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND
    );
    assert_eq!(archive_record.payload_commit, result.payload_digest);
    assert_eq!(
        archive_record.meta.boundary_commit,
        run.boundary.ism_candidate_digest
    );
    assert_eq!(archive_record.meta.flags, 0);
    assert_eq!(archive_record.meta.tier, 3);
}

#[test]
fn geist_ism_append_preserves_candidate_audit_boundary_sleep_replay_provenance() {
    let run = run_pipeline();
    let payload = MinimalSpineGeistIsmAppendPayload::from_artifacts(
        &run.candidate,
        &run.audit,
        &run.boundary,
    )
    .expect("payload");

    assert_eq!(
        payload.geist_projection_candidate_digest,
        run.candidate.projection_digest
    );
    assert_eq!(
        payload.geist_projection_audit_digest,
        run.audit.audit_digest
    );
    assert_eq!(
        payload.ism_candidate_boundary_digest,
        run.boundary.ism_candidate_digest
    );
    assert_eq!(
        payload.sleep_plan_audit_digest,
        run.sleep_audit.audit_digest
    );
    assert_eq!(
        payload.sleep_plan_candidate_digest,
        run.sleep_audit.sleep_plan_candidate_digest
    );
    assert_eq!(
        payload.sleep_applied_boundary_digest,
        Some(run.sleep_boundary.applied_boundary_digest)
    );
    assert_eq!(
        payload.replay_audit_digest,
        run.sleep_audit.replay_audit_digest
    );
    assert_eq!(
        payload.replay_schedule_digest,
        run.sleep_audit.replay_schedule_digest
    );
    assert_eq!(payload.token_count, run.sleep_audit.token_count);
    assert_eq!(payload.candidate_source, run.candidate.source);
    assert_eq!(payload.audit_source, run.audit.source);
    assert_eq!(payload.boundary_source, run.boundary.source);
    assert_eq!(payload.sleep_source, run.candidate.sleep_source);
    assert_eq!(payload.source, MINIMAL_SPINE_GEIST_ISM_APPEND_CONTRACT);
    assert_eq!(
        payload.evidence_archive_appended_meaning,
        MINIMAL_SPINE_GEIST_ISM_APPEND_MEANING
    );
}

#[test]
fn geist_ism_append_is_deterministic_for_fresh_stores() {
    let run = run_pipeline();
    let first_evidence = InMemoryEvidenceStore::new();
    let first_archive = InMemoryArchiveStore::new();
    let mut first_appender = ArchiveAppender::new();
    let second_evidence = InMemoryEvidenceStore::new();
    let second_archive = InMemoryArchiveStore::new();
    let mut second_appender = ArchiveAppender::new();

    let first = append_minimal_spine_geist_ism_record(
        &run.candidate,
        &run.audit,
        &run.boundary,
        &first_evidence,
        &first_archive,
        &mut first_appender,
    )
    .expect("first append");
    let second = append_minimal_spine_geist_ism_record(
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
fn geist_ism_builders_remain_append_free() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let run = run_pipeline();

    assert_eq!(run.candidate.projection_digest, run.candidate.digest());
    assert_eq!(run.audit.audit_digest, run.audit.digest());
    assert_eq!(run.boundary.ism_candidate_digest, run.boundary.digest());
    assert!(evidence_store.is_empty());
    assert!(archive_store.root_commit().is_none());
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, None)
            .count(),
        0
    );
}

#[test]
fn geist_ism_append_rejects_invalid_or_mismatched_inputs() {
    let run = run_pipeline();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();

    let mut mismatched_boundary = run.boundary.clone();
    mismatched_boundary.token_count = mismatched_boundary.token_count.saturating_add(1);
    mismatched_boundary.ism_candidate_digest = mismatched_boundary.digest();
    assert!(append_minimal_spine_geist_ism_record(
        &run.candidate,
        &run.audit,
        &mismatched_boundary,
        &evidence_store,
        &archive_store,
        &mut ArchiveAppender::new(),
    )
    .is_err());

    let mut failed_candidate = run.candidate.clone();
    failed_candidate.geist_applied = true;
    failed_candidate.projection_digest = failed_candidate.digest();
    let failed_audit = verify_minimal_spine_geist_projection_candidate(&failed_candidate);
    assert_eq!(
        failed_audit.status,
        MinimalSpineGeistProjectionAuditStatus::Fail
    );
    assert!(append_minimal_spine_geist_ism_record(
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
    zero_candidate.projection_digest = zero_candidate.digest();
    let zero_audit = verify_minimal_spine_geist_projection_candidate(&zero_candidate);
    assert!(append_minimal_spine_geist_ism_record(
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
    zero_token_candidate.projection_digest = zero_token_candidate.digest();
    let zero_token_audit = verify_minimal_spine_geist_projection_candidate(&zero_token_candidate);
    assert!(append_minimal_spine_geist_ism_record(
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
fn geist_ism_append_does_not_trigger_runtime_ism_identity_policy_or_gateway() {
    let run = run_pipeline();
    let payload = MinimalSpineGeistIsmAppendPayload::from_artifacts(
        &run.candidate,
        &run.audit,
        &run.boundary,
    )
    .expect("payload");

    assert!(!payload.geist_runtime_applied);
    assert!(!payload.ism_written);
    assert!(!payload.ism_upserted);
    assert!(!payload.identity_anchor);
    assert!(!payload.identity_finalized);
    assert!(!payload.memory_stabilized);
    assert!(!payload.policy_mutated);
    assert!(!payload.gateway_visible);
    assert_eq!(
        payload.evidence_archive_appended_meaning,
        MINIMAL_SPINE_GEIST_ISM_APPEND_MEANING
    );
}

#[test]
fn geist_ism_append_does_not_create_second_event_log() {
    let run = run_pipeline();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    append_minimal_spine_geist_ism_record(
        &run.candidate,
        &run.audit,
        &run.boundary,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("Geist/ISM append");

    assert_eq!(evidence_store.len(), 1);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, None)
            .count(),
        1
    );
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND, Some(1))
            .count(),
        1
    );
}

fn hex(digest: Digest32) -> String {
    digest
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
