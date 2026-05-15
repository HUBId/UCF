#![forbid(unsafe_code)]

use ucf_archive_store::{ArchiveAppender, ArchiveStore, InMemoryArchiveStore};
use ucf_consolidation::{
    append_minimal_spine_micro_milestone, build_micro_milestone_from_minimal_spine_candidate,
    ConsolidationError, MinimalSpineMicroMilestoneAppendPayload,
    MinimalSpineMicroMilestoneBuildOutput, MinimalSpineMicroMilestoneCandidate,
    MINIMAL_SPINE_CONSOLIDATION_SOURCE, MINIMAL_SPINE_MICRO_MILESTONE_APPEND_CONTRACT,
    MINIMAL_SPINE_MICRO_MILESTONE_ARCHIVE_KIND,
};
use ucf_evidence::{EvidenceStore, InMemoryEvidenceStore};
use ucf_types::{Digest32, EvidenceId};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn candidate_fixture() -> MinimalSpineMicroMilestoneCandidate {
    MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        7,
        EvidenceId::new("minimal-spine-evidence-append"),
        digest(1),
        digest(2),
        digest(3),
        digest(4),
        digest(5),
        "allow",
        "materialized-test-output",
    )
}

fn build_fixture() -> MinimalSpineMicroMilestoneBuildOutput {
    build_micro_milestone_from_minimal_spine_candidate(&candidate_fixture()).unwrap()
}

#[test]
fn micro_append_contract_is_explicit_and_readbackable() {
    let build_output = build_fixture();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    let result = append_minimal_spine_micro_milestone(
        &build_output,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .unwrap();

    let payload =
        MinimalSpineMicroMilestoneAppendPayload::from_build_output(&build_output).unwrap();
    let readback = evidence_store
        .get(result.appended_evidence_id.clone())
        .expect("explicit evidence append is readable");
    let proof = readback.proof.expect("append stores proof payload");
    let archive_readback = archive_store
        .get(result.archive_key)
        .expect("explicit archive append is readable");

    assert_eq!(
        payload.append_contract,
        MINIMAL_SPINE_MICRO_MILESTONE_APPEND_CONTRACT
    );
    assert_eq!(result.payload_digest, payload.digest());
    assert_eq!(result.build_output_digest, build_output.digest());
    assert_eq!(
        result.micro_milestone_digest,
        build_output.micro_milestone_digest
    );
    assert_eq!(proof.payload, payload.deterministic_bytes());
    assert_eq!(
        archive_readback.kind,
        MINIMAL_SPINE_MICRO_MILESTONE_ARCHIVE_KIND
    );
    assert_eq!(archive_readback.payload_commit, payload.digest());
    assert_eq!(
        archive_readback.meta.boundary_commit,
        build_output.micro_milestone_digest
    );
}

#[test]
fn micro_append_preserves_builder_provenance() {
    let build_output = build_fixture();
    let payload =
        MinimalSpineMicroMilestoneAppendPayload::from_build_output(&build_output).unwrap();

    assert_eq!(payload.build_output_digest, build_output.digest());
    assert_eq!(payload.candidate_digest, build_output.candidate_digest);
    assert_eq!(
        payload.micro_milestone_digest,
        build_output.micro_milestone_digest
    );
    assert_eq!(payload.input_digest, build_output.input_digest);
    assert_eq!(
        payload.candidate_set_record_digest,
        build_output.candidate_set_record_digest
    );
    assert_eq!(
        payload.output_record_digest,
        build_output.output_record_digest
    );
    assert_eq!(payload.source_evidence_id, build_output.evidence_id);
    assert_eq!(payload.archive_output_key, build_output.archive_output_key);
    assert_eq!(
        payload.archive_output_event_digest,
        build_output.archive_output_event_digest
    );
    assert_eq!(payload.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
}

#[test]
fn micro_append_is_deterministic_for_fresh_stores() {
    let build_output = build_fixture();
    let first_evidence_store = InMemoryEvidenceStore::new();
    let first_archive_store = InMemoryArchiveStore::new();
    let mut first_archive_appender = ArchiveAppender::new();
    let second_evidence_store = InMemoryEvidenceStore::new();
    let second_archive_store = InMemoryArchiveStore::new();
    let mut second_archive_appender = ArchiveAppender::new();

    let first = append_minimal_spine_micro_milestone(
        &build_output,
        &first_evidence_store,
        &first_archive_store,
        &mut first_archive_appender,
    )
    .unwrap();
    let second = append_minimal_spine_micro_milestone(
        &build_output,
        &second_evidence_store,
        &second_archive_store,
        &mut second_archive_appender,
    )
    .unwrap();

    assert_eq!(first, second);
}

#[test]
fn micro_builder_remains_append_free() {
    let candidate = candidate_fixture();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();

    let _build_output = build_micro_milestone_from_minimal_spine_candidate(&candidate).unwrap();

    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);
}

#[test]
fn micro_append_does_not_trigger_replay_geist_or_macro() {
    let build_output = build_fixture();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    let result = append_minimal_spine_micro_milestone(
        &build_output,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .unwrap();
    let payload =
        MinimalSpineMicroMilestoneAppendPayload::from_build_output(&build_output).unwrap();
    let payload_bytes = payload.deterministic_bytes();
    let forbidden = [
        b"replay".as_slice(),
        b"Geist".as_slice(),
        b"ISM".as_slice(),
        b"meso".as_slice(),
        b"macro".as_slice(),
        b"finalized".as_slice(),
    ];

    for needle in forbidden {
        assert!(!payload_bytes
            .windows(needle.len())
            .any(|window| window == needle));
    }
    assert_eq!(evidence_store.len(), 1);
    assert_eq!(
        archive_store
            .iter_kind(MINIMAL_SPINE_MICRO_MILESTONE_ARCHIVE_KIND, None)
            .count(),
        1
    );
    assert_eq!(
        result.micro_milestone_digest,
        build_output.micro_milestone_digest
    );
}

#[test]
fn micro_append_rejects_invalid_build_output() {
    let mut build_output = build_fixture();
    build_output.input_digest = Digest32::new([0; Digest32::LEN]);
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    let error = append_minimal_spine_micro_milestone(
        &build_output,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .unwrap_err();

    assert_eq!(
        error,
        ConsolidationError::InvalidMinimalSpineMicroMilestoneAppendPayloadLinks
    );
    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);
}
