#![forbid(unsafe_code)]

use ucf_archive_store::{ArchiveAppender, ArchiveStore, InMemoryArchiveStore};
use ucf_consolidation::{
    append_minimal_spine_meso_milestone,
    build_meso_milestone_from_minimal_spine_micro_build_outputs,
    build_micro_milestone_from_minimal_spine_candidate, ConsolidationError,
    MinimalSpineMesoMilestoneAppendPayload, MinimalSpineMicroMilestoneAppendPayload,
    MinimalSpineMicroMilestoneBuildOutput, MinimalSpineMicroMilestoneCandidate,
    MINIMAL_SPINE_CONSOLIDATION_SOURCE, MINIMAL_SPINE_MESO_MILESTONE_APPEND_CONTRACT,
    MINIMAL_SPINE_MESO_MILESTONE_ARCHIVE_KIND,
};
use ucf_evidence::{EvidenceStore, InMemoryEvidenceStore};
use ucf_types::{Digest32, EvidenceId};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn candidate_fixture(
    sequence: u64,
    evidence_suffix: &str,
    output_record_byte: u8,
) -> MinimalSpineMicroMilestoneCandidate {
    MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        sequence,
        EvidenceId::new(format!(
            "minimal-spine-evidence-meso-append-{evidence_suffix}"
        )),
        digest(1),
        digest(2),
        digest(output_record_byte),
        digest(4),
        digest(5),
        "allow",
        "materialized-test-output",
    )
}

fn build_output_fixture(
    sequence: u64,
    evidence_suffix: &str,
    output_record_byte: u8,
) -> MinimalSpineMicroMilestoneBuildOutput {
    build_micro_milestone_from_minimal_spine_candidate(&candidate_fixture(
        sequence,
        evidence_suffix,
        output_record_byte,
    ))
    .expect("valid micro build output")
}

fn micro_outputs_fixture() -> Vec<MinimalSpineMicroMilestoneBuildOutput> {
    vec![
        build_output_fixture(7, "a", 3),
        build_output_fixture(8, "b", 6),
        build_output_fixture(9, "c", 9),
    ]
}

fn micro_payloads_fixture() -> Vec<MinimalSpineMicroMilestoneAppendPayload> {
    micro_outputs_fixture()
        .iter()
        .map(MinimalSpineMicroMilestoneAppendPayload::from_build_output)
        .collect::<Result<Vec<_>, _>>()
        .expect("valid micro append payloads")
}

#[test]
fn meso_append_contract_is_explicit_and_readbackable() {
    let micro_outputs = micro_outputs_fixture();
    let build_output =
        build_meso_milestone_from_minimal_spine_micro_build_outputs(&micro_outputs).unwrap();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    let result = append_minimal_spine_meso_milestone(
        &build_output,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .unwrap();

    let payload = MinimalSpineMesoMilestoneAppendPayload::from_build_output(&build_output).unwrap();
    let readback = evidence_store
        .get(result.appended_evidence_id.clone())
        .expect("explicit meso evidence append is readable");
    let proof = readback.proof.expect("append stores meso proof payload");
    let archive_readback = archive_store
        .get(result.archive_key)
        .expect("explicit meso archive append is readable");

    assert_eq!(
        payload.append_contract,
        MINIMAL_SPINE_MESO_MILESTONE_APPEND_CONTRACT
    );
    assert_eq!(result.payload_digest, payload.digest());
    assert_eq!(result.build_output_digest, build_output.digest());
    assert_eq!(
        result.meso_milestone_digest,
        build_output.meso_milestone_digest
    );
    assert_eq!(result.aggregation_digest, build_output.aggregation_digest);
    assert_eq!(
        result.micro_payload_digests,
        build_output.micro_payload_digests
    );
    assert_eq!(
        result.micro_milestone_digests,
        build_output.micro_milestone_digests
    );
    assert_eq!(proof.payload, payload.deterministic_bytes());
    assert_eq!(
        archive_readback.kind,
        MINIMAL_SPINE_MESO_MILESTONE_ARCHIVE_KIND
    );
    assert_eq!(archive_readback.payload_commit, payload.digest());
    assert_eq!(
        archive_readback.meta.boundary_commit,
        build_output.meso_milestone_digest
    );
    assert_eq!(archive_readback.meta.tier, 2);
}

#[test]
fn meso_append_preserves_micro_and_meso_provenance() {
    let micro_payloads = micro_payloads_fixture();
    let micro_outputs = micro_outputs_fixture();
    let build_output =
        build_meso_milestone_from_minimal_spine_micro_build_outputs(&micro_outputs).unwrap();
    let payload = MinimalSpineMesoMilestoneAppendPayload::from_build_output(&build_output).unwrap();
    let mut expected_micro_payload_digests: Vec<_> = micro_payloads
        .iter()
        .map(|payload| payload.digest())
        .collect();
    expected_micro_payload_digests.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));

    assert_eq!(payload.build_output_digest, build_output.digest());
    assert_eq!(
        payload.meso_milestone_digest,
        build_output.meso_milestone_digest
    );
    assert_eq!(payload.aggregation_digest, build_output.aggregation_digest);
    assert_eq!(payload.micro_count, build_output.micro_count);
    assert_eq!(payload.micro_count, 3);
    assert_eq!(
        payload.micro_payload_digests,
        expected_micro_payload_digests
    );
    assert_eq!(
        payload.micro_milestone_digests,
        build_output.micro_milestone_digests
    );
    assert_eq!(payload.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
}

#[test]
fn meso_append_is_deterministic_for_fresh_stores() {
    let build_output =
        build_meso_milestone_from_minimal_spine_micro_build_outputs(&micro_outputs_fixture())
            .unwrap();
    let first_evidence_store = InMemoryEvidenceStore::new();
    let first_archive_store = InMemoryArchiveStore::new();
    let mut first_archive_appender = ArchiveAppender::new();
    let second_evidence_store = InMemoryEvidenceStore::new();
    let second_archive_store = InMemoryArchiveStore::new();
    let mut second_archive_appender = ArchiveAppender::new();

    let first = append_minimal_spine_meso_milestone(
        &build_output,
        &first_evidence_store,
        &first_archive_store,
        &mut first_archive_appender,
    )
    .unwrap();
    let second = append_minimal_spine_meso_milestone(
        &build_output,
        &second_evidence_store,
        &second_archive_store,
        &mut second_archive_appender,
    )
    .unwrap();

    assert_eq!(first, second);
}

#[test]
fn meso_builder_remains_append_free() {
    let micro_outputs = micro_outputs_fixture();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();

    let _build_output =
        build_meso_milestone_from_minimal_spine_micro_build_outputs(&micro_outputs).unwrap();

    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);
}

#[test]
fn meso_append_does_not_trigger_replay_geist_or_macro() {
    let build_output =
        build_meso_milestone_from_minimal_spine_micro_build_outputs(&micro_outputs_fixture())
            .unwrap();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    let result = append_minimal_spine_meso_milestone(
        &build_output,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .unwrap();
    let payload = MinimalSpineMesoMilestoneAppendPayload::from_build_output(&build_output).unwrap();
    let payload_bytes = payload.deterministic_bytes();
    let forbidden = [
        b"replay".as_slice(),
        b"Geist".as_slice(),
        b"ISM".as_slice(),
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
            .iter_kind(MINIMAL_SPINE_MESO_MILESTONE_ARCHIVE_KIND, None)
            .count(),
        1
    );
    assert_eq!(
        result.meso_milestone_digest,
        build_output.meso_milestone_digest
    );
}

#[test]
fn meso_append_rejects_invalid_build_output() {
    let mut build_output =
        build_meso_milestone_from_minimal_spine_micro_build_outputs(&micro_outputs_fixture())
            .unwrap();
    build_output.aggregation_digest = Digest32::new([0; Digest32::LEN]);
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    let error = append_minimal_spine_meso_milestone(
        &build_output,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .unwrap_err();

    assert_eq!(
        error,
        ConsolidationError::InvalidMinimalSpineMesoMilestoneAppendPayloadLinks
    );
    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);
}

#[test]
fn meso_append_does_not_mutate_micro_payloads() {
    let micro_outputs = micro_outputs_fixture();
    let micro_payloads = micro_payloads_fixture();
    let original_micro_payloads = micro_payloads.clone();
    let build_output =
        build_meso_milestone_from_minimal_spine_micro_build_outputs(&micro_outputs).unwrap();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    append_minimal_spine_meso_milestone(
        &build_output,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .unwrap();

    assert_eq!(micro_payloads, original_micro_payloads);
}
