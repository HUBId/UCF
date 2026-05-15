#![forbid(unsafe_code)]

use prost::Message;
use ucf_consolidation::{
    build_micro_milestone_from_minimal_spine_candidate, ConsolidationError,
    MinimalSpineMicroMilestoneBuildOutput, MinimalSpineMicroMilestoneCandidate,
    MINIMAL_SPINE_CONSOLIDATION_SOURCE, MINIMAL_SPINE_MICRO_MILESTONE_BUILD_OUTPUT_VERSION,
};
use ucf_types::v1::spec::MicroMilestone;
use ucf_types::{Digest32, EvidenceId};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn candidate_fixture() -> MinimalSpineMicroMilestoneCandidate {
    MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        7,
        EvidenceId::new("minimal-spine-evidence-builder"),
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
    build_micro_milestone_from_minimal_spine_candidate(&candidate_fixture()).expect("valid build")
}

#[test]
fn micro_builder_from_minimal_spine_candidate_is_deterministic() {
    let first = build_fixture();
    let second = build_fixture();

    assert_eq!(first, second);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.digest(), second.digest());
    assert_eq!(first.micro_milestone_digest, second.micro_milestone_digest);
}

#[test]
fn micro_builder_changes_when_candidate_link_changes() {
    let first = build_fixture();
    let changed_output_record = MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        7,
        EvidenceId::new("minimal-spine-evidence-builder"),
        digest(1),
        digest(2),
        digest(9),
        digest(4),
        digest(5),
        "allow",
        "materialized-test-output",
    );
    let changed_archive_key = MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        7,
        EvidenceId::new("minimal-spine-evidence-builder"),
        digest(1),
        digest(2),
        digest(3),
        digest(8),
        digest(5),
        "allow",
        "materialized-test-output",
    );

    let changed_output =
        build_micro_milestone_from_minimal_spine_candidate(&changed_output_record).unwrap();
    let changed_archive =
        build_micro_milestone_from_minimal_spine_candidate(&changed_archive_key).unwrap();

    assert_ne!(first.digest(), changed_output.digest());
    assert_ne!(first.digest(), changed_archive.digest());
    assert_ne!(
        first.deterministic_bytes(),
        changed_output.deterministic_bytes()
    );
}

#[test]
fn micro_builder_outputs_protocol_micro_milestone() {
    let output = build_fixture();

    assert_eq!(
        output.version,
        MINIMAL_SPINE_MICRO_MILESTONE_BUILD_OUTPUT_VERSION
    );
    assert!(output
        .micro_milestone
        .milestone_id
        .starts_with("minimal-spine-micro-"));
    assert_eq!(output.micro_milestone.achieved_at_ms, 7);
    assert_eq!(
        output.micro_milestone.label,
        "minimal-spine-v1 micro candidate allow materialized-test-output"
    );

    let prost_bytes = output.micro_milestone.encode_to_vec();
    let decoded = MicroMilestone::decode(prost_bytes.as_slice()).expect("prost roundtrip");
    assert_eq!(decoded, output.micro_milestone);
}

#[test]
fn micro_builder_preserves_minimal_spine_provenance() {
    let candidate = candidate_fixture();
    let output = build_micro_milestone_from_minimal_spine_candidate(&candidate).unwrap();

    assert_eq!(output.candidate_digest, candidate.digest());
    assert_eq!(output.input_digest, digest(1));
    assert_eq!(output.candidate_set_record_digest, digest(2));
    assert_eq!(output.output_record_digest, digest(3));
    assert_eq!(
        output.evidence_id,
        EvidenceId::new("minimal-spine-evidence-builder")
    );
    assert_eq!(output.archive_output_key, digest(4));
    assert_eq!(output.archive_output_event_digest, digest(5));
    assert_eq!(output.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
}

#[test]
fn micro_builder_rejects_invalid_candidate_links() {
    let candidate = MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        7,
        EvidenceId::new("minimal-spine-evidence-builder"),
        Digest32::new([0; Digest32::LEN]),
        digest(2),
        digest(3),
        digest(4),
        digest(5),
        "allow",
        "materialized-test-output",
    );

    assert_eq!(
        build_micro_milestone_from_minimal_spine_candidate(&candidate).unwrap_err(),
        ConsolidationError::InvalidMinimalSpineMicroMilestoneCandidateLinks
    );
}

#[test]
fn micro_builder_has_no_archive_replay_geist_side_effects() {
    let output = build_fixture();
    let bytes = output.deterministic_bytes();

    assert_eq!(output.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
    assert!(!bytes
        .windows(b"ArchiveMilestoneSink".len())
        .any(|window| window == b"ArchiveMilestoneSink"));
    assert!(!bytes
        .windows(b"replay".len())
        .any(|window| window == b"replay"));
    assert!(!bytes
        .windows(b"Geist".len())
        .any(|window| window == b"Geist"));
    assert!(!bytes.windows(b"ISM".len()).any(|window| window == b"ISM"));
}

#[test]
fn micro_builder_does_not_build_meso_or_macro() {
    let output = build_fixture();
    let bytes = output.deterministic_bytes();

    assert!(!bytes.windows(b"meso".len()).any(|window| window == b"meso"));
    assert!(!bytes
        .windows(b"macro".len())
        .any(|window| window == b"macro"));
    assert!(!bytes
        .windows(b"finalized".len())
        .any(|window| window == b"finalized"));
}
