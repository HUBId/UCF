#![forbid(unsafe_code)]

use prost::Message;
use ucf_consolidation::{
    build_meso_milestone_from_minimal_spine_micro_build_outputs,
    build_meso_milestone_from_minimal_spine_micro_payloads,
    build_micro_milestone_from_minimal_spine_candidate, ConsolidationError,
    MinimalSpineMicroMilestoneAppendPayload, MinimalSpineMicroMilestoneBuildOutput,
    MinimalSpineMicroMilestoneCandidate, MINIMAL_SPINE_CONSOLIDATION_SOURCE,
    MINIMAL_SPINE_MESO_MILESTONE_BUILD_OUTPUT_VERSION,
};
use ucf_types::v1::spec::MesoMilestone;
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
        EvidenceId::new(format!("minimal-spine-evidence-meso-{evidence_suffix}")),
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

fn payloads_fixture() -> Vec<MinimalSpineMicroMilestoneAppendPayload> {
    micro_outputs_fixture()
        .iter()
        .map(MinimalSpineMicroMilestoneAppendPayload::from_build_output)
        .collect::<Result<Vec<_>, _>>()
        .expect("valid payloads")
}

#[test]
fn meso_builder_from_micro_payloads_is_deterministic() {
    let payloads = payloads_fixture();

    let first = build_meso_milestone_from_minimal_spine_micro_payloads(&payloads).unwrap();
    let second = build_meso_milestone_from_minimal_spine_micro_payloads(&payloads).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.digest(), second.digest());
    assert_eq!(first.meso_milestone_digest, second.meso_milestone_digest);
}

#[test]
fn meso_builder_normalizes_input_order() {
    let payloads = payloads_fixture();
    let mut reversed = payloads.clone();
    reversed.reverse();

    let sorted_output = build_meso_milestone_from_minimal_spine_micro_payloads(&payloads).unwrap();
    let reversed_output =
        build_meso_milestone_from_minimal_spine_micro_payloads(&reversed).unwrap();

    assert_eq!(sorted_output, reversed_output);
    assert_eq!(sorted_output.digest(), reversed_output.digest());
}

#[test]
fn meso_builder_changes_when_micro_payload_changes() {
    let payloads = payloads_fixture();
    let mut changed = payloads.clone();
    changed[1] = MinimalSpineMicroMilestoneAppendPayload::from_build_output(&build_output_fixture(
        8, "b", 8,
    ))
    .unwrap();

    let original_output =
        build_meso_milestone_from_minimal_spine_micro_payloads(&payloads).unwrap();
    let changed_output = build_meso_milestone_from_minimal_spine_micro_payloads(&changed).unwrap();

    assert_ne!(
        original_output.aggregation_digest,
        changed_output.aggregation_digest
    );
    assert_ne!(
        original_output.meso_milestone_digest,
        changed_output.meso_milestone_digest
    );
    assert_ne!(original_output.digest(), changed_output.digest());
}

#[test]
fn meso_builder_outputs_protocol_meso_milestone() {
    let outputs = micro_outputs_fixture();
    let meso_output =
        build_meso_milestone_from_minimal_spine_micro_build_outputs(&outputs).unwrap();
    let canonical = ucf::canonical_bytes(&meso_output.meso_milestone);
    let prost_bytes = meso_output.meso_milestone.encode_to_vec();
    let decoded = MesoMilestone::decode(prost_bytes.as_slice()).expect("prost meso roundtrip");

    assert_eq!(
        meso_output.version,
        MINIMAL_SPINE_MESO_MILESTONE_BUILD_OUTPUT_VERSION
    );
    assert_eq!(decoded, meso_output.meso_milestone);
    assert!(!canonical.is_empty());
    assert!(meso_output
        .meso_milestone
        .milestone_id
        .starts_with("minimal-spine-meso-"));
    assert_eq!(meso_output.meso_milestone.achieved_at_ms, 9);
    assert_eq!(
        meso_output.meso_milestone.micro_milestone_ids.len(),
        outputs.len()
    );
}

#[test]
fn meso_builder_preserves_micro_provenance() {
    let payloads = payloads_fixture();
    let meso_output = build_meso_milestone_from_minimal_spine_micro_payloads(&payloads).unwrap();
    let mut expected_payload_digests: Vec<_> =
        payloads.iter().map(|payload| payload.digest()).collect();
    expected_payload_digests.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    let mut expected_micro_digests: Vec<_> = payloads
        .iter()
        .map(|payload| (payload.digest(), payload.micro_milestone_digest))
        .collect();
    expected_micro_digests.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let expected_micro_digests: Vec<_> = expected_micro_digests
        .into_iter()
        .map(|(_, micro_digest)| micro_digest)
        .collect();

    assert_eq!(meso_output.micro_payload_digests, expected_payload_digests);
    assert_eq!(meso_output.micro_milestone_digests, expected_micro_digests);
    assert_eq!(meso_output.micro_count, 3);
    assert_ne!(
        meso_output.aggregation_digest,
        Digest32::new([0; Digest32::LEN])
    );
    assert_eq!(meso_output.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
}

#[test]
fn meso_builder_rejects_empty_or_duplicate_inputs() {
    let empty_error = build_meso_milestone_from_minimal_spine_micro_payloads(&[]).unwrap_err();
    assert_eq!(
        empty_error,
        ConsolidationError::MinimalSpineMesoMilestoneEmptyInput
    );

    let mut payloads = payloads_fixture();
    payloads.push(payloads[0].clone());
    let duplicate_error =
        build_meso_milestone_from_minimal_spine_micro_payloads(&payloads).unwrap_err();
    assert_eq!(
        duplicate_error,
        ConsolidationError::DuplicateMinimalSpineMesoMilestoneInput
    );
}

#[test]
fn meso_builder_has_no_archive_replay_geist_side_effects() {
    let payloads = payloads_fixture();
    let meso_output = build_meso_milestone_from_minimal_spine_micro_payloads(&payloads).unwrap();
    let bytes = meso_output.deterministic_bytes();
    let forbidden = [
        b"ArchiveMilestoneSink".as_slice(),
        b"Replay".as_slice(),
        b"Sleep".as_slice(),
        b"Geist".as_slice(),
        b"ISM".as_slice(),
        b"finalized".as_slice(),
    ];

    for needle in forbidden {
        assert!(!bytes.windows(needle.len()).any(|window| window == needle));
    }
    assert_eq!(meso_output.micro_count, payloads.len() as u32);
}

#[test]
fn meso_builder_does_not_build_macro() {
    let payloads = payloads_fixture();
    let meso_output = build_meso_milestone_from_minimal_spine_micro_payloads(&payloads).unwrap();
    let bytes = meso_output.deterministic_bytes();

    assert!(!bytes
        .windows(b"macro".len())
        .any(|window| window == b"macro"));
    assert_eq!(meso_output.meso_milestone.micro_milestone_ids.len(), 3);
}
