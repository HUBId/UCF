#![forbid(unsafe_code)]

use prost::Message;
use ucf_archive_store::{ArchiveAppender, ArchiveStore, InMemoryArchiveStore};
use ucf_consolidation::{
    build_macro_milestone_candidate_from_minimal_spine_meso_build_outputs,
    build_macro_milestone_candidate_from_minimal_spine_meso_payloads,
    build_meso_milestone_from_minimal_spine_micro_build_outputs,
    build_micro_milestone_from_minimal_spine_candidate, ConsolidationError,
    MinimalSpineMacroMilestoneCandidate, MinimalSpineMesoMilestoneAppendPayload,
    MinimalSpineMesoMilestoneBuildOutput, MinimalSpineMicroMilestoneBuildOutput,
    MinimalSpineMicroMilestoneCandidate, MINIMAL_SPINE_CONSOLIDATION_SOURCE,
    MINIMAL_SPINE_MACRO_MILESTONE_CANDIDATE_SOURCE,
    MINIMAL_SPINE_MACRO_MILESTONE_CANDIDATE_VERSION,
};
use ucf_evidence::{EvidenceStore, InMemoryEvidenceStore};
use ucf_types::v1::spec::MacroMilestone;
use ucf_types::{Digest32, EvidenceId};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn micro_candidate_fixture(
    sequence: u64,
    evidence_suffix: &str,
    output_record_byte: u8,
) -> MinimalSpineMicroMilestoneCandidate {
    MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        sequence,
        EvidenceId::new(format!("minimal-spine-evidence-macro-{evidence_suffix}")),
        digest(1),
        digest(2),
        digest(output_record_byte),
        digest(4),
        digest(5),
        "allow",
        "materialized-test-output",
    )
}

fn micro_output_fixture(
    sequence: u64,
    evidence_suffix: &str,
    output_record_byte: u8,
) -> MinimalSpineMicroMilestoneBuildOutput {
    build_micro_milestone_from_minimal_spine_candidate(&micro_candidate_fixture(
        sequence,
        evidence_suffix,
        output_record_byte,
    ))
    .expect("valid micro build output")
}

fn meso_output_fixture(offset: u8) -> MinimalSpineMesoMilestoneBuildOutput {
    let outputs = vec![
        micro_output_fixture(10 + u64::from(offset), &format!("{offset}-a"), 10 + offset),
        micro_output_fixture(20 + u64::from(offset), &format!("{offset}-b"), 20 + offset),
    ];
    build_meso_milestone_from_minimal_spine_micro_build_outputs(&outputs)
        .expect("valid meso build output")
}

fn meso_outputs_fixture() -> Vec<MinimalSpineMesoMilestoneBuildOutput> {
    vec![
        meso_output_fixture(1),
        meso_output_fixture(2),
        meso_output_fixture(3),
    ]
}

fn meso_payloads_fixture() -> Vec<MinimalSpineMesoMilestoneAppendPayload> {
    meso_outputs_fixture()
        .iter()
        .map(MinimalSpineMesoMilestoneAppendPayload::from_build_output)
        .collect::<Result<Vec<_>, _>>()
        .expect("valid meso payloads")
}

#[test]
fn macro_candidate_from_meso_payloads_is_deterministic() {
    let payloads = meso_payloads_fixture();

    let first = build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&payloads)
        .expect("macro candidate");
    let second = build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&payloads)
        .expect("macro candidate");

    assert_eq!(first, second);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.digest(), second.digest());
    assert_eq!(first.macro_candidate_digest, first.digest());
}

#[test]
fn macro_candidate_normalizes_input_order() {
    let payloads = meso_payloads_fixture();
    let mut reversed = payloads.clone();
    reversed.reverse();

    let sorted_candidate =
        build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&payloads)
            .expect("macro candidate");
    let reversed_candidate =
        build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&reversed)
            .expect("macro candidate");

    assert_eq!(sorted_candidate, reversed_candidate);
    assert_eq!(sorted_candidate.digest(), reversed_candidate.digest());
}

#[test]
fn macro_candidate_changes_when_meso_payload_changes() {
    let payloads = meso_payloads_fixture();
    let mut changed = payloads.clone();
    changed[1] = MinimalSpineMesoMilestoneAppendPayload::from_build_output(&meso_output_fixture(8))
        .expect("changed meso payload");

    let original_candidate =
        build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&payloads)
            .expect("macro candidate");
    let changed_candidate =
        build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&changed)
            .expect("macro candidate");

    assert_ne!(
        original_candidate.macro_aggregation_digest,
        changed_candidate.macro_aggregation_digest
    );
    assert_ne!(
        original_candidate.macro_milestone_digest,
        changed_candidate.macro_milestone_digest
    );
    assert_ne!(original_candidate.digest(), changed_candidate.digest());
}

#[test]
fn macro_candidate_outputs_protocol_macro_milestone() {
    let outputs = meso_outputs_fixture();
    let candidate = build_macro_milestone_candidate_from_minimal_spine_meso_build_outputs(&outputs)
        .expect("macro candidate");
    let canonical = ucf::canonical_bytes(&candidate.macro_milestone);
    let prost_bytes = candidate.macro_milestone.encode_to_vec();
    let decoded = MacroMilestone::decode(prost_bytes.as_slice()).expect("prost macro roundtrip");

    assert_eq!(
        candidate.version,
        MINIMAL_SPINE_MACRO_MILESTONE_CANDIDATE_VERSION
    );
    assert_eq!(decoded, candidate.macro_milestone);
    assert!(!canonical.is_empty());
    assert!(candidate
        .macro_milestone
        .milestone_id
        .starts_with("minimal-spine-macro-candidate-"));
    assert_eq!(candidate.macro_milestone.achieved_at_ms, 23);
    assert_eq!(
        candidate.macro_milestone.meso_milestone_ids.len(),
        outputs.len()
    );
}

#[test]
fn macro_candidate_preserves_meso_provenance() {
    let payloads = meso_payloads_fixture();
    let candidate = build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&payloads)
        .expect("macro candidate");
    let mut expected_payload_digests: Vec<_> =
        payloads.iter().map(|payload| payload.digest()).collect();
    expected_payload_digests.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    let mut expected_meso_links: Vec<_> = payloads
        .iter()
        .map(|payload| {
            (
                payload.digest(),
                payload.build_output_digest,
                payload.meso_milestone_digest,
                payload.aggregation_digest,
            )
        })
        .collect();
    expected_meso_links.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let expected_build_output_digests: Vec<_> = expected_meso_links
        .iter()
        .map(|(_, build_output_digest, _, _)| *build_output_digest)
        .collect();
    let expected_meso_milestone_digests: Vec<_> = expected_meso_links
        .iter()
        .map(|(_, _, meso_milestone_digest, _)| *meso_milestone_digest)
        .collect();
    let expected_meso_aggregation_digests: Vec<_> = expected_meso_links
        .iter()
        .map(|(_, _, _, aggregation_digest)| *aggregation_digest)
        .collect();

    assert_eq!(candidate.meso_payload_digests, expected_payload_digests);
    assert_eq!(
        candidate.meso_build_output_digests,
        expected_build_output_digests
    );
    assert_eq!(
        candidate.meso_milestone_digests,
        expected_meso_milestone_digests
    );
    assert_eq!(
        candidate.meso_aggregation_digests,
        expected_meso_aggregation_digests
    );
    assert_eq!(candidate.meso_count, 3);
    assert_ne!(
        candidate.macro_aggregation_digest,
        Digest32::new([0; Digest32::LEN])
    );
    assert_eq!(
        candidate.source,
        MINIMAL_SPINE_MACRO_MILESTONE_CANDIDATE_SOURCE
    );
}

#[test]
fn macro_candidate_rejects_empty_or_duplicate_inputs() {
    let payloads = meso_payloads_fixture();
    let mut duplicate = payloads.clone();
    duplicate.push(payloads[0].clone());
    let mut duplicate_meso_digest = payloads.clone();
    duplicate_meso_digest[1].meso_milestone_digest = duplicate_meso_digest[0].meso_milestone_digest;

    assert_eq!(
        build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&[]).unwrap_err(),
        ConsolidationError::MinimalSpineMacroMilestoneEmptyInput
    );
    assert_eq!(
        build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&duplicate).unwrap_err(),
        ConsolidationError::DuplicateMinimalSpineMacroMilestoneInput
    );
    assert_eq!(
        build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&duplicate_meso_digest)
            .unwrap_err(),
        ConsolidationError::DuplicateMinimalSpineMacroMilestoneInput
    );
}

#[test]
fn macro_candidate_is_not_finalized_or_identity_anchor() {
    let payloads = meso_payloads_fixture();
    let candidate = build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&payloads)
        .expect("macro candidate");
    let candidate_bytes = candidate.deterministic_bytes();
    let forbidden = [
        b"finalized".as_slice(),
        b"identity".as_slice(),
        b"Geist".as_slice(),
        b"ISM".as_slice(),
    ];

    assert!(!candidate.finalized);
    assert!(!candidate.identity_anchor);
    for needle in forbidden {
        assert!(!candidate_bytes
            .windows(needle.len())
            .any(|window| window == needle));
    }
}

#[test]
fn macro_candidate_has_no_archive_replay_geist_side_effects() {
    let payloads = meso_payloads_fixture();
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let _archive_appender = ArchiveAppender::new();

    let candidate = build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&payloads)
        .expect("macro candidate");
    let candidate_bytes = candidate.deterministic_bytes();
    let forbidden = [
        b"replay".as_slice(),
        b"Replay".as_slice(),
        b"sleep".as_slice(),
        b"Sleep".as_slice(),
        b"Geist".as_slice(),
        b"ISM".as_slice(),
        b"append".as_slice(),
    ];

    for needle in forbidden {
        assert!(!candidate_bytes
            .windows(needle.len())
            .any(|window| window == needle));
    }
    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);
    assert_eq!(
        candidate.source,
        MINIMAL_SPINE_MACRO_MILESTONE_CANDIDATE_SOURCE
    );
}

#[test]
fn macro_candidate_does_not_trigger_macro_finalized_event() {
    let payloads = meso_payloads_fixture();
    let candidate: MinimalSpineMacroMilestoneCandidate =
        build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&payloads)
            .expect("macro candidate");
    let candidate_bytes = candidate.deterministic_bytes();
    let forbidden = [
        b"MacroMilestoneFinalized".as_slice(),
        b"publish".as_slice(),
        b"finalized".as_slice(),
    ];

    for needle in forbidden {
        assert!(!candidate_bytes
            .windows(needle.len())
            .any(|window| window == needle));
    }
    assert!(!candidate.finalized);
    assert_eq!(
        candidate.source,
        MINIMAL_SPINE_MACRO_MILESTONE_CANDIDATE_SOURCE
    );
    assert_ne!(candidate.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
}
