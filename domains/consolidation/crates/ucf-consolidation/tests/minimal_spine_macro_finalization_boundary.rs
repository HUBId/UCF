#![forbid(unsafe_code)]

use ucf_consolidation::{
    build_macro_milestone_candidate_from_minimal_spine_meso_build_outputs,
    build_meso_milestone_from_minimal_spine_micro_build_outputs,
    build_micro_milestone_from_minimal_spine_candidate, ConsolidationError,
    MinimalSpineMacroConsolidationFinalization, MinimalSpineMacroMilestoneCandidate,
    MinimalSpineMesoMilestoneBuildOutput, MinimalSpineMicroMilestoneBuildOutput,
    MinimalSpineMicroMilestoneCandidate, MINIMAL_SPINE_MACRO_CONSOLIDATION_FINALIZATION_SOURCE,
    MINIMAL_SPINE_MACRO_CONSOLIDATION_FINALIZATION_VERSION,
};
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
        EvidenceId::new(format!(
            "minimal-spine-evidence-macro-finalization-{evidence_suffix}"
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

fn macro_candidate_fixture() -> MinimalSpineMacroMilestoneCandidate {
    let meso_outputs = vec![
        meso_output_fixture(1),
        meso_output_fixture(2),
        meso_output_fixture(3),
    ];
    build_macro_milestone_candidate_from_minimal_spine_meso_build_outputs(&meso_outputs)
        .expect("valid macro candidate")
}

#[test]
fn macro_consolidation_finalization_is_deterministic() {
    let candidate = macro_candidate_fixture();

    let first = MinimalSpineMacroConsolidationFinalization::from_candidate(&candidate)
        .expect("finalization boundary");
    let second = MinimalSpineMacroConsolidationFinalization::from_candidate(&candidate)
        .expect("finalization boundary");

    assert_eq!(first, second);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.digest(), second.digest());
    assert_eq!(
        first.macro_candidate_digest,
        candidate.macro_candidate_digest
    );
    assert_eq!(
        first.macro_milestone_digest,
        candidate.macro_milestone_digest
    );
    assert_eq!(first.meso_count, candidate.meso_count);
    assert_eq!(
        first.version,
        MINIMAL_SPINE_MACRO_CONSOLIDATION_FINALIZATION_VERSION
    );
    assert_eq!(
        first.source,
        MINIMAL_SPINE_MACRO_CONSOLIDATION_FINALIZATION_SOURCE
    );
}

#[test]
fn macro_consolidation_finalization_is_not_identity_anchor() {
    let candidate = macro_candidate_fixture();
    let boundary = MinimalSpineMacroConsolidationFinalization::from_candidate(&candidate)
        .expect("finalization boundary");

    assert!(boundary.consolidation_finalized);
    assert!(!boundary.identity_anchor);
    assert!(!boundary.geist_ingested);
    assert!(!boundary.replay_completed);
    assert!(!boundary.evidence_archive_appended);
    assert!(!boundary.gateway_visible);
}

#[test]
fn macro_consolidation_finalization_does_not_publish_macro_finalized_event() {
    let candidate = macro_candidate_fixture();
    let boundary = MinimalSpineMacroConsolidationFinalization::from_candidate(&candidate)
        .expect("finalization boundary");
    let bytes = boundary.deterministic_bytes();

    for forbidden in [
        b"MacroMilestoneFinalized".as_slice(),
        b"publish_macro".as_slice(),
        b"ArchiveMilestoneSink".as_slice(),
    ] {
        assert!(!bytes
            .windows(forbidden.len())
            .any(|window| window == forbidden));
    }
    assert!(boundary.consolidation_finalized);
}

#[test]
fn macro_consolidation_finalization_does_not_append_or_trigger_replay_geist() {
    let candidate = macro_candidate_fixture();
    let boundary = MinimalSpineMacroConsolidationFinalization::from_candidate(&candidate)
        .expect("finalization boundary");
    let bytes = boundary.deterministic_bytes();

    for forbidden in [
        b"append".as_slice(),
        b"Replay".as_slice(),
        b"Sleep".as_slice(),
        b"Geist".as_slice(),
        b"ISM".as_slice(),
        b"gateway".as_slice(),
    ] {
        assert!(!bytes
            .windows(forbidden.len())
            .any(|window| window == forbidden));
    }
    assert!(!boundary.evidence_archive_appended);
    assert!(!boundary.replay_completed);
    assert!(!boundary.geist_ingested);
}

#[test]
fn macro_consolidation_finalization_rejects_invalid_candidate() {
    let candidate = macro_candidate_fixture();

    let mut zero_digest_candidate = candidate.clone();
    zero_digest_candidate.macro_candidate_digest = Digest32::new([0; Digest32::LEN]);
    assert_eq!(
        MinimalSpineMacroConsolidationFinalization::from_candidate(&zero_digest_candidate)
            .unwrap_err(),
        ConsolidationError::InvalidMinimalSpineMacroFinalizationBoundaryCandidate
    );

    let mut already_finalized_candidate = candidate.clone();
    already_finalized_candidate.finalized = true;
    assert_eq!(
        MinimalSpineMacroConsolidationFinalization::from_candidate(&already_finalized_candidate)
            .unwrap_err(),
        ConsolidationError::InvalidMinimalSpineMacroFinalizationBoundaryCandidate
    );

    let mut identity_anchor_candidate = candidate;
    identity_anchor_candidate.identity_anchor = true;
    assert_eq!(
        MinimalSpineMacroConsolidationFinalization::from_candidate(&identity_anchor_candidate)
            .unwrap_err(),
        ConsolidationError::InvalidMinimalSpineMacroFinalizationBoundaryCandidate
    );
}

#[test]
fn macro_candidate_remains_candidate_only() {
    let candidate = macro_candidate_fixture();
    let candidate_before = candidate.clone();
    let boundary = MinimalSpineMacroConsolidationFinalization::from_candidate(&candidate)
        .expect("finalization boundary");

    assert!(!candidate.finalized);
    assert!(!candidate.identity_anchor);
    assert_eq!(candidate, candidate_before);
    assert!(boundary.consolidation_finalized);
}
