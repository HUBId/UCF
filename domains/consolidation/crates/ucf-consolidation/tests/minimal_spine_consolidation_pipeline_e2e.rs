#![forbid(unsafe_code)]

use ucf_archive_store::{ArchiveAppender, ArchiveStore, InMemoryArchiveStore};
use ucf_consolidation::{
    append_minimal_spine_meso_milestone, append_minimal_spine_micro_milestone,
    build_macro_milestone_candidate_from_minimal_spine_meso_payloads,
    build_meso_milestone_from_minimal_spine_micro_payloads,
    build_micro_milestone_from_minimal_spine_candidate, ConsolidationError,
    MinimalSpineMacroConsolidationFinalization, MinimalSpineMacroMilestoneCandidate,
    MinimalSpineMesoMilestoneAppendPayload, MinimalSpineMesoMilestoneAppendResult,
    MinimalSpineMesoMilestoneBuildOutput, MinimalSpineMicroMilestoneAppendPayload,
    MinimalSpineMicroMilestoneAppendResult, MinimalSpineMicroMilestoneBuildOutput,
    MinimalSpineMicroMilestoneCandidate, MINIMAL_SPINE_CONSOLIDATION_SOURCE,
    MINIMAL_SPINE_MACRO_CONSOLIDATION_FINALIZATION_SOURCE,
    MINIMAL_SPINE_MESO_MILESTONE_ARCHIVE_KIND, MINIMAL_SPINE_MICRO_MILESTONE_ARCHIVE_KIND,
};
use ucf_evidence::{EvidenceStore, InMemoryEvidenceStore};
use ucf_types::{Digest32, EvidenceId};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn micro_candidate_fixture(
    sequence: u64,
    evidence_suffix: &str,
    input_byte: u8,
    candidate_set_byte: u8,
    output_record_byte: u8,
    archive_key_byte: u8,
    archive_event_byte: u8,
) -> MinimalSpineMicroMilestoneCandidate {
    MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        sequence,
        EvidenceId::new(format!(
            "minimal-spine-evidence-pipeline-e2e-{evidence_suffix}"
        )),
        digest(input_byte),
        digest(candidate_set_byte),
        digest(output_record_byte),
        digest(archive_key_byte),
        digest(archive_event_byte),
        "allow",
        "materialized-test-output",
    )
}

fn micro_candidates_fixture() -> Vec<MinimalSpineMicroMilestoneCandidate> {
    vec![
        micro_candidate_fixture(101, "a", 1, 2, 3, 4, 5),
        micro_candidate_fixture(202, "b", 11, 12, 13, 14, 15),
    ]
}

#[derive(Debug)]
struct PipelineRun {
    micro_outputs: Vec<MinimalSpineMicroMilestoneBuildOutput>,
    micro_payloads: Vec<MinimalSpineMicroMilestoneAppendPayload>,
    micro_append_results: Vec<MinimalSpineMicroMilestoneAppendResult>,
    meso_output: MinimalSpineMesoMilestoneBuildOutput,
    meso_payload: MinimalSpineMesoMilestoneAppendPayload,
    meso_append_result: MinimalSpineMesoMilestoneAppendResult,
    macro_candidate: MinimalSpineMacroMilestoneCandidate,
    finalization: MinimalSpineMacroConsolidationFinalization,
    evidence_len: usize,
    micro_archive_count: usize,
    meso_archive_count: usize,
}

impl PipelineRun {
    fn micro_build_digests(&self) -> Vec<Digest32> {
        self.micro_outputs
            .iter()
            .map(|output| output.digest())
            .collect()
    }

    fn micro_payload_digests(&self) -> Vec<Digest32> {
        self.micro_payloads
            .iter()
            .map(MinimalSpineMicroMilestoneAppendPayload::digest)
            .collect()
    }

    fn micro_readback_digests(&self) -> Vec<Digest32> {
        self.micro_append_results
            .iter()
            .map(|result| result.readback_digest)
            .collect()
    }
}

fn run_pipeline() -> PipelineRun {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();
    let micro_candidates = micro_candidates_fixture();

    let micro_outputs: Vec<_> = micro_candidates
        .iter()
        .map(|candidate| {
            build_micro_milestone_from_minimal_spine_candidate(candidate)
                .expect("valid micro build output")
        })
        .collect();
    let micro_payloads: Vec<_> = micro_outputs
        .iter()
        .map(|output| {
            MinimalSpineMicroMilestoneAppendPayload::from_build_output(output)
                .expect("valid micro payload")
        })
        .collect();
    let micro_append_results: Vec<_> = micro_outputs
        .iter()
        .map(|output| {
            append_minimal_spine_micro_milestone(
                output,
                &evidence_store,
                &archive_store,
                &mut archive_appender,
            )
            .expect("explicit micro append")
        })
        .collect();

    let meso_output = build_meso_milestone_from_minimal_spine_micro_payloads(&micro_payloads)
        .expect("valid meso build output from explicit micro append payloads");
    let meso_payload = MinimalSpineMesoMilestoneAppendPayload::from_build_output(&meso_output)
        .expect("valid meso payload");
    let meso_append_result = append_minimal_spine_meso_milestone(
        &meso_output,
        &evidence_store,
        &archive_store,
        &mut archive_appender,
    )
    .expect("explicit meso append");
    let macro_candidate = build_macro_milestone_candidate_from_minimal_spine_meso_payloads(
        std::slice::from_ref(&meso_payload),
    )
    .expect("valid macro candidate from explicit meso append payload");
    let finalization = MinimalSpineMacroConsolidationFinalization::from_candidate(&macro_candidate)
        .expect("valid local consolidation finalization boundary");

    let evidence_len = evidence_store.len();
    let micro_archive_count = archive_store
        .iter_kind(MINIMAL_SPINE_MICRO_MILESTONE_ARCHIVE_KIND, None)
        .count();
    let meso_archive_count = archive_store
        .iter_kind(MINIMAL_SPINE_MESO_MILESTONE_ARCHIVE_KIND, None)
        .count();

    PipelineRun {
        micro_outputs,
        micro_payloads,
        micro_append_results,
        meso_output,
        meso_payload,
        meso_append_result,
        macro_candidate,
        finalization,
        evidence_len,
        micro_archive_count,
        meso_archive_count,
    }
}

#[test]
fn consolidation_pipeline_e2e_is_deterministic_across_fresh_runs() {
    let first = run_pipeline();
    let second = run_pipeline();

    assert_eq!(first.micro_build_digests(), second.micro_build_digests());
    assert_eq!(
        first.micro_payload_digests(),
        second.micro_payload_digests()
    );
    assert_eq!(
        first.micro_readback_digests(),
        second.micro_readback_digests()
    );
    assert_eq!(first.micro_append_results, second.micro_append_results);
    assert_eq!(first.meso_output.digest(), second.meso_output.digest());
    assert_eq!(first.meso_payload.digest(), second.meso_payload.digest());
    assert_eq!(first.meso_append_result, second.meso_append_result);
    assert_eq!(
        first.macro_candidate.digest(),
        second.macro_candidate.digest()
    );
    assert_eq!(first.finalization.digest(), second.finalization.digest());
}

#[test]
fn consolidation_pipeline_preserves_micro_to_macro_provenance() {
    let run = run_pipeline();
    let micro_payload_digests = run.micro_payload_digests();
    let micro_milestone_digests: Vec<_> = run
        .micro_outputs
        .iter()
        .map(|output| output.micro_milestone_digest)
        .collect();

    for payload_digest in &micro_payload_digests {
        assert!(run
            .meso_output
            .micro_payload_digests
            .contains(payload_digest));
        assert!(run
            .meso_payload
            .micro_payload_digests
            .contains(payload_digest));
    }
    for milestone_digest in &micro_milestone_digests {
        assert!(run
            .meso_output
            .micro_milestone_digests
            .contains(milestone_digest));
        assert!(run
            .meso_payload
            .micro_milestone_digests
            .contains(milestone_digest));
    }

    assert!(run
        .macro_candidate
        .meso_payload_digests
        .contains(&run.meso_payload.digest()));
    assert!(run
        .macro_candidate
        .meso_build_output_digests
        .contains(&run.meso_output.digest()));
    assert!(run
        .macro_candidate
        .meso_milestone_digests
        .contains(&run.meso_output.meso_milestone_digest));
    assert!(run
        .macro_candidate
        .meso_aggregation_digests
        .contains(&run.meso_output.aggregation_digest));
    assert_eq!(
        run.finalization.macro_candidate_digest,
        run.macro_candidate.digest()
    );
    assert_eq!(
        run.finalization.macro_milestone_digest,
        run.macro_candidate.macro_milestone_digest
    );
    assert_eq!(run.meso_payload.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
    assert_eq!(
        run.macro_candidate.source,
        "minimal_spine_v1_macro_candidate"
    );
    assert_eq!(
        run.finalization.source,
        MINIMAL_SPINE_MACRO_CONSOLIDATION_FINALIZATION_SOURCE
    );
}

#[test]
fn consolidation_pipeline_requires_explicit_append_stages() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let micro_candidates = micro_candidates_fixture();
    let micro_outputs: Vec<_> = micro_candidates
        .iter()
        .map(|candidate| {
            build_micro_milestone_from_minimal_spine_candidate(candidate)
                .expect("valid micro build output")
        })
        .collect();
    let micro_payloads: Vec<_> = micro_outputs
        .iter()
        .map(|output| {
            MinimalSpineMicroMilestoneAppendPayload::from_build_output(output)
                .expect("valid micro payload")
        })
        .collect();
    let meso_output = build_meso_milestone_from_minimal_spine_micro_payloads(&micro_payloads)
        .expect("valid meso build output");
    let meso_payload = MinimalSpineMesoMilestoneAppendPayload::from_build_output(&meso_output)
        .expect("valid meso payload");
    let macro_candidate = build_macro_milestone_candidate_from_minimal_spine_meso_payloads(
        std::slice::from_ref(&meso_payload),
    )
    .expect("valid macro candidate");
    let _finalization =
        MinimalSpineMacroConsolidationFinalization::from_candidate(&macro_candidate)
            .expect("valid finalization boundary");

    assert!(evidence_store.is_empty());
    assert_eq!(archive_store.root_commit(), None);

    let run = run_pipeline();
    assert_eq!(run.evidence_len, 3);
    assert_eq!(run.micro_archive_count, 2);
    assert_eq!(run.meso_archive_count, 1);
}

#[test]
fn consolidation_pipeline_has_no_replay_geist_gateway_or_identity_side_effects() {
    let run = run_pipeline();
    let finalization_bytes = run.finalization.deterministic_bytes();
    let macro_candidate_bytes = run.macro_candidate.deterministic_bytes();

    assert!(run.finalization.consolidation_finalized);
    assert!(!run.finalization.identity_anchor);
    assert!(!run.finalization.geist_ingested);
    assert!(!run.finalization.replay_completed);
    assert!(!run.finalization.evidence_archive_appended);
    assert!(!run.finalization.gateway_visible);
    assert!(!run.macro_candidate.finalized);
    assert!(!run.macro_candidate.identity_anchor);

    for forbidden in [
        b"Replay".as_slice(),
        b"replay".as_slice(),
        b"Sleep".as_slice(),
        b"sleep".as_slice(),
        b"Geist".as_slice(),
        b"geist".as_slice(),
        b"ISM".as_slice(),
        b"gateway".as_slice(),
        b"Gateway".as_slice(),
        b"capability".as_slice(),
        b"Capability".as_slice(),
        b"real_compute".as_slice(),
        b"identity_anchor".as_slice(),
        b"ArchiveMilestoneSink".as_slice(),
        b"MacroMilestoneFinalized".as_slice(),
    ] {
        assert!(!contains_bytes(&finalization_bytes, forbidden));
        assert!(!contains_bytes(&macro_candidate_bytes, forbidden));
    }
}

#[test]
fn consolidation_pipeline_rejects_invalid_or_duplicate_inputs() {
    let run = run_pipeline();
    let duplicate_micro_error = build_meso_milestone_from_minimal_spine_micro_payloads(&[
        run.micro_payloads[0].clone(),
        run.micro_payloads[0].clone(),
    ])
    .unwrap_err();
    assert_eq!(
        duplicate_micro_error,
        ConsolidationError::DuplicateMinimalSpineMesoMilestoneInput
    );

    let duplicate_meso_error = build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&[
        run.meso_payload.clone(),
        run.meso_payload.clone(),
    ])
    .unwrap_err();
    assert_eq!(
        duplicate_meso_error,
        ConsolidationError::DuplicateMinimalSpineMacroMilestoneInput
    );

    let invalid_candidate = MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        303,
        EvidenceId::new("minimal-spine-evidence-pipeline-e2e-invalid"),
        Digest32::new([0; Digest32::LEN]),
        digest(22),
        digest(23),
        digest(24),
        digest(25),
        "allow",
        "materialized-test-output",
    );
    assert_eq!(
        build_micro_milestone_from_minimal_spine_candidate(&invalid_candidate).unwrap_err(),
        ConsolidationError::InvalidMinimalSpineMicroMilestoneCandidateLinks
    );

    let mut invalid_payload = run.micro_payloads[0].clone();
    invalid_payload.micro_milestone_digest = Digest32::new([0; Digest32::LEN]);
    assert_eq!(
        build_meso_milestone_from_minimal_spine_micro_payloads(&[invalid_payload]).unwrap_err(),
        ConsolidationError::InvalidMinimalSpineMesoMilestoneInputLinks
    );
}
