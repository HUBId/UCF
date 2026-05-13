#![forbid(unsafe_code)]

use ucf_consolidation::{
    MinimalSpineMicroMilestoneCandidate, MINIMAL_SPINE_CONSOLIDATION_SOURCE,
    MINIMAL_SPINE_MICRO_MILESTONE_CANDIDATE_VERSION,
};
use ucf_types::{Digest32, EvidenceId};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn candidate_fixture() -> MinimalSpineMicroMilestoneCandidate {
    MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        7,
        EvidenceId::new("minimal-spine-evidence-consolidation-hook"),
        digest(1),
        digest(2),
        digest(3),
        digest(4),
        digest(5),
        "allow",
        "materialized-test-output",
    )
}

#[test]
fn micro_candidate_is_derived_from_canonical_spine_links() {
    let candidate = candidate_fixture();

    assert_eq!(
        candidate.version,
        MINIMAL_SPINE_MICRO_MILESTONE_CANDIDATE_VERSION
    );
    assert_eq!(candidate.sequence, 7);
    assert_eq!(
        candidate.evidence_id.as_str(),
        "minimal-spine-evidence-consolidation-hook"
    );
    assert_eq!(candidate.input_digest, digest(1));
    assert_eq!(candidate.candidate_set_record_digest, digest(2));
    assert_eq!(candidate.output_record_digest, digest(3));
    assert_eq!(candidate.archive_output_key, digest(4));
    assert_eq!(candidate.archive_output_event_digest, digest(5));
    assert_eq!(candidate.policy_status, "allow");
    assert_eq!(candidate.output_status, "materialized-test-output");
    assert_eq!(candidate.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
    assert!(candidate.validate_links_nonzero());
}

#[test]
fn micro_candidate_digest_is_deterministic() {
    let first = candidate_fixture();
    let second = candidate_fixture();

    assert_eq!(first, second);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.digest(), second.digest());

    let changed = MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        7,
        EvidenceId::new("minimal-spine-evidence-consolidation-hook"),
        digest(1),
        digest(2),
        digest(9),
        digest(4),
        digest(5),
        "allow",
        "materialized-test-output",
    );

    assert_ne!(first.digest(), changed.digest());
}

#[test]
fn micro_hook_does_not_finalize_macro_or_trigger_replay() {
    let candidate = candidate_fixture();
    let bytes = candidate.deterministic_bytes();

    assert_eq!(candidate.source, MINIMAL_SPINE_CONSOLIDATION_SOURCE);
    assert!(!bytes
        .windows(b"macro".len())
        .any(|window| window == b"macro"));
    assert!(!bytes
        .windows(b"replay".len())
        .any(|window| window == b"replay"));
}

#[test]
fn micro_hook_does_not_replace_evidence_archive_authority() {
    let candidate = candidate_fixture();

    assert_eq!(
        candidate.evidence_id.as_str(),
        "minimal-spine-evidence-consolidation-hook"
    );
    assert_eq!(candidate.candidate_set_record_digest, digest(2));
    assert_eq!(candidate.output_record_digest, digest(3));
    assert_eq!(candidate.archive_output_key, digest(4));
    assert_eq!(candidate.archive_output_event_digest, digest(5));
    assert!(candidate.validate_links_nonzero());
}

#[test]
fn micro_candidate_rejects_zero_links() {
    let candidate = MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links(
        7,
        EvidenceId::new("minimal-spine-evidence-consolidation-hook"),
        Digest32::new([0; Digest32::LEN]),
        digest(2),
        digest(3),
        digest(4),
        digest(5),
        "allow",
        "materialized-test-output",
    );

    assert!(!candidate.validate_links_nonzero());
}
