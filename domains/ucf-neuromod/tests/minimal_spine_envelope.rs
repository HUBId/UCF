#![forbid(unsafe_code)]

use ucf_neuromod::minimal_spine::{
    MinimalSpineNeuromodEnvelope, MinimalSpineNeuromodHints, MinimalSpineNeuromodLinks,
    NeuromodHint, MINIMAL_SPINE_NEUROMOD_ENVELOPE_VERSION, MINIMAL_SPINE_NEUROMOD_SOURCE,
};
use ucf_types::{Digest32, EvidenceId};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn links(policy_status: &str, output_status: &str) -> MinimalSpineNeuromodLinks {
    MinimalSpineNeuromodLinks::new(
        11,
        EvidenceId::new("minimal-spine-evidence-neuromod-envelope"),
        digest(1),
        digest(2),
        digest(3),
        digest(4),
        digest(5),
        policy_status,
        output_status,
    )
}

fn hints() -> MinimalSpineNeuromodHints {
    MinimalSpineNeuromodHints::new(
        NeuromodHint::new(450).expect("bounded salience"),
        NeuromodHint::new(650).expect("bounded stability"),
        NeuromodHint::new(150).expect("bounded risk"),
        NeuromodHint::new(250).expect("bounded noise"),
        NeuromodHint::new(350).expect("bounded learning"),
    )
}

fn envelope_fixture() -> MinimalSpineNeuromodEnvelope {
    MinimalSpineNeuromodEnvelope::from_minimal_spine_links(
        links("allow", "materialized-test-output"),
        hints(),
    )
}

#[test]
fn neuromod_envelope_is_derived_from_canonical_spine_links() {
    let envelope = envelope_fixture();

    assert_eq!(envelope.version, MINIMAL_SPINE_NEUROMOD_ENVELOPE_VERSION);
    assert_eq!(envelope.sequence, 11);
    assert_eq!(
        envelope.evidence_id.as_str(),
        "minimal-spine-evidence-neuromod-envelope"
    );
    assert_eq!(envelope.input_digest, digest(1));
    assert_eq!(envelope.candidate_set_record_digest, digest(2));
    assert_eq!(envelope.output_record_digest, digest(3));
    assert_eq!(envelope.archive_output_key, digest(4));
    assert_eq!(envelope.archive_output_event_digest, digest(5));
    assert_eq!(envelope.policy_status, "allow");
    assert_eq!(envelope.output_status, "materialized-test-output");
    assert_eq!(envelope.source, MINIMAL_SPINE_NEUROMOD_SOURCE);
    assert!(envelope.validate_links_nonzero());
}

#[test]
fn neuromod_envelope_digest_is_deterministic() {
    let first = envelope_fixture();
    let second = envelope_fixture();

    assert_eq!(first, second);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.digest(), second.digest());

    let changed_output = MinimalSpineNeuromodEnvelope::from_minimal_spine_links(
        MinimalSpineNeuromodLinks::new(
            11,
            EvidenceId::new("minimal-spine-evidence-neuromod-envelope"),
            digest(1),
            digest(2),
            digest(9),
            digest(4),
            digest(5),
            "allow",
            "materialized-test-output",
        ),
        hints(),
    );
    assert_ne!(first.digest(), changed_output.digest());

    let changed_hint = MinimalSpineNeuromodEnvelope::from_minimal_spine_links(
        links("allow", "materialized-test-output"),
        MinimalSpineNeuromodHints::new(
            NeuromodHint::new(451).expect("bounded salience"),
            NeuromodHint::new(650).expect("bounded stability"),
            NeuromodHint::new(150).expect("bounded risk"),
            NeuromodHint::new(250).expect("bounded noise"),
            NeuromodHint::new(350).expect("bounded learning"),
        ),
    );
    assert_ne!(first.digest(), changed_hint.digest());
}

#[test]
fn neuromod_envelope_hints_are_bounded() {
    let envelope = envelope_fixture();

    assert!(envelope.validate_bounds());
    assert!(hints().validate_bounds());
    assert_eq!(NeuromodHint::new(0).expect("zero is bounded").raw(), 0);
    assert_eq!(NeuromodHint::new(1000).expect("max is bounded").raw(), 1000);
    assert!(NeuromodHint::new(1001).is_err());
}

#[test]
fn neuromod_hook_does_not_override_policy_or_output() {
    let envelope = envelope_fixture();

    assert_eq!(envelope.policy_status, "allow");
    assert_eq!(envelope.output_status, "materialized-test-output");
    assert!(!envelope.allows_decision_override());
}

#[test]
fn neuromod_hook_does_not_replace_evidence_archive_authority() {
    let envelope = envelope_fixture();

    assert_eq!(
        envelope.evidence_id.as_str(),
        "minimal-spine-evidence-neuromod-envelope"
    );
    assert_eq!(envelope.candidate_set_record_digest, digest(2));
    assert_eq!(envelope.output_record_digest, digest(3));
    assert_eq!(envelope.archive_output_key, digest(4));
    assert_eq!(envelope.archive_output_event_digest, digest(5));
    assert!(envelope.validate_links_nonzero());
}

#[test]
fn deny_policy_produces_metadata_only_risk_noise_hints() {
    let envelope = MinimalSpineNeuromodEnvelope::from_minimal_spine_links_with_conservative_hints(
        links("deny", "suppressed"),
    );

    assert_eq!(envelope.policy_status, "deny");
    assert_eq!(envelope.output_status, "suppressed");
    assert!(envelope.risk_hint.raw() > envelope.stability_hint.raw());
    assert!(envelope.noise_hint.raw() > envelope.learning_hint.raw());
    assert!(!envelope.allows_decision_override());
}
