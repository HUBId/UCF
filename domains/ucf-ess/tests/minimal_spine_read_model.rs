#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use ucf_ess::v1::{MinimalSpineEssProjection, MinimalSpineEssReadModel, MINIMAL_SPINE_ESS_SOURCE};
use ucf_types::{Digest32, EvidenceId};

fn fixture_digest(domain: &[u8], material: &[u8]) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(material);
    Digest32::new(hasher.finalize().into())
}

fn projection_fixture() -> MinimalSpineEssProjection {
    let input_digest = fixture_digest(
        b"ucf.minimal_spine.input.v1",
        b"ess-read-model-input-from-canonical-spine-record",
    );
    let candidate_set_record_digest = fixture_digest(
        b"ucf.minimal_spine.candidate_set_record.v1",
        b"canonical-candidate-set-record-digest-fixture",
    );
    let output_record_digest = fixture_digest(
        b"ucf.minimal_spine.output_record.v1",
        b"canonical-output-record-digest-fixture",
    );
    let archive_output_key = fixture_digest(
        b"ucf.minimal_spine.archive_output_key.v1",
        output_record_digest.as_bytes(),
    );

    MinimalSpineEssProjection::from_canonical_links(
        EvidenceId::new("minimal-spine-evidence-ess-read-model"),
        input_digest,
        candidate_set_record_digest,
        output_record_digest,
        archive_output_key,
        "allow",
        "materialized-test-output",
    )
}

#[test]
fn ess_projection_is_derived_from_canonical_spine_records() {
    let projection = projection_fixture();

    assert_eq!(projection.version, 1);
    assert_eq!(
        projection.evidence_id.as_str(),
        "minimal-spine-evidence-ess-read-model"
    );
    assert_ne!(projection.input_digest, Digest32::new([0; Digest32::LEN]));
    assert_ne!(
        projection.candidate_set_record_digest,
        Digest32::new([0; Digest32::LEN])
    );
    assert_ne!(
        projection.output_record_digest,
        Digest32::new([0; Digest32::LEN])
    );
    assert_ne!(
        projection.archive_output_key,
        Digest32::new([0; Digest32::LEN])
    );
    assert_eq!(projection.policy_status, "allow");
    assert_eq!(projection.output_status, "materialized-test-output");
    assert_eq!(projection.source, MINIMAL_SPINE_ESS_SOURCE);
}

#[test]
fn ess_read_model_indexes_by_output_digest_and_evidence_id() {
    let projection = projection_fixture();
    let output_record_digest = projection.output_record_digest;
    let evidence_id = projection.evidence_id.clone();
    let archive_output_key = projection.archive_output_key;

    let model = MinimalSpineEssReadModel::from_projection(projection.clone());

    assert_eq!(model.len(), 1);
    assert!(!model.is_empty());
    assert_eq!(
        model.get_by_output_digest(output_record_digest),
        Some(&projection)
    );
    assert_eq!(model.get_by_evidence_id(&evidence_id), Some(&projection));
    assert_eq!(
        model.get_by_archive_output_key(archive_output_key),
        Some(&projection)
    );
}

#[test]
fn ess_read_model_does_not_replace_archive_authority() {
    let projection = projection_fixture();
    let model = MinimalSpineEssReadModel::from_projection(projection.clone());

    let readback = model
        .get_by_output_digest(projection.output_record_digest)
        .expect("projection should be indexed by output digest");

    assert_eq!(readback.source, MINIMAL_SPINE_ESS_SOURCE);
    assert_eq!(
        readback.output_record_digest,
        projection.output_record_digest
    );
    assert_eq!(readback.archive_output_key, projection.archive_output_key);
    assert_eq!(
        readback.candidate_set_record_digest,
        projection.candidate_set_record_digest
    );
    assert_eq!(model.len(), 1);
}

#[test]
fn ess_projection_is_deterministic() {
    let a = projection_fixture();
    let b = projection_fixture();

    assert_eq!(a, b);
    assert_eq!(a.projection_digest(), b.projection_digest());
}
