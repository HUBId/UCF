#![forbid(unsafe_code)]

use ucf_archive_store::{
    ArchiveAppender, ArchiveStore, InMemoryArchiveStore, RecordKind, RecordMeta,
};
use ucf_evidence::{EvidenceEnvelope, EvidenceStore, InMemoryEvidenceStore};
use ucf_gateway::spine_read::{SpineReadError, SpineReadService};
use ucf_protocol::canonical_bytes;
use ucf_protocol::v1::spec::{CandidateSetRecord, ExperienceRecord, OutputRecord, ProofEnvelope};
use ucf_types::{Digest32, EvidenceId, LogicalTime, WallTime};

const FIXED_OBSERVED_AT_MS: u64 = 1_700_000_000_000;

struct MinimalSpineFixture {
    evidence_id: EvidenceId,
    candidate_set_record_digest: Digest32,
    output_record_digest: Digest32,
    output_record_bytes: Vec<u8>,
    archive_key: Digest32,
    archive_payload_commit: Digest32,
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn payload_commit(bytes: &[u8]) -> Digest32 {
    Digest32::new(*blake3::hash(bytes).as_bytes())
}

fn digest_vec(digest: Digest32) -> Vec<u8> {
    digest.as_bytes().to_vec()
}

fn append_minimal_spine_fixture(
    evidence_store: &InMemoryEvidenceStore,
    archive_store: &InMemoryArchiveStore,
) -> MinimalSpineFixture {
    let input_digest = digest_bytes(
        b"ucf.minimal_spine.input.v1",
        b"gateway-spine-read-api-v1.1-input",
    );
    let policy_decision_digest = digest_bytes(
        b"ucf.minimal_spine.policy_decision.v1",
        b"allow:minimal-spine-gateway-read-api-v1.1",
    );
    let selected_candidate_digest = digest_bytes(
        b"ucf.minimal_spine.candidate.v1",
        b"candidate-0:no-real-compute",
    );
    let candidate_set_record = CandidateSetRecord {
        version: 1,
        input_digest: digest_vec(input_digest),
        policy_decision_digest: digest_vec(policy_decision_digest),
        candidate_count: 1,
        candidate_digests: vec![digest_vec(selected_candidate_digest)],
        candidates_digest: digest_vec(digest_bytes(
            b"ucf.minimal_spine.candidates.v1",
            selected_candidate_digest.as_bytes(),
        )),
        provenance: "minimal-spine-v1.1-gateway-read-fixture-no-real-compute".to_string(),
    };
    let candidate_set_bytes = canonical_bytes(&candidate_set_record);
    let candidate_set_record_digest = digest_bytes(
        b"ucf.minimal_spine.candidate_set_record.v1",
        &candidate_set_bytes,
    );

    let output_material_digest = digest_bytes(
        b"ucf.minimal_spine.output_material.v1",
        b"gateway-read-api-output:no-real-compute",
    );
    let output_record = OutputRecord {
        version: 1,
        input_digest: digest_vec(input_digest),
        candidate_set_digest: digest_vec(candidate_set_record_digest),
        selected_candidate_digest: digest_vec(selected_candidate_digest),
        output_digest: digest_vec(output_material_digest),
        policy_status: "allow".to_string(),
        status: "materialized-test-output".to_string(),
        provenance: "minimal-spine-v1.1-gateway-read-fixture-no-real-compute".to_string(),
        evidence_id: None,
    };
    let output_record_bytes = canonical_bytes(&output_record);
    let output_record_digest =
        digest_bytes(b"ucf.minimal_spine.output_record.v1", &output_record_bytes);

    let evidence_payload = format!(
        "minimal_spine_evidence=v1;status=allow;frame_id=gateway-read-api;input_digest={};candidate_set_record_digest={};output_record_digest={};",
        hex::encode(input_digest.as_bytes()),
        hex::encode(candidate_set_record_digest.as_bytes()),
        hex::encode(output_record_digest.as_bytes())
    )
    .into_bytes();
    let evidence_id = EvidenceId::new(format!(
        "minimal-spine-gateway-read-{}",
        &hex::encode(
            digest_bytes(b"ucf.minimal_spine.evidence_id.v1", &evidence_payload).as_bytes()
        )[..16]
    ));
    let evidence_record = ExperienceRecord {
        record_id: evidence_id.as_str().to_string(),
        observed_at_ms: FIXED_OBSERVED_AT_MS,
        subject_id: "minimal-spine-route-result".to_string(),
        payload: evidence_payload,
        digest: None,
        vrf_tag: None,
        proof_ref: None,
    };
    let proof = ProofEnvelope {
        envelope_id: format!("minimal-spine-proof-{}", evidence_record.record_id),
        payload: canonical_bytes(&evidence_record),
        payload_digest: None,
        vrf_tags: Vec::new(),
        signature_ids: vec!["minimal-spine-v1.1-gateway-read-fixture".to_string()],
    };
    let envelope = EvidenceEnvelope {
        evidence_id: evidence_id.clone(),
        proof: Some(proof),
        fold_proof: None,
        logical_time: LogicalTime::new(1),
        wall_time: WallTime::new(FIXED_OBSERVED_AT_MS),
    };
    assert_eq!(evidence_store.append(envelope), evidence_id);

    let meta = RecordMeta {
        cycle_id: 1,
        tier: 0,
        flags: 0,
        boundary_commit: output_record_digest,
    };
    let mut appender = ArchiveAppender::new();
    let archive_record = appender.build_record(RecordKind::OutputEvent, &output_record_bytes, meta);
    archive_store.append(archive_record);

    let archive_payload_commit = payload_commit(&output_record_bytes);

    MinimalSpineFixture {
        evidence_id,
        candidate_set_record_digest,
        output_record_digest,
        output_record_bytes,
        archive_key: archive_record.key,
        archive_payload_commit,
    }
}

#[test]
fn spine_read_health_is_read_only() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let service = SpineReadService::new(&evidence_store, &archive_store);

    let health = service.spine_read_health();

    assert_eq!(health.status, "ok");
    assert_eq!(health.mode, "read_only");
    assert_eq!(health.spine_version, "v1.1");
    assert!(evidence_store.is_empty());
    assert!(archive_store.root_commit().is_none());
}

#[test]
fn spine_read_evidence_returns_linked_output_digest() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let fixture = append_minimal_spine_fixture(&evidence_store, &archive_store);
    let service = SpineReadService::new(&evidence_store, &archive_store);

    let summary = service
        .read_evidence(fixture.evidence_id.clone())
        .expect("evidence readback");

    assert_eq!(summary.evidence_id, fixture.evidence_id.as_str());
    assert_eq!(summary.proof_signature_count, 1);
    assert_eq!(
        summary.experience_subject_id.as_deref(),
        Some("minimal-spine-route-result")
    );
    assert_eq!(
        summary.candidate_set_record_digest_hex.as_deref(),
        Some(hex::encode(fixture.candidate_set_record_digest.as_bytes()).as_str())
    );
    assert_eq!(
        summary.output_record_digest_hex.as_deref(),
        Some(hex::encode(fixture.output_record_digest.as_bytes()).as_str())
    );
}

#[test]
fn spine_read_output_event_returns_output_commit() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let fixture = append_minimal_spine_fixture(&evidence_store, &archive_store);
    let service = SpineReadService::new(&evidence_store, &archive_store);

    let summary = service
        .read_output_event(fixture.archive_key)
        .expect("output event readback");

    assert_eq!(summary.record_kind, "OutputEvent");
    assert_eq!(
        summary.payload_commit_hex,
        hex::encode(fixture.archive_payload_commit.as_bytes())
    );
    assert_eq!(
        summary.boundary_commit_hex,
        hex::encode(fixture.output_record_digest.as_bytes())
    );
    assert_eq!(
        summary.output_record_digest_hex,
        hex::encode(fixture.output_record_digest.as_bytes())
    );
    assert_eq!(summary.cycle_id, 1);
    assert!(summary.root_commit_hex.is_some());
    assert_eq!(summary.payload_bytes_len, None);
    assert_eq!(
        fixture.archive_payload_commit,
        payload_commit(&fixture.output_record_bytes)
    );

    let missing = Digest32::new([9u8; Digest32::LEN]);
    assert_eq!(
        service.read_output_event(missing),
        Err(SpineReadError::OutputEventNotFound)
    );
}

#[test]
fn spine_read_api_has_no_write_path() {
    let evidence_store = InMemoryEvidenceStore::new();
    let archive_store = InMemoryArchiveStore::new();
    let service = SpineReadService::new(&evidence_store, &archive_store);

    assert_eq!(service.spine_read_health().mode, "read_only");
    assert_eq!(evidence_store.len(), 0);
    assert!(archive_store.root_commit().is_none());
    assert_eq!(
        service.read_evidence(EvidenceId::new("missing")),
        Err(SpineReadError::EvidenceNotFound)
    );
    assert_eq!(evidence_store.len(), 0);
    assert!(archive_store.root_commit().is_none());
}
