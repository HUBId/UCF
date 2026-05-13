#![forbid(unsafe_code)]

use ucf::v1::spec::{
    ActionCode, CandidateSetRecord, ControlFrame, DecisionKind, ExperienceRecord, OutputRecord,
    PolicyDecision, ProofEnvelope,
};
use ucf::{canonical_bytes, DecodeError};
use ucf_archive::{build_compact_record, ExperienceAppender, InMemoryArchive};
use ucf_archive_store::{
    ArchiveAppender, ArchiveStore, InMemoryArchiveStore, RecordKind, RecordMeta,
};
use ucf_policy_ecology::{PolicyEcology, PolicyRule, PolicyWeights, ReplayGate};
use ucf_types::{Digest32, EvidenceId};

const FIXED_OBSERVED_AT_MS: u64 = 1_700_000_000_123;
const MINIMAL_SPINE_DENY_REASON: &str = "minimal_spine_policy_denied_decision_class";
const MINIMAL_SPINE_ALLOW_REASON: &str = "minimal_spine_policy_allowed";

#[derive(Clone, Debug, Eq, PartialEq)]
enum MinimalSpinePolicyDecision {
    Allow { reason: &'static str },
    Deny { reason: &'static str },
}

#[derive(Clone, Debug, PartialEq)]
struct MinimalSpineRunResult {
    input_bytes: Vec<u8>,
    input_digest: Digest32,
    policy_decision: MinimalSpinePolicyDecision,
    candidate_set_record: Option<CandidateSetRecord>,
    candidate_set_bytes: Option<Vec<u8>>,
    candidate_set_digest: Option<Digest32>,
    output_record: Option<OutputRecord>,
    output_record_bytes: Option<Vec<u8>>,
    output_record_digest: Option<Digest32>,
    evidence_id: Option<EvidenceId>,
    evidence_record: Option<ExperienceRecord>,
    evidence_bytes: Option<Vec<u8>>,
    evidence_digest: Option<Digest32>,
    archive_key: Option<Digest32>,
    archive_payload_commit: Option<Digest32>,
    archive_append_commit: Option<Digest32>,
    archive_root_commit: Option<Digest32>,
    archive_readback_output_record_bytes: Option<Vec<u8>>,
    evidence_entries_len: usize,
    archive_output_records_len: usize,
}

fn minimal_spine_frame(frame_id: &str, decision_kind: DecisionKind) -> ControlFrame {
    ControlFrame {
        frame_id: frame_id.to_string(),
        issued_at_ms: FIXED_OBSERVED_AT_MS,
        decision: Some(PolicyDecision {
            kind: decision_kind as i32,
            action: ActionCode::ActionCodeContinue as i32,
            rationale: format!("minimal-spine-{decision_kind:?}"),
            confidence_bp: 9_000,
            constraint_ids: vec!["minimal-spine-v1".to_string()],
        }),
        evidence_ids: vec!["minimal-spine-input-evidence".to_string()],
        policy_id: "minimal-spine-policy-v1".to_string(),
    }
}

fn minimal_spine_policy() -> PolicyEcology {
    PolicyEcology::new(
        1,
        vec![PolicyRule::DenyReplayIfDecisionClass {
            class: DecisionKind::DecisionKindDeny as u16,
        }],
        PolicyWeights,
    )
}

fn frame_decision_kind(frame: &ControlFrame) -> DecisionKind {
    frame
        .decision
        .as_ref()
        .and_then(|decision| DecisionKind::try_from(decision.kind).ok())
        .expect("minimal spine fixture carries a valid policy decision kind")
}

fn control_frame_roundtrips(frame: &ControlFrame) -> Result<(), DecodeError> {
    let input_bytes = canonical_bytes(frame);
    let decoded = ControlFrame::decode_canonical(&input_bytes)?;
    assert_eq!(&decoded, frame);
    Ok(())
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn archive_payload_commit(bytes: &[u8]) -> Digest32 {
    Digest32::new(*blake3::hash(bytes).as_bytes())
}

fn digest_vec(digest: Digest32) -> Vec<u8> {
    digest.as_bytes().to_vec()
}

fn minimal_spine_policy_record(frame: &ControlFrame, input_digest: Digest32) -> ExperienceRecord {
    let payload = format!(
        "minimal_spine=v1;frame_id={};decision_kind={};input_digest={};",
        frame.frame_id,
        frame_decision_kind(frame) as u16,
        hex::encode(input_digest.as_bytes())
    )
    .into_bytes();

    build_compact_record(
        format!("minimal-spine-policy-{}", frame.frame_id),
        FIXED_OBSERVED_AT_MS,
        "minimal-spine-policy-gate",
        payload,
    )
}

fn evaluate_minimal_spine_policy(
    policy: &PolicyEcology,
    policy_record: &ExperienceRecord,
) -> MinimalSpinePolicyDecision {
    if policy.allow_replay(policy_record) {
        MinimalSpinePolicyDecision::Allow {
            reason: MINIMAL_SPINE_ALLOW_REASON,
        }
    } else {
        MinimalSpinePolicyDecision::Deny {
            reason: MINIMAL_SPINE_DENY_REASON,
        }
    }
}

fn minimal_spine_candidate_payload(
    frame: &ControlFrame,
    input_digest: Digest32,
    policy_decision: &MinimalSpinePolicyDecision,
) -> Vec<u8> {
    format!(
        "minimal_spine_candidate=v1;frame_id={};input_digest={};policy={policy_decision:?};no_real_compute=true;",
        frame.frame_id,
        hex::encode(input_digest.as_bytes())
    )
    .into_bytes()
}

fn minimal_spine_candidate_set_record(
    frame: &ControlFrame,
    input_digest: Digest32,
    policy_decision: &MinimalSpinePolicyDecision,
) -> CandidateSetRecord {
    let policy_decision_bytes = format!("{policy_decision:?}").into_bytes();
    let policy_decision_digest = digest_bytes(
        b"ucf.minimal_spine.policy_decision.v1",
        &policy_decision_bytes,
    );
    let candidate_payload = minimal_spine_candidate_payload(frame, input_digest, policy_decision);
    let candidate_digest = digest_bytes(b"ucf.minimal_spine.candidate.v1", &candidate_payload);
    let candidates_digest = digest_bytes(
        b"ucf.minimal_spine.candidates.v1",
        candidate_digest.as_bytes(),
    );

    CandidateSetRecord {
        version: 1,
        input_digest: digest_vec(input_digest),
        policy_decision_digest: digest_vec(policy_decision_digest),
        candidate_count: 1,
        candidate_digests: vec![digest_vec(candidate_digest)],
        candidates_digest: digest_vec(candidates_digest),
        provenance: "minimal-spine-v1-test-fixture-no-real-compute".to_string(),
    }
}

fn minimal_spine_output_record(
    input_digest: Digest32,
    candidate_set_digest: Digest32,
    candidate_set: &CandidateSetRecord,
) -> OutputRecord {
    let selected_candidate_digest = candidate_set
        .candidate_digests
        .first()
        .expect("minimal spine allow path has one candidate")
        .clone();
    let mut output_material = Vec::new();
    output_material.extend_from_slice(input_digest.as_bytes());
    output_material.extend_from_slice(candidate_set_digest.as_bytes());
    output_material.extend_from_slice(&selected_candidate_digest);
    output_material.extend_from_slice(b"minimal-spine-output-record-v1");
    let output_digest = digest_bytes(b"ucf.minimal_spine.output_payload.v1", &output_material);

    OutputRecord {
        version: 1,
        input_digest: digest_vec(input_digest),
        candidate_set_digest: digest_vec(candidate_set_digest),
        selected_candidate_digest,
        output_digest: digest_vec(output_digest),
        policy_status: "allow".to_string(),
        status: "materialized-test-output".to_string(),
        provenance: "minimal-spine-v1-test-fixture-no-real-compute".to_string(),
        evidence_id: None,
    }
}

fn minimal_spine_evidence_record(
    frame: &ControlFrame,
    input_digest: Digest32,
    candidate_set_digest: Digest32,
    output_record_digest: Digest32,
) -> ExperienceRecord {
    let payload = format!(
        "minimal_spine_evidence=v1;status=allow;frame_id={};input_digest={};candidate_set_record_digest={};output_record_digest={};",
        frame.frame_id,
        hex::encode(input_digest.as_bytes()),
        hex::encode(candidate_set_digest.as_bytes()),
        hex::encode(output_record_digest.as_bytes())
    )
    .into_bytes();
    let record_id = format!(
        "minimal-spine-evidence-{}",
        &hex::encode(digest_bytes(b"ucf.minimal_spine.evidence_id.v1", &payload).as_bytes())[..16]
    );

    build_compact_record(
        record_id,
        FIXED_OBSERVED_AT_MS,
        "minimal-spine-route-result",
        payload,
    )
}

fn minimal_spine_proof(record: &ExperienceRecord) -> ProofEnvelope {
    let payload = canonical_bytes(record);
    ProofEnvelope {
        envelope_id: format!("minimal-spine-proof-{}", record.record_id),
        payload,
        payload_digest: None,
        vrf_tags: Vec::new(),
        signature_ids: vec!["minimal-spine-v1-fixture".to_string()],
    }
}

fn run_minimal_spine(frame: ControlFrame) -> MinimalSpineRunResult {
    control_frame_roundtrips(&frame).expect("canonical control frame roundtrip");
    let input_bytes = canonical_bytes(&frame);
    let input_digest = digest_bytes(b"ucf.minimal_spine.input.v1", &input_bytes);
    let policy_record = minimal_spine_policy_record(&frame, input_digest);
    let policy = minimal_spine_policy();
    let policy_decision = evaluate_minimal_spine_policy(&policy, &policy_record);
    let archive = InMemoryArchive::new();
    let archive_store = InMemoryArchiveStore::new();
    let mut archive_appender = ArchiveAppender::new();

    // Minimal Spine v1 deny semantics for this E2E fixture: deny happens before route output
    // materialization, evidence append, and archive-store append. A denied run therefore has no
    // candidate set, output record, or hidden archive/evidence mutation.
    if matches!(policy_decision, MinimalSpinePolicyDecision::Deny { .. }) {
        return MinimalSpineRunResult {
            input_bytes,
            input_digest,
            policy_decision,
            candidate_set_record: None,
            candidate_set_bytes: None,
            candidate_set_digest: None,
            output_record: None,
            output_record_bytes: None,
            output_record_digest: None,
            evidence_id: None,
            evidence_record: None,
            evidence_bytes: None,
            evidence_digest: None,
            archive_key: None,
            archive_payload_commit: None,
            archive_append_commit: None,
            archive_root_commit: archive_store.root_commit(),
            archive_readback_output_record_bytes: None,
            evidence_entries_len: archive.list().len(),
            archive_output_records_len: archive_store
                .iter_kind(RecordKind::OutputEvent, None)
                .count(),
        };
    }

    let candidate_set_record =
        minimal_spine_candidate_set_record(&frame, input_digest, &policy_decision);
    let candidate_set_bytes = canonical_bytes(&candidate_set_record);
    let candidate_set_digest = digest_bytes(
        b"ucf.minimal_spine.candidate_set_record.v1",
        &candidate_set_bytes,
    );
    let output_record =
        minimal_spine_output_record(input_digest, candidate_set_digest, &candidate_set_record);
    let output_record_bytes = canonical_bytes(&output_record);
    let output_record_digest =
        digest_bytes(b"ucf.minimal_spine.output_record.v1", &output_record_bytes);
    let evidence_record = minimal_spine_evidence_record(
        &frame,
        input_digest,
        candidate_set_digest,
        output_record_digest,
    );
    let evidence_bytes = canonical_bytes(&evidence_record);
    let evidence_digest = digest_bytes(b"ucf.minimal_spine.evidence_record.v1", &evidence_bytes);
    let proof = minimal_spine_proof(&evidence_record);
    let evidence_id = archive.append_with_proof(evidence_record.clone(), Some(proof.clone()));

    let meta = RecordMeta {
        cycle_id: 1,
        tier: 0,
        flags: 0,
        boundary_commit: output_record_digest,
    };
    let archive_record =
        archive_appender.build_record(RecordKind::OutputEvent, &output_record_bytes, meta);
    let archive_append_commit = archive_store.append(archive_record);
    let archive_readback = archive_store
        .get(archive_record.key)
        .expect("archive-store readback by deterministic key");
    assert_eq!(archive_readback, archive_record);
    assert_eq!(
        archive_readback.payload_commit,
        archive_payload_commit(&output_record_bytes)
    );
    assert_eq!(archive_readback.meta.boundary_commit, output_record_digest);

    let evidence_entries = archive.list();
    assert_eq!(evidence_entries.len(), 1);
    assert_eq!(evidence_entries[0].evidence_id, evidence_id);
    assert_eq!(evidence_entries[0].proof, Some(proof));
    assert_eq!(evidence_entries[0].logical_time.tick, 0);
    assert_eq!(evidence_entries[0].wall_time.unix_ms, FIXED_OBSERVED_AT_MS);

    MinimalSpineRunResult {
        input_bytes,
        input_digest,
        policy_decision,
        candidate_set_record: Some(candidate_set_record),
        candidate_set_bytes: Some(candidate_set_bytes),
        candidate_set_digest: Some(candidate_set_digest),
        output_record: Some(output_record),
        output_record_bytes: Some(output_record_bytes.clone()),
        output_record_digest: Some(output_record_digest),
        evidence_id: Some(evidence_id),
        evidence_record: Some(evidence_record),
        evidence_bytes: Some(evidence_bytes),
        evidence_digest: Some(evidence_digest),
        archive_key: Some(archive_record.key),
        archive_payload_commit: Some(archive_record.payload_commit),
        archive_append_commit: Some(archive_append_commit),
        archive_root_commit: archive_store.root_commit(),
        archive_readback_output_record_bytes: Some(output_record_bytes),
        evidence_entries_len: evidence_entries.len(),
        archive_output_records_len: archive_store
            .iter_kind(RecordKind::OutputEvent, None)
            .count(),
    }
}

#[test]
fn minimal_spine_allow_path_appends_and_reads_back_evidence() {
    let frame = minimal_spine_frame("minimal-spine-allow-1", DecisionKind::DecisionKindAllow);
    let result = run_minimal_spine(frame.clone());

    assert_eq!(frame.frame_id, "minimal-spine-allow-1");
    assert!(!result.input_bytes.is_empty());
    assert_eq!(
        ControlFrame::decode_canonical(&result.input_bytes),
        Ok(frame)
    );
    assert_eq!(
        result.policy_decision,
        MinimalSpinePolicyDecision::Allow {
            reason: MINIMAL_SPINE_ALLOW_REASON,
        }
    );

    let candidate_set_record = result
        .candidate_set_record
        .expect("allow candidate set record");
    let candidate_set_bytes = result
        .candidate_set_bytes
        .expect("allow candidate set bytes");
    let candidate_set_digest = result
        .candidate_set_digest
        .expect("allow candidate set digest");
    assert_eq!(candidate_set_record.version, 1);
    assert_eq!(candidate_set_record.candidate_count, 1);
    assert_eq!(candidate_set_record.candidate_digests.len(), 1);
    assert_eq!(canonical_bytes(&candidate_set_record), candidate_set_bytes);
    assert_ne!(candidate_set_digest, Digest32::new([0; 32]));

    let output_record = result.output_record.expect("allow output record");
    let output_record_bytes = result
        .output_record_bytes
        .expect("allow output record bytes");
    let output_record_digest = result
        .output_record_digest
        .expect("allow output record digest");
    assert_eq!(output_record.version, 1);
    assert_eq!(
        output_record.candidate_set_digest,
        digest_vec(candidate_set_digest)
    );
    assert_eq!(canonical_bytes(&output_record), output_record_bytes);
    assert_ne!(output_record_digest, Digest32::new([0; 32]));

    let evidence_id = result.evidence_id.expect("allow evidence id");
    let evidence_record = result.evidence_record.expect("allow evidence record");
    let evidence_bytes = result.evidence_bytes.expect("allow evidence bytes");
    let evidence_payload = String::from_utf8_lossy(&evidence_record.payload);
    assert_eq!(
        evidence_id,
        EvidenceId::new(evidence_record.record_id.clone())
    );
    assert!(evidence_payload.contains("status=allow"));
    assert!(evidence_payload.contains(&format!(
        "candidate_set_record_digest={}",
        hex::encode(candidate_set_digest.as_bytes())
    )));
    assert!(evidence_payload.contains(&format!(
        "output_record_digest={}",
        hex::encode(output_record_digest.as_bytes())
    )));
    assert_eq!(canonical_bytes(&evidence_record), evidence_bytes);
    assert_eq!(
        Some(output_record_bytes.clone()),
        result.archive_readback_output_record_bytes
    );
    assert_eq!(
        result.archive_payload_commit,
        Some(archive_payload_commit(&output_record_bytes))
    );
    assert_eq!(result.evidence_entries_len, 1);
    assert_eq!(result.archive_output_records_len, 1);
    assert!(result.archive_key.is_some());
    assert!(result.archive_append_commit.is_some());
    assert!(result.archive_root_commit.is_some());
}

#[test]
fn minimal_spine_deny_path_is_explicit_and_safe() {
    let frame = minimal_spine_frame("minimal-spine-deny-1", DecisionKind::DecisionKindDeny);
    let result = run_minimal_spine(frame.clone());

    assert_eq!(
        ControlFrame::decode_canonical(&result.input_bytes),
        Ok(frame)
    );
    assert_eq!(
        result.policy_decision,
        MinimalSpinePolicyDecision::Deny {
            reason: MINIMAL_SPINE_DENY_REASON,
        }
    );
    assert!(result.candidate_set_record.is_none());
    assert!(result.candidate_set_bytes.is_none());
    assert!(result.candidate_set_digest.is_none());
    assert!(result.output_record.is_none());
    assert!(result.output_record_bytes.is_none());
    assert!(result.output_record_digest.is_none());
    assert!(result.evidence_id.is_none());
    assert!(result.evidence_record.is_none());
    assert!(result.evidence_bytes.is_none());
    assert!(result.evidence_digest.is_none());
    assert!(result.archive_key.is_none());
    assert!(result.archive_payload_commit.is_none());
    assert!(result.archive_append_commit.is_none());
    assert!(result.archive_root_commit.is_none());
    assert!(result.archive_readback_output_record_bytes.is_none());
    assert_eq!(result.evidence_entries_len, 0);
    assert_eq!(result.archive_output_records_len, 0);
}

#[test]
fn minimal_spine_allow_path_is_deterministic_across_fresh_runs() {
    let frame = minimal_spine_frame("minimal-spine-replay-1", DecisionKind::DecisionKindAllow);
    let first = run_minimal_spine(frame.clone());
    let second = run_minimal_spine(frame);

    assert_eq!(first.input_bytes, second.input_bytes);
    assert_eq!(first.input_digest, second.input_digest);
    assert_eq!(first.policy_decision, second.policy_decision);
    assert_eq!(first.candidate_set_record, second.candidate_set_record);
    assert_eq!(first.candidate_set_bytes, second.candidate_set_bytes);
    assert_eq!(first.candidate_set_digest, second.candidate_set_digest);
    assert_eq!(first.output_record, second.output_record);
    assert_eq!(first.output_record_bytes, second.output_record_bytes);
    assert_eq!(first.output_record_digest, second.output_record_digest);
    assert_eq!(first.evidence_id, second.evidence_id);
    assert_eq!(first.evidence_record, second.evidence_record);
    assert_eq!(first.evidence_bytes, second.evidence_bytes);
    assert_eq!(first.evidence_digest, second.evidence_digest);
    assert_eq!(first.archive_key, second.archive_key);
    assert_eq!(first.archive_payload_commit, second.archive_payload_commit);
    assert_eq!(first.archive_append_commit, second.archive_append_commit);
    assert_eq!(first.archive_root_commit, second.archive_root_commit);
    assert_eq!(
        first.archive_readback_output_record_bytes,
        second.archive_readback_output_record_bytes
    );
    assert_eq!(first.evidence_entries_len, 1);
    assert_eq!(second.evidence_entries_len, 1);
    assert_eq!(first.archive_output_records_len, 1);
    assert_eq!(second.archive_output_records_len, 1);
}
