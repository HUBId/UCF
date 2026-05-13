#![forbid(unsafe_code)]

use ucf::v1::spec::{
    ActionCode, ControlFrame, DecisionKind, ExperienceRecord, PolicyDecision, ProofEnvelope,
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct MinimalSpineOutputCandidate {
    id: Digest32,
    payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
struct MinimalSpineRunResult {
    input_bytes: Vec<u8>,
    input_digest: Digest32,
    policy_decision: MinimalSpinePolicyDecision,
    output_candidate: Option<MinimalSpineOutputCandidate>,
    evidence_id: Option<EvidenceId>,
    evidence_record: Option<ExperienceRecord>,
    evidence_bytes: Option<Vec<u8>>,
    evidence_digest: Option<Digest32>,
    archive_key: Option<Digest32>,
    archive_payload_commit: Option<Digest32>,
    archive_append_commit: Option<Digest32>,
    archive_root_commit: Option<Digest32>,
    archive_readback_bytes: Option<Vec<u8>>,
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

fn minimal_spine_output_candidate(
    frame: &ControlFrame,
    input_digest: Digest32,
    policy_decision: &MinimalSpinePolicyDecision,
) -> MinimalSpineOutputCandidate {
    let payload = format!(
        "minimal_spine_output=v1;frame_id={};input_digest={};policy={policy_decision:?};",
        frame.frame_id,
        hex::encode(input_digest.as_bytes())
    )
    .into_bytes();
    let id = digest_bytes(b"ucf.minimal_spine.output_candidate.v1", &payload);
    MinimalSpineOutputCandidate { id, payload }
}

fn minimal_spine_evidence_record(
    frame: &ControlFrame,
    input_digest: Digest32,
    candidate: &MinimalSpineOutputCandidate,
) -> ExperienceRecord {
    let payload = format!(
        "minimal_spine_evidence=v1;status=allow;frame_id={};input_digest={};output_candidate={};",
        frame.frame_id,
        hex::encode(input_digest.as_bytes()),
        hex::encode(candidate.id.as_bytes())
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
    // output candidate and no hidden archive/evidence mutation.
    if matches!(policy_decision, MinimalSpinePolicyDecision::Deny { .. }) {
        return MinimalSpineRunResult {
            input_bytes,
            input_digest,
            policy_decision,
            output_candidate: None,
            evidence_id: None,
            evidence_record: None,
            evidence_bytes: None,
            evidence_digest: None,
            archive_key: None,
            archive_payload_commit: None,
            archive_append_commit: None,
            archive_root_commit: archive_store.root_commit(),
            archive_readback_bytes: None,
            evidence_entries_len: archive.list().len(),
            archive_output_records_len: archive_store
                .iter_kind(RecordKind::OutputEvent, None)
                .count(),
        };
    }

    let output_candidate = minimal_spine_output_candidate(&frame, input_digest, &policy_decision);
    let evidence_record = minimal_spine_evidence_record(&frame, input_digest, &output_candidate);
    let evidence_bytes = canonical_bytes(&evidence_record);
    let evidence_digest = digest_bytes(b"ucf.minimal_spine.evidence_record.v1", &evidence_bytes);
    let proof = minimal_spine_proof(&evidence_record);
    let evidence_id = archive.append_with_proof(evidence_record.clone(), Some(proof.clone()));

    let meta = RecordMeta {
        cycle_id: 1,
        tier: 0,
        flags: 0,
        boundary_commit: input_digest,
    };
    let archive_record =
        archive_appender.build_record(RecordKind::OutputEvent, &evidence_bytes, meta);
    let archive_append_commit = archive_store.append(archive_record);
    let archive_readback = archive_store
        .get(archive_record.key)
        .expect("archive-store readback by deterministic key");
    assert_eq!(archive_readback, archive_record);

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
        output_candidate: Some(output_candidate),
        evidence_id: Some(evidence_id),
        evidence_record: Some(evidence_record),
        evidence_bytes: Some(evidence_bytes.clone()),
        evidence_digest: Some(evidence_digest),
        archive_key: Some(archive_record.key),
        archive_payload_commit: Some(archive_record.payload_commit),
        archive_append_commit: Some(archive_append_commit),
        archive_root_commit: archive_store.root_commit(),
        archive_readback_bytes: Some(evidence_bytes),
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

    let output_candidate = result.output_candidate.expect("allow output candidate");
    assert!(!output_candidate.payload.is_empty());
    assert_ne!(output_candidate.id, Digest32::new([0; 32]));

    let evidence_id = result.evidence_id.expect("allow evidence id");
    let evidence_record = result.evidence_record.expect("allow evidence record");
    let evidence_bytes = result.evidence_bytes.expect("allow evidence bytes");
    assert_eq!(
        evidence_id,
        EvidenceId::new(evidence_record.record_id.clone())
    );
    assert!(String::from_utf8_lossy(&evidence_record.payload).contains("status=allow"));
    assert_eq!(canonical_bytes(&evidence_record), evidence_bytes);
    assert_eq!(Some(evidence_bytes.clone()), result.archive_readback_bytes);
    assert_eq!(result.evidence_entries_len, 1);
    assert_eq!(result.archive_output_records_len, 1);
    assert!(result.archive_key.is_some());
    assert!(result.archive_payload_commit.is_some());
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
    assert!(result.output_candidate.is_none());
    assert!(result.evidence_id.is_none());
    assert!(result.evidence_record.is_none());
    assert!(result.evidence_bytes.is_none());
    assert!(result.evidence_digest.is_none());
    assert!(result.archive_key.is_none());
    assert!(result.archive_payload_commit.is_none());
    assert!(result.archive_append_commit.is_none());
    assert!(result.archive_root_commit.is_none());
    assert!(result.archive_readback_bytes.is_none());
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
    assert_eq!(first.output_candidate, second.output_candidate);
    assert_eq!(first.evidence_id, second.evidence_id);
    assert_eq!(first.evidence_record, second.evidence_record);
    assert_eq!(first.evidence_bytes, second.evidence_bytes);
    assert_eq!(first.evidence_digest, second.evidence_digest);
    assert_eq!(first.archive_key, second.archive_key);
    assert_eq!(first.archive_payload_commit, second.archive_payload_commit);
    assert_eq!(first.archive_append_commit, second.archive_append_commit);
    assert_eq!(first.archive_root_commit, second.archive_root_commit);
    assert_eq!(first.archive_readback_bytes, second.archive_readback_bytes);
    assert_eq!(first.evidence_entries_len, 1);
    assert_eq!(second.evidence_entries_len, 1);
    assert_eq!(first.archive_output_records_len, 1);
    assert_eq!(second.archive_output_records_len, 1);
}
