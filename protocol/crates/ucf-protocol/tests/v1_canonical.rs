use ucf::{canonical_bytes, v1};

fn sample_control_frame(
    evidence_ids: Vec<&str>,
    constraint_ids: Vec<&str>,
) -> v1::spec::ControlFrame {
    let decision = v1::spec::PolicyDecision {
        kind: v1::spec::DecisionKind::DecisionKindAllow as i32,
        action: v1::spec::ActionCode::ActionCodeContinue as i32,
        rationale: "ok".to_string(),
        confidence_bp: 8_500,
        constraint_ids: constraint_ids.into_iter().map(String::from).collect(),
    };

    v1::spec::ControlFrame {
        frame_id: "frame-1".to_string(),
        issued_at_ms: 1_700_000_000_000,
        decision: Some(decision),
        evidence_ids: evidence_ids.into_iter().map(String::from).collect(),
        policy_id: "policy-1".to_string(),
    }
}

fn sample_candidate_set_record() -> v1::spec::CandidateSetRecord {
    v1::spec::CandidateSetRecord {
        version: 1,
        input_digest: vec![1u8; 32],
        policy_decision_digest: vec![2u8; 32],
        candidate_count: 1,
        candidate_digests: vec![vec![3u8; 32]],
        candidates_digest: vec![4u8; 32],
        provenance: "minimal-spine-v1-test-fixture".to_string(),
    }
}

fn sample_output_record() -> v1::spec::OutputRecord {
    v1::spec::OutputRecord {
        version: 1,
        input_digest: vec![1u8; 32],
        candidate_set_digest: vec![4u8; 32],
        selected_candidate_digest: vec![3u8; 32],
        output_digest: vec![5u8; 32],
        policy_status: "allow".to_string(),
        status: "materialized".to_string(),
        provenance: "minimal-spine-v1-test-fixture".to_string(),
        evidence_id: Some("evidence-1".to_string()),
    }
}

fn sample_record() -> v1::spec::ExperienceRecord {
    v1::spec::ExperienceRecord {
        record_id: "rec-1".to_string(),
        observed_at_ms: 1_700_000_000_001,
        subject_id: "subject-1".to_string(),
        payload: vec![1, 2, 3],
        digest: Some(v1::spec::Digest {
            algorithm: "blake3-256".to_string(),
            value: vec![9, 9, 9],
            algo_id: Some(1),
            domain: Some(7),
            value_32: Some(vec![7u8; 32]),
        }),
        vrf_tag: Some(v1::spec::VrfTag {
            algorithm: "vrf-1".to_string(),
            proof: vec![4, 5, 6],
            output: vec![7, 8, 9],
            suite_id: Some(2),
            domain: Some(3),
            tag: Some(vec![8u8; 32]),
        }),
        proof_ref: Some(v1::spec::ProofRef {
            proof_id: "proof-1".to_string(),
            algo_id: Some(1),
            suite_id: Some(2),
            opaque: Some(vec![3, 3, 3]),
        }),
    }
}

#[test]
fn canonical_encoding_is_deterministic() {
    let frame = sample_control_frame(vec!["e1", "e2"], vec!["c1", "c2"]);
    let first = canonical_bytes(&frame);
    let second = canonical_bytes(&frame);
    assert_eq!(first, second);
}

#[test]
fn normalized_control_frames_match() {
    let frame_a = sample_control_frame(vec!["e1", "e2"], vec!["c1", "c2"]);
    let frame_b = sample_control_frame(vec!["e2", "e1"], vec!["c2", "c1"]);

    let normalized_a = v1::spec::ControlFrameNormalized::from(frame_a);
    let normalized_b = v1::spec::ControlFrameNormalized::from(frame_b);

    assert_eq!(
        canonical_bytes(&normalized_a),
        canonical_bytes(&normalized_b)
    );
}

#[test]
fn canonical_roundtrip_control_frame() {
    let frame = sample_control_frame(vec!["e1", "e2"], vec!["c1", "c2"]);
    let bytes = canonical_bytes(&frame);
    let decoded = v1::spec::ControlFrame::decode_canonical(&bytes).expect("decode");
    assert_eq!(frame, decoded);
}

#[test]
fn canonical_roundtrip_experience_record() {
    let record = sample_record();
    let bytes = canonical_bytes(&record);
    let decoded = v1::spec::ExperienceRecord::decode_canonical(&bytes).expect("decode");
    assert_eq!(record, decoded);
}

#[test]
fn candidate_set_record_canonical_encoding_is_stable() {
    let record = sample_candidate_set_record();
    let first = canonical_bytes(&record);
    let second = canonical_bytes(&record);

    assert_eq!(first, second);
    assert!(!first.is_empty());
}

#[test]
fn output_record_canonical_encoding_is_stable() {
    let record = sample_output_record();
    let first = canonical_bytes(&record);
    let second = canonical_bytes(&record);

    assert_eq!(first, second);
    assert!(!first.is_empty());
}

#[test]
fn candidate_set_and_output_records_prost_roundtrip() {
    use prost::Message;

    let candidate_set = sample_candidate_set_record();
    let candidate_set_bytes = candidate_set.encode_to_vec();
    let decoded_candidate_set =
        v1::spec::CandidateSetRecord::decode(candidate_set_bytes.as_slice()).expect("decode");
    assert_eq!(candidate_set, decoded_candidate_set);

    let output = sample_output_record();
    let output_bytes = output.encode_to_vec();
    let decoded_output = v1::spec::OutputRecord::decode(output_bytes.as_slice()).expect("decode");
    assert_eq!(output, decoded_output);
}
