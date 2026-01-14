use std::sync::Arc;

use prost::Message;
use ucf_ai_port::{MockAiPort, PolicySpeechGate};
use ucf_archive::InMemoryArchive;
use ucf_digital_brain::InMemoryDigitalBrain;
use ucf_policy_ecology::PolicyEcology;
use ucf_policy_gateway::NoOpPolicyEvaluator;
use ucf_router::Router;
use ucf_sandbox::normalize;
use ucf_types::v1::spec::ExperienceRecord;
use ucf_types::v1::spec::{ControlFrame, DecisionKind};
use ucf_types::EvidenceId;

#[test]
fn handle_control_frame_routes_end_to_end() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let brain = Arc::new(InMemoryDigitalBrain::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
    let router = Router::new(
        policy,
        archive.clone(),
        Some(brain.clone()),
        ai_port,
        speech_gate,
    );

    let frame = ControlFrame {
        frame_id: "frame-1".to_string(),
        issued_at_ms: 1_700_000_000_000,
        decision: None,
        evidence_ids: Vec::new(),
        policy_id: "policy-1".to_string(),
    };

    let outcome = router
        .handle_control_frame(normalize(frame.clone()))
        .expect("route frame");

    assert_eq!(outcome.evidence_id, EvidenceId::new("exp-frame-1"));
    assert_eq!(outcome.decision_kind, DecisionKind::DecisionKindUnspecified);
    assert_eq!(archive.list().len(), 1);
    assert_eq!(brain.records().len(), 1);

    let envelope = &archive.list()[0];
    let proof = envelope.proof.as_ref().expect("missing proof envelope");
    let record =
        ExperienceRecord::decode(proof.payload.as_slice()).expect("decode experience record");
    let payload_text = String::from_utf8(record.payload.clone()).expect("payload utf8");

    assert_eq!(record.record_id, "exp-frame-1");
    assert_eq!(record.observed_at_ms, frame.issued_at_ms);
    assert_eq!(record.subject_id, frame.policy_id);
    assert!(payload_text.contains("frame_id=frame-1"));
    assert!(payload_text.contains("decision_kind=0"));
    assert!(payload_text.contains("ai_thoughts=ok"));
}
