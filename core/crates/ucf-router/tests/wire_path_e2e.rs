use std::sync::{Arc, Mutex};

use prost::Message;
use ucf_ai_port::{
    AiPillars, MockAiPort, OutputSuppressed, OutputSuppressionSink, PolicySpeechGate,
};
use ucf_archive::InMemoryArchive;
use ucf_attn_controller::{AttentionEventSink, AttentionUpdated};
use ucf_cde_port::MockCdePort;
use ucf_digital_brain::InMemoryDigitalBrain;
use ucf_nsr_port::{NsrBackend, NsrPort, NsrReport};
use ucf_policy_ecology::{PolicyEcology, PolicyRule, PolicyWeights};
use ucf_policy_gateway::NoOpPolicyEvaluator;
use ucf_risk_gate::PolicyRiskGate;
use ucf_router::Router;
use ucf_sandbox::normalize;
use ucf_scm_port::{CausalNode, CounterfactualQuery, CounterfactualResult, ScmDag, ScmPort};
use ucf_tom_port::{
    ActorProfile, IntentHypothesis, IntentType, KnowledgeGap, SocialRiskSignals, TomPort, TomReport,
};
use ucf_types::v1::spec::ExperienceRecord;
use ucf_types::v1::spec::{ActionCode, ControlFrame, DecisionKind, PolicyDecision};
use ucf_types::EvidenceId;

#[derive(Clone, Default)]
struct CaptureSuppression {
    events: Arc<Mutex<Vec<OutputSuppressed>>>,
}

impl OutputSuppressionSink for CaptureSuppression {
    fn publish(&self, event: OutputSuppressed) {
        if let Ok(mut guard) = self.events.lock() {
            guard.push(event);
        }
    }
}

#[derive(Clone, Default)]
struct CaptureAttention {
    events: Arc<Mutex<Vec<AttentionUpdated>>>,
}

impl AttentionEventSink for CaptureAttention {
    fn publish(&self, event: AttentionUpdated) {
        if let Ok(mut guard) = self.events.lock() {
            guard.push(event);
        }
    }
}

#[derive(Clone)]
struct LowRiskTomPort;

impl TomPort for LowRiskTomPort {
    fn analyze(
        &self,
        cf: &ucf_sandbox::ControlFrameNormalized,
        _outputs: &[ucf_types::AiOutput],
    ) -> TomReport {
        TomReport {
            actors: vec![ActorProfile {
                id: 1,
                label: "actor-1".to_string(),
            }],
            intent: IntentHypothesis {
                intent: IntentType::Unknown,
                confidence: 0,
            },
            gaps: vec![KnowledgeGap {
                topic: "context".to_string(),
                uncertainty: 0,
            }],
            risk: SocialRiskSignals {
                deception_likelihood: 0,
                consent_uncertainty: 0,
                manipulation_risk: 0,
                overall: 0,
            },
            commit: cf.commitment().digest,
        }
    }
}

fn allow_speech_policy() -> PolicyEcology {
    PolicyEcology::new(
        1,
        vec![PolicyRule::AllowExternalSpeechIfDecisionClass {
            class: DecisionKind::DecisionKindAllow as u16,
        }],
        PolicyWeights,
    )
}

fn decision_frame(frame_id: &str) -> ControlFrame {
    ControlFrame {
        frame_id: frame_id.to_string(),
        issued_at_ms: 1_700_000_000_000,
        decision: Some(PolicyDecision {
            kind: DecisionKind::DecisionKindAllow as i32,
            action: ActionCode::ActionCodeContinue as i32,
            rationale: "ok".to_string(),
            confidence_bp: 1000,
            constraint_ids: Vec::new(),
        }),
        evidence_ids: Vec::new(),
        policy_id: "policy-1".to_string(),
    }
}

#[test]
fn handle_control_frame_routes_end_to_end() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let brain = Arc::new(InMemoryDigitalBrain::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive.clone(),
        Some(brain.clone()),
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
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
    assert_eq!(archive.list().len(), 2);
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

#[test]
fn risk_gate_denies_speech_when_nsr_not_ok() {
    struct DenyNsr;
    impl NsrBackend for DenyNsr {
        fn check(&self, _claims: &ucf_types::SymbolicClaims) -> NsrReport {
            NsrReport {
                ok: false,
                violations: vec!["violation".to_string()],
            }
        }
    }

    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
        nsr: Some(Arc::new(NsrPort::new(Arc::new(DenyNsr)))),
        ..AiPillars::default()
    }));
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let suppression = CaptureSuppression::default();
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive.clone(),
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        Some(Arc::new(suppression.clone())),
    );

    let outcome = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    assert!(outcome.speech_outputs.is_empty());
    let events = suppression.events.lock().expect("suppression events");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].channel, ucf_types::OutputChannel::Speech);
    assert!(events[0].risk > 0);

    let envelope = &archive.list()[0];
    let proof = envelope.proof.as_ref().expect("missing proof envelope");
    let record =
        ExperienceRecord::decode(proof.payload.as_slice()).expect("decode experience record");
    let payload_text = String::from_utf8(record.payload.clone()).expect("payload utf8");
    assert!(payload_text.contains("output_suppressed="));
}

#[test]
fn risk_gate_denies_speech_on_unsafe_scm_probe() {
    struct LargeDagScm {
        dag: ScmDag,
    }

    impl ScmPort for LargeDagScm {
        fn update(
            &mut self,
            _obs: &ucf_types::WorldStateVec,
            _hint: Option<&ucf_cde_port::CdeHypothesis>,
        ) -> ScmDag {
            self.dag.clone()
        }

        fn counterfactual(&self, _q: &CounterfactualQuery) -> CounterfactualResult {
            CounterfactualResult {
                predicted: 0,
                confidence: 9000,
            }
        }
    }

    let nodes = (0..150)
        .map(|id| CausalNode::new(id, format!("node-{id}")))
        .collect::<Vec<_>>();
    let dag = ScmDag::new(nodes, Vec::new());

    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
        cde: Some(Arc::new(MockCdePort::new())),
        scm: Some(Arc::new(Mutex::new(LargeDagScm { dag }))),
        nsr: Some(Arc::new(NsrPort::default())),
        ..AiPillars::default()
    }));
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    );

    let outcome = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    assert!(outcome.speech_outputs.is_empty());
}

#[test]
fn risk_gate_permits_speech_when_risk_is_low() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
        nsr: Some(Arc::new(NsrPort::default())),
        ..AiPillars::default()
    }));
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    );

    let outcome = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    assert_eq!(outcome.speech_outputs.len(), 1);
}

#[test]
fn attention_event_is_emitted() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let capture = CaptureAttention::default();
    let router = Router::new(
        policy,
        archive,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_attention_sink(Arc::new(capture.clone()));

    let _ = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    let events = capture.events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert!(events[0].gain > 0);
}
