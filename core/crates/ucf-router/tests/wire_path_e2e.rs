use std::sync::{Arc, Mutex};

use prost::Message;
use ucf_ai_port::{
    AiPillars, MockAiPort, OutputSuppressed, OutputSuppressionSink, PolicySpeechGate,
};
use ucf_archive::InMemoryArchive;
use ucf_archive_store::InMemoryArchiveStore;
use ucf_attn_controller::{AttentionEventSink, AttentionUpdated};
use ucf_bluebrain_port::MockBlueBrainPort;
use ucf_cde_port::MockCdePort;
use ucf_digital_brain::InMemoryDigitalBrain;
use ucf_nsr_port::{
    NsrBackend, NsrInput, NsrPort, NsrReport, NsrStubBackend, NsrVerdict, NsrViolation,
};
use ucf_policy_ecology::{PolicyEcology, PolicyRule, PolicyWeights};
use ucf_policy_gateway::NoOpPolicyEvaluator;
use ucf_risk_gate::PolicyRiskGate;
use ucf_router::Router;
use ucf_sandbox::{
    normalize, AiCallRequest, AiCallResult, IntentSummary, SandboxPort, SandboxReport,
    SandboxVerdict,
};
use ucf_scm_port::{CausalNode, CounterfactualQuery, CounterfactualResult, ScmDag, ScmPort};
use ucf_tcf_port::{CyclePlan, Pulse, PulseKind, TcfPort, TcfState};
use ucf_tom_port::{
    ActorProfile, IntentHypothesis, IntentType, KnowledgeGap, SocialRiskSignals, TomPort, TomReport,
};
use ucf_types::v1::spec::ExperienceRecord;
use ucf_types::v1::spec::{ActionCode, ControlFrame, DecisionKind, PolicyDecision};
use ucf_types::{Digest32, EvidenceId};

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

#[derive(Clone, Default)]
struct TraceStages {
    stages: Arc<Mutex<Vec<PulseKind>>>,
}

impl ucf_router::StageTrace for TraceStages {
    fn record(&self, stage: PulseKind) {
        if let Ok(mut guard) = self.stages.lock() {
            guard.push(stage);
        }
    }
}

struct FixedTcf {
    plan: CyclePlan,
    state: TcfState,
}

impl FixedTcf {
    fn new() -> Self {
        let pulses = vec![
            Pulse {
                kind: PulseKind::Think,
                weight: 5000,
                slot: 0,
            },
            Pulse {
                kind: PulseKind::Sense,
                weight: 4500,
                slot: 1,
            },
            Pulse {
                kind: PulseKind::Verify,
                weight: 4000,
                slot: 2,
            },
            Pulse {
                kind: PulseKind::Consolidate,
                weight: 3500,
                slot: 3,
            },
            Pulse {
                kind: PulseKind::Broadcast,
                weight: 3000,
                slot: 4,
            },
        ];
        Self {
            plan: CyclePlan {
                cycle_id: 7,
                pulses,
                commit: ucf_types::Digest32::new([3u8; 32]),
            },
            state: TcfState {
                phase: ucf_tcf_port::Phase { q: 0 },
                energy: 0,
                commit: ucf_types::Digest32::new([0u8; 32]),
            },
        }
    }
}

impl TcfPort for FixedTcf {
    fn step(
        &mut self,
        _attn: &ucf_attn_controller::AttentionWeights,
        _surprise: Option<&ucf_predictive_coding::SurpriseSignal>,
    ) -> CyclePlan {
        self.plan.clone()
    }

    fn state(&self) -> &TcfState {
        &self.state
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
fn brain_response_updates_pending_delta() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive,
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_tcf_port(Box::new(FixedTcf::new()))
    .with_bluebrain_port(Box::new(MockBlueBrainPort::new()));

    router
        .handle_control_frame(normalize(decision_frame("brain-1")))
        .expect("route frame");

    let snapshot = router
        .last_workspace_snapshot()
        .expect("workspace snapshot");
    assert!(snapshot
        .broadcast
        .iter()
        .any(|signal| signal.summary.contains("BRAIN_RESP")));
    assert!(router.pending_neuromod_delta().is_some());
}

#[test]
fn handle_control_frame_routes_end_to_end() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let brain = Arc::new(InMemoryDigitalBrain::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive.clone(),
        archive_store,
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
    assert_eq!(archive.list().len(), 10);
    assert_eq!(brain.records().len(), 1);

    let record = archive
        .list()
        .iter()
        .find_map(|envelope| {
            let proof = envelope.proof.as_ref()?;
            let record = ExperienceRecord::decode(proof.payload.as_slice()).ok()?;
            (record.record_id == "exp-frame-1").then_some(record)
        })
        .expect("experience record");
    let payload_text = String::from_utf8(record.payload.clone()).expect("payload utf8");

    assert_eq!(record.record_id, "exp-frame-1");
    assert_eq!(record.observed_at_ms, frame.issued_at_ms);
    assert_eq!(record.subject_id, frame.policy_id);
    assert!(payload_text.contains("frame_id=frame-1"));
    assert!(payload_text.contains("decision_kind=0"));
    assert!(payload_text.contains("ai_thoughts=ok"));
}

#[test]
fn orchestrator_respects_cycle_plan_ordering() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let trace = TraceStages::default();

    let router = Router::new(
        policy,
        archive,
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_stage_trace(Arc::new(trace.clone()))
    .with_tcf_port(Box::new(FixedTcf::new()));

    let frame = normalize(decision_frame("trace-1"));
    let _ = router
        .handle_control_frame(frame)
        .expect("route control frame");

    let recorded = trace.stages.lock().unwrap().clone();
    let expected = vec![
        PulseKind::Think,
        PulseKind::Sense,
        PulseKind::Verify,
        PulseKind::Consolidate,
        PulseKind::Broadcast,
    ];
    assert_eq!(recorded, expected);
}

#[test]
fn risk_gate_denies_speech_when_nsr_not_ok() {
    struct DenyNsr;
    impl NsrBackend for DenyNsr {
        fn evaluate(&self, _input: &NsrInput) -> NsrReport {
            NsrReport {
                verdict: NsrVerdict::Deny,
                causal_report_commit: Digest32::new([0u8; 32]),
                violations: vec![NsrViolation {
                    code: "NSR_RULE_FORBIDDEN_INTENT".to_string(),
                    detail_digest: Digest32::new([8u8; 32]),
                    severity: 9000,
                    commit: Digest32::new([9u8; 32]),
                }],
                proof_digest: Digest32::new([7u8; 32]),
                commit: Digest32::new([6u8; 32]),
            }
        }
    }

    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
        nsr: Some(Arc::new(NsrPort::default())),
        ..AiPillars::default()
    }));
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let suppression = CaptureSuppression::default();
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive.clone(),
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        Some(Arc::new(suppression.clone())),
    )
    .with_nsr_port(Arc::new(NsrPort::new(Arc::new(DenyNsr))));

    let outcome = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    assert!(outcome.speech_outputs.is_empty());
    let events = suppression.events.lock().expect("suppression events");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].channel, ucf_types::OutputChannel::Speech);
    assert!(events[0].risk > 0);

    let payload_text = archive
        .list()
        .iter()
        .find_map(|envelope| {
            let proof = envelope.proof.as_ref()?;
            let record = ExperienceRecord::decode(proof.payload.as_slice()).ok()?;
            let payload_text = String::from_utf8(record.payload).ok()?;
            payload_text
                .contains("output_suppressed=")
                .then_some(payload_text)
        })
        .expect("suppression payload");
    assert!(payload_text.contains("output_suppressed="));
}

#[test]
fn cde_runs_before_nsr_in_verify() {
    #[derive(Clone)]
    struct CaptureNsr {
        seen: Arc<Mutex<Option<NsrInput>>>,
    }

    impl NsrBackend for CaptureNsr {
        fn evaluate(&self, input: &NsrInput) -> NsrReport {
            if let Ok(mut guard) = self.seen.lock() {
                *guard = Some(input.clone());
            }
            NsrReport {
                verdict: NsrVerdict::Ok,
                causal_report_commit: input.causal_report_commit,
                violations: Vec::new(),
                proof_digest: Digest32::new([0u8; 32]),
                commit: input.commit,
            }
        }
    }

    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let capture = CaptureNsr {
        seen: Arc::new(Mutex::new(None)),
    };
    let router = Router::new(
        policy,
        archive,
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_tcf_port(Box::new(FixedTcf::new()))
    .with_nsr_port(Arc::new(NsrPort::new(Arc::new(capture.clone()))));

    let _ = router
        .handle_control_frame(normalize(decision_frame("order-1")))
        .expect("route frame");

    let input = capture
        .seen
        .lock()
        .expect("lock input")
        .clone()
        .expect("nsr input captured");
    assert_ne!(input.causal_report_commit, Digest32::new([0u8; 32]));
    assert_eq!(input.counterfactuals.len(), 2);
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
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
        cde: Some(Arc::new(MockCdePort::new())),
        scm: Some(Arc::new(Mutex::new(LargeDagScm { dag }))),
        nsr: Some(Arc::new(NsrPort::default())),
        ..AiPillars::default()
    }));
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let nsr_backend = Arc::new(NsrStubBackend::new());
    let router = Router::new(
        policy,
        archive,
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_nsr_port(Arc::new(NsrPort::new(nsr_backend)));

    let outcome = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    assert!(outcome.speech_outputs.is_empty());
}

#[test]
fn sandbox_allows_speech_outputs() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let nsr_backend = Arc::new(NsrStubBackend::new());
    let router = Router::new(
        policy,
        archive,
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_nsr_port(Arc::new(NsrPort::new(nsr_backend)));

    let outcome = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    assert_eq!(outcome.speech_outputs.len(), 1);
    assert!(outcome.speech_outputs.iter().all(|out| {
        matches!(out.channel, ucf_types::OutputChannel::Speech) && out.content == "ok"
    }));
}

#[test]
fn sandbox_denied_blocks_external_speech() {
    #[derive(Default)]
    struct DenySandbox;

    impl SandboxPort for DenySandbox {
        fn evaluate_call(
            &mut self,
            _cf: &ucf_sandbox::ControlFrameNormalized,
            _intent: &IntentSummary,
            _req: &AiCallRequest,
        ) -> SandboxReport {
            SandboxReport {
                verdict: SandboxVerdict::Deny {
                    reason: "BUDGET_EXCEEDED".to_string(),
                },
                ops_used: 9001,
                commit: ucf_types::Digest32::new([9u8; 32]),
            }
        }

        fn run_ai(&mut self, _req: &AiCallRequest) -> Result<AiCallResult, SandboxReport> {
            Err(SandboxReport {
                verdict: SandboxVerdict::Deny {
                    reason: "BUDGET_EXCEEDED".to_string(),
                },
                ops_used: 9001,
                commit: ucf_types::Digest32::new([9u8; 32]),
            })
        }
    }

    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive,
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_sandbox_port(Box::new(DenySandbox));

    let outcome = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    assert!(outcome.speech_outputs.is_empty());
}

#[test]
fn risk_gate_permits_speech_when_risk_is_low() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
        nsr: Some(Arc::new(NsrPort::default())),
        ..AiPillars::default()
    }));
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let nsr_backend = Arc::new(NsrStubBackend::new());
    let router = Router::new(
        policy,
        archive,
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_nsr_port(Arc::new(NsrPort::new(nsr_backend)));

    let outcome = router
        .handle_control_frame(normalize(decision_frame("ping")))
        .expect("route frame");

    assert_eq!(outcome.speech_outputs.len(), 1);
}

#[test]
fn verify_pulse_emits_causal_report() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let router = Router::new(
        policy,
        archive.clone(),
        archive_store,
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    )
    .with_tcf_port(Box::new(FixedTcf::new()));

    let frame = normalize(decision_frame("causal-1"));
    router
        .handle_control_frame(frame)
        .expect("route control frame");

    let snapshot = router
        .last_workspace_snapshot()
        .expect("workspace snapshot");
    assert!(snapshot
        .broadcast
        .iter()
        .any(|signal| signal.summary.contains("CDE ok")));

    let record = archive
        .list()
        .iter()
        .find_map(|envelope| {
            let proof = envelope.proof.as_ref()?;
            let record = ExperienceRecord::decode(proof.payload.as_slice()).ok()?;
            (record.subject_id == "causal").then_some(record)
        })
        .expect("causal record");
    let payload_text = String::from_utf8(record.payload).expect("payload utf8");
    assert!(payload_text.contains("dag="));
}

#[test]
fn attention_event_is_emitted() {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(allow_speech_policy()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let capture = CaptureAttention::default();
    let router = Router::new(
        policy,
        archive,
        archive_store,
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
