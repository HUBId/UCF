use std::sync::Arc;

use ucf_ai_port::{MockAiPort, PolicySpeechGate};
use ucf_archive::InMemoryArchive;
use ucf_archive_store::{ArchiveStore, InMemoryArchiveStore};
use ucf_nsr_port::{NsrBackend, NsrInput, NsrPort, NsrReport, NsrVerdict};
use ucf_output_router::{GateBundle, NsrSummary, OutputRouter, OutputRouterEvent, RouterConfig};
use ucf_policy_ecology::{PolicyEcology, RiskDecision, RiskGateResult};
use ucf_policy_gateway::NoOpPolicyEvaluator;
use ucf_risk_gate::PolicyRiskGate;
use ucf_router::{Router, PIPELINE};
use ucf_sandbox::{normalize, ControlFrameNormalized, SandboxVerdict};
use ucf_sle::{SleCore, SleInputs};
use ucf_tom_port::{
    ActorProfile, IntentHypothesis, IntentType, KnowledgeGap, SocialRiskSignals, TomPort, TomReport,
};
use ucf_types::v1::spec::{ActionCode, ControlFrame, DecisionKind, PolicyDecision};
use ucf_types::{AiOutput, Digest32, OutputChannel};

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

struct RestrictNsr;

impl NsrBackend for RestrictNsr {
    fn evaluate(&self, _input: &NsrInput) -> NsrReport {
        NsrReport {
            verdict: NsrVerdict::Restrict,
            causal_report_commit: Digest32::new([0u8; 32]),
            violations: Vec::new(),
            proof_digest: Digest32::new([0u8; 32]),
            commit: Digest32::new([1u8; 32]),
        }
    }
}

fn allow_decision(frame_id: &str) -> ControlFrame {
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

fn build_router() -> (Router, Arc<InMemoryArchiveStore>) {
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
        archive_store.clone(),
        None,
        ai_port,
        speech_gate,
        risk_gate,
        tom_port,
        None,
    );
    (router, archive_store)
}

fn run_cycles(
    frames: &[ControlFrameNormalized],
) -> (Digest32, Digest32, Vec<ControlFrameNormalized>) {
    let (router, archive_store) = build_router();
    for frame in frames {
        router
            .handle_control_frame(frame.clone())
            .expect("route control frame");
    }
    let snapshot = router
        .last_workspace_snapshot()
        .expect("workspace snapshot");
    let archive_root = archive_store.root_commit().expect("archive root");
    (snapshot.commit, archive_root, frames.to_vec())
}

#[test]
fn pipeline_order_is_authoritative() {
    let expected = [
        "onn", "spikebus", "coupling", "jepa", "iit", "tcf", "nsr", "sle", "ncde", "ssm", "cde",
        "output", "archive",
    ];
    assert_eq!(PIPELINE, expected);
}

#[test]
fn determinism_smoke_test_matches_workspace_and_archive_roots() {
    let frames: Vec<ControlFrameNormalized> = (0..10)
        .map(|idx| normalize(allow_decision(&format!("determinism-{idx}"))))
        .collect();
    let (workspace_a, archive_a, _) = run_cycles(&frames);
    let (workspace_b, archive_b, _) = run_cycles(&frames);
    assert_eq!(workspace_a, workspace_b);
    assert_eq!(archive_a, archive_b);
}

#[test]
fn thought_only_non_leak_drops_speech_and_forces_stabilize() {
    let mut sle_core = SleCore::default();
    let sle_inputs = SleInputs::new(
        21,
        Digest32::new([9u8; 32]),
        2,
        Digest32::new([10u8; 32]),
        8200,
        4000,
        Digest32::new([11u8; 32]),
        7800,
        Digest32::new([12u8; 32]),
        1,
        Digest32::new([13u8; 32]),
        6200,
        7200,
        false,
        false,
        1000,
        1200,
        9000,
    );
    let sle_outputs = sle_core.tick(&sle_inputs);
    assert_ne!(sle_outputs.thought_only_root, Digest32::new([0u8; 32]));

    let config = RouterConfig {
        thought_capacity: 8,
        max_thought_frames_per_cycle: 8,
        external_enabled: true,
    };
    let mut output_router = OutputRouter::new(config);
    output_router.apply_coherence(0, 2000);

    let decision = PolicyDecision {
        kind: DecisionKind::DecisionKindAllow as i32,
        action: ActionCode::ActionCodeContinue as i32,
        rationale: "ok".to_string(),
        confidence_bp: 1000,
        constraint_ids: Vec::new(),
    };
    let risk_result = RiskGateResult {
        decision: RiskDecision::Permit,
        risk: 0,
        reasons: Vec::new(),
        evidence: Digest32::new([0u8; 32]),
    };
    let gates = GateBundle {
        policy_decision: decision,
        sandbox: SandboxVerdict::Allow,
        risk_results: vec![risk_result],
        nsr_summary: NsrSummary {
            verdict: NsrVerdict::Allow,
            violations_digest: Digest32::new([0u8; 32]),
        },
        speech_gate: vec![true],
        coherence_plv: 0,
        coherence_threshold: 2000,
        phi_proxy: 9000,
        phi_threshold: 3200,
        speak_lock: 0,
        speak_lock_min: 3000,
        damp_output: false,
        output_gain_cap: 10_000,
    };
    let cf = normalize(allow_decision("ping"));
    let outputs = vec![AiOutput {
        channel: OutputChannel::Speech,
        content: "escape".to_string(),
        confidence: 1000,
        rationale_commit: None,
        integration_score: None,
    }];
    let decisions = output_router.route(&cf, outputs, &gates);
    let events = output_router.drain_events();

    assert_eq!(decisions.len(), 1);
    assert!(!decisions[0].permitted);
    assert_eq!(decisions[0].reason_code, "onn_low_coherence");
    assert!(events.iter().any(|event| {
        matches!(
            event,
            OutputRouterEvent::OutputSuppressed {
                reason_code, ..
            } if reason_code == "onn_low_coherence"
        )
    }));

    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::new());
    let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
    let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
    let tom_port = Arc::new(LowRiskTomPort);
    let nsr_port = Arc::new(NsrPort::new(Arc::new(RestrictNsr)));
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
    .with_nsr_port(nsr_port);

    router
        .handle_control_frame(normalize(allow_decision("ping")))
        .expect("route control frame");
    assert!(router.force_stabilize_cycles() > 0);
}
