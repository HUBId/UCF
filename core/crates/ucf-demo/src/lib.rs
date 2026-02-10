#![forbid(unsafe_code)]

use blake3::Hasher;
use std::sync::Arc;
use ucf_ai_port::MockAiPort;
use ucf_archive::InMemoryArchive;
use ucf_archive_store::InMemoryArchiveStore;
use ucf_policy_ecology::PolicyEcology;
use ucf_policy_gateway::NoOpPolicyEvaluator;
use ucf_risk_gate::PolicyRiskGate;
use ucf_router::Router;
use ucf_sandbox::ControlFrameNormalized;
use ucf_tom_port::MockTomPort;
use ucf_types::v1::spec::{ActionCode, ControlFrame, DecisionKind, PolicyDecision};
use ucf_types::Digest32;
use ucf_workspace::CoherenceSummary;

pub fn build_router() -> Router {
    let policy = Arc::new(NoOpPolicyEvaluator::new());
    let archive = Arc::new(InMemoryArchive::new());
    let archive_store = Arc::new(InMemoryArchiveStore::new());
    let ai_port = Arc::new(MockAiPort::default());
    let policy_ecology = PolicyEcology::allow_all();
    let speech_gate = Arc::new(ucf_ai_port::PolicySpeechGate::new(policy_ecology.clone()));
    let risk_gate = Arc::new(PolicyRiskGate::new(policy_ecology));
    let tom_port = Arc::new(MockTomPort::new());

    Router::new(
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
}

pub fn deterministic_control_frame(seed: u64, cycle_id: u64) -> ControlFrameNormalized {
    let external_commit = deterministic_commit(seed, cycle_id, b"external");
    let policy_snapshot_commit = deterministic_commit(seed, cycle_id, b"policy");
    let decision = PolicyDecision {
        kind: DecisionKind::DecisionKindAllow as i32,
        action: ActionCode::ActionCodeContinue as i32,
        rationale: "demo".to_string(),
        confidence_bp: 10_000,
        constraint_ids: vec![
            format!("ext:{external_commit}"),
            format!("pol:{policy_snapshot_commit}"),
        ],
    };

    let cf = ControlFrame {
        frame_id: format!("demo-{seed}-{cycle_id}-{external_commit}"),
        issued_at_ms: cycle_id,
        decision: Some(decision),
        evidence_ids: vec![
            format!("{external_commit}"),
            format!("{policy_snapshot_commit}"),
        ],
        policy_id: format!("demo-policy-{seed}-{policy_snapshot_commit}"),
    };
    ucf_sandbox::normalize(cf)
}

pub fn run_cycles(cycles: u64, seed: u64) -> Vec<CoherenceSummary> {
    let router = build_router();
    let mut summaries = Vec::with_capacity(cycles as usize);

    for cycle_id in 1..=cycles {
        let cf = deterministic_control_frame(seed, cycle_id);
        let _ = router
            .handle_control_frame(cf)
            .expect("demo control frame should be accepted");
        let snapshot = router
            .last_workspace_snapshot()
            .expect("workspace snapshot should be present");
        summaries.push(CoherenceSummary::from_snapshot(&snapshot));
    }

    summaries
}

fn deterministic_commit(seed: u64, cycle_id: u64, domain: &[u8]) -> Digest32 {
    let mut h = Hasher::new();
    h.update(b"ucf.demo.seed.v1");
    h.update(domain);
    h.update(&seed.to_be_bytes());
    h.update(&cycle_id.to_be_bytes());
    Digest32::new(*h.finalize().as_bytes())
}
