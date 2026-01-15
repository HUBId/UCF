#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex};

use blake3::Hasher;
#[cfg(feature = "ai-runtime")]
use ucf_ai_runtime::backend::{AiRuntimeBackend, NoopRuntimeBackend};
use ucf_cde_port::{CdeHypothesis, CdePort};
use ucf_iit_monitor::IitMonitor;
use ucf_ncde_port::{NcdeContext, NcdePort};
use ucf_nsr_port::{NsrPort, NsrReport};
use ucf_policy_ecology::{PolicyEcology, PolicyRule};
use ucf_sandbox::ControlFrameNormalized;
use ucf_sle::{LoopFrame, StrangeLoopEngine};
use ucf_ssm_port::{SsmPort, SsmState};
use ucf_tcf_port::{TcfPort, TcfSignal};
use ucf_types::v1::spec::DecisionKind;
use ucf_types::{CausalGraphStub, Claim, Digest32, SymbolicClaims, ThoughtVec, WorldStateVec};

pub use ucf_types::{AiOutput, OutputChannel};

pub trait AiPort {
    fn infer(&self, input: &ControlFrameNormalized) -> Vec<AiOutput>;
}

#[derive(Clone, Default)]
pub struct AiPillars {
    pub tcf: Option<Arc<dyn TcfPort + Send + Sync>>,
    pub ssm: Option<Arc<dyn SsmPort + Send + Sync>>,
    pub cde: Option<Arc<dyn CdePort + Send + Sync>>,
    pub nsr: Option<Arc<dyn NsrPort + Send + Sync>>,
    pub ncde: Option<Arc<dyn NcdePort + Send + Sync>>,
    pub sle: Option<Arc<StrangeLoopEngine>>,
    pub iit_monitor: Option<Arc<Mutex<IitMonitor>>>,
}

impl AiPillars {
    fn enabled(&self) -> bool {
        self.tcf.is_some()
            || self.ssm.is_some()
            || self.cde.is_some()
            || self.nsr.is_some()
            || self.ncde.is_some()
            || self.sle.is_some()
            || self.iit_monitor.is_some()
    }
}

#[derive(Clone, Default)]
pub struct MockAiPort {
    pillars: AiPillars,
}

impl MockAiPort {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pillars(pillars: AiPillars) -> Self {
        Self { pillars }
    }
}

#[cfg(feature = "ai-runtime")]
pub fn noop_runtime_backend(mock: MockAiPort) -> NoopRuntimeBackend {
    let delegate = Arc::new(move |cf: &ControlFrameNormalized| mock.infer(cf));
    NoopRuntimeBackend::new(delegate)
}

#[derive(Clone)]
pub struct AiOrchestrator {
    mock: MockAiPort,
    #[cfg(feature = "ai-runtime")]
    runtime_backend: Option<Arc<dyn AiRuntimeBackend + Send + Sync>>,
}

impl AiOrchestrator {
    pub fn new(mock: MockAiPort) -> Self {
        Self {
            mock,
            #[cfg(feature = "ai-runtime")]
            runtime_backend: None,
        }
    }

    pub fn with_pillars(pillars: AiPillars) -> Self {
        Self::new(MockAiPort::with_pillars(pillars))
    }

    #[cfg(feature = "ai-runtime")]
    pub fn with_runtime_backend(
        mut self,
        backend: Arc<dyn AiRuntimeBackend + Send + Sync>,
    ) -> Self {
        self.runtime_backend = Some(backend);
        self
    }

    #[cfg(feature = "ai-runtime")]
    pub fn with_noop_runtime(mut self) -> Self {
        let backend = noop_runtime_backend(self.mock.clone());
        self.runtime_backend = Some(Arc::new(backend));
        self
    }
}

impl Default for AiOrchestrator {
    fn default() -> Self {
        Self::new(MockAiPort::new())
    }
}

impl AiPort for AiOrchestrator {
    fn infer(&self, input: &ControlFrameNormalized) -> Vec<AiOutput> {
        #[cfg(feature = "ai-runtime")]
        if let Some(runtime) = &self.runtime_backend {
            return runtime.infer_runtime(input);
        }
        self.mock.infer(input)
    }
}

impl AiPort for MockAiPort {
    fn infer(&self, input: &ControlFrameNormalized) -> Vec<AiOutput> {
        let artifacts = if self.pillars.enabled() {
            run_pillars(&self.pillars, input)
        } else {
            PillarArtifacts::baseline(input.commitment().digest)
        };
        let mut thought = AiOutput {
            channel: OutputChannel::Thought,
            content: "ok".to_string(),
            confidence: 1000,
            rationale_commit: Some(artifacts.rationale_commit),
            integration_score: None,
        };

        let loop_frame = reflect_sle(
            &self.pillars,
            input.commitment().digest,
            &artifacts,
            &thought,
        );
        if let Some(frame) = loop_frame {
            thought.rationale_commit = Some(frame.report_commit);
        }
        let integration_score = sample_iit_monitor(&self.pillars, &artifacts, &thought);
        if let Some(score) = integration_score {
            thought.integration_score = Some(score);
        }

        let allow_speech = artifacts.nsr_report.as_ref().is_none_or(|report| report.ok);
        if input.as_ref().frame_id == "ping" && allow_speech {
            let speech = AiOutput {
                channel: OutputChannel::Speech,
                content: "ok".to_string(),
                confidence: 1000,
                rationale_commit: thought.rationale_commit,
                integration_score: thought.integration_score,
            };
            vec![thought, speech]
        } else {
            vec![thought]
        }
    }
}

struct PillarArtifacts {
    rationale_commit: Digest32,
    nsr_report: Option<NsrReport>,
    ssm_digest: Option<Digest32>,
    cde_hyp: Option<CdeHypothesis>,
    nsr_digest: Option<Digest32>,
}

impl PillarArtifacts {
    fn baseline(digest: Digest32) -> Self {
        Self {
            rationale_commit: digest,
            nsr_report: None,
            ssm_digest: None,
            cde_hyp: None,
            nsr_digest: None,
        }
    }
}

fn run_pillars(pillars: &AiPillars, input: &ControlFrameNormalized) -> PillarArtifacts {
    let base_digest = input.commitment().digest;
    let mut signal = TcfSignal::new(base_digest, 0);
    let mut ssm_state = SsmState::new(base_digest, 0);
    let mut graph = CausalGraphStub::new(Vec::new(), Vec::new());
    let ctx = NcdeContext::new(base_digest);
    let world_state = WorldStateVec::new(base_digest.as_bytes().to_vec(), vec![Digest32::LEN]);
    let mut thought = ThoughtVec::new(base_digest.as_bytes().to_vec());
    let claims = SymbolicClaims::new(vec![Claim::new_from_strs(
        "frame",
        vec![input.as_ref().frame_id.as_str()],
    )]);

    let mut artifacts = Vec::new();
    let mut nsr_report = None;
    let mut ssm_digest = None;
    let mut cde_hyp = None;
    let mut nsr_digest = None;

    if let Some(monitor) = &pillars.iit_monitor {
        if let Ok(guard) = monitor.lock() {
            let intensity = tcf_intensity_stub(guard.aggregate());
            signal.intensity = intensity;
        }
    }

    if let Some(tcf) = &pillars.tcf {
        let pulse = tcf.tick(&mut signal, &[base_digest]);
        artifacts.push(pulse.digest);
    }
    if let Some(ssm) = &pillars.ssm {
        let output = ssm.update(&mut ssm_state, &world_state);
        artifacts.push(output.digest);
        ssm_digest = Some(output.digest);
    }
    if let Some(cde) = &pillars.cde {
        let hypothesis = cde.infer(&mut graph, &world_state);
        artifacts.push(hypothesis.digest);
        cde_hyp = Some(hypothesis);
    }
    if let Some(nsr) = &pillars.nsr {
        let report = nsr.check(&claims);
        let report_digest = digest_nsr_report(&report);
        artifacts.push(report_digest);
        nsr_report = Some(report);
        nsr_digest = Some(report_digest);
    }
    if let Some(ncde) = &pillars.ncde {
        thought = ncde.integrate(&ctx, &thought);
        artifacts.push(digest_thought(&thought));
    }

    let rationale_commit = if artifacts.is_empty() {
        base_digest
    } else {
        hash_digests(&artifacts)
    };

    PillarArtifacts {
        rationale_commit,
        nsr_report,
        ssm_digest,
        cde_hyp,
        nsr_digest,
    }
}

fn reflect_sle(
    pillars: &AiPillars,
    base_digest: Digest32,
    artifacts: &PillarArtifacts,
    output: &AiOutput,
) -> Option<LoopFrame> {
    pillars.sle.as_ref().map(|sle| {
        let prev = sle.latest().unwrap_or_else(|| LoopFrame::seed(base_digest));
        sle.reflect(
            &prev,
            output,
            artifacts.nsr_report.as_ref(),
            artifacts.cde_hyp.as_ref(),
        )
    })
}

fn sample_iit_monitor(
    pillars: &AiPillars,
    artifacts: &PillarArtifacts,
    output: &AiOutput,
) -> Option<u16> {
    let monitor = pillars.iit_monitor.as_ref()?;
    let mut guard = monitor.lock().ok()?;
    if let (Some(ssm), Some(cde)) = (artifacts.ssm_digest, artifacts.cde_hyp.as_ref()) {
        guard.sample(ssm, cde.digest);
    }
    if let (Some(cde), Some(nsr)) = (artifacts.cde_hyp.as_ref(), artifacts.nsr_digest) {
        guard.sample(cde.digest, nsr);
    }
    if let Some(nsr) = artifacts.nsr_digest {
        let output_digest = digest_ai_output(output);
        guard.sample(nsr, output_digest);
    }
    Some(guard.aggregate())
}

fn digest_ai_output(output: &AiOutput) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ai.output.v1");
    let channel_tag: u8 = match output.channel {
        OutputChannel::Thought => 0,
        OutputChannel::Speech => 1,
    };
    hasher.update(&[channel_tag]);
    hasher.update(&output.confidence.to_be_bytes());
    hasher.update(
        &u64::try_from(output.content.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    hasher.update(output.content.as_bytes());
    match output.rationale_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match output.integration_score {
        Some(score) => {
            hasher.update(&[1]);
            hasher.update(&score.to_be_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn tcf_intensity_stub(score: u16) -> u16 {
    score / 100
}

fn hash_digests(digests: &[Digest32]) -> Digest32 {
    let mut hasher = Hasher::new();
    for digest in digests {
        hasher.update(digest.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_thought(thought: &ThoughtVec) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&thought.bytes);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_nsr_report(report: &NsrReport) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&[report.ok as u8]);
    hasher.update(
        &u64::try_from(report.violations.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for violation in &report.violations {
        hasher.update(violation.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

pub trait SpeechGate {
    fn allow_speech(&self, cf: &ControlFrameNormalized, out: &AiOutput) -> bool;
}

#[derive(Clone, Debug)]
pub struct PolicySpeechGate {
    policy: PolicyEcology,
}

impl PolicySpeechGate {
    pub fn new(policy: PolicyEcology) -> Self {
        Self { policy }
    }

    fn decision_class(cf: &ControlFrameNormalized) -> Option<u16> {
        cf.as_ref()
            .decision
            .as_ref()
            .and_then(|decision| DecisionKind::try_from(decision.kind).ok())
            .map(|kind| kind as u16)
    }

    fn allow_for_class(&self, class: u16) -> bool {
        self.policy.rules().iter().any(|rule| {
            matches!(
                rule,
                PolicyRule::AllowExternalSpeechIfDecisionClass { class: rule_class }
                    if *rule_class == class
            )
        })
    }
}

impl SpeechGate for PolicySpeechGate {
    fn allow_speech(&self, cf: &ControlFrameNormalized, out: &AiOutput) -> bool {
        if out.channel != OutputChannel::Speech {
            return true;
        }

        let Some(class) = Self::decision_class(cf) else {
            return false;
        };

        self.allow_for_class(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ai-runtime")]
    use ucf_ai_runtime::backend::{AiRuntimeBackend, NoopRuntimeBackend};
    use ucf_sandbox::normalize;
    use ucf_types::v1::spec::ControlFrame;
    use ucf_types::SymbolicClaims;
    use ucf_types::ThoughtVec;
    use ucf_types::WorldStateVec;

    use ucf_cde_port::{CdeHypothesis, CdePort, MockCdePort};
    use ucf_iit_monitor::IitMonitor;
    use ucf_ncde_port::{NcdeContext, NcdePort};
    use ucf_nsr_port::{MockNsrPort, NsrPort, NsrReport};
    use ucf_sle::StrangeLoopEngine;
    use ucf_ssm_port::{MockSsmPort, SsmOutput, SsmPort, SsmState};
    use ucf_tcf_port::{TcfPort, TcfPulse, TcfSignal};

    fn base_frame(frame_id: &str) -> ControlFrame {
        ControlFrame {
            frame_id: frame_id.to_string(),
            issued_at_ms: 1_700_000_000_000,
            decision: None,
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        }
    }

    #[test]
    fn mock_ai_port_emits_thought_and_speech_for_ping() {
        let port = MockAiPort::new();
        let normalized = normalize(base_frame("ping"));

        let outputs = port.infer(&normalized);

        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .any(|out| out.channel == OutputChannel::Thought));
        assert!(outputs
            .iter()
            .any(|out| out.channel == OutputChannel::Speech));
        assert!(outputs.iter().all(|out| out.content == "ok"));
    }

    #[test]
    fn mock_ai_port_emits_only_thought_for_other_frames() {
        let port = MockAiPort::new();
        let normalized = normalize(base_frame("frame-1"));

        let outputs = port.infer(&normalized);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].channel, OutputChannel::Thought);
        assert_eq!(outputs[0].content, "ok");
    }

    #[test]
    fn pipeline_order_is_deterministic() {
        let order: Arc<std::sync::Mutex<Vec<&'static str>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        struct OrderTcf {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl TcfPort for OrderTcf {
            fn tick(&self, _signal: &mut TcfSignal, _inputs: &[Digest32]) -> TcfPulse {
                self.order.lock().unwrap().push("tcf");
                TcfPulse {
                    digest: Digest32::new([1u8; 32]),
                    tick: 1,
                    inputs: 0,
                }
            }
        }

        struct OrderSsm {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl SsmPort for OrderSsm {
            fn update(&self, _state: &mut SsmState, _input: &WorldStateVec) -> SsmOutput {
                self.order.lock().unwrap().push("ssm");
                SsmOutput {
                    digest: Digest32::new([2u8; 32]),
                    state_digest: Digest32::new([3u8; 32]),
                    step: 1,
                }
            }
        }

        struct OrderCde {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl CdePort for OrderCde {
            fn infer(&self, _graph: &mut CausalGraphStub, _obs: &WorldStateVec) -> CdeHypothesis {
                self.order.lock().unwrap().push("cde");
                CdeHypothesis {
                    digest: Digest32::new([4u8; 32]),
                    nodes: 0,
                    edges: 0,
                }
            }
        }

        struct OrderNsr {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl NsrPort for OrderNsr {
            fn check(&self, _claims: &SymbolicClaims) -> NsrReport {
                self.order.lock().unwrap().push("nsr");
                NsrReport {
                    ok: true,
                    violations: Vec::new(),
                }
            }
        }

        struct OrderNcde {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl NcdePort for OrderNcde {
            fn integrate(&self, _ctx: &NcdeContext, _control: &ThoughtVec) -> ThoughtVec {
                self.order.lock().unwrap().push("ncde");
                ThoughtVec::new(vec![1, 2, 3])
            }
        }

        let pillars = AiPillars {
            tcf: Some(Arc::new(OrderTcf {
                order: order.clone(),
            })),
            ssm: Some(Arc::new(OrderSsm {
                order: order.clone(),
            })),
            cde: Some(Arc::new(OrderCde {
                order: order.clone(),
            })),
            nsr: Some(Arc::new(OrderNsr {
                order: order.clone(),
            })),
            ncde: Some(Arc::new(OrderNcde {
                order: order.clone(),
            })),
            ..AiPillars::default()
        };

        let port = MockAiPort::with_pillars(pillars);
        let normalized = normalize(base_frame("frame-1"));

        let _ = port.infer(&normalized);

        let collected = order.lock().unwrap().clone();
        assert_eq!(collected, vec!["tcf", "ssm", "cde", "nsr", "ncde"]);
    }

    #[test]
    fn nsr_violations_can_deny_speech() {
        struct DenyNsr;
        impl NsrPort for DenyNsr {
            fn check(&self, claims: &SymbolicClaims) -> NsrReport {
                let violations = claims
                    .claims
                    .iter()
                    .map(|claim| claim.predicate.clone())
                    .collect::<Vec<_>>();
                NsrReport {
                    ok: false,
                    violations,
                }
            }
        }

        let pillars = AiPillars {
            nsr: Some(Arc::new(DenyNsr)),
            ..AiPillars::default()
        };
        let port = MockAiPort::with_pillars(pillars);
        let normalized = normalize(base_frame("ping"));

        let outputs = port.infer(&normalized);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].channel, OutputChannel::Thought);
    }

    #[test]
    fn pipeline_emits_stable_meta_observer_outputs() {
        let build_port = || {
            let pillars = AiPillars {
                ssm: Some(Arc::new(MockSsmPort::new())),
                cde: Some(Arc::new(MockCdePort::new())),
                nsr: Some(Arc::new(MockNsrPort::new())),
                sle: Some(Arc::new(StrangeLoopEngine::new(4))),
                iit_monitor: Some(Arc::new(Mutex::new(IitMonitor::new(4)))),
                ..AiPillars::default()
            };
            MockAiPort::with_pillars(pillars)
        };

        let normalized = normalize(base_frame("frame-1"));
        let port_a = build_port();
        let port_b = build_port();

        let out_a = port_a.infer(&normalized);
        let out_b = port_b.infer(&normalized);
        let thought_a = out_a
            .iter()
            .find(|out| out.channel == OutputChannel::Thought)
            .expect("thought output");
        let thought_b = out_b
            .iter()
            .find(|out| out.channel == OutputChannel::Thought)
            .expect("thought output");

        assert!(thought_a.rationale_commit.is_some());
        assert!(thought_a.integration_score.is_some());
        assert_eq!(thought_a.rationale_commit, thought_b.rationale_commit);
        assert_eq!(thought_a.integration_score, thought_b.integration_score);
    }

    #[cfg(feature = "ai-runtime")]
    #[test]
    fn noop_runtime_backend_compiles_and_delegates() {
        let mock = MockAiPort::new();
        let backend = NoopRuntimeBackend::new(Arc::new(move |cf| mock.infer(cf)));
        let normalized = normalize(base_frame("ping"));

        let outputs = backend.infer_runtime(&normalized);

        assert!(!outputs.is_empty());
    }
}
