#![forbid(unsafe_code)]

use std::sync::Arc;

use blake3::Hasher;
use ucf_cde_port::CdePort;
use ucf_ncde_port::{NcdeContext, NcdePort};
use ucf_nsr_port::{NsrPort, NsrReport};
use ucf_policy_ecology::{PolicyEcology, PolicyRule};
use ucf_sandbox::ControlFrameNormalized;
use ucf_ssm_port::{SsmPort, SsmState};
use ucf_tcf_port::{TcfPort, TcfSignal};
use ucf_types::v1::spec::DecisionKind;
use ucf_types::{CausalGraphStub, Claim, Digest32, SymbolicClaims, ThoughtVec, WorldStateVec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputChannel {
    Thought,
    Speech,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiOutput {
    pub channel: OutputChannel,
    pub content: String,
    pub confidence: u16,
    pub rationale_commit: Option<Digest32>,
}

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
}

impl AiPillars {
    fn enabled(&self) -> bool {
        self.tcf.is_some()
            || self.ssm.is_some()
            || self.cde.is_some()
            || self.nsr.is_some()
            || self.ncde.is_some()
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

impl AiPort for MockAiPort {
    fn infer(&self, input: &ControlFrameNormalized) -> Vec<AiOutput> {
        let (rationale_commit, nsr_report) = if self.pillars.enabled() {
            run_pillars(&self.pillars, input)
        } else {
            (input.commitment().digest, None)
        };
        let rationale_commit = Some(rationale_commit);
        let thought = AiOutput {
            channel: OutputChannel::Thought,
            content: "ok".to_string(),
            confidence: 1000,
            rationale_commit,
        };

        let allow_speech = nsr_report.as_ref().is_none_or(|report| report.ok);
        if input.as_ref().frame_id == "ping" && allow_speech {
            let speech = AiOutput {
                channel: OutputChannel::Speech,
                content: "ok".to_string(),
                confidence: 1000,
                rationale_commit,
            };
            vec![thought, speech]
        } else {
            vec![thought]
        }
    }
}

fn run_pillars(
    pillars: &AiPillars,
    input: &ControlFrameNormalized,
) -> (Digest32, Option<NsrReport>) {
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

    if let Some(tcf) = &pillars.tcf {
        let pulse = tcf.tick(&mut signal, &[base_digest]);
        artifacts.push(pulse.digest);
    }
    if let Some(ssm) = &pillars.ssm {
        let output = ssm.update(&mut ssm_state, &world_state);
        artifacts.push(output.digest);
    }
    if let Some(cde) = &pillars.cde {
        let hypothesis = cde.infer(&mut graph, &world_state);
        artifacts.push(hypothesis.digest);
    }
    if let Some(nsr) = &pillars.nsr {
        let report = nsr.check(&claims);
        artifacts.push(digest_nsr_report(&report));
        nsr_report = Some(report);
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

    (rationale_commit, nsr_report)
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

    use ucf_sandbox::normalize;
    use ucf_types::v1::spec::ControlFrame;
    use ucf_types::SymbolicClaims;
    use ucf_types::ThoughtVec;
    use ucf_types::WorldStateVec;

    use ucf_cde_port::{CdeHypothesis, CdePort};
    use ucf_ncde_port::{NcdeContext, NcdePort};
    use ucf_nsr_port::{NsrPort, NsrReport};
    use ucf_ssm_port::{SsmOutput, SsmPort, SsmState};
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
}
