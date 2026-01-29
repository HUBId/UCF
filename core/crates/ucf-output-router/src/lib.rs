#![forbid(unsafe_code)]

use std::collections::VecDeque;

use blake3::Hasher;
use ucf_nsr_port::NsrVerdict;
use ucf_policy_ecology::{RiskDecision, RiskGateResult};
use ucf_recursion_controller::RecursionBudget;
use ucf_sandbox::ControlFrameNormalized;
use ucf_types::v1::spec::{DecisionKind, PolicyDecision};
use ucf_types::{AiOutput, Digest32};

pub use ucf_types::OutputChannel;

const COHERENCE_THOUGHT_CAP: u16 = 1;
const NSR_WARN_VERBOSE_LIMIT: usize = 240;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputFrame {
    pub channel: OutputChannel,
    pub text: String,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteDecision {
    pub permitted: bool,
    pub reason_code: String,
    pub evidence: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterConfig {
    pub thought_capacity: usize,
    pub max_thought_frames_per_cycle: u16,
    pub external_enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrSummary {
    pub verdict: NsrVerdict,
    pub violations_digest: Digest32,
}

pub use ucf_sandbox::SandboxVerdict;

#[derive(Clone, Debug)]
pub struct GateBundle {
    pub policy_decision: PolicyDecision,
    pub sandbox: SandboxVerdict,
    pub risk_results: Vec<RiskGateResult>,
    pub nsr_summary: NsrSummary,
    pub speech_gate: Vec<bool>,
    pub coherence_plv: u16,
    pub coherence_threshold: u16,
    pub phi_proxy: u16,
    pub phi_threshold: u16,
    pub speak_lock: u16,
    pub speak_lock_min: u16,
    pub damp_output: bool,
    pub output_gain_cap: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRef {
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputRouterEvent {
    ThoughtBuffered {
        frame: OutputFrame,
    },
    SpeechEmitted {
        frame: OutputFrame,
    },
    OutputSuppressed {
        frame: OutputFrame,
        reason_code: String,
        evidence: Digest32,
        risk: u16,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubVocalBuffer {
    cap: usize,
    frames: VecDeque<OutputFrame>,
}

impl SubVocalBuffer {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            frames: VecDeque::with_capacity(cap),
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn frames(&self) -> &VecDeque<OutputFrame> {
        &self.frames
    }

    pub fn push(&mut self, frame: OutputFrame) {
        if self.cap == 0 {
            return;
        }
        if self.frames.len() == self.cap {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    pub fn promote_to_evidence<F>(&self, rule: F) -> Vec<EvidenceRef>
    where
        F: Fn(&OutputFrame) -> bool,
    {
        self.frames
            .iter()
            .filter(|frame| rule(frame))
            .map(|frame| EvidenceRef {
                commit: frame.commit,
            })
            .collect()
    }
}

pub struct OutputRouter {
    config: RouterConfig,
    subvocal: SubVocalBuffer,
    events: Vec<OutputRouterEvent>,
    cycle: u64,
    recursion_thought_cap: Option<u16>,
    coherence_thought_cap: Option<u16>,
    force_thought_only: bool,
    suppress_verbose_thoughts: bool,
    verbose_thought_limit: usize,
}

impl OutputRouter {
    pub fn new(config: RouterConfig) -> Self {
        let subvocal = SubVocalBuffer::new(config.thought_capacity);
        Self {
            config,
            subvocal,
            events: Vec::new(),
            cycle: 0,
            recursion_thought_cap: None,
            coherence_thought_cap: None,
            force_thought_only: false,
            suppress_verbose_thoughts: false,
            verbose_thought_limit: 480,
        }
    }

    pub fn subvocal(&self) -> &SubVocalBuffer {
        &self.subvocal
    }

    pub fn max_thought_frames_per_cycle(&self) -> u16 {
        self.config.max_thought_frames_per_cycle
    }

    pub fn drain_events(&mut self) -> Vec<OutputRouterEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn set_max_thought_frames_per_cycle(&mut self, max: u16) {
        self.config.max_thought_frames_per_cycle = max.max(1);
    }

    pub fn apply_recursion_budget(&mut self, budget: &RecursionBudget) {
        if budget.max_depth <= 1 {
            self.recursion_thought_cap = Some(2);
            self.suppress_verbose_thoughts = true;
        } else {
            self.recursion_thought_cap = None;
            self.suppress_verbose_thoughts = false;
        }
    }

    pub fn apply_coherence(&mut self, coherence_plv: u16, threshold: u16) {
        if coherence_plv < threshold {
            self.force_thought_only = true;
            self.coherence_thought_cap = Some(COHERENCE_THOUGHT_CAP);
        } else {
            self.force_thought_only = false;
            self.coherence_thought_cap = None;
        }
    }

    pub fn route(
        &mut self,
        cf: &ControlFrameNormalized,
        outputs: Vec<AiOutput>,
        gates: &GateBundle,
    ) -> Vec<RouteDecision> {
        self.cycle = self.cycle.wrapping_add(1);
        let mut thought_count: u16 = 0;
        let mut decisions = Vec::with_capacity(outputs.len());
        let nsr_warn = matches!(gates.nsr_summary.verdict, NsrVerdict::Warn);
        let suppress_verbose = self.suppress_verbose_thoughts || nsr_warn;
        let verbose_limit = if nsr_warn {
            self.verbose_thought_limit.min(NSR_WARN_VERBOSE_LIMIT)
        } else {
            self.verbose_thought_limit
        };
        let mut max_thought_frames_per_cycle = if nsr_warn {
            (self.config.max_thought_frames_per_cycle / 2).max(1)
        } else {
            self.config.max_thought_frames_per_cycle
        };
        if let Some(cap) = self.coherence_thought_cap {
            max_thought_frames_per_cycle = max_thought_frames_per_cycle.min(cap).max(1);
        }

        for (idx, output) in outputs.into_iter().enumerate() {
            let commit = output_commit(cf, &output);
            let frame = OutputFrame {
                channel: output.channel,
                text: output.content,
                commit,
            };
            match frame.channel {
                OutputChannel::Thought => {
                    let decision = self.handle_thought(
                        frame,
                        &mut thought_count,
                        idx,
                        gates,
                        max_thought_frames_per_cycle,
                        suppress_verbose,
                        verbose_limit,
                    );
                    decisions.push(decision);
                }
                OutputChannel::Speech => {
                    let decision = self.handle_speech(frame, idx, gates);
                    decisions.push(decision);
                }
            }
        }

        decisions
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_thought(
        &mut self,
        frame: OutputFrame,
        thought_count: &mut u16,
        idx: usize,
        gates: &GateBundle,
        max_thought_frames_per_cycle: u16,
        suppress_verbose: bool,
        verbose_limit: usize,
    ) -> RouteDecision {
        if suppress_verbose && frame.text.len() > verbose_limit {
            let reason_code = "thought_verbose_suppressed".to_string();
            let evidence = decision_evidence(&frame, &reason_code, gates, idx, None);
            self.events.push(OutputRouterEvent::OutputSuppressed {
                frame,
                reason_code: reason_code.clone(),
                evidence,
                risk: 0,
            });
            return RouteDecision {
                permitted: false,
                reason_code,
                evidence,
            };
        }
        let thought_cap = self
            .recursion_thought_cap
            .map(|cap| cap.min(max_thought_frames_per_cycle))
            .unwrap_or(max_thought_frames_per_cycle);
        if *thought_count >= thought_cap {
            let reason_code = "thought_cycle_cap".to_string();
            let evidence = decision_evidence(&frame, &reason_code, gates, idx, None);
            self.events.push(OutputRouterEvent::OutputSuppressed {
                frame,
                reason_code: reason_code.clone(),
                evidence,
                risk: 0,
            });
            return RouteDecision {
                permitted: false,
                reason_code,
                evidence,
            };
        }
        if !output_gain_allows(frame.commit, gates.output_gain_cap) {
            let reason_code = "tcf_output_cap".to_string();
            let evidence = decision_evidence(&frame, &reason_code, gates, idx, None);
            self.events.push(OutputRouterEvent::OutputSuppressed {
                frame,
                reason_code: reason_code.clone(),
                evidence,
                risk: 0,
            });
            return RouteDecision {
                permitted: false,
                reason_code,
                evidence,
            };
        }
        *thought_count = thought_count.saturating_add(1);
        let reason_code = "thought_buffered".to_string();
        let evidence = decision_evidence(&frame, &reason_code, gates, idx, None);
        self.subvocal.push(frame.clone());
        self.events
            .push(OutputRouterEvent::ThoughtBuffered { frame });
        RouteDecision {
            permitted: true,
            reason_code,
            evidence,
        }
    }

    fn handle_speech(
        &mut self,
        frame: OutputFrame,
        idx: usize,
        gates: &GateBundle,
    ) -> RouteDecision {
        if !matches!(gates.nsr_summary.verdict, NsrVerdict::Allow) {
            return self.deny_speech(frame, idx, gates, "nsr_not_allow", 0);
        }
        if self.force_thought_only {
            return self.deny_speech(frame, idx, gates, "onn_low_coherence", 0);
        }
        if gates.coherence_plv < gates.coherence_threshold {
            return self.deny_speech(frame, idx, gates, "coherence_low", 0);
        }
        if gates.phi_proxy < gates.phi_threshold {
            return self.deny_speech(frame, idx, gates, "phi_low", 0);
        }
        if gates.damp_output {
            return self.deny_speech(frame, idx, gates, "iit_damp_output", 0);
        }
        if !output_gain_allows(frame.commit, gates.output_gain_cap) {
            return self.deny_speech(frame, idx, gates, "tcf_output_cap", 0);
        }
        if gates.speak_lock < gates.speak_lock_min {
            return self.deny_speech(frame, idx, gates, "speak_lock_low", 0);
        }
        let policy_allowed = policy_allows(&gates.policy_decision);
        if !policy_allowed {
            return self.deny_speech(frame, idx, gates, "policy_denied", 0);
        }
        if !sandbox_allows(&gates.sandbox) {
            return self.deny_speech(frame, idx, gates, "sandbox_denied", 0);
        }
        if !self.config.external_enabled {
            return self.deny_speech(frame, idx, gates, "external_disabled", 0);
        }
        let allow_speech = gates.speech_gate.get(idx).copied().unwrap_or(false);
        if !allow_speech {
            return self.deny_speech(frame, idx, gates, "speech_gate_denied", 0);
        }
        let risk = gates
            .risk_results
            .get(idx)
            .map(|result| result.risk)
            .unwrap_or(0);
        let risk_result = gates.risk_results.get(idx);
        if let Some(result) = risk_result {
            if matches!(result.decision, RiskDecision::Permit) {
                let reason_code = "speech_permitted".to_string();
                let evidence = decision_evidence(&frame, &reason_code, gates, idx, Some(result));
                self.events.push(OutputRouterEvent::SpeechEmitted { frame });
                return RouteDecision {
                    permitted: true,
                    reason_code,
                    evidence,
                };
            }
            let reason_code = risk_reason(result);
            return self.deny_speech(frame, idx, gates, &reason_code, risk);
        }

        self.deny_speech(frame, idx, gates, "risk_missing", risk)
    }

    fn deny_speech(
        &mut self,
        frame: OutputFrame,
        idx: usize,
        gates: &GateBundle,
        reason_code: &str,
        risk: u16,
    ) -> RouteDecision {
        let reason = reason_code.to_string();
        let evidence = decision_evidence(&frame, &reason, gates, idx, gates.risk_results.get(idx));
        self.events.push(OutputRouterEvent::OutputSuppressed {
            frame,
            reason_code: reason.clone(),
            evidence,
            risk,
        });
        RouteDecision {
            permitted: false,
            reason_code: reason,
            evidence,
        }
    }
}

fn policy_allows(decision: &PolicyDecision) -> bool {
    let kind =
        DecisionKind::try_from(decision.kind).unwrap_or(DecisionKind::DecisionKindUnspecified);
    matches!(
        kind,
        DecisionKind::DecisionKindAllow | DecisionKind::DecisionKindUnspecified
    )
}

fn sandbox_allows(verdict: &SandboxVerdict) -> bool {
    matches!(verdict, SandboxVerdict::Allow)
}

fn risk_reason(result: &RiskGateResult) -> String {
    result
        .reasons
        .first()
        .cloned()
        .unwrap_or_else(|| "risk_denied".to_string())
}

fn output_commit(cf: &ControlFrameNormalized, output: &AiOutput) -> Digest32 {
    if let Some(commit) = output.rationale_commit {
        return commit;
    }
    let mut hasher = Hasher::new();
    hasher.update(b"ucf-output-router-output/v1");
    hasher.update(match output.channel {
        OutputChannel::Thought => b"thought",
        OutputChannel::Speech => b"speech",
    });
    hasher.update(output.content.as_bytes());
    hasher.update(cf.commitment().digest.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn output_gain_allows(commit: Digest32, cap: u16) -> bool {
    if cap >= 10_000 {
        return true;
    }
    if cap == 0 {
        return false;
    }
    let bytes = commit.as_bytes();
    let sample = u16::from_be_bytes([bytes[0], bytes[1]]) % 10_000;
    sample < cap
}

fn decision_evidence(
    frame: &OutputFrame,
    reason_code: &str,
    gates: &GateBundle,
    index: usize,
    risk: Option<&RiskGateResult>,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf-output-router-decision/v1");
    hasher.update(frame.commit.as_bytes());
    hasher.update(reason_code.as_bytes());
    hasher.update(&[gates.nsr_summary.verdict.as_u8()]);
    hasher.update(gates.nsr_summary.violations_digest.as_bytes());
    hasher.update(&(index as u64).to_be_bytes());
    if let Some(risk) = risk {
        hasher.update(&risk.risk.to_be_bytes());
        hasher.update(risk.evidence.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_sandbox::normalize;
    use ucf_types::v1::spec::{ActionCode, ControlFrame};

    fn decision_allow() -> PolicyDecision {
        PolicyDecision {
            kind: DecisionKind::DecisionKindAllow as i32,
            action: ActionCode::ActionCodeContinue as i32,
            rationale: "ok".to_string(),
            confidence_bp: 1000,
            constraint_ids: Vec::new(),
        }
    }

    fn gates_for(outputs: &[AiOutput], risk: Vec<RiskGateResult>) -> GateBundle {
        GateBundle {
            policy_decision: decision_allow(),
            sandbox: SandboxVerdict::Allow,
            risk_results: risk,
            nsr_summary: NsrSummary {
                verdict: NsrVerdict::Allow,
                violations_digest: Digest32::new([0u8; 32]),
            },
            speech_gate: outputs.iter().map(|_| true).collect(),
            coherence_plv: 4000,
            coherence_threshold: 3000,
            phi_proxy: 4000,
            phi_threshold: 3200,
            speak_lock: 8000,
            speak_lock_min: 6000,
            damp_output: false,
            output_gain_cap: 10_000,
        }
    }

    fn cf() -> ControlFrameNormalized {
        normalize(ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 1,
            decision: None,
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        })
    }

    fn thought_output(text: &str) -> AiOutput {
        AiOutput {
            channel: OutputChannel::Thought,
            content: text.to_string(),
            confidence: 1000,
            rationale_commit: None,
            integration_score: None,
        }
    }

    fn speech_output(text: &str) -> AiOutput {
        AiOutput {
            channel: OutputChannel::Speech,
            content: text.to_string(),
            confidence: 1000,
            rationale_commit: None,
            integration_score: None,
        }
    }

    fn permit_risk() -> RiskGateResult {
        RiskGateResult {
            decision: RiskDecision::Permit,
            risk: 0,
            reasons: Vec::new(),
            evidence: Digest32::new([1u8; 32]),
        }
    }

    fn deny_risk(reason: &str) -> RiskGateResult {
        RiskGateResult {
            decision: RiskDecision::Deny,
            risk: 9000,
            reasons: vec![reason.to_string()],
            evidence: Digest32::new([2u8; 32]),
        }
    }

    #[test]
    fn thought_is_buffered_and_never_emitted() {
        let config = RouterConfig {
            thought_capacity: 4,
            max_thought_frames_per_cycle: 4,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        let outputs = vec![thought_output("inner"), speech_output("hello")];
        let decisions = router.route(
            &cf(),
            outputs.clone(),
            &gates_for(&outputs, vec![permit_risk(), permit_risk()]),
        );

        assert!(decisions[0].permitted);
        assert_eq!(router.subvocal().len(), 1);
        assert_eq!(router.subvocal().frames()[0].text, "inner");
        let events = router.drain_events();
        assert!(events
            .iter()
            .any(|event| matches!(event, OutputRouterEvent::ThoughtBuffered { .. })));
        assert!(events
            .iter()
            .any(|event| matches!(event, OutputRouterEvent::SpeechEmitted { .. })));
    }

    #[test]
    fn speech_denied_when_risk_gate_denies() {
        let config = RouterConfig {
            thought_capacity: 2,
            max_thought_frames_per_cycle: 2,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        let outputs = vec![speech_output("nope")];
        let decisions = router.route(
            &cf(),
            outputs.clone(),
            &gates_for(&outputs, vec![deny_risk("nsr_not_ok")]),
        );

        assert!(!decisions[0].permitted);
        assert_eq!(decisions[0].reason_code, "nsr_not_ok");
    }

    #[test]
    fn speech_allowed_when_all_gates_allow() {
        let config = RouterConfig {
            thought_capacity: 2,
            max_thought_frames_per_cycle: 2,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        let outputs = vec![speech_output("hi")];
        let decisions = router.route(
            &cf(),
            outputs.clone(),
            &gates_for(&outputs, vec![permit_risk()]),
        );

        assert!(decisions[0].permitted);
        assert_eq!(decisions[0].reason_code, "speech_permitted");
    }

    #[test]
    fn speech_denied_when_nsr_denies() {
        let config = RouterConfig {
            thought_capacity: 2,
            max_thought_frames_per_cycle: 2,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        let outputs = vec![speech_output("nope")];
        let mut gates = gates_for(&outputs, vec![permit_risk()]);
        gates.nsr_summary.verdict = NsrVerdict::Deny;

        let decisions = router.route(&cf(), outputs, &gates);

        assert!(!decisions[0].permitted);
        assert_eq!(decisions[0].reason_code, "nsr_not_allow");
    }

    #[test]
    fn warn_forces_thought_only_and_throttles() {
        let config = RouterConfig {
            thought_capacity: 4,
            max_thought_frames_per_cycle: 4,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        router.apply_coherence(1000, 3000);
        let outputs = vec![
            thought_output("t1"),
            thought_output("t2"),
            thought_output("t3"),
            speech_output("hi"),
        ];
        let mut gates = gates_for(
            &outputs,
            vec![permit_risk(), permit_risk(), permit_risk(), permit_risk()],
        );
        gates.nsr_summary.verdict = NsrVerdict::Warn;

        let decisions = router.route(&cf(), outputs, &gates);

        assert!(decisions[0].permitted);
        assert!(!decisions[1].permitted);
        assert_eq!(decisions[1].reason_code, "thought_cycle_cap");
        assert!(!decisions[2].permitted);
        assert_eq!(decisions[2].reason_code, "thought_cycle_cap");
        assert!(!decisions[3].permitted);
        assert_eq!(decisions[3].reason_code, "nsr_not_allow");
    }

    #[test]
    fn low_coherence_forces_thought_only_and_caps_thoughts() {
        let config = RouterConfig {
            thought_capacity: 4,
            max_thought_frames_per_cycle: 4,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        router.apply_coherence(1000, 3000);
        let outputs = vec![
            thought_output("t1"),
            thought_output("t2"),
            speech_output("hi"),
        ];
        let gates = gates_for(&outputs, vec![permit_risk(), permit_risk(), permit_risk()]);

        let decisions = router.route(&cf(), outputs, &gates);

        assert!(decisions[0].permitted);
        assert_eq!(decisions[1].reason_code, "thought_cycle_cap");
        assert_eq!(decisions[2].reason_code, "onn_low_coherence");
    }

    #[test]
    fn speech_denied_when_coherence_below_threshold() {
        let config = RouterConfig {
            thought_capacity: 2,
            max_thought_frames_per_cycle: 2,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        let outputs = vec![speech_output("hi")];
        let mut gates = gates_for(&outputs, vec![permit_risk()]);
        gates.coherence_plv = 1000;
        gates.coherence_threshold = 3000;

        let decisions = router.route(&cf(), outputs, &gates);

        assert!(!decisions[0].permitted);
        assert_eq!(decisions[0].reason_code, "coherence_low");
    }

    #[test]
    fn speech_denied_when_phi_below_threshold() {
        let config = RouterConfig {
            thought_capacity: 2,
            max_thought_frames_per_cycle: 2,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        let outputs = vec![speech_output("hi")];
        let mut gates = gates_for(&outputs, vec![permit_risk()]);
        gates.phi_proxy = 1000;
        gates.phi_threshold = 3200;

        let decisions = router.route(&cf(), outputs, &gates);

        assert!(!decisions[0].permitted);
        assert_eq!(decisions[0].reason_code, "phi_low");
    }

    #[test]
    fn speech_denied_when_speak_lock_below_threshold() {
        let config = RouterConfig {
            thought_capacity: 2,
            max_thought_frames_per_cycle: 2,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);
        let outputs = vec![speech_output("hi")];
        let mut gates = gates_for(&outputs, vec![permit_risk()]);
        gates.speak_lock = 1000;
        gates.speak_lock_min = 6000;

        let decisions = router.route(&cf(), outputs, &gates);

        assert!(!decisions[0].permitted);
        assert_eq!(decisions[0].reason_code, "speak_lock_low");
    }

    #[test]
    fn retention_is_deterministic_across_cycles() {
        let config = RouterConfig {
            thought_capacity: 2,
            max_thought_frames_per_cycle: 1,
            external_enabled: true,
        };
        let mut router = OutputRouter::new(config);

        let outputs = vec![thought_output("first"), thought_output("drop")];
        let _ = router.route(
            &cf(),
            outputs.clone(),
            &gates_for(&outputs, vec![permit_risk(), permit_risk()]),
        );
        assert_eq!(router.subvocal().len(), 1);
        assert_eq!(router.subvocal().frames()[0].text, "first");

        let outputs = vec![thought_output("second"), thought_output("drop-2")];
        let _ = router.route(
            &cf(),
            outputs.clone(),
            &gates_for(&outputs, vec![permit_risk(), permit_risk()]),
        );
        assert_eq!(router.subvocal().len(), 2);
        assert_eq!(router.subvocal().frames()[0].text, "first");
        assert_eq!(router.subvocal().frames()[1].text, "second");

        let outputs = vec![thought_output("third")];
        let _ = router.route(
            &cf(),
            outputs.clone(),
            &gates_for(&outputs, vec![permit_risk()]),
        );
        assert_eq!(router.subvocal().len(), 2);
        assert_eq!(router.subvocal().frames()[0].text, "second");
        assert_eq!(router.subvocal().frames()[1].text, "third");
    }
}
