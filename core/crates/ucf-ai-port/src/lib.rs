#![forbid(unsafe_code)]

use ucf_policy_ecology::{PolicyEcology, PolicyRule};
use ucf_sandbox::ControlFrameNormalized;
use ucf_types::v1::spec::DecisionKind;
use ucf_types::Digest32;

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
pub struct MockAiPort;

impl MockAiPort {
    pub fn new() -> Self {
        Self
    }
}

impl AiPort for MockAiPort {
    fn infer(&self, input: &ControlFrameNormalized) -> Vec<AiOutput> {
        let rationale_commit = Some(input.commitment().digest);
        let thought = AiOutput {
            channel: OutputChannel::Thought,
            content: "ok".to_string(),
            confidence: 1000,
            rationale_commit,
        };

        if input.as_ref().frame_id == "ping" {
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
}
