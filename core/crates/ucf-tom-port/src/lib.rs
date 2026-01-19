#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_sandbox::ControlFrameNormalized;
use ucf_types::{AiOutput, Digest32};

pub type ActorId = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActorProfile {
    pub id: ActorId,
    pub label: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntentType {
    AskInfo,
    Negotiate,
    RequestAction,
    SocialBond,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntentHypothesis {
    pub intent: IntentType,
    pub confidence: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnowledgeGap {
    pub topic: String,
    pub uncertainty: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SocialRiskSignals {
    pub deception_likelihood: u16,
    pub consent_uncertainty: u16,
    pub manipulation_risk: u16,
    pub overall: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TomReport {
    pub actors: Vec<ActorProfile>,
    pub intent: IntentHypothesis,
    pub gaps: Vec<KnowledgeGap>,
    pub risk: SocialRiskSignals,
    pub commit: Digest32,
}

pub trait TomPort {
    fn analyze(&self, cf: &ControlFrameNormalized, outputs: &[AiOutput]) -> TomReport;
}

#[derive(Clone, Default)]
pub struct MockTomPort;

impl MockTomPort {
    pub fn new() -> Self {
        Self
    }
}

impl TomPort for MockTomPort {
    fn analyze(&self, cf: &ControlFrameNormalized, outputs: &[AiOutput]) -> TomReport {
        let output_digest = output_text_digest(outputs);
        let seed = seed_digest(cf, &output_digest);
        let intent = infer_intent(outputs);
        let confidence = bounded_u16(&seed, b"intent_confidence");
        let intent_hyp = IntentHypothesis { intent, confidence };

        let actor = ActorProfile {
            id: actor_id(&seed),
            label: format!("actor:{}", cf.as_ref().policy_id),
        };

        let gaps = vec![KnowledgeGap {
            topic: "context".to_string(),
            uncertainty: bounded_u16(&seed, b"gap_context"),
        }];

        let deception_likelihood = bounded_u16(&seed, b"deception");
        let consent_uncertainty = bounded_u16(&seed, b"consent");
        let manipulation_risk = bounded_u16(&seed, b"manipulation");
        let overall = overall_score(deception_likelihood, consent_uncertainty, manipulation_risk);
        let risk = SocialRiskSignals {
            deception_likelihood,
            consent_uncertainty,
            manipulation_risk,
            overall,
        };

        let commit = report_commit(&actor, &intent_hyp, &gaps, &risk, &seed, &output_digest);

        TomReport {
            actors: vec![actor],
            intent: intent_hyp,
            gaps,
            risk,
            commit,
        }
    }
}

fn output_text_digest(outputs: &[AiOutput]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tom.output_text.v1");
    hasher.update(&u64::try_from(outputs.len()).unwrap_or(0).to_be_bytes());
    for output in outputs {
        hasher.update(
            &u64::try_from(output.content.len())
                .unwrap_or(0)
                .to_be_bytes(),
        );
        hasher.update(output.content.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn seed_digest(cf: &ControlFrameNormalized, output_digest: &Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tom.seed.v1");
    hasher.update(cf.commitment().digest.as_bytes());
    hasher.update(output_digest.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn infer_intent(outputs: &[AiOutput]) -> IntentType {
    let text = outputs
        .iter()
        .map(|out| out.content.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    if text.contains("bitte") || text.contains("please") {
        IntentType::RequestAction
    } else if text.contains("warum") || text.contains("why") {
        IntentType::AskInfo
    } else {
        IntentType::Unknown
    }
}

fn bounded_u16(seed: &Digest32, tag: &[u8]) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tom.bound.v1");
    hasher.update(seed.as_bytes());
    hasher.update(tag);
    let digest = hasher.finalize();
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&digest.as_bytes()[0..2]);
    let raw = u16::from_be_bytes(bytes);
    raw % 10001
}

fn overall_score(deception: u16, consent: u16, manipulation: u16) -> u16 {
    let sum = u32::from(deception) + u32::from(consent) + u32::from(manipulation);
    (sum / 3) as u16
}

fn actor_id(seed: &Digest32) -> ActorId {
    let bytes = seed.as_bytes();
    let mut id_bytes = [0u8; 8];
    id_bytes.copy_from_slice(&bytes[0..8]);
    u64::from_be_bytes(id_bytes)
}

fn report_commit(
    actor: &ActorProfile,
    intent: &IntentHypothesis,
    gaps: &[KnowledgeGap],
    risk: &SocialRiskSignals,
    seed: &Digest32,
    output_digest: &Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.tom.report.v1");
    hasher.update(seed.as_bytes());
    hasher.update(output_digest.as_bytes());
    hasher.update(&actor.id.to_be_bytes());
    hasher.update(actor.label.as_bytes());
    hasher.update(&[intent_code(intent.intent)]);
    hasher.update(&intent.confidence.to_be_bytes());
    hasher.update(&u64::try_from(gaps.len()).unwrap_or(0).to_be_bytes());
    for gap in gaps {
        hasher.update(&u64::try_from(gap.topic.len()).unwrap_or(0).to_be_bytes());
        hasher.update(gap.topic.as_bytes());
        hasher.update(&gap.uncertainty.to_be_bytes());
    }
    hasher.update(&risk.deception_likelihood.to_be_bytes());
    hasher.update(&risk.consent_uncertainty.to_be_bytes());
    hasher.update(&risk.manipulation_risk.to_be_bytes());
    hasher.update(&risk.overall.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn intent_code(intent: IntentType) -> u8 {
    match intent {
        IntentType::AskInfo => 1,
        IntentType::Negotiate => 2,
        IntentType::RequestAction => 3,
        IntentType::SocialBond => 4,
        IntentType::Unknown => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ucf_sandbox::normalize;
    use ucf_types::v1::spec::ControlFrame;
    use ucf_types::OutputChannel;

    fn base_frame() -> ControlFrame {
        ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 1,
            decision: None,
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        }
    }

    #[test]
    fn mock_tom_port_is_deterministic() {
        let port = MockTomPort::new();
        let cf = normalize(base_frame());
        let outputs = vec![AiOutput {
            channel: OutputChannel::Speech,
            content: "please help".to_string(),
            confidence: 1000,
            rationale_commit: None,
            integration_score: None,
        }];

        let report_a = port.analyze(&cf, &outputs);
        let report_b = port.analyze(&cf, &outputs);

        assert_eq!(report_a, report_b);
    }
}
