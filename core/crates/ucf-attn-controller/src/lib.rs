#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusChannel {
    Threat,
    Task,
    Social,
    Memory,
    Exploration,
    Idle,
}

impl FocusChannel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Threat => "threat",
            Self::Task => "task",
            Self::Social => "social",
            Self::Memory => "memory",
            Self::Exploration => "exploration",
            Self::Idle => "idle",
        }
    }
}

impl std::str::FromStr for FocusChannel {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "threat" => Ok(Self::Threat),
            "task" => Ok(Self::Task),
            "social" => Ok(Self::Social),
            "memory" => Ok(Self::Memory),
            "exploration" => Ok(Self::Exploration),
            "idle" => Ok(Self::Idle),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionWeights {
    pub channel: FocusChannel,
    pub gain: u16,
    pub noise_suppress: u16,
    pub replay_bias: u16,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnInputs {
    pub policy_class: u16,
    pub risk_score: u16,
    pub integration_score: u16,
    pub consistency_instability: u16,
    pub intent_type: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionUpdated {
    pub channel: FocusChannel,
    pub gain: u16,
    pub replay_bias: u16,
    pub commit: Digest32,
}

pub trait AttentionEventSink {
    fn publish(&self, event: AttentionUpdated);
}

#[derive(Clone, Default)]
pub struct AttnController;

impl AttnController {
    pub const INTENT_ASK_INFO: u16 = 0;
    pub const INTENT_NEGOTIATE: u16 = 1;
    pub const INTENT_REQUEST_ACTION: u16 = 2;
    pub const INTENT_SOCIAL_BOND: u16 = 3;
    pub const INTENT_UNKNOWN: u16 = 4;

    pub fn compute(&self, inputs: &AttnInputs) -> AttentionWeights {
        let commit = commit_inputs(inputs);

        let mut channel = FocusChannel::Idle;
        let mut gain = 1000;
        let mut noise_suppress = 1000;
        let mut replay_bias = 1000;

        if inputs.risk_score >= 7000 || is_policy_high_risk(inputs.policy_class) {
            channel = FocusChannel::Threat;
            gain = clamp_scale(4000u32 + (inputs.risk_score as u32 / 2));
            noise_suppress = clamp_scale(3000u32 + (inputs.risk_score as u32 / 2));
            replay_bias = clamp_scale(2000u32 + (inputs.risk_score as u32 / 3));
        } else if inputs.intent_type == Self::INTENT_REQUEST_ACTION && inputs.risk_score <= 3000 {
            channel = FocusChannel::Task;
            gain = clamp_scale(6000u32 + (inputs.integration_score as u32 / 5));
            noise_suppress = 2000;
            replay_bias = 1500;
        } else if inputs.integration_score <= 3000 {
            channel = FocusChannel::Memory;
            gain = 3000;
            noise_suppress = 2500;
            replay_bias = 8500;
        } else if inputs.intent_type == Self::INTENT_SOCIAL_BOND {
            channel = FocusChannel::Social;
            gain = 3500;
            noise_suppress = 1500;
            replay_bias = 2000;
        } else if inputs.consistency_instability >= 6000 {
            channel = FocusChannel::Exploration;
            gain = 4500;
            noise_suppress = 1200;
            replay_bias = 3000;
        }

        AttentionWeights {
            channel,
            gain,
            noise_suppress,
            replay_bias,
            commit,
        }
    }
}

fn is_policy_high_risk(policy_class: u16) -> bool {
    matches!(policy_class, 2 | 3)
}

fn clamp_scale(value: u32) -> u16 {
    value.min(10_000) as u16
}

fn commit_inputs(inputs: &AttnInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.attn.inputs.v1");
    hasher.update(&inputs.policy_class.to_be_bytes());
    hasher.update(&inputs.risk_score.to_be_bytes());
    hasher.update(&inputs.integration_score.to_be_bytes());
    hasher.update(&inputs.consistency_instability.to_be_bytes());
    hasher.update(&inputs.intent_type.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_is_deterministic() {
        let controller = AttnController;
        let inputs = AttnInputs {
            policy_class: 1,
            risk_score: 2500,
            integration_score: 4200,
            consistency_instability: 1000,
            intent_type: AttnController::INTENT_ASK_INFO,
        };

        let first = controller.compute(&inputs);
        let second = controller.compute(&inputs);

        assert_eq!(first, second);
        assert_eq!(first.commit, second.commit);
    }

    #[test]
    fn threat_channel_selected_for_high_risk() {
        let controller = AttnController;
        let inputs = AttnInputs {
            policy_class: 1,
            risk_score: 9000,
            integration_score: 5000,
            consistency_instability: 0,
            intent_type: AttnController::INTENT_UNKNOWN,
        };

        let weights = controller.compute(&inputs);

        assert_eq!(weights.channel, FocusChannel::Threat);
        assert!(weights.gain > 1000);
        assert!(weights.noise_suppress > 1000);
    }

    #[test]
    fn replay_bias_increases_when_integration_low() {
        let controller = AttnController;
        let low = AttnInputs {
            policy_class: 1,
            risk_score: 1000,
            integration_score: 1000,
            consistency_instability: 0,
            intent_type: AttnController::INTENT_UNKNOWN,
        };
        let high = AttnInputs {
            integration_score: 9000,
            ..low
        };

        let low_weights = controller.compute(&low);
        let high_weights = controller.compute(&high);

        assert!(low_weights.replay_bias > high_weights.replay_bias);
        assert_eq!(low_weights.channel, FocusChannel::Memory);
    }
}
