use crate::{adapter::ActionAdapter, errors::PolicyError};
use ucf_frames::v1::{ChannelCode, ControlFrame, ControlPayload, DecisionCode, DecisionFrame};

pub struct Gem;

impl Gem {
    pub fn execute<A: ActionAdapter>(
        adapter: &mut A,
        ctrl: &ControlFrame,
        decision: Option<&DecisionFrame>,
    ) -> Result<(), PolicyError> {
        let decision = decision.ok_or(PolicyError::MissingDecision)?;

        match ctrl.channel {
            ChannelCode::InternalThought => Ok(()),
            _ => match decision.decision {
                DecisionCode::Deny | DecisionCode::Defer => Ok(()),
                DecisionCode::Allow => match (&ctrl.channel, &ctrl.payload) {
                    (ChannelCode::ExternalOutput, ControlPayload::Text(text)) => {
                        adapter.emit_text(text)
                    }
                    (ChannelCode::ExternalOutput, _) => {
                        Err(PolicyError::InvalidFrame("payload/channel mismatch"))
                    }
                    (ChannelCode::BrainStimulus, ControlPayload::Bytes(_))
                    | (ChannelCode::BrainStimulus, ControlPayload::Empty) => Ok(()),
                    (ChannelCode::BrainStimulus, ControlPayload::Text(_)) => {
                        Err(PolicyError::InvalidFrame("payload/channel mismatch"))
                    }
                    (ChannelCode::MemoryWrite, ControlPayload::Bytes(bytes)) => {
                        adapter.write_memory(bytes)
                    }
                    (ChannelCode::MemoryWrite, _) => {
                        Err(PolicyError::InvalidFrame("payload/channel mismatch"))
                    }
                    (ChannelCode::InternalThought, _) => Ok(()),
                },
            },
        }
    }
}
