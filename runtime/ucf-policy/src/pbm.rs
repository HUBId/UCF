use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, DecisionFrame, DenyReasonCode, IntentType,
    ReasonCode,
};

pub struct Pbm;

impl Pbm {
    pub fn infer_intent(ctrl: &ControlFrame) -> IntentType {
        match ctrl.channel {
            ChannelCode::InternalThought => IntentType::InternalThought,
            ChannelCode::ExternalOutput => IntentType::ExternalCommunicate,
            ChannelCode::MemoryWrite => IntentType::WriteMemory,
            ChannelCode::BrainStimulus => IntentType::StimulateBrain,
        }
    }

    pub fn decide(ctrl: &ControlFrame) -> DecisionFrame {
        let intent = Self::infer_intent(ctrl);
        match ctrl.channel {
            ChannelCode::InternalThought => DecisionFrame::allow_with_reason(
                ctrl.time,
                ctrl.corr,
                intent,
                ReasonCode("allow_internal"),
                "allow_internal",
            ),
            ChannelCode::ExternalOutput => match &ctrl.payload {
                ControlPayload::Text(text) => {
                    if text.len() <= 512 {
                        DecisionFrame::allow_with_reason(
                            ctrl.time,
                            ctrl.corr,
                            intent,
                            ReasonCode("allow_text_external"),
                            "allow_text_external",
                        )
                    } else {
                        DecisionFrame::deny_with_reason(
                            ctrl.time,
                            ctrl.corr,
                            intent,
                            ReasonCode("deny_external_too_long"),
                            DenyReasonCode::PolicyViolation,
                            "deny_external_too_long",
                        )
                    }
                }
                _ => DecisionFrame::deny_with_reason(
                    ctrl.time,
                    ctrl.corr,
                    intent,
                    ReasonCode("deny_external_nontext"),
                    DenyReasonCode::PolicyViolation,
                    "deny_external_nontext",
                ),
            },
            ChannelCode::MemoryWrite => DecisionFrame::deny_with_reason(
                ctrl.time,
                ctrl.corr,
                intent,
                ReasonCode("deny_memorywrite_default"),
                DenyReasonCode::PolicyViolation,
                "deny_memorywrite_default",
            ),
            ChannelCode::BrainStimulus => DecisionFrame::deny_with_reason(
                ctrl.time,
                ctrl.corr,
                intent,
                ReasonCode("deny_brainstim_default"),
                DenyReasonCode::PolicyViolation,
                "deny_brainstim_default",
            ),
        }
    }
}
