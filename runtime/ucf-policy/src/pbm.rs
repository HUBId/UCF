use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, DecisionFrame, DecisionMeta, DenyReasonCode,
    IntentType, NeuromodulatorSnapshot, ReasonCode,
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

    pub fn decide(ctrl: &ControlFrame, neuromod: Option<NeuromodulatorSnapshot>) -> DecisionFrame {
        let intent = Self::infer_intent(ctrl);
        let decision = match ctrl.channel {
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
        };

        decision.with_meta(Self::meta_from_neuromod(neuromod))
    }

    fn meta_from_neuromod(neuromod: Option<NeuromodulatorSnapshot>) -> DecisionMeta {
        let Some(neuromod) = neuromod else {
            return DecisionMeta::baseline();
        };

        let attention_gain =
            Self::clamp01(0.4 + 0.6 * (neuromod.norepinephrine * 0.6 + neuromod.stress * 0.4));
        let learning_gate = Self::clamp01(0.3 + 0.7 * neuromod.acetylcholine);

        let recursion_budget = if neuromod.dopamine > 0.7 && neuromod.stress < 0.4 {
            3
        } else if neuromod.stress > 0.7 {
            0
        } else if neuromod.serotonin > 0.7 {
            2
        } else {
            1
        };

        DecisionMeta {
            attention_gain,
            learning_gate,
            recursion_budget,
        }
    }

    fn clamp01(value: f32) -> f32 {
        value.clamp(0.0, 1.0)
    }
}
