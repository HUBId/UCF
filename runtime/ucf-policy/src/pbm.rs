use ucf_frames::v1::{ChannelCode, ControlFrame, DecisionFrame, DenyReasonCode};

pub struct Pbm;

impl Pbm {
    pub fn decide(ctrl: &ControlFrame) -> DecisionFrame {
        match ctrl.channel {
            ChannelCode::InternalThought => {
                DecisionFrame::allow(ctrl.time, ctrl.corr, "allow_internal")
            }
            _ => DecisionFrame::deny(
                ctrl.time,
                ctrl.corr,
                DenyReasonCode::PolicyViolation,
                "deny_external",
            ),
        }
    }
}
