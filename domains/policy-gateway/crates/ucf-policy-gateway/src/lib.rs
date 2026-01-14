#![forbid(unsafe_code)]

use ucf_bus::MessageEnvelope;
use ucf_types::v1::spec::{ActionCode, ControlFrame, DecisionKind, PolicyDecision};
use ucf_types::{LogicalTime, NodeId, StreamId, WallTime};

pub trait PolicyEvaluator {
    fn evaluate(&self, cf: ControlFrame) -> PolicyDecision;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyDecisionClass {
    Unspecified,
    Allow,
    Deny,
    Escalate,
    Observe,
    Unknown(i32),
}

impl PolicyDecisionClass {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::Unspecified => DecisionKind::DecisionKindUnspecified as u16,
            Self::Allow => DecisionKind::DecisionKindAllow as u16,
            Self::Deny => DecisionKind::DecisionKindDeny as u16,
            Self::Escalate => DecisionKind::DecisionKindEscalate as u16,
            Self::Observe => DecisionKind::DecisionKindObserve as u16,
            Self::Unknown(value) => u16::try_from(value).unwrap_or(0),
        }
    }
}

impl From<&PolicyDecision> for PolicyDecisionClass {
    fn from(decision: &PolicyDecision) -> Self {
        match DecisionKind::try_from(decision.kind) {
            Ok(DecisionKind::DecisionKindUnspecified) => Self::Unspecified,
            Ok(DecisionKind::DecisionKindAllow) => Self::Allow,
            Ok(DecisionKind::DecisionKindDeny) => Self::Deny,
            Ok(DecisionKind::DecisionKindEscalate) => Self::Escalate,
            Ok(DecisionKind::DecisionKindObserve) => Self::Observe,
            Err(_) => Self::Unknown(decision.kind),
        }
    }
}

pub fn decision_class_id(decision: &PolicyDecision) -> u16 {
    PolicyDecisionClass::from(decision).as_u16()
}

#[derive(Default)]
pub struct NoOpPolicyEvaluator {
    rationale: String,
}

impl NoOpPolicyEvaluator {
    pub fn new() -> Self {
        Self {
            rationale: "no decision".to_string(),
        }
    }
}

impl PolicyEvaluator for NoOpPolicyEvaluator {
    fn evaluate(&self, _cf: ControlFrame) -> PolicyDecision {
        PolicyDecision {
            kind: DecisionKind::DecisionKindUnspecified as i32,
            action: ActionCode::ActionCodeUnspecified as i32,
            rationale: self.rationale.clone(),
            confidence_bp: 0,
            constraint_ids: Vec::new(),
        }
    }
}

pub fn wrap_decision(
    decision: PolicyDecision,
    node_id: NodeId,
    stream_id: StreamId,
    logical_time: LogicalTime,
    wall_time: WallTime,
) -> MessageEnvelope<PolicyDecision> {
    MessageEnvelope {
        node_id,
        stream_id,
        logical_time,
        wall_time,
        payload: decision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_evaluator_returns_no_decision() {
        let evaluator = NoOpPolicyEvaluator::new();
        let frame = ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 1,
            decision: None,
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        };

        let decision = evaluator.evaluate(frame);

        assert_eq!(decision.kind, DecisionKind::DecisionKindUnspecified as i32);
        assert_eq!(decision.action, ActionCode::ActionCodeUnspecified as i32);
        assert_eq!(decision.rationale, "no decision");
        assert_eq!(decision.confidence_bp, 0);
        assert!(decision.constraint_ids.is_empty());
    }
}
