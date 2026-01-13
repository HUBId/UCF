#![forbid(unsafe_code)]

use ucf_bus::MessageEnvelope;
use ucf_types::v1::spec::{ActionCode, ControlFrame, DecisionKind, PolicyDecision};
use ucf_types::{LogicalTime, NodeId, StreamId, WallTime};

pub trait PolicyEvaluator {
    fn evaluate(&self, cf: ControlFrame) -> PolicyDecision;
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
