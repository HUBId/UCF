use std::sync::Arc;

use ucf_core::types::SimTime;

use crate::v1::{CorrelationId, DecisionCode, DenyReasonCode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionFrame {
    pub time: SimTime,
    pub corr: CorrelationId,
    pub decision: DecisionCode,
    pub deny_reason: Option<DenyReasonCode>,
    pub rationale: Arc<str>,
}

impl DecisionFrame {
    pub fn allow(time: SimTime, corr: CorrelationId, rationale: impl Into<Arc<str>>) -> Self {
        Self {
            time,
            corr,
            decision: DecisionCode::Allow,
            deny_reason: None,
            rationale: rationale.into(),
        }
    }

    pub fn deny(
        time: SimTime,
        corr: CorrelationId,
        reason: DenyReasonCode,
        rationale: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            time,
            corr,
            decision: DecisionCode::Deny,
            deny_reason: Some(reason),
            rationale: rationale.into(),
        }
    }

    pub fn defer(time: SimTime, corr: CorrelationId, rationale: impl Into<Arc<str>>) -> Self {
        Self {
            time,
            corr,
            decision: DecisionCode::Defer,
            deny_reason: None,
            rationale: rationale.into(),
        }
    }
}
