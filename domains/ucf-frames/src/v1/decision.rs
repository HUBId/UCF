use std::sync::Arc;

use ucf_core::types::SimTime;

use crate::v1::{CorrelationId, DecisionCode, DenyReasonCode, IntentType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionFrame {
    pub time: SimTime,
    pub corr: CorrelationId,
    pub decision: DecisionCode,
    pub intent: IntentType,
    pub reason_code: ReasonCode,
    pub deny_reason: Option<DenyReasonCode>,
    pub rationale: Arc<str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasonCode(pub &'static str);

impl DecisionFrame {
    pub fn allow(time: SimTime, corr: CorrelationId, rationale: impl Into<Arc<str>>) -> Self {
        Self {
            time,
            corr,
            decision: DecisionCode::Allow,
            intent: IntentType::Unknown,
            reason_code: ReasonCode("allow_default"),
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
            intent: IntentType::Unknown,
            reason_code: ReasonCode("deny_default"),
            deny_reason: Some(reason),
            rationale: rationale.into(),
        }
    }

    pub fn defer(time: SimTime, corr: CorrelationId, rationale: impl Into<Arc<str>>) -> Self {
        Self {
            time,
            corr,
            decision: DecisionCode::Defer,
            intent: IntentType::Unknown,
            reason_code: ReasonCode("defer_default"),
            deny_reason: None,
            rationale: rationale.into(),
        }
    }

    pub fn allow_with_reason(
        time: SimTime,
        corr: CorrelationId,
        intent: IntentType,
        reason_code: ReasonCode,
        rationale: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            time,
            corr,
            decision: DecisionCode::Allow,
            intent,
            reason_code,
            deny_reason: None,
            rationale: rationale.into(),
        }
    }

    pub fn deny_with_reason(
        time: SimTime,
        corr: CorrelationId,
        intent: IntentType,
        reason_code: ReasonCode,
        deny_reason: DenyReasonCode,
        rationale: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            time,
            corr,
            decision: DecisionCode::Deny,
            intent,
            reason_code,
            deny_reason: Some(deny_reason),
            rationale: rationale.into(),
        }
    }

    pub fn defer_with_reason(
        time: SimTime,
        corr: CorrelationId,
        intent: IntentType,
        reason_code: ReasonCode,
        rationale: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            time,
            corr,
            decision: DecisionCode::Defer,
            intent,
            reason_code,
            deny_reason: None,
            rationale: rationale.into(),
        }
    }
}
