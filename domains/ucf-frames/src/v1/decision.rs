use std::sync::Arc;

use ucf_core::types::SimTime;

use crate::v1::{CorrelationId, DecisionCode, DenyReasonCode, IntentType};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionMeta {
    pub attention_gain: f32,
    pub learning_gate: f32,
    pub recursion_budget: u8,
}

impl DecisionMeta {
    pub fn baseline() -> Self {
        Self {
            attention_gain: 0.5,
            learning_gate: 0.5,
            recursion_budget: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionFrame {
    pub time: SimTime,
    pub corr: CorrelationId,
    pub decision: DecisionCode,
    pub intent: IntentType,
    pub reason_code: ReasonCode,
    pub deny_reason: Option<DenyReasonCode>,
    pub rationale: Arc<str>,
    pub meta: DecisionMeta,
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
            meta: DecisionMeta::baseline(),
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
            meta: DecisionMeta::baseline(),
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
            meta: DecisionMeta::baseline(),
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
            meta: DecisionMeta::baseline(),
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
            meta: DecisionMeta::baseline(),
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
            meta: DecisionMeta::baseline(),
        }
    }

    pub fn with_meta(mut self, meta: DecisionMeta) -> Self {
        self.meta = meta;
        self
    }
}
