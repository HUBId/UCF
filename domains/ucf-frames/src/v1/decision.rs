use std::sync::Arc;

use ucf_core::types::SimTime;

use crate::v1::{CorrelationId, DecisionCode, DenyReasonCode, IntentType};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComputeSignalsSummary {
    pub backend: &'static str,
    pub surprise: f32,
    pub pressure: f32,
    pub risk: f32,
    pub confidence: f32,
    pub spike_count: u16,
    pub spikes_digest: [u8; 32],
    pub sparsity: Option<f32>,
    pub energy: Option<f32>,
    pub ssm_readout: Option<f32>,
    pub ssm_digest: Option<[u8; 32]>,
    pub world_digest: Option<[u8; 32]>,
    pub risk_quality: Option<u8>,
    pub evidence_context_digest: Option<[u8; 32]>,
    pub evidence_world_digest: Option<[u8; 32]>,
    pub evidence_spikes_digest: Option<[u8; 32]>,
    pub evidence_ssm_digest: Option<[u8; 32]>,
    pub backend_profile: Option<&'static str>,
    pub budget_profile_id: Option<u32>,
    pub seed: Option<u64>,
    pub risk_contract_version: Option<u16>,
    pub compute_schema_version: Option<u16>,
    pub compute_chain_digest: Option<[u8; 32]>,
    pub compute_code_version: Option<&'static str>,
    pub budget_exceeded_stage: Option<&'static str>,
    pub coherence: Option<f32>,
    pub instability: Option<f32>,
    pub phi_proxy: Option<f32>,
    pub coherence_digest: Option<[u8; 32]>,
}

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
    pub compute_summary: Option<ComputeSignalsSummary>,
    pub gating_reason: Option<&'static str>,
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
            compute_summary: None,
            gating_reason: None,
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
            compute_summary: None,
            gating_reason: None,
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
            compute_summary: None,
            gating_reason: None,
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
            compute_summary: None,
            gating_reason: None,
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
            compute_summary: None,
            gating_reason: None,
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
            compute_summary: None,
            gating_reason: None,
        }
    }

    pub fn with_meta(mut self, meta: DecisionMeta) -> Self {
        self.meta = meta;
        self
    }
}

impl DecisionFrame {
    pub fn with_compute_summary(mut self, compute_summary: ComputeSignalsSummary) -> Self {
        self.compute_summary = Some(compute_summary);
        self
    }

    pub fn with_gating_reason(mut self, gating_reason: Option<&'static str>) -> Self {
        self.gating_reason = gating_reason;
        self
    }
}
