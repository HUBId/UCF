use metrics::counter;
use tracing::info_span;
use ucf_bluebrain_bridge::BrainStimulusEncoder;
use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, DecisionCode, DecisionFrame, DenyReasonCode,
};

use crate::{
    adapter::ActionAdapter,
    capability::{
        CapabilityDenyReason, CapabilityKind, CapabilityLimits, CapabilityScope, CapabilitySet,
        CapabilityToken,
    },
    errors::PolicyError,
    rate_limiter::{RateKey, RateLimiter},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PayloadHint {
    pub bytes_out: Option<u32>,
    pub bytes_in: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolRequest {
    pub id: u64,
    pub kind: CapabilityKind,
    pub target: String,
    pub payload_hint: PayloadHint,
    pub requested_at_t: u64,
    pub decision_id: u64,
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    AllowedExecuted,
    Denied,
    RateLimited,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolResultSummary {
    pub status: ToolStatus,
    pub bytes_out: Option<u32>,
    pub bytes_in: Option<u32>,
    pub error_code: Option<String>,
    pub finished_at_t: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorizationOutcome {
    Allowed { token_digest: [u8; 32] },
    Denied { reason: CapabilityDenyReason },
    RateLimited { retry_after_ticks: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolExecutionAudit {
    pub request: ToolRequest,
    pub auth: AuthorizationOutcome,
    pub result: ToolResultSummary,
}

pub struct ToolGate {
    pub capabilities: CapabilitySet,
    pub rate_limiter: RateLimiter,
}

impl ToolGate {
    pub fn new(capabilities: CapabilitySet, rate_limiter: RateLimiter) -> Self {
        Self {
            capabilities,
            rate_limiter,
        }
    }

    pub fn authorize(&mut self, req: &ToolRequest, now_t: u64) -> AuthorizationOutcome {
        let span = info_span!(
            "tool_gate.authorize",
            kind = req.kind.as_tag(),
            target = req.target.as_str()
        );
        let _entered = span.enter();
        counter!("ucf_tool_requests_total", "kind" => req.kind.as_tag().to_string()).increment(1);

        let token = match self.capabilities.allows(req, now_t) {
            Ok(token) => token,
            Err(reason) => {
                counter!("ucf_tool_denied_total", "reason" => format!("{reason:?}")).increment(1);
                return AuthorizationOutcome::Denied { reason };
            }
        };

        let rate = self.rate_limiter.check_and_record(
            RateKey {
                kind: req.kind.clone(),
                target: req.target.clone(),
                token_digest: token.token_digest,
            },
            now_t,
            token.limits.max_calls_per_window,
            token.limits.window_ticks,
        );

        if !rate.allowed {
            counter!("ucf_tool_rate_limited_total").increment(1);
            return AuthorizationOutcome::RateLimited {
                retry_after_ticks: rate.retry_after_ticks,
            };
        }

        AuthorizationOutcome::Allowed {
            token_digest: token.token_digest,
        }
    }
}

pub struct Gem;

impl Gem {
    pub fn execute<A: ActionAdapter>(
        adapter: &mut A,
        ctrl: &ControlFrame,
        decision: Option<&DecisionFrame>,
    ) -> Result<(), PolicyError> {
        let mut gate = ToolGate::new(
            issue_capabilities(decision, ctrl.time.tick.get()),
            RateLimiter::new(1024),
        );
        Self::execute_with_gate(adapter, ctrl, decision, ctrl.corr.0, &mut gate).map(|_| ())
    }

    pub fn execute_with_gate<A: ActionAdapter>(
        adapter: &mut A,
        ctrl: &ControlFrame,
        decision: Option<&DecisionFrame>,
        decision_id: u64,
        gate: &mut ToolGate,
    ) -> Result<ToolExecutionAudit, PolicyError> {
        let decision = decision.ok_or(PolicyError::MissingDecision)?;
        let req = request_from(ctrl, decision, decision_id);
        let auth = gate.authorize(&req, req.requested_at_t);
        let finished_at_t = req.requested_at_t;

        if matches!(decision.decision, DecisionCode::Deny | DecisionCode::Defer) {
            return Ok(ToolExecutionAudit {
                request: req,
                auth: AuthorizationOutcome::Denied {
                    reason: CapabilityDenyReason::MissingToken,
                },
                result: ToolResultSummary {
                    status: ToolStatus::Denied,
                    bytes_out: None,
                    bytes_in: None,
                    error_code: Some("decision_not_allow".to_string()),
                    finished_at_t,
                },
            });
        }

        match auth {
            AuthorizationOutcome::Allowed { .. } => {
                let outcome = match (&ctrl.channel, &ctrl.payload) {
                    (ChannelCode::InternalThought, _) => Ok(()),
                    (ChannelCode::ExternalOutput, ControlPayload::Text(text)) => {
                        adapter.emit_text(text)
                    }
                    (ChannelCode::ExternalOutput, _) => {
                        Err(PolicyError::InvalidFrame("payload/channel mismatch"))
                    }
                    (ChannelCode::BrainStimulus, ControlPayload::BrainStimulus(payload)) => {
                        let spikes = BrainStimulusEncoder::encode_to_spikes(ctrl, payload);
                        adapter.emit_brain_spikes(spikes)
                    }
                    (ChannelCode::BrainStimulus, _) => {
                        Err(PolicyError::InvalidFrame("payload/channel mismatch"))
                    }
                    (ChannelCode::MemoryWrite, ControlPayload::Bytes(bytes)) => {
                        adapter.write_memory(bytes)
                    }
                    (ChannelCode::MemoryWrite, _) => {
                        Err(PolicyError::InvalidFrame("payload/channel mismatch"))
                    }
                };

                let (status, error_code) = if let Err(error) = outcome {
                    (ToolStatus::Failed, Some(error.to_string()))
                } else {
                    (ToolStatus::AllowedExecuted, None)
                };

                Ok(ToolExecutionAudit {
                    request: req.clone(),
                    auth,
                    result: ToolResultSummary {
                        status,
                        bytes_out: req.payload_hint.bytes_out,
                        bytes_in: req.payload_hint.bytes_in,
                        error_code,
                        finished_at_t,
                    },
                })
            }
            AuthorizationOutcome::Denied { reason } => Ok(ToolExecutionAudit {
                request: req.clone(),
                auth,
                result: ToolResultSummary {
                    status: ToolStatus::Denied,
                    bytes_out: req.payload_hint.bytes_out,
                    bytes_in: req.payload_hint.bytes_in,
                    error_code: Some(format!("{reason:?}")),
                    finished_at_t,
                },
            }),
            AuthorizationOutcome::RateLimited { retry_after_ticks } => Ok(ToolExecutionAudit {
                request: req.clone(),
                auth,
                result: ToolResultSummary {
                    status: ToolStatus::RateLimited,
                    bytes_out: req.payload_hint.bytes_out,
                    bytes_in: req.payload_hint.bytes_in,
                    error_code: Some(format!("retry_after:{retry_after_ticks}")),
                    finished_at_t,
                },
            }),
        }
    }
}

pub fn request_from(
    ctrl: &ControlFrame,
    decision: &DecisionFrame,
    decision_id: u64,
) -> ToolRequest {
    let (kind, target, payload_hint) = match (&ctrl.channel, &ctrl.payload) {
        (ChannelCode::ExternalOutput, ControlPayload::Text(text)) => (
            CapabilityKind::ExternalApi,
            "external_output".to_string(),
            PayloadHint {
                bytes_out: Some(text.len() as u32),
                bytes_in: None,
            },
        ),
        (ChannelCode::MemoryWrite, ControlPayload::Bytes(bytes)) => (
            CapabilityKind::FileWrite,
            "memory_write".to_string(),
            PayloadHint {
                bytes_out: Some(bytes.len() as u32),
                bytes_in: None,
            },
        ),
        (ChannelCode::BrainStimulus, ControlPayload::BrainStimulus(_payload)) => (
            CapabilityKind::UiAutomation,
            "brain_target".to_string(),
            PayloadHint {
                bytes_out: Some(4),
                bytes_in: None,
            },
        ),
        (ChannelCode::InternalThought, _) => (
            CapabilityKind::Custom("internal_thought".to_string()),
            "internal".to_string(),
            PayloadHint::default(),
        ),
        _ => (
            CapabilityKind::Custom("invalid".to_string()),
            "invalid".to_string(),
            PayloadHint::default(),
        ),
    };

    let evidence_chain_digest = decision
        .compute_summary
        .and_then(|s| s.compute_chain_digest)
        .unwrap_or([0u8; 32]);

    ToolRequest {
        id: ctrl.corr.0,
        kind,
        target,
        payload_hint,
        requested_at_t: ctrl.time.tick.get(),
        decision_id,
        evidence_chain_digest,
    }
}

pub fn issue_capabilities(decision: Option<&DecisionFrame>, now_t: u64) -> CapabilitySet {
    let Some(decision) = decision else {
        return CapabilitySet::empty();
    };

    let risk = decision
        .compute_summary
        .map(|s| s.risk)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let confidence = decision
        .compute_summary
        .map(|s| s.confidence)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);

    let mut tokens = Vec::new();
    if risk > 0.7 || confidence < 0.3 {
        tokens.push(CapabilityToken::issue(
            CapabilityKind::FileRead,
            CapabilityScope::Paths(vec!["/workspace/UCF/config".to_string()]),
            CapabilityLimits {
                max_calls_per_window: 1,
                window_ticks: 10,
                max_bytes_out: None,
                max_bytes_in: Some(1024),
                max_concurrent: 1,
            },
            "pbm_v0",
            now_t,
            Some(now_t.saturating_add(10)),
        ));
        return CapabilitySet { tokens };
    }

    if decision.decision == DecisionCode::Allow {
        tokens.push(CapabilityToken::issue(
            CapabilityKind::ExternalApi,
            CapabilityScope::ApiNames(vec!["external_output".to_string()]),
            CapabilityLimits {
                max_calls_per_window: 4,
                window_ticks: 10,
                max_bytes_out: Some(4096),
                max_bytes_in: Some(1024),
                max_concurrent: 1,
            },
            "pbm_v0",
            now_t,
            Some(now_t.saturating_add(10)),
        ));
        tokens.push(CapabilityToken::issue(
            CapabilityKind::FileWrite,
            CapabilityScope::Paths(vec!["memory_write".to_string()]),
            CapabilityLimits {
                max_calls_per_window: 2,
                window_ticks: 10,
                max_bytes_out: Some(2048),
                max_bytes_in: None,
                max_concurrent: 1,
            },
            "pbm_v0",
            now_t,
            Some(now_t.saturating_add(10)),
        ));
        tokens.push(CapabilityToken::issue(
            CapabilityKind::UiAutomation,
            CapabilityScope::ApiNames(vec!["brain_target".to_string()]),
            CapabilityLimits {
                max_calls_per_window: 2,
                window_ticks: 10,
                max_bytes_out: Some(64),
                max_bytes_in: None,
                max_concurrent: 1,
            },
            "pbm_v0",
            now_t,
            Some(now_t.saturating_add(10)),
        ));
    }

    CapabilitySet { tokens }
}

pub fn policy_gate(decision: &DecisionFrame) -> bool {
    !matches!(decision.decision, DecisionCode::Deny)
        && decision.deny_reason != Some(DenyReasonCode::PolicyViolation)
}
