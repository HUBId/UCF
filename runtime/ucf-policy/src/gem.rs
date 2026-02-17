use metrics::counter;
use sha2::{Digest, Sha256};
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

const TOOL_KIND_SLOTS: usize = 9;
const ISSUANCE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GovernanceSignals {
    pub t: u64,
    pub risk: f32,
    pub confidence: f32,
    pub nsr_risk: Option<f32>,
    pub coherence: Option<f32>,
    pub instability: Option<f32>,
    pub pressure: f32,
    pub surprise: f32,
    pub lfm_uncertainty: Option<f32>,
    pub lfm_stability: Option<f32>,
    pub hormone_stress: Option<f32>,
    pub digest: [u8; 32],
}

impl GovernanceSignals {
    pub fn from_inputs(
        decision: Option<&DecisionFrame>,
        t: u64,
        nsr_risk: Option<f32>,
        hormone_stress: Option<f32>,
    ) -> Self {
        let summary = decision.and_then(|d| d.compute_summary);
        let mut out = Self {
            t,
            risk: summary.map(|s| s.risk).unwrap_or(1.0),
            confidence: summary.map(|s| s.confidence).unwrap_or(0.0),
            nsr_risk,
            coherence: summary.and_then(|s| s.coherence),
            instability: summary.and_then(|s| s.instability),
            pressure: summary.map(|s| s.pressure).unwrap_or(1.0),
            surprise: summary.map(|s| s.surprise).unwrap_or(1.0),
            lfm_uncertainty: summary.and_then(|s| s.lfm_uncertainty),
            lfm_stability: summary.and_then(|s| s.lfm_stability),
            hormone_stress,
            digest: [0; 32],
        };
        out.risk = out.risk.clamp(0.0, 1.0);
        out.confidence = out.confidence.clamp(0.0, 1.0);
        out.pressure = out.pressure.clamp(0.0, 1.0);
        out.surprise = out.surprise.clamp(0.0, 1.0);
        out.nsr_risk = out.nsr_risk.map(|v| v.clamp(0.0, 1.0));
        out.coherence = out.coherence.map(|v| v.clamp(0.0, 1.0));
        out.instability = out.instability.map(|v| v.clamp(0.0, 1.0));
        out.lfm_uncertainty = out.lfm_uncertainty.map(|v| v.clamp(0.0, 1.0));
        out.lfm_stability = out.lfm_stability.map(|v| v.clamp(0.0, 1.0));
        out.hormone_stress = out.hormone_stress.map(|v| v.clamp(0.0, 1.0));
        out.digest = out.compute_digest();
        out
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.t.to_le_bytes());
        hasher.update(self.risk.to_bits().to_le_bytes());
        hasher.update(self.confidence.to_bits().to_le_bytes());
        put_opt_f32(&mut hasher, self.nsr_risk);
        put_opt_f32(&mut hasher, self.coherence);
        put_opt_f32(&mut hasher, self.instability);
        hasher.update(self.pressure.to_bits().to_le_bytes());
        hasher.update(self.surprise.to_bits().to_le_bytes());
        put_opt_f32(&mut hasher, self.lfm_uncertainty);
        put_opt_f32(&mut hasher, self.lfm_stability);
        put_opt_f32(&mut hasher, self.hormone_stress);
        hasher.finalize().into()
    }
}

fn put_opt_f32(hasher: &mut Sha256, value: Option<f32>) {
    if let Some(v) = value {
        hasher.update([1]);
        hasher.update(v.to_bits().to_le_bytes());
    } else {
        hasher.update([0]);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssuanceTier {
    Tier0,
    Tier1,
    Tier2,
    Tier3,
}

impl IssuanceTier {
    pub fn from_score(score: f32) -> Self {
        if score < 0.25 {
            Self::Tier0
        } else if score < 0.5 {
            Self::Tier1
        } else if score < 0.75 {
            Self::Tier2
        } else {
            Self::Tier3
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Self::Tier0 => 0,
            Self::Tier1 => 1,
            Self::Tier2 => 2,
            Self::Tier3 => 3,
        }
    }
}

pub fn governor_score(signals: GovernanceSignals) -> f32 {
    (0.35 * signals.nsr_risk.unwrap_or(signals.risk)
        + 0.20 * (1.0 - signals.coherence.unwrap_or(1.0))
        + 0.20 * signals.instability.unwrap_or(0.0)
        + 0.15 * signals.lfm_uncertainty.unwrap_or(0.0)
        + 0.10 * signals.hormone_stress.unwrap_or(0.0))
    .clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolGovernorSlot {
    pub tokens: u16,
    pub cooldown_ticks: u16,
    pub deny_count: u16,
}

#[derive(Clone, Debug)]
pub struct ToolGovernor {
    slots: [ToolGovernorSlot; TOOL_KIND_SLOTS],
    last_tick: u64,
}

impl Default for ToolGovernor {
    fn default() -> Self {
        let mut slots = [ToolGovernorSlot {
            tokens: 0,
            cooldown_ticks: 0,
            deny_count: 0,
        }; TOOL_KIND_SLOTS];
        for (idx, slot) in slots.iter_mut().enumerate() {
            slot.tokens = bucket_cfg(slot_kind(idx)).capacity;
        }
        Self {
            slots,
            last_tick: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct BucketCfg {
    capacity: u16,
    refill_per_tick: u16,
    cooldown_step: u16,
    cooldown_max: u16,
}

fn bucket_cfg(kind: CapabilityKind) -> BucketCfg {
    match kind {
        CapabilityKind::FileRead => BucketCfg {
            capacity: 3,
            refill_per_tick: 1,
            cooldown_step: 1,
            cooldown_max: 8,
        },
        CapabilityKind::FileWrite => BucketCfg {
            capacity: 2,
            refill_per_tick: 1,
            cooldown_step: 2,
            cooldown_max: 12,
        },
        CapabilityKind::UiAutomation => BucketCfg {
            capacity: 1,
            refill_per_tick: 1,
            cooldown_step: 2,
            cooldown_max: 12,
        },
        CapabilityKind::ExternalApi | CapabilityKind::NetHttp => BucketCfg {
            capacity: 1,
            refill_per_tick: 1,
            cooldown_step: 3,
            cooldown_max: 16,
        },
        CapabilityKind::ProcessExec => BucketCfg {
            capacity: 0,
            refill_per_tick: 0,
            cooldown_step: 4,
            cooldown_max: 32,
        },
        CapabilityKind::ClipboardRead
        | CapabilityKind::ClipboardWrite
        | CapabilityKind::Custom(_) => BucketCfg {
            capacity: 1,
            refill_per_tick: 1,
            cooldown_step: 2,
            cooldown_max: 10,
        },
    }
}

fn kind_slot(kind: &CapabilityKind) -> usize {
    match kind {
        CapabilityKind::NetHttp => 0,
        CapabilityKind::FileRead => 1,
        CapabilityKind::FileWrite => 2,
        CapabilityKind::ProcessExec => 3,
        CapabilityKind::ClipboardRead => 4,
        CapabilityKind::ClipboardWrite => 5,
        CapabilityKind::UiAutomation => 6,
        CapabilityKind::ExternalApi => 7,
        CapabilityKind::Custom(_) => 8,
    }
}

fn slot_kind(slot: usize) -> CapabilityKind {
    match slot {
        0 => CapabilityKind::NetHttp,
        1 => CapabilityKind::FileRead,
        2 => CapabilityKind::FileWrite,
        3 => CapabilityKind::ProcessExec,
        4 => CapabilityKind::ClipboardRead,
        5 => CapabilityKind::ClipboardWrite,
        6 => CapabilityKind::UiAutomation,
        7 => CapabilityKind::ExternalApi,
        _ => CapabilityKind::Custom("custom".to_string()),
    }
}

impl ToolGovernor {
    pub fn on_tick(&mut self, t: u64) {
        if t <= self.last_tick {
            return;
        }
        let dt = (t - self.last_tick).min(32);
        for (idx, slot) in self.slots.iter_mut().enumerate() {
            let cfg = bucket_cfg(slot_kind(idx));
            let refill = cfg.refill_per_tick.saturating_mul(dt as u16);
            slot.tokens = slot.tokens.saturating_add(refill).min(cfg.capacity);
            slot.cooldown_ticks = slot.cooldown_ticks.saturating_sub(dt as u16);
        }
        self.last_tick = t;
    }

    fn allow_and_consume(&mut self, kind: &CapabilityKind) -> Result<(), &'static str> {
        let idx = kind_slot(kind);
        let cfg = bucket_cfg(slot_kind(idx));
        let slot = &mut self.slots[idx];
        if slot.cooldown_ticks > 0 {
            slot.deny_count = slot.deny_count.saturating_add(1).min(1024);
            return Err("cooldown_active");
        }
        if slot.tokens == 0 {
            slot.deny_count = slot.deny_count.saturating_add(1).min(1024);
            slot.cooldown_ticks =
                (slot.cooldown_ticks.saturating_add(cfg.cooldown_step)).min(cfg.cooldown_max);
            return Err("token_exhausted");
        }
        slot.tokens = slot.tokens.saturating_sub(1);
        slot.deny_count = slot.deny_count.saturating_sub(1);
        Ok(())
    }

    fn deny(&mut self, kind: &CapabilityKind) {
        let idx = kind_slot(kind);
        let cfg = bucket_cfg(slot_kind(idx));
        let slot = &mut self.slots[idx];
        slot.deny_count = slot.deny_count.saturating_add(1).min(1024);
        let escalation = cfg.cooldown_step.saturating_add(slot.deny_count / 2);
        slot.cooldown_ticks = slot
            .cooldown_ticks
            .saturating_add(escalation)
            .min(cfg.cooldown_max);
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.last_tick.to_le_bytes());
        for slot in self.slots {
            hasher.update(slot.tokens.to_le_bytes());
            hasher.update(slot.cooldown_ticks.to_le_bytes());
            hasher.update(slot.deny_count.to_le_bytes());
        }
        hasher.finalize().into()
    }

    pub fn slot(&self, kind: &CapabilityKind) -> ToolGovernorSlot {
        self.slots[kind_slot(kind)]
    }

    pub fn snapshot(&self) -> [(CapabilityKind, ToolGovernorSlot); TOOL_KIND_SLOTS] {
        std::array::from_fn(|idx| (slot_kind(idx), self.slots[idx]))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityDecisionItem {
    pub kind: CapabilityKind,
    pub reason_code: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityIssuanceDecision {
    pub requested_kinds: Vec<CapabilityKind>,
    pub granted_kinds: Vec<CapabilityKind>,
    pub denied: Vec<CapabilityDecisionItem>,
    pub tier: IssuanceTier,
    pub governor_score: f32,
    pub governance_signals_digest: [u8; 32],
    pub throttle_state_digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
    pub schema_version: u16,
}

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
    pub candidate_id: Option<u16>,
    pub tool_intent_digest: Option<[u8; 32]>,
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
    pub policy_bundle_hash: Option<String>,
}

impl ToolGate {
    pub fn new(
        capabilities: CapabilitySet,
        rate_limiter: RateLimiter,
        policy_bundle_hash: Option<String>,
    ) -> Self {
        Self {
            capabilities,
            rate_limiter,
            policy_bundle_hash,
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

        if self.policy_bundle_hash.is_none() {
            counter!("ucf_tool_denied_total", "reason" => "policy_bundle_unverified".to_string())
                .increment(1);
            return AuthorizationOutcome::Denied {
                reason: CapabilityDenyReason::PolicyBundleUnverified,
            };
        }

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
            None,
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
        candidate_id: None,
        tool_intent_digest: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn request_from_intent(
    decision: &DecisionFrame,
    decision_id: u64,
    request_id: u64,
    kind: CapabilityKind,
    target: String,
    payload_hint: PayloadHint,
    candidate_id: u16,
    tool_intent_digest: [u8; 32],
) -> ToolRequest {
    let evidence_chain_digest = decision
        .compute_summary
        .and_then(|s| s.compute_chain_digest)
        .unwrap_or([0u8; 32]);

    ToolRequest {
        id: request_id,
        kind,
        target,
        payload_hint,
        requested_at_t: decision.time.tick.get(),
        decision_id,
        evidence_chain_digest,
        candidate_id: Some(candidate_id),
        tool_intent_digest: Some(tool_intent_digest),
    }
}

pub fn issue_capabilities(decision: Option<&DecisionFrame>, now_t: u64) -> CapabilitySet {
    let mut governor = ToolGovernor::default();
    let signals = GovernanceSignals::from_inputs(decision, now_t, None, None);
    issue_capabilities_governed(decision, now_t, signals, &mut governor).0
}

pub fn issue_capabilities_governed(
    decision: Option<&DecisionFrame>,
    now_t: u64,
    signals: GovernanceSignals,
    governor: &mut ToolGovernor,
) -> (CapabilitySet, CapabilityIssuanceDecision) {
    let Some(decision) = decision else {
        let denied = vec![CapabilityDecisionItem {
            kind: CapabilityKind::Custom("all".to_string()),
            reason_code: "missing_decision",
        }];
        return (
            CapabilitySet::empty(),
            CapabilityIssuanceDecision {
                requested_kinds: Vec::new(),
                granted_kinds: Vec::new(),
                denied,
                tier: IssuanceTier::Tier3,
                governor_score: 1.0,
                governance_signals_digest: signals.digest,
                throttle_state_digest: governor.digest(),
                evidence_chain_digest: [0; 32],
                schema_version: ISSUANCE_SCHEMA_VERSION,
            },
        );
    };
    governor.on_tick(now_t);
    let score = governor_score(signals);
    let tier = IssuanceTier::from_score(score);
    counter!("ucf_governor_tier", "tier" => tier.as_u8().to_string()).increment(1);
    metrics::gauge!("ucf_governor_score").set(f64::from(score));
    let mut tokens = Vec::new();
    let mut granted_kinds = Vec::new();
    let mut denied = Vec::new();
    let requested = [
        CapabilityKind::FileRead,
        CapabilityKind::FileWrite,
        CapabilityKind::ExternalApi,
        CapabilityKind::UiAutomation,
    ];

    for kind in requested.iter().cloned() {
        let maybe_token =
            base_policy_token(&kind, now_t, tier, decision.decision == DecisionCode::Allow);
        match maybe_token {
            Some(token) => match governor.allow_and_consume(&kind) {
                Ok(()) => {
                    counter!("ucf_cap_issued_total", "kind" => kind.as_tag().to_string())
                        .increment(1);
                    granted_kinds.push(kind);
                    tokens.push(token);
                }
                Err(reason_code) => {
                    counter!("ucf_cap_denied_total", "kind" => kind.as_tag().to_string(), "reason" => reason_code.to_string()).increment(1);
                    denied.push(CapabilityDecisionItem { kind, reason_code });
                }
            },
            None => {
                governor.deny(&kind);
                let reason_code = "tier_policy";
                counter!("ucf_cap_denied_total", "kind" => kind.as_tag().to_string(), "reason" => reason_code.to_string()).increment(1);
                denied.push(CapabilityDecisionItem { kind, reason_code });
            }
        }
    }
    let throttle_state_digest = governor.digest();
    let evidence_chain_digest = decision
        .compute_summary
        .and_then(|s| s.compute_chain_digest)
        .unwrap_or([0; 32]);

    (
        CapabilitySet { tokens },
        CapabilityIssuanceDecision {
            requested_kinds: requested.to_vec(),
            granted_kinds,
            denied,
            tier,
            governor_score: score,
            governance_signals_digest: signals.digest,
            throttle_state_digest,
            evidence_chain_digest,
            schema_version: ISSUANCE_SCHEMA_VERSION,
        },
    )
}

fn base_policy_token(
    kind: &CapabilityKind,
    now_t: u64,
    tier: IssuanceTier,
    allow_decision: bool,
) -> Option<CapabilityToken> {
    if !allow_decision || matches!(tier, IssuanceTier::Tier3) {
        return None;
    }
    match kind {
        CapabilityKind::FileRead
            if matches!(
                tier,
                IssuanceTier::Tier0 | IssuanceTier::Tier1 | IssuanceTier::Tier2
            ) =>
        {
            Some(CapabilityToken::issue(
                CapabilityKind::FileRead,
                CapabilityScope::Paths(vec!["/workspace/UCF/config".to_string()]),
                CapabilityLimits {
                    max_calls_per_window: 2,
                    window_ticks: 10,
                    max_bytes_out: None,
                    max_bytes_in: Some(1024),
                    max_concurrent: 1,
                },
                "pbm_v0",
                now_t,
                Some(now_t.saturating_add(10)),
            ))
        }
        CapabilityKind::FileWrite if matches!(tier, IssuanceTier::Tier0 | IssuanceTier::Tier1) => {
            Some(CapabilityToken::issue(
                CapabilityKind::FileWrite,
                CapabilityScope::Paths(vec!["memory_write".to_string()]),
                CapabilityLimits {
                    max_calls_per_window: if matches!(tier, IssuanceTier::Tier0) {
                        2
                    } else {
                        1
                    },
                    window_ticks: 10,
                    max_bytes_out: Some(1024),
                    max_bytes_in: None,
                    max_concurrent: 1,
                },
                "pbm_v0",
                now_t,
                Some(now_t.saturating_add(10)),
            ))
        }
        CapabilityKind::ExternalApi if matches!(tier, IssuanceTier::Tier0) => {
            Some(CapabilityToken::issue(
                CapabilityKind::ExternalApi,
                CapabilityScope::ApiNames(vec!["external_output".to_string()]),
                CapabilityLimits {
                    max_calls_per_window: 2,
                    window_ticks: 10,
                    max_bytes_out: Some(2048),
                    max_bytes_in: Some(1024),
                    max_concurrent: 1,
                },
                "pbm_v0",
                now_t,
                Some(now_t.saturating_add(10)),
            ))
        }
        CapabilityKind::UiAutomation
            if matches!(tier, IssuanceTier::Tier0 | IssuanceTier::Tier1) =>
        {
            Some(CapabilityToken::issue(
                CapabilityKind::UiAutomation,
                CapabilityScope::ApiNames(vec!["brain_target".to_string()]),
                CapabilityLimits {
                    max_calls_per_window: 1,
                    window_ticks: 10,
                    max_bytes_out: Some(64),
                    max_bytes_in: None,
                    max_concurrent: 1,
                },
                "pbm_v0",
                now_t,
                Some(now_t.saturating_add(10)),
            ))
        }
        _ => None,
    }
}

pub fn policy_gate(decision: &DecisionFrame) -> bool {
    !matches!(decision.decision, DecisionCode::Deny)
        && decision.deny_reason != Some(DenyReasonCode::PolicyViolation)
}
