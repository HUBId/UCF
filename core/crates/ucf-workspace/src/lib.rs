#![forbid(unsafe_code)]

use blake3::Hasher;
use std::cmp::Ordering;
use ucf_attn_controller::{AttentionUpdated, FocusChannel};
use ucf_output_router::{OutputRouterEvent, RouteDecision};
use ucf_policy_ecology::{ConsistencyReport, ConsistencyVerdict, RiskDecision, RiskGateResult};
use ucf_sleep_coordinator::{SleepTrigger, SleepTriggered};
use ucf_types::v1::spec::{ActionCode, DecisionKind, PolicyDecision};
use ucf_types::Digest32;

const SUMMARY_MAX_BYTES: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum SignalKind {
    World,
    Policy,
    Risk,
    Attention,
    Integration,
    Consistency,
    Output,
    Sleep,
}

impl SignalKind {
    const COUNT: usize = 8;

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceSignal {
    pub kind: SignalKind,
    pub priority: u16,
    pub digest: Digest32,
    pub summary: String,
}

/// Workspace signal helpers.
///
/// # Priority scheme
/// - Base priority is `1000`.
/// - High threat/risk (`risk >= 7000` or attention channel `Threat`) is raised to `>= 9000`.
/// - Policy denials are raised to `>= 9500` to guarantee broadcast.
/// - Attention gain boosts related signals by `gain / 5` (capped at `2000`).
///
/// # Summary format
/// - Summaries are compact, uppercase tokens like `RISK=4200 DENY` or `POLICY=DENY`.
/// - Summaries never contain user-provided text; only categorical labels and digests.
impl WorkspaceSignal {
    pub fn from_policy_decision(decision: &PolicyDecision, attention_gain: Option<u16>) -> Self {
        let kind = SignalKind::Policy;
        let decision_kind =
            DecisionKind::try_from(decision.kind).unwrap_or(DecisionKind::DecisionKindUnspecified);
        let action =
            ActionCode::try_from(decision.action).unwrap_or(ActionCode::ActionCodeUnspecified);
        let summary = format!(
            "POLICY={} ACT={}",
            decision_kind_token(decision_kind),
            action_token(action)
        );
        let digest = digest_policy_decision(decision_kind, action, decision.confidence_bp);
        let priority = priority_with_attention(policy_priority(decision_kind), attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }

    pub fn from_risk_result(result: &RiskGateResult, attention_gain: Option<u16>) -> Self {
        let kind = SignalKind::Risk;
        let summary = format!(
            "RISK={} {}",
            result.risk,
            risk_decision_token(result.decision)
        );
        let digest = digest_risk_result(result);
        let base_priority = risk_priority(result.risk, result.decision);
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }

    pub fn from_attention_update(update: &AttentionUpdated) -> Self {
        let kind = SignalKind::Attention;
        let summary = format!(
            "ATTN={} GAIN={}",
            focus_channel_token(update.channel),
            update.gain
        );
        let priority = attention_priority(update.channel, update.gain);
        Self {
            kind,
            priority,
            digest: update.commit,
            summary,
        }
    }

    pub fn from_integration_score(score: u16, attention_gain: Option<u16>) -> Self {
        let kind = SignalKind::Integration;
        let summary = format!("IIT={score}");
        let digest = digest_integration_score(score);
        let base_priority = integration_priority(score);
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }

    pub fn from_consistency_report(
        report: &ConsistencyReport,
        attention_gain: Option<u16>,
    ) -> Self {
        let kind = SignalKind::Consistency;
        let summary = format!(
            "NSR={} SCORE={}",
            consistency_verdict_token(report.verdict),
            report.score
        );
        let digest = digest_consistency_report(report);
        let base_priority = consistency_priority(report.verdict);
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }

    pub fn from_consistency_verdict(
        verdict: ConsistencyVerdict,
        attention_gain: Option<u16>,
    ) -> Self {
        let kind = SignalKind::Consistency;
        let summary = format!("NSR={}", consistency_verdict_token(verdict));
        let digest = digest_consistency_verdict(verdict);
        let base_priority = consistency_priority(verdict);
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }

    pub fn from_output_event(event: &OutputRouterEvent, attention_gain: Option<u16>) -> Self {
        let kind = SignalKind::Output;
        let (summary, digest, base_priority) = output_event_payload(event);
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }

    pub fn from_output_decision(decision: &RouteDecision, attention_gain: Option<u16>) -> Self {
        let kind = SignalKind::Output;
        let reason = output_reason_token(&decision.reason_code);
        let summary = if decision.permitted {
            format!("OUTPUT=ALLOW REASON={reason}")
        } else {
            format!("OUTPUT=BLOCK REASON={reason}")
        };
        let digest = digest_output_decision(decision);
        let base_priority = if decision.permitted { 3500 } else { 6000 };
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }

    pub fn from_sleep_triggered(triggered: &SleepTriggered, attention_gain: Option<u16>) -> Self {
        let kind = SignalKind::Sleep;
        let summary = format!(
            "SLEEP=TRIGGERED REASON={}",
            sleep_trigger_token(triggered.reason)
        );
        let digest = digest_sleep_triggered(triggered);
        let base_priority = sleep_priority(triggered.reason);
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }

    pub fn from_world_state_digest(digest: Digest32) -> Self {
        let kind = SignalKind::World;
        let summary = format!("WORLD=STATE DIG={digest}");
        let priority = 3000;
        Self {
            kind,
            priority,
            digest,
            summary,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceConfig {
    pub cap: usize,
    pub broadcast_cap: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceSnapshot {
    pub cycle_id: u64,
    pub broadcast: Vec<WorkspaceSignal>,
    pub commit: Digest32,
}

/// Encode a workspace snapshot into a compact, deterministic payload for archiving.
///
/// Summaries included here are already sanitized and non-sensitive (categorical labels
/// and digests only), making this safe to store without raw user content.
pub fn encode_workspace_snapshot(snapshot: &WorkspaceSnapshot) -> Vec<u8> {
    const SNAPSHOT_DOMAIN_TAG: u16 = 0x5753;
    let signals = if snapshot.broadcast.len() > u16::MAX as usize {
        &snapshot.broadcast[..u16::MAX as usize]
    } else {
        snapshot.broadcast.as_slice()
    };
    let mut payload = Vec::with_capacity(
        2 + 8 + 2 + signals.len() * (1 + 2 + Digest32::LEN + 2 + SUMMARY_MAX_BYTES),
    );
    payload.extend_from_slice(&SNAPSHOT_DOMAIN_TAG.to_be_bytes());
    payload.extend_from_slice(&snapshot.cycle_id.to_be_bytes());
    payload.extend_from_slice(&(signals.len() as u16).to_be_bytes());
    for signal in signals {
        payload.push(signal.kind as u8);
        payload.extend_from_slice(&signal.priority.to_be_bytes());
        payload.extend_from_slice(signal.digest.as_bytes());
        let summary_bytes = signal.summary.as_bytes();
        let summary_len = summary_bytes.len().min(SUMMARY_MAX_BYTES);
        payload.extend_from_slice(&(summary_len as u16).to_be_bytes());
        payload.extend_from_slice(&summary_bytes[..summary_len]);
    }
    payload
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DropCounters {
    pub total: u64,
    pub by_kind: [u64; SignalKind::COUNT],
}

pub struct Workspace {
    config: WorkspaceConfig,
    signals: Vec<SignalEntry>,
    next_seq: u64,
    drops: DropCounters,
}

impl Workspace {
    pub fn new(config: WorkspaceConfig) -> Self {
        Self {
            config,
            signals: Vec::new(),
            next_seq: 0,
            drops: DropCounters::default(),
        }
    }

    pub fn drop_counters(&self) -> DropCounters {
        self.drops
    }

    pub fn publish(&mut self, mut sig: WorkspaceSignal) {
        sig.summary = sanitize_summary(&sig.summary);
        let entry = SignalEntry {
            signal: sig,
            seq: self.next_seq,
        };
        self.next_seq = self.next_seq.wrapping_add(1);
        self.signals.push(entry);
        if self.signals.len() > self.config.cap {
            self.drop_excess();
        }
    }

    pub fn arbitrate(&mut self, cycle_id: u64) -> WorkspaceSnapshot {
        let mut entries = std::mem::take(&mut self.signals);
        entries.sort_by(compare_for_broadcast);
        let broadcast_len = self.config.broadcast_cap.min(entries.len());
        let broadcast_entries: Vec<SignalEntry> = entries.drain(..broadcast_len).collect();
        for entry in entries {
            self.note_drop(entry.signal.kind);
        }
        let broadcast: Vec<WorkspaceSignal> = broadcast_entries
            .iter()
            .map(|entry| entry.signal.clone())
            .collect();
        let commit = commit_snapshot(cycle_id, &broadcast);
        WorkspaceSnapshot {
            cycle_id,
            broadcast,
            commit,
        }
    }

    fn drop_excess(&mut self) {
        if let Some(idx) = self
            .signals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| compare_for_drop(a, b))
            .map(|(idx, _)| idx)
        {
            let entry = self.signals.remove(idx);
            self.note_drop(entry.signal.kind);
        }
    }

    fn note_drop(&mut self, kind: SignalKind) {
        self.drops.total += 1;
        self.drops.by_kind[kind.index()] += 1;
    }
}

#[derive(Clone, Debug)]
struct SignalEntry {
    signal: WorkspaceSignal,
    seq: u64,
}

const PRIORITY_BASE: u16 = 1000;
const PRIORITY_HIGH: u16 = 9000;
const PRIORITY_POLICY_DENY: u16 = 9500;
const ATTENTION_BOOST_CAP: u16 = 2000;

fn priority_with_attention(base: u16, attention_gain: Option<u16>) -> u16 {
    base.saturating_add(attention_boost(attention_gain))
}

fn attention_boost(attention_gain: Option<u16>) -> u16 {
    attention_gain
        .map(|gain| (gain / 5).min(ATTENTION_BOOST_CAP))
        .unwrap_or(0)
}

fn policy_priority(kind: DecisionKind) -> u16 {
    match kind {
        DecisionKind::DecisionKindDeny => PRIORITY_POLICY_DENY,
        DecisionKind::DecisionKindEscalate => 8500,
        DecisionKind::DecisionKindObserve => 4500,
        DecisionKind::DecisionKindAllow | DecisionKind::DecisionKindUnspecified => 3500,
    }
}

fn risk_priority(risk: u16, decision: RiskDecision) -> u16 {
    let base = if risk >= 7000 {
        PRIORITY_HIGH.saturating_add((risk - 7000) / 2)
    } else {
        PRIORITY_BASE.saturating_add(risk / 3)
    };
    match decision {
        RiskDecision::Permit => base,
        RiskDecision::Deny => base.max(PRIORITY_HIGH),
    }
}

fn attention_priority(channel: FocusChannel, gain: u16) -> u16 {
    let boost = (gain / 4).min(ATTENTION_BOOST_CAP);
    match channel {
        FocusChannel::Threat => PRIORITY_HIGH.saturating_add(boost),
        _ => PRIORITY_BASE.saturating_add(boost),
    }
}

fn integration_priority(score: u16) -> u16 {
    let inverted = 10_000u16.saturating_sub(score);
    PRIORITY_BASE.saturating_add(inverted / 5)
}

fn consistency_priority(verdict: ConsistencyVerdict) -> u16 {
    match verdict {
        ConsistencyVerdict::Accept => 3000,
        ConsistencyVerdict::Damp => 6000,
        ConsistencyVerdict::Reject => 8500,
    }
}

fn sleep_priority(reason: SleepTrigger) -> u16 {
    match reason {
        SleepTrigger::Instability | SleepTrigger::LowIntegration => 7000,
        SleepTrigger::Density => 6000,
        SleepTrigger::Manual => 5500,
        SleepTrigger::None => 3000,
    }
}

fn output_event_payload(event: &OutputRouterEvent) -> (String, Digest32, u16) {
    match event {
        OutputRouterEvent::ThoughtBuffered { frame } => (
            "OUTPUT=THOUGHT BUFFERED".to_string(),
            digest_output_event(b"thought_buffered", frame.commit, None, 0),
            3000,
        ),
        OutputRouterEvent::SpeechEmitted { frame } => (
            "OUTPUT=SPEECH EMIT".to_string(),
            digest_output_event(b"speech_emitted", frame.commit, None, 0),
            5000,
        ),
        OutputRouterEvent::OutputSuppressed {
            frame,
            reason_code,
            evidence,
            risk,
        } => {
            let reason = output_reason_token(reason_code);
            let summary = format!("OUTPUT=SUPPRESS REASON={reason} RISK={risk}");
            let digest =
                digest_output_event(b"output_suppressed", frame.commit, Some(*evidence), *risk);
            let mut base_priority = if *risk >= 7000 { PRIORITY_HIGH } else { 6000 };
            if reason == "POLICY" {
                base_priority = base_priority.max(PRIORITY_POLICY_DENY);
            }
            (summary, digest, base_priority)
        }
    }
}

fn decision_kind_token(kind: DecisionKind) -> &'static str {
    match kind {
        DecisionKind::DecisionKindAllow => "ALLOW",
        DecisionKind::DecisionKindDeny => "DENY",
        DecisionKind::DecisionKindEscalate => "ESCALATE",
        DecisionKind::DecisionKindObserve => "OBSERVE",
        DecisionKind::DecisionKindUnspecified => "UNSPEC",
    }
}

fn action_token(action: ActionCode) -> &'static str {
    match action {
        ActionCode::ActionCodeContinue => "CONTINUE",
        ActionCode::ActionCodePause => "PAUSE",
        ActionCode::ActionCodeTerminate => "TERMINATE",
        ActionCode::ActionCodeRequireHuman => "REQUIRE_HUMAN",
        ActionCode::ActionCodeUnspecified => "UNSPEC",
    }
}

fn risk_decision_token(decision: RiskDecision) -> &'static str {
    match decision {
        RiskDecision::Permit => "PERMIT",
        RiskDecision::Deny => "DENY",
    }
}

fn focus_channel_token(channel: FocusChannel) -> &'static str {
    match channel {
        FocusChannel::Threat => "THREAT",
        FocusChannel::Task => "TASK",
        FocusChannel::Social => "SOCIAL",
        FocusChannel::Memory => "MEMORY",
        FocusChannel::Exploration => "EXPLORE",
        FocusChannel::Idle => "IDLE",
    }
}

fn consistency_verdict_token(verdict: ConsistencyVerdict) -> &'static str {
    match verdict {
        ConsistencyVerdict::Accept => "OK",
        ConsistencyVerdict::Damp => "DAMP",
        ConsistencyVerdict::Reject => "VIOL",
    }
}

fn sleep_trigger_token(trigger: SleepTrigger) -> &'static str {
    match trigger {
        SleepTrigger::None => "NONE",
        SleepTrigger::Instability => "INSTABILITY",
        SleepTrigger::Density => "DENSITY",
        SleepTrigger::LowIntegration => "LOW_INTEGRATION",
        SleepTrigger::Manual => "MANUAL",
    }
}

fn sleep_trigger_code(trigger: SleepTrigger) -> u8 {
    match trigger {
        SleepTrigger::None => 0,
        SleepTrigger::Instability => 1,
        SleepTrigger::Density => 2,
        SleepTrigger::LowIntegration => 3,
        SleepTrigger::Manual => 4,
    }
}

fn output_reason_token(reason_code: &str) -> &'static str {
    let reason = reason_code.to_ascii_lowercase();
    if reason.contains("policy") {
        "POLICY"
    } else if reason.contains("risk") {
        "RISK"
    } else if reason.contains("sandbox") {
        "SANDBOX"
    } else if reason.contains("thought") && reason.contains("cap") {
        "CAP"
    } else {
        "OTHER"
    }
}

fn digest_policy_decision(kind: DecisionKind, action: ActionCode, confidence_bp: u32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.policy.v1");
    hasher.update(&(kind as i32).to_be_bytes());
    hasher.update(&(action as i32).to_be_bytes());
    hasher.update(&confidence_bp.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_risk_result(result: &RiskGateResult) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.risk.v1");
    hasher.update(&[matches!(result.decision, RiskDecision::Deny) as u8]);
    hasher.update(&result.risk.to_be_bytes());
    hasher.update(result.evidence.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_integration_score(score: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.integration.v1");
    hasher.update(&score.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_consistency_report(report: &ConsistencyReport) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.consistency.v1");
    hasher.update(&report.score.to_be_bytes());
    hasher.update(&[report.verdict.as_u8()]);
    hasher.update(&(report.matched_anchors as u64).to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_consistency_verdict(verdict: ConsistencyVerdict) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.consistency.verdict.v1");
    hasher.update(&[verdict.as_u8()]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_output_event(
    tag: &[u8],
    frame_commit: Digest32,
    evidence: Option<Digest32>,
    risk: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.output.event.v1");
    hasher.update(tag);
    hasher.update(frame_commit.as_bytes());
    if let Some(evidence) = evidence {
        hasher.update(evidence.as_bytes());
    }
    hasher.update(&risk.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_output_decision(decision: &RouteDecision) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.output.decision.v1");
    hasher.update(&[decision.permitted as u8]);
    hasher.update(output_reason_token(&decision.reason_code).as_bytes());
    hasher.update(decision.evidence.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_sleep_triggered(triggered: &SleepTriggered) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.sleep.v1");
    hasher.update(&triggered.cycle_id.to_be_bytes());
    hasher.update(&[sleep_trigger_code(triggered.reason)]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn compare_for_drop(a: &SignalEntry, b: &SignalEntry) -> Ordering {
    let priority_cmp = a.signal.priority.cmp(&b.signal.priority);
    if priority_cmp != Ordering::Equal {
        return priority_cmp;
    }
    let kind_cmp = a.signal.kind.cmp(&b.signal.kind);
    if kind_cmp != Ordering::Equal {
        return kind_cmp;
    }
    let digest_cmp = a.signal.digest.as_bytes().cmp(b.signal.digest.as_bytes());
    if digest_cmp != Ordering::Equal {
        return digest_cmp;
    }
    a.seq.cmp(&b.seq)
}

fn compare_for_broadcast(a: &SignalEntry, b: &SignalEntry) -> Ordering {
    let priority_cmp = b.signal.priority.cmp(&a.signal.priority);
    if priority_cmp != Ordering::Equal {
        return priority_cmp;
    }
    let kind_cmp = a.signal.kind.cmp(&b.signal.kind);
    if kind_cmp != Ordering::Equal {
        return kind_cmp;
    }
    let digest_cmp = a.signal.digest.as_bytes().cmp(b.signal.digest.as_bytes());
    if digest_cmp != Ordering::Equal {
        return digest_cmp;
    }
    a.seq.cmp(&b.seq)
}

fn commit_snapshot(cycle_id: u64, broadcast: &[WorkspaceSignal]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&cycle_id.to_be_bytes());
    for signal in broadcast {
        hasher.update(&[signal.kind as u8]);
        hasher.update(&signal.priority.to_be_bytes());
        hasher.update(signal.digest.as_bytes());
        hasher.update(signal.summary.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn sanitize_summary(summary: &str) -> String {
    let mut cleaned = String::with_capacity(summary.len().min(SUMMARY_MAX_BYTES));
    for ch in summary.chars() {
        if ch.is_control() {
            continue;
        }
        cleaned.push(ch);
        if cleaned.len() >= SUMMARY_MAX_BYTES {
            break;
        }
    }
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "summary redacted".to_string()
    } else {
        trimmed.to_string()
    }
}
