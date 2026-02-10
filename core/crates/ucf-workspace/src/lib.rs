#![forbid(unsafe_code)]

use blake3::Hasher;
use std::cmp::Ordering;
use ucf_attn_controller::{AttentionUpdated, FocusChannel};
use ucf_consistency_engine::{ConsistencyReport as DriftReport, DriftBand};
use ucf_iit::IitOutput;
use ucf_output_router::{OutputRouterEvent, RouteDecision};
use ucf_policy_ecology::{
    ConsistencyReport as PolicyConsistencyReport, ConsistencyVerdict, RiskDecision, RiskGateResult,
};
use ucf_predictive_coding::{SurpriseBand, SurpriseUpdated};
use ucf_sleep_coordinator::{SleepTrigger, SleepTriggered};
use ucf_spikebus::{SpikeKind, SpikeOutputs};
use ucf_structural_store::StructuralDeltaProposal;
use ucf_types::v1::spec::{ActionCode, DecisionKind, PolicyDecision};
use ucf_types::{Digest32, LearningSignal, StructuralDelta};

const SUMMARY_MAX_BYTES: usize = 160;
const NSR_HIT_SUMMARY_MAX: usize = 8;
const CDE_TOP_EDGES_MAX: usize = 8;

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
    Brain,
    Sleep,
}

impl SignalKind {
    const COUNT: usize = 9;

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
    pub slot: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InternalUtterance {
    pub commit: Digest32,
    pub src: Digest32,
    pub severity: u16,
}

impl InternalUtterance {
    pub fn new(src: Digest32, severity: u16) -> Self {
        let commit = digest_internal_utterance(src, severity);
        Self {
            commit,
            src,
            severity,
        }
    }
}

/// Workspace signal helpers.
///
/// # Priority scheme
/// - Base priority is `1000`.
/// - High threat/risk (`risk >= 7000` or attention channel `Threat`) is raised to `>= 9000`.
/// - Policy denials are raised to `>= 9500` to guarantee broadcast.
/// - Attention gain boosts related signals by `gain / 5` (capped at `2000`).
/// - Surprise signals follow the band priority scale.
///
/// # Summary format
/// - Summaries are compact, uppercase tokens like `RISK=4200 DENY` or `POLICY=DENY`.
/// - Summaries never contain user-provided text; only categorical labels and digests.
impl WorkspaceSignal {
    pub fn from_policy_decision(
        decision: &PolicyDecision,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
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
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_world_state(digest: Digest32, slot: Option<u8>) -> Self {
        let kind = SignalKind::World;
        let summary = format!("SSM state={digest}");
        Self {
            kind,
            priority: 3000,
            digest,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_risk_result(
        result: &RiskGateResult,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
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
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_attention_update(update: &AttentionUpdated, slot: Option<u8>) -> Self {
        let kind = SignalKind::Attention;
        let mut summary = format!(
            "ATTN={} GAIN={}",
            focus_channel_token(update.channel),
            update.gain
        );
        if let Some(commit) = update.wm_commit {
            summary.push_str(&format!(" WM={commit}"));
        }
        let priority = attention_priority(update.channel, update.gain);
        Self {
            kind,
            priority,
            digest: update.commit,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_integration_score(
        score: u16,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
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
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_influence_update(
        node_count: usize,
        root_commit: Digest32,
        pulses_root: Digest32,
        outputs_commit: Digest32,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
        let kind = SignalKind::Integration;
        let summary = format!("INFL nodes={node_count} root={root_commit} pulses={pulses_root}");
        let base_priority = 2400;
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest: outputs_commit,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_consistency_report(
        report: &PolicyConsistencyReport,
        attention_gain: Option<u16>,
        slot: Option<u8>,
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
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_consistency_drift(
        report: &DriftReport,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
        let kind = SignalKind::Consistency;
        let summary = format!(
            "DRIFT={} SCORE={}",
            drift_band_token(report.band),
            report.drift_score
        );
        let digest = report.commit;
        let base_priority = drift_priority(report.band);
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_consistency_verdict(
        verdict: ConsistencyVerdict,
        attention_gain: Option<u16>,
        slot: Option<u8>,
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
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_sle_reflex(
        loop_level: u8,
        delta: i16,
        commit: Digest32,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
        let kind = SignalKind::Consistency;
        let summary = format!("SLE=LEVEL {loop_level} DELTA={delta}");
        let base_priority = 4000u16.saturating_add(u16::from(loop_level).saturating_mul(500));
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest: commit,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_recursion_budget(
        max_depth: u8,
        per_cycle_steps: u16,
        commit: Digest32,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
        let kind = SignalKind::Consistency;
        let summary = format!("RDC depth={max_depth} steps={per_cycle_steps}");
        let base_priority = 4200u16.saturating_add(u16::from(max_depth).saturating_mul(250));
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest: commit,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_output_event(
        event: &OutputRouterEvent,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
        let kind = SignalKind::Output;
        let (summary, digest, base_priority) = output_event_payload(event);
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_output_decision(
        decision: &RouteDecision,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
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
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_brain_stimulated(stim_commit: Digest32, slot: Option<u8>) -> Self {
        let kind = SignalKind::Brain;
        let summary = format!("BRAIN_STIM={stim_commit}");
        Self {
            kind,
            priority: 7000,
            digest: stim_commit,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_brain_responded(
        resp_commit: Digest32,
        arousal: u16,
        valence: i16,
        slot: Option<u8>,
    ) -> Self {
        let kind = SignalKind::Brain;
        let summary = format!("BRAIN_RESP={resp_commit} AROUSAL={arousal} VALENCE={valence}");
        let priority = 7500u16.saturating_add(arousal / 2).min(10_000);
        Self {
            kind,
            priority,
            digest: resp_commit,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_sleep_triggered(
        triggered: &SleepTriggered,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
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
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_sleep_proposals(
        proposal_count: usize,
        digest: Digest32,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
        let kind = SignalKind::Sleep;
        let summary = format!("SLEEP=PROPOSALS COUNT={proposal_count}");
        let base_priority = if proposal_count > 0 { 6500 } else { 2000 };
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_replay_summary(
        micro_count: usize,
        meso_count: usize,
        macro_count: usize,
        attention_gain: Option<u16>,
        slot: Option<u8>,
    ) -> Self {
        let kind = SignalKind::Sleep;
        let summary = format!("REPLAY micro={micro_count} meso={meso_count} macro={macro_count}");
        let digest = digest_replay_summary(micro_count, meso_count, macro_count);
        let base_priority = if micro_count + meso_count + macro_count > 0 {
            7000
        } else {
            2500
        };
        let priority = priority_with_attention(base_priority, attention_gain);
        Self {
            kind,
            priority,
            digest,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_world_state_digest(digest: Digest32, slot: Option<u8>) -> Self {
        let kind = SignalKind::World;
        let summary = format!("WORLD=STATE DIG={digest}");
        let priority = 3000;
        Self {
            kind,
            priority,
            digest,
            summary,
            slot: slot.unwrap_or(0),
        }
    }

    pub fn from_surprise_update(update: &SurpriseUpdated, slot: Option<u8>) -> Self {
        let kind = SignalKind::World;
        let summary = format!(
            "SURPRISE={} BAND={}",
            update.score,
            surprise_band_token(update.band)
        );
        let priority = surprise_priority(update.band);
        Self {
            kind,
            priority,
            digest: update.commit,
            summary,
            slot: slot.unwrap_or(0),
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
    pub recursion_used: u16,
    pub spike_accepted_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub spike_max_intensity: u16,
    pub ncde_commit: Digest32,
    pub ncde_state_digest: Digest32,
    pub ncde_energy: u16,
    pub replay_pressure_hint: u16,
    pub cde_commit: Digest32,
    pub cde_graph_commit: Digest32,
    pub cde_top_edges: Vec<(u16, u16, u16, u8)>,
    pub cde_top_edge_commits: Vec<Digest32>,
    pub cde_intervention_commit: Option<Digest32>,
    pub cde_observation_commit: Digest32,
    pub cde_last_query_result: Option<Digest32>,
    pub ssm_commit: Digest32,
    pub ssm_state_commit: Digest32,
    pub ssm_state_digest: Digest32,
    pub ssm_salience: u16,
    pub ssm_novelty: u16,
    pub ssm_attention_gain: u16,
    pub jepa_world_state: Digest32,
    pub jepa_prediction: Digest32,
    pub jepa_surprise: u16,
    pub learning_signal_commit: Digest32,
    pub learning_signal_learn_rate: u16,
    pub learning_signal_mode: u8,
    pub structural_delta_commit: Digest32,
    pub structural_delta_mass: u16,
    pub structural_delta_targets: [u16; 4],
    pub influence_v2_commit: Digest32,
    pub influence_pulses_root: Digest32,
    pub influence_node_values: Vec<(u16, i16)>,
    pub coupling_influences_root: Digest32,
    pub coupling_top_influences: Vec<(u16, i16)>,
    pub coupling_lag_commits: Vec<(u16, Digest32)>,
    pub gain_budget_commit: Digest32,
    pub budget_low_plv_streak: u8,
    pub budget_high_novelty_streak: u8,
    pub budget_violation_streak: u8,
    pub budget_triggers: u8,
    pub tcf_plan_commit: Digest32,
    pub tcf_attention_gain_cap: u16,
    pub tcf_learning_gain_cap: u16,
    pub tcf_output_gain_cap: u16,
    pub tcf_sleep_active: bool,
    pub tcf_replay_active: bool,
    pub tcf_lock_window_buckets: u8,
    pub lock_window_source_cycle: u64,
    pub coherence_lag_commit: Digest32,
    pub update_mode: u8,
    pub onn_phase_commit: Digest32,
    pub onn_gamma_bucket: u8,
    pub onn_global_plv: u16,
    pub iit_output: Option<IitOutput>,
    pub nsr_trace_root: Option<Digest32>,
    pub nsr_prev_commit: Option<Digest32>,
    pub nsr_verdict: Option<u8>,
    pub nsr_triggered_rules_root: Option<Digest32>,
    pub nsr_derived_facts_root: Option<Digest32>,
    pub nsr_fact_flags: u8,
    pub nsr_hit_counts: [u16; 3],
    pub nsr_hit_summaries: Vec<NsrHitSummary>,
    pub rsa_commit: Digest32,
    pub rsa_proposal_commit: Option<Digest32>,
    pub rsa_decision_apply: bool,
    pub rsa_apply_allowed: bool,
    pub rsa_apply_commit: Digest32,
    pub rsa_reason_mask: u32,
    pub rsa_applied_params_root: Digest32,
    pub rsa_snapshot_chain_commit: Digest32,
    pub sle_commit: Digest32,
    pub sle_reflection_commit: Digest32,
    pub sle_reflection_class: u8,
    pub sle_reflection_intensity: u16,
    pub sle_thought_only_root: Digest32,
    pub sle_ssm_bias: i16,
    pub sle_cde_bias: i16,
    pub sle_request_replay: bool,
    pub internal_utterances: Vec<InternalUtterance>,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SleOutputsSnapshot {
    pub sle_commit: Digest32,
    pub reflection_commit: Digest32,
    pub reflection_class: u8,
    pub reflection_intensity: u16,
    pub thought_only_root: Digest32,
    pub ssm_bias: i16,
    pub cde_bias: i16,
    pub request_replay: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrHitSummary {
    pub rule_id: u16,
    pub severity: u8,
    pub reason: u16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrTraceSummary {
    pub trace_root: Digest32,
    pub prev_commit: Option<Digest32>,
    pub verdict: u8,
    pub derived_facts_root: Option<Digest32>,
    pub triggered_rules_root: Option<Digest32>,
    pub fact_flags: u8,
    pub hit_counts: [u16; 3],
    pub hit_summaries: Vec<NsrHitSummary>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoherenceSummary {
    pub cycle_id: u64,
    pub gamma_bucket: u8,
    pub plv: u16,
    pub lock_window: u8,
    pub surprise: u16,
    pub novelty: u16,
    pub salience: u16,
    pub attention_gain: u16,
    pub learn_rate: u16,
    pub learn_mode: u8,
    pub delta_mass: u16,
    pub delta_targets: [u16; 4],
    pub nsr_verdict: Option<u8>,
    pub nsr_hit_counts: [u16; 3],
    pub violations: Vec<NsrHitSummary>,
}

impl CoherenceSummary {
    pub fn from_snapshot(snapshot: &WorkspaceSnapshot) -> Self {
        Self {
            cycle_id: snapshot.cycle_id,
            gamma_bucket: snapshot.onn_gamma_bucket,
            plv: snapshot.onn_global_plv,
            lock_window: snapshot.tcf_lock_window_buckets,
            surprise: snapshot.jepa_surprise,
            novelty: snapshot.ssm_novelty,
            salience: snapshot.ssm_salience,
            attention_gain: snapshot.ssm_attention_gain,
            learn_rate: snapshot.learning_signal_learn_rate,
            learn_mode: snapshot.learning_signal_mode,
            delta_mass: snapshot.structural_delta_mass,
            delta_targets: snapshot.structural_delta_targets,
            nsr_verdict: snapshot.nsr_verdict,
            nsr_hit_counts: snapshot.nsr_hit_counts,
            violations: snapshot.nsr_hit_summaries.clone(),
        }
    }
}

/// Encode a workspace snapshot into a compact, deterministic payload for archiving.
///
/// Summaries included here are already sanitized and non-sensitive (categorical labels
/// and digests only), making this safe to store without raw user content.
pub fn encode_workspace_snapshot(snapshot: &WorkspaceSnapshot) -> Vec<u8> {
    const SNAPSHOT_DOMAIN_TAG: u16 = 0x5753;
    let cde_edges = if snapshot.cde_top_edges.len() > CDE_TOP_EDGES_MAX {
        &snapshot.cde_top_edges[..CDE_TOP_EDGES_MAX]
    } else {
        snapshot.cde_top_edges.as_slice()
    };
    let cde_edge_commits = if snapshot.cde_top_edge_commits.len() > CDE_TOP_EDGES_MAX {
        &snapshot.cde_top_edge_commits[..CDE_TOP_EDGES_MAX]
    } else {
        snapshot.cde_top_edge_commits.as_slice()
    };
    let signals = if snapshot.broadcast.len() > u16::MAX as usize {
        &snapshot.broadcast[..u16::MAX as usize]
    } else {
        snapshot.broadcast.as_slice()
    };
    let mut payload = Vec::new();
    payload.extend_from_slice(&SNAPSHOT_DOMAIN_TAG.to_be_bytes());
    payload.extend_from_slice(&snapshot.cycle_id.to_be_bytes());
    payload.extend_from_slice(&(signals.len() as u16).to_be_bytes());
    payload.extend_from_slice(&snapshot.recursion_used.to_be_bytes());
    payload.extend_from_slice(snapshot.spike_accepted_root.as_bytes());
    payload.extend_from_slice(
        &u16::try_from(snapshot.spike_counts.len())
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    for (kind, count) in &snapshot.spike_counts {
        payload.extend_from_slice(&kind.as_u16().to_be_bytes());
        payload.extend_from_slice(&count.to_be_bytes());
    }
    payload.extend_from_slice(&snapshot.spike_max_intensity.to_be_bytes());
    payload.extend_from_slice(snapshot.ncde_commit.as_bytes());
    payload.extend_from_slice(snapshot.ncde_state_digest.as_bytes());
    payload.extend_from_slice(&snapshot.ncde_energy.to_be_bytes());
    payload.extend_from_slice(&snapshot.replay_pressure_hint.to_be_bytes());
    payload.extend_from_slice(snapshot.cde_commit.as_bytes());
    payload.extend_from_slice(snapshot.cde_graph_commit.as_bytes());
    payload.extend_from_slice(&(cde_edges.len() as u16).to_be_bytes());
    for (from, to, conf, lag) in cde_edges {
        payload.extend_from_slice(&from.to_be_bytes());
        payload.extend_from_slice(&to.to_be_bytes());
        payload.extend_from_slice(&conf.to_be_bytes());
        payload.push(*lag);
    }
    payload.extend_from_slice(&(cde_edge_commits.len() as u16).to_be_bytes());
    for commit in cde_edge_commits {
        payload.extend_from_slice(commit.as_bytes());
    }
    match snapshot.cde_intervention_commit {
        Some(commit) => {
            payload.push(1);
            payload.extend_from_slice(commit.as_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
        }
    }
    payload.extend_from_slice(snapshot.cde_observation_commit.as_bytes());
    match snapshot.cde_last_query_result {
        Some(commit) => {
            payload.push(1);
            payload.extend_from_slice(commit.as_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
        }
    }
    payload.extend_from_slice(snapshot.ssm_commit.as_bytes());
    payload.extend_from_slice(snapshot.ssm_state_commit.as_bytes());
    payload.extend_from_slice(snapshot.ssm_state_digest.as_bytes());
    payload.extend_from_slice(&snapshot.ssm_salience.to_be_bytes());
    payload.extend_from_slice(&snapshot.ssm_novelty.to_be_bytes());
    payload.extend_from_slice(&snapshot.ssm_attention_gain.to_be_bytes());
    payload.extend_from_slice(snapshot.jepa_world_state.as_bytes());
    payload.extend_from_slice(snapshot.jepa_prediction.as_bytes());
    payload.extend_from_slice(&snapshot.jepa_surprise.to_be_bytes());
    payload.extend_from_slice(snapshot.learning_signal_commit.as_bytes());
    payload.extend_from_slice(&snapshot.learning_signal_learn_rate.to_be_bytes());
    payload.push(snapshot.learning_signal_mode);
    payload.extend_from_slice(snapshot.structural_delta_commit.as_bytes());
    payload.extend_from_slice(&snapshot.structural_delta_mass.to_be_bytes());
    for target in snapshot.structural_delta_targets {
        payload.extend_from_slice(&target.to_be_bytes());
    }
    payload.extend_from_slice(snapshot.influence_v2_commit.as_bytes());
    payload.extend_from_slice(snapshot.influence_pulses_root.as_bytes());
    payload.extend_from_slice(&(snapshot.influence_node_values.len() as u16).to_be_bytes());
    for (node, value) in &snapshot.influence_node_values {
        payload.extend_from_slice(&node.to_be_bytes());
        payload.extend_from_slice(&value.to_be_bytes());
    }
    payload.extend_from_slice(snapshot.coupling_influences_root.as_bytes());
    payload.extend_from_slice(&(snapshot.coupling_top_influences.len() as u16).to_be_bytes());
    for (signal, value) in &snapshot.coupling_top_influences {
        payload.extend_from_slice(&signal.to_be_bytes());
        payload.extend_from_slice(&value.to_be_bytes());
    }
    payload.extend_from_slice(&(snapshot.coupling_lag_commits.len() as u16).to_be_bytes());
    for (signal, commit) in &snapshot.coupling_lag_commits {
        payload.extend_from_slice(&signal.to_be_bytes());
        payload.extend_from_slice(commit.as_bytes());
    }
    payload.extend_from_slice(snapshot.gain_budget_commit.as_bytes());
    payload.push(snapshot.budget_low_plv_streak);
    payload.push(snapshot.budget_high_novelty_streak);
    payload.push(snapshot.budget_violation_streak);
    payload.push(snapshot.budget_triggers);
    payload.extend_from_slice(snapshot.tcf_plan_commit.as_bytes());
    payload.extend_from_slice(&snapshot.tcf_attention_gain_cap.to_be_bytes());
    payload.extend_from_slice(&snapshot.tcf_learning_gain_cap.to_be_bytes());
    payload.extend_from_slice(&snapshot.tcf_output_gain_cap.to_be_bytes());
    payload.push(snapshot.tcf_sleep_active as u8);
    payload.push(snapshot.tcf_replay_active as u8);
    payload.push(snapshot.tcf_lock_window_buckets);
    payload.extend_from_slice(&snapshot.lock_window_source_cycle.to_be_bytes());
    payload.extend_from_slice(snapshot.coherence_lag_commit.as_bytes());
    payload.push(snapshot.update_mode);
    payload.extend_from_slice(snapshot.onn_phase_commit.as_bytes());
    payload.push(snapshot.onn_gamma_bucket);
    payload.extend_from_slice(&snapshot.onn_global_plv.to_be_bytes());
    match snapshot.iit_output.as_ref() {
        Some(output) => {
            payload.push(1);
            payload.extend_from_slice(output.commit.as_bytes());
            payload.extend_from_slice(&output.phi_proxy.to_be_bytes());
            payload.push(iit_hint_flags(output));
            payload.extend_from_slice(output.hints_commit.as_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
            payload.extend_from_slice(&0u16.to_be_bytes());
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
        }
    }
    match snapshot.nsr_trace_root {
        Some(commit) => {
            payload.push(1);
            payload.extend_from_slice(commit.as_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
        }
    }
    match snapshot.nsr_prev_commit {
        Some(commit) => {
            payload.push(1);
            payload.extend_from_slice(commit.as_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
        }
    }
    payload.push(snapshot.nsr_verdict.unwrap_or(0));
    match snapshot.nsr_triggered_rules_root {
        Some(commit) => {
            payload.push(1);
            payload.extend_from_slice(commit.as_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
        }
    }
    match snapshot.nsr_derived_facts_root {
        Some(commit) => {
            payload.push(1);
            payload.extend_from_slice(commit.as_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
        }
    }
    payload.push(snapshot.nsr_fact_flags);
    for count in snapshot.nsr_hit_counts {
        payload.extend_from_slice(&count.to_be_bytes());
    }
    payload.extend_from_slice(
        &u16::try_from(snapshot.nsr_hit_summaries.len())
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    for summary in &snapshot.nsr_hit_summaries {
        payload.extend_from_slice(&summary.rule_id.to_be_bytes());
        payload.push(summary.severity);
        payload.extend_from_slice(&summary.reason.to_be_bytes());
        payload.extend_from_slice(summary.commit.as_bytes());
    }
    payload.extend_from_slice(snapshot.rsa_commit.as_bytes());
    match snapshot.rsa_proposal_commit {
        Some(commit) => {
            payload.push(1);
            payload.extend_from_slice(commit.as_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0u8; Digest32::LEN]);
        }
    }
    payload.push(snapshot.rsa_decision_apply as u8);
    payload.push(snapshot.rsa_apply_allowed as u8);
    payload.extend_from_slice(snapshot.rsa_apply_commit.as_bytes());
    payload.extend_from_slice(&snapshot.rsa_reason_mask.to_be_bytes());
    payload.extend_from_slice(snapshot.rsa_applied_params_root.as_bytes());
    payload.extend_from_slice(snapshot.rsa_snapshot_chain_commit.as_bytes());
    payload.extend_from_slice(snapshot.sle_commit.as_bytes());
    payload.extend_from_slice(snapshot.sle_reflection_commit.as_bytes());
    payload.push(snapshot.sle_reflection_class);
    payload.extend_from_slice(&snapshot.sle_reflection_intensity.to_be_bytes());
    payload.extend_from_slice(snapshot.sle_thought_only_root.as_bytes());
    payload.extend_from_slice(&snapshot.sle_ssm_bias.to_be_bytes());
    payload.extend_from_slice(&snapshot.sle_cde_bias.to_be_bytes());
    payload.push(snapshot.sle_request_replay as u8);
    payload.extend_from_slice(
        &u16::try_from(snapshot.internal_utterances.len())
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    for utterance in &snapshot.internal_utterances {
        payload.extend_from_slice(utterance.commit.as_bytes());
        payload.extend_from_slice(utterance.src.as_bytes());
        payload.extend_from_slice(&utterance.severity.to_be_bytes());
    }
    for signal in signals {
        payload.push(signal.kind as u8);
        payload.push(signal.slot);
        payload.extend_from_slice(&signal.priority.to_be_bytes());
        payload.extend_from_slice(signal.digest.as_bytes());
        let summary_bytes = signal.summary.as_bytes();
        let summary_len = summary_bytes.len().min(SUMMARY_MAX_BYTES);
        payload.extend_from_slice(&(summary_len as u16).to_be_bytes());
        payload.extend_from_slice(&summary_bytes[..summary_len]);
    }
    payload
}

fn iit_hint_flags(output: &IitOutput) -> u8 {
    (output.tighten_sync as u8)
        | ((output.damp_output as u8) << 1)
        | ((output.damp_learning as u8) << 2)
        | ((output.request_replay as u8) << 3)
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
    recursion_used: u16,
    spike_cycle_id: u64,
    spike_accepted_root: Digest32,
    spike_counts: Vec<(SpikeKind, u16)>,
    spike_max_intensity: u16,
    structural_proposal: Option<StructuralDeltaProposal>,
    rsa_commit: Digest32,
    rsa_proposal_commit: Option<Digest32>,
    rsa_decision_apply: bool,
    rsa_apply_allowed: bool,
    rsa_apply_commit: Digest32,
    rsa_reason_mask: u32,
    rsa_applied_params_root: Digest32,
    rsa_snapshot_chain_commit: Digest32,
    ncde_commit: Digest32,
    ncde_state_digest: Digest32,
    ncde_energy: u16,
    replay_pressure_hint: u16,
    cde_commit: Digest32,
    cde_graph_commit: Digest32,
    cde_top_edges: Vec<(u16, u16, u16, u8)>,
    cde_top_edge_commits: Vec<Digest32>,
    cde_intervention_commit: Option<Digest32>,
    cde_observation_commit: Digest32,
    cde_last_query_result: Option<Digest32>,
    ssm_commit: Digest32,
    ssm_state_commit: Digest32,
    ssm_state_digest: Digest32,
    ssm_salience: u16,
    ssm_novelty: u16,
    ssm_attention_gain: u16,
    jepa_world_state: Digest32,
    jepa_prediction: Digest32,
    jepa_surprise: u16,
    learning_signal_commit: Digest32,
    learning_signal_learn_rate: u16,
    learning_signal_mode: u8,
    structural_delta_commit: Digest32,
    structural_delta_mass: u16,
    structural_delta_targets: [u16; 4],
    influence_v2_commit: Digest32,
    influence_pulses_root: Digest32,
    influence_node_values: Vec<(u16, i16)>,
    coupling_influences_root: Digest32,
    coupling_top_influences: Vec<(u16, i16)>,
    coupling_lag_commits: Vec<(u16, Digest32)>,
    gain_budget_commit: Digest32,
    budget_low_plv_streak: u8,
    budget_high_novelty_streak: u8,
    budget_violation_streak: u8,
    budget_triggers: u8,
    tcf_plan_commit: Digest32,
    tcf_attention_gain_cap: u16,
    tcf_learning_gain_cap: u16,
    tcf_output_gain_cap: u16,
    tcf_sleep_active: bool,
    tcf_replay_active: bool,
    tcf_lock_window_buckets: u8,
    lock_window_source_cycle: u64,
    coherence_lag_commit: Digest32,
    update_mode: u8,
    onn_phase_commit: Digest32,
    onn_gamma_bucket: u8,
    onn_global_plv: u16,
    iit_output: Option<IitOutput>,
    nsr_trace_root: Option<Digest32>,
    nsr_prev_commit: Option<Digest32>,
    nsr_verdict: Option<u8>,
    nsr_triggered_rules_root: Option<Digest32>,
    nsr_derived_facts_root: Option<Digest32>,
    nsr_fact_flags: u8,
    nsr_hit_counts: [u16; 3],
    nsr_hit_summaries: Vec<NsrHitSummary>,
    sle_commit: Digest32,
    sle_reflection_commit: Digest32,
    sle_reflection_class: u8,
    sle_reflection_intensity: u16,
    sle_thought_only_root: Digest32,
    sle_ssm_bias: i16,
    sle_cde_bias: i16,
    sle_request_replay: bool,
    internal_utterances: Vec<InternalUtterance>,
}

impl Workspace {
    pub fn new(config: WorkspaceConfig) -> Self {
        Self {
            config,
            signals: Vec::new(),
            next_seq: 0,
            drops: DropCounters::default(),
            recursion_used: 0,
            spike_cycle_id: 0,
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_max_intensity: 0,
            structural_proposal: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_proposal_commit: None,
            rsa_decision_apply: false,
            rsa_apply_allowed: false,
            rsa_apply_commit: Digest32::new([0u8; 32]),
            rsa_reason_mask: 0,
            rsa_applied_params_root: Digest32::new([0u8; 32]),
            rsa_snapshot_chain_commit: Digest32::new([0u8; 32]),
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_digest: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            replay_pressure_hint: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            cde_top_edge_commits: Vec::new(),
            cde_intervention_commit: None,
            cde_observation_commit: Digest32::new([0u8; 32]),
            cde_last_query_result: None,
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            ssm_state_digest: Digest32::new([0u8; 32]),
            ssm_salience: 0,
            ssm_novelty: 0,
            ssm_attention_gain: 0,
            jepa_world_state: Digest32::new([0u8; 32]),
            jepa_prediction: Digest32::new([0u8; 32]),
            jepa_surprise: 0,
            learning_signal_commit: Digest32::new([0u8; 32]),
            learning_signal_learn_rate: 0,
            learning_signal_mode: 0,
            structural_delta_commit: Digest32::new([0u8; 32]),
            structural_delta_mass: 0,
            structural_delta_targets: [0; 4],
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            coupling_influences_root: Digest32::new([0u8; 32]),
            coupling_top_influences: Vec::new(),
            coupling_lag_commits: Vec::new(),
            gain_budget_commit: Digest32::new([0u8; 32]),
            budget_low_plv_streak: 0,
            budget_high_novelty_streak: 0,
            budget_violation_streak: 0,
            budget_triggers: 0,
            tcf_plan_commit: Digest32::new([0u8; 32]),
            tcf_attention_gain_cap: 0,
            tcf_learning_gain_cap: 0,
            tcf_output_gain_cap: 0,
            tcf_sleep_active: false,
            tcf_replay_active: false,
            tcf_lock_window_buckets: 0,
            lock_window_source_cycle: 0,
            coherence_lag_commit: Digest32::new([0u8; 32]),
            update_mode: 0,
            onn_phase_commit: Digest32::new([0u8; 32]),
            onn_gamma_bucket: 0,
            onn_global_plv: 0,
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            nsr_triggered_rules_root: None,
            nsr_derived_facts_root: None,
            nsr_fact_flags: 0,
            nsr_hit_counts: [0u16; 3],
            nsr_hit_summaries: Vec::new(),
            sle_commit: Digest32::new([0u8; 32]),
            sle_reflection_commit: Digest32::new([0u8; 32]),
            sle_reflection_class: 0,
            sle_reflection_intensity: 0,
            sle_thought_only_root: Digest32::new([0u8; 32]),
            sle_ssm_bias: 0,
            sle_cde_bias: 0,
            sle_request_replay: false,
            internal_utterances: Vec::new(),
        }
    }

    pub fn drop_counters(&self) -> DropCounters {
        self.drops
    }

    pub fn coherence_summary(&self) -> CoherenceSummary {
        CoherenceSummary {
            cycle_id: self.spike_cycle_id,
            gamma_bucket: self.onn_gamma_bucket,
            plv: self.onn_global_plv,
            lock_window: self.tcf_lock_window_buckets,
            surprise: self.jepa_surprise,
            novelty: self.ssm_novelty,
            salience: self.ssm_salience,
            attention_gain: self.ssm_attention_gain,
            learn_rate: self.learning_signal_learn_rate,
            learn_mode: self.learning_signal_mode,
            delta_mass: self.structural_delta_mass,
            delta_targets: self.structural_delta_targets,
            nsr_verdict: self.nsr_verdict,
            nsr_hit_counts: self.nsr_hit_counts,
            violations: self.nsr_hit_summaries.clone(),
        }
    }

    pub fn record_recursion_used(&mut self, used: u16) {
        self.recursion_used = used;
    }

    pub fn set_broadcast_cap(&mut self, broadcast_cap: usize) {
        self.config.broadcast_cap = broadcast_cap.min(self.config.cap);
    }

    pub fn set_spike_outputs(&mut self, outputs: SpikeOutputs) {
        self.spike_cycle_id = outputs.cycle_id;
        self.spike_accepted_root = outputs.accepted_root;
        self.spike_counts = outputs.counts;
        self.spike_max_intensity = outputs.max_intensity;
    }

    pub fn set_structural_proposal(&mut self, proposal: StructuralDeltaProposal) {
        self.structural_proposal = Some(proposal);
    }

    pub fn take_structural_proposal(&mut self) -> Option<StructuralDeltaProposal> {
        self.structural_proposal.take()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_rsa_output(
        &mut self,
        rsa_commit: Digest32,
        proposal_commit: Option<Digest32>,
        decision_apply: bool,
        apply_allowed: bool,
        apply_commit: Digest32,
        reason_mask: u32,
        applied_params_root: Digest32,
        snapshot_chain_commit: Digest32,
    ) {
        self.rsa_commit = rsa_commit;
        self.rsa_proposal_commit = proposal_commit;
        self.rsa_decision_apply = decision_apply;
        self.rsa_apply_allowed = apply_allowed;
        self.rsa_apply_commit = apply_commit;
        self.rsa_reason_mask = reason_mask;
        self.rsa_applied_params_root = applied_params_root;
        self.rsa_snapshot_chain_commit = snapshot_chain_commit;
    }

    pub fn set_coherence_state(&mut self, lag_commit: Digest32, update_mode: u8) {
        self.coherence_lag_commit = lag_commit;
        self.update_mode = update_mode;
    }

    pub fn spike_summary(&self) -> SpikeOutputs {
        SpikeOutputs {
            cycle_id: self.spike_cycle_id,
            accepted_root: self.spike_accepted_root,
            accepted: Vec::new(),
            counts: self.spike_counts.clone(),
            max_intensity: self.spike_max_intensity,
            commit: Digest32::new([0u8; 32]),
        }
    }

    pub fn set_ncde_snapshot(
        &mut self,
        commit: Digest32,
        state_digest: Digest32,
        energy: u16,
        replay_pressure_hint: u16,
    ) {
        self.ncde_commit = commit;
        self.ncde_state_digest = state_digest;
        self.ncde_energy = energy.min(10_000);
        self.replay_pressure_hint = replay_pressure_hint.min(10_000);
    }

    pub fn ncde_commit(&self) -> Digest32 {
        self.ncde_commit
    }

    pub fn ncde_state_digest(&self) -> Digest32 {
        self.ncde_state_digest
    }

    pub fn ncde_energy(&self) -> u16 {
        self.ncde_energy
    }

    pub fn ncde_replay_pressure_hint(&self) -> u16 {
        self.replay_pressure_hint
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_cde_output(
        &mut self,
        commit: Digest32,
        graph_commit: Digest32,
        top_edges: Vec<(u16, u16, u16, u8)>,
        top_edge_commits: Vec<Digest32>,
        intervention_commit: Option<Digest32>,
        observation_commit: Digest32,
        last_query_result: Option<Digest32>,
    ) {
        self.cde_commit = commit;
        self.cde_graph_commit = graph_commit;
        self.cde_top_edges = top_edges;
        self.cde_top_edge_commits = top_edge_commits;
        self.cde_intervention_commit = intervention_commit;
        self.cde_observation_commit = observation_commit;
        self.cde_last_query_result = last_query_result;
    }

    pub fn set_ssm_snapshot(
        &mut self,
        commit: Digest32,
        state_commit: Digest32,
        state_digest: Digest32,
        salience: u16,
        novelty: u16,
        attention_gain: u16,
    ) {
        self.ssm_commit = commit;
        self.ssm_state_commit = state_commit;
        self.ssm_state_digest = state_digest;
        self.ssm_salience = salience;
        self.ssm_novelty = novelty;
        self.ssm_attention_gain = attention_gain;
    }

    pub fn set_jepa_snapshot(
        &mut self,
        world_state: Digest32,
        prediction: Digest32,
        surprise: u16,
    ) {
        self.jepa_world_state = world_state;
        self.jepa_prediction = prediction;
        self.jepa_surprise = surprise.min(10_000);
    }

    pub fn set_ssm_commits(&mut self, commit: Digest32, state_commit: Digest32) {
        self.set_ssm_snapshot(
            commit,
            state_commit,
            self.ssm_state_digest,
            self.ssm_salience,
            self.ssm_novelty,
            self.ssm_attention_gain,
        );
    }

    pub fn set_learning_signal(&mut self, signal: LearningSignal) {
        self.learning_signal_commit = signal.commit;
        self.learning_signal_learn_rate = signal.learn_rate.min(10_000);
        self.learning_signal_mode = signal.mode.min(2);
    }

    pub fn set_structural_delta(&mut self, delta: StructuralDelta) {
        self.structural_delta_commit = delta.commit;
        self.structural_delta_mass = delta.delta_mass.min(10_000);
        self.structural_delta_targets = delta.targets;
    }

    pub fn set_influence_snapshot(
        &mut self,
        influence_commit: Digest32,
        pulses_root: Digest32,
        node_values: Vec<(u16, i16)>,
    ) {
        self.influence_v2_commit = influence_commit;
        self.influence_pulses_root = pulses_root;
        self.influence_node_values = node_values;
    }

    pub fn set_coupling_snapshot(
        &mut self,
        influences_root: Digest32,
        top_influences: Vec<(u16, i16)>,
        lag_commits: Vec<(u16, Digest32)>,
    ) {
        self.coupling_influences_root = influences_root;
        self.coupling_top_influences = top_influences;
        self.coupling_lag_commits = lag_commits;
    }

    pub fn set_gain_budget_state(
        &mut self,
        gain_budget_commit: Digest32,
        low_plv_streak: u8,
        high_novelty_streak: u8,
        violation_streak: u8,
        triggers: u8,
    ) {
        self.gain_budget_commit = gain_budget_commit;
        self.budget_low_plv_streak = low_plv_streak;
        self.budget_high_novelty_streak = high_novelty_streak;
        self.budget_violation_streak = violation_streak;
        self.budget_triggers = triggers;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_tcf_plan(
        &mut self,
        plan_commit: Digest32,
        attention_gain_cap: u16,
        learning_gain_cap: u16,
        output_gain_cap: u16,
        sleep_active: bool,
        replay_active: bool,
        lock_window_buckets: u8,
    ) {
        self.tcf_plan_commit = plan_commit;
        self.tcf_attention_gain_cap = attention_gain_cap.min(10_000);
        self.tcf_learning_gain_cap = learning_gain_cap.min(10_000);
        self.tcf_output_gain_cap = output_gain_cap.min(10_000);
        self.tcf_sleep_active = sleep_active;
        self.tcf_replay_active = replay_active;
        self.tcf_lock_window_buckets = lock_window_buckets;
    }

    pub fn set_onn_snapshot(
        &mut self,
        phase_commit: Digest32,
        gamma_bucket: u8,
        global_plv: u16,
        lock_window_source_cycle: u64,
    ) {
        self.onn_phase_commit = phase_commit;
        self.onn_gamma_bucket = gamma_bucket;
        self.onn_global_plv = global_plv;
        self.lock_window_source_cycle = lock_window_source_cycle;
    }

    pub fn set_iit_output(&mut self, output: IitOutput) {
        self.iit_output = Some(output);
    }

    pub fn set_nsr_trace(&mut self, mut summary: NsrTraceSummary) {
        self.nsr_trace_root = Some(summary.trace_root);
        self.nsr_prev_commit = summary.prev_commit;
        self.nsr_verdict = Some(summary.verdict);
        self.nsr_derived_facts_root = summary.derived_facts_root;
        self.nsr_triggered_rules_root = summary.triggered_rules_root;
        self.nsr_fact_flags = summary.fact_flags;
        self.nsr_hit_counts = summary.hit_counts;
        if summary.hit_summaries.len() > NSR_HIT_SUMMARY_MAX {
            summary.hit_summaries.truncate(NSR_HIT_SUMMARY_MAX);
        }
        self.nsr_hit_summaries = summary.hit_summaries;
    }

    pub fn set_sle_outputs(&mut self, outputs: SleOutputsSnapshot) {
        self.sle_commit = outputs.sle_commit;
        self.sle_reflection_commit = outputs.reflection_commit;
        self.sle_reflection_class = outputs.reflection_class;
        self.sle_reflection_intensity = outputs.reflection_intensity;
        self.sle_thought_only_root = outputs.thought_only_root;
        self.sle_ssm_bias = outputs.ssm_bias;
        self.sle_cde_bias = outputs.cde_bias;
        self.sle_request_replay = outputs.request_replay;
    }

    pub fn rsa_applied(&self) -> bool {
        self.rsa_decision_apply
    }

    pub fn sle_reflection_commit(&self) -> Digest32 {
        self.sle_reflection_commit
    }

    pub fn sle_ssm_bias(&self) -> i16 {
        self.sle_ssm_bias
    }

    pub fn sle_cde_bias(&self) -> i16 {
        self.sle_cde_bias
    }

    pub fn sle_request_replay(&self) -> bool {
        self.sle_request_replay
    }

    pub fn push_internal_utterance(&mut self, utterance: InternalUtterance) {
        self.internal_utterances.push(utterance);
    }

    pub fn ssm_commit(&self) -> Digest32 {
        self.ssm_commit
    }

    pub fn ssm_state_commit(&self) -> Digest32 {
        self.ssm_state_commit
    }

    pub fn ssm_state_digest(&self) -> Digest32 {
        self.ssm_state_digest
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
        let recursion_used = self.recursion_used;
        self.recursion_used = 0;
        let spike_summary = self.spike_summary();
        let ncde_commit = self.ncde_commit;
        let ncde_state_digest = self.ncde_state_digest;
        let ncde_energy = self.ncde_energy;
        let replay_pressure_hint = self.replay_pressure_hint;
        let cde_commit = self.cde_commit;
        let cde_graph_commit = self.cde_graph_commit;
        let cde_top_edges = std::mem::take(&mut self.cde_top_edges);
        let cde_top_edge_commits = std::mem::take(&mut self.cde_top_edge_commits);
        let cde_intervention_commit = self.cde_intervention_commit.take();
        let cde_observation_commit = self.cde_observation_commit;
        let cde_last_query_result = self.cde_last_query_result.take();
        let ssm_commit = self.ssm_commit;
        let ssm_state_commit = self.ssm_state_commit;
        let ssm_state_digest = self.ssm_state_digest;
        let ssm_salience = self.ssm_salience;
        let ssm_novelty = self.ssm_novelty;
        let ssm_attention_gain = self.ssm_attention_gain;
        let jepa_world_state = self.jepa_world_state;
        let jepa_prediction = self.jepa_prediction;
        let jepa_surprise = self.jepa_surprise;
        let learning_signal_commit = self.learning_signal_commit;
        let learning_signal_learn_rate = self.learning_signal_learn_rate;
        let learning_signal_mode = self.learning_signal_mode;
        let structural_delta_commit = self.structural_delta_commit;
        let structural_delta_mass = self.structural_delta_mass;
        let structural_delta_targets = self.structural_delta_targets;
        let influence_v2_commit = self.influence_v2_commit;
        let influence_pulses_root = self.influence_pulses_root;
        let influence_node_values = std::mem::take(&mut self.influence_node_values);
        let coupling_influences_root = self.coupling_influences_root;
        let coupling_top_influences = std::mem::take(&mut self.coupling_top_influences);
        let coupling_lag_commits = std::mem::take(&mut self.coupling_lag_commits);
        let gain_budget_commit = self.gain_budget_commit;
        let budget_low_plv_streak = self.budget_low_plv_streak;
        let budget_high_novelty_streak = self.budget_high_novelty_streak;
        let budget_violation_streak = self.budget_violation_streak;
        let budget_triggers = self.budget_triggers;
        let tcf_plan_commit = self.tcf_plan_commit;
        let tcf_attention_gain_cap = self.tcf_attention_gain_cap;
        let tcf_learning_gain_cap = self.tcf_learning_gain_cap;
        let tcf_output_gain_cap = self.tcf_output_gain_cap;
        let tcf_sleep_active = self.tcf_sleep_active;
        let tcf_replay_active = self.tcf_replay_active;
        let tcf_lock_window_buckets = self.tcf_lock_window_buckets;
        let lock_window_source_cycle = self.lock_window_source_cycle;
        let coherence_lag_commit = self.coherence_lag_commit;
        let update_mode = self.update_mode;
        let onn_phase_commit = self.onn_phase_commit;
        let onn_gamma_bucket = self.onn_gamma_bucket;
        let onn_global_plv = self.onn_global_plv;
        let iit_output = self.iit_output.take();
        let nsr_trace_root = self.nsr_trace_root.take();
        let nsr_prev_commit = self.nsr_prev_commit.take();
        let nsr_verdict = self.nsr_verdict.take();
        let nsr_triggered_rules_root = self.nsr_triggered_rules_root.take();
        let nsr_derived_facts_root = self.nsr_derived_facts_root.take();
        let nsr_fact_flags = self.nsr_fact_flags;
        let nsr_hit_counts = self.nsr_hit_counts;
        let nsr_hit_summaries = std::mem::take(&mut self.nsr_hit_summaries);
        let rsa_commit = self.rsa_commit;
        let rsa_proposal_commit = self.rsa_proposal_commit;
        let rsa_decision_apply = self.rsa_decision_apply;
        let rsa_apply_allowed = self.rsa_apply_allowed;
        let rsa_apply_commit = self.rsa_apply_commit;
        let rsa_reason_mask = self.rsa_reason_mask;
        let rsa_applied_params_root = self.rsa_applied_params_root;
        let rsa_snapshot_chain_commit = self.rsa_snapshot_chain_commit;
        let sle_commit = self.sle_commit;
        let sle_reflection_commit = self.sle_reflection_commit;
        let sle_reflection_class = self.sle_reflection_class;
        let sle_reflection_intensity = self.sle_reflection_intensity;
        let sle_thought_only_root = self.sle_thought_only_root;
        let sle_ssm_bias = self.sle_ssm_bias;
        let sle_cde_bias = self.sle_cde_bias;
        let sle_request_replay = self.sle_request_replay;
        let internal_utterances = std::mem::take(&mut self.internal_utterances);
        let commit = commit_snapshot(
            cycle_id,
            recursion_used,
            spike_summary.accepted_root,
            &spike_summary.counts,
            spike_summary.max_intensity,
            ncde_commit,
            ncde_state_digest,
            ncde_energy,
            replay_pressure_hint,
            cde_commit,
            cde_graph_commit,
            &cde_top_edges,
            &cde_top_edge_commits,
            cde_intervention_commit,
            cde_observation_commit,
            cde_last_query_result,
            ssm_commit,
            ssm_state_commit,
            ssm_state_digest,
            ssm_salience,
            ssm_novelty,
            ssm_attention_gain,
            jepa_world_state,
            jepa_prediction,
            jepa_surprise,
            learning_signal_commit,
            learning_signal_learn_rate,
            learning_signal_mode,
            structural_delta_commit,
            structural_delta_mass,
            structural_delta_targets,
            influence_v2_commit,
            influence_pulses_root,
            &influence_node_values,
            coupling_influences_root,
            &coupling_top_influences,
            &coupling_lag_commits,
            gain_budget_commit,
            budget_low_plv_streak,
            budget_high_novelty_streak,
            budget_violation_streak,
            budget_triggers,
            tcf_plan_commit,
            tcf_attention_gain_cap,
            tcf_learning_gain_cap,
            tcf_output_gain_cap,
            tcf_sleep_active,
            tcf_replay_active,
            tcf_lock_window_buckets,
            lock_window_source_cycle,
            coherence_lag_commit,
            update_mode,
            onn_phase_commit,
            onn_gamma_bucket,
            onn_global_plv,
            iit_output.as_ref(),
            nsr_trace_root,
            nsr_prev_commit,
            nsr_verdict,
            nsr_triggered_rules_root,
            nsr_derived_facts_root,
            nsr_fact_flags,
            nsr_hit_counts,
            &nsr_hit_summaries,
            rsa_commit,
            rsa_proposal_commit,
            rsa_decision_apply,
            rsa_apply_allowed,
            rsa_apply_commit,
            rsa_reason_mask,
            rsa_applied_params_root,
            rsa_snapshot_chain_commit,
            sle_commit,
            sle_reflection_commit,
            sle_reflection_class,
            sle_reflection_intensity,
            sle_thought_only_root,
            sle_ssm_bias,
            sle_cde_bias,
            sle_request_replay,
            &internal_utterances,
            &broadcast,
        );
        WorkspaceSnapshot {
            cycle_id,
            broadcast,
            recursion_used,
            spike_accepted_root: spike_summary.accepted_root,
            spike_counts: spike_summary.counts,
            spike_max_intensity: spike_summary.max_intensity,
            ncde_commit,
            ncde_state_digest,
            ncde_energy,
            replay_pressure_hint,
            cde_commit,
            cde_graph_commit,
            cde_top_edges,
            cde_top_edge_commits,
            cde_intervention_commit,
            cde_observation_commit,
            cde_last_query_result,
            ssm_commit,
            ssm_state_commit,
            ssm_state_digest,
            ssm_salience,
            ssm_novelty,
            ssm_attention_gain,
            jepa_world_state,
            jepa_prediction,
            jepa_surprise,
            learning_signal_commit,
            learning_signal_learn_rate,
            learning_signal_mode,
            structural_delta_commit,
            structural_delta_mass,
            structural_delta_targets,
            influence_v2_commit,
            influence_pulses_root,
            influence_node_values,
            coupling_influences_root,
            coupling_top_influences,
            coupling_lag_commits,
            gain_budget_commit,
            budget_low_plv_streak,
            budget_high_novelty_streak,
            budget_violation_streak,
            budget_triggers,
            tcf_plan_commit,
            tcf_attention_gain_cap,
            tcf_learning_gain_cap,
            tcf_output_gain_cap,
            tcf_sleep_active,
            tcf_replay_active,
            tcf_lock_window_buckets,
            lock_window_source_cycle,
            coherence_lag_commit,
            update_mode,
            onn_phase_commit,
            onn_gamma_bucket,
            onn_global_plv,
            iit_output,
            nsr_trace_root,
            nsr_prev_commit,
            nsr_verdict,
            nsr_triggered_rules_root,
            nsr_derived_facts_root,
            nsr_fact_flags,
            nsr_hit_counts,
            nsr_hit_summaries,
            rsa_commit,
            rsa_proposal_commit,
            rsa_decision_apply,
            rsa_apply_allowed,
            rsa_apply_commit,
            rsa_reason_mask,
            rsa_applied_params_root,
            rsa_snapshot_chain_commit,
            sle_commit,
            sle_reflection_commit,
            sle_reflection_class,
            sle_reflection_intensity,
            sle_thought_only_root,
            sle_ssm_bias,
            sle_cde_bias,
            sle_request_replay,
            internal_utterances,
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

fn drift_priority(band: DriftBand) -> u16 {
    match band {
        DriftBand::Low => 3500,
        DriftBand::Medium => 5000,
        DriftBand::High => 8000,
        DriftBand::Critical => 9500,
    }
}

fn sleep_priority(reason: SleepTrigger) -> u16 {
    match reason {
        SleepTrigger::Instability | SleepTrigger::LowIntegration => 7000,
        SleepTrigger::Density => 6000,
        SleepTrigger::Manual => 5500,
        SleepTrigger::SurpriseCritical => 9000,
        SleepTrigger::None => 3000,
    }
}

fn surprise_priority(band: SurpriseBand) -> u16 {
    match band {
        SurpriseBand::Low => 3000,
        SurpriseBand::Medium => 5000,
        SurpriseBand::High => 8000,
        SurpriseBand::Critical => 9500,
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

fn drift_band_token(band: DriftBand) -> &'static str {
    match band {
        DriftBand::Low => "LOW",
        DriftBand::Medium => "MED",
        DriftBand::High => "HIGH",
        DriftBand::Critical => "CRIT",
    }
}

fn sleep_trigger_token(trigger: SleepTrigger) -> &'static str {
    match trigger {
        SleepTrigger::None => "NONE",
        SleepTrigger::Instability => "INSTABILITY",
        SleepTrigger::Density => "DENSITY",
        SleepTrigger::LowIntegration => "LOW_INTEGRATION",
        SleepTrigger::Manual => "MANUAL",
        SleepTrigger::SurpriseCritical => "SURPRISE_CRITICAL",
    }
}

fn sleep_trigger_code(trigger: SleepTrigger) -> u8 {
    match trigger {
        SleepTrigger::None => 0,
        SleepTrigger::Instability => 1,
        SleepTrigger::Density => 2,
        SleepTrigger::LowIntegration => 3,
        SleepTrigger::Manual => 4,
        SleepTrigger::SurpriseCritical => 5,
    }
}

fn surprise_band_token(band: SurpriseBand) -> &'static str {
    match band {
        SurpriseBand::Low => "LOW",
        SurpriseBand::Medium => "MED",
        SurpriseBand::High => "HIGH",
        SurpriseBand::Critical => "CRIT",
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

fn digest_consistency_report(report: &PolicyConsistencyReport) -> Digest32 {
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

pub fn output_event_commit(
    tag: &[u8],
    frame_commit: Digest32,
    evidence: Option<Digest32>,
    risk: u16,
) -> Digest32 {
    digest_output_event(tag, frame_commit, evidence, risk)
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

fn digest_internal_utterance(src: Digest32, severity: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.internal_utterance.v1");
    hasher.update(src.as_bytes());
    hasher.update(&severity.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_replay_summary(micro: usize, meso: usize, macro_: usize) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.workspace.replay.summary.v1");
    hasher.update(&(micro as u64).to_be_bytes());
    hasher.update(&(meso as u64).to_be_bytes());
    hasher.update(&(macro_ as u64).to_be_bytes());
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

#[allow(clippy::too_many_arguments)]
fn commit_snapshot(
    cycle_id: u64,
    recursion_used: u16,
    spike_accepted_root: Digest32,
    spike_counts: &[(SpikeKind, u16)],
    spike_max_intensity: u16,
    ncde_commit: Digest32,
    ncde_state_digest: Digest32,
    ncde_energy: u16,
    replay_pressure_hint: u16,
    cde_commit: Digest32,
    cde_graph_commit: Digest32,
    cde_top_edges: &[(u16, u16, u16, u8)],
    cde_top_edge_commits: &[Digest32],
    cde_intervention_commit: Option<Digest32>,
    cde_observation_commit: Digest32,
    cde_last_query_result: Option<Digest32>,
    ssm_commit: Digest32,
    ssm_state_commit: Digest32,
    ssm_state_digest: Digest32,
    ssm_salience: u16,
    ssm_novelty: u16,
    ssm_attention_gain: u16,
    jepa_world_state: Digest32,
    jepa_prediction: Digest32,
    jepa_surprise: u16,
    learning_signal_commit: Digest32,
    learning_signal_learn_rate: u16,
    learning_signal_mode: u8,
    structural_delta_commit: Digest32,
    structural_delta_mass: u16,
    structural_delta_targets: [u16; 4],
    influence_v2_commit: Digest32,
    influence_pulses_root: Digest32,
    influence_node_values: &[(u16, i16)],
    coupling_influences_root: Digest32,
    coupling_top_influences: &[(u16, i16)],
    coupling_lag_commits: &[(u16, Digest32)],
    gain_budget_commit: Digest32,
    budget_low_plv_streak: u8,
    budget_high_novelty_streak: u8,
    budget_violation_streak: u8,
    budget_triggers: u8,
    tcf_plan_commit: Digest32,
    tcf_attention_gain_cap: u16,
    tcf_learning_gain_cap: u16,
    tcf_output_gain_cap: u16,
    tcf_sleep_active: bool,
    tcf_replay_active: bool,
    tcf_lock_window_buckets: u8,
    lock_window_source_cycle: u64,
    coherence_lag_commit: Digest32,
    update_mode: u8,
    onn_phase_commit: Digest32,
    onn_gamma_bucket: u8,
    onn_global_plv: u16,
    iit_output: Option<&IitOutput>,
    nsr_trace_root: Option<Digest32>,
    nsr_prev_commit: Option<Digest32>,
    nsr_verdict: Option<u8>,
    nsr_triggered_rules_root: Option<Digest32>,
    nsr_derived_facts_root: Option<Digest32>,
    nsr_fact_flags: u8,
    nsr_hit_counts: [u16; 3],
    nsr_hit_summaries: &[NsrHitSummary],
    rsa_commit: Digest32,
    rsa_proposal_commit: Option<Digest32>,
    rsa_decision_apply: bool,
    rsa_apply_allowed: bool,
    rsa_apply_commit: Digest32,
    rsa_reason_mask: u32,
    rsa_applied_params_root: Digest32,
    rsa_snapshot_chain_commit: Digest32,
    sle_commit: Digest32,
    sle_reflection_commit: Digest32,
    sle_reflection_class: u8,
    sle_reflection_intensity: u16,
    sle_thought_only_root: Digest32,
    sle_ssm_bias: i16,
    sle_cde_bias: i16,
    sle_request_replay: bool,
    internal_utterances: &[InternalUtterance],
    broadcast: &[WorkspaceSignal],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&recursion_used.to_be_bytes());
    hasher.update(spike_accepted_root.as_bytes());
    hasher.update(
        &u32::try_from(spike_counts.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (kind, count) in spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(&spike_max_intensity.to_be_bytes());
    hasher.update(ncde_commit.as_bytes());
    hasher.update(ncde_state_digest.as_bytes());
    hasher.update(&ncde_energy.to_be_bytes());
    hasher.update(&replay_pressure_hint.to_be_bytes());
    hasher.update(cde_commit.as_bytes());
    hasher.update(cde_graph_commit.as_bytes());
    hasher.update(
        &u64::try_from(cde_top_edges.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for (from, to, conf, lag) in cde_top_edges {
        hasher.update(&from.to_be_bytes());
        hasher.update(&to.to_be_bytes());
        hasher.update(&conf.to_be_bytes());
        hasher.update(&[*lag]);
    }
    hasher.update(
        &u64::try_from(cde_top_edge_commits.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for commit in cde_top_edge_commits {
        hasher.update(commit.as_bytes());
    }
    match cde_intervention_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(cde_observation_commit.as_bytes());
    match cde_last_query_result {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(ssm_commit.as_bytes());
    hasher.update(ssm_state_commit.as_bytes());
    hasher.update(ssm_state_digest.as_bytes());
    hasher.update(&ssm_salience.to_be_bytes());
    hasher.update(&ssm_novelty.to_be_bytes());
    hasher.update(&ssm_attention_gain.to_be_bytes());
    hasher.update(jepa_world_state.as_bytes());
    hasher.update(jepa_prediction.as_bytes());
    hasher.update(&jepa_surprise.to_be_bytes());
    hasher.update(learning_signal_commit.as_bytes());
    hasher.update(&learning_signal_learn_rate.to_be_bytes());
    hasher.update(&[learning_signal_mode]);
    hasher.update(structural_delta_commit.as_bytes());
    hasher.update(&structural_delta_mass.to_be_bytes());
    for target in structural_delta_targets {
        hasher.update(&target.to_be_bytes());
    }
    hasher.update(influence_v2_commit.as_bytes());
    hasher.update(influence_pulses_root.as_bytes());
    hasher.update(
        &u64::try_from(influence_node_values.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for (node, value) in influence_node_values {
        hasher.update(&node.to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    hasher.update(coupling_influences_root.as_bytes());
    hasher.update(
        &u64::try_from(coupling_top_influences.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for (signal, value) in coupling_top_influences {
        hasher.update(&signal.to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    hasher.update(
        &u64::try_from(coupling_lag_commits.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for (signal, commit) in coupling_lag_commits {
        hasher.update(&signal.to_be_bytes());
        hasher.update(commit.as_bytes());
    }
    hasher.update(gain_budget_commit.as_bytes());
    hasher.update(&[
        budget_low_plv_streak,
        budget_high_novelty_streak,
        budget_violation_streak,
        budget_triggers,
    ]);
    hasher.update(tcf_plan_commit.as_bytes());
    hasher.update(&tcf_attention_gain_cap.to_be_bytes());
    hasher.update(&tcf_learning_gain_cap.to_be_bytes());
    hasher.update(&tcf_output_gain_cap.to_be_bytes());
    hasher.update(&[tcf_sleep_active as u8]);
    hasher.update(&[tcf_replay_active as u8]);
    hasher.update(&[tcf_lock_window_buckets]);
    hasher.update(&lock_window_source_cycle.to_be_bytes());
    hasher.update(coherence_lag_commit.as_bytes());
    hasher.update(&[update_mode]);
    hasher.update(onn_phase_commit.as_bytes());
    hasher.update(&[onn_gamma_bucket]);
    hasher.update(&onn_global_plv.to_be_bytes());
    match iit_output {
        Some(output) => {
            hasher.update(&[1]);
            hasher.update(output.commit.as_bytes());
            hasher.update(&output.phi_proxy.to_be_bytes());
            hasher.update(&[iit_hint_flags(output)]);
            hasher.update(output.hints_commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match nsr_trace_root {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match nsr_prev_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&[nsr_verdict.unwrap_or(0)]);
    match nsr_triggered_rules_root {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match nsr_derived_facts_root {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&[nsr_fact_flags]);
    for count in nsr_hit_counts {
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(
        &u32::try_from(nsr_hit_summaries.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for summary in nsr_hit_summaries {
        hasher.update(&summary.rule_id.to_be_bytes());
        hasher.update(&[summary.severity]);
        hasher.update(&summary.reason.to_be_bytes());
        hasher.update(summary.commit.as_bytes());
    }
    hasher.update(rsa_commit.as_bytes());
    match rsa_proposal_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&[rsa_decision_apply as u8]);
    hasher.update(&[rsa_apply_allowed as u8]);
    hasher.update(rsa_apply_commit.as_bytes());
    hasher.update(&rsa_reason_mask.to_be_bytes());
    hasher.update(rsa_applied_params_root.as_bytes());
    hasher.update(rsa_snapshot_chain_commit.as_bytes());
    hasher.update(sle_commit.as_bytes());
    hasher.update(sle_reflection_commit.as_bytes());
    hasher.update(&[sle_reflection_class]);
    hasher.update(&sle_reflection_intensity.to_be_bytes());
    hasher.update(sle_thought_only_root.as_bytes());
    hasher.update(&sle_ssm_bias.to_be_bytes());
    hasher.update(&sle_cde_bias.to_be_bytes());
    hasher.update(&[sle_request_replay as u8]);
    hasher.update(
        &u32::try_from(internal_utterances.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for utterance in internal_utterances {
        hasher.update(utterance.commit.as_bytes());
        hasher.update(utterance.src.as_bytes());
        hasher.update(&utterance.severity.to_be_bytes());
    }
    for signal in broadcast {
        hasher.update(&[signal.kind as u8]);
        hasher.update(&[signal.slot]);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(
        kind: SignalKind,
        priority: u16,
        digest_byte: u8,
        summary: &str,
    ) -> WorkspaceSignal {
        WorkspaceSignal {
            kind,
            priority,
            digest: Digest32::new([digest_byte; 32]),
            summary: summary.to_string(),
            slot: 0,
        }
    }

    #[test]
    fn publish_over_cap_drops_lowest_priority_then_kind_then_digest_then_seq() {
        let mut workspace = Workspace::new(WorkspaceConfig {
            cap: 2,
            broadcast_cap: 2,
        });

        // Drop rule: remove the lowest priority, then lowest kind, then lowest digest, then seq.
        workspace.publish(make_signal(SignalKind::World, 1000, 1, "WORLD=STATE DIG=1"));
        workspace.publish(make_signal(SignalKind::Policy, 1000, 2, "POLICY=ALLOW"));
        workspace.publish(make_signal(SignalKind::Risk, 1000, 3, "RISK=42 PERMIT"));

        let snapshot = workspace.arbitrate(1);

        assert_eq!(snapshot.broadcast.len(), 2);
        assert_eq!(snapshot.broadcast[0].kind, SignalKind::Policy);
        assert_eq!(snapshot.broadcast[1].kind, SignalKind::Risk);

        let drops = workspace.drop_counters();
        assert_eq!(drops.total, 1);
        assert_eq!(drops.by_kind[SignalKind::World.index()], 1);
    }

    #[test]
    fn arbitrate_orders_by_priority_then_kind_then_digest_and_enforces_broadcast_cap() {
        let mut workspace = Workspace::new(WorkspaceConfig {
            cap: 10,
            broadcast_cap: 2,
        });

        workspace.publish(make_signal(SignalKind::Policy, 5000, 2, "POLICY=ALLOW"));
        workspace.publish(make_signal(SignalKind::Policy, 5000, 1, "POLICY=ALLOW"));
        workspace.publish(make_signal(SignalKind::Risk, 7000, 9, "RISK=9000 DENY"));
        workspace.publish(make_signal(SignalKind::World, 1000, 4, "WORLD=STATE DIG=4"));

        let snapshot = workspace.arbitrate(7);

        assert_eq!(snapshot.broadcast.len(), 2);
        assert_eq!(snapshot.broadcast[0].kind, SignalKind::Risk);
        assert_eq!(snapshot.broadcast[1].digest, Digest32::new([1u8; 32]));

        let drops = workspace.drop_counters();
        assert_eq!(drops.total, 2);
        assert_eq!(drops.by_kind[SignalKind::Policy.index()], 1);
        assert_eq!(drops.by_kind[SignalKind::World.index()], 1);
    }

    #[test]
    fn snapshot_commit_changes_when_signal_digest_changes() {
        let mut workspace_a = Workspace::new(WorkspaceConfig {
            cap: 2,
            broadcast_cap: 2,
        });
        workspace_a.publish(make_signal(SignalKind::Policy, 5000, 1, "POLICY=ALLOW"));
        workspace_a.publish(make_signal(SignalKind::Risk, 7000, 2, "RISK=100 DENY"));
        let snapshot_a = workspace_a.arbitrate(12);

        let mut workspace_b = Workspace::new(WorkspaceConfig {
            cap: 2,
            broadcast_cap: 2,
        });
        workspace_b.publish(make_signal(SignalKind::Policy, 5000, 1, "POLICY=ALLOW"));
        workspace_b.publish(make_signal(SignalKind::Risk, 7000, 3, "RISK=100 DENY"));
        let snapshot_b = workspace_b.arbitrate(12);

        assert_ne!(snapshot_a.commit, snapshot_b.commit);
    }
}
