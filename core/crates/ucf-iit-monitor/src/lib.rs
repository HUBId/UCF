#![forbid(unsafe_code)]

//! Deterministic proxy coupling metric between module digests.
//!
//! This is **not** a real IIT/Phi implementation. The score is a stable hash mapping
//! into `0..=10000` and only serves as a repeatable integration/coupling proxy.

use std::collections::{HashSet, VecDeque};

use blake3::Hasher;
use ucf_types::Digest32;
use ucf_workspace::{SignalKind, WorkspaceSnapshot};

const DOMAIN_SCORE: &[u8] = b"ucf.iit.proxy.v1";
const DOMAIN_REPORT: &[u8] = b"ucf.iit.report.v1";
const DOMAIN_ACTION: &[u8] = b"ucf.iit.action.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CouplingSample {
    pub a: Digest32,
    pub b: Digest32,
    pub score: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IitBand {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IitReport {
    pub phi: u16,
    pub band: IitBand,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IitActionKind {
    Fusion,
    Isolate,
    ReplayBias,
    Throttle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IitAction {
    pub kind: IitActionKind,
    pub intensity: u16,
    pub commit: Digest32,
}

#[derive(Debug)]
pub struct IitMonitor {
    pub window: usize,
    samples: VecDeque<CouplingSample>,
    phi_history: VecDeque<u16>,
}

impl IitMonitor {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            samples: VecDeque::new(),
            phi_history: VecDeque::new(),
        }
    }

    pub fn sample(&mut self, module_a: Digest32, module_b: Digest32) -> CouplingSample {
        let score = coupling_score(&module_a, &module_b);
        let sample = CouplingSample {
            a: module_a,
            b: module_b,
            score,
        };
        self.samples.push_back(sample);
        if self.window > 0 {
            while self.samples.len() > self.window {
                self.samples.pop_front();
            }
        }
        sample
    }

    pub fn aggregate(&self) -> u16 {
        if self.samples.is_empty() {
            return 0;
        }
        let sum: u32 = self
            .samples
            .iter()
            .map(|sample| u32::from(sample.score))
            .sum();
        let avg = sum / self.samples.len().max(1) as u32;
        u16::try_from(avg.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
    }

    pub fn evaluate(
        &mut self,
        snapshot: &WorkspaceSnapshot,
        risk: u16,
        coherence_plv: Option<u16>,
    ) -> (IitReport, Vec<IitAction>) {
        let coherence_plv = coherence_plv.or_else(|| coherence_from_snapshot(snapshot));
        let phi = integration_score(snapshot, &self.phi_history, coherence_plv);
        if self.window > 0 {
            self.phi_history.push_back(phi);
            while self.phi_history.len() > self.window {
                self.phi_history.pop_front();
            }
        }
        let band = band_for_phi(phi);
        let report = IitReport {
            phi,
            band,
            commit: report_commit(phi, band),
        };
        let actions = actions_for_phi(phi, risk);
        (report, actions)
    }
}

fn coupling_score(a: &Digest32, b: &Digest32) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_SCORE);
    hasher.update(a.as_bytes());
    hasher.update(b.as_bytes());
    let bytes = hasher.finalize();
    let raw = u32::from_be_bytes([
        bytes.as_bytes()[0],
        bytes.as_bytes()[1],
        bytes.as_bytes()[2],
        bytes.as_bytes()[3],
    ]);
    let scaled = (u64::from(raw) * 10_000) / u64::from(u32::MAX);
    u16::try_from(scaled.min(10_000)).unwrap_or(10_000)
}

fn integration_score(
    snapshot: &WorkspaceSnapshot,
    history: &VecDeque<u16>,
    coherence_plv: Option<u16>,
) -> u16 {
    let mut kinds: HashSet<SignalKind> = HashSet::new();
    for signal in &snapshot.broadcast {
        kinds.insert(signal.kind);
    }
    let kinds_count = u16::try_from(kinds.len()).unwrap_or(0);
    let cross_module = kinds_count.saturating_sub(1).saturating_mul(1200);
    let fusion = kinds_count.saturating_mul(800);
    let density = u16::try_from(snapshot.broadcast.len().min(16)).unwrap_or(0) * 300;

    let coherence_bonus = coherence_bonus(coherence_plv);
    let base = cross_module
        .saturating_add(fusion)
        .saturating_add(density)
        .saturating_add(coherence_bonus);
    let stability_bonus = stability_bonus(base, history);
    (base.saturating_add(stability_bonus)).min(10_000)
}

fn coherence_bonus(coherence_plv: Option<u16>) -> u16 {
    let Some(coherence) = coherence_plv else {
        return 0;
    };
    (coherence / 4).min(2500)
}

fn coherence_from_snapshot(snapshot: &WorkspaceSnapshot) -> Option<u16> {
    snapshot.broadcast.iter().find_map(|signal| {
        signal
            .summary
            .split_whitespace()
            .find_map(|token| token.strip_prefix("COH="))
            .and_then(|value| value.parse::<u16>().ok())
    })
}

fn stability_bonus(base: u16, history: &VecDeque<u16>) -> u16 {
    let Some(avg) = average(history) else {
        return 0;
    };
    let delta = avg.abs_diff(base);
    if delta <= 500 {
        1500
    } else if delta <= 1500 {
        500
    } else {
        0
    }
}

fn average(history: &VecDeque<u16>) -> Option<u16> {
    if history.is_empty() {
        return None;
    }
    let sum: u32 = history.iter().map(|value| u32::from(*value)).sum();
    let avg = sum / history.len().max(1) as u32;
    Some(u16::try_from(avg.min(u32::from(u16::MAX))).unwrap_or(u16::MAX))
}

pub fn band_for_phi(phi: u16) -> IitBand {
    match phi {
        0..=3299 => IitBand::Low,
        3300..=6599 => IitBand::Medium,
        _ => IitBand::High,
    }
}

pub fn report_for_phi(phi: u16) -> IitReport {
    let band = band_for_phi(phi);
    IitReport {
        phi,
        band,
        commit: report_commit(phi, band),
    }
}

pub fn report_commit(phi: u16, band: IitBand) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_REPORT);
    hasher.update(&phi.to_be_bytes());
    hasher.update(&[band as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

pub fn actions_for_phi(phi: u16, risk: u16) -> Vec<IitAction> {
    let mut actions = Vec::new();

    if phi < 3000 {
        let intensity = 3000u16.saturating_sub(phi).min(3000);
        actions.push(build_action(IitActionKind::Fusion, intensity));
        actions.push(build_action(
            IitActionKind::ReplayBias,
            intensity.saturating_add(500),
        ));
    }

    if phi > 8000 && risk >= 7000 {
        let intensity = phi
            .saturating_sub(8000)
            .saturating_add(risk.saturating_sub(7000));
        actions.push(build_action(IitActionKind::Isolate, intensity.min(4000)));
        actions.push(build_action(IitActionKind::Throttle, intensity.min(3000)));
    }

    actions
}

fn build_action(kind: IitActionKind, intensity: u16) -> IitAction {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_ACTION);
    hasher.update(&[kind as u8]);
    hasher.update(&intensity.to_be_bytes());
    let commit = Digest32::new(*hasher.finalize().as_bytes());
    IitAction {
        kind,
        intensity,
        commit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::Digest32;
    use ucf_workspace::WorkspaceSignal;

    #[test]
    fn sample_is_deterministic_and_bounded() {
        let mut monitor = IitMonitor::new(4);
        let a = Digest32::new([1u8; 32]);
        let b = Digest32::new([2u8; 32]);

        let first = monitor.sample(a, b);
        let second = monitor.sample(a, b);

        assert_eq!(first.score, second.score);
        assert!(first.score <= 10_000);
    }

    #[test]
    fn evaluation_is_deterministic_for_same_snapshot_history() {
        let mut monitor_a = IitMonitor::new(3);
        let mut monitor_b = IitMonitor::new(3);
        let snapshot = WorkspaceSnapshot {
            cycle_id: 1,
            broadcast: vec![
                WorkspaceSignal {
                    kind: SignalKind::World,
                    priority: 4000,
                    digest: Digest32::new([1u8; 32]),
                    summary: "WORLD=STATE".to_string(),
                    slot: 0,
                },
                WorkspaceSignal {
                    kind: SignalKind::Risk,
                    priority: 5000,
                    digest: Digest32::new([2u8; 32]),
                    summary: "RISK=LOW".to_string(),
                    slot: 0,
                },
            ],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([9u8; 32]),
        };

        let (report_a, actions_a) = monitor_a.evaluate(&snapshot, 1000, None);
        let (report_b, actions_b) = monitor_b.evaluate(&snapshot, 1000, None);

        assert_eq!(report_a, report_b);
        assert_eq!(actions_a, actions_b);
    }

    #[test]
    fn low_phi_triggers_fusion_actions() {
        let mut monitor = IitMonitor::new(2);
        let snapshot = WorkspaceSnapshot {
            cycle_id: 1,
            broadcast: vec![],
            recursion_used: 0,
            spike_seen_root: Digest32::new([0u8; 32]),
            spike_accepted_root: Digest32::new([0u8; 32]),
            spike_counts: Vec::new(),
            spike_causal_link_count: 0,
            spike_consistency_alert_count: 0,
            spike_thought_only_count: 0,
            spike_output_intent_count: 0,
            spike_cap_hit: false,
            ncde_commit: Digest32::new([0u8; 32]),
            ncde_state_commit: Digest32::new([0u8; 32]),
            ncde_energy: 0,
            cde_commit: Digest32::new([0u8; 32]),
            cde_graph_commit: Digest32::new([0u8; 32]),
            cde_top_edges: Vec::new(),
            ssm_commit: Digest32::new([0u8; 32]),
            ssm_state_commit: Digest32::new([0u8; 32]),
            influence_v2_commit: Digest32::new([0u8; 32]),
            influence_pulses_root: Digest32::new([0u8; 32]),
            influence_node_values: Vec::new(),
            onn_states_commit: Digest32::new([0u8; 32]),
            onn_global_plv: 0,
            onn_pair_locks_commit: Digest32::new([0u8; 32]),
            onn_phase_frame_commit: Digest32::new([0u8; 32]),
            iit_output: None,
            nsr_trace_root: None,
            nsr_prev_commit: None,
            nsr_verdict: None,
            rsa_commit: Digest32::new([0u8; 32]),
            rsa_chosen: None,
            rsa_applied: false,
            rsa_new_params_commit: None,
            sle_commit: Digest32::new([0u8; 32]),
            sle_self_symbol_commit: Digest32::new([0u8; 32]),
            sle_rate_limited: false,
            internal_utterances: Vec::new(),
            commit: Digest32::new([3u8; 32]),
        };
        let (_report, actions) = monitor.evaluate(&snapshot, 0, None);

        assert!(actions
            .iter()
            .any(|action| action.kind == IitActionKind::Fusion));
        assert!(actions
            .iter()
            .any(|action| action.kind == IitActionKind::ReplayBias));
    }
}
