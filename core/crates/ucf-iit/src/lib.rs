#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_influence::InfluenceNodeId;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const DOMAIN_INPUT: &[u8] = b"ucf.iit.inputs.v1";
const DOMAIN_HINTS: &[u8] = b"ucf.iit.hints.v1";
const DOMAIN_REPORT: &[u8] = b"ucf.iit.report.v1";
const DOMAIN_OUTPUT: &[u8] = b"ucf.iit.output.v1";
const DOMAIN_PARAMS: &[u8] = b"ucf.iit.params.v1";
const DOMAIN_CORE: &[u8] = b"ucf.iit.core.v1";

const MAX_SCORE: u16 = 10_000;
const SPIKE_REWARD_CAP: u32 = 24;
const SPIKE_THREAT_CAP: u32 = 12;
const SPIKE_SUPPRESS_PENALTY: u16 = 1000;
const DEPENDENCY_UNIT: u16 = 2500;
const NCDE_ENERGY_MIN: u16 = 1200;
const NCDE_ENERGY_MAX: u16 = 8500;

const DRIFT_HIGH: u16 = 7000;
const RISK_HIGH: u16 = 7000;
const SURPRISE_HIGH: u16 = 7000;
const PLV_LOW: u16 = 3500;
const REPLAY_PRESSURE_MIN: i16 = 3000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IitInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub global_plv: u16,
    pub pair_locks_commit: Digest32,
    pub influence_pulses_root: Digest32,
    pub influence_nodes: Vec<(InfluenceNodeId, i16)>,
    pub spike_seen_root: Digest32,
    pub spike_accepted_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub cde_summary_commit: Option<Digest32>,
    pub nsr_trace_root: Digest32,
    pub geist_consistency: bool,
    pub ncde_state_digest: Digest32,
    pub ncde_energy: u16,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub commit: Digest32,
}

impl IitInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        global_plv: u16,
        pair_locks_commit: Digest32,
        influence_pulses_root: Digest32,
        influence_nodes: Vec<(InfluenceNodeId, i16)>,
        spike_seen_root: Digest32,
        spike_accepted_root: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        cde_summary_commit: Option<Digest32>,
        nsr_trace_root: Digest32,
        geist_consistency: bool,
        ncde_state_digest: Digest32,
        ncde_energy: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
    ) -> Self {
        let mut inputs = Self {
            cycle_id,
            phase_commit,
            global_plv,
            pair_locks_commit,
            influence_pulses_root,
            influence_nodes,
            spike_seen_root,
            spike_accepted_root,
            spike_counts,
            cde_summary_commit,
            nsr_trace_root,
            geist_consistency,
            ncde_state_digest,
            ncde_energy,
            risk,
            drift,
            surprise,
            commit: Digest32::new([0u8; 32]),
        };
        inputs.commit = commit_inputs(&inputs);
        inputs
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IitHints {
    pub tighten_sync: bool,
    pub damp_output: bool,
    pub damp_learning: bool,
    pub request_replay: bool,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IitOutputs {
    pub cycle_id: u64,
    pub phi_proxy: u16,
    pub integration_report_commit: Digest32,
    pub hints: IitHints,
    pub commit: Digest32,
}

pub type IitOutput = IitOutputs;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IitParams {
    pub w_plv: u16,
    pub w_influence: u16,
    pub w_spike: u16,
    pub w_dependency: u16,
    pub penalty_risk: u16,
    pub penalty_drift: u16,
    pub penalty_surprise: u16,
    pub phi_low: u16,
    pub phi_high: u16,
    pub commit: Digest32,
}

impl Default for IitParams {
    fn default() -> Self {
        let mut params = Self {
            w_plv: 3,
            w_influence: 3,
            w_spike: 3,
            w_dependency: 1,
            penalty_risk: 3500,
            penalty_drift: 2800,
            penalty_surprise: 2200,
            phi_low: 3200,
            phi_high: 7200,
            commit: Digest32::new([0u8; 32]),
        };
        params.commit = commit_params(&params);
        params
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IitCore {
    pub params: IitParams,
    pub commit: Digest32,
}

impl IitCore {
    pub fn new(params: IitParams) -> Self {
        let commit = commit_core(&params);
        Self { params, commit }
    }

    pub fn tick(&self, inp: &IitInputs) -> IitOutputs {
        let phase_score = inp.global_plv.min(MAX_SCORE);
        let influence_score = influence_score(&inp.influence_nodes);
        let spike_score = spike_score(inp);
        let dependency_score = dependency_score(inp);

        let weighted_sum = weighted_sum(&[
            (phase_score, self.params.w_plv),
            (influence_score, self.params.w_influence),
            (spike_score, self.params.w_spike),
            (dependency_score, self.params.w_dependency),
        ]);
        let penalties = penalty_score(
            inp.risk,
            inp.drift,
            inp.surprise,
            self.params.penalty_risk,
            self.params.penalty_drift,
            self.params.penalty_surprise,
        );
        let phi_raw = weighted_sum.min(MAX_SCORE);
        let phi_proxy = phi_raw.saturating_sub(penalties).min(MAX_SCORE);

        let hints = build_hints(inp, phi_proxy, self.params.phi_low, self.params.phi_high);
        let integration_report_commit = commit_report(
            inp,
            phi_proxy,
            phase_score,
            influence_score,
            spike_score,
            dependency_score,
            penalties,
            &hints,
        );
        let commit = commit_outputs(inp.cycle_id, phi_proxy, integration_report_commit, &hints);

        IitOutputs {
            cycle_id: inp.cycle_id,
            phi_proxy,
            integration_report_commit,
            hints,
            commit,
        }
    }
}

impl Default for IitCore {
    fn default() -> Self {
        Self::new(IitParams::default())
    }
}

fn influence_score(nodes: &[(InfluenceNodeId, i16)]) -> u16 {
    let integration = node_value(nodes, InfluenceNodeId::Integration);
    let coherence = node_value(nodes, InfluenceNodeId::Coherence);
    let attention = node_value(nodes, InfluenceNodeId::AttentionGain);
    weighted_sum(&[(integration, 4), (coherence, 3), (attention, 3)])
}

fn node_value(nodes: &[(InfluenceNodeId, i16)], target: InfluenceNodeId) -> u16 {
    let value = nodes
        .iter()
        .find_map(|(id, value)| (*id == target).then_some(*value))
        .unwrap_or(0);
    let clamped = value.clamp(-10_000, 10_000) as i32;
    let normalized = (clamped + 10_000) as u32 * 10_000 / 20_000;
    clamp_score(normalized)
}

fn spike_score(inp: &IitInputs) -> u16 {
    let mut reward_total: u32 = 0;
    let mut threat_total: u32 = 0;

    for (kind, count) in &inp.spike_counts {
        if *count == 0 {
            continue;
        }
        match kind {
            SpikeKind::CausalLink
            | SpikeKind::ConsistencyAlert
            | SpikeKind::Feature
            | SpikeKind::Novelty => {
                reward_total = reward_total.saturating_add(u32::from(*count));
            }
            SpikeKind::Threat => {
                threat_total = threat_total.saturating_add(u32::from(*count));
            }
            _ => {}
        }
    }

    let reward_score = scale_to_max(reward_total, SPIKE_REWARD_CAP);
    let threat_penalty = scale_to_max(threat_total, SPIKE_THREAT_CAP) / 2;
    let suppression_penalty = if inp.spike_seen_root != inp.spike_accepted_root {
        SPIKE_SUPPRESS_PENALTY
    } else {
        0
    };

    reward_score
        .saturating_sub(threat_penalty)
        .saturating_sub(suppression_penalty)
        .min(MAX_SCORE)
}

fn dependency_score(inp: &IitInputs) -> u16 {
    let mut score = 0u16;
    if inp.cde_summary_commit.is_some() {
        score = score.saturating_add(DEPENDENCY_UNIT);
    }
    if !is_zero_digest(&inp.nsr_trace_root) {
        score = score.saturating_add(DEPENDENCY_UNIT);
    }
    if inp.geist_consistency {
        score = score.saturating_add(DEPENDENCY_UNIT);
    }
    if inp.ncde_energy >= NCDE_ENERGY_MIN && inp.ncde_energy <= NCDE_ENERGY_MAX {
        score = score.saturating_add(DEPENDENCY_UNIT);
    }
    score.min(MAX_SCORE)
}

fn penalty_score(
    risk: u16,
    drift: u16,
    surprise: u16,
    penalty_risk: u16,
    penalty_drift: u16,
    penalty_surprise: u16,
) -> u16 {
    let risk_term = u32::from(risk.min(MAX_SCORE)) * u32::from(penalty_risk);
    let drift_term = u32::from(drift.min(MAX_SCORE)) * u32::from(penalty_drift);
    let surprise_term = u32::from(surprise.min(MAX_SCORE)) * u32::from(penalty_surprise);
    let total = risk_term
        .saturating_add(drift_term)
        .saturating_add(surprise_term);
    clamp_score(total / u32::from(MAX_SCORE))
}

fn build_hints(inp: &IitInputs, phi: u16, phi_low: u16, _phi_high: u16) -> IitHints {
    let tighten_sync = phi < phi_low || inp.global_plv < PLV_LOW || inp.drift >= DRIFT_HIGH;
    let damp_output = phi < phi_low || inp.risk >= RISK_HIGH || !inp.geist_consistency;
    let damp_learning = inp.risk >= RISK_HIGH || (inp.surprise >= SURPRISE_HIGH && phi < phi_low);
    let replay_pressure = replay_pressure_present(&inp.influence_nodes);
    let request_replay = phi < phi_low
        && (inp.drift >= DRIFT_HIGH || inp.surprise >= SURPRISE_HIGH)
        && replay_pressure;
    let flags = hints_flags(tighten_sync, damp_output, damp_learning, request_replay);
    let commit = commit_hints(inp.cycle_id, phi, flags);

    IitHints {
        tighten_sync,
        damp_output,
        damp_learning,
        request_replay,
        commit,
    }
}

fn replay_pressure_present(nodes: &[(InfluenceNodeId, i16)]) -> bool {
    nodes
        .iter()
        .find_map(|(id, value)| (*id == InfluenceNodeId::ReplayPressure).then_some(*value))
        .unwrap_or(0)
        >= REPLAY_PRESSURE_MIN
}

fn hints_flags(
    tighten_sync: bool,
    damp_output: bool,
    damp_learning: bool,
    request_replay: bool,
) -> u8 {
    (tighten_sync as u8)
        | ((damp_output as u8) << 1)
        | ((damp_learning as u8) << 2)
        | ((request_replay as u8) << 3)
}

fn weighted_sum(values: &[(u16, u16)]) -> u16 {
    let mut sum: u32 = 0;
    let mut weight: u32 = 0;
    for (value, w) in values {
        if *w == 0 {
            continue;
        }
        sum = sum.saturating_add(u32::from(*value) * u32::from(*w));
        weight = weight.saturating_add(u32::from(*w));
    }
    if weight == 0 {
        return 0;
    }
    clamp_score(sum / weight)
}

fn scale_to_max(value: u32, cap: u32) -> u16 {
    if cap == 0 {
        return 0;
    }
    let scaled = (value.min(cap) * u32::from(MAX_SCORE)) / cap;
    clamp_score(scaled)
}

fn clamp_score(value: u32) -> u16 {
    u16::try_from(value.min(u32::from(MAX_SCORE))).unwrap_or(MAX_SCORE)
}

fn is_zero_digest(digest: &Digest32) -> bool {
    digest.as_bytes().iter().all(|value| *value == 0)
}

fn commit_inputs(inputs: &IitInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_INPUT);
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.phase_commit.as_bytes());
    hasher.update(&inputs.global_plv.to_be_bytes());
    hasher.update(inputs.pair_locks_commit.as_bytes());
    hasher.update(inputs.influence_pulses_root.as_bytes());
    let mut influence_nodes = inputs.influence_nodes.clone();
    influence_nodes.sort_by(|(left_id, left_value), (right_id, right_value)| {
        left_id
            .to_u16()
            .cmp(&right_id.to_u16())
            .then_with(|| left_value.cmp(right_value))
    });
    hasher.update(
        &u32::try_from(influence_nodes.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (node, value) in influence_nodes {
        hasher.update(&node.to_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    hasher.update(inputs.spike_seen_root.as_bytes());
    hasher.update(inputs.spike_accepted_root.as_bytes());
    let mut spike_counts = inputs.spike_counts.clone();
    spike_counts.sort_by(|(left_kind, left_count), (right_kind, right_count)| {
        left_kind
            .as_u16()
            .cmp(&right_kind.as_u16())
            .then_with(|| left_count.cmp(right_count))
    });
    hasher.update(
        &u32::try_from(spike_counts.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (kind, count) in spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    match inputs.cde_summary_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(inputs.nsr_trace_root.as_bytes());
    hasher.update(&[inputs.geist_consistency as u8]);
    hasher.update(inputs.ncde_state_digest.as_bytes());
    hasher.update(&inputs.ncde_energy.to_be_bytes());
    hasher.update(&inputs.risk.to_be_bytes());
    hasher.update(&inputs.drift.to_be_bytes());
    hasher.update(&inputs.surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_hints(cycle_id: u64, phi: u16, flags: u8) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_HINTS);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&phi.to_be_bytes());
    hasher.update(&[flags]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_report(
    inputs: &IitInputs,
    phi: u16,
    phase_score: u16,
    influence_score: u16,
    spike_score: u16,
    dependency_score: u16,
    penalties: u16,
    hints: &IitHints,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_REPORT);
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.phase_commit.as_bytes());
    hasher.update(&phi.to_be_bytes());
    hasher.update(&phase_score.to_be_bytes());
    hasher.update(&influence_score.to_be_bytes());
    hasher.update(&spike_score.to_be_bytes());
    hasher.update(&dependency_score.to_be_bytes());
    hasher.update(&penalties.to_be_bytes());
    hasher.update(hints.commit.as_bytes());
    hasher.update(inputs.pair_locks_commit.as_bytes());
    hasher.update(inputs.influence_pulses_root.as_bytes());
    hasher.update(inputs.spike_seen_root.as_bytes());
    hasher.update(inputs.spike_accepted_root.as_bytes());
    if let Some(commit) = inputs.cde_summary_commit {
        hasher.update(commit.as_bytes());
    }
    hasher.update(inputs.nsr_trace_root.as_bytes());
    hasher.update(inputs.ncde_state_digest.as_bytes());
    hasher.update(&[inputs.geist_consistency as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(cycle_id: u64, phi: u16, report_commit: Digest32, hints: &IitHints) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_OUTPUT);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&phi.to_be_bytes());
    hasher.update(report_commit.as_bytes());
    hasher.update(hints.commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_params(params: &IitParams) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PARAMS);
    hasher.update(&params.w_plv.to_be_bytes());
    hasher.update(&params.w_influence.to_be_bytes());
    hasher.update(&params.w_spike.to_be_bytes());
    hasher.update(&params.w_dependency.to_be_bytes());
    hasher.update(&params.penalty_risk.to_be_bytes());
    hasher.update(&params.penalty_drift.to_be_bytes());
    hasher.update(&params.penalty_surprise.to_be_bytes());
    hasher.update(&params.phi_low.to_be_bytes());
    hasher.update(&params.phi_high.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params: &IitParams) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_CORE);
    hasher.update(params.commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inputs(global_plv: u16, risk: u16) -> IitInputs {
        IitInputs::new(
            42,
            Digest32::new([1u8; 32]),
            global_plv,
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            vec![
                (InfluenceNodeId::Integration, 4000),
                (InfluenceNodeId::Coherence, 2500),
                (InfluenceNodeId::AttentionGain, 1500),
                (InfluenceNodeId::ReplayPressure, 4000),
            ],
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            vec![
                (SpikeKind::CausalLink, 4),
                (SpikeKind::ConsistencyAlert, 2),
                (SpikeKind::Feature, 1),
                (SpikeKind::Threat, 0),
            ],
            Some(Digest32::new([6u8; 32])),
            Digest32::new([7u8; 32]),
            true,
            Digest32::new([8u8; 32]),
            4200,
            risk,
            1200,
            1500,
        )
    }

    #[test]
    fn deterministic_outputs_for_same_inputs() {
        let core = IitCore::default();
        let inputs = sample_inputs(5000, 2000);
        let out_a = core.tick(&inputs);
        let out_b = core.tick(&inputs);

        assert_eq!(out_a.phi_proxy, out_b.phi_proxy);
        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(
            out_a.integration_report_commit,
            out_b.integration_report_commit
        );
        assert_eq!(out_a.hints.commit, out_b.hints.commit);
    }

    #[test]
    fn higher_plv_increases_phi_proxy() {
        let core = IitCore::default();
        let low = sample_inputs(2000, 2000);
        let high = sample_inputs(8000, 2000);

        assert!(core.tick(&high).phi_proxy > core.tick(&low).phi_proxy);
    }

    #[test]
    fn higher_risk_decreases_phi_proxy() {
        let core = IitCore::default();
        let low_risk = sample_inputs(6000, 1000);
        let high_risk = sample_inputs(6000, 9000);

        assert!(core.tick(&low_risk).phi_proxy > core.tick(&high_risk).phi_proxy);
    }

    #[test]
    fn low_phi_sets_tighten_and_damp_output() {
        let core = IitCore::default();
        let mut inputs = sample_inputs(1500, 2000);
        inputs.drift = 8000;
        let output = core.tick(&inputs);

        assert!(output.hints.tighten_sync);
        assert!(output.hints.damp_output);
    }
}
