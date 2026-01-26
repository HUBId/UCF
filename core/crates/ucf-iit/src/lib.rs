#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const DOMAIN_INPUT: &[u8] = b"ucf.iit.inputs.v2";
const DOMAIN_OUTPUT: &[u8] = b"ucf.iit.output.v2";
const DOMAIN_STATE: &[u8] = b"ucf.iit.state.v2";

const MAX_SCORE: u16 = 10_000;
const SPIKE_TOTAL_CAP: u32 = 200;
const SPIKE_KIND_CAP: u32 = 8;
const SPIKE_CROSS_CAP: u32 = 5;

const COUPLING_LOW_THRESHOLD: u16 = 3_500;
const COUPLING_LOW_STREAK: u16 = 3;
const COUPLING_LOW_PENALTY: u16 = 900;

const SURPRISE_EXTREME: u16 = 9_000;
const COHERENCE_LOW: u16 = 3_000;
const SURPRISE_COHERENCE_PENALTY: u16 = 1_200;

const REASONING_BASE: u16 = 5_000;
const REASONING_NO_NSR: u16 = 4_200;

const WARN_SURPRISE_COHERENCE: u16 = 0b0001;
const WARN_LOW_COUPLING: u16 = 0b0010;
const WARN_NSR_DENY_SURPRISE: u16 = 0b0100;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IitInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub coherence_plv: u16,
    pub spike_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub ssm_commit: Digest32,
    pub wm_salience: u16,
    pub wm_novelty: u16,
    pub ncde_commit: Digest32,
    pub ncde_energy: u16,
    pub nsr_commit: Option<Digest32>,
    pub nsr_verdict: Option<u8>,
    pub cde_commit: Option<Digest32>,
    pub drift: u16,
    pub surprise: u16,
    pub risk: u16,
    pub commit: Digest32,
}

impl IitInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        coherence_plv: u16,
        spike_root: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        ssm_commit: Digest32,
        wm_salience: u16,
        wm_novelty: u16,
        ncde_commit: Digest32,
        ncde_energy: u16,
        nsr_commit: Option<Digest32>,
        nsr_verdict: Option<u8>,
        cde_commit: Option<Digest32>,
        drift: u16,
        surprise: u16,
        risk: u16,
    ) -> Self {
        let mut inputs = Self {
            cycle_id,
            phase_commit,
            coherence_plv,
            spike_root,
            spike_counts,
            ssm_commit,
            wm_salience,
            wm_novelty,
            ncde_commit,
            ncde_energy,
            nsr_commit,
            nsr_verdict,
            cde_commit,
            drift,
            surprise,
            risk,
            commit: Digest32::new([0u8; 32]),
        };
        inputs.commit = commit_inputs(&inputs);
        inputs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IitOutput {
    pub cycle_id: u64,
    pub phi_proxy: u16,
    pub coupling_proxy: u16,
    pub coherence: u16,
    pub warnings: u16,
    pub commit: Digest32,
}

#[derive(Debug)]
pub struct IitMonitor {
    pub state_commit: Digest32,
    low_coupling_streak: u16,
}

impl IitMonitor {
    pub fn new() -> Self {
        Self {
            state_commit: Digest32::new([0u8; 32]),
            low_coupling_streak: 0,
        }
    }

    pub fn tick(&mut self, inp: &IitInputs) -> IitOutput {
        let (coupling_spike, cross_checks) = coupling_from_spikes(&inp.spike_counts);
        let coupling_state = coupling_state(inp.wm_salience, inp.ncde_energy);
        let (coupling_reasoning, mut warnings) = coupling_reasoning(inp, cross_checks);

        let coupling_proxy = weighted_avg(&[
            (coupling_spike, 4),
            (coupling_state, 3),
            (coupling_reasoning, 3),
        ]);

        self.low_coupling_streak = if coupling_proxy < COUPLING_LOW_THRESHOLD {
            self.low_coupling_streak.saturating_add(1)
        } else {
            0
        };

        let drift_scaled = inp.drift.min(MAX_SCORE);
        let risk_scaled = inp.risk.min(MAX_SCORE);
        let drift_inverse = MAX_SCORE.saturating_sub(drift_scaled);
        let risk_inverse = MAX_SCORE.saturating_sub(risk_scaled);

        let mut phi_proxy = weighted_avg(&[
            (inp.coherence_plv.min(MAX_SCORE), 3),
            (coupling_proxy, 4),
            (drift_inverse, 2),
            (risk_inverse, 1),
        ]);

        if inp.surprise >= SURPRISE_EXTREME && inp.coherence_plv <= COHERENCE_LOW {
            phi_proxy = phi_proxy.saturating_sub(SURPRISE_COHERENCE_PENALTY);
            warnings |= WARN_SURPRISE_COHERENCE;
        }

        if self.low_coupling_streak >= COUPLING_LOW_STREAK {
            phi_proxy = phi_proxy.saturating_sub(COUPLING_LOW_PENALTY);
            warnings |= WARN_LOW_COUPLING;
        }

        phi_proxy = phi_proxy.min(MAX_SCORE);

        let output = IitOutput {
            cycle_id: inp.cycle_id,
            phi_proxy,
            coupling_proxy,
            coherence: inp.coherence_plv,
            warnings,
            commit: commit_output(inp, phi_proxy, coupling_proxy, warnings),
        };

        self.state_commit = commit_state(self.state_commit, inp.commit, output.commit);
        output
    }
}

impl Default for IitMonitor {
    fn default() -> Self {
        Self::new()
    }
}

fn coupling_from_spikes(spike_counts: &[(SpikeKind, u16)]) -> (u16, u16) {
    if spike_counts.is_empty() {
        return (0, 0);
    }
    let mut total: u32 = 0;
    let mut kinds: u32 = 0;
    let mut cross_total: u32 = 0;

    for (kind, count) in spike_counts {
        if *count == 0 {
            continue;
        }
        kinds = kinds.saturating_add(1);
        total = total.saturating_add(u32::from(*count));
        match kind {
            SpikeKind::CausalLink | SpikeKind::ConsistencyAlert => {
                cross_total = cross_total.saturating_add(u32::from(*count));
            }
            _ => {}
        }
    }

    let total_scaled = scale_to_max(total, SPIKE_TOTAL_CAP);
    let kinds_scaled = scale_to_max(kinds, SPIKE_KIND_CAP);
    let cross_pairs = cross_total / 2;
    let cross_scaled = scale_to_max(cross_pairs, SPIKE_CROSS_CAP);

    (
        weighted_avg(&[(total_scaled, 4), (kinds_scaled, 3), (cross_scaled, 3)]),
        cross_pairs.min(u32::from(u16::MAX)) as u16,
    )
}

fn coupling_state(wm_salience: u16, ncde_energy: u16) -> u16 {
    let stability = MAX_SCORE.saturating_sub(ncde_energy.min(MAX_SCORE));
    let diff = wm_salience.min(MAX_SCORE).abs_diff(stability);
    let alignment = MAX_SCORE.saturating_sub(diff);
    let avg = (u32::from(wm_salience.min(MAX_SCORE)) + u32::from(stability)) / 2;
    clamp_score((avg * u32::from(alignment)) / u32::from(MAX_SCORE))
}

fn coupling_reasoning(inp: &IitInputs, cross_checks: u16) -> (u16, u16) {
    let mut warnings = 0u16;
    let mut score = if inp.nsr_verdict.is_some() {
        REASONING_BASE
    } else {
        REASONING_NO_NSR
    };

    if let Some(verdict) = inp.nsr_verdict {
        match verdict {
            0 => {
                if inp.drift <= 3_000 {
                    score = score.saturating_add(1_500);
                } else {
                    score = score.saturating_add(600);
                }
            }
            1 => {
                score = score.saturating_sub(300);
            }
            _ => {
                score = score.saturating_sub(2_000);
            }
        }
    }

    if inp.risk >= 7_000 {
        score = score.saturating_sub(1_200);
    }
    if inp.drift >= 7_000 {
        score = score.saturating_sub(800);
    }

    if inp.nsr_verdict == Some(2) && inp.cde_commit.is_some() {
        let surprise_gate = inp.surprise >= 7_000 || cross_checks >= 4;
        if surprise_gate {
            warnings |= WARN_NSR_DENY_SURPRISE;
            score = score.saturating_sub(800);
        }
    }

    (score.min(MAX_SCORE), warnings)
}

fn weighted_avg(values: &[(u16, u16)]) -> u16 {
    let mut sum: u32 = 0;
    let mut weight: u32 = 0;
    for (value, w) in values {
        sum = sum.saturating_add(u32::from(*value) * u32::from(*w));
        weight = weight.saturating_add(u32::from(*w));
    }
    if weight == 0 {
        return 0;
    }
    clamp_score(sum / weight)
}

fn scale_to_max(value: u32, max: u32) -> u16 {
    if max == 0 {
        return 0;
    }
    let scaled = (value.min(max) * u32::from(MAX_SCORE)) / max;
    clamp_score(scaled)
}

fn clamp_score(value: u32) -> u16 {
    u16::try_from(value.min(u32::from(MAX_SCORE))).unwrap_or(MAX_SCORE)
}

fn commit_inputs(inputs: &IitInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_INPUT);
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.phase_commit.as_bytes());
    hasher.update(&inputs.coherence_plv.to_be_bytes());
    hasher.update(inputs.spike_root.as_bytes());
    hasher.update(&(inputs.spike_counts.len() as u32).to_be_bytes());
    for (kind, count) in &inputs.spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(inputs.ssm_commit.as_bytes());
    hasher.update(&inputs.wm_salience.to_be_bytes());
    hasher.update(&inputs.wm_novelty.to_be_bytes());
    hasher.update(inputs.ncde_commit.as_bytes());
    hasher.update(&inputs.ncde_energy.to_be_bytes());
    match inputs.nsr_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match inputs.nsr_verdict {
        Some(verdict) => {
            hasher.update(&[1]);
            hasher.update(&[verdict]);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match inputs.cde_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&inputs.drift.to_be_bytes());
    hasher.update(&inputs.surprise.to_be_bytes());
    hasher.update(&inputs.risk.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_output(inputs: &IitInputs, phi: u16, coupling: u16, warnings: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_OUTPUT);
    hasher.update(inputs.commit.as_bytes());
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(&phi.to_be_bytes());
    hasher.update(&coupling.to_be_bytes());
    hasher.update(&inputs.coherence_plv.to_be_bytes());
    hasher.update(&warnings.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(prev: Digest32, input_commit: Digest32, output_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_STATE);
    hasher.update(prev.as_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(output_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inputs(coherence: u16, coupling_bump: u16) -> IitInputs {
        let spike_counts = vec![
            (SpikeKind::CausalLink, 4u16 + coupling_bump / 1000),
            (SpikeKind::ConsistencyAlert, 2u16 + coupling_bump / 2000),
        ];
        IitInputs::new(
            42,
            Digest32::new([1u8; 32]),
            coherence,
            Digest32::new([2u8; 32]),
            spike_counts,
            Digest32::new([3u8; 32]),
            6_500,
            1_200,
            Digest32::new([4u8; 32]),
            3_200,
            Some(Digest32::new([5u8; 32])),
            Some(0),
            Some(Digest32::new([6u8; 32])),
            2_000,
            1_500,
            2_500,
        )
    }

    #[test]
    fn output_commit_is_deterministic() {
        let inputs = sample_inputs(4_200, 0);
        let mut monitor_a = IitMonitor::new();
        let mut monitor_b = IitMonitor::new();

        let out_a = monitor_a.tick(&inputs);
        let out_b = monitor_b.tick(&inputs);

        assert_eq!(out_a.commit, out_b.commit);
    }

    #[test]
    fn phi_proxy_rises_with_coherence_and_coupling() {
        let mut monitor = IitMonitor::new();
        let low = sample_inputs(3_000, 0);
        let high = sample_inputs(7_500, 4_000);

        let out_low = monitor.tick(&low);
        let out_high = monitor.tick(&high);

        assert!(out_high.phi_proxy > out_low.phi_proxy);
    }
}
