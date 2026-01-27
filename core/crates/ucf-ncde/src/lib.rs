#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const DIM_H_MIN: u16 = 16;
const DIM_H_MAX: u16 = 64;
const DIM_U_MIN: u16 = 8;
const DIM_U_MAX: u16 = 32;
const DT_Q_MIN: u16 = 1;
const DT_Q_MAX: u16 = 1000;
const STEPS_MIN: u8 = 1;
const STEPS_MAX: u8 = 8;
const CONTROL_SCALE: i32 = 1024;
const DT_SCALE: i32 = 1000;
const MAX_COEFF: i32 = 64;
const MAX_H: i32 = 4_194_304;
const SUMMARY_LEN: usize = 8;
const STRENGTH_MAX: u16 = 10_000;

const PARAMS_DOMAIN: &[u8] = b"ucf.ncde.params.v1";
const STATE_DOMAIN: &[u8] = b"ucf.ncde.state.v1";
const CONTROL_DOMAIN: &[u8] = b"ucf.ncde.control.v1";
const OUTPUT_DOMAIN: &[u8] = b"ucf.ncde.output.v1";
const COEFF_A_DOMAIN: &[u8] = b"ucf.ncde.coeff.a";
const COEFF_B_DOMAIN: &[u8] = b"ucf.ncde.coeff.b";
const COEFF_C_DOMAIN: &[u8] = b"ucf.ncde.coeff.c";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeParams {
    pub dim_h: u16,
    pub dim_u: u16,
    pub dt_q: u16,
    pub steps: u8,
    pub attractor_strength: u16,
    pub commit: Digest32,
}

impl NcdeParams {
    pub fn new(dim_h: u16, dim_u: u16, dt_q: u16, steps: u8, attractor_strength: u16) -> Self {
        let dim_h = dim_h.clamp(DIM_H_MIN, DIM_H_MAX);
        let dim_u = dim_u.clamp(DIM_U_MIN, DIM_U_MAX);
        let dt_q = dt_q.clamp(DT_Q_MIN, DT_Q_MAX);
        let steps = steps.clamp(STEPS_MIN, STEPS_MAX);
        let attractor_strength = attractor_strength.min(STRENGTH_MAX);
        let commit = commit_params(dim_h, dim_u, dt_q, steps, attractor_strength);
        Self {
            dim_h,
            dim_u,
            dt_q,
            steps,
            attractor_strength,
            commit,
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.dim_h as usize, self.dim_u as usize)
    }
}

impl Default for NcdeParams {
    fn default() -> Self {
        Self::new(32, 16, 100, 2, 2500)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcdeState {
    pub h: Vec<i32>,
    pub t_q: u64,
    pub commit: Digest32,
}

impl NcdeState {
    pub fn new(params: &NcdeParams) -> Self {
        let h = vec![0; params.dim_h as usize];
        let t_q = 0;
        let commit = commit_state(&h, t_q, params.commit);
        Self { h, t_q, commit }
    }

    pub fn reset_if_dim_mismatch(&mut self, params: &NcdeParams) {
        if self.h.len() != params.dim_h as usize {
            *self = Self::new(params);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlFrame {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub global_phase: u16,
    pub coherence_plv: u16,
    pub influence_pulses_root: Digest32,
    pub spike_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub drift: u16,
    pub surprise: u16,
    pub risk: u16,
    pub attn_gain: u16,
    pub commit: Digest32,
}

impl ControlFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        global_phase: u16,
        coherence_plv: u16,
        influence_pulses_root: Digest32,
        spike_root: Digest32,
        mut spike_counts: Vec<(SpikeKind, u16)>,
        drift: u16,
        surprise: u16,
        risk: u16,
        attn_gain: u16,
    ) -> Self {
        spike_counts.sort_by(|(kind_a, _), (kind_b, _)| kind_a.cmp(kind_b));
        let commit = commit_control(
            cycle_id,
            phase_commit,
            global_phase,
            coherence_plv,
            influence_pulses_root,
            spike_root,
            &spike_counts,
            drift,
            surprise,
            risk,
            attn_gain,
        );
        Self {
            cycle_id,
            phase_commit,
            global_phase,
            coherence_plv,
            influence_pulses_root,
            spike_root,
            spike_counts,
            drift,
            surprise,
            risk,
            attn_gain,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcdeOutput {
    pub cycle_id: u64,
    pub h_commit: Digest32,
    pub h_summary: Vec<i16>,
    pub energy: u16,
    pub commit: Digest32,
}

pub struct NcdeCore {
    pub params: NcdeParams,
    pub state: NcdeState,
}

impl NcdeCore {
    pub fn new(params: NcdeParams) -> Self {
        let state = NcdeState::new(&params);
        Self { params, state }
    }

    pub fn tick(&mut self, ctrl: &ControlFrame) -> NcdeOutput {
        self.state.reset_if_dim_mismatch(&self.params);
        let (dim_h, dim_u) = self.params.dims();
        let mut h = self.state.h.clone();
        let u = build_control_vector(ctrl, dim_u);
        let attractor = triangle_wave(ctrl.global_phase, CONTROL_SCALE);
        let dt_q = i64::from(self.params.dt_q);
        let steps = self.params.steps.max(1);

        for _ in 0..steps {
            let k1 = derivative(&h, &u, attractor, &self.params);
            let h_tmp = apply_step(&h, &k1, dt_q);
            let k2 = derivative(&h_tmp, &u, attractor, &self.params);
            h = apply_heun(&h, &k1, &k2, dt_q);
            clamp_vec(&mut h);
        }

        if h.len() != dim_h {
            h.resize(dim_h, 0);
        }

        let t_q = self
            .state
            .t_q
            .saturating_add(u64::from(self.params.dt_q) * u64::from(self.params.steps));
        let state_commit = commit_state(&h, t_q, self.params.commit);
        self.state = NcdeState {
            h: h.clone(),
            t_q,
            commit: state_commit,
        };

        let h_commit = commit_h(&h);
        let energy = energy_from_state(&h);
        let h_summary = summarize_h(&h);
        let commit = commit_output(
            ctrl.cycle_id,
            h_commit,
            energy,
            ctrl.commit,
            state_commit,
            self.params.commit,
        );

        NcdeOutput {
            cycle_id: ctrl.cycle_id,
            h_commit,
            h_summary,
            energy,
            commit,
        }
    }
}

impl Default for NcdeCore {
    fn default() -> Self {
        Self::new(NcdeParams::default())
    }
}

fn commit_params(
    dim_h: u16,
    dim_u: u16,
    dt_q: u16,
    steps: u8,
    attractor_strength: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PARAMS_DOMAIN);
    hasher.update(&dim_h.to_be_bytes());
    hasher.update(&dim_u.to_be_bytes());
    hasher.update(&dt_q.to_be_bytes());
    hasher.update(&[steps]);
    hasher.update(&attractor_strength.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(h: &[i32], t_q: u64, params_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STATE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    hasher.update(&t_q.to_be_bytes());
    for value in h {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_h(h: &[i32]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ncde.state.h.v1");
    for value in h {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_control(
    cycle_id: u64,
    phase_commit: Digest32,
    global_phase: u16,
    coherence_plv: u16,
    influence_pulses_root: Digest32,
    spike_root: Digest32,
    spike_counts: &[(SpikeKind, u16)],
    drift: u16,
    surprise: u16,
    risk: u16,
    attn_gain: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CONTROL_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(&global_phase.to_be_bytes());
    hasher.update(&coherence_plv.to_be_bytes());
    hasher.update(influence_pulses_root.as_bytes());
    hasher.update(spike_root.as_bytes());
    for (kind, count) in spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&attn_gain.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_output(
    cycle_id: u64,
    h_commit: Digest32,
    energy: u16,
    control_commit: Digest32,
    state_commit: Digest32,
    params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OUTPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(h_commit.as_bytes());
    hasher.update(&energy.to_be_bytes());
    hasher.update(control_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn build_control_vector(ctrl: &ControlFrame, dim_u: usize) -> Vec<i32> {
    let mut values = Vec::with_capacity(dim_u);
    values.push(centered_value(ctrl.global_phase, u16::MAX));
    values.push(centered_value(ctrl.coherence_plv, STRENGTH_MAX));
    values.push(centered_value(ctrl.drift, STRENGTH_MAX));
    values.push(centered_value(ctrl.surprise, STRENGTH_MAX));
    values.push(centered_value(ctrl.risk, STRENGTH_MAX));
    values.push(centered_value(ctrl.attn_gain, STRENGTH_MAX));

    let total_spikes: u32 = ctrl
        .spike_counts
        .iter()
        .map(|(_, count)| u32::from(*count))
        .sum();
    for (_, count) in ctrl.spike_counts.iter() {
        if values.len() >= dim_u {
            break;
        }
        let norm = if total_spikes == 0 {
            0
        } else {
            (i32::from(*count) * CONTROL_SCALE) / total_spikes as i32
        };
        values.push(norm);
    }

    while values.len() < dim_u {
        values.push(0);
    }

    values
}

fn derivative(h: &[i32], u: &[i32], attractor: i32, params: &NcdeParams) -> Vec<i32> {
    let dim_h = params.dim_h as usize;
    let dim_u = params.dim_u as usize;
    let strength = params.attractor_strength.min(STRENGTH_MAX) as i64;
    let strength_scale = i64::from(STRENGTH_MAX);
    let mut out = vec![0; dim_h];

    for i in 0..dim_h {
        let a_coeff = coeff_signed(params.commit, COEFF_A_DOMAIN, i as u32) as i64;
        let mut acc = (a_coeff * h[i] as i64) / i64::from(CONTROL_SCALE);
        for (j, value) in u.iter().take(dim_u).enumerate() {
            let b_coeff =
                coeff_signed(params.commit, COEFF_B_DOMAIN, (i * dim_u + j) as u32) as i64;
            acc = acc.saturating_add((b_coeff * i64::from(*value)) / i64::from(CONTROL_SCALE));
        }
        let c_coeff = coeff_abs(params.commit, COEFF_C_DOMAIN, i as u32) as i64;
        let pull = i64::from(attractor) - h[i] as i64;
        let pull_term =
            (c_coeff * pull * strength) / (i64::from(CONTROL_SCALE) * strength_scale.max(1));
        acc = acc.saturating_add(pull_term);
        out[i] = clamp_i32(acc);
    }

    out
}

fn apply_step(h: &[i32], k: &[i32], dt_q: i64) -> Vec<i32> {
    let mut out = Vec::with_capacity(h.len());
    for (value, deriv) in h.iter().zip(k.iter()) {
        let delta = (dt_q * i64::from(*deriv)) / i64::from(DT_SCALE);
        out.push(clamp_i32(i64::from(*value).saturating_add(delta)));
    }
    out
}

fn apply_heun(h: &[i32], k1: &[i32], k2: &[i32], dt_q: i64) -> Vec<i32> {
    let mut out = Vec::with_capacity(h.len());
    for ((value, deriv1), deriv2) in h.iter().zip(k1.iter()).zip(k2.iter()) {
        let avg = i64::from(*deriv1).saturating_add(i64::from(*deriv2)) / 2;
        let delta = (dt_q * avg) / i64::from(DT_SCALE);
        out.push(clamp_i32(i64::from(*value).saturating_add(delta)));
    }
    out
}

fn summarize_h(h: &[i32]) -> Vec<i16> {
    h.iter()
        .take(SUMMARY_LEN)
        .map(|value| clamp_i16(i64::from(*value) / 8))
        .collect()
}

fn energy_from_state(h: &[i32]) -> u16 {
    if h.is_empty() {
        return 0;
    }
    let total: i64 = h.iter().map(|value| i64::from(value.abs())).sum();
    let avg = total / h.len() as i64;
    clamp_u16(avg / 32)
}

fn triangle_wave(phase: u16, scale: i32) -> i32 {
    let phase = i32::from(phase);
    let span = 65_536i32;
    let half = span / 2;
    let value = if phase < half { phase } else { span - phase };
    let signed = value * 2 - half;
    (signed * scale) / half
}

fn centered_value(value: u16, max: u16) -> i32 {
    if max == 0 {
        return 0;
    }
    let max_i32 = i32::from(max);
    let doubled = i32::from(value) * 2 - max_i32;
    (doubled * CONTROL_SCALE) / max_i32
}

fn coeff_signed(seed: Digest32, domain: &[u8], idx: u32) -> i32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(seed.as_bytes());
    hasher.update(&idx.to_be_bytes());
    let bytes = hasher.finalize();
    let raw = i16::from_be_bytes([bytes.as_bytes()[0], bytes.as_bytes()[1]]);
    (raw as i32).rem_euclid(MAX_COEFF * 2 + 1) - MAX_COEFF
}

fn coeff_abs(seed: Digest32, domain: &[u8], idx: u32) -> i32 {
    let value = coeff_signed(seed, domain, idx).abs();
    value.max(1)
}

fn clamp_vec(values: &mut [i32]) {
    for value in values {
        *value = (*value).clamp(-MAX_H, MAX_H);
    }
}

fn clamp_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn clamp_i16(value: i64) -> i16 {
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

fn clamp_u16(value: i64) -> u16 {
    value.clamp(0, 10_000) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncde_is_deterministic_for_same_control() {
        let params = NcdeParams::new(24, 10, 80, 2, 1800);
        let mut core_a = NcdeCore::new(params);
        let mut core_b = NcdeCore::new(params);
        let control = ControlFrame::new(
            7,
            Digest32::new([1u8; 32]),
            12_000,
            8000,
            Digest32::new([2u8; 32]),
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Novelty, 3), (SpikeKind::Threat, 1)],
            1200,
            4200,
            2000,
            1500,
        );

        let out_a = core_a.tick(&control);
        let out_b = core_b.tick(&control);

        assert_eq!(out_a.commit, out_b.commit);
    }

    #[test]
    fn stronger_attractor_reduces_energy_variance() {
        let base = NcdeParams::new(24, 10, 80, 2, 500);
        let mut weak = NcdeCore::new(base);
        let strong_params = NcdeParams::new(24, 10, 80, 2, 4500);
        let mut strong = NcdeCore::new(strong_params);
        let mut weak_energy = Vec::new();
        let mut strong_energy = Vec::new();

        for (idx, phase) in [1000u16, 8000, 16000, 24000, 32000, 40000]
            .iter()
            .enumerate()
        {
            let control = ControlFrame::new(
                idx as u64,
                Digest32::new([1u8; 32]),
                *phase,
                6000,
                Digest32::new([2u8; 32]),
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Novelty, 2), (SpikeKind::ConsistencyAlert, 1)],
                1000,
                2000,
                1500,
                1800,
            );
            weak_energy.push(weak.tick(&control).energy as i64);
            strong_energy.push(strong.tick(&control).energy as i64);
        }

        let weak_var = variance(&weak_energy);
        let strong_var = variance(&strong_energy);

        assert!(strong_var <= weak_var);
    }

    fn variance(values: &[i64]) -> i64 {
        if values.is_empty() {
            return 0;
        }
        let mean: i64 = values.iter().sum::<i64>() / values.len() as i64;
        values
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<i64>()
            / values.len() as i64
    }
}
