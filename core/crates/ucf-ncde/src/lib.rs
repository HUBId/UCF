#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::{Digest32, GainBudget};

const PARAMS_DOMAIN: &[u8] = b"ucf.ncde.params.v3";
const INPUT_DOMAIN: &[u8] = b"ucf.ncde.inputs.v3";
const OUTPUT_DOMAIN: &[u8] = b"ucf.ncde.outputs.v3";
const CORE_DOMAIN: &[u8] = b"ucf.ncde.core.v3";
const FLOW_DOMAIN: &[u8] = b"ucf.ncde.flow.v1";
const HASH16_DOMAIN: &[u8] = b"ucf.ncde.hash16.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeParams {
    pub dim: usize,
    pub dt_q: u16,
    pub gain_spike: u16,
    pub gain_phase: u16,
    pub leak: u16,
    pub clamp: i32,
    pub commit: Digest32,
}

impl NcdeParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        dt_q: u16,
        gain_spike: u16,
        gain_phase: u16,
        leak: u16,
        clamp: i32,
    ) -> Self {
        let dim = dim.clamp(1, 64);
        let dt_q = dt_q.clamp(1, 10_000);
        let gain_phase = gain_phase.min(10_000);
        let gain_spike = gain_spike.min(10_000);
        let leak = leak.min(10_000);
        let clamp = clamp.max(1);
        let commit = commit_params(dim, dt_q, gain_phase, gain_spike, leak, clamp);
        Self {
            dim,
            dt_q,
            gain_spike,
            gain_phase,
            leak,
            clamp,
            commit,
        }
    }
}

impl Default for NcdeParams {
    fn default() -> Self {
        Self::new(16, 512, 4_200, 2_800, 600, 50_000)
    }
}

pub fn apply_gain_phase_delta(params: &NcdeParams, delta: i16) -> NcdeParams {
    NcdeParams::new(
        params.dim,
        params.dt_q,
        params.gain_spike,
        apply_i16_delta(params.gain_phase, delta),
        params.leak,
        params.clamp,
    )
}

pub fn apply_gain_spike_delta(params: &NcdeParams, delta: i16) -> NcdeParams {
    NcdeParams::new(
        params.dim,
        params.dt_q,
        apply_i16_delta(params.gain_spike, delta),
        params.gain_phase,
        params.leak,
        params.clamp,
    )
}

pub fn apply_leak_delta(params: &NcdeParams, delta: i16) -> NcdeParams {
    NcdeParams::new(
        params.dim,
        params.dt_q,
        params.gain_spike,
        params.gain_phase,
        apply_i16_delta(params.leak, delta),
        params.clamp,
    )
}

fn apply_i16_delta(value: u16, delta: i16) -> u16 {
    i32::from(value)
        .saturating_add(i32::from(delta))
        .clamp(0, 10_000) as u16
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeCountsSummary {
    pub total: u16,
}

impl SpikeCountsSummary {
    pub fn from_counts(counts: &[(ucf_spikebus::SpikeKind, u16)]) -> Self {
        let total = counts
            .iter()
            .map(|(_, c)| u32::from(*c))
            .sum::<u32>()
            .min(u32::from(u16::MAX));
        Self {
            total: total as u16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub gamma_bucket: u8,
    pub plv: u16,
    pub spike_root: Digest32,
    pub spike_counts: SpikeCountsSummary,
    pub attention_gain: u16,
    pub surprise: u16,
    pub gain_budget_commit: Digest32,
    pub commit: Digest32,
}

impl NcdeInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        gamma_bucket: u8,
        plv: u16,
        spike_root: Digest32,
        spike_counts: SpikeCountsSummary,
        attention_gain: u16,
        surprise: u16,
        gain_budget_commit: Digest32,
    ) -> Self {
        let plv = plv.min(10_000);
        let attention_gain = attention_gain.min(10_000);
        let surprise = surprise.min(10_000);
        let commit = commit_inputs(
            cycle_id,
            phase_commit,
            gamma_bucket,
            plv,
            spike_root,
            spike_counts,
            attention_gain,
            surprise,
            gain_budget_commit,
        );
        Self {
            cycle_id,
            phase_commit,
            gamma_bucket,
            plv,
            spike_root,
            spike_counts,
            attention_gain,
            surprise,
            gain_budget_commit,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeOutputs {
    pub cycle_id: u64,
    pub flow_state: Digest32,
    pub flow_energy: u16,
    pub ncde_state_digest: Digest32,
    pub ncde_energy: u16,
    pub replay_pressure_hint: u16,
    pub commit: Digest32,
}

pub trait ContinuousDynamics {
    fn tick(&mut self, inp: &NcdeInputs) -> NcdeOutputs;

    fn tick_with_budget(&mut self, inp: &NcdeInputs, _budget: &GainBudget) -> NcdeOutputs {
        self.tick(inp)
    }

    fn params(&self) -> NcdeParams;

    fn set_params(&mut self, params: NcdeParams);
}

pub struct NcdeCore {
    pub params: NcdeParams,
    pub prev_flow_state: Digest32,
    pub prev_flow_energy: u16,
    pub commit: Digest32,
}

impl NcdeCore {
    pub fn new(params: NcdeParams) -> Self {
        let prev_flow_state = Digest32::new([0u8; 32]);
        let commit = commit_core(params.commit, prev_flow_state, 0);
        Self {
            params,
            prev_flow_state,
            prev_flow_energy: 0,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &NcdeInputs, budget: &GainBudget) -> NcdeOutputs {
        let flow_state = commit_flow_state(
            self.prev_flow_state,
            inp.phase_commit,
            inp.spike_root,
            inp.attention_gain,
            inp.surprise,
            inp.gamma_bucket,
        );
        let mut flow_energy = compute_flow_energy(self.prev_flow_state, flow_state, inp);
        flow_energy = GainBudget::apply(flow_energy, budget.ncde);
        flow_energy = GainBudget::apply(flow_energy, budget.master);

        let commit = commit_outputs(flow_state, flow_energy, inp.commit, self.params.commit);

        self.prev_flow_state = flow_state;
        self.prev_flow_energy = flow_energy;
        self.commit = commit_core(self.params.commit, flow_state, flow_energy);

        NcdeOutputs {
            cycle_id: inp.cycle_id,
            flow_state,
            flow_energy,
            ncde_state_digest: flow_state,
            ncde_energy: flow_energy,
            replay_pressure_hint: flow_energy / 2,
            commit,
        }
    }
}

impl Default for NcdeCore {
    fn default() -> Self {
        Self::new(NcdeParams::default())
    }
}

impl ContinuousDynamics for NcdeCore {
    fn tick(&mut self, inp: &NcdeInputs) -> NcdeOutputs {
        NcdeCore::tick(self, inp, &GainBudget::default())
    }

    fn tick_with_budget(&mut self, inp: &NcdeInputs, budget: &GainBudget) -> NcdeOutputs {
        NcdeCore::tick(self, inp, budget)
    }

    fn params(&self) -> NcdeParams {
        self.params
    }

    fn set_params(&mut self, params: NcdeParams) {
        *self = Self::new(params);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockContinuousDynamics {
    params: NcdeParams,
    flow_state: Digest32,
    flow_energy: u16,
}

impl MockContinuousDynamics {
    pub fn new(flow_state: Digest32) -> Self {
        Self {
            params: NcdeParams::default(),
            flow_state,
            flow_energy: 1200,
        }
    }
}

impl Default for MockContinuousDynamics {
    fn default() -> Self {
        Self::new(Digest32::new([7u8; 32]))
    }
}

impl ContinuousDynamics for MockContinuousDynamics {
    fn tick(&mut self, inp: &NcdeInputs) -> NcdeOutputs {
        let commit = commit_outputs(
            self.flow_state,
            self.flow_energy,
            inp.commit,
            self.params.commit,
        );
        NcdeOutputs {
            cycle_id: inp.cycle_id,
            flow_state: self.flow_state,
            flow_energy: self.flow_energy,
            ncde_state_digest: self.flow_state,
            ncde_energy: self.flow_energy,
            replay_pressure_hint: self.flow_energy / 2,
            commit,
        }
    }

    fn params(&self) -> NcdeParams {
        self.params
    }

    fn set_params(&mut self, params: NcdeParams) {
        self.params = params;
    }
}

fn commit_params(
    dim: usize,
    dt_q: u16,
    gain_phase: u16,
    gain_spike: u16,
    leak: u16,
    clamp: i32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PARAMS_DOMAIN);
    hasher.update(&(dim as u32).to_be_bytes());
    hasher.update(&dt_q.to_be_bytes());
    hasher.update(&gain_phase.to_be_bytes());
    hasher.update(&gain_spike.to_be_bytes());
    hasher.update(&leak.to_be_bytes());
    hasher.update(&clamp.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    phase_commit: Digest32,
    gamma_bucket: u8,
    plv: u16,
    spike_root: Digest32,
    spike_counts: SpikeCountsSummary,
    attention_gain: u16,
    surprise: u16,
    gain_budget_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(&[gamma_bucket]);
    hasher.update(&plv.to_be_bytes());
    hasher.update(spike_root.as_bytes());
    hasher.update(&spike_counts.total.to_be_bytes());
    hasher.update(&attention_gain.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(gain_budget_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_flow_state(
    prev_flow_state: Digest32,
    phase_commit: Digest32,
    spike_root: Digest32,
    attention_gain: u16,
    surprise: u16,
    gamma_bucket: u8,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(FLOW_DOMAIN);
    hasher.update(prev_flow_state.as_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(spike_root.as_bytes());
    hasher.update(&attention_gain.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&[gamma_bucket]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hash16(value: Digest32) -> u16 {
    let mut hasher = Hasher::new();
    hasher.update(HASH16_DOMAIN);
    hasher.update(value.as_bytes());
    let out = hasher.finalize();
    u16::from_be_bytes([out.as_bytes()[0], out.as_bytes()[1]])
}

fn compute_flow_energy(prev_flow_state: Digest32, flow_state: Digest32, inp: &NcdeInputs) -> u16 {
    let diff = hash16(flow_state) ^ hash16(prev_flow_state);
    let mut energy = (diff.count_ones() as i32).saturating_mul(400);
    energy = energy.min(10_000);

    energy = energy
        .saturating_add(i32::from(inp.spike_counts.total).saturating_mul(50))
        .min(10_000);
    energy = energy.saturating_add(i32::from(inp.surprise) / 3);
    energy = energy.saturating_sub(i32::from(inp.attention_gain) / 10);
    if inp.plv < 3_000 {
        energy = energy.saturating_add(500);
    }

    energy.clamp(0, 10_000) as u16
}

fn commit_outputs(
    flow_state: Digest32,
    flow_energy: u16,
    input_commit: Digest32,
    params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OUTPUT_DOMAIN);
    hasher.update(flow_state.as_bytes());
    hasher.update(&flow_energy.to_be_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, flow_state: Digest32, flow_energy: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CORE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    hasher.update(flow_state.as_bytes());
    hasher.update(&flow_energy.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs() -> NcdeInputs {
        NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            3,
            4_500,
            Digest32::new([2u8; 32]),
            SpikeCountsSummary { total: 3 },
            5_000,
            4_200,
            Digest32::new([3u8; 32]),
        )
    }

    #[test]
    fn deterministic_same_inputs_same_outputs() {
        let params = NcdeParams::default();
        let mut core_a = NcdeCore::new(params);
        let mut core_b = NcdeCore::new(params);
        let inp = base_inputs();
        assert_eq!(
            core_a.tick(&inp, &GainBudget::default()),
            core_b.tick(&inp, &GainBudget::default())
        );
    }

    #[test]
    fn higher_attention_damps_energy() {
        let prev = Digest32::new([9u8; 32]);
        let flow = Digest32::new([7u8; 32]);
        let mut low = base_inputs();
        let mut high = base_inputs();
        low.attention_gain = 1_000;
        high.attention_gain = 9_000;
        low.commit = commit_inputs(
            low.cycle_id,
            low.phase_commit,
            low.gamma_bucket,
            low.plv,
            low.spike_root,
            low.spike_counts,
            low.attention_gain,
            low.surprise,
            low.gain_budget_commit,
        );
        high.commit = commit_inputs(
            high.cycle_id,
            high.phase_commit,
            high.gamma_bucket,
            high.plv,
            high.spike_root,
            high.spike_counts,
            high.attention_gain,
            high.surprise,
            high.gain_budget_commit,
        );
        let low_e = compute_flow_energy(prev, flow, &low);
        let high_e = compute_flow_energy(prev, flow, &high);
        assert!(high_e < low_e);
    }

    #[test]
    fn more_spikes_raise_energy() {
        let params = NcdeParams::default();
        let mut core_low = NcdeCore::new(params);
        let mut core_high = NcdeCore::new(params);
        let mut low = base_inputs();
        let mut high = base_inputs();
        low.spike_counts = SpikeCountsSummary { total: 1 };
        high.spike_counts = SpikeCountsSummary { total: 7 };
        low.commit = commit_inputs(
            low.cycle_id,
            low.phase_commit,
            low.gamma_bucket,
            low.plv,
            low.spike_root,
            low.spike_counts,
            low.attention_gain,
            low.surprise,
            low.gain_budget_commit,
        );
        high.commit = commit_inputs(
            high.cycle_id,
            high.phase_commit,
            high.gamma_bucket,
            high.plv,
            high.spike_root,
            high.spike_counts,
            high.attention_gain,
            high.surprise,
            high.gain_budget_commit,
        );
        assert!(
            core_high.tick(&high, &GainBudget::default()).flow_energy
                > core_low.tick(&low, &GainBudget::default()).flow_energy
        );
    }
}
