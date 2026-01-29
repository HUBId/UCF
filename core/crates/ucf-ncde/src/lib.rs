#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_coupling::SignalId;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const MAX_DIM: usize = 32;
const GAIN_SCALE: i32 = 10_000;
const CONTROL_SCALE: i32 = 1024;
const DT_MIN: u16 = 1;
const DT_MAX: u16 = 10_000;
const LEAK_MAX: u16 = 10_000;
const GAIN_MAX: u16 = 10_000;
const GAIN_MIN: u16 = 0;
const DEFAULT_CLAMP: i32 = 50_000;
const Q_SHIFT: u8 = 12;
const Q_SCALE: i64 = 1 << Q_SHIFT;
const SPIKE_NORM_MAX: u32 = 16;

const PARAMS_DOMAIN: &[u8] = b"ucf.ncde.params.v2";
const STATE_DOMAIN: &[u8] = b"ucf.ncde.state.v2";
const INPUT_DOMAIN: &[u8] = b"ucf.ncde.inputs.v2";
const OUTPUT_DOMAIN: &[u8] = b"ucf.ncde.outputs.v2";
const DIGEST_DOMAIN: &[u8] = b"ucf.ncde.state.digest.v2";
const CONTROL_DOMAIN: &[u8] = b"ucf.ncde.control.v2";
const CORE_DOMAIN: &[u8] = b"ucf.ncde.core.v2";

const PHASE_TABLE: [i16; 16] = [
    -512, -384, -256, -128, 0, 128, 256, 384, 512, 384, 256, 128, 0, -128, -256, -384,
];

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
        let dim = dim.clamp(1, MAX_DIM);
        let dt_q = dt_q.clamp(DT_MIN, DT_MAX);
        let gain_spike = gain_spike.min(GAIN_MAX);
        let gain_phase = gain_phase.min(GAIN_MAX);
        let leak = leak.min(LEAK_MAX);
        let clamp = clamp.abs().max(1);
        let commit = commit_params(dim, dt_q, gain_spike, gain_phase, leak, clamp);
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

pub fn apply_gain_phase_delta(params: &NcdeParams, delta: i16) -> NcdeParams {
    let gain_phase = apply_i16_delta(params.gain_phase, delta, GAIN_MIN, GAIN_MAX);
    NcdeParams::new(
        params.dim,
        params.dt_q,
        params.gain_spike,
        gain_phase,
        params.leak,
        params.clamp,
    )
}

pub fn apply_gain_spike_delta(params: &NcdeParams, delta: i16) -> NcdeParams {
    let gain_spike = apply_i16_delta(params.gain_spike, delta, GAIN_MIN, GAIN_MAX);
    NcdeParams::new(
        params.dim,
        params.dt_q,
        gain_spike,
        params.gain_phase,
        params.leak,
        params.clamp,
    )
}

pub fn apply_leak_delta(params: &NcdeParams, delta: i16) -> NcdeParams {
    let leak = apply_i16_delta(params.leak, delta, GAIN_MIN, LEAK_MAX);
    NcdeParams::new(
        params.dim,
        params.dt_q,
        params.gain_spike,
        params.gain_phase,
        leak,
        params.clamp,
    )
}

fn apply_i16_delta(value: u16, delta: i16, min: u16, max: u16) -> u16 {
    let value = i32::from(value);
    let delta = i32::from(delta);
    let updated = value
        .saturating_add(delta)
        .clamp(i32::from(min), i32::from(max));
    updated as u16
}

impl Default for NcdeParams {
    fn default() -> Self {
        Self::new(16, 512, 4200, 2800, 600, DEFAULT_CLAMP)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcdeState {
    pub y: Vec<i32>,
    pub commit: Digest32,
}

impl NcdeState {
    pub fn new(params: &NcdeParams) -> Self {
        let y = vec![0; params.dim];
        let commit = commit_state(&y, params.commit);
        Self { y, commit }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcdeInputs {
    pub cycle_id: u64,
    pub phase_bus_commit: Digest32,
    pub gamma_bucket: u8,
    pub spike_accepted_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub attention_gain: u16,
    pub coupling_influences_root: Digest32,
    pub coupling_influences: Vec<(SignalId, i16)>,
    pub ssm_state_commit: Digest32,
    pub ssm_salience: u16,
    pub ssm_novelty: u16,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub learning_gain_cap: u16,
    pub commit: Digest32,
}

impl NcdeInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_bus_commit: Digest32,
        gamma_bucket: u8,
        spike_accepted_root: Digest32,
        mut spike_counts: Vec<(SpikeKind, u16)>,
        attention_gain: u16,
        coupling_influences_root: Digest32,
        mut coupling_influences: Vec<(SignalId, i16)>,
        ssm_state_commit: Digest32,
        ssm_salience: u16,
        ssm_novelty: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
        learning_gain_cap: u16,
    ) -> Self {
        spike_counts.sort_by(|(kind_a, _), (kind_b, _)| kind_a.cmp(kind_b));
        coupling_influences.sort_by(|(id_a, _), (id_b, _)| id_a.cmp(id_b));
        let learning_gain_cap = learning_gain_cap.min(GAIN_MAX);
        let commit = commit_inputs(
            cycle_id,
            phase_bus_commit,
            gamma_bucket,
            spike_accepted_root,
            &spike_counts,
            attention_gain,
            coupling_influences_root,
            &coupling_influences,
            ssm_state_commit,
            ssm_salience,
            ssm_novelty,
            risk,
            drift,
            surprise,
            learning_gain_cap,
        );
        Self {
            cycle_id,
            phase_bus_commit,
            gamma_bucket,
            spike_accepted_root,
            spike_counts,
            attention_gain,
            coupling_influences_root,
            coupling_influences,
            ssm_state_commit,
            ssm_salience,
            ssm_novelty,
            risk,
            drift,
            surprise,
            learning_gain_cap,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeOutputs {
    pub cycle_id: u64,
    pub ncde_state_digest: Digest32,
    pub ncde_energy: u16,
    pub replay_pressure_hint: u16,
    pub commit: Digest32,
}

pub struct NcdeCore {
    pub params: NcdeParams,
    pub state: NcdeState,
    pub prev_energy: u16,
    pub commit: Digest32,
}

impl NcdeCore {
    pub fn new(params: NcdeParams) -> Self {
        let state = NcdeState::new(&params);
        let commit = commit_core(params.commit, state.commit);
        Self {
            params,
            state,
            prev_energy: 0,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &NcdeInputs) -> NcdeOutputs {
        self.reset_state_if_dim_mismatch();
        let dim = self.params.dim;
        let control = build_control_vector(inp, &self.params, dim);
        let phase_terms =
            build_phase_terms(inp.gamma_bucket, &self.params, inp.learning_gain_cap, dim);
        let mut next_y = self.state.y.clone();

        for (idx, value) in next_y.iter_mut().enumerate().take(dim) {
            let current = self.state.y[idx];
            let k1 = flow(current, control.u[idx], phase_terms[idx], &self.params);
            let y_pred = current.saturating_add(scale_dt(k1, self.params.dt_q));
            let k2 = flow(y_pred, control.u[idx], phase_terms[idx], &self.params);
            let delta = scale_dt(avg_i32(k1, k2), self.params.dt_q);
            let updated = current.saturating_add(delta);
            *value = clamp_i32(updated, self.params.clamp);
        }

        let ncde_state_digest = commit_state_digest(&next_y);
        let ncde_energy =
            update_energy(&next_y, &control, inp, self.prev_energy, self.params.clamp);
        let replay_pressure_hint = replay_pressure_hint(ncde_energy, inp, control.phi_influence);

        let state_commit = commit_state(&next_y, self.params.commit);
        self.state = NcdeState {
            y: next_y,
            commit: state_commit,
        };
        self.prev_energy = ncde_energy;
        self.commit = commit_core(self.params.commit, self.state.commit);

        let commit = commit_outputs(
            inp.cycle_id,
            ncde_state_digest,
            ncde_energy,
            replay_pressure_hint,
            inp.commit,
            self.params.commit,
            state_commit,
        );

        NcdeOutputs {
            cycle_id: inp.cycle_id,
            ncde_state_digest,
            ncde_energy,
            replay_pressure_hint,
            commit,
        }
    }

    fn reset_state_if_dim_mismatch(&mut self) {
        if self.state.y.len() != self.params.dim {
            self.state = NcdeState::new(&self.params);
            self.prev_energy = 0;
            self.commit = commit_core(self.params.commit, self.state.commit);
        }
    }
}

impl Default for NcdeCore {
    fn default() -> Self {
        Self::new(NcdeParams::default())
    }
}

struct ControlSignal {
    u: Vec<i32>,
    phi_influence: i16,
}

fn commit_params(
    dim: usize,
    dt_q: u16,
    gain_spike: u16,
    gain_phase: u16,
    leak: u16,
    clamp: i32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PARAMS_DOMAIN);
    hasher.update(&(dim as u32).to_be_bytes());
    hasher.update(&dt_q.to_be_bytes());
    hasher.update(&gain_spike.to_be_bytes());
    hasher.update(&gain_phase.to_be_bytes());
    hasher.update(&leak.to_be_bytes());
    hasher.update(&clamp.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(y: &[i32], params_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STATE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    for value in y.iter() {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    phase_bus_commit: Digest32,
    gamma_bucket: u8,
    spike_accepted_root: Digest32,
    spike_counts: &[(SpikeKind, u16)],
    attention_gain: u16,
    coupling_influences_root: Digest32,
    coupling_influences: &[(SignalId, i16)],
    ssm_state_commit: Digest32,
    ssm_salience: u16,
    ssm_novelty: u16,
    risk: u16,
    drift: u16,
    surprise: u16,
    learning_gain_cap: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(&[gamma_bucket]);
    hasher.update(spike_accepted_root.as_bytes());
    hasher.update(&attention_gain.to_be_bytes());
    hasher.update(coupling_influences_root.as_bytes());
    hasher.update(
        &u32::try_from(spike_counts.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (kind, count) in spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(
        &u32::try_from(coupling_influences.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (signal, value) in coupling_influences {
        hasher.update(&signal.as_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    hasher.update(ssm_state_commit.as_bytes());
    hasher.update(&ssm_salience.to_be_bytes());
    hasher.update(&ssm_novelty.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&learning_gain_cap.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state_digest(y: &[i32]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DIGEST_DOMAIN);
    for value in y.iter() {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    ncde_state_digest: Digest32,
    ncde_energy: u16,
    replay_pressure_hint: u16,
    input_commit: Digest32,
    params_commit: Digest32,
    state_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OUTPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(ncde_state_digest.as_bytes());
    hasher.update(&ncde_energy.to_be_bytes());
    hasher.update(&replay_pressure_hint.to_be_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, state_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CORE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn build_control_vector(inp: &NcdeInputs, params: &NcdeParams, dim: usize) -> ControlSignal {
    let total_spikes: u32 = inp
        .spike_counts
        .iter()
        .map(|(_, count)| u32::from(*count))
        .sum();
    let feature = spike_count(inp, SpikeKind::Feature);
    let novelty = spike_count(inp, SpikeKind::Novelty);
    let threat = spike_count(inp, SpikeKind::Threat);
    let causal = spike_count(inp, SpikeKind::CausalLink);
    let thought = spike_count(inp, SpikeKind::ThoughtOnly);

    let feature_norm = normalize_spike(feature, total_spikes);
    let novelty_norm = normalize_spike(novelty, total_spikes);
    let threat_norm = normalize_spike(threat, total_spikes);
    let causal_norm = normalize_spike(causal, total_spikes);
    let thought_norm = normalize_spike(thought, total_spikes);

    let combined =
        feature_norm * 6 + novelty_norm * 4 + causal_norm * 3 + thought_norm * 2 - threat_norm * 6;
    let combined = combined.clamp(-CONTROL_SCALE * 10, CONTROL_SCALE * 10);

    let (phi_influence, attention_influence) = coupling_biases(inp);
    let attention_gain = apply_gain_influence(inp.attention_gain, attention_influence);
    let spike_gain = (i32::from(params.gain_spike) * i32::from(inp.learning_gain_cap)) / GAIN_SCALE;
    let spike_gain = spike_gain.max(1);
    let mut base = (combined * spike_gain) / GAIN_SCALE;
    base = (base * i32::from(attention_gain)) / GAIN_SCALE;

    let coupling_bias = (i32::from(phi_influence) * CONTROL_SCALE) / GAIN_SCALE;

    let mut u = vec![0i32; dim];
    for (idx, value) in u.iter_mut().enumerate() {
        let sign = hashed_sign(inp.spike_accepted_root, idx as u16)
            * gamma_sign(inp.gamma_bucket, idx as u8);
        let mut drive = base.saturating_mul(sign);
        drive = drive.saturating_add(coupling_bias);
        *value = clamp_i32(drive, params.clamp);
    }

    ControlSignal { u, phi_influence }
}

fn coupling_biases(inp: &NcdeInputs) -> (i16, i16) {
    let mut phi = 0i16;
    let mut attention = 0i16;
    for (signal, value) in &inp.coupling_influences {
        match signal {
            SignalId::PhiProxy => phi = phi.saturating_add(*value),
            SignalId::AttentionFinalGain => attention = attention.saturating_add(*value),
            _ => {}
        }
    }
    (phi, attention)
}

fn apply_gain_influence(base: u16, influence: i16) -> u16 {
    let base = i32::from(base);
    let updated = base.saturating_add(i32::from(influence));
    updated.clamp(0, GAIN_SCALE) as u16
}

fn build_phase_terms(
    gamma_bucket: u8,
    params: &NcdeParams,
    learning_gain_cap: u16,
    dim: usize,
) -> Vec<i32> {
    let mut terms = vec![0i32; dim];
    for (idx, value) in terms.iter_mut().enumerate() {
        let table_idx = ((gamma_bucket as usize).wrapping_add(idx)) & 0x0f;
        let coeff = i32::from(PHASE_TABLE[table_idx]);
        let gain_phase = (i32::from(params.gain_phase) * i32::from(learning_gain_cap)) / GAIN_SCALE;
        let scaled = (coeff * gain_phase) / GAIN_SCALE;
        *value = scaled;
    }
    terms
}

fn flow(y: i32, u: i32, phase: i32, params: &NcdeParams) -> i32 {
    let leak_term = (i64::from(params.leak) * i64::from(y)) / i64::from(GAIN_SCALE);
    let value = i64::from(u) - leak_term + i64::from(phase);
    clamp_i64(value, params.clamp)
}

fn scale_dt(value: i32, dt_q: u16) -> i32 {
    let scaled = (i64::from(value) * i64::from(dt_q)) / Q_SCALE;
    scaled as i32
}

fn avg_i32(a: i32, b: i32) -> i32 {
    ((i64::from(a) + i64::from(b)) / 2) as i32
}

fn update_energy(
    y: &[i32],
    control: &ControlSignal,
    inp: &NcdeInputs,
    prev_energy: u16,
    clamp: i32,
) -> u16 {
    let mean_abs = mean_abs(y);
    let mut energy = if clamp == 0 {
        0
    } else {
        (mean_abs * GAIN_SCALE) / clamp.abs().max(1)
    };

    let attention_boost = i32::from(inp.attention_gain) / 8;
    let risk_penalty = i32::from(inp.risk) / 6;
    let drift_penalty = i32::from(inp.drift) / 10;
    let salience_boost = i32::from(inp.ssm_salience) / 12;
    let novelty_boost = i32::from(inp.ssm_novelty) / 16;
    let phi_penalty = i32::from(control.phi_influence.abs()) / 20;

    energy = energy
        .saturating_add(attention_boost)
        .saturating_add(salience_boost)
        .saturating_add(novelty_boost)
        .saturating_sub(risk_penalty)
        .saturating_sub(drift_penalty)
        .saturating_sub(phi_penalty);

    let smoothed = (i32::from(prev_energy) + energy) / 2;
    smoothed.clamp(0, GAIN_SCALE) as u16
}

fn replay_pressure_hint(energy: u16, inp: &NcdeInputs, phi_influence: i16) -> u16 {
    let energy = i32::from(energy);
    let surprise = i32::from(inp.surprise);
    let phi_level = ((i32::from(phi_influence) + GAIN_SCALE) / 2).clamp(0, GAIN_SCALE);
    let phi_low = GAIN_SCALE - phi_level;

    let mut pressure = energy.saturating_mul(surprise) / GAIN_SCALE;
    pressure = pressure.saturating_mul(phi_low) / GAIN_SCALE;
    pressure.clamp(0, GAIN_SCALE) as u16
}

fn spike_count(inp: &NcdeInputs, kind: SpikeKind) -> u16 {
    inp.spike_counts
        .iter()
        .find_map(|(entry_kind, count)| (*entry_kind == kind).then_some(*count))
        .unwrap_or(0)
}

fn normalize_spike(count: u16, total: u32) -> i32 {
    if total == 0 {
        return 0;
    }
    let denom = total.max(SPIKE_NORM_MAX);
    let value = (i32::from(count) * CONTROL_SCALE) / denom as i32;
    value.clamp(0, CONTROL_SCALE)
}

fn hashed_sign(root: Digest32, idx: u16) -> i32 {
    let mut hasher = Hasher::new();
    hasher.update(CONTROL_DOMAIN);
    hasher.update(root.as_bytes());
    hasher.update(&idx.to_be_bytes());
    let bytes = hasher.finalize();
    if bytes.as_bytes()[0] & 1 == 0 {
        1
    } else {
        -1
    }
}

fn gamma_sign(gamma_bucket: u8, idx: u8) -> i32 {
    let pattern = gamma_bucket.wrapping_add(idx) & 1;
    if pattern == 0 {
        1
    } else {
        -1
    }
}

fn mean_abs(values: &[i32]) -> i32 {
    if values.is_empty() {
        return 0;
    }
    let sum: i32 = values.iter().map(|value| value.abs()).sum();
    sum / values.len() as i32
}

fn clamp_i32(value: i32, max_state: i32) -> i32 {
    clamp_i64(i64::from(value), max_state)
}

fn clamp_i64(value: i64, max_state: i32) -> i32 {
    let max_state = i64::from(max_state.abs().max(1));
    value.clamp(-max_state, max_state) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs() -> NcdeInputs {
        NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            3,
            Digest32::new([2u8; 32]),
            vec![
                (SpikeKind::Feature, 2),
                (SpikeKind::Threat, 1),
                (SpikeKind::CausalLink, 1),
            ],
            5000,
            Digest32::new([3u8; 32]),
            vec![(SignalId::PhiProxy, -1200)],
            Digest32::new([4u8; 32]),
            2400,
            1800,
            1200,
            800,
            4200,
            10_000,
        )
    }

    #[test]
    fn ncde_is_deterministic_for_same_inputs() {
        let params = NcdeParams::new(8, 512, 3200, 2100, 500, 20_000);
        let mut core_a = NcdeCore::new(params);
        let mut core_b = NcdeCore::new(params);
        let inputs = base_inputs();

        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);

        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(out_a.ncde_state_digest, out_b.ncde_state_digest);
    }

    #[test]
    fn state_decays_with_leak_and_zero_control() {
        let params = NcdeParams::new(6, 512, 0, 0, 4000, 30_000);
        let mut core = NcdeCore::new(params);
        core.state.y = vec![12_000; core.params.dim];
        core.state.commit = commit_state(&core.state.y, core.params.commit);
        let inputs = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            2,
            Digest32::new([2u8; 32]),
            vec![],
            0,
            Digest32::new([3u8; 32]),
            vec![],
            Digest32::new([4u8; 32]),
            0,
            0,
            0,
            0,
            0,
            10_000,
        );

        let before = core.state.y[0].abs();
        let out = core.tick(&inputs);
        let after = core.state.y[0].abs();

        assert!(after < before, "state should decay with leak");
        assert_eq!(out.ncde_energy, core.prev_energy);
    }

    #[test]
    fn phase_bucket_changes_state_deterministically() {
        let params = NcdeParams::new(8, 512, 2000, 5000, 200, 30_000);
        let mut core_a = NcdeCore::new(params);
        let mut core_b = NcdeCore::new(params);
        let mut inputs_a = base_inputs();
        let mut inputs_b = base_inputs();
        inputs_a.gamma_bucket = 1;
        inputs_b.gamma_bucket = 9;

        let out_a = core_a.tick(&inputs_a);
        let out_b = core_b.tick(&inputs_b);

        assert_ne!(out_a.ncde_state_digest, out_b.ncde_state_digest);
    }

    #[test]
    fn feature_spikes_increase_energy() {
        let params = NcdeParams::new(8, 512, 4500, 2000, 500, 40_000);
        let inputs_low = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            2,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Feature, 1)],
            4000,
            Digest32::new([3u8; 32]),
            vec![],
            Digest32::new([4u8; 32]),
            0,
            0,
            200,
            100,
            2000,
            10_000,
        );
        let inputs_high = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            2,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Feature, 6)],
            4000,
            Digest32::new([3u8; 32]),
            vec![],
            Digest32::new([4u8; 32]),
            0,
            0,
            200,
            100,
            2000,
            10_000,
        );
        let mut core_low = NcdeCore::new(params);
        let mut core_high = NcdeCore::new(params);

        let out_low = core_low.tick(&inputs_low);
        let out_high = core_high.tick(&inputs_high);

        assert!(out_high.ncde_energy > out_low.ncde_energy);
    }
}
