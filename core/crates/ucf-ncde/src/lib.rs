#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const DIM: usize = 16;
const CONTROL_SCALE: i32 = 1024;
const GAIN_SCALE: i32 = 10_000;
const ENERGY_MAX: i32 = 10_000;
const DT_MIN: u16 = 1;
const DT_MAX: u16 = 10_000;
const LEAK_MAX: u16 = 10_000;
const GAIN_MAX: u16 = 10_000;
const DEFAULT_MAX_STATE: i32 = 50_000;
const PHASE_MAX: i32 = 65_536;
const SPIKE_NORM_MAX: u16 = 16;

const PARAMS_DOMAIN: &[u8] = b"ucf.ncde.params.v1";
const STATE_DOMAIN: &[u8] = b"ucf.ncde.state.v1";
const INPUT_DOMAIN: &[u8] = b"ucf.ncde.inputs.v1";
const OUTPUT_DOMAIN: &[u8] = b"ucf.ncde.outputs.v1";
const DIGEST_DOMAIN: &[u8] = b"ucf.ncde.state.digest.v1";
const CONTROL_DOMAIN: &[u8] = b"ucf.ncde.control.v1";
const CORE_DOMAIN: &[u8] = b"ucf.ncde.core.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeParams {
    pub dt: u16,
    pub gain_phase: u16,
    pub gain_spike: u16,
    pub gain_influence: u16,
    pub leak: u16,
    pub max_state: i32,
    pub commit: Digest32,
}

impl NcdeParams {
    pub fn new(
        dt: u16,
        gain_phase: u16,
        gain_spike: u16,
        gain_influence: u16,
        leak: u16,
        max_state: i32,
    ) -> Self {
        let dt = dt.clamp(DT_MIN, DT_MAX);
        let gain_phase = gain_phase.min(GAIN_MAX);
        let gain_spike = gain_spike.min(GAIN_MAX);
        let gain_influence = gain_influence.min(GAIN_MAX);
        let leak = leak.min(LEAK_MAX);
        let max_state = max_state.abs().max(1);
        let commit = commit_params(dt, gain_phase, gain_spike, gain_influence, leak, max_state);
        Self {
            dt,
            gain_phase,
            gain_spike,
            gain_influence,
            leak,
            max_state,
            commit,
        }
    }
}

impl Default for NcdeParams {
    fn default() -> Self {
        Self::new(400, 3500, 4200, 2200, 600, DEFAULT_MAX_STATE)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeState {
    pub x: [i32; DIM],
    pub energy: u16,
    pub commit: Digest32,
}

impl NcdeState {
    pub fn new(params: &NcdeParams) -> Self {
        let x = [0; DIM];
        let energy = ENERGY_MAX as u16 / 2;
        let commit = commit_state(&x, energy, params.commit);
        Self { x, energy, commit }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcdeInputs {
    pub cycle_id: u64,
    pub phase_frame_commit: Digest32,
    pub phase_u16: u16,
    pub tcf_energy_smooth: u16,
    pub spike_accepted_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub influence_pulses_root: Digest32,
    pub coherence_plv: u16,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub commit: Digest32,
}

impl NcdeInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_frame_commit: Digest32,
        phase_u16: u16,
        tcf_energy_smooth: u16,
        spike_accepted_root: Digest32,
        mut spike_counts: Vec<(SpikeKind, u16)>,
        influence_pulses_root: Digest32,
        coherence_plv: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
    ) -> Self {
        spike_counts.sort_by(|(kind_a, _), (kind_b, _)| kind_a.cmp(kind_b));
        let commit = commit_inputs(
            cycle_id,
            phase_frame_commit,
            phase_u16,
            tcf_energy_smooth,
            spike_accepted_root,
            &spike_counts,
            influence_pulses_root,
            coherence_plv,
            risk,
            drift,
            surprise,
        );
        Self {
            cycle_id,
            phase_frame_commit,
            phase_u16,
            tcf_energy_smooth,
            spike_accepted_root,
            spike_counts,
            influence_pulses_root,
            coherence_plv,
            risk,
            drift,
            surprise,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeOutputs {
    pub cycle_id: u64,
    pub state_commit: Digest32,
    pub energy: u16,
    pub state_digest: Digest32,
    pub commit: Digest32,
}

pub struct NcdeCore {
    pub params: NcdeParams,
    pub state: NcdeState,
    pub commit: Digest32,
}

impl NcdeCore {
    pub fn new(params: NcdeParams) -> Self {
        let state = NcdeState::new(&params);
        let commit = commit_core(params.commit, state.commit);
        Self {
            params,
            state,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &NcdeInputs) -> NcdeOutputs {
        let control = build_control_vector(inp, &self.params);
        let mut next_x = self.state.x;
        let tcf_centered = centered_value(inp.tcf_energy_smooth, GAIN_MAX);

        for (idx, next_value) in next_x.iter_mut().enumerate().take(DIM) {
            let f = compute_flow(idx, &self.state.x, &control.u, tcf_centered, &self.params);
            let dt = i64::from(self.params.dt);
            let leak = i64::from(self.params.leak);
            let current = i64::from(self.state.x[idx]);
            let delta = (dt * i64::from(f)) / i64::from(GAIN_SCALE);
            let leak_term = (leak * current) / i64::from(GAIN_SCALE);
            let updated = current + delta - leak_term;
            *next_value = clamp_i32(updated, self.params.max_state);
        }

        let next_energy = update_energy(&self.state, &control, inp, &self.params);
        let state_commit = commit_state(&next_x, next_energy, self.params.commit);
        let state_digest = commit_state_digest(&next_x);
        self.state = NcdeState {
            x: next_x,
            energy: next_energy,
            commit: state_commit,
        };
        self.commit = commit_core(self.params.commit, self.state.commit);

        let commit = commit_outputs(
            inp.cycle_id,
            state_commit,
            next_energy,
            state_digest,
            inp.commit,
            self.params.commit,
        );

        NcdeOutputs {
            cycle_id: inp.cycle_id,
            state_commit,
            energy: next_energy,
            state_digest,
            commit,
        }
    }
}

impl Default for NcdeCore {
    fn default() -> Self {
        Self::new(NcdeParams::default())
    }
}

struct ControlSignal {
    u: [i32; DIM],
    feature_norm: i32,
    novelty_norm: i32,
    threat_norm: i32,
}

fn commit_params(
    dt: u16,
    gain_phase: u16,
    gain_spike: u16,
    gain_influence: u16,
    leak: u16,
    max_state: i32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PARAMS_DOMAIN);
    hasher.update(&dt.to_be_bytes());
    hasher.update(&gain_phase.to_be_bytes());
    hasher.update(&gain_spike.to_be_bytes());
    hasher.update(&gain_influence.to_be_bytes());
    hasher.update(&leak.to_be_bytes());
    hasher.update(&max_state.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(x: &[i32; DIM], energy: u16, params_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STATE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    hasher.update(&energy.to_be_bytes());
    for value in x.iter() {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    phase_frame_commit: Digest32,
    phase_u16: u16,
    tcf_energy_smooth: u16,
    spike_accepted_root: Digest32,
    spike_counts: &[(SpikeKind, u16)],
    influence_pulses_root: Digest32,
    coherence_plv: u16,
    risk: u16,
    drift: u16,
    surprise: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_frame_commit.as_bytes());
    hasher.update(&phase_u16.to_be_bytes());
    hasher.update(&tcf_energy_smooth.to_be_bytes());
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
    hasher.update(influence_pulses_root.as_bytes());
    hasher.update(&coherence_plv.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state_digest(x: &[i32; DIM]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DIGEST_DOMAIN);
    for value in x.iter() {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    state_commit: Digest32,
    energy: u16,
    state_digest: Digest32,
    input_commit: Digest32,
    params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OUTPUT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(state_commit.as_bytes());
    hasher.update(&energy.to_be_bytes());
    hasher.update(state_digest.as_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, state_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CORE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn build_control_vector(inp: &NcdeInputs, params: &NcdeParams) -> ControlSignal {
    let mut u = [0i32; DIM];
    let total_spikes: u32 = inp
        .spike_counts
        .iter()
        .map(|(_, count)| u32::from(*count))
        .sum();
    let feature = spike_count(inp, SpikeKind::Feature);
    let novelty = spike_count(inp, SpikeKind::Novelty);
    let threat = spike_count(inp, SpikeKind::Threat);
    let feature_norm = normalize_spike(feature, total_spikes);
    let novelty_norm = normalize_spike(novelty, total_spikes);
    let threat_norm = normalize_spike(threat, total_spikes);

    let phase_signal = scale_gain(phase_signal(inp.phase_u16), params.gain_phase);
    let coherence = scale_gain(
        centered_value(inp.coherence_plv, GAIN_MAX),
        params.gain_phase,
    );
    let surprise = scale_gain(centered_value(inp.surprise, GAIN_MAX), params.gain_phase);

    u[0] = phase_signal;
    u[1] = scale_gain(feature_norm, params.gain_spike);
    u[2] = scale_gain(novelty_norm, params.gain_spike);
    u[3] = -scale_gain(threat_norm, params.gain_spike);
    u[4] = coherence;
    u[5] = surprise;

    for (idx, value) in u.iter_mut().enumerate().take(DIM).skip(6) {
        let hashed = influence_hash(inp.influence_pulses_root, idx as u16);
        *value = scale_gain(hashed, params.gain_influence);
    }

    ControlSignal {
        u,
        feature_norm,
        novelty_norm,
        threat_norm,
    }
}

fn compute_flow(
    idx: usize,
    x: &[i32; DIM],
    u: &[i32; DIM],
    tcf_centered: i32,
    params: &NcdeParams,
) -> i32 {
    let a_coeff = -512;
    let b_coeff = 768;
    let c_coeff = 256;
    let x_i = x[idx];
    let u_i = u[idx];
    let cross =
        (i64::from(u[(idx + 1) % DIM]) * i64::from(x[(idx + 2) % DIM])) / i64::from(CONTROL_SCALE);
    let linear = (i64::from(a_coeff) * i64::from(x_i)) / i64::from(CONTROL_SCALE);
    let input = (i64::from(b_coeff) * i64::from(u_i)) / i64::from(CONTROL_SCALE);
    let cross_term = (i64::from(c_coeff) * cross) / i64::from(CONTROL_SCALE);
    let tcf_term = i64::from(tcf_centered) / 4;
    let gain = i64::from(params.gain_phase.max(1));
    let combined = linear + input + cross_term + (tcf_term * gain / i64::from(GAIN_SCALE));
    clamp_i32(combined, params.max_state)
}

fn update_energy(
    state: &NcdeState,
    control: &ControlSignal,
    inp: &NcdeInputs,
    params: &NcdeParams,
) -> u16 {
    let control_mag: i32 = control.u.iter().map(|value| value.abs()).sum::<i32>() / DIM as i32;
    let control_drive = (control_mag * ENERGY_MAX / CONTROL_SCALE).min(ENERGY_MAX);
    let feature_drive = (control.feature_norm * ENERGY_MAX / CONTROL_SCALE) / 2;
    let novelty_drive = (control.novelty_norm * ENERGY_MAX / CONTROL_SCALE) / 4;
    let threat_penalty = (control.threat_norm * ENERGY_MAX / CONTROL_SCALE) / 2;
    let tcf_drive = i32::from(inp.tcf_energy_smooth) / 4;
    let risk_penalty = i32::from(inp.risk) / 3;
    let leak_penalty = (i32::from(params.leak) * i32::from(state.energy)) / GAIN_SCALE.max(1);

    let mut energy =
        i32::from(state.energy) + control_drive / 3 + feature_drive + novelty_drive + tcf_drive
            - threat_penalty
            - risk_penalty
            - leak_penalty;

    energy = energy.clamp(0, ENERGY_MAX);
    energy as u16
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
    let denom = total.max(u32::from(SPIKE_NORM_MAX));
    let value = (i32::from(count) * CONTROL_SCALE) / denom as i32;
    value.clamp(0, CONTROL_SCALE)
}

fn scale_gain(value: i32, gain: u16) -> i32 {
    (i64::from(value) * i64::from(gain) / i64::from(GAIN_SCALE)) as i32
}

fn phase_signal(phase: u16) -> i32 {
    let phase = i32::from(phase);
    let half = PHASE_MAX / 2;
    let value = if phase < half {
        phase
    } else {
        PHASE_MAX - phase
    };
    let signed = value * 2 - half;
    (signed * CONTROL_SCALE) / half
}

fn centered_value(value: u16, max: u16) -> i32 {
    if max == 0 {
        return 0;
    }
    let max_i32 = i32::from(max);
    let doubled = i32::from(value) * 2 - max_i32;
    (doubled * CONTROL_SCALE) / max_i32
}

fn influence_hash(root: Digest32, idx: u16) -> i32 {
    let mut hasher = Hasher::new();
    hasher.update(CONTROL_DOMAIN);
    hasher.update(root.as_bytes());
    hasher.update(&idx.to_be_bytes());
    let bytes = hasher.finalize();
    let raw = i16::from_be_bytes([bytes.as_bytes()[0], bytes.as_bytes()[1]]);
    (i32::from(raw) * CONTROL_SCALE) / i32::from(i16::MAX)
}

fn clamp_i32(value: i64, max_state: i32) -> i32 {
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
            16_384,
            5000,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Feature, 2), (SpikeKind::Threat, 1)],
            Digest32::new([3u8; 32]),
            5000,
            2000,
            1200,
            4000,
        )
    }

    #[test]
    fn ncde_is_deterministic_for_same_inputs() {
        let params = NcdeParams::new(400, 3000, 3500, 2000, 500, 20_000);
        let mut core_a = NcdeCore::new(params);
        let mut core_b = NcdeCore::new(params);
        let inputs = base_inputs();

        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);

        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(out_a.state_commit, out_b.state_commit);
    }

    #[test]
    fn state_decays_with_leak_and_zero_control() {
        let params = NcdeParams::new(400, 0, 0, 0, 2000, 50_000);
        let mut core = NcdeCore::new(params);
        core.state.x = [10_000; DIM];
        core.state.commit = commit_state(&core.state.x, core.state.energy, core.params.commit);
        let inputs = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            16_384,
            5000,
            Digest32::new([2u8; 32]),
            vec![],
            Digest32::new([3u8; 32]),
            5000,
            0,
            0,
            5000,
        );

        let before = core.state.x[0].abs();
        let out = core.tick(&inputs);
        let after = core.state.x[0].abs();

        assert!(after < before, "state should decay with leak");
        assert_eq!(out.energy, core.state.energy);
    }

    #[test]
    fn feature_spikes_increase_energy() {
        let params = NcdeParams::new(400, 3000, 5000, 2000, 500, 50_000);
        let inputs_low = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            16_384,
            5000,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Feature, 1)],
            Digest32::new([3u8; 32]),
            5000,
            1000,
            1000,
            5000,
        );
        let inputs_high = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            16_384,
            5000,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Feature, 6)],
            Digest32::new([3u8; 32]),
            5000,
            1000,
            1000,
            5000,
        );
        let mut core_low = NcdeCore::new(params);
        let mut core_high = NcdeCore::new(params);

        let out_low = core_low.tick(&inputs_low);
        let out_high = core_high.tick(&inputs_high);

        assert!(out_high.energy > out_low.energy);
        assert_ne!(out_high.state_digest, out_low.state_digest);
    }

    #[test]
    fn threat_spikes_reduce_energy() {
        let params = NcdeParams::new(400, 3000, 5000, 2000, 500, 50_000);
        let inputs_feature = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            16_384,
            5000,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Feature, 4)],
            Digest32::new([3u8; 32]),
            5000,
            1000,
            1000,
            5000,
        );
        let inputs_threat = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            16_384,
            5000,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Threat, 4)],
            Digest32::new([3u8; 32]),
            5000,
            1000,
            1000,
            5000,
        );
        let mut core_feature = NcdeCore::new(params);
        let mut core_threat = NcdeCore::new(params);

        let out_feature = core_feature.tick(&inputs_feature);
        let out_threat = core_threat.tick(&inputs_threat);

        assert!(out_threat.energy < out_feature.energy);
    }

    #[test]
    fn risk_inhibits_energy() {
        let params = NcdeParams::new(400, 3000, 5000, 2000, 500, 50_000);
        let inputs_low = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            16_384,
            5000,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Feature, 3)],
            Digest32::new([3u8; 32]),
            5000,
            1000,
            1000,
            5000,
        );
        let inputs_high = NcdeInputs::new(
            1,
            Digest32::new([1u8; 32]),
            16_384,
            5000,
            Digest32::new([2u8; 32]),
            vec![(SpikeKind::Feature, 3)],
            Digest32::new([3u8; 32]),
            5000,
            8000,
            1000,
            5000,
        );
        let mut core_low = NcdeCore::new(params);
        let mut core_high = NcdeCore::new(params);

        let out_low = core_low.tick(&inputs_low);
        let out_high = core_high.tick(&inputs_high);

        assert!(out_high.energy < out_low.energy);
    }
}
