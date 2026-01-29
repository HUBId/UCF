#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_coupling::SignalId;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const MAX_DIM: usize = 64;
const Q15_SCALE: i64 = 1 << 15;
const MAX_U16: u16 = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsmParams {
    pub dim: usize,
    pub a_q15: i16,
    pub b_q15: i16,
    pub k_phase: u16,
    pub k_spike: u16,
    pub k_coupling: u16,
    pub k_sle: u16,
    pub k_ncde: u16,
    pub novelty_hi: u16,
    pub novelty_lo: u16,
    pub salience_hi: u16,
    pub clamp: i32,
    pub commit: Digest32,
}

impl SsmParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        a_q15: i16,
        b_q15: i16,
        k_phase: u16,
        k_spike: u16,
        k_coupling: u16,
        k_sle: u16,
        k_ncde: u16,
        novelty_hi: u16,
        novelty_lo: u16,
        salience_hi: u16,
        clamp: i32,
    ) -> Self {
        let dim = dim.clamp(1, MAX_DIM);
        let k_phase = k_phase.min(MAX_U16);
        let k_spike = k_spike.min(MAX_U16);
        let k_coupling = k_coupling.min(MAX_U16);
        let k_sle = k_sle.min(MAX_U16);
        let k_ncde = k_ncde.min(MAX_U16);
        let novelty_hi = novelty_hi.min(MAX_U16).max(novelty_lo);
        let novelty_lo = novelty_lo.min(novelty_hi);
        let salience_hi = salience_hi.clamp(1, MAX_U16);
        let clamp = clamp.max(1);
        let commit = commit_params(
            dim,
            a_q15,
            b_q15,
            k_phase,
            k_spike,
            k_coupling,
            k_sle,
            k_ncde,
            novelty_hi,
            novelty_lo,
            salience_hi,
            clamp,
        );
        Self {
            dim,
            a_q15,
            b_q15,
            k_phase,
            k_spike,
            k_coupling,
            k_sle,
            k_ncde,
            novelty_hi,
            novelty_lo,
            salience_hi,
            clamp,
            commit,
        }
    }
}

impl Default for SsmParams {
    fn default() -> Self {
        Self::new(
            32, -512, 768, 1200, 1600, 800, 700, 1400, 9000, 0, 10_000, 40_000,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmState {
    pub x: Vec<i32>,
    pub last_state_digest: Digest32,
    pub commit: Digest32,
}

impl SsmState {
    pub fn new(params: &SsmParams) -> Self {
        let x = vec![0; params.dim];
        let last_state_digest = digest_state(&x, 0, Digest32::new([0u8; 32]));
        let commit = Digest32::new([0u8; 32]);
        Self {
            x,
            last_state_digest,
            commit,
        }
    }

    pub fn reset_if_dim_mismatch(&mut self, params: &SsmParams) {
        if self.x.len() != params.dim {
            self.x = vec![0; params.dim];
            self.last_state_digest = digest_state(&self.x, 0, Digest32::new([0u8; 32]));
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmInputs {
    pub cycle_id: u64,
    pub phase_bus_commit: Digest32,
    pub gamma_bucket: u8,
    pub percept_commit: Digest32,
    pub percept_energy: u16,
    pub spike_accepted_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub coupling_influences_root: Digest32,
    pub coupling_influences: Vec<(SignalId, i16)>,
    pub tcf_attention_cap: u16,
    pub tcf_learning_cap: u16,
    pub sle_ssm_bias: i16,
    pub ncde_energy: u16,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub commit: Digest32,
}

impl SsmInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_bus_commit: Digest32,
        gamma_bucket: u8,
        percept_commit: Digest32,
        percept_energy: u16,
        spike_accepted_root: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        coupling_influences_root: Digest32,
        coupling_influences: Vec<(SignalId, i16)>,
        tcf_attention_cap: u16,
        tcf_learning_cap: u16,
        sle_ssm_bias: i16,
        ncde_energy: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
    ) -> Self {
        let commit = commit_inputs(
            cycle_id,
            phase_bus_commit,
            gamma_bucket,
            percept_commit,
            percept_energy,
            spike_accepted_root,
            &spike_counts,
            coupling_influences_root,
            &coupling_influences,
            tcf_attention_cap,
            tcf_learning_cap,
            sle_ssm_bias,
            ncde_energy,
            risk,
            drift,
            surprise,
        );
        Self {
            cycle_id,
            phase_bus_commit,
            gamma_bucket,
            percept_commit,
            percept_energy,
            spike_accepted_root,
            spike_counts,
            coupling_influences_root,
            coupling_influences,
            tcf_attention_cap,
            tcf_learning_cap,
            sle_ssm_bias,
            ncde_energy,
            risk,
            drift,
            surprise,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmOutputs {
    pub cycle_id: u64,
    pub ssm_state_commit: Digest32,
    pub ssm_state_digest: Digest32,
    pub ssm_salience: u16,
    pub ssm_novelty: u16,
    pub ssm_attention_gain: u16,
    pub commit: Digest32,
}

pub struct SsmCore {
    pub params: SsmParams,
    pub state: SsmState,
    pub last_salience: u16,
    pub last_novelty: u16,
    pub commit: Digest32,
}

impl SsmCore {
    pub fn new(params: SsmParams) -> Self {
        let state = SsmState::new(&params);
        let commit = commit_core(params.commit, state.commit, state.last_state_digest);
        Self {
            params,
            state,
            last_salience: 0,
            last_novelty: 0,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &SsmInputs) -> SsmOutputs {
        self.state.reset_if_dim_mismatch(&self.params);
        let dim = self.params.dim;
        let u_q15 = build_input_drive(inp, &self.params);
        let sign_pattern = build_sign_pattern(
            dim,
            inp.percept_commit,
            inp.spike_accepted_root,
            inp.phase_bus_commit,
        );
        let effective_b = effective_b_q15(inp, &self.params);

        for (idx, value) in self.state.x.iter_mut().enumerate().take(dim) {
            let u_i = u_q15.saturating_mul(sign_pattern[idx]);
            let a_term = i64::from(self.params.a_q15) * i64::from(*value);
            let b_term = i64::from(effective_b) * i64::from(u_i);
            let mut updated = (a_term + b_term) >> 15;
            updated = updated.clamp(-i64::from(self.params.clamp), i64::from(self.params.clamp));
            *value = updated as i32;
        }

        let ssm_state_digest = digest_state(&self.state.x, inp.cycle_id, inp.phase_bus_commit);
        let ssm_state_commit = commit_state(ssm_state_digest, self.params.commit, inp.commit);
        let ssm_novelty =
            novelty_score(ssm_state_digest, self.state.last_state_digest, &self.params);
        let ssm_salience = salience_score(inp, &self.params);
        let ssm_attention_gain = attention_gain_score(inp, ssm_salience, ssm_novelty);

        self.state.last_state_digest = ssm_state_digest;
        self.state.commit = ssm_state_commit;
        self.last_salience = ssm_salience;
        self.last_novelty = ssm_novelty;
        self.commit = commit_core(self.params.commit, ssm_state_commit, ssm_state_digest);

        let commit = commit_outputs(
            inp.cycle_id,
            ssm_state_commit,
            ssm_salience,
            ssm_novelty,
            ssm_attention_gain,
        );

        SsmOutputs {
            cycle_id: inp.cycle_id,
            ssm_state_commit,
            ssm_state_digest,
            ssm_salience,
            ssm_novelty,
            ssm_attention_gain,
            commit,
        }
    }
}

fn build_input_drive(inp: &SsmInputs, params: &SsmParams) -> i32 {
    if is_zero_input(inp) {
        return 0;
    }

    let percept_q15 = q15_from_u16(inp.percept_energy);
    let spike_energy = spike_scalar(&inp.spike_counts);
    let spike_q15 = q15_from_u16(spike_energy);
    let coupling_sum = coupling_scalar(&inp.coupling_influences);
    let coupling_q15 = q15_from_i16(coupling_sum);
    let sle_q15 = q15_from_i16(inp.sle_ssm_bias);
    let ncde_q15 = q15_from_u16(inp.ncde_energy);
    let phase_q15 = q15_from_phase(inp.gamma_bucket, params.k_phase);

    let mut u_q15 = percept_q15;
    u_q15 = u_q15.saturating_add(scale_q15(spike_q15, params.k_spike));
    u_q15 = u_q15.saturating_add(scale_q15(coupling_q15, params.k_coupling));
    u_q15 = u_q15.saturating_add(scale_q15(sle_q15, params.k_sle));
    u_q15 = u_q15.saturating_add(scale_q15(ncde_q15, params.k_ncde));
    u_q15 = u_q15.saturating_add(phase_q15);

    u_q15.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}

fn build_sign_pattern(
    dim: usize,
    percept_commit: Digest32,
    spike_accepted_root: Digest32,
    phase_bus_commit: Digest32,
) -> Vec<i32> {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.signs.v1");
    hasher.update(percept_commit.as_bytes());
    hasher.update(spike_accepted_root.as_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    let mut reader = hasher.finalize_xof();
    let mut bytes = vec![0u8; dim.max(1)];
    reader.fill(&mut bytes);
    bytes
        .iter()
        .take(dim)
        .enumerate()
        .map(|(idx, byte)| {
            let bit = (byte >> (idx % 8)) & 1;
            if bit == 1 {
                1
            } else {
                -1
            }
        })
        .collect()
}

fn effective_b_q15(inp: &SsmInputs, params: &SsmParams) -> i16 {
    let base = i64::from(params.b_q15);
    let cap = u16::min(inp.tcf_attention_cap, inp.tcf_learning_cap).min(MAX_U16);
    let cap_scale = i64::from(cap);
    let risk_scale = 10_000i64
        .saturating_sub(i64::from(inp.risk) / 2)
        .clamp(0, 10_000);
    let surprise_scale = 10_000i64
        .saturating_add(i64::from(inp.surprise) / 6)
        .clamp(10_000, 12_000);

    let mut scaled = base.saturating_mul(cap_scale);
    scaled = scaled.saturating_mul(risk_scale) / 10_000;
    scaled = scaled.saturating_mul(surprise_scale) / 10_000;
    scaled /= 10_000;

    scaled.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

fn spike_scalar(spike_counts: &[(SpikeKind, u16)]) -> u16 {
    let mut feature = 0u32;
    let mut novelty = 0u32;
    let mut threat = 0u32;
    let mut causal = 0u32;
    let mut thought_only = 0u32;
    for (kind, count) in spike_counts {
        match kind {
            SpikeKind::Feature => feature = feature.saturating_add(u32::from(*count)),
            SpikeKind::Novelty => novelty = novelty.saturating_add(u32::from(*count)),
            SpikeKind::Threat => threat = threat.saturating_add(u32::from(*count)),
            SpikeKind::CausalLink => causal = causal.saturating_add(u32::from(*count)),
            SpikeKind::ThoughtOnly => thought_only = thought_only.saturating_add(u32::from(*count)),
            SpikeKind::ConsistencyAlert
            | SpikeKind::MemoryCue
            | SpikeKind::ReplayCue
            | SpikeKind::OutputIntent
            | SpikeKind::Unknown(_) => {}
        }
    }
    let combined = feature
        .saturating_add(novelty.saturating_mul(2))
        .saturating_add(threat.saturating_mul(3))
        .saturating_add(causal)
        .saturating_add(thought_only);
    combined.min(u32::from(MAX_U16)) as u16
}

fn coupling_scalar(coupling_influences: &[(SignalId, i16)]) -> i16 {
    let mut sum = 0i32;
    for (_, value) in coupling_influences {
        sum = sum.saturating_add(i32::from(*value));
    }
    sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn q15_from_u16(value: u16) -> i32 {
    let scaled = i64::from(value.min(MAX_U16)) * Q15_SCALE / i64::from(MAX_U16);
    scaled as i32
}

fn q15_from_i16(value: i16) -> i32 {
    let scaled = i64::from(value) * Q15_SCALE / i64::from(MAX_U16);
    scaled.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i32
}

fn q15_from_phase(gamma_bucket: u8, k_phase: u16) -> i32 {
    if k_phase == 0 {
        return 0;
    }
    let centered = i32::from(gamma_bucket) - 128;
    let raw = (i64::from(centered) * Q15_SCALE) / 128;
    let scaled = raw.saturating_mul(i64::from(k_phase)) / 10_000;
    scaled.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i32
}

fn scale_q15(value: i32, gain: u16) -> i32 {
    let scaled = i64::from(value) * i64::from(gain.min(MAX_U16)) / 10_000;
    scaled.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i32
}

fn salience_score(inp: &SsmInputs, params: &SsmParams) -> u16 {
    let threat = inp
        .spike_counts
        .iter()
        .filter_map(|(kind, count)| (*kind == SpikeKind::Threat).then_some(*count))
        .sum::<u16>();
    let mut score = u32::from(inp.percept_energy)
        .saturating_add(u32::from(threat))
        .saturating_add(u32::from(inp.ncde_energy));
    score = score.min(u32::from(params.salience_hi));
    score.min(u32::from(MAX_U16)) as u16
}

fn novelty_score(current: Digest32, previous: Digest32, params: &SsmParams) -> u16 {
    let current_trunc = u16::from_be_bytes([current.as_bytes()[0], current.as_bytes()[1]]);
    let previous_trunc = u16::from_be_bytes([previous.as_bytes()[0], previous.as_bytes()[1]]);
    let xor = current_trunc ^ previous_trunc;
    let bits = xor.count_ones();
    let scaled = (bits.saturating_mul(10_000) / 16) as u16;
    let mut novelty = if scaled == 0 {
        0
    } else {
        scaled.max(params.novelty_lo)
    };
    novelty = novelty.min(params.novelty_hi);
    novelty
}

fn attention_gain_score(inp: &SsmInputs, salience: u16, novelty: u16) -> u16 {
    let base = inp.tcf_attention_cap.min(MAX_U16);
    let boost = u32::from(salience) / 2 + u32::from(novelty) / 4;
    let penalty = u32::from(inp.risk) / 2 + u32::from(inp.drift) / 3;
    let mut gain = u32::from(base).saturating_add(boost);
    gain = gain.saturating_sub(penalty);
    gain.min(u32::from(base)).min(u32::from(MAX_U16)) as u16
}

fn is_zero_input(inp: &SsmInputs) -> bool {
    inp.percept_commit.as_bytes().iter().all(|byte| *byte == 0)
        && inp.percept_energy == 0
        && inp.spike_counts.is_empty()
        && inp.coupling_influences.is_empty()
        && inp.sle_ssm_bias == 0
        && inp.ncde_energy == 0
        && inp.risk == 0
        && inp.drift == 0
        && inp.surprise == 0
}

#[allow(clippy::too_many_arguments)]
fn commit_params(
    dim: usize,
    a_q15: i16,
    b_q15: i16,
    k_phase: u16,
    k_spike: u16,
    k_coupling: u16,
    k_sle: u16,
    k_ncde: u16,
    novelty_hi: u16,
    novelty_lo: u16,
    salience_hi: u16,
    clamp: i32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.params.v2");
    hasher.update(&u64::try_from(dim).unwrap_or(0).to_be_bytes());
    hasher.update(&a_q15.to_be_bytes());
    hasher.update(&b_q15.to_be_bytes());
    hasher.update(&k_phase.to_be_bytes());
    hasher.update(&k_spike.to_be_bytes());
    hasher.update(&k_coupling.to_be_bytes());
    hasher.update(&k_sle.to_be_bytes());
    hasher.update(&k_ncde.to_be_bytes());
    hasher.update(&novelty_hi.to_be_bytes());
    hasher.update(&novelty_lo.to_be_bytes());
    hasher.update(&salience_hi.to_be_bytes());
    hasher.update(&clamp.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_state(state: &[i32], cycle_id: u64, phase_bus_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.state.digest.v2");
    let mut chunk_count = 0u16;
    for chunk in state.chunks(8) {
        let mut chunk_hasher = Hasher::new();
        chunk_hasher.update(&u16::try_from(chunk.len()).unwrap_or(0).to_be_bytes());
        for value in chunk {
            let trunc = (*value as i16).to_be_bytes();
            chunk_hasher.update(&trunc);
        }
        let chunk_digest = chunk_hasher.finalize();
        hasher.update(chunk_digest.as_bytes());
        chunk_count = chunk_count.saturating_add(1);
    }
    hasher.update(&chunk_count.to_be_bytes());
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(
    state_digest: Digest32,
    params_commit: Digest32,
    input_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.state.commit.v2");
    hasher.update(state_digest.as_bytes());
    hasher.update(params_commit.as_bytes());
    hasher.update(input_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    phase_bus_commit: Digest32,
    gamma_bucket: u8,
    percept_commit: Digest32,
    percept_energy: u16,
    spike_accepted_root: Digest32,
    spike_counts: &[(SpikeKind, u16)],
    coupling_influences_root: Digest32,
    coupling_influences: &[(SignalId, i16)],
    tcf_attention_cap: u16,
    tcf_learning_cap: u16,
    sle_ssm_bias: i16,
    ncde_energy: u16,
    risk: u16,
    drift: u16,
    surprise: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.input.v2");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(&[gamma_bucket]);
    hasher.update(percept_commit.as_bytes());
    hasher.update(&percept_energy.to_be_bytes());
    hasher.update(spike_accepted_root.as_bytes());
    hasher.update(&(spike_counts.len() as u16).to_be_bytes());
    for (kind, count) in spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(coupling_influences_root.as_bytes());
    hasher.update(&(coupling_influences.len() as u16).to_be_bytes());
    for (id, value) in coupling_influences {
        hasher.update(&id.as_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    hasher.update(&tcf_attention_cap.to_be_bytes());
    hasher.update(&tcf_learning_cap.to_be_bytes());
    hasher.update(&sle_ssm_bias.to_be_bytes());
    hasher.update(&ncde_energy.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    ssm_state_commit: Digest32,
    ssm_salience: u16,
    ssm_novelty: u16,
    ssm_attention_gain: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.output.v2");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(ssm_state_commit.as_bytes());
    hasher.update(&ssm_salience.to_be_bytes());
    hasher.update(&ssm_novelty.to_be_bytes());
    hasher.update(&ssm_attention_gain.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, state_commit: Digest32, digest: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.core.v2");
    hasher.update(params_commit.as_bytes());
    hasher.update(state_commit.as_bytes());
    hasher.update(digest.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inputs(percept_commit: Digest32) -> SsmInputs {
        SsmInputs::new(
            7,
            Digest32::new([1u8; 32]),
            128,
            percept_commit,
            1200,
            Digest32::new([4u8; 32]),
            vec![
                (SpikeKind::Feature, 4),
                (SpikeKind::Novelty, 3),
                (SpikeKind::Threat, 2),
            ],
            Digest32::new([9u8; 32]),
            vec![(SignalId::SsmSalience, 600)],
            4000,
            3000,
            200,
            3200,
            800,
            600,
            1200,
        )
    }

    #[test]
    fn deterministic_output_commit() {
        let params = SsmParams::default();
        let mut core_a = SsmCore::new(params);
        let mut core_b = SsmCore::new(params);
        let input = sample_inputs(Digest32::new([9u8; 32]));

        let out_a = core_a.tick(&input);
        let out_b = core_b.tick(&input);

        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(out_a.ssm_state_digest, out_b.ssm_state_digest);
    }

    #[test]
    fn decay_drives_state_toward_zero() {
        let params = SsmParams::new(8, 30000, 0, 0, 0, 0, 0, 0, 9000, 0, 10_000, 20_000);
        let mut core = SsmCore::new(params);
        core.state.x = vec![10_000; params.dim];
        let input = SsmInputs::new(
            1,
            Digest32::new([0u8; 32]),
            0,
            Digest32::new([0u8; 32]),
            0,
            Digest32::new([0u8; 32]),
            Vec::new(),
            Digest32::new([0u8; 32]),
            Vec::new(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let before: i64 = core
            .state
            .x
            .iter()
            .map(|value| i64::from(value.abs()))
            .sum();
        core.tick(&input);
        let after: i64 = core
            .state
            .x
            .iter()
            .map(|value| i64::from(value.abs()))
            .sum();
        assert!(after < before);
    }

    #[test]
    fn surprise_increases_update_magnitude() {
        let params = SsmParams::default();
        let mut core = SsmCore::new(params);
        let low_surprise = SsmInputs::new(
            1,
            Digest32::new([1u8; 32]),
            120,
            Digest32::new([2u8; 32]),
            2000,
            Digest32::new([3u8; 32]),
            vec![(SpikeKind::Feature, 2)],
            Digest32::new([4u8; 32]),
            vec![(SignalId::SsmSalience, 300)],
            5000,
            5000,
            100,
            1000,
            200,
            100,
            200,
        );
        let high_surprise = SsmInputs::new(
            2,
            Digest32::new([1u8; 32]),
            120,
            Digest32::new([2u8; 32]),
            2000,
            Digest32::new([3u8; 32]),
            vec![(SpikeKind::Feature, 2)],
            Digest32::new([4u8; 32]),
            vec![(SignalId::SsmSalience, 300)],
            5000,
            5000,
            100,
            1000,
            200,
            100,
            8000,
        );
        let before = core.state.x.clone();
        let out_low = core.tick(&low_surprise);
        let after_low = core.state.x.clone();
        let _ = out_low;
        let out_high = core.tick(&high_surprise);
        let after_high = core.state.x.clone();
        let magnitude_low: i64 = before
            .iter()
            .zip(after_low.iter())
            .map(|(a, b)| i64::from((b - a).abs()))
            .sum();
        let magnitude_high: i64 = before
            .iter()
            .zip(after_high.iter())
            .map(|(a, b)| i64::from((b - a).abs()))
            .sum();
        assert!(magnitude_high >= magnitude_low);
        assert!(out_high.ssm_attention_gain <= 10_000);
    }

    #[test]
    fn novelty_tracks_digest_distance() {
        let params = SsmParams::default();
        let mut core = SsmCore::new(params);
        let first = sample_inputs(Digest32::new([2u8; 32]));
        let out_a = core.tick(&first);
        let out_b = core.tick(&first);

        assert!(out_b.ssm_novelty <= out_a.ssm_novelty);
    }

    #[test]
    fn attention_gain_respects_tcf_cap() {
        let params = SsmParams::default();
        let mut core = SsmCore::new(params);
        let input = SsmInputs::new(
            1,
            Digest32::new([1u8; 32]),
            120,
            Digest32::new([2u8; 32]),
            8000,
            Digest32::new([3u8; 32]),
            vec![(SpikeKind::Threat, 6)],
            Digest32::new([4u8; 32]),
            vec![(SignalId::SsmSalience, 500)],
            3000,
            8000,
            0,
            4000,
            200,
            200,
            200,
        );
        let output = core.tick(&input);
        assert!(output.ssm_attention_gain <= 3000);
    }
}
