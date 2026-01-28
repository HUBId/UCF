#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const SCALE: i64 = 10_000;
const MAX_DIM: usize = 64;
const NOVELTY_BITS: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsmParams {
    pub dim: usize,
    pub dt: u16,
    pub k_att: u16,
    pub k_novelty: u16,
    pub leak: u16,
    pub max_state: i32,
    pub commit: Digest32,
}

impl SsmParams {
    pub fn new(dim: usize, dt: u16, k_att: u16, k_novelty: u16, leak: u16, max_state: i32) -> Self {
        let dim = dim.clamp(1, MAX_DIM);
        let dt = dt.min(10_000);
        let k_att = k_att.min(10_000);
        let k_novelty = k_novelty.min(10_000);
        let leak = leak.min(10_000);
        let max_state = max_state.max(1);
        let commit = commit_params(dim, dt, k_att, k_novelty, leak, max_state);
        Self {
            dim,
            dt,
            k_att,
            k_novelty,
            leak,
            max_state,
            commit,
        }
    }
}

impl Default for SsmParams {
    fn default() -> Self {
        Self::new(32, 250, 800, 5200, 800, 40_000)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmState {
    pub x: Vec<i32>,
    pub commit: Digest32,
}

impl SsmState {
    pub fn new(params: &SsmParams) -> Self {
        let x = vec![0; params.dim];
        let commit = commit_state(&x, params.commit);
        Self { x, commit }
    }

    pub fn reset_if_dim_mismatch(&mut self, params: &SsmParams) {
        if self.x.len() != params.dim {
            self.x = vec![0; params.dim];
        }
        self.commit = commit_state(&self.x, params.commit);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub percept_commit: Digest32,
    pub percept_energy: u16,
    pub spike_accepted_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub prev_attention_gain: u16,
    pub ncde_state_digest: Digest32,
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
        phase_commit: Digest32,
        percept_commit: Digest32,
        percept_energy: u16,
        spike_accepted_root: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        prev_attention_gain: u16,
        ncde_state_digest: Digest32,
        ncde_energy: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
    ) -> Self {
        let commit = commit_inputs(
            cycle_id,
            phase_commit,
            percept_commit,
            percept_energy,
            spike_accepted_root,
            &spike_counts,
            prev_attention_gain,
            ncde_state_digest,
            ncde_energy,
            risk,
            drift,
            surprise,
        );
        Self {
            cycle_id,
            phase_commit,
            percept_commit,
            percept_energy,
            spike_accepted_root,
            spike_counts,
            prev_attention_gain,
            ncde_state_digest,
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
    pub salience: u16,
    pub novelty: u16,
    pub attention_gain: u16,
    pub commit: Digest32,
}

pub struct SsmCore {
    pub params: SsmParams,
    pub state: SsmState,
    pub last_state_digest: Digest32,
    pub commit: Digest32,
}

impl SsmCore {
    pub fn new(params: SsmParams) -> Self {
        let state = SsmState::new(&params);
        let last_state_digest = digest_state(&state.x);
        let commit = commit_core(params.commit, state.commit, last_state_digest);
        Self {
            params,
            state,
            last_state_digest,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &SsmInputs) -> SsmOutputs {
        self.state.reset_if_dim_mismatch(&self.params);
        let dim = self.params.dim;
        let control = build_control_vector(inp, dim);
        let dt = i64::from(self.params.dt);
        let leak = i64::from(self.params.leak);
        let attn_boost =
            (i64::from(self.params.k_att) * i64::from(inp.prev_attention_gain)) / SCALE;
        let max_state = i64::from(self.params.max_state);

        for (idx, value) in self.state.x.iter_mut().enumerate().take(dim) {
            let current = i64::from(*value);
            let leak_term = (leak * current) / SCALE;
            let delta = (dt * (control[idx] - leak_term)) / SCALE;
            let mut updated = current.saturating_add(delta);
            if control[idx] != 0 {
                updated = updated.saturating_add(attn_boost.saturating_mul(sign_i64(control[idx])));
            }
            updated = updated.clamp(-max_state, max_state);
            *value = updated as i32;
        }

        let ssm_state_commit = commit_state(&self.state.x, self.params.commit);
        self.state.commit = ssm_state_commit;
        let state_digest = digest_state(&self.state.x);
        let novelty = novelty_score(state_digest, self.last_state_digest, inp, &self.params);
        self.last_state_digest = state_digest;
        let salience = salience_score(&control, inp);
        let attention_gain = attention_gain_score(salience, novelty, inp);
        self.commit = commit_core(
            self.params.commit,
            self.state.commit,
            self.last_state_digest,
        );

        let commit = commit_outputs(
            inp.cycle_id,
            ssm_state_commit,
            salience,
            novelty,
            attention_gain,
            inp.commit,
            self.params.commit,
        );

        SsmOutputs {
            cycle_id: inp.cycle_id,
            ssm_state_commit,
            salience,
            novelty,
            attention_gain,
            commit,
        }
    }
}

fn build_control_vector(inp: &SsmInputs, dim: usize) -> Vec<i64> {
    if is_zero_input(inp) {
        return vec![0; dim];
    }
    let mut components = hash_signed_components(inp.percept_commit, dim);
    let (feature, novelty, threat) = spike_summary(&inp.spike_counts);
    let spike_mix = i64::from(feature)
        .saturating_add(i64::from(novelty).saturating_mul(2))
        .saturating_add(i64::from(threat).saturating_mul(3))
        .min(SCALE);
    let energy_mix = i64::from(inp.ncde_energy).saturating_add(i64::from(inp.percept_energy) / 2);
    let attn_mix = i64::from(inp.prev_attention_gain) / 3;
    let bias = spike_mix
        .saturating_add(energy_mix)
        .saturating_add(attn_mix);

    for value in &mut components {
        let signed_bias = sign_i64(*value) * bias;
        *value = value.saturating_add(signed_bias).clamp(-SCALE, SCALE);
    }
    components
}

fn hash_signed_components(commit: Digest32, dim: usize) -> Vec<i64> {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.control.v1");
    hasher.update(commit.as_bytes());
    hasher.update(&u64::try_from(dim).unwrap_or(0).to_be_bytes());
    let mut reader = hasher.finalize_xof();
    let mut bytes = vec![0u8; dim.saturating_mul(2).max(2)];
    reader.fill(&mut bytes);
    bytes
        .chunks_exact(2)
        .take(dim)
        .map(|chunk| {
            let raw = i16::from_be_bytes([chunk[0], chunk[1]]);
            let scaled = (i64::from(raw) * SCALE) / i64::from(i16::MAX);
            scaled.clamp(-SCALE, SCALE)
        })
        .collect()
}

fn spike_summary(spike_counts: &[(SpikeKind, u16)]) -> (u16, u16, u16) {
    let mut feature = 0u16;
    let mut novelty = 0u16;
    let mut threat = 0u16;
    for (kind, count) in spike_counts {
        match kind {
            SpikeKind::Feature => feature = feature.saturating_add(*count),
            SpikeKind::Novelty => novelty = novelty.saturating_add(*count),
            SpikeKind::Threat => threat = threat.saturating_add(*count),
            SpikeKind::CausalLink
            | SpikeKind::ConsistencyAlert
            | SpikeKind::ThoughtOnly
            | SpikeKind::MemoryCue
            | SpikeKind::ReplayCue
            | SpikeKind::OutputIntent
            | SpikeKind::Unknown(_) => {}
        }
    }
    (feature.min(10_000), novelty.min(10_000), threat.min(10_000))
}

fn salience_score(control: &[i64], inp: &SsmInputs) -> u16 {
    if control.is_empty() {
        return 0;
    }
    let sum: i64 = control.iter().map(|value| value.abs()).sum();
    let avg = sum / i64::try_from(control.len()).unwrap_or(1);
    let mut score = avg
        .saturating_add(i64::from(inp.ncde_energy))
        .saturating_add(i64::from(inp.percept_energy) / 2)
        .saturating_sub(i64::from(inp.risk) / 2);
    score = score.clamp(0, SCALE);
    score as u16
}

fn novelty_score(current: Digest32, last: Digest32, inp: &SsmInputs, params: &SsmParams) -> u16 {
    let mut xor = [0u8; 32];
    for (idx, byte) in xor.iter_mut().enumerate() {
        *byte = current.as_bytes()[idx] ^ last.as_bytes()[idx];
    }
    let mut bits = 0u32;
    for byte in xor {
        bits += byte.count_ones();
    }
    let base = (bits.saturating_mul(10_000) / NOVELTY_BITS) as u16;
    let (_, spike_novelty, _) = spike_summary(&inp.spike_counts);
    let surprise_boost = (u32::from(params.k_novelty) * u32::from(inp.surprise)) / 10_000;
    let spike_boost = (u32::from(params.k_novelty) * u32::from(spike_novelty)) / 10_000;
    let mut score = u32::from(base)
        .saturating_add(spike_boost)
        .saturating_add(surprise_boost / 2);
    if spike_novelty > 0 {
        score = score.saturating_add(u32::from(params.k_novelty) / 10);
    }
    score.min(10_000) as u16
}

fn attention_gain_score(salience: u16, novelty: u16, inp: &SsmInputs) -> u16 {
    let weighted = (u32::from(salience) * 2 + u32::from(novelty)) / 3;
    let penalty = u32::from(inp.risk) / 2 + u32::from(inp.drift) / 3;
    weighted.saturating_sub(penalty).min(10_000) as u16
}

fn is_zero_input(inp: &SsmInputs) -> bool {
    inp.percept_commit.as_bytes().iter().all(|byte| *byte == 0)
        && inp.spike_counts.is_empty()
        && inp.prev_attention_gain == 0
        && inp.ncde_energy == 0
        && inp.percept_energy == 0
        && inp.risk == 0
        && inp.drift == 0
        && inp.surprise == 0
}

fn sign_i64(value: i64) -> i64 {
    if value >= 0 {
        1
    } else {
        -1
    }
}

fn commit_params(
    dim: usize,
    dt: u16,
    k_att: u16,
    k_novelty: u16,
    leak: u16,
    max_state: i32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.params.v1");
    hasher.update(&u64::try_from(dim).unwrap_or(0).to_be_bytes());
    hasher.update(&dt.to_be_bytes());
    hasher.update(&k_att.to_be_bytes());
    hasher.update(&k_novelty.to_be_bytes());
    hasher.update(&leak.to_be_bytes());
    hasher.update(&max_state.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_state(state: &[i32], params_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.state.v1");
    hasher.update(params_commit.as_bytes());
    hasher.update(&u64::try_from(state.len()).unwrap_or(0).to_be_bytes());
    for value in state {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_state(state: &[i32]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.state.digest.v1");
    hasher.update(&u64::try_from(state.len()).unwrap_or(0).to_be_bytes());
    for value in state {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    phase_commit: Digest32,
    percept_commit: Digest32,
    percept_energy: u16,
    spike_accepted_root: Digest32,
    spike_counts: &[(SpikeKind, u16)],
    prev_attention_gain: u16,
    ncde_state_digest: Digest32,
    ncde_energy: u16,
    risk: u16,
    drift: u16,
    surprise: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.input.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(percept_commit.as_bytes());
    hasher.update(&percept_energy.to_be_bytes());
    hasher.update(spike_accepted_root.as_bytes());
    hasher.update(&(spike_counts.len() as u16).to_be_bytes());
    for (kind, count) in spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(&prev_attention_gain.to_be_bytes());
    hasher.update(ncde_state_digest.as_bytes());
    hasher.update(&ncde_energy.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    ssm_state_commit: Digest32,
    salience: u16,
    novelty: u16,
    attention_gain: u16,
    input_commit: Digest32,
    params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.output.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(ssm_state_commit.as_bytes());
    hasher.update(&salience.to_be_bytes());
    hasher.update(&novelty.to_be_bytes());
    hasher.update(&attention_gain.to_be_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, state_commit: Digest32, digest: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.core.v1");
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
            percept_commit,
            1200,
            Digest32::new([4u8; 32]),
            vec![
                (SpikeKind::Feature, 4),
                (SpikeKind::Novelty, 3),
                (SpikeKind::Threat, 2),
            ],
            2400,
            Digest32::new([5u8; 32]),
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
        assert_eq!(out_a.ssm_state_commit, out_b.ssm_state_commit);
    }

    #[test]
    fn leak_drives_state_toward_zero_for_zero_input() {
        let params = SsmParams::new(8, 8000, 0, 0, 9000, 20_000);
        let mut core = SsmCore::new(params);
        core.state.x = vec![10_000; params.dim];
        core.state.commit = commit_state(&core.state.x, params.commit);
        let input = SsmInputs::new(
            1,
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            0,
            Digest32::new([0u8; 32]),
            Vec::new(),
            0,
            Digest32::new([0u8; 32]),
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
    fn novelty_increases_when_percept_commit_changes() {
        let params = SsmParams::default();
        let mut baseline = SsmCore::new(params);
        let mut changed = SsmCore::new(params);
        let first = sample_inputs(Digest32::new([2u8; 32]));
        baseline.tick(&first);
        changed.tick(&first);

        let same = baseline.tick(&first);
        let altered = changed.tick(&sample_inputs(Digest32::new([3u8; 32])));

        assert!(altered.novelty >= same.novelty);
    }

    #[test]
    fn higher_inputs_raise_attention_gain() {
        let params = SsmParams::default();
        let mut core = SsmCore::new(params);
        let low = SsmInputs::new(
            1,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            200,
            Digest32::new([3u8; 32]),
            vec![(SpikeKind::Feature, 1)],
            0,
            Digest32::new([4u8; 32]),
            200,
            500,
            200,
            100,
        );
        let high = SsmInputs::new(
            2,
            Digest32::new([1u8; 32]),
            Digest32::new([9u8; 32]),
            6000,
            Digest32::new([3u8; 32]),
            vec![
                (SpikeKind::Feature, 10),
                (SpikeKind::Novelty, 8),
                (SpikeKind::Threat, 6),
            ],
            7000,
            Digest32::new([4u8; 32]),
            8000,
            300,
            200,
            200,
        );

        let low_out = core.tick(&low);
        let high_out = core.tick(&high);

        assert!(high_out.attention_gain >= low_out.attention_gain);
        assert!(high_out.attention_gain <= 10_000);
    }
}
