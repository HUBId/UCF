#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_spikebus::SpikeKind;
use ucf_types::Digest32;

const SCALE: i64 = 10_000;
const INPUT_SUMMARY_DIMS: usize = 4;
const WM_MAX_DIMS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsmParams {
    pub dim_x: u16,
    pub dim_u: u16,
    pub scan_blocks: u8,
    pub dt_q: u16,
    pub leak: u16,
    pub selectivity: u16,
    pub commit: Digest32,
}

impl SsmParams {
    pub fn new(
        dim_x: u16,
        dim_u: u16,
        scan_blocks: u8,
        dt_q: u16,
        leak: u16,
        selectivity: u16,
    ) -> Self {
        let dim_x = dim_x.clamp(16, 128);
        let dim_u = dim_u.clamp(8, 64);
        let scan_blocks = scan_blocks.clamp(1, 8);
        let leak = leak.min(10_000);
        let selectivity = selectivity.min(10_000);
        let commit = commit_params(dim_x, dim_u, scan_blocks, dt_q, leak, selectivity);
        Self {
            dim_x,
            dim_u,
            scan_blocks,
            dt_q,
            leak,
            selectivity,
            commit,
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.dim_x as usize, self.dim_u as usize)
    }
}

impl Default for SsmParams {
    fn default() -> Self {
        Self::new(32, 16, 4, 250, 800, 6000)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmState {
    pub x: Vec<i32>,
    pub commit: Digest32,
}

impl SsmState {
    pub fn new(params: &SsmParams) -> Self {
        let x = vec![0; params.dim_x as usize];
        let commit = commit_state(&x, params.commit);
        Self { x, commit }
    }

    pub fn reset_if_dim_mismatch(&mut self, params: &SsmParams) {
        let dim_x = params.dim_x as usize;
        if self.x.len() != dim_x {
            self.x = vec![0; dim_x];
        }
        self.commit = commit_state(&self.x, params.commit);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmInput {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub global_phase: u16,
    pub coherence_plv: u16,
    pub ncde_commit: Digest32,
    pub ncde_energy: u16,
    pub ncde_summary: Vec<i16>,
    pub spike_root: Digest32,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub drift: u16,
    pub surprise: u16,
    pub risk: u16,
    pub commit: Digest32,
}

impl SsmInput {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        global_phase: u16,
        coherence_plv: u16,
        ncde_commit: Digest32,
        ncde_energy: u16,
        ncde_summary: Vec<i16>,
        spike_root: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        drift: u16,
        surprise: u16,
        risk: u16,
    ) -> Self {
        let commit = commit_input(
            cycle_id,
            phase_commit,
            global_phase,
            coherence_plv,
            ncde_commit,
            ncde_energy,
            &ncde_summary,
            spike_root,
            &spike_counts,
            drift,
            surprise,
            risk,
        );
        Self {
            cycle_id,
            phase_commit,
            global_phase,
            coherence_plv,
            ncde_commit,
            ncde_energy,
            ncde_summary,
            spike_root,
            spike_counts,
            drift,
            surprise,
            risk,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmOutput {
    pub cycle_id: u64,
    pub x_commit: Digest32,
    pub wm_vector: Vec<i16>,
    pub wm_salience: u16,
    pub wm_novelty: u16,
    pub commit: Digest32,
}

pub struct SsmCore {
    pub params: SsmParams,
    pub state: SsmState,
}

impl SsmCore {
    pub fn new(params: SsmParams) -> Self {
        let state = SsmState::new(&params);
        Self { params, state }
    }

    pub fn tick(&mut self, inp: &SsmInput) -> SsmOutput {
        self.state.reset_if_dim_mismatch(&self.params);
        let (dim_x, dim_u) = self.params.dims();
        let u = build_input_vector(inp, dim_u);
        let gate_base = gate_base(inp);
        let gate_mods = gate_modifiers(self.params.commit, dim_x);
        let b = matrix_from_commit(b"ucf.ssm.matrix.b.v1", dim_x, dim_u, self.params.commit);
        let blocks = usize::from(self.params.scan_blocks.max(1));
        let block_size = dim_x.div_ceil(blocks);
        let block_index = (inp.cycle_id as usize) % blocks;
        let block_start = block_index * block_size;
        let block_end = (block_start + block_size).min(dim_x);
        let dt_q = i64::from(self.params.dt_q);
        let leak = i64::from(self.params.leak);
        let selectivity = i64::from(self.params.selectivity);

        let mut next = vec![0i32; dim_x];
        for idx in 0..dim_x {
            let prev = i64::from(self.state.x[idx]);
            let decay = prev.saturating_mul(SCALE.saturating_sub(leak)) / SCALE;
            let gate = gate_mods[idx]
                .saturating_mul(gate_base)
                .saturating_mul(selectivity)
                / (SCALE * SCALE);
            let gated = prev.saturating_mul(gate) / SCALE;
            let mut acc = decay.saturating_add(gated);
            if idx >= block_start && idx < block_end {
                let row_offset = idx * dim_u;
                let mut proj = 0i64;
                for (col, coeff) in b[row_offset..row_offset + dim_u].iter().enumerate() {
                    proj = proj.saturating_add(i64::from(*coeff).saturating_mul(u[col]));
                }
                let scaled_proj = proj.saturating_mul(dt_q) / SCALE;
                acc = acc.saturating_add(scaled_proj);
            }
            next[idx] = clamp_i32(acc);
        }
        self.state.x = next;
        let x_commit = commit_state(&self.state.x, self.params.commit);
        self.state.commit = x_commit;
        let wm_vector = build_wm_vector(&self.state.x, dim_x);
        let wm_salience = wm_salience(inp);
        let wm_novelty = wm_novelty(inp);
        let commit = commit_output(
            inp.cycle_id,
            x_commit,
            &wm_vector,
            wm_salience,
            wm_novelty,
            inp.commit,
            self.params.commit,
        );
        SsmOutput {
            cycle_id: inp.cycle_id,
            x_commit,
            wm_vector,
            wm_salience,
            wm_novelty,
            commit,
        }
    }
}

fn build_input_vector(inp: &SsmInput, dim_u: usize) -> Vec<i64> {
    let mut u = Vec::with_capacity(dim_u);
    u.push(i64::from(inp.ncde_energy));
    u.push(i64::from(inp.coherence_plv));
    u.push(i64::from(inp.global_phase >> 8));
    u.push(i64::from(inp.drift));
    u.push(i64::from(inp.surprise));
    u.push(i64::from(inp.risk));
    let (novelty, threat, causal, replay, attention) = spike_summary(&inp.spike_counts);
    u.push(i64::from(novelty));
    u.push(i64::from(threat));
    u.push(i64::from(causal));
    u.push(i64::from(replay));
    u.push(i64::from(attention));

    let summary = inp.ncde_summary.iter().copied().take(INPUT_SUMMARY_DIMS);
    for value in summary {
        u.push(i64::from(value));
    }
    while u.len() < dim_u {
        u.push(0);
    }
    u.truncate(dim_u);
    u
}

fn spike_summary(spike_counts: &[(SpikeKind, u16)]) -> (u16, u16, u16, u16, u16) {
    let mut novelty = 0u16;
    let mut threat = 0u16;
    let mut causal = 0u16;
    let mut replay = 0u16;
    let mut attention = 0u16;
    for (kind, count) in spike_counts {
        match kind {
            SpikeKind::Novelty => novelty = novelty.saturating_add(*count),
            SpikeKind::Threat => threat = threat.saturating_add(*count),
            SpikeKind::CausalLink => causal = causal.saturating_add(*count),
            SpikeKind::ReplayTrigger => replay = replay.saturating_add(*count),
            SpikeKind::AttentionShift => attention = attention.saturating_add(*count),
            SpikeKind::ConsistencyAlert | SpikeKind::Thought | SpikeKind::Unknown(_) => {}
        }
    }
    (
        novelty.min(10_000),
        threat.min(10_000),
        causal.min(10_000),
        replay.min(10_000),
        attention.min(10_000),
    )
}

fn gate_base(inp: &SsmInput) -> i64 {
    let (novelty, threat, _, _, attention) = spike_summary(&inp.spike_counts);
    let spike_boost = i64::from(novelty.saturating_add(threat).saturating_add(attention)) / 2;
    let mut base = i64::from(inp.coherence_plv)
        .saturating_add(spike_boost)
        .saturating_sub(i64::from(inp.drift) / 2)
        .saturating_sub(i64::from(inp.risk) / 3);
    base = base.clamp(0, SCALE);
    base
}

fn wm_salience(inp: &SsmInput) -> u16 {
    let (novelty, threat, _, _, attention) = spike_summary(&inp.spike_counts);
    let spike_boost = i64::from(novelty.saturating_add(threat).saturating_add(attention)) / 2;
    let mut score = i64::from(inp.coherence_plv)
        .saturating_add(spike_boost)
        .saturating_sub(i64::from(inp.drift) / 2)
        .saturating_sub(i64::from(inp.risk) / 2);
    score = score.clamp(0, SCALE);
    score as u16
}

fn wm_novelty(inp: &SsmInput) -> u16 {
    let (novelty, _, _, replay, _) = spike_summary(&inp.spike_counts);
    let spike_boost = i64::from(novelty.saturating_add(replay)) / 2;
    let mut score = i64::from(inp.surprise)
        .saturating_add(spike_boost)
        .saturating_sub(i64::from(inp.coherence_plv) / 2);
    score = score.clamp(0, SCALE);
    score as u16
}

fn build_wm_vector(x: &[i32], dim_x: usize) -> Vec<i16> {
    let len = WM_MAX_DIMS.min(dim_x);
    let mut wm = Vec::with_capacity(len);
    for value in x.iter().take(len) {
        let scaled = i64::from(*value) / SCALE;
        wm.push(clamp_i16(scaled));
    }
    wm
}

fn gate_modifiers(commit: Digest32, dim_x: usize) -> Vec<i64> {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.gate.v1");
    hasher.update(commit.as_bytes());
    hasher.update(&u64::try_from(dim_x).unwrap_or(0).to_be_bytes());
    let mut reader = hasher.finalize_xof();
    let mut bytes = vec![0u8; dim_x.max(1)];
    reader.fill(&mut bytes);
    bytes
        .into_iter()
        .map(|byte| {
            let scaled = i64::from(byte) * SCALE / 255;
            scaled.clamp(0, SCALE)
        })
        .collect()
}

fn matrix_from_commit(domain: &[u8], rows: usize, cols: usize, commit: Digest32) -> Vec<i32> {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(commit.as_bytes());
    hasher.update(&u64::try_from(rows).unwrap_or(0).to_be_bytes());
    hasher.update(&u64::try_from(cols).unwrap_or(0).to_be_bytes());
    let mut reader = hasher.finalize_xof();
    let mut bytes = vec![0u8; rows.saturating_mul(cols).max(1)];
    reader.fill(&mut bytes);
    bytes
        .into_iter()
        .map(|byte| {
            let mapped = (byte % 7) as i32 - 3;
            if mapped == 0 {
                1
            } else {
                mapped
            }
        })
        .collect()
}

fn commit_params(
    dim_x: u16,
    dim_u: u16,
    scan_blocks: u8,
    dt_q: u16,
    leak: u16,
    selectivity: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.params.v1");
    hasher.update(&dim_x.to_be_bytes());
    hasher.update(&dim_u.to_be_bytes());
    hasher.update(&[scan_blocks]);
    hasher.update(&dt_q.to_be_bytes());
    hasher.update(&leak.to_be_bytes());
    hasher.update(&selectivity.to_be_bytes());
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

#[allow(clippy::too_many_arguments)]
fn commit_input(
    cycle_id: u64,
    phase_commit: Digest32,
    global_phase: u16,
    coherence_plv: u16,
    ncde_commit: Digest32,
    ncde_energy: u16,
    ncde_summary: &[i16],
    spike_root: Digest32,
    spike_counts: &[(SpikeKind, u16)],
    drift: u16,
    surprise: u16,
    risk: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.input.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(&global_phase.to_be_bytes());
    hasher.update(&coherence_plv.to_be_bytes());
    hasher.update(ncde_commit.as_bytes());
    hasher.update(&ncde_energy.to_be_bytes());
    hasher.update(&(ncde_summary.len() as u16).to_be_bytes());
    for value in ncde_summary {
        hasher.update(&value.to_be_bytes());
    }
    hasher.update(spike_root.as_bytes());
    hasher.update(&(spike_counts.len() as u16).to_be_bytes());
    for (kind, count) in spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&risk.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_output(
    cycle_id: u64,
    x_commit: Digest32,
    wm_vector: &[i16],
    wm_salience: u16,
    wm_novelty: u16,
    input_commit: Digest32,
    params_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.output.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(x_commit.as_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(params_commit.as_bytes());
    hasher.update(&wm_salience.to_be_bytes());
    hasher.update(&wm_novelty.to_be_bytes());
    hasher.update(&(wm_vector.len() as u16).to_be_bytes());
    for value in wm_vector {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn clamp_i16(value: i64) -> i16 {
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

fn clamp_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input(coherence_plv: u16) -> SsmInput {
        SsmInput::new(
            7,
            Digest32::new([1u8; 32]),
            42_000,
            coherence_plv,
            Digest32::new([2u8; 32]),
            3200,
            vec![10, -5, 12, 1],
            Digest32::new([3u8; 32]),
            vec![(SpikeKind::Novelty, 3), (SpikeKind::Threat, 2)],
            1200,
            3400,
            800,
        )
    }

    #[test]
    fn deterministic_output_commit() {
        let params = SsmParams::default();
        let mut core_a = SsmCore::new(params);
        let mut core_b = SsmCore::new(params);
        let input = sample_input(6200);

        let out_a = core_a.tick(&input);
        let out_b = core_b.tick(&input);

        assert_eq!(out_a.commit, out_b.commit);
        assert_eq!(out_a.x_commit, out_b.x_commit);
    }

    #[test]
    fn higher_leak_reduces_state_magnitude() {
        let params_low = SsmParams::new(32, 16, 4, 250, 300, 6000);
        let params_high = SsmParams::new(32, 16, 4, 250, 8000, 6000);
        let mut low = SsmCore::new(params_low);
        let mut high = SsmCore::new(params_high);
        let input = sample_input(5000);

        for _ in 0..5 {
            low.tick(&input);
            high.tick(&input);
        }

        let low_mag: i64 = low.state.x.iter().map(|value| i64::from(value.abs())).sum();
        let high_mag: i64 = high
            .state
            .x
            .iter()
            .map(|value| i64::from(value.abs()))
            .sum();
        assert!(high_mag < low_mag);
    }

    #[test]
    fn coherence_increases_salience() {
        let mut core = SsmCore::new(SsmParams::default());
        let low = core.tick(&sample_input(2000));
        let high = core.tick(&sample_input(9000));
        assert!(high.wm_salience >= low.wm_salience);
    }
}
