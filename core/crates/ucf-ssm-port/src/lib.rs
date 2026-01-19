#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_attn_controller::AttentionWeights;
use ucf_types::Digest32;

const GATE_SCALE: i64 = 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmConfig {
    pub state_dim: usize,
    pub input_dim: usize,
    pub scan_chunk: usize,
}

impl SsmConfig {
    pub fn new(state_dim: usize, input_dim: usize, scan_chunk: usize) -> Self {
        Self {
            state_dim,
            input_dim,
            scan_chunk: scan_chunk.max(1),
        }
    }

    pub fn validate_state(&self, state: &SsmState) -> Result<(), SsmError> {
        if state.s.len() != self.state_dim {
            return Err(SsmError::StoreError(format!(
                "state_dim mismatch: expected {}, got {}",
                self.state_dim,
                state.s.len()
            )));
        }
        Ok(())
    }

    pub fn validate_input(&self, input: &SsmInput) -> Result<(), SsmError> {
        if input.x.len() != self.input_dim {
            return Err(SsmError::ConfigError(format!(
                "input_dim mismatch: expected {}, got {}",
                self.input_dim,
                input.x.len()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmState {
    pub s: Vec<i32>,
    pub commit: Digest32,
}

impl SsmState {
    pub fn new(s: Vec<i32>) -> Self {
        let commit = commit_state(&s);
        Self { s, commit }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmInput {
    pub x: Vec<i16>,
    pub commit: Digest32,
}

impl SsmInput {
    pub fn from_commitment(commit: Digest32, input_dim: usize) -> Result<Self, SsmError> {
        if input_dim == 0 {
            return Err(SsmError::ConfigError(
                "input_dim must be greater than zero".to_string(),
            ));
        }
        let mut bytes = vec![0u8; input_dim.saturating_mul(2)];
        let mut hasher = Hasher::new();
        hasher.update(b"ucf.ssm.input.v1");
        hasher.update(commit.as_bytes());
        hasher.update(&u64::try_from(input_dim).unwrap_or(0).to_be_bytes());
        let mut reader = hasher.finalize_xof();
        reader.fill(&mut bytes);
        let mut x = Vec::with_capacity(input_dim);
        for chunk in bytes.chunks_exact(2) {
            let pair = [chunk[0], chunk[1]];
            x.push(i16::from_be_bytes(pair));
        }
        Ok(Self { x, commit })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmOutput {
    pub y: Vec<i16>,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SsmError {
    ConfigError(String),
    StoreError(String),
}

pub trait SsmPort {
    fn reset(&mut self);
    fn update(&mut self, inp: &SsmInput, attn: Option<&AttentionWeights>) -> SsmOutput;
    fn state(&self) -> &SsmState;
}

#[derive(Clone)]
pub struct DeterministicSsmPort {
    config: SsmConfig,
    state: SsmState,
    matrices: SsmMatrices,
}

#[derive(Clone)]
struct SsmMatrices {
    a: Vec<i32>,
    b: Vec<i32>,
    c: Vec<i32>,
}

impl DeterministicSsmPort {
    pub fn new(config: SsmConfig) -> Self {
        let state = SsmState::new(vec![0; config.state_dim]);
        let matrices = SsmMatrices::new(&config);
        Self {
            config,
            state,
            matrices,
        }
    }

    pub fn update_many(
        &mut self,
        inputs: &[SsmInput],
        attn: Option<&AttentionWeights>,
    ) -> Vec<SsmOutput> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(self.config.scan_chunk) {
            for input in chunk {
                outputs.push(self.update(input, attn));
            }
        }
        outputs
    }

    fn try_update(
        &mut self,
        input: &SsmInput,
        attn: Option<&AttentionWeights>,
    ) -> Result<SsmOutput, SsmError> {
        self.config.validate_input(input)?;
        self.config.validate_state(&self.state)?;
        let gate = gate_scale(attn);
        let proposed = self.proposed_state(&input.x);
        for (idx, next) in proposed.into_iter().enumerate() {
            let prev = self.state.s[idx];
            let blended = blend_state(prev, next, gate);
            self.state.s[idx] = blended;
        }
        self.state.commit = commit_state(&self.state.s);
        let y = self.output_from_state();
        let commit = commit_output(&y, &self.state.commit, &input.commit);
        Ok(SsmOutput { y, commit })
    }

    fn proposed_state(&self, input: &[i16]) -> Vec<i32> {
        let mut next = vec![0i32; self.config.state_dim];
        for (row, next_value) in next.iter_mut().enumerate() {
            let mut acc = 0i64;
            let row_offset = row * self.config.state_dim;
            for (col, coeff) in self.matrices.a[row_offset..row_offset + self.config.state_dim]
                .iter()
                .enumerate()
            {
                acc = acc
                    .saturating_add(i64::from(*coeff).saturating_mul(i64::from(self.state.s[col])));
            }
            let row_offset = row * self.config.input_dim;
            for (col, coeff) in self.matrices.b[row_offset..row_offset + self.config.input_dim]
                .iter()
                .enumerate()
            {
                acc = acc.saturating_add(i64::from(*coeff).saturating_mul(i64::from(input[col])));
            }
            *next_value = clamp_i32(acc);
        }
        next
    }

    fn output_from_state(&self) -> Vec<i16> {
        let mut y = vec![0i16; self.config.input_dim];
        for (row, out) in y.iter_mut().enumerate() {
            let mut acc = 0i64;
            let row_offset = row * self.config.state_dim;
            for (col, coeff) in self.matrices.c[row_offset..row_offset + self.config.state_dim]
                .iter()
                .enumerate()
            {
                acc = acc
                    .saturating_add(i64::from(*coeff).saturating_mul(i64::from(self.state.s[col])));
            }
            *out = clamp_i16(acc);
        }
        y
    }
}

impl SsmPort for DeterministicSsmPort {
    fn reset(&mut self) {
        self.state.s.fill(0);
        self.state.commit = commit_state(&self.state.s);
    }

    fn update(&mut self, inp: &SsmInput, attn: Option<&AttentionWeights>) -> SsmOutput {
        match self.try_update(inp, attn) {
            Ok(output) => output,
            Err(_) => {
                self.reset();
                let y = vec![0i16; self.config.input_dim];
                let commit = commit_output(&y, &self.state.commit, &inp.commit);
                SsmOutput { y, commit }
            }
        }
    }

    fn state(&self) -> &SsmState {
        &self.state
    }
}

impl SsmMatrices {
    fn new(config: &SsmConfig) -> Self {
        let a = matrix_from_hash(b"ucf.ssm.matrix.a.v1", config.state_dim, config.state_dim);
        let b = matrix_from_hash(b"ucf.ssm.matrix.b.v1", config.state_dim, config.input_dim);
        let c = matrix_from_hash(b"ucf.ssm.matrix.c.v1", config.input_dim, config.state_dim);
        Self { a, b, c }
    }
}

fn gate_scale(attn: Option<&AttentionWeights>) -> i64 {
    let Some(attn) = attn else {
        return GATE_SCALE;
    };
    let gain = i64::from(attn.gain.max(1));
    let suppress = i64::from(attn.noise_suppress);
    let denom = gain.saturating_add(suppress).max(1);
    (gain.saturating_mul(GATE_SCALE) / denom).clamp(0, GATE_SCALE)
}

fn blend_state(prev: i32, next: i32, gate: i64) -> i32 {
    let weighted_next = gate.saturating_mul(i64::from(next));
    let weighted_prev = (GATE_SCALE - gate).saturating_mul(i64::from(prev));
    clamp_i32((weighted_next + weighted_prev) / GATE_SCALE)
}

fn clamp_i16(value: i64) -> i16 {
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

fn clamp_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn matrix_from_hash(domain: &[u8], rows: usize, cols: usize) -> Vec<i32> {
    let mut hasher = Hasher::new();
    hasher.update(domain);
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

fn commit_state(state: &[i32]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.state.v1");
    hasher.update(&u64::try_from(state.len()).unwrap_or(0).to_be_bytes());
    for value in state {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_output(y: &[i16], state_commit: &Digest32, input_commit: &Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ssm.output.v1");
    hasher.update(state_commit.as_bytes());
    hasher.update(input_commit.as_bytes());
    for value in y {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_attn_controller::{AttentionWeights, FocusChannel};

    fn config() -> SsmConfig {
        SsmConfig::new(4, 3, 2)
    }

    fn input() -> SsmInput {
        SsmInput {
            x: vec![10, -4, 7],
            commit: Digest32::new([2u8; 32]),
        }
    }

    fn attention(gain: u16, suppress: u16) -> AttentionWeights {
        AttentionWeights {
            channel: FocusChannel::Task,
            gain,
            noise_suppress: suppress,
            replay_bias: 1000,
            commit: Digest32::new([9u8; 32]),
        }
    }

    fn l1_distance(a: &[i32], b: &[i32]) -> i64 {
        a.iter()
            .zip(b.iter())
            .map(|(left, right)| i64::from(left.abs_diff(*right)))
            .sum()
    }

    #[test]
    fn deterministic_state_evolution() {
        let config = config();
        let mut port_a = DeterministicSsmPort::new(config.clone());
        let mut port_b = DeterministicSsmPort::new(config);
        let input = input();

        let out_a = port_a.update(&input, None);
        let out_b = port_b.update(&input, None);

        assert_eq!(out_a, out_b);
        assert_eq!(port_a.state(), port_b.state());
    }

    #[test]
    fn attention_gain_affects_state_change() {
        let config = config();
        let mut port_high = DeterministicSsmPort::new(config.clone());
        let mut port_low = DeterministicSsmPort::new(config);
        let input = input();
        let baseline = port_high.state().s.clone();

        let _ = port_high.update(&input, Some(&attention(9000, 500)));
        let _ = port_low.update(&input, Some(&attention(500, 9000)));

        let high_distance = l1_distance(&baseline, &port_high.state().s);
        let low_distance = l1_distance(&baseline, &port_low.state().s);

        assert!(high_distance > low_distance);
    }

    #[test]
    fn mismatched_dimensions_surface_errors() {
        let config = config();
        let bad_state = SsmState::new(vec![0; config.state_dim + 1]);
        let bad_input = SsmInput {
            x: vec![1, 2],
            commit: Digest32::new([1u8; 32]),
        };

        assert!(matches!(
            config.validate_state(&bad_state),
            Err(SsmError::StoreError(_))
        ));
        assert!(matches!(
            config.validate_input(&bad_input),
            Err(SsmError::ConfigError(_))
        ));
    }
}
