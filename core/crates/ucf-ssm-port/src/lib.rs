#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::{Digest32, WorldStateVec};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmState {
    pub digest: Digest32,
    pub step: u64,
}

impl SsmState {
    pub fn new(digest: Digest32, step: u64) -> Self {
        Self { digest, step }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsmOutput {
    pub digest: Digest32,
    pub state_digest: Digest32,
    pub step: u64,
}

pub trait SsmPort {
    fn update(&self, state: &mut SsmState, input: &WorldStateVec) -> SsmOutput;
}

#[derive(Clone, Default)]
pub struct MockSsmPort;

impl MockSsmPort {
    pub fn new() -> Self {
        Self
    }
}

impl SsmPort for MockSsmPort {
    fn update(&self, state: &mut SsmState, input: &WorldStateVec) -> SsmOutput {
        let input_digest = hash_world_state(input);
        let next_digest = hash_two(&state.digest, &input_digest);
        state.digest = next_digest;
        state.step = state.step.saturating_add(1);

        let output_digest = hash_step(&state.digest, state.step);
        SsmOutput {
            digest: output_digest,
            state_digest: state.digest,
            step: state.step,
        }
    }
}

fn hash_world_state(input: &WorldStateVec) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&u64::try_from(input.dims.len()).unwrap_or(0).to_be_bytes());
    for dim in &input.dims {
        hasher.update(&dim.to_be_bytes());
    }
    hasher.update(&input.bytes);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hash_two(left: &Digest32, right: &Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(left.as_bytes());
    hasher.update(right.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hash_step(digest: &Digest32, step: u64) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(digest.as_bytes());
    hasher.update(&step.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_ssm_is_deterministic() {
        let input = WorldStateVec::new(vec![1, 2, 3], vec![3]);
        let digest = Digest32::new([7u8; 32]);
        let mut state_a = SsmState::new(digest, 0);
        let mut state_b = SsmState::new(digest, 0);
        let port = MockSsmPort::new();

        let out_a = port.update(&mut state_a, &input);
        let out_b = port.update(&mut state_b, &input);

        assert_eq!(out_a, out_b);
        assert_eq!(state_a, state_b);
        assert_eq!(state_a.step, 1);
    }
}
