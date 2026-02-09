#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const PARAMS_DOMAIN: &[u8] = b"ucf.jepa.params.v1";
const INPUTS_DOMAIN: &[u8] = b"ucf.jepa.inputs.v1";
const OUTPUTS_DOMAIN: &[u8] = b"ucf.jepa.outputs.v1";
const CORE_DOMAIN: &[u8] = b"ucf.jepa.core.v1";
const WORLD_STATE_DOMAIN: &[u8] = b"ucf.jepa.world_state.v1";
const PREDICTION_DOMAIN: &[u8] = b"ucf.jepa.prediction.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JepaParams {
    /// config for placeholder predictor
    pub horizon: u8, // <= 4
    pub surprise_hi: u16, // 0..10000
    pub commit: Digest32,
}

impl JepaParams {
    pub fn new(horizon: u8, surprise_hi: u16) -> Self {
        let horizon = horizon.clamp(1, 4);
        let surprise_hi = surprise_hi.min(10_000);
        let commit = commit_params(horizon, surprise_hi);
        Self {
            horizon,
            surprise_hi,
            commit,
        }
    }
}

impl Default for JepaParams {
    fn default() -> Self {
        Self::new(2, 7_000)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JepaInputs {
    pub cycle_id: u64,
    // percept summary:
    pub percept_commit: Digest32,
    pub percept_energy: u16,
    // memory/context:
    pub ssm_state_digest: Digest32,
    // timing:
    pub phase_bus_commit: Digest32,
    pub gamma_bucket: u8,
    // last world state:
    pub last_world_state: Digest32,
    // optional exogenous:
    pub coupling_root: Digest32,
    pub commit: Digest32,
}

impl JepaInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        percept_commit: Digest32,
        percept_energy: u16,
        ssm_state_digest: Digest32,
        phase_bus_commit: Digest32,
        gamma_bucket: u8,
        last_world_state: Digest32,
        coupling_root: Digest32,
    ) -> Self {
        let commit = commit_inputs(
            cycle_id,
            percept_commit,
            percept_energy,
            ssm_state_digest,
            phase_bus_commit,
            gamma_bucket,
            last_world_state,
            coupling_root,
        );
        Self {
            cycle_id,
            percept_commit,
            percept_energy,
            ssm_state_digest,
            phase_bus_commit,
            gamma_bucket,
            last_world_state,
            coupling_root,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JepaOutputs {
    pub cycle_id: u64,
    pub world_state: Digest32,
    pub prediction: Digest32,
    pub surprise: u16, // 0..10000
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JepaCore {
    pub params: JepaParams,
    pub last_world_state: Digest32,
    pub last_prediction: Digest32,
    pub commit: Digest32,
}

impl JepaCore {
    pub fn new(params: JepaParams) -> Self {
        let last_world_state = Digest32::new([0u8; 32]);
        let last_prediction = Digest32::new([0u8; 32]);
        let commit = commit_core(params.commit, last_world_state, last_prediction);
        Self {
            params,
            last_world_state,
            last_prediction,
            commit,
        }
    }

    pub fn tick(&mut self, inp: &JepaInputs) -> JepaOutputs {
        let world_state = digest_world_state(
            inp.percept_commit,
            inp.ssm_state_digest,
            inp.phase_bus_commit,
            inp.coupling_root,
        );
        let prediction =
            digest_prediction(inp.last_world_state, inp.phase_bus_commit, inp.gamma_bucket);
        let surprise = surprise_from_digests(world_state, prediction);
        let commit = commit_outputs(
            inp.cycle_id,
            world_state,
            prediction,
            surprise,
            self.params.commit,
            inp.commit,
        );
        self.last_world_state = world_state;
        self.last_prediction = prediction;
        self.commit = commit_core(self.params.commit, world_state, prediction);
        JepaOutputs {
            cycle_id: inp.cycle_id,
            world_state,
            prediction,
            surprise,
            commit,
        }
    }
}

impl Default for JepaCore {
    fn default() -> Self {
        Self::new(JepaParams::default())
    }
}

fn commit_params(horizon: u8, surprise_hi: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PARAMS_DOMAIN);
    hasher.update(&[horizon]);
    hasher.update(&surprise_hi.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_inputs(
    cycle_id: u64,
    percept_commit: Digest32,
    percept_energy: u16,
    ssm_state_digest: Digest32,
    phase_bus_commit: Digest32,
    gamma_bucket: u8,
    last_world_state: Digest32,
    coupling_root: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(INPUTS_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(percept_commit.as_bytes());
    hasher.update(&percept_energy.to_be_bytes());
    hasher.update(ssm_state_digest.as_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(&[gamma_bucket]);
    hasher.update(last_world_state.as_bytes());
    hasher.update(coupling_root.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    world_state: Digest32,
    prediction: Digest32,
    surprise: u16,
    params_commit: Digest32,
    inputs_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(OUTPUTS_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(world_state.as_bytes());
    hasher.update(prediction.as_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(params_commit.as_bytes());
    hasher.update(inputs_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_core(params_commit: Digest32, world_state: Digest32, prediction: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(CORE_DOMAIN);
    hasher.update(params_commit.as_bytes());
    hasher.update(world_state.as_bytes());
    hasher.update(prediction.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_world_state(
    percept_commit: Digest32,
    ssm_state_digest: Digest32,
    phase_bus_commit: Digest32,
    coupling_root: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(WORLD_STATE_DOMAIN);
    hasher.update(percept_commit.as_bytes());
    hasher.update(ssm_state_digest.as_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(coupling_root.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_prediction(
    last_world_state: Digest32,
    phase_bus_commit: Digest32,
    gamma_bucket: u8,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(PREDICTION_DOMAIN);
    hasher.update(last_world_state.as_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(&[gamma_bucket]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn surprise_from_digests(world_state: Digest32, prediction: Digest32) -> u16 {
    let left = trunc16(world_state);
    let right = trunc16(prediction);
    let xor = left ^ right;
    let count = xor.count_ones() as u32;
    let scaled = count.saturating_mul(10_000) / 16;
    u16::try_from(scaled.min(10_000)).unwrap_or(10_000)
}

fn trunc16(digest: Digest32) -> u16 {
    let bytes = digest.as_bytes();
    u16::from_be_bytes([bytes[0], bytes[1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_is_deterministic() {
        let params = JepaParams::new(3, 6_500);
        let inputs = JepaInputs::new(
            7,
            Digest32::new([1u8; 32]),
            120,
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            2,
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
        );
        let mut core_a = JepaCore::new(params);
        let mut core_b = JepaCore::new(params);

        let out_a = core_a.tick(&inputs);
        let out_b = core_b.tick(&inputs);

        assert_eq!(out_a, out_b);
        assert_eq!(core_a.last_world_state, core_b.last_world_state);
        assert_eq!(core_a.last_prediction, core_b.last_prediction);
    }

    #[test]
    fn surprise_is_small_when_digests_match() {
        let digest = Digest32::new([9u8; 32]);
        let surprise = surprise_from_digests(digest, digest);
        assert_eq!(surprise, 0);
    }
}
