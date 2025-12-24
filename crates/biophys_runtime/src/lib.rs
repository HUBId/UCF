#![forbid(unsafe_code)]

use biophys_core::{clamp_usize, LifParams, LifState, NeuronId, PopCode};
use biophys_solver::{LifSolver, StepSolver};

pub trait BiophysCircuit<I: ?Sized, O> {
    fn step(&mut self, input: &I) -> O;
    fn config_digest(&self) -> [u8; 32];
}

#[derive(Debug, Clone)]
pub struct BiophysRuntime {
    pub params: Vec<LifParams>,
    pub states: Vec<LifState>,
    pub dt_ms: u16,
    pub step_count: u64,
    pub max_spikes_per_step: usize,
}

impl BiophysRuntime {
    pub fn new(
        params: Vec<LifParams>,
        states: Vec<LifState>,
        dt_ms: u16,
        max_spikes_per_step: usize,
    ) -> Self {
        assert!(dt_ms > 0, "dt_ms must be non-zero");
        assert_eq!(params.len(), states.len(), "params/state mismatch");
        Self {
            params,
            states,
            dt_ms,
            step_count: 0,
            max_spikes_per_step,
        }
    }

    pub fn step(&mut self, inputs: &[i32]) -> PopCode {
        assert_eq!(inputs.len(), self.states.len(), "input length mismatch");
        let mut spikes = Vec::new();
        for (idx, (state, params)) in self.states.iter_mut().zip(self.params.iter()).enumerate() {
            let mut solver = LifSolver::new(*params, self.dt_ms);
            if solver.step(state, &inputs[idx]) {
                spikes.push(NeuronId(idx as u32));
            }
        }
        spikes.sort_by_key(|id| id.0);
        let max_spikes = clamp_usize(spikes.len(), self.max_spikes_per_step);
        spikes.truncate(max_spikes);
        self.step_count = self.step_count.saturating_add(1);
        PopCode { spikes }
    }

    pub fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:CFG");
        update_u16(&mut hasher, self.dt_ms);
        update_u32(&mut hasher, self.params.len() as u32);
        for params in &self.params {
            update_u16(&mut hasher, params.tau_ms);
            update_i32(&mut hasher, params.v_rest);
            update_i32(&mut hasher, params.v_reset);
            update_i32(&mut hasher, params.v_threshold);
        }
        *hasher.finalize().as_bytes()
    }

    pub fn snapshot_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:SNAP");
        update_u64(&mut hasher, self.step_count);
        update_u32(&mut hasher, self.states.len() as u32);
        for state in &self.states {
            update_i32(&mut hasher, state.v);
            update_u16(&mut hasher, state.refractory_steps);
        }
        *hasher.finalize().as_bytes()
    }
}

impl BiophysCircuit<[i32], PopCode> for BiophysRuntime {
    fn step(&mut self, input: &[i32]) -> PopCode {
        BiophysRuntime::step(self, input)
    }

    fn config_digest(&self) -> [u8; 32] {
        BiophysRuntime::config_digest(self)
    }
}

fn update_u16(hasher: &mut blake3::Hasher, value: u16) {
    hasher.update(&value.to_le_bytes());
}

fn update_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn update_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn update_i32(hasher: &mut blake3::Hasher, value: i32) {
    hasher.update(&value.to_le_bytes());
}
