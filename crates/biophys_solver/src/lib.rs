#![forbid(unsafe_code)]

use biophys_core::{LifParams, LifState, DEFAULT_REFRACTORY_STEPS};

pub trait StepSolver<S, I, O> {
    fn step(&mut self, state: &mut S, input: &I) -> O;
}

#[derive(Debug, Clone, Copy)]
pub struct LifSolver {
    pub params: LifParams,
    pub dt_ms: u16,
}

impl LifSolver {
    pub fn new(params: LifParams, dt_ms: u16) -> Self {
        assert!(params.tau_ms > 0, "tau_ms must be non-zero");
        Self { params, dt_ms }
    }

    fn update_voltage(&self, v: i32, input_current: i32) -> i32 {
        let tau = self.params.tau_ms as i64;
        let dv = (-(v as i64 - self.params.v_rest as i64) + input_current as i64)
            * self.dt_ms as i64
            / tau;
        v.saturating_add(dv as i32)
    }
}

impl StepSolver<LifState, i32, bool> for LifSolver {
    fn step(&mut self, state: &mut LifState, input: &i32) -> bool {
        if state.refractory_steps > 0 {
            state.refractory_steps = state.refractory_steps.saturating_sub(1);
            state.v = self.params.v_reset;
            return false;
        }

        state.v = self.update_voltage(state.v, *input);
        if state.v >= self.params.v_threshold {
            state.v = self.params.v_reset;
            state.refractory_steps = DEFAULT_REFRACTORY_STEPS;
            return true;
        }
        false
    }
}
