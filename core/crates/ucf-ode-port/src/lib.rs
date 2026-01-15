#![forbid(unsafe_code)]

/// ODE solver interface for runtime integration backends.
///
/// Real implementations will later live behind feature gates (diffsol/sundials-sys).
pub trait OdeSolver {
    fn step(&self, state: &mut [f32], control: &[f32], dt: f32);
}

/// NCDE solver interface for runtime integration backends.
///
/// Real implementations will later live behind feature gates (diffsol/sundials-sys).
pub trait NcdeSolver {
    fn integrate(&self, state: &mut [f32], control_path: &[[f32; 4]]);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MockOdeSolver;

impl OdeSolver for MockOdeSolver {
    fn step(&self, state: &mut [f32], control: &[f32], dt: f32) {
        let control_len = control.len();
        for (idx, value) in state.iter_mut().enumerate() {
            let control_value = if control_len == 0 {
                0.0
            } else {
                control[idx % control_len]
            };
            *value += (control_value + (idx as f32 * 0.01)) * dt;
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MockNcdeSolver;

impl NcdeSolver for MockNcdeSolver {
    fn integrate(&self, state: &mut [f32], control_path: &[[f32; 4]]) {
        for (step, control) in control_path.iter().enumerate() {
            for (idx, value) in state.iter_mut().enumerate() {
                let control_value = control[idx % control.len()];
                let step_bias = step as f32 * 0.001;
                *value += control_value * 0.01 + step_bias;
            }
        }
    }
}
