#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Integrator {
    Euler,
}

pub fn step_euler(x: f32, dxdt: f32, dt_s: f32) -> f32 {
    x + dxdt * dt_s
}

pub fn clamp01(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

pub fn prod_clear_step(h: f32, prod: f32, clearance: f32, dt_s: f32) -> f32 {
    let dxdt = prod - clearance.max(0.0) * h;
    clamp01(step_euler(h, dxdt, dt_s.max(0.0)))
}
