pub const D: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsmCfg {
    pub alpha: f32,
    pub beta: f32,
    pub leak: f32,
    pub gate_k: f32,
}

impl Default for SsmCfg {
    fn default() -> Self {
        Self {
            alpha: 0.10,
            beta: 0.50,
            leak: 0.01,
            gate_k: 4.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SsmState {
    pub x: [f32; D],
}

impl Default for SsmState {
    fn default() -> Self {
        Self { x: [0.0; D] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsmInputs {
    pub attention: f32,
    pub integration: f32,
    pub dopamine: f32,
    pub noise: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SsmOut {
    pub ctx: [f32; D],
    pub gate: f32,
    pub norm_l2: f32,
    pub sparsity: f32,
}

pub fn mix_inputs(a: &[f32; D], b: &[f32; D], w: f32) -> [f32; D] {
    let wa = (1.0 - w).clamp(0.0, 1.0);
    let wb = w.clamp(0.0, 1.0);
    let mut out = [0.0; D];
    for i in 0..D {
        out[i] = wa * a[i] + wb * b[i];
    }
    out
}

pub fn step(state: &mut SsmState, u: &[f32; D], inp: SsmInputs, cfg: SsmCfg) -> SsmOut {
    let g_raw = 0.5 * inp.attention + 0.3 * inp.integration + 0.2 * inp.dopamine;
    let gate = sigmoid(cfg.gate_k * (g_raw - 0.5));
    let eta = (cfg.alpha * gate * (1.0 - inp.noise)).clamp(0.0, 1.0);

    for (x, ui) in state.x.iter_mut().zip(u.iter()) {
        *x *= 1.0 - cfg.leak;
        *x = (1.0 - eta) * *x + eta * (cfg.beta * *ui);
    }

    let norm_l2 = state.x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let sparsity = state.x.iter().filter(|v| v.abs() < 1e-3).count() as f32 / D as f32;

    SsmOut {
        ctx: state.x,
        gate,
        norm_l2,
        sparsity,
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
