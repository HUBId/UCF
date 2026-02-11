#![forbid(unsafe_code)]

pub type Dim = usize;

#[derive(Clone, Debug, PartialEq)]
pub struct SsmCfg {
    pub n: Dim,
    pub in_dim: Dim,
    pub out_dim: Dim,
    pub step_ms: u64,
    pub alpha: f32,
    pub gate_floor: f32,
}

impl SsmCfg {
    pub fn default_small() -> Self {
        Self {
            n: 32,
            in_dim: 8,
            out_dim: 8,
            step_ms: 10,
            alpha: 0.98,
            gate_floor: 0.05,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SsmState {
    pub x: Vec<f32>,
    pub last_y: Vec<f32>,
    pub last_update_ms: u64,
}

impl SsmState {
    pub fn new(cfg: &SsmCfg, now_ms: u64) -> Self {
        Self {
            x: vec![0.0; cfg.n],
            last_y: vec![0.0; cfg.out_dim],
            last_update_ms: now_ms,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SsmStep {
    pub now_ms: u64,
    pub gate: f32,
    pub y: Vec<f32>,
}

pub fn weight(i: usize, j: usize) -> f32 {
    ((((i * 131 + j * 17) % 97) as f32) / 97.0 - 0.5) * 0.1
}

pub fn ssm_step(cfg: &SsmCfg, st: &mut SsmState, now_ms: u64, input: &[f32], gate: f32) -> SsmStep {
    let g = gate.clamp(0.0, 1.0).max(cfg.gate_floor);
    let mut next = vec![0.0; cfg.n];

    for (k, next_k) in next.iter_mut().enumerate() {
        let mut bu = 0.0;
        for j in 0..cfg.in_dim {
            let u = input.get(j).copied().unwrap_or(0.0);
            bu += weight(k, j) * u;
        }

        let offdiag_idx = (k + 1) % cfg.n.max(1);
        let offdiag = 0.01 * weight(k, offdiag_idx) * st.x[offdiag_idx];
        *next_k = cfg.alpha * st.x[k] + g * bu + offdiag;
    }

    st.x = next;

    let mut y = vec![0.0; cfg.out_dim];
    for (o, y_o) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..cfg.n {
            acc += weight(o, k) * st.x[k];
        }
        *y_o = acc;
    }

    st.last_y = y.clone();
    st.last_update_ms = now_ms;

    SsmStep { now_ms, gate: g, y }
}

#[cfg(test)]
mod tests {
    use super::{ssm_step, SsmCfg, SsmState};

    fn energy(y: &[f32]) -> f32 {
        y.iter().map(|v| v.abs()).sum::<f32>() / (y.len() as f32)
    }

    #[test]
    fn deterministic_with_same_inputs() {
        let cfg = SsmCfg::default_small();
        let input = vec![0.1; cfg.in_dim];

        let mut a = SsmState::new(&cfg, 0);
        let mut b = SsmState::new(&cfg, 0);

        let _ = ssm_step(&cfg, &mut a, 10, &input, 0.4);
        let out_a = ssm_step(&cfg, &mut a, 20, &input, 0.4);

        let _ = ssm_step(&cfg, &mut b, 10, &input, 0.4);
        let out_b = ssm_step(&cfg, &mut b, 20, &input, 0.4);

        assert_eq!(out_a.y, out_b.y);
    }

    #[test]
    fn enforces_gate_floor() {
        let cfg = SsmCfg::default_small();
        let mut st = SsmState::new(&cfg, 0);
        let out = ssm_step(&cfg, &mut st, 10, &[0.2; 8], 0.0);
        assert_eq!(out.gate, cfg.gate_floor);
    }

    #[test]
    fn energy_increases_with_stronger_input() {
        let cfg = SsmCfg::default_small();
        let mut low = SsmState::new(&cfg, 0);
        let mut high = SsmState::new(&cfg, 0);

        let low_out = ssm_step(&cfg, &mut low, 10, &[0.05; 8], 0.8);
        let high_out = ssm_step(&cfg, &mut high, 10, &[0.8; 8], 0.8);

        assert!(energy(&high_out.y) >= energy(&low_out.y));
    }
}
