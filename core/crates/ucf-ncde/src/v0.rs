#[derive(Clone, Debug, PartialEq)]
pub enum NcdeIntegrator {
    Euler,
    Rk2,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NcdeCfg {
    pub dim: usize,
    pub dt_ms: u32,
    pub integrator: NcdeIntegrator,
    pub leak: f32,
    pub input_gain: f32,
    pub phase_gain: f32,
    pub clamp: f32,
}

impl NcdeCfg {
    pub fn default_v0() -> Self {
        Self {
            dim: 8,
            dt_ms: 10,
            integrator: NcdeIntegrator::Rk2,
            leak: 0.05,
            input_gain: 0.25,
            phase_gain: 0.10,
            clamp: 4.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NcdeInput {
    pub u: Vec<f32>,
    pub phase: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NcdeState {
    pub x: Vec<f32>,
    pub t_ms: u64,
}

impl NcdeState {
    pub fn new(cfg: &NcdeCfg) -> Self {
        let dim = cfg.dim.max(1);
        Self {
            x: vec![0.0; dim],
            t_ms: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NcdeTick {
    pub now_ms: u64,
    pub l2_q: u8,
    pub phase_q: u8,
}

pub fn ncde_step(cfg: &NcdeCfg, st: &mut NcdeState, now_ms: u64, inp: &NcdeInput) -> NcdeTick {
    let dim = cfg.dim.max(1);
    if st.x.len() != dim {
        st.x.resize(dim, 0.0);
    }

    let dt = (cfg.dt_ms.max(1) as f32) / 1000.0;
    let phase = inp.phase.clamp(0.0, 1.0);
    let phase_mod = 1.0 + cfg.phase_gain * (2.0 * phase - 1.0);

    match cfg.integrator {
        NcdeIntegrator::Euler => {
            for i in 0..dim {
                let u_i = inp.u.get(i).copied().unwrap_or(0.0);
                let drift = phase_mod * (cfg.input_gain * u_i - cfg.leak * st.x[i]);
                st.x[i] += dt * drift;
            }
        }
        NcdeIntegrator::Rk2 => {
            let mut k1 = vec![0.0; dim];
            for (i, k1_i) in k1.iter_mut().enumerate() {
                let u_i = inp.u.get(i).copied().unwrap_or(0.0);
                *k1_i = phase_mod * (cfg.input_gain * u_i - cfg.leak * st.x[i]);
            }

            for (i, x_i) in st.x.iter_mut().enumerate() {
                let u_i = inp.u.get(i).copied().unwrap_or(0.0);
                let x_mid = *x_i + 0.5 * dt * k1[i];
                let k2 = phase_mod * (cfg.input_gain * u_i - cfg.leak * x_mid);
                *x_i += dt * k2;
            }
        }
    }

    let clamp = cfg.clamp.abs().max(1.0e-6);
    for x_i in &mut st.x {
        *x_i = x_i.clamp(-clamp, clamp);
    }

    st.t_ms = now_ms;

    let l2 = st.x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let denom = (clamp * (dim as f32).sqrt()).max(1.0e-6);
    let l2_norm = (l2 / denom).clamp(0.0, 1.0);

    NcdeTick {
        now_ms,
        l2_q: quantize_unit(l2_norm),
        phase_q: quantize_unit(phase),
    }
}

fn quantize_unit(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::{ncde_step, NcdeCfg, NcdeInput, NcdeIntegrator, NcdeState};

    #[test]
    fn rk2_is_closer_than_euler_for_linear_step() {
        let cfg_euler = NcdeCfg {
            dim: 1,
            dt_ms: 100,
            integrator: NcdeIntegrator::Euler,
            leak: 1.0,
            input_gain: 1.0,
            phase_gain: 0.0,
            clamp: 10.0,
        };
        let cfg_rk2 = NcdeCfg {
            integrator: NcdeIntegrator::Rk2,
            ..cfg_euler.clone()
        };

        let mut st_euler = NcdeState::new(&cfg_euler);
        let mut st_rk2 = NcdeState::new(&cfg_rk2);
        let inp = NcdeInput {
            u: vec![1.0],
            phase: 0.5,
        };

        ncde_step(&cfg_euler, &mut st_euler, 100, &inp);
        ncde_step(&cfg_rk2, &mut st_rk2, 100, &inp);

        let expected = 1.0 - (-0.1_f32).exp();
        let err_euler = (st_euler.x[0] - expected).abs();
        let err_rk2 = (st_rk2.x[0] - expected).abs();
        assert!(err_rk2 < err_euler);
    }

    #[test]
    fn clamp_limits_state() {
        let cfg = NcdeCfg {
            dim: 2,
            dt_ms: 1000,
            integrator: NcdeIntegrator::Euler,
            leak: 0.0,
            input_gain: 100.0,
            phase_gain: 0.0,
            clamp: 0.5,
        };
        let mut st = NcdeState::new(&cfg);
        let inp = NcdeInput {
            u: vec![1.0, -1.0],
            phase: 0.5,
        };

        ncde_step(&cfg, &mut st, 1_000, &inp);
        assert_eq!(st.x, vec![0.5, -0.5]);
    }

    #[test]
    fn same_sequence_is_deterministic() {
        let cfg = NcdeCfg::default_v0();
        let mut st_a = NcdeState::new(&cfg);
        let mut st_b = NcdeState::new(&cfg);

        let inputs = [
            NcdeInput {
                u: vec![1.0, 0.2, 0.4],
                phase: 0.1,
            },
            NcdeInput {
                u: vec![0.5, 0.8, 0.1],
                phase: 0.4,
            },
            NcdeInput {
                u: vec![0.0, 0.3, 0.9],
                phase: 0.9,
            },
        ];

        let seq_a: Vec<u8> = inputs
            .iter()
            .enumerate()
            .map(|(idx, inp)| ncde_step(&cfg, &mut st_a, idx as u64 * 10, inp).l2_q)
            .collect();
        let seq_b: Vec<u8> = inputs
            .iter()
            .enumerate()
            .map(|(idx, inp)| ncde_step(&cfg, &mut st_b, idx as u64 * 10, inp).l2_q)
            .collect();

        assert_eq!(seq_a, seq_b);
    }
}
