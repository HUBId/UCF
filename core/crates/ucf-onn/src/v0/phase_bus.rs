use core::f32::consts::PI;

use super::types::PhaseDeg;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OnnNode {
    Global,
    Ssm,
    Nsr,
    Cde,
    Iit,
    Sle,
    Ncde,
    Spikes,
    Tcf,
    Custom(u16),
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnnCfg {
    pub dt_ms: u32,
    pub omega_hz: f32,
    pub k_couple: f32,
    pub global_pull: f32,
    pub clamp_step: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnnState {
    pub phase: Vec<f32>,
    pub omega: Vec<f32>,
    pub nodes: Vec<OnnNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnnInput {
    pub now_ms: u64,
    pub anchors: Vec<(OnnNode, f32)>,
    pub gate: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnnOut {
    pub now_ms: u64,
    pub global_phase_0_1: f32,
    pub lock_nsr_jepa_proxy: f32,
    pub lock_nsr_cde: f32,
    pub lock_nsr_ssm: f32,
}

impl OnnCfg {
    pub fn default_v0() -> Self {
        Self {
            dt_ms: 10,
            omega_hz: 40.0,
            k_couple: 0.8,
            global_pull: 0.6,
            clamp_step: 0.35,
        }
    }
}

impl OnnState {
    pub fn new(cfg: &OnnCfg, nodes: &[OnnNode]) -> Self {
        let phase = vec![0.0; nodes.len()];
        let omega = vec![2.0 * PI * cfg.omega_hz; nodes.len()];
        Self {
            phase,
            omega,
            nodes: nodes.to_vec(),
        }
    }

    pub fn phase_deg(&self, node: OnnNode) -> Option<PhaseDeg> {
        self.index_of(node)
            .map(|idx| PhaseDeg((self.phase[idx] * 180.0 / PI).rem_euclid(360.0)))
    }

    fn index_of(&self, node: OnnNode) -> Option<usize> {
        self.nodes.iter().position(|n| *n == node)
    }
}

pub fn wrap_2pi(x: f32) -> f32 {
    x.rem_euclid(2.0 * PI)
}

fn wrap_pi(x: f32) -> f32 {
    (x + PI).rem_euclid(2.0 * PI) - PI
}

fn lock_metric(a: f32, b: f32) -> f32 {
    1.0 - (wrap_pi(a - b).abs() / PI)
}

fn phase_at(st: &OnnState, node: OnnNode) -> Option<f32> {
    st.index_of(node).map(|idx| st.phase[idx])
}

pub fn onn_step(cfg: &OnnCfg, st: &mut OnnState, inp: &OnnInput) -> OnnOut {
    let dt = cfg.dt_ms as f32 / 1000.0;
    let base_omega = 2.0 * PI * cfg.omega_hz;
    for omega in &mut st.omega {
        *omega = base_omega;
    }

    let coupling_scale = cfg.k_couple * inp.gate.clamp(0.0, 1.0);
    let global_idx = st
        .index_of(OnnNode::Global)
        .expect("OnnNode::Global must be registered");

    let old = st.phase.clone();
    let phase_g = old[global_idx];
    let cde_phase = phase_at(st, OnnNode::Cde).unwrap_or(phase_g);
    let nsr_phase = phase_at(st, OnnNode::Nsr).unwrap_or(phase_g);
    let ssm_phase = phase_at(st, OnnNode::Ssm).unwrap_or(phase_g);

    for (i, phase_i_ref) in old.iter().enumerate().take(st.phase.len()) {
        if i == global_idx {
            continue;
        }

        let node_i = st.nodes[i];
        let phase_i = *phase_i_ref;
        let mut sum = (phase_g - phase_i).sin() * cfg.global_pull;

        match node_i {
            OnnNode::Nsr => {
                sum += (cde_phase - phase_i).sin() * 0.35;
                sum += (ssm_phase - phase_i).sin() * 0.25;
            }
            OnnNode::Cde => {
                sum += (nsr_phase - phase_i).sin() * 0.35;
            }
            OnnNode::Ssm => {
                sum += (nsr_phase - phase_i).sin() * 0.25;
            }
            _ => {}
        }

        let dtheta =
            (dt * (st.omega[i] + coupling_scale * sum)).clamp(-cfg.clamp_step, cfg.clamp_step);
        st.phase[i] = wrap_2pi(st.phase[i] + dtheta);
    }

    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    for node in [OnnNode::Nsr, OnnNode::Cde, OnnNode::Ssm] {
        if let Some(phi) = phase_at(st, node) {
            sum_sin += phi.sin();
            sum_cos += phi.cos();
        }
    }
    let mean_phase = sum_sin.atan2(sum_cos);
    let dtheta_g = (dt
        * (st.omega[global_idx] + coupling_scale * 0.2 * (mean_phase - old[global_idx]).sin()))
    .clamp(-cfg.clamp_step, cfg.clamp_step);
    st.phase[global_idx] = wrap_2pi(st.phase[global_idx] + dtheta_g);

    for (node, phase_01) in &inp.anchors {
        if let Some(idx) = st.index_of(*node) {
            let target = wrap_2pi(*phase_01 * 2.0 * PI);
            st.phase[idx] = wrap_2pi(st.phase[idx] + 0.15 * (target - st.phase[idx]).sin());
        }
    }

    let phase_global = st.phase[global_idx];
    let phase_nsr = phase_at(st, OnnNode::Nsr).unwrap_or(phase_global);
    let phase_cde = phase_at(st, OnnNode::Cde).unwrap_or(phase_global);
    let phase_ssm = phase_at(st, OnnNode::Ssm).unwrap_or(phase_global);

    let lock_nsr_cde = lock_metric(phase_nsr, phase_cde).clamp(0.0, 1.0);
    let lock_nsr_ssm = lock_metric(phase_nsr, phase_ssm).clamp(0.0, 1.0);

    OnnOut {
        now_ms: inp.now_ms,
        global_phase_0_1: (phase_global / (2.0 * PI)).clamp(0.0, 1.0),
        lock_nsr_jepa_proxy: lock_nsr_cde,
        lock_nsr_cde,
        lock_nsr_ssm,
    }
}

#[cfg(test)]
mod tests {
    use super::{lock_metric, onn_step, wrap_2pi, OnnCfg, OnnInput, OnnNode, OnnState};
    use core::f32::consts::PI;

    #[test]
    fn wrap_2pi_stability() {
        assert!((wrap_2pi(0.0) - 0.0).abs() < 1e-6);
        assert!((wrap_2pi(2.0 * PI) - 0.0).abs() < 1e-6);
        assert!((wrap_2pi(-0.5) - (2.0 * PI - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn lock_metric_properties() {
        let a = 1.23;
        assert!((lock_metric(a, a) - 1.0).abs() < 1e-6);
        assert!(lock_metric(a, a + PI).abs() < 1e-6);
    }

    #[test]
    fn coherence_increases_for_nsr_and_cde() {
        let cfg = OnnCfg::default_v0();
        let nodes = [OnnNode::Global, OnnNode::Ssm, OnnNode::Nsr, OnnNode::Cde];
        let mut st = OnnState::new(&cfg, &nodes);
        st.phase[2] = 0.0;
        st.phase[3] = PI;
        let baseline = lock_metric(st.phase[2], st.phase[3]);

        for step in 0..300 {
            let _ = onn_step(
                &cfg,
                &mut st,
                &OnnInput {
                    now_ms: step,
                    anchors: Vec::new(),
                    gate: 1.0,
                },
            );
        }

        let lock = lock_metric(st.phase[2], st.phase[3]);
        assert!(lock > baseline);
    }
}
