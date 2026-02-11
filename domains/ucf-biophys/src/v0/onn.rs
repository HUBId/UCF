use crate::v0::phase::TAU;

pub type OscId = u16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnnCfg {
    pub omega_hz: f32,
    pub coupling: f32,
    pub damping: f32,
    pub lock_eps: f32,
}

impl Default for OnnCfg {
    fn default() -> Self {
        Self {
            omega_hz: 40.0,
            coupling: 0.12,
            damping: 0.02,
            lock_eps: 0.35,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnnState {
    pub global_phase: f32,
    pub osc: Vec<(OscId, f32)>,
}

impl Default for OnnState {
    fn default() -> Self {
        Self {
            global_phase: 0.0,
            osc: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnnOut {
    pub global_phase: f32,
    pub locks: Vec<(OscId, OscId, f32)>,
    pub mean_lock: f32,
}

pub fn ensure_osc(state: &mut OnnState, id: OscId) {
    if !state.osc.iter().any(|(oid, _)| *oid == id) {
        state.osc.push((id, state.global_phase));
    }
}

pub fn phase_diff_abs(a: f32, b: f32) -> f32 {
    let delta = (wrap_phase(a) - wrap_phase(b)).abs();
    delta.min(TAU - delta)
}

pub fn phase_bin(phase: f32, bins: u8) -> u8 {
    if bins <= 1 {
        return 0;
    }
    let wrapped = wrap_phase(phase);
    let ratio = wrapped / TAU;
    let idx = (ratio * bins as f32).floor() as i32;
    idx.clamp(0, bins as i32 - 1) as u8
}

pub fn step(
    state: &mut OnnState,
    dt_ms: u32,
    cfg: OnnCfg,
    couplings: &[(OscId, OscId, f32)],
) -> OnnOut {
    let dt_s = dt_ms as f32 / 1000.0;
    let advance = TAU * cfg.omega_hz * dt_s;
    state.global_phase = wrap_phase(state.global_phase + advance);

    let prev = state.osc.clone();
    for (id_i, phase_i) in &mut state.osc {
        let mut coupled_sum = 0.0;
        for (a, b, kij) in couplings {
            if *a == *id_i {
                if let Some((_, phase_j)) = prev.iter().find(|(id, _)| id == b) {
                    coupled_sum += *kij * (*phase_j - *phase_i).sin();
                }
            }
        }

        let damping = -cfg.damping * (*phase_i - state.global_phase).sin();
        *phase_i = wrap_phase(*phase_i + advance + cfg.coupling * coupled_sum + damping);
    }

    let mut locks = Vec::with_capacity(couplings.len());
    let mut lock_sum = 0.0;
    let mut count = 0_usize;
    for (a, b, _) in couplings {
        let phase_a = state.osc.iter().find(|(id, _)| id == a).map(|(_, p)| *p);
        let phase_b = state.osc.iter().find(|(id, _)| id == b).map(|(_, p)| *p);
        if let (Some(pa), Some(pb)) = (phase_a, phase_b) {
            let diff = phase_diff_abs(pa, pb);
            locks.push((*a, *b, diff));
            lock_sum += (1.0 - diff / core::f32::consts::PI).clamp(0.0, 1.0);
            count += 1;
        }
    }

    let mean_lock = if count == 0 {
        1.0
    } else {
        lock_sum / count as f32
    };
    OnnOut {
        global_phase: state.global_phase,
        locks,
        mean_lock,
    }
}

fn wrap_phase(mut x: f32) -> f32 {
    x %= TAU;
    if x < 0.0 {
        x += TAU;
    }
    x
}
