pub const TAU: f32 = core::f32::consts::PI * 2.0;

pub fn wrap_phase(mut x: f32) -> f32 {
    x %= TAU;
    if x < 0.0 {
        x += TAU;
    }
    x
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Osc {
    pub phase: f32,
    pub omega_hz: f32,
}

impl Default for Osc {
    fn default() -> Self {
        Self {
            phase: 0.0,
            omega_hz: 40.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseCfg {
    pub coupling: f32,
    pub omega_hz_min: f32,
    pub omega_hz_max: f32,
}

impl Default for PhaseCfg {
    fn default() -> Self {
        Self {
            coupling: 0.15,
            omega_hz_min: 1.0,
            omega_hz_max: 120.0,
        }
    }
}

pub fn osc_step(self_osc: Osc, dt_s: f32) -> Osc {
    Osc {
        phase: wrap_phase(self_osc.phase + TAU * self_osc.omega_hz * dt_s),
        omega_hz: self_osc.omega_hz,
    }
}

pub fn couple_pair(a: Osc, b: Osc, dt_s: f32, cfg: PhaseCfg) -> (Osc, Osc) {
    let dphi = cfg.coupling * (b.phase - a.phase).sin();
    (
        Osc {
            phase: wrap_phase(a.phase + dphi * dt_s),
            omega_hz: a.omega_hz,
        },
        Osc {
            phase: wrap_phase(b.phase - dphi * dt_s),
            omega_hz: b.omega_hz,
        },
    )
}

pub fn phase_lock(a: Osc, b: Osc) -> f32 {
    let delta = (a.phase - b.phase).abs();
    let dist = delta.min(TAU - delta);
    (1.0 - (dist / core::f32::consts::PI)).clamp(0.0, 1.0)
}
