#![forbid(unsafe_code)]

pub type OscId = u8;

#[derive(Clone, Debug, PartialEq)]
pub struct TcfCfg {
    pub tick_ms: u64,
    pub hz: f32,
    pub coupling_k: f32,
    pub damping: f32,
    pub num_oscs: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Osc {
    pub phase: f32,
    pub omega: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TcfState {
    pub global_phase: f32,
    pub oscs: Vec<Osc>,
    pub last_now_ms: u64,
    pub mean_lock: f32,
    pub jitter: f32,
    pub phase_spread: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TcfTick {
    pub now_ms: u64,
    pub global_phase: f32,
    pub mean_lock: f32,
    pub jitter: f32,
    pub phase_spread: f32,
}

impl TcfCfg {
    pub fn default_gamma40() -> Self {
        Self {
            tick_ms: 10,
            hz: 40.0,
            coupling_k: 0.15,
            damping: 0.98,
            num_oscs: 6,
        }
    }
}

impl TcfState {
    pub fn new(cfg: &TcfCfg, now_ms: u64) -> Self {
        let num = cfg.num_oscs.max(1);
        let oscs = (0..num)
            .map(|i| Osc {
                phase: (i as f32) / (num as f32),
                omega: cfg.hz,
            })
            .collect::<Vec<_>>();
        Self {
            global_phase: 0.0,
            oscs,
            last_now_ms: now_ms,
            mean_lock: 0.0,
            jitter: 0.0,
            phase_spread: 1.0,
        }
    }
}

pub fn frac(x: f32) -> f32 {
    x - x.floor()
}

pub fn wrap_diff(a: f32, b: f32) -> f32 {
    let mut d = frac(a) - frac(b);
    if d > 0.5 {
        d -= 1.0;
    } else if d < -0.5 {
        d += 1.0;
    }
    d
}

pub fn tcf_tick(
    cfg: &TcfCfg,
    st: &mut TcfState,
    now_ms: u64,
    coupling_targets: &[(OscId, f32)],
) -> TcfTick {
    let prev_global_phase = st.global_phase;
    let dt = (now_ms.saturating_sub(st.last_now_ms) as f32 / 1000.0).clamp(0.0, 0.1);
    st.last_now_ms = now_ms;

    st.global_phase = frac(st.global_phase + cfg.hz * dt);

    for osc in &mut st.oscs {
        let d = wrap_diff(osc.phase, st.global_phase);
        osc.phase += dt * osc.omega + cfg.coupling_k * (-d) * dt;
        osc.phase = frac(osc.phase * cfg.damping + st.global_phase * (1.0 - cfg.damping));
    }

    for (id, target_phase) in coupling_targets {
        if let Some(osc) = st.oscs.get_mut(*id as usize) {
            let d = wrap_diff(osc.phase, *target_phase);
            osc.phase = frac(osc.phase + cfg.coupling_k * 0.5 * (-d) * dt);
        }
    }

    let spread = if st.oscs.is_empty() {
        0.0
    } else {
        st.oscs
            .iter()
            .map(|osc| wrap_diff(osc.phase, st.global_phase).abs())
            .sum::<f32>()
            / st.oscs.len() as f32
    };
    st.phase_spread = (spread * 2.0).clamp(0.0, 1.0);
    st.mean_lock = (1.0 - st.phase_spread).clamp(0.0, 1.0);
    st.jitter = (wrap_diff(st.global_phase, prev_global_phase).abs() * 1000.0).max(0.0);

    TcfTick {
        now_ms,
        global_phase: st.global_phase,
        mean_lock: st.mean_lock,
        jitter: st.jitter,
        phase_spread: st.phase_spread,
    }
}

#[cfg(test)]
mod tests {
    use super::{tcf_tick, TcfCfg, TcfState};

    #[test]
    fn determinism_for_equal_inputs() {
        let cfg = TcfCfg::default_gamma40();
        let seq = [0_u64, 10, 20, 30, 40, 55, 70, 80, 95];
        let mut a = TcfState::new(&cfg, seq[0]);
        let mut b = TcfState::new(&cfg, seq[0]);

        for now in seq {
            let ta = tcf_tick(&cfg, &mut a, now, &[(0, 0.25), (1, 0.75)]);
            let tb = tcf_tick(&cfg, &mut b, now, &[(0, 0.25), (1, 0.75)]);
            assert_eq!(ta, tb);
        }
    }

    #[test]
    fn mean_lock_always_bounded() {
        let cfg = TcfCfg::default_gamma40();
        let mut st = TcfState::new(&cfg, 0);
        for i in 1..=200 {
            let tick = tcf_tick(&cfg, &mut st, i * 10, &[]);
            assert!((0.0..=1.0).contains(&tick.mean_lock));
        }
    }

    #[test]
    fn lock_converges_up_without_targets() {
        let cfg = TcfCfg::default_gamma40();
        let mut st = TcfState::new(&cfg, 0);
        let mut first_spread = None;
        let mut min_spread: f32 = 1.0;
        for i in 1..=50 {
            let tick = tcf_tick(&cfg, &mut st, i * 10, &[]);
            if first_spread.is_none() {
                first_spread = Some(tick.phase_spread);
            }
            min_spread = min_spread.min(tick.phase_spread);
        }
        assert!(min_spread < first_spread.expect("first spread should exist"));
    }
}
