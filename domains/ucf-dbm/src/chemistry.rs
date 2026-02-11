#[derive(Clone, Debug, PartialEq)]
pub struct NeuromodState {
    pub dopa: f32,
    pub serotonin: f32,
    pub oxytocin: f32,
    pub endorphin: f32,
}

impl NeuromodState {
    pub fn baseline() -> Self {
        Self {
            dopa: 0.2,
            serotonin: 0.2,
            oxytocin: 0.2,
            endorphin: 0.2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChemistryCfg {
    pub tau_dopa_s: f32,
    pub tau_5ht_s: f32,
    pub tau_oxy_s: f32,
    pub tau_end_s: f32,
    pub prod_gain: f32,
    pub stress_to_5ht: f32,
    pub reward_to_dopa: f32,
    pub safety_to_oxy: f32,
    pub pain_to_end: f32,
}

impl ChemistryCfg {
    pub fn default_v0() -> Self {
        Self {
            tau_dopa_s: 8.0,
            tau_5ht_s: 10.0,
            tau_oxy_s: 12.0,
            tau_end_s: 9.0,
            prod_gain: 0.8,
            stress_to_5ht: 0.9,
            reward_to_dopa: 1.0,
            safety_to_oxy: 0.8,
            pain_to_end: 1.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn chemistry_step(
    dt_s: f32,
    st: &mut NeuromodState,
    reward: f32,
    stress: f32,
    safety: f32,
    pain: f32,
    homeo_err: f32,
    cfg: &ChemistryCfg,
) {
    let dt = dt_s.max(0.000_1);
    let reward = reward.clamp(0.0, 1.0);
    let stress = stress.clamp(0.0, 1.0);
    let safety = safety.clamp(0.0, 1.0);
    let pain = pain.clamp(0.0, 1.0);
    let homeo_err = homeo_err.max(0.0);

    let dopa_prod =
        cfg.prod_gain * (cfg.reward_to_dopa * reward - 0.15 * stress) - 0.10 * homeo_err;
    let ht5_prod = cfg.prod_gain * (cfg.stress_to_5ht * stress - 0.10 * reward) + 0.12 * safety
        - 0.08 * homeo_err;
    let oxy_prod = cfg.prod_gain * (cfg.safety_to_oxy * safety - 0.10 * stress);
    let end_prod = cfg.prod_gain * (cfg.pain_to_end * pain + 0.10 * safety - 0.10 * reward);

    let dopa_decay = st.dopa / cfg.tau_dopa_s.max(0.1);
    let ht5_decay = st.serotonin / cfg.tau_5ht_s.max(0.1);
    let oxy_decay = st.oxytocin / cfg.tau_oxy_s.max(0.1);
    let end_decay = st.endorphin / cfg.tau_end_s.max(0.1);

    st.dopa = (st.dopa + dt * (dopa_prod - dopa_decay)).clamp(0.0, 1.0);
    st.serotonin = (st.serotonin + dt * (ht5_prod - ht5_decay)).clamp(0.0, 1.0);
    st.oxytocin = (st.oxytocin + dt * (oxy_prod - oxy_decay)).clamp(0.0, 1.0);
    st.endorphin = (st.endorphin + dt * (end_prod - end_decay)).clamp(0.0, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chemistry_reward_increases_dopa() {
        let cfg = ChemistryCfg::default_v0();
        let mut st = NeuromodState::baseline();
        let d0 = st.dopa;
        for _ in 0..100 {
            chemistry_step(0.05, &mut st, 1.0, 0.0, 0.2, 0.0, 0.0, &cfg);
        }
        assert!(st.dopa > d0);
    }

    #[test]
    fn chemistry_stress_increases_serotonin_and_reduces_dopa() {
        let cfg = ChemistryCfg::default_v0();
        let mut st = NeuromodState::baseline();
        let d0 = st.dopa;
        let s0 = st.serotonin;
        for _ in 0..100 {
            chemistry_step(0.05, &mut st, 0.0, 1.0, 0.0, 0.0, 0.0, &cfg);
        }
        assert!(st.serotonin > s0);
        assert!(st.dopa < d0);
    }

    #[test]
    fn chemistry_clamps_0_1() {
        let cfg = ChemistryCfg::default_v0();
        let mut st = NeuromodState {
            dopa: 2.0,
            serotonin: -1.0,
            oxytocin: 3.0,
            endorphin: -2.0,
        };
        chemistry_step(0.1, &mut st, 10.0, 10.0, 10.0, 10.0, 0.0, &cfg);
        assert!((0.0..=1.0).contains(&st.dopa));
        assert!((0.0..=1.0).contains(&st.serotonin));
        assert!((0.0..=1.0).contains(&st.oxytocin));
        assert!((0.0..=1.0).contains(&st.endorphin));
    }
}
