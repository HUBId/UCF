use crate::v0::{field::Unit01, ode::clamp01, prod_clear_step};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HpaState {
    pub crh: f32,
    pub acth: f32,
    pub cortisol: f32,
}

impl Default for HpaState {
    fn default() -> Self {
        Self {
            crh: 0.1,
            acth: 0.1,
            cortisol: 0.1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HpaCfg {
    pub crh_clear: f32,
    pub acth_clear: f32,
    pub cortisol_clear: f32,
    pub stress_gain: f32,
    pub crh_to_acth_gain: f32,
    pub acth_to_cort_gain: f32,
    pub cortisol_feedback_gain: f32,
}

impl Default for HpaCfg {
    fn default() -> Self {
        Self {
            crh_clear: 0.4,
            acth_clear: 0.35,
            cortisol_clear: 0.25,
            stress_gain: 1.0,
            crh_to_acth_gain: 0.95,
            acth_to_cort_gain: 1.05,
            cortisol_feedback_gain: 0.9,
        }
    }
}

pub fn hpa_step(st: HpaState, stress: f32, dt_s: f32, cfg: HpaCfg) -> HpaState {
    let stress = clamp01(stress);
    let crh_prod = (cfg.stress_gain * stress - cfg.cortisol_feedback_gain * st.cortisol).max(0.0);
    let acth_prod = cfg.crh_to_acth_gain * st.crh;
    let cort_prod = cfg.acth_to_cort_gain * st.acth;

    HpaState {
        crh: prod_clear_step(clamp01(st.crh), crh_prod, cfg.crh_clear, dt_s),
        acth: prod_clear_step(clamp01(st.acth), acth_prod, cfg.acth_clear, dt_s),
        cortisol: prod_clear_step(clamp01(st.cortisol), cort_prod, cfg.cortisol_clear, dt_s),
    }
}

pub fn cortisol_unit(st: HpaState) -> Unit01 {
    Unit01::new(st.cortisol)
}
