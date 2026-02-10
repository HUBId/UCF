use crate::v0::NeuromodulatorField;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HhParams {
    pub g_na: f32,
    pub g_k: f32,
    pub g_l: f32,
    pub threshold_shift_mv: f32,
    pub max_firing_hz: f32,
}

impl Default for HhParams {
    fn default() -> Self {
        Self {
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            threshold_shift_mv: 0.0,
            max_firing_hz: 200.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModulationCfg {
    pub dopamine_gna_gain: f32,
    pub serotonin_gk_gain: f32,
    pub oxytocin_gaba_bias: f32,
    pub endorphin_firing_cap: f32,
    pub gaba_threshold_mv: f32,
    pub glutamate_threshold_mv: f32,
}

impl Default for ModulationCfg {
    fn default() -> Self {
        Self {
            dopamine_gna_gain: 0.25,
            serotonin_gk_gain: 0.25,
            oxytocin_gaba_bias: 0.20,
            endorphin_firing_cap: -0.30,
            gaba_threshold_mv: 5.0,
            glutamate_threshold_mv: -3.0,
        }
    }
}

pub fn modulate_hh(base: HhParams, field: NeuromodulatorField, cfg: ModulationCfg) -> HhParams {
    // Dopamine increases sodium conductance.
    let g_na = base.g_na * (1.0 + field.dopamine.get() * cfg.dopamine_gna_gain);
    // Serotonin increases potassium conductance.
    let g_k = base.g_k * (1.0 + field.serotonin.get() * cfg.serotonin_gk_gain);
    // GABA raises threshold (inhibitory).
    let gaba_shift = field.gaba.get() * cfg.gaba_threshold_mv;
    // Glutamate lowers threshold via negative default gain.
    let glutamate_shift = field.glutamate.get() * cfg.glutamate_threshold_mv;
    // Oxytocin biases inhibitory stability as an additional threshold raise.
    let oxytocin_shift = field.oxytocin.get() * cfg.oxytocin_gaba_bias * cfg.gaba_threshold_mv;
    let threshold_shift_mv =
        base.threshold_shift_mv + gaba_shift + glutamate_shift + oxytocin_shift;

    // Endorphin reduces firing cap with a floor at 1 Hz.
    let cap_scale = (1.0 - field.endorphin.get() * cfg.endorphin_firing_cap.abs()).max(0.0);
    let max_firing_hz = (base.max_firing_hz * cap_scale).max(1.0);

    HhParams {
        g_na,
        g_k,
        g_l: base.g_l,
        threshold_shift_mv,
        max_firing_hz,
    }
}

pub fn summarize(field: NeuromodulatorField) -> [f32; 7] {
    [
        field.dopamine.get(),
        field.serotonin.get(),
        field.oxytocin.get(),
        field.endorphin.get(),
        field.glutamate.get(),
        field.gaba.get(),
        field.acetylcholine.get(),
    ]
}
