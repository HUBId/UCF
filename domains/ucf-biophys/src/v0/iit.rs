#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoherenceState {
    Stable,
    Drifting,
    Fragmenting,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IITInputs {
    pub lock_nsr_jepa: f32,
    pub lock_micro_nsr: f32,
    pub spike_rate_hz: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IITCfg {
    pub stable_th: f32,
    pub drifting_th: f32,
    pub spike_rate_norm_hz: f32,
    pub ema_alpha: f32,
}

impl Default for IITCfg {
    fn default() -> Self {
        Self {
            stable_th: 0.75,
            drifting_th: 0.55,
            spike_rate_norm_hz: 20.0,
            ema_alpha: 0.15,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IITState {
    pub integration_ema: f32,
}

impl Default for IITState {
    fn default() -> Self {
        Self {
            integration_ema: 0.0,
        }
    }
}

pub fn compute_integration(raw: IITInputs, cfg: IITCfg) -> f32 {
    let lock_mean = (0.5 * (raw.lock_nsr_jepa + raw.lock_micro_nsr)).clamp(0.0, 1.0);
    let spike_bonus = (raw.spike_rate_hz / cfg.spike_rate_norm_hz).clamp(0.0, 1.0) * 0.1;
    (lock_mean + spike_bonus).clamp(0.0, 1.0)
}

pub fn classify(integration: f32, cfg: IITCfg) -> CoherenceState {
    if integration >= cfg.stable_th {
        CoherenceState::Stable
    } else if integration >= cfg.drifting_th {
        CoherenceState::Drifting
    } else {
        CoherenceState::Fragmenting
    }
}

impl IITState {
    pub fn step(&mut self, raw: IITInputs, cfg: IITCfg) -> (f32, CoherenceState) {
        let integration = compute_integration(raw, cfg);
        let alpha = cfg.ema_alpha.clamp(0.0, 1.0);
        self.integration_ema = (1.0 - alpha) * self.integration_ema + alpha * integration;
        let state = classify(self.integration_ema, cfg);
        (self.integration_ema, state)
    }
}
