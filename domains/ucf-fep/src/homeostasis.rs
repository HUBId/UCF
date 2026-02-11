#[derive(Clone, Debug, PartialEq)]
pub struct HomeoCfg {
    pub target_event_rate: f32,
    pub target_brain_spike_rate: f32,
    pub k_p: f32,
    pub k_i: f32,
    pub integ_cap: f32,
    pub out_cap: f32,
}

impl HomeoCfg {
    pub fn default_v0() -> Self {
        Self {
            target_event_rate: 0.2,
            target_brain_spike_rate: 0.3,
            k_p: 0.7,
            k_i: 0.15,
            integ_cap: 1.0,
            out_cap: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HomeoState {
    pub integ: f32,
    pub last_out: f32,
}

impl HomeoState {
    pub fn new() -> Self {
        Self {
            integ: 0.0,
            last_out: 0.0,
        }
    }
}

impl Default for HomeoState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn homeostasis_step(
    cfg: &HomeoCfg,
    st: &mut HomeoState,
    dt_s: f32,
    observed_event_rate: f32,
    observed_brain_rate: f32,
    ess_pressure: f32,
) -> f32 {
    let event_rate = observed_event_rate.clamp(0.0, 1.0);
    let brain_rate = observed_brain_rate.clamp(0.0, 1.0);
    let mem_pressure = ess_pressure.clamp(0.0, 1.0);

    let err = (event_rate - cfg.target_event_rate)
        + 0.6 * (brain_rate - cfg.target_brain_spike_rate)
        + 0.7 * (mem_pressure - 0.5);

    st.integ = (st.integ + err * dt_s.max(0.0)).clamp(-cfg.integ_cap.abs(), cfg.integ_cap.abs());
    let out = (cfg.k_p * err + cfg.k_i * st.integ).clamp(0.0, cfg.out_cap.max(0.0));
    st.last_out = out;
    out
}
