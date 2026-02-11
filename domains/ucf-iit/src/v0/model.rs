#[derive(Clone, Debug, PartialEq)]
pub struct IitCfg {
    pub window: usize,
    pub min_samples: usize,
    pub coherence_weight: f32,
    pub flow_weight: f32,
    pub enforce_threshold: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IitSample {
    pub now_ms: u64,
    pub tcf_lock: f32,
    pub ssm_gate: f32,
    pub nsr_risk: f32,
    pub cde_conf: f32,
    pub spike_rate: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IitState {
    pub ring: Vec<IitSample>,
    pub idx: usize,
    pub filled: usize,
    pub phi_proxy: f32,
    pub coherence: f32,
    pub flow: f32,
    pub enforce: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IitTick {
    pub now_ms: u64,
    pub phi_proxy: f32,
    pub coherence: f32,
    pub flow: f32,
    pub enforce: bool,
}

impl IitCfg {
    pub fn default_v0() -> Self {
        Self {
            window: 32,
            min_samples: 8,
            coherence_weight: 0.5,
            flow_weight: 0.5,
            enforce_threshold: 0.35,
        }
    }
}

impl IitState {
    pub fn new(cfg: &IitCfg) -> Self {
        let window = cfg.window.max(1);
        Self {
            ring: vec![
                IitSample {
                    now_ms: 0,
                    tcf_lock: 0.0,
                    ssm_gate: 0.0,
                    nsr_risk: 0.0,
                    cde_conf: 0.0,
                    spike_rate: 0.0,
                };
                window
            ],
            idx: 0,
            filled: 0,
            phi_proxy: 0.0,
            coherence: 0.0,
            flow: 0.0,
            enforce: false,
        }
    }
}

pub fn iit_push_and_eval(cfg: &IitCfg, st: &mut IitState, s: IitSample) -> IitTick {
    if st.ring.is_empty() {
        st.ring.push(s.clone());
    }
    let pos = st.idx % st.ring.len();
    st.ring[pos] = s.clone();
    st.idx = (st.idx + 1) % st.ring.len();
    st.filled = st.filled.saturating_add(1).min(st.ring.len());

    if st.filled < cfg.min_samples {
        st.phi_proxy = 0.0;
        st.coherence = 0.0;
        st.flow = 0.0;
        st.enforce = false;
        return IitTick {
            now_ms: s.now_ms,
            phi_proxy: st.phi_proxy,
            coherence: st.coherence,
            flow: st.flow,
            enforce: st.enforce,
        };
    }

    let filled = st.filled as f32;
    let samples = &st.ring[..st.filled];
    st.coherence = (samples
        .iter()
        .map(|it| it.tcf_lock.clamp(0.0, 1.0))
        .sum::<f32>()
        / filled)
        .clamp(0.0, 1.0);

    let flow_sum = samples
        .iter()
        .map(|it| {
            let term_ssm_nsr =
                1.0 - (it.ssm_gate.clamp(0.0, 1.0) - (1.0 - it.nsr_risk.clamp(0.0, 1.0))).abs();
            let term_cde_tcf =
                1.0 - (it.cde_conf.clamp(0.0, 1.0) - it.tcf_lock.clamp(0.0, 1.0)).abs();
            let term_spike_nsr =
                1.0 - (it.spike_rate.clamp(0.0, 1.0) - (1.0 - it.nsr_risk.clamp(0.0, 1.0))).abs();
            ((term_ssm_nsr + term_cde_tcf + term_spike_nsr) / 3.0).clamp(0.0, 1.0)
        })
        .sum::<f32>();
    st.flow = (flow_sum / filled).clamp(0.0, 1.0);

    st.phi_proxy =
        (cfg.coherence_weight * st.coherence + cfg.flow_weight * st.flow).clamp(0.0, 1.0);
    st.enforce = st.phi_proxy < cfg.enforce_threshold.clamp(0.0, 1.0);

    IitTick {
        now_ms: s.now_ms,
        phi_proxy: st.phi_proxy,
        coherence: st.coherence,
        flow: st.flow,
        enforce: st.enforce,
    }
}
