use crate::v0::modulation::HhParams;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HHState {
    pub v_mv: f32,
    pub m: f32,
    pub h: f32,
    pub n: f32,
}

impl Default for HHState {
    fn default() -> Self {
        Self {
            v_mv: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HHNeuron {
    pub state: HHState,
    pub params: HhParams,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HhStepIn {
    pub i_ext: f32,
    pub syn_i: f32,
    pub dt_s: f32,
    pub threshold_mv: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HhStepOut {
    pub spiked: bool,
    pub v_mv: f32,
    pub spike_latency_s: Option<f32>,
}

impl HHNeuron {
    pub fn step_stub(&mut self, input: HhStepIn) -> HhStepOut {
        self.state.v_mv += (input.i_ext + input.syn_i) * input.dt_s * 10.0;
        self.state.v_mv += (-65.0 - self.state.v_mv) * input.dt_s * 2.0;

        // TODO: Replace with real Hodgkin-Huxley gating ODE integration for m/h/n.
        let spiked = self.state.v_mv >= input.threshold_mv;
        if spiked {
            self.state.v_mv = -70.0;
        }

        HhStepOut {
            spiked,
            v_mv: self.state.v_mv,
            spike_latency_s: None,
        }
    }
}
