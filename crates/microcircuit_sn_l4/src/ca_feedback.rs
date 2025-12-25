#![forbid(unsafe_code)]

use biophys_synapses_l4::{clamp_inhibitory_boost_q, INHIBITORY_BOOST_SCALE_Q};

pub const MAX_HOLD_BIAS_STEPS: u8 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaFeedbackPolicy {
    pub enabled: bool,
    pub gaba_boost_q: u16,
    pub hold_bias_steps: u8,
}

impl CaFeedbackPolicy {
    pub fn normalized(self) -> Self {
        Self {
            enabled: self.enabled,
            gaba_boost_q: clamp_inhibitory_boost_q(self.gaba_boost_q),
            hold_bias_steps: self.hold_bias_steps.min(MAX_HOLD_BIAS_STEPS),
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            gaba_boost_q: INHIBITORY_BOOST_SCALE_Q,
            hold_bias_steps: 0,
        }
    }
}

impl Default for CaFeedbackPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            gaba_boost_q: 1100,
            hold_bias_steps: 2,
        }
    }
}
