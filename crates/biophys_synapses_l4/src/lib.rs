#![forbid(unsafe_code)]

pub const FIXED_POINT_SCALE: u32 = 1 << 16;
const FIXED_POINT_SCALE_I64: i64 = 1 << 16;
const DECAY_SCALE: u32 = 1024;
const MAX_SYNAPSE_G: f32 = 1000.0;
const MAX_ACCUMULATOR_G: f32 = 5000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynKind {
    AMPA,
    NMDA,
    GABA,
}

#[derive(Debug, Clone)]
pub struct SynapseL4 {
    pub pre_neuron: u32,
    pub post_neuron: u32,
    pub post_compartment: u32,
    pub kind: SynKind,
    pub g_max: f32,
    pub e_rev: f32,
    pub tau_rise_ms: f32,
    pub tau_decay_ms: f32,
    pub delay_steps: u16,
}

impl SynapseL4 {
    pub fn g_max_fixed(&self) -> u32 {
        f32_to_fixed_u32(self.g_max)
    }

    pub fn e_rev_fixed(&self) -> i32 {
        f32_to_fixed_i32(self.e_rev)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynapseState {
    pub g_fixed: u32,
}

impl SynapseState {
    pub fn apply_spike(&mut self, synapse: &SynapseL4) {
        let g_max_fixed = synapse.g_max_fixed();
        let max_fixed = f32_to_fixed_u32(MAX_SYNAPSE_G).max(g_max_fixed);
        self.g_fixed = (self.g_fixed + g_max_fixed).min(max_fixed);
    }

    pub fn decay(&mut self, decay_k: u16) {
        if decay_k as u32 >= DECAY_SCALE {
            self.g_fixed = 0;
            return;
        }
        let decay = (self.g_fixed as u64 * decay_k as u64) / DECAY_SCALE as u64;
        self.g_fixed = self.g_fixed.saturating_sub(decay as u32);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynapseConductance {
    pub g_fixed: u32,
    pub g_e_rev_fixed: i64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynapseAccumulator {
    pub ampa: SynapseConductance,
    pub nmda: SynapseConductance,
    pub gaba: SynapseConductance,
}

impl SynapseAccumulator {
    pub fn add(&mut self, kind: SynKind, g_fixed: u32, e_rev: f32) {
        let e_rev_fixed = f32_to_fixed_i32(e_rev) as i64;
        let max_fixed = f32_to_fixed_u32(MAX_ACCUMULATOR_G);
        let target = match kind {
            SynKind::AMPA => &mut self.ampa,
            SynKind::NMDA => &mut self.nmda,
            SynKind::GABA => &mut self.gaba,
        };
        let remaining = max_fixed.saturating_sub(target.g_fixed);
        let add_g = g_fixed.min(remaining);
        if add_g == 0 {
            return;
        }
        target.g_fixed = target.g_fixed.saturating_add(add_g);
        let delta = (add_g as i64 * e_rev_fixed) >> 16;
        target.g_e_rev_fixed = target.g_e_rev_fixed.saturating_add(delta);
    }

    pub fn total_current(&self, v: f32) -> f32 {
        syn_current(self.ampa, v) + syn_current(self.nmda, v) + syn_current(self.gaba, v)
    }
}

pub fn decay_k(dt_ms: f32, tau_decay_ms: f32) -> u16 {
    if tau_decay_ms <= 0.0 {
        return DECAY_SCALE as u16;
    }
    let ratio = dt_ms / tau_decay_ms;
    let scaled = (ratio * DECAY_SCALE as f32).round();
    scaled.clamp(0.0, DECAY_SCALE as f32) as u16
}

fn syn_current(conductance: SynapseConductance, v: f32) -> f32 {
    let g = fixed_to_f32(conductance.g_fixed);
    if g == 0.0 {
        return 0.0;
    }
    let g_e_rev = fixed_to_f32_i64(conductance.g_e_rev_fixed);
    g_e_rev - g * v
}

fn f32_to_fixed_u32(value: f32) -> u32 {
    if value <= 0.0 {
        return 0;
    }
    (value * FIXED_POINT_SCALE as f32).round() as u32
}

fn f32_to_fixed_i32(value: f32) -> i32 {
    (value * FIXED_POINT_SCALE as f32).round() as i32
}

fn fixed_to_f32(value: u32) -> f32 {
    value as f32 / FIXED_POINT_SCALE as f32
}

fn fixed_to_f32_i64(value: i64) -> f32 {
    value as f32 / FIXED_POINT_SCALE_I64 as f32
}
