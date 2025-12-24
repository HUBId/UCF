#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NeuronId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SynapseId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompartmentId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifParams {
    pub tau_ms: u16,
    pub v_rest: i32,
    pub v_reset: i32,
    pub v_threshold: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifState {
    pub v: i32,
    pub refractory_steps: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopCode {
    pub spikes: Vec<NeuronId>,
}

pub const DEFAULT_REFRACTORY_STEPS: u16 = 0;

pub fn clamp_i32(value: i32, min: i32, max: i32) -> i32 {
    value.max(min).min(max)
}

pub fn clamp_usize(value: usize, max: usize) -> usize {
    if value > max {
        max
    } else {
        value
    }
}
