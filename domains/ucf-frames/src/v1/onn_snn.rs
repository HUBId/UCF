#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnnFrame {
    pub now_ms: u64,
    pub global_phase: f32,
    pub mean_lock: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnnFrame {
    pub now_ms: u64,
    pub spikes: u32,
    pub feature: u32,
    pub causal: u32,
    pub verify: u32,
    pub attention: u32,
}
