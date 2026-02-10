#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicrocircuitFrame {
    pub now_ms: u64,
    pub n: u32,
    pub spike_count: u32,
    pub avg_v: f32,
}
