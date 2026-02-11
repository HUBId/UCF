#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SleFrame {
    pub now_ms: u64,
    pub fired: u8,
    pub reason: u8,
    pub depth: u8,
    pub weight_q: u8,
    pub tok_n: u8,
}
