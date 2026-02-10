#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NsrFrame {
    pub now_ms: u64,
    pub verdict: u8,
    pub verified_ratio: f32,
}
