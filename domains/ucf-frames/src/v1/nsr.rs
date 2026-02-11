#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NsrFrame {
    pub now_ms: u64,
    pub verdict: u8,
    pub satisfied: u16,
    pub total: u16,
    pub verified_q: u8,
}
