#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcdeFrame {
    pub now_ms: u64,
    pub l2_q: u8,
    pub phase_q: u8,
}
