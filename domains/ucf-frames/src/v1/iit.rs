#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IitFrame {
    pub now_ms: u64,
    pub phi_q: u8,
    pub coh_q: u8,
    pub flow_q: u8,
    pub enforce: u8,
}
