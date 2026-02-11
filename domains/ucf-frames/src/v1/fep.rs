#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FepFrame {
    pub now_ms: u64,
    pub attention_q: u8,
    pub learn_gate_q: u8,
    pub memprio_q: u8,
    pub inhibit_q: u8,
    pub confidence_q: u8,
    pub homeo_err_q: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoherenceFrame {
    pub now_ms: u64,
    pub coupling_q: u8,
    pub drift_q: u8,
    pub risk_q: u8,
    pub lock_q: u8,
}
