#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CdeFrame {
    pub now_ms: u64,
    pub hyps: u16,
    pub changed: u16,
    pub pruned: u16,
    pub top_conf_q: u8,
}
