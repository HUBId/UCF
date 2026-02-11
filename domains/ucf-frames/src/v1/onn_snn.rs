#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnnFrame {
    pub now_ms: u64,
    pub global_phase_q: u8,
    pub lock_nsr_cde_q: u8,
    pub lock_nsr_ssm_q: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnnFrame {
    pub now_ms: u64,
    pub fired: u16,
    pub suppressed: u16,
    pub max_amp_q: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeFrame {
    pub now_ms: u64,
    pub kind: u8,
    pub chan: u8,
    pub phase: u8,
    pub strength_q: u8,
    pub ttfs_q: u8,
}
