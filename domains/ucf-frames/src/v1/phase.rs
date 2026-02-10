#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseFrame {
    pub now_ms: u64,
    pub jepa_phase: f32,
    pub nsr_phase: f32,
    pub micro_phase: f32,
    pub lock_nsr_jepa: f32,
    pub lock_micro_nsr: f32,
}
