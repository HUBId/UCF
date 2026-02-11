#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcfFrame {
    pub now_ms: u64,
    pub phase_bin: u8,
    pub lock_q: u8,
    pub jitter_q: u8,
    pub spread_q: u8,
}

impl TcfFrame {
    pub fn from_metrics(
        now_ms: u64,
        global_phase: f32,
        mean_lock: f32,
        jitter: f32,
        phase_spread: f32,
    ) -> Self {
        let phase_bin = (global_phase.clamp(0.0, 1.0) * 255.0).round() as u8;
        let lock_q = (mean_lock.clamp(0.0, 1.0) * 255.0).round() as u8;
        let jitter_q = ((jitter.max(0.0) / 10.0).clamp(0.0, 1.0) * 255.0).round() as u8;
        let spread_q = (phase_spread.clamp(0.0, 1.0) * 255.0).round() as u8;
        Self {
            now_ms,
            phase_bin,
            lock_q,
            jitter_q,
            spread_q,
        }
    }
}
