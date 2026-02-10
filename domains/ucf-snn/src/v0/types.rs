use ucf_onn::v0::PhaseDeg;

pub type SpikeChan = u16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeTimeMs(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TtfsMs(pub u16);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpikePayload {
    Binary,
    Ttfs { ttfs_ms: TtfsMs },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnnSpike {
    pub chan: SpikeChan,
    pub t: SpikeTimeMs,
    pub payload: SpikePayload,
    pub phase: Option<PhaseDeg>,
}

pub(crate) fn clamp_u16(v: u16, lo: u16, hi: u16) -> u16 {
    v.max(lo).min(hi)
}

pub(crate) fn clamp01(x: f32) -> f32 {
    if x.is_nan() {
        return 0.0;
    }
    x.clamp(0.0, 1.0)
}

pub(crate) fn norm_phase_opt(p: Option<PhaseDeg>) -> Option<PhaseDeg> {
    p.map(PhaseDeg::norm)
}
