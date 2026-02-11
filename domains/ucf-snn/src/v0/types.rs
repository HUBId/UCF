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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpikeSrc {
    Sae,
    Lens,
    Cde,
    Nsr,
    Ssm,
    Iit,
    Sle,
    Ncde,
    External,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpikeDst {
    Nsr,
    Cde,
    Ssm,
    Rsa,
    Policy,
    Global,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpikeEvent {
    pub now_ms: u64,
    pub src: SpikeSrc,
    pub dst: SpikeDst,
    pub feature_id: u32,
    pub phase_q: u8,
    pub ttfsp_q: u8,
    pub amp_q: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SnnCfg {
    pub ttfsp_window_ms: u32,
    pub min_fire_q: u8,
    pub route_to_nsr: bool,
    pub route_to_cde: bool,
    pub route_to_ssm: bool,
    pub max_events_per_tick: usize,
}

impl SnnCfg {
    pub fn default_v0() -> Self {
        Self {
            ttfsp_window_ms: 25,
            min_fire_q: 32,
            route_to_nsr: true,
            route_to_cde: true,
            route_to_ssm: true,
            max_events_per_tick: 64,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SnnOut {
    pub emitted: Vec<SpikeEvent>,
    pub fired_count: u16,
    pub suppressed_count: u16,
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
