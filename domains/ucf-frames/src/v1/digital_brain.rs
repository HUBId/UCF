#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChemFrame {
    pub now_ms: u64,
    pub dopa_q: u8,
    pub s5ht_q: u8,
    pub oxy_q: u8,
    pub end_q: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrainFrame {
    pub now_ms: u64,
    pub amyg_spikes: u16,
    pub pfc_spikes: u16,
    pub amyg_avg_v_q: i16,
    pub pfc_avg_v_q: i16,
}

pub fn quantize_hormone(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn quantize_avg_v_mv(v_mv: f32) -> i16 {
    (v_mv.clamp(-100.0, 50.0) * 10.0).round() as i16
}
