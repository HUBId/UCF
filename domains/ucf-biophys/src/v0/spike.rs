use crate::v0::phase::{wrap_phase, TAU};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpikeEvent {
    pub neuron: u32,
    pub t_ms: u64,
    pub phase: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpikeCodecCfg {
    pub window_ms: u64,
    pub min_phase: f32,
    pub max_phase: f32,
}

impl Default for SpikeCodecCfg {
    fn default() -> Self {
        Self {
            window_ms: 25,
            min_phase: 0.0,
            max_phase: TAU,
        }
    }
}

pub fn ttfs_phase(latency_ms: u64, cfg: SpikeCodecCfg) -> f32 {
    let clamped = latency_ms.min(cfg.window_ms);
    let ratio = if cfg.window_ms == 0 {
        0.0
    } else {
        clamped as f32 / cfg.window_ms as f32
    };
    let phase = cfg.min_phase + ratio * (cfg.max_phase - cfg.min_phase);
    wrap_phase(phase)
}

pub fn spikes_from_ids(now_ms: u64, spike_ids: &[u32], base_phase: f32) -> Vec<SpikeEvent> {
    spike_ids
        .iter()
        .map(|neuron| SpikeEvent {
            neuron: *neuron,
            t_ms: now_ms,
            phase: base_phase,
        })
        .collect()
}
