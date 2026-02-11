use ucf_onn::v0::PhaseDeg;

use crate::v0::types::{
    clamp01, clamp_u16, norm_phase_opt, SnnCfg, SnnOut, SnnSpike, SpikeChan, SpikeDst, SpikeEvent,
    SpikePayload, SpikeSrc, SpikeTimeMs, TtfsMs,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureEvent {
    pub chan: SpikeChan,
    pub intensity: f32,
    pub novelty: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnnEncodeCfg {
    pub novelty_gate: f32,
    pub ttfs_max_ms: u16,
    pub use_ttfs: bool,
}

impl Default for SnnEncodeCfg {
    fn default() -> Self {
        Self {
            novelty_gate: 0.2,
            ttfs_max_ms: 100,
            use_ttfs: true,
        }
    }
}

pub fn encode(
    now_ms: u64,
    phase: Option<PhaseDeg>,
    cfg: SnnEncodeCfg,
    feats: &[FeatureEvent],
) -> Vec<SnnSpike> {
    let normalized_phase = norm_phase_opt(phase);
    let mut spikes = Vec::new();

    for feat in feats {
        if clamp01(feat.novelty) < cfg.novelty_gate {
            continue;
        }

        let (t, payload) = if cfg.use_ttfs {
            let intensity = clamp01(feat.intensity);
            let ttfs = ((1.0 - intensity) * f32::from(cfg.ttfs_max_ms)).round() as u16;
            let ttfs = clamp_u16(ttfs, 0, 1000);
            (
                SpikeTimeMs(now_ms + u64::from(ttfs)),
                SpikePayload::Ttfs {
                    ttfs_ms: TtfsMs(ttfs),
                },
            )
        } else {
            (SpikeTimeMs(now_ms), SpikePayload::Binary)
        };

        spikes.push(SnnSpike {
            chan: feat.chan,
            t,
            payload,
            phase: normalized_phase,
        });
    }

    spikes.sort_by_key(|s| (s.t.0, s.chan));
    spikes
}

pub fn encode_ttfsp(phase_0_1: f32, intensity_0_1: f32) -> u8 {
    encode_ttfsp_with_window(phase_0_1, intensity_0_1, 25)
}

fn encode_ttfsp_with_window(phase_0_1: f32, intensity_0_1: f32, ttfsp_window_ms: u32) -> u8 {
    let phase = clamp01(phase_0_1);
    let intensity = clamp01(intensity_0_1);
    let window = (ttfsp_window_ms.max(1)) as f32;
    let phase_bias = 0.15 * (1.0 - phase);
    let tt_ms = (1.0 - intensity) * window * (1.0 - phase_bias);
    ((tt_ms / window).clamp(0.0, 1.0) * 255.0).round() as u8
}

fn route_for(src: SpikeSrc, intensity: f32, cfg: &SnnCfg) -> Option<SpikeDst> {
    match src {
        SpikeSrc::Sae | SpikeSrc::Lens => {
            if intensity >= 0.65 && cfg.route_to_nsr {
                Some(SpikeDst::Nsr)
            } else if intensity >= 0.45 && cfg.route_to_cde {
                Some(SpikeDst::Cde)
            } else if intensity >= 0.30 && cfg.route_to_ssm {
                Some(SpikeDst::Ssm)
            } else {
                None
            }
        }
        SpikeSrc::Cde => {
            if intensity >= 0.40 && cfg.route_to_nsr {
                Some(SpikeDst::Nsr)
            } else {
                None
            }
        }
        SpikeSrc::Nsr => {
            if intensity >= 0.30 && cfg.route_to_cde {
                Some(SpikeDst::Cde)
            } else {
                None
            }
        }
        SpikeSrc::Ssm => {
            if intensity >= 0.30 && cfg.route_to_global() {
                Some(SpikeDst::Global)
            } else {
                None
            }
        }
        _ => None,
    }
}

trait GlobalRoute {
    fn route_to_global(&self) -> bool;
}

impl GlobalRoute for SnnCfg {
    fn route_to_global(&self) -> bool {
        self.route_to_nsr || self.route_to_cde || self.route_to_ssm
    }
}

pub fn snn_emit(
    cfg: &SnnCfg,
    now_ms: u64,
    phase_0_1: f32,
    src: SpikeSrc,
    candidates: &[(u32, f32)],
) -> SnnOut {
    let phase_q = (clamp01(phase_0_1) * 255.0).round() as u8;
    let mut suppressed_count = 0_u16;
    let mut selected: Vec<(f32, SpikeEvent)> = Vec::new();

    for (feature_id, intensity_raw) in candidates {
        let intensity = clamp01(*intensity_raw);
        let amp_q = (intensity * 255.0).round() as u8;
        if amp_q < cfg.min_fire_q {
            suppressed_count = suppressed_count.saturating_add(1);
            continue;
        }

        let Some(dst) = route_for(src, intensity, cfg) else {
            suppressed_count = suppressed_count.saturating_add(1);
            continue;
        };

        selected.push((
            intensity,
            SpikeEvent {
                now_ms,
                src,
                dst,
                feature_id: *feature_id,
                phase_q,
                ttfsp_q: encode_ttfsp_with_window(phase_0_1, intensity, cfg.ttfsp_window_ms),
                amp_q,
            },
        ));
    }

    selected.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.feature_id.cmp(&b.1.feature_id))
    });
    if selected.len() > cfg.max_events_per_tick {
        suppressed_count = suppressed_count.saturating_add(
            (selected.len() - cfg.max_events_per_tick).min(u16::MAX as usize) as u16,
        );
        selected.truncate(cfg.max_events_per_tick);
    }

    let emitted = selected.into_iter().map(|(_, evt)| evt).collect::<Vec<_>>();
    SnnOut {
        fired_count: emitted.len().min(u16::MAX as usize) as u16,
        emitted,
        suppressed_count,
    }
}
