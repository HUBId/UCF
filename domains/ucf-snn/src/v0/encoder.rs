use ucf_onn::v0::PhaseDeg;

use crate::v0::types::{
    clamp01, clamp_u16, norm_phase_opt, SnnSpike, SpikeChan, SpikePayload, SpikeTimeMs, TtfsMs,
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
