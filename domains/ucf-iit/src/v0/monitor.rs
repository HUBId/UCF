use ucf_frames::v1::PhiProxySnapshot;
use ucf_onn::v0::{OnnCore, OscId};

use super::config::IitConfig;

pub const MOD_JEPA: OscId = 10;
pub const MOD_SSM: OscId = 11;
pub const MOD_NSR: OscId = 12;
pub const MOD_PBM: OscId = 13;
pub const MOD_GEIST: OscId = 14;
pub const MOD_BLUE: OscId = 15;

const TRACKED_MODULES: [OscId; 6] = [MOD_JEPA, MOD_SSM, MOD_NSR, MOD_PBM, MOD_GEIST, MOD_BLUE];

#[derive(Clone, Debug)]
pub struct IitMonitor {
    cfg: IitConfig,
}

impl IitMonitor {
    pub fn new(cfg: IitConfig) -> Self {
        Self { cfg }
    }

    pub fn compute(&self, onn: &OnnCore) -> PhiProxySnapshot {
        if !self.cfg.enabled {
            return PhiProxySnapshot::baseline();
        }

        let mut total = 0.0_f32;
        let mut min = 1.0_f32;
        let mut count = 0_u16;

        for (i, module_a) in TRACKED_MODULES.iter().enumerate() {
            for module_b in TRACKED_MODULES.iter().skip(i + 1) {
                let Some(coherence) = onn.coherence_pair(*module_a, *module_b) else {
                    continue;
                };

                total += coherence;
                if coherence < min {
                    min = coherence;
                }
                count = count.saturating_add(1);
            }
        }

        if count == 0 {
            return PhiProxySnapshot::baseline();
        }

        let coherence_mean = total / f32::from(count);
        let coherence_min = min;
        let phi_raw = (0.7 * coherence_mean + 0.3 * coherence_min) * self.cfg.phi_gain;
        let phi = clamp(phi_raw, self.cfg.phi_floor, self.cfg.phi_ceiling);

        PhiProxySnapshot {
            phi,
            coherence_mean,
            coherence_min,
            n_pairs: count,
        }
    }
}

fn clamp(value: f32, floor: f32, ceiling: f32) -> f32 {
    value.max(floor).min(ceiling)
}
