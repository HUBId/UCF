#![forbid(unsafe_code)]

pub type PhaseBin = u8;
pub type SpikeChan = u8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpikeKind {
    Novelty,
    Verify,
    CausalHit,
    MemoryMark,
    AttentionShift,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spike {
    pub now_ms: u64,
    pub kind: SpikeKind,
    pub chan: SpikeChan,
    pub phase: PhaseBin,
    pub strength: f32,
    pub ttfs_us: u32,
}

#[derive(Clone, Debug, Default)]
pub struct SpikeBus {
    pub buf: Vec<Spike>,
}

impl SpikeBus {
    pub fn push(&mut self, s: Spike) {
        self.buf.push(s);
    }

    pub fn drain(&mut self) -> Vec<Spike> {
        std::mem::take(&mut self.buf)
    }
}

pub fn encode_ttfs_us(strength: f32) -> u32 {
    let s = strength.clamp(0.0, 1.0);
    (5000.0 - s * (5000.0 - 50.0)).round() as u32
}

pub fn phase_dist(a: PhaseBin, b: PhaseBin) -> u8 {
    let d = u16::from(a.abs_diff(b));
    let wrapped = 256_u16 - d;
    d.min(wrapped) as u8
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhaseLockCfg {
    pub max_dist: u8,
    pub attenuate: bool,
}

pub fn filter_phase_locked(
    cfg: &PhaseLockCfg,
    ref_phase: PhaseBin,
    spikes: &[Spike],
) -> Vec<Spike> {
    spikes
        .iter()
        .filter_map(|spike| {
            let dist = phase_dist(spike.phase, ref_phase);
            if dist <= cfg.max_dist {
                return Some(*spike);
            }
            if cfg.attenuate {
                let mut attenuated = *spike;
                attenuated.strength *= 0.25;
                attenuated.ttfs_us = attenuated.ttfs_us.saturating_add(1000);
                return Some(attenuated);
            }
            None
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_ttfs_bounds_and_monotonicity() {
        assert_eq!(encode_ttfs_us(1.0), 50);
        assert_eq!(encode_ttfs_us(0.0), 5000);
        assert!(encode_ttfs_us(0.8) < encode_ttfs_us(0.3));
    }

    #[test]
    fn phase_distance_is_circular() {
        assert_eq!(phase_dist(0, 255), 1);
        assert_eq!(phase_dist(12, 12), 0);
    }

    #[test]
    fn phase_lock_filter_handles_attenuate_and_drop() {
        let spike = Spike {
            now_ms: 1,
            kind: SpikeKind::Novelty,
            chan: 1,
            phase: 120,
            strength: 0.8,
            ttfs_us: 100,
        };

        let kept = filter_phase_locked(
            &PhaseLockCfg {
                max_dist: 24,
                attenuate: true,
            },
            0,
            &[spike],
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].strength, 0.2);
        assert_eq!(kept[0].ttfs_us, 1100);

        let dropped = filter_phase_locked(
            &PhaseLockCfg {
                max_dist: 24,
                attenuate: false,
            },
            0,
            &[spike],
        );
        assert!(dropped.is_empty());
    }
}
