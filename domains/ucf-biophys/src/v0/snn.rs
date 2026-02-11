use crate::v0::OscId;

pub type SpikeId = u32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpikeKind {
    Feature = 1,
    Causal = 2,
    Verify = 3,
    Attention = 4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpikeEvent {
    pub now_ms: u64,
    pub src: OscId,
    pub kind: SpikeKind,
    pub spike_id: SpikeId,
    pub phase_bin: u8,
    pub ttfs_code: u8,
    pub magnitude: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventBus {
    pub q: Vec<SpikeEvent>,
    pub cap: usize,
}

impl Default for EventBus {
    fn default() -> Self {
        Self {
            q: Vec::new(),
            cap: 4096,
        }
    }
}

impl EventBus {
    pub fn push(&mut self, ev: SpikeEvent) {
        if self.q.len() >= self.cap {
            let _ = self.q.remove(0);
        }
        self.q.push(ev);
    }

    pub fn drain(&mut self) -> Vec<SpikeEvent> {
        core::mem::take(&mut self.q)
    }

    pub fn count_kind(&self, kind: SpikeKind) -> usize {
        self.q.iter().filter(|ev| ev.kind == kind).count()
    }
}

pub fn ttfs_from_strength(strength: f32) -> u8 {
    let s = strength.clamp(0.0, 1.0);
    ((1.0 - s) * 255.0).round() as u8
}
