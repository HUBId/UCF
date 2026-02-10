use super::{compute_delta, NeuromodulatorField};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuromodInputs {
    pub surprise: f32,
    pub reward: f32,
    pub threat: f32,
    pub social: f32,
}

impl NeuromodInputs {
    pub fn baseline() -> Self {
        Self {
            surprise: 0.0,
            reward: 0.0,
            threat: 0.0,
            social: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuromodScheduler {
    pub tick_ms: u64,
    last_tick: u64,
}

impl NeuromodScheduler {
    pub fn new(tick_ms: u64) -> Self {
        Self {
            tick_ms,
            last_tick: 0,
        }
    }

    pub fn should_tick(&self, now_ms: u64) -> bool {
        now_ms >= self.last_tick.saturating_add(self.tick_ms)
    }

    pub fn advance(
        &mut self,
        now_ms: u64,
        field: &mut NeuromodulatorField,
        inputs: NeuromodInputs,
    ) {
        while self.should_tick(now_ms) {
            let delta = compute_delta(inputs);
            field.apply_delta(delta);
            self.last_tick = self.last_tick.saturating_add(self.tick_ms);
        }
    }
}
