use std::collections::VecDeque;

use ucf_core::types::SimTime;

use crate::v0::{BrainBus, BrainBusError, BrainEvent};

#[derive(Debug, Clone)]
pub struct InMemoryBrainQueue {
    cap: usize,
    events: VecDeque<BrainEvent>,
}

impl InMemoryBrainQueue {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            events: VecDeque::new(),
        }
    }
}

impl BrainBus for InMemoryBrainQueue {
    fn push(&mut self, ev: BrainEvent) -> Result<(), BrainBusError> {
        if self.events.len() >= self.cap {
            return Err(BrainBusError::QueueFull);
        }
        self.events.push_back(ev);
        Ok(())
    }

    fn pop_ready(&mut self, now: SimTime) -> Option<BrainEvent> {
        for (idx, event) in self.events.iter().enumerate() {
            let ready = match event_time(event) {
                Some(time) => sim_time_le(time, now),
                None => true,
            };
            if ready {
                return self.events.remove(idx);
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.events.len()
    }
}

fn event_time(ev: &BrainEvent) -> Option<SimTime> {
    match ev {
        BrainEvent::Spike(spike) => Some(spike.time),
        BrainEvent::Frame(_) => None,
    }
}

fn sim_time_le(lhs: SimTime, rhs: SimTime) -> bool {
    lhs.window.get() < rhs.window.get()
        || (lhs.window.get() == rhs.window.get() && lhs.tick.get() <= rhs.tick.get())
}
