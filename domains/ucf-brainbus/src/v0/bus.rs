use ucf_core::types::SimTime;

use crate::v0::{BrainBusError, BrainEvent};

pub trait BrainBus {
    fn push(&mut self, ev: BrainEvent) -> Result<(), BrainBusError>;
    fn pop_ready(&mut self, now: SimTime) -> Option<BrainEvent>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
