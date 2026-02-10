use std::sync::Arc;

use ucf_core::types::{RegionId, SimTime, Q16};

use crate::v1::CorrelationId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrainFrame {
    pub time: SimTime,
    pub corr: CorrelationId,
    pub region: RegionId,
    pub signal: BrainSignal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrainSignal {
    Spike {
        intensity: Q16,
    },
    Neuromod {
        dopamine: Q16,
        serotonin: Q16,
        cortisol: Q16,
    },
    Tag {
        key: Arc<str>,
        value: Arc<str>,
    },
}
