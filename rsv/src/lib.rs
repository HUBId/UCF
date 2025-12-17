#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type OverlayId = String;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum HealthFlag {
    Nominal,
    Degraded,
    Faulted,
}

impl Default for HealthFlag {
    fn default() -> Self {
        Self::Nominal
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub struct RegulatorState {
    pub profile: String,
    pub active_overlays: Vec<OverlayId>,
    pub window_index: u64,
    pub health: HealthFlag,
}

#[derive(Debug, Error)]
pub enum StateError {
    #[error("state store not implemented")]
    StorageUnavailable,
    #[error("state serialization failed: {0}")]
    Serialization(String),
}

pub trait StateStore {
    fn load(&self) -> Result<RegulatorState, StateError>;
    fn persist(&self, state: &RegulatorState) -> Result<(), StateError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_nominal() {
        let state = RegulatorState::default();
        assert_eq!(state.health, HealthFlag::Nominal);
    }
}
