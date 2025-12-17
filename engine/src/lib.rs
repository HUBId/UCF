#![forbid(unsafe_code)]

use profiles::{ProfileComposer, ProfileResolution, ProfileResolutionRequest};
use rsv::{RegulatorState, StateStore};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wire::{SignedFrame, WireError};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EngineInputs {
    pub tick: u64,
    pub inbound: Vec<SignedFrame>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EngineOutcome {
    pub state: RegulatorState,
    pub outbound: Vec<SignedFrame>,
    pub resolution: ProfileResolution,
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("profile resolution failed")]
    Profile(profiles::ProfileError),
    #[error("state storage failed")]
    State(rsv::StateError),
    #[error("wire handling failed")]
    Wire(WireError),
    #[error("update logic is not implemented")]
    NotImplemented,
}

pub trait UpdateEngine {
    type Store: StateStore;
    type Composer: ProfileComposer;

    fn store(&self) -> &Self::Store;
    fn composer(&self) -> &Self::Composer;

    fn apply(
        &mut self,
        state: &mut RegulatorState,
        inputs: EngineInputs,
    ) -> Result<EngineOutcome, EngineError>;
}

pub fn stage_resolution<C: ProfileComposer>(
    composer: &C,
    state: &RegulatorState,
) -> Result<ProfileResolution, EngineError> {
    composer
        .compose(ProfileResolutionRequest {
            profile: state.profile.clone(),
            overlays: state.active_overlays.clone(),
        })
        .map_err(EngineError::from)
}

impl From<profiles::ProfileError> for EngineError {
    fn from(error: profiles::ProfileError) -> Self {
        EngineError::Profile(error)
    }
}

impl From<rsv::StateError> for EngineError {
    fn from(error: rsv::StateError) -> Self {
        EngineError::State(error)
    }
}

impl From<WireError> for EngineError {
    fn from(error: WireError) -> Self {
        EngineError::Wire(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use profiles::{ProfileError, StaticProfileComposer};
    use rsv::HealthFlag;

    #[test]
    fn resolution_request_round_trip() {
        let state = RegulatorState {
            profile: "baseline".to_string(),
            active_overlays: vec!["overlay-a".to_string()],
            window_index: 0,
            health: HealthFlag::Nominal,
        };

        let composer = StaticProfileComposer;
        let resolution = stage_resolution(&composer, &state).expect("resolution should succeed");
        assert_eq!(resolution.active_profile, state.profile);
        assert_eq!(resolution.active_overlays, state.active_overlays);

        let outcome = EngineOutcome {
            state: state.clone(),
            outbound: Vec::new(),
            resolution,
        };
        assert_eq!(outcome.state.profile, state.profile);
    }

    #[test]
    fn engine_error_conversion() {
        let error = ProfileError::InactiveProfile("missing".into());
        let engine_error = EngineError::from(error);
        match engine_error {
            EngineError::Profile(ProfileError::InactiveProfile(name)) => {
                assert_eq!(name, "missing".to_string())
            }
            _ => panic!("unexpected variant"),
        }
    }
}
