#![forbid(unsafe_code)]

use rsv::OverlayId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ProfileDefinition {
    pub name: String,
    pub overlays: Vec<OverlayDefinition>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct OverlayDefinition {
    pub name: OverlayId,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ProfileResolutionRequest {
    pub profile: String,
    pub overlays: Vec<OverlayId>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ProfileResolution {
    pub active_profile: String,
    pub active_overlays: Vec<OverlayId>,
}

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("profile {0} is not active")]
    InactiveProfile(String),
    #[error("overlay {0} is not supported")]
    UnsupportedOverlay(String),
}

pub trait ProfileComposer {
    fn compose(&self, request: ProfileResolutionRequest)
        -> Result<ProfileResolution, ProfileError>;
}

#[derive(Debug, Default)]
pub struct StaticProfileComposer;

impl ProfileComposer for StaticProfileComposer {
    fn compose(
        &self,
        request: ProfileResolutionRequest,
    ) -> Result<ProfileResolution, ProfileError> {
        if request.profile.is_empty() {
            return Err(ProfileError::InactiveProfile("<empty>".to_string()));
        }
        Ok(ProfileResolution {
            active_profile: request.profile,
            active_overlays: request.overlays,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_composer_passes_through_values() {
        let composer = StaticProfileComposer;
        let request = ProfileResolutionRequest {
            profile: "baseline".to_string(),
            overlays: vec!["overlay".to_string()],
        };

        let resolution = composer.compose(request).expect("resolution should pass");
        assert_eq!(resolution.active_profile, "baseline");
        assert_eq!(resolution.active_overlays, vec!["overlay".to_string()]);
    }

    #[test]
    fn static_composer_rejects_empty_profile() {
        let composer = StaticProfileComposer;
        let request = ProfileResolutionRequest {
            profile: String::new(),
            overlays: Vec::new(),
        };

        let err = composer
            .compose(request)
            .expect_err("empty profiles are invalid");
        match err {
            ProfileError::InactiveProfile(name) => assert_eq!(name, "<empty>"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
