#![forbid(unsafe_code)]

use engine::{stage_resolution, EngineError};
use hpa::{HpaClient, HpaConfig, HpaError, NoopHpaClient};
use profiles::StaticProfileComposer;
use pvgs_client::{PvgsClient, PvgsClientConfig};
use rsv::{HealthFlag, RegulatorState, StateError, StateStore};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
struct ConfigPaths {
    regulator_profiles: String,
    regulator_overlays: String,
    regulator_update_tables: String,
    windowing: String,
    class_thresholds: String,
    hpa: String,
}

impl Default for ConfigPaths {
    fn default() -> Self {
        Self {
            regulator_profiles: "config/regulator_profiles.yaml".to_string(),
            regulator_overlays: "config/regulator_overlays.yaml".to_string(),
            regulator_update_tables: "config/regulator_update_tables.yaml".to_string(),
            windowing: "config/windowing.yaml".to_string(),
            class_thresholds: "config/class_thresholds.yaml".to_string(),
            hpa: "config/hpa.yaml".to_string(),
        }
    }
}

#[derive(Debug)]
enum AppError {
    MissingPath(&'static str),
    State(StateError),
    Engine(EngineError),
    Hpa(HpaError),
    ConfigSerde(serde_yaml::Error),
}

impl From<StateError> for AppError {
    fn from(error: StateError) -> Self {
        AppError::State(error)
    }
}

impl From<EngineError> for AppError {
    fn from(error: EngineError) -> Self {
        AppError::Engine(error)
    }
}

impl From<HpaError> for AppError {
    fn from(error: HpaError) -> Self {
        AppError::Hpa(error)
    }
}

impl From<serde_yaml::Error> for AppError {
    fn from(error: serde_yaml::Error) -> Self {
        AppError::ConfigSerde(error)
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::MissingPath(path) => write!(f, "configuration path missing for {path}"),
            AppError::State(err) => {
                let _ = err;
                write!(f, "state error")
            }
            AppError::Engine(err) => {
                let _ = err;
                write!(f, "engine error")
            }
            AppError::Hpa(err) => {
                let _ = err;
                write!(f, "HPA error")
            }
            AppError::ConfigSerde(err) => {
                let _ = err;
                write!(f, "config serialization failed")
            }
        }
    }
}

impl std::error::Error for AppError {}

struct MemoryStore {
    state: RegulatorState,
}

impl MemoryStore {
    fn new(state: RegulatorState) -> Self {
        Self { state }
    }
}

impl StateStore for MemoryStore {
    fn load(&self) -> Result<RegulatorState, StateError> {
        Ok(self.state.clone())
    }

    fn persist(&self, _state: &RegulatorState) -> Result<(), StateError> {
        Ok(())
    }
}

fn validate_paths(paths: &ConfigPaths) -> Result<(), AppError> {
    if paths.regulator_profiles.is_empty() {
        return Err(AppError::MissingPath("regulator_profiles"));
    }
    if paths.regulator_overlays.is_empty() {
        return Err(AppError::MissingPath("regulator_overlays"));
    }
    if paths.regulator_update_tables.is_empty() {
        return Err(AppError::MissingPath("regulator_update_tables"));
    }
    if paths.windowing.is_empty() {
        return Err(AppError::MissingPath("windowing"));
    }
    if paths.class_thresholds.is_empty() {
        return Err(AppError::MissingPath("class_thresholds"));
    }
    if paths.hpa.is_empty() {
        return Err(AppError::MissingPath("hpa"));
    }

    Ok(())
}

fn main() -> Result<(), AppError> {
    let paths = ConfigPaths::default();
    validate_paths(&paths)?;

    let _ = serde_yaml::to_string(&paths)?;

    let state_store = MemoryStore::new(RegulatorState {
        profile: "baseline".to_string(),
        active_overlays: vec!["default".to_string()],
        window_index: 0,
        health: HealthFlag::Nominal,
    });

    let mut hpa_client = NoopHpaClient::default();
    hpa_client.configure(HpaConfig::default())?;

    let mut pvgs_client = PvgsClient::default();
    pvgs_client.configure(PvgsClientConfig {
        cbv_endpoint: "cbv".to_string(),
        hbv_endpoint: "hbv".to_string(),
    });

    let composer = StaticProfileComposer;
    let current_state = state_store.load()?;
    let resolution = stage_resolution(&composer, &current_state)?;

    // future: wire up RSV, PVGS, and HPA components using the placeholders above
    let _ = pvgs_client;
    let _ = hpa_client;
    println!("boot ok: loaded configs from {:?}", paths);
    println!("active resolution: {resolution:?}");
    Ok(())
}
