#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub struct HpaConfig {
    pub endpoint: String,
    pub calibration_table: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct HpaSnapshot {
    pub timestamp: u64,
    pub payload: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum HpaError {
    #[error("HPA configuration missing: {0}")]
    MissingConfig(String),
    #[error("HPA interaction is not implemented")]
    NotImplemented,
}

pub trait HpaClient {
    fn configure(&mut self, config: HpaConfig) -> Result<(), HpaError>;
    fn measure(&mut self) -> Result<HpaSnapshot, HpaError>;
}

#[derive(Debug, Default)]
pub struct NoopHpaClient {
    pub last_config: Option<HpaConfig>,
}

impl HpaClient for NoopHpaClient {
    fn configure(&mut self, config: HpaConfig) -> Result<(), HpaError> {
        self.last_config = Some(config);
        Ok(())
    }

    fn measure(&mut self) -> Result<HpaSnapshot, HpaError> {
        Err(HpaError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_client_stores_config() {
        let mut client = NoopHpaClient::default();
        let config = HpaConfig {
            endpoint: "memristor:0".to_string(),
            calibration_table: Some("default".to_string()),
        };

        client
            .configure(config.clone())
            .expect("config should be accepted");
        assert_eq!(client.last_config, Some(config));
    }

    #[test]
    fn noop_client_measure_not_implemented() {
        let mut client = NoopHpaClient::default();
        let result = client.measure();
        assert!(matches!(result, Err(HpaError::NotImplemented)));
    }
}
