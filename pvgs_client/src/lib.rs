#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub struct PvgsClientConfig {
    pub cbv_endpoint: String,
    pub hbv_endpoint: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct PvgsSnapshot {
    pub channel: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum PvgsError {
    #[error("PVGS endpoints not configured")]
    MissingConfig,
    #[error("PVGS fetch is not implemented")]
    NotImplemented,
}

#[derive(Debug, Default)]
pub struct PvgsClient {
    config: Option<PvgsClientConfig>,
}

impl PvgsClient {
    pub fn new(config: PvgsClientConfig) -> Self {
        Self {
            config: Some(config),
        }
    }

    pub fn configure(&mut self, config: PvgsClientConfig) {
        self.config = Some(config);
    }

    pub fn fetch_cbv(&self) -> Result<PvgsSnapshot, PvgsError> {
        self.check_config()?;
        Err(PvgsError::NotImplemented)
    }

    pub fn fetch_hbv(&self) -> Result<PvgsSnapshot, PvgsError> {
        self.check_config()?;
        Err(PvgsError::NotImplemented)
    }

    fn check_config(&self) -> Result<(), PvgsError> {
        if self.config.is_none() {
            return Err(PvgsError::MissingConfig);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_requires_configuration() {
        let client = PvgsClient::default();
        let err = client
            .fetch_cbv()
            .expect_err("missing config should be rejected");
        assert!(matches!(err, PvgsError::MissingConfig));
    }

    #[test]
    fn configured_client_is_placeholder_only() {
        let mut client = PvgsClient::default();
        client.configure(PvgsClientConfig {
            cbv_endpoint: "http://cbv".to_string(),
            hbv_endpoint: "http://hbv".to_string(),
        });

        let err = client.fetch_hbv().expect_err("fetch is placeholder");
        assert!(matches!(err, PvgsError::NotImplemented));
    }
}
