use crate::{AiComputeBackend, ComputeBudget, ComputeError, CpuStubBackend};

#[cfg(feature = "compute-candle")]
mod candle_backend;

#[cfg(feature = "compute-candle")]
pub use candle_backend::CandleBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackendKind {
    #[default]
    Stub,
    Candle,
    Burn,
}

impl ComputeBackendKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "stub" | "cpu_stub" => Some(Self::Stub),
            "candle" | "candle_dummy" => Some(Self::Candle),
            "burn" | "burn_dummy" => Some(Self::Burn),
            _ => None,
        }
    }

    pub fn as_env_str(self) -> &'static str {
        match self {
            Self::Stub => "stub",
            Self::Candle => "candle",
            Self::Burn => "burn",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComputeBudgetProfile {
    pub max_micros: u64,
    pub hard_timeout_micros: u64,
}

impl Default for ComputeBudgetProfile {
    fn default() -> Self {
        let budget = ComputeBudget::default();
        Self {
            max_micros: budget.max_micros,
            hard_timeout_micros: budget.hard_timeout_micros,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComputeBackendConfig {
    pub kind: ComputeBackendKind,
    pub seed: u64,
    pub budgets: ComputeBudgetProfile,
}

impl Default for ComputeBackendConfig {
    fn default() -> Self {
        let budget = ComputeBudget::default();
        Self {
            kind: ComputeBackendKind::default(),
            seed: budget.seed,
            budgets: ComputeBudgetProfile::default(),
        }
    }
}

impl ComputeBackendConfig {
    pub fn to_budget(self) -> ComputeBudget {
        ComputeBudget {
            max_micros: self.budgets.max_micros,
            hard_timeout_micros: self.budgets.hard_timeout_micros,
            seed: self.seed,
        }
    }

    pub fn from_env() -> Result<Self, ComputeError> {
        let mut cfg = Self::default();

        if let Ok(value) = std::env::var("UCF_COMPUTE_BACKEND") {
            cfg.kind =
                ComputeBackendKind::parse(&value).ok_or_else(|| ComputeError::InvalidInput {
                    reason: format!("unsupported UCF_COMPUTE_BACKEND={value}"),
                })?;
        }
        if let Ok(value) = std::env::var("UCF_COMPUTE_SEED") {
            cfg.seed = value
                .parse::<u64>()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_COMPUTE_SEED={value}"),
                })?;
        }
        if let Ok(value) = std::env::var("UCF_COMPUTE_MAX_MICROS") {
            cfg.budgets.max_micros =
                value
                    .parse::<u64>()
                    .map_err(|_| ComputeError::InvalidInput {
                        reason: format!("invalid UCF_COMPUTE_MAX_MICROS={value}"),
                    })?;
        }
        if let Ok(value) = std::env::var("UCF_COMPUTE_HARD_TIMEOUT_MICROS") {
            cfg.budgets.hard_timeout_micros =
                value
                    .parse::<u64>()
                    .map_err(|_| ComputeError::InvalidInput {
                        reason: format!("invalid UCF_COMPUTE_HARD_TIMEOUT_MICROS={value}"),
                    })?;
        }

        Ok(cfg)
    }
}

pub fn build_backend(
    cfg: &ComputeBackendConfig,
) -> Result<Box<dyn AiComputeBackend + Send + Sync>, ComputeError> {
    match cfg.kind {
        ComputeBackendKind::Stub => Ok(Box::new(CpuStubBackend)),
        ComputeBackendKind::Candle => {
            #[cfg(feature = "compute-candle")]
            {
                Ok(Box::new(CandleBackend::new(cfg.seed)))
            }
            #[cfg(not(feature = "compute-candle"))]
            {
                Err(ComputeError::BackendDisabled)
            }
        }
        ComputeBackendKind::Burn => Err(ComputeError::BackendDisabled),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_parse_defaults() {
        let cfg = ComputeBackendConfig::default();
        assert_eq!(cfg.kind, ComputeBackendKind::Stub);
        assert!(cfg.budgets.max_micros > 0);
    }

    #[test]
    fn parse_backend_kind_aliases() {
        assert_eq!(
            ComputeBackendKind::parse("stub"),
            Some(ComputeBackendKind::Stub)
        );
        assert_eq!(
            ComputeBackendKind::parse("cpu_stub"),
            Some(ComputeBackendKind::Stub)
        );
        assert_eq!(
            ComputeBackendKind::parse("candle"),
            Some(ComputeBackendKind::Candle)
        );
        assert_eq!(
            ComputeBackendKind::parse("burn"),
            Some(ComputeBackendKind::Burn)
        );
        assert_eq!(ComputeBackendKind::parse("unknown"), None);
    }

    #[test]
    fn candle_disabled_without_feature() {
        let cfg = ComputeBackendConfig {
            kind: ComputeBackendKind::Candle,
            ..ComputeBackendConfig::default()
        };
        #[cfg(not(feature = "compute-candle"))]
        {
            let result = build_backend(&cfg);
            assert!(matches!(result, Err(ComputeError::BackendDisabled)));
        }

        #[cfg(feature = "compute-candle")]
        {
            let backend = build_backend(&cfg).expect("candle backend available");
            assert_eq!(backend.name(), "candle_dummy");
        }
    }
}
