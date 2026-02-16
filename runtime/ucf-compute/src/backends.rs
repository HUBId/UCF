use std::sync::{Arc, Mutex};

use crate::feature_extractor::ToySaeExtractor;
use crate::pipeline::{ComputePipelineBackend, FusionConfig, LimitsConfig};
use crate::ssm::MockSsmSelectiveScan;
use crate::world_model::MockJepaPredictor;
use crate::{AiComputeBackend, ComputeBudget, ComputeBudgetProfile, ComputeError};

#[cfg(feature = "compute-candle")]
mod candle_backend;

#[cfg(feature = "compute-candle")]
pub use candle_backend::CandleSaeExtractor;

#[cfg(feature = "compute-burn")]
mod burn_backend;

#[cfg(feature = "compute-burn")]
pub use burn_backend::BurnSaeExtractor;

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
pub struct ComputeTimeBudget {
    pub max_micros: u64,
    pub hard_timeout_micros: u64,
}

impl Default for ComputeTimeBudget {
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
    pub budgets: ComputeTimeBudget,
    pub profile: ComputeBudgetProfile,
}

impl Default for ComputeBackendConfig {
    fn default() -> Self {
        let budget = ComputeBudget::default();
        Self {
            kind: ComputeBackendKind::default(),
            seed: budget.seed,
            budgets: ComputeTimeBudget::default(),
            profile: ComputeBudgetProfile::default_profile(),
        }
    }
}

impl ComputeBackendConfig {
    pub fn to_budget(self) -> ComputeBudget {
        let profile = self.profile;
        ComputeBudget {
            max_micros: self.budgets.max_micros,
            hard_timeout_micros: self.budgets.hard_timeout_micros,
            seed: self.seed,
            profile_id: profile.profile_id,
            global_work_units: profile.global_work_units,
            world_units: profile.world_units,
            sae_units: profile.sae_units,
            ssm_units: profile.ssm_units,
            degrade_policy: profile.degrade_policy,
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
        if let Ok(value) = std::env::var("UCF_COMPUTE_BUDGET_PROFILE") {
            cfg.profile = match value.trim().to_ascii_lowercase().as_str() {
                "default" => ComputeBudgetProfile::default_profile(),
                "tight" => ComputeBudgetProfile::tight_profile(),
                "stress" => ComputeBudgetProfile::stress_profile(),
                _ => {
                    return Err(ComputeError::InvalidInput {
                        reason: format!("invalid UCF_COMPUTE_BUDGET_PROFILE={value}"),
                    })
                }
            };
        }

        Ok(cfg)
    }
}

pub fn build_backend(
    cfg: &ComputeBackendConfig,
) -> Result<Box<dyn AiComputeBackend + Send + Sync>, ComputeError> {
    let world = Arc::new(Mutex::new(MockJepaPredictor::default()));
    let ssm = Arc::new(MockSsmSelectiveScan);
    let fusion = FusionConfig::default();
    let limits = LimitsConfig::default();

    let backend = match cfg.kind {
        ComputeBackendKind::Stub => ComputePipelineBackend::new(
            "stub",
            world,
            Arc::new(ToySaeExtractor::default()),
            ssm,
            fusion,
            limits,
        ),
        ComputeBackendKind::Candle => {
            #[cfg(feature = "compute-candle")]
            {
                ComputePipelineBackend::new(
                    "candle",
                    world,
                    Arc::new(CandleSaeExtractor::new(cfg.seed)),
                    ssm,
                    fusion,
                    limits,
                )
            }
            #[cfg(not(feature = "compute-candle"))]
            {
                return Err(ComputeError::BackendDisabled);
            }
        }
        ComputeBackendKind::Burn => {
            #[cfg(feature = "compute-burn")]
            {
                ComputePipelineBackend::new(
                    "burn",
                    world,
                    Arc::new(BurnSaeExtractor::new(cfg.seed)),
                    ssm,
                    fusion,
                    limits,
                )
            }
            #[cfg(not(feature = "compute-burn"))]
            {
                return Err(ComputeError::BackendDisabled);
            }
        }
    };

    Ok(Box::new(backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "compute-candle", feature = "compute-burn"))]
    use crate::{ComputeInput, FrameId};

    #[test]
    fn env_parse_defaults() {
        let cfg = ComputeBackendConfig::default();
        assert_eq!(cfg.kind, ComputeBackendKind::Stub);
        assert!(cfg.budgets.max_micros > 0);
        assert_eq!(cfg.profile.profile_id, 1);
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
    fn parse_budget_profile_from_env() {
        std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", "stress");
        let cfg = ComputeBackendConfig::from_env().expect("parse env");
        assert_eq!(cfg.profile.profile_id, 3);
        std::env::remove_var("UCF_COMPUTE_BUDGET_PROFILE");
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
            assert_eq!(backend.name(), "candle");
        }
    }

    #[test]
    fn burn_profile_behavior() {
        let cfg = ComputeBackendConfig {
            kind: ComputeBackendKind::Burn,
            ..ComputeBackendConfig::default()
        };

        #[cfg(not(feature = "compute-burn"))]
        {
            let result = build_backend(&cfg);
            assert!(matches!(result, Err(ComputeError::BackendDisabled)));
        }

        #[cfg(feature = "compute-burn")]
        {
            let backend = build_backend(&cfg).expect("burn profile available");
            let err = backend
                .compute(
                    &ComputeInput {
                        frame_id: FrameId(1),
                        t: 1,
                        context_digest: [1_u8; 32],
                    },
                    ComputeBudget::default(),
                )
                .expect_err("burn skeleton should return explicit error");
            assert!(matches!(err, ComputeError::NotImplemented));
        }
    }

    #[cfg(feature = "compute-candle")]
    #[test]
    fn candle_profile_differs_from_stub_deterministically() {
        let input = ComputeInput {
            frame_id: FrameId(11),
            t: 5,
            context_digest: [9_u8; 32],
        };
        let budget = ComputeBudget::default();

        let stub = build_backend(&ComputeBackendConfig {
            kind: ComputeBackendKind::Stub,
            ..ComputeBackendConfig::default()
        })
        .expect("stub");
        let candle = build_backend(&ComputeBackendConfig {
            kind: ComputeBackendKind::Candle,
            seed: 77,
            budgets: ComputeTimeBudget::default(),
            profile: ComputeBudgetProfile::default_profile(),
        })
        .expect("candle");

        let a = candle.compute(&input, budget).expect("candle compute");
        let b = candle.compute(&input, budget).expect("candle compute");
        assert_eq!(a, b);

        let stub_out = stub.compute(&input, budget).expect("stub compute");
        assert_ne!(
            a.summary("candle").spikes_digest,
            stub_out.summary("stub").spikes_digest
        );
    }
}
