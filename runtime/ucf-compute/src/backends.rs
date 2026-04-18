use crate::pipeline::{ComputePipelineBackend, FusionConfig, LimitsConfig};
use crate::{
    AiComputeBackend, BackendPackConfig, BackendPackFactory, BackendPackKind, ComputeBudget,
    ComputeBudgetProfile, ComputeError, EnablementComputeBackend, RealEnablementMode,
    RuntimeProfile,
};

pub const CANONICAL_ONBOARDING_BACKEND: ComputeBackendKind = ComputeBackendKind::Burn;
pub const CANONICAL_ONBOARDING_PACK: BackendPackKind = BackendPackKind::BurnToyV1;

#[cfg(feature = "compute-candle")]
mod candle_backend;

#[cfg(feature = "compute-candle")]
pub use candle_backend::{CandleSaeExtractor, CandleSsmKernel, CandleWorldPredictor};

#[cfg(feature = "compute-burn")]
mod burn_backend;

#[cfg(feature = "compute-burn")]
pub use burn_backend::{BurnSaeExtractor, BurnSsmKernel, BurnWorldPredictor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackendKind {
    /// Legacy/dev compatibility lane.
    ///
    /// This backend is deterministic and bounded, but it is not the canonical
    /// production onboarding path.
    #[default]
    Stub,
    /// Compatibility seam for candle-based experiments and parity checks.
    Candle,
    /// Canonical production onboarding backend kind.
    Burn,
    /// Internal worker execution lane used by process-isolated workers.
    Worker,
}

impl ComputeBackendKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "stub" => Some(Self::Stub),
            "candle" => Some(Self::Candle),
            "burn" => Some(Self::Burn),
            "worker" => Some(Self::Worker),
            _ => None,
        }
    }

    pub fn as_env_str(self) -> &'static str {
        match self {
            Self::Stub => "stub",
            Self::Candle => "candle",
            Self::Burn => "burn",
            Self::Worker => "worker",
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
            lfm_units: profile.lfm_units,
            degrade_policy: profile.degrade_policy,
            governor_tier: 0,
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
    let backend = build_service_compute_backend(cfg)?;
    let fusion = FusionConfig::default();
    let limits = LimitsConfig::default();

    let primary: Box<dyn AiComputeBackend + Send + Sync> = Box::new(backend);
    let runtime_profile = RuntimeProfile::from_env(cfg)?;
    let enablement = runtime_profile.enablement;
    if enablement.mode == RealEnablementMode::Off {
        return Ok(primary);
    }

    let shadow_kind = match cfg.kind {
        ComputeBackendKind::Stub => BackendPackKind::CandleToyV1,
        ComputeBackendKind::Candle | ComputeBackendKind::Burn | ComputeBackendKind::Worker => {
            BackendPackKind::ToyV1
        }
    };
    let shadow_pack = BackendPackFactory::build(BackendPackConfig {
        pack: shadow_kind,
        seed: cfg.seed,
    });
    let shadow_backend = shadow_pack
        .ok()
        .map(|pack| ComputePipelineBackend::new(pack, fusion, limits))
        .map(|b| Box::new(b) as Box<dyn AiComputeBackend + Send + Sync>);

    Ok(Box::new(EnablementComputeBackend::new(
        primary,
        shadow_backend,
        enablement,
    )))
}

/// Build a pipeline backend that can be mounted behind `CanonicalComputeEntryPoint`.
///
/// This keeps consumer integration on the canonical service contracts while
/// still honoring configured backend kinds (including compatibility/internal lanes).
pub fn build_service_compute_backend(
    cfg: &ComputeBackendConfig,
) -> Result<ComputePipelineBackend, ComputeError> {
    let fusion = FusionConfig::default();
    let limits = LimitsConfig::default();

    let pack_kind = match cfg.kind {
        ComputeBackendKind::Stub => BackendPackKind::ToyV1,
        ComputeBackendKind::Candle => BackendPackKind::CandleToyV1,
        ComputeBackendKind::Burn => BackendPackKind::BurnToyV1,
        ComputeBackendKind::Worker => BackendPackKind::WorkerV1,
    };
    let pack = BackendPackFactory::build(BackendPackConfig {
        pack: pack_kind,
        seed: cfg.seed,
    })?;

    match cfg.kind {
        ComputeBackendKind::Stub => Ok(ComputePipelineBackend::new(pack, fusion, limits)),
        ComputeBackendKind::Candle => {
            #[cfg(feature = "compute-candle")]
            {
                Ok(ComputePipelineBackend::new(pack, fusion, limits))
            }
            #[cfg(not(feature = "compute-candle"))]
            {
                Err(ComputeError::BackendDisabled)
            }
        }
        ComputeBackendKind::Burn => {
            #[cfg(feature = "compute-burn")]
            {
                Ok(ComputePipelineBackend::new(pack, fusion, limits))
            }
            #[cfg(not(feature = "compute-burn"))]
            {
                Err(ComputeError::BackendDisabled)
            }
        }
        ComputeBackendKind::Worker => Ok(ComputePipelineBackend::new(pack, fusion, limits)),
    }
}

/// Build the canonical production onboarding backend.
///
/// This is intentionally separate from `build_backend` so call sites can
/// explicitly opt into the canonical Burn onboarding lane rather than relying
/// on compatibility-oriented `ComputeBackendConfig` defaults.
pub fn build_canonical_production_backend(
    seed: u64,
) -> Result<Box<dyn AiComputeBackend + Send + Sync>, ComputeError> {
    Ok(Box::new(build_onboarding_reference_backend(seed)?))
}

pub fn build_onboarding_reference_backend(
    seed: u64,
) -> Result<ComputePipelineBackend, ComputeError> {
    let pack = BackendPackFactory::build(BackendPackConfig {
        pack: CANONICAL_ONBOARDING_PACK,
        seed,
    })?;
    Ok(ComputePipelineBackend::new(
        pack,
        FusionConfig::default(),
        LimitsConfig::default(),
    ))
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
    fn rejects_legacy_backend_kind_aliases() {
        for legacy in ["cpu_stub", "candle_dummy", "burn_dummy", "worker_v1"] {
            assert_eq!(ComputeBackendKind::parse(legacy), None);
        }
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
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        let cfg = ComputeBackendConfig {
            kind: ComputeBackendKind::Candle,
            ..ComputeBackendConfig::default()
        };
        #[cfg(not(feature = "compute-candle"))]
        {
            let result = build_backend(&cfg);
            assert!(matches!(
                result,
                Err(ComputeError::BackendDisabled) | Err(ComputeError::InvalidInput { .. })
            ));
        }

        #[cfg(feature = "compute-candle")]
        {
            let Ok(backend) = build_backend(&cfg) else {
                return;
            };
            assert!(backend.name().contains("candle"));
        }
    }

    #[test]
    fn burn_profile_behavior() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        let cfg = ComputeBackendConfig {
            kind: ComputeBackendKind::Burn,
            ..ComputeBackendConfig::default()
        };

        #[cfg(not(feature = "compute-burn"))]
        {
            let result = build_backend(&cfg);
            assert!(matches!(
                result,
                Err(ComputeError::BackendDisabled) | Err(ComputeError::InvalidInput { .. })
            ));
        }

        #[cfg(feature = "compute-burn")]
        {
            let Ok(backend) = build_backend(&cfg) else {
                return;
            };
            let result = backend.compute(
                &ComputeInput {
                    frame_id: FrameId(1),
                    t: 1,
                    context_digest: [1_u8; 32],
                },
                ComputeBudget::default(),
            );
            match result {
                Ok(out) => assert!((0.0..=1.0).contains(&out.surprise)),
                Err(ComputeError::BackendDisabled) => {}
                Err(other) => panic!("unexpected burn compute error: {other:?}"),
            }
        }
    }

    #[test]
    fn onboarding_reference_profile_is_pinned_to_burn_pack() {
        assert_eq!(CANONICAL_ONBOARDING_BACKEND, ComputeBackendKind::Burn);
        assert_eq!(CANONICAL_ONBOARDING_PACK, BackendPackKind::BurnToyV1);
    }

    #[test]
    fn compatibility_constructor_default_is_not_the_canonical_production_lane() {
        let cfg = ComputeBackendConfig::default();
        assert_eq!(cfg.kind, ComputeBackendKind::Stub);
        assert_ne!(cfg.kind, CANONICAL_ONBOARDING_BACKEND);
    }

    #[test]
    fn onboarding_reference_backend_build_state_is_honest() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();

        let result = build_onboarding_reference_backend(7);
        #[cfg(all(feature = "compute-burn", feature = "lfm-burn"))]
        {
            if let Ok(backend) = result {
                assert_eq!(backend.name(), CANONICAL_ONBOARDING_PACK.as_str());
            }
        }
        #[cfg(not(all(feature = "compute-burn", feature = "lfm-burn")))]
        {
            assert!(matches!(
                result,
                Err(ComputeError::BackendDisabled) | Err(ComputeError::InvalidInput { .. })
            ));
        }
    }

    #[cfg(feature = "compute-candle")]
    #[test]
    fn candle_profile_differs_from_stub_deterministically() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
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
        let candle = match build_backend(&ComputeBackendConfig {
            kind: ComputeBackendKind::Candle,
            seed: 77,
            budgets: ComputeTimeBudget::default(),
            profile: ComputeBudgetProfile::default_profile(),
        }) {
            Ok(backend) => backend,
            Err(ComputeError::BackendDisabled) | Err(ComputeError::InvalidInput { .. }) => return,
            Err(other) => panic!("unexpected candle init error: {other:?}"),
        };

        let a = candle.compute(&input, budget).expect("candle compute");
        let candle2 = match build_backend(&ComputeBackendConfig {
            kind: ComputeBackendKind::Candle,
            seed: 77,
            budgets: ComputeTimeBudget::default(),
            profile: ComputeBudgetProfile::default_profile(),
        }) {
            Ok(backend) => backend,
            Err(ComputeError::BackendDisabled) | Err(ComputeError::InvalidInput { .. }) => return,
            Err(other) => panic!("unexpected candle init error: {other:?}"),
        };
        let b = candle2.compute(&input, budget).expect("candle compute");
        assert_eq!(a.summary("a").spikes_digest, b.summary("b").spikes_digest);

        let stub_out = stub.compute(&input, budget).expect("stub compute");
        assert_ne!(
            a.summary("candle").spikes_digest,
            stub_out.summary("stub").spikes_digest
        );
    }

    #[cfg(all(feature = "compute-candle", feature = "compute-burn"))]
    #[test]
    fn candle_burn_envelope_parity_smoke() {
        let input = ComputeInput {
            frame_id: FrameId(7),
            t: 3,
            context_digest: [3_u8; 32],
        };
        let budget = ComputeBudget::default();
        let candle = build_backend(&ComputeBackendConfig {
            kind: ComputeBackendKind::Candle,
            ..ComputeBackendConfig::default()
        })
        .expect("candle");
        let burn = build_backend(&ComputeBackendConfig {
            kind: ComputeBackendKind::Burn,
            ..ComputeBackendConfig::default()
        })
        .expect("burn");

        let candle_out = candle.compute(&input, budget).expect("candle compute");
        let burn_out = burn.compute(&input, budget).expect("burn compute");
        assert!((0.0..=1.0).contains(&candle_out.pressure));
        assert!((0.0..=1.0).contains(&burn_out.pressure));
        assert!((candle_out.pressure - burn_out.pressure).abs() <= 0.15);
    }
}
