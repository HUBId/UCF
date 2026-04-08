use crate::{
    ComputeBackendConfig, ComputeBackendKind, ComputeError, EnablementConfig, ModelSlot,
    RealEnablementMode, SlotMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    Production,
    Diagnostic,
    Test,
}

impl RuntimeMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "production" | "prod" => Some(Self::Production),
            "diagnostic" | "diag" | "compare" => Some(Self::Diagnostic),
            "test" | "dev" => Some(Self::Test),
            _ => None,
        }
    }

    pub fn as_env_str(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::Diagnostic => "diagnostic",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentProfile {
    LocalOnly,
    MultiWorker,
}

impl DeploymentProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "local_only" | "local" | "single_worker" => Some(Self::LocalOnly),
            "multi_worker" | "worker" | "remote" => Some(Self::MultiWorker),
            _ => None,
        }
    }

    pub fn as_env_str(self) -> &'static str {
        match self {
            Self::LocalOnly => "local_only",
            Self::MultiWorker => "multi_worker",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeDiagnosticFlags {
    pub compare_enabled: bool,
    pub shadow_enabled: bool,
    pub slot_shadow_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeProfile {
    pub mode: RuntimeMode,
    pub deployment: DeploymentProfile,
    pub enablement: EnablementConfig,
    pub diagnostics: RuntimeDiagnosticFlags,
}

impl RuntimeProfile {
    pub fn from_env(cfg: &ComputeBackendConfig) -> Result<Self, ComputeError> {
        let enablement = EnablementConfig::from_env()?;

        let mode = match std::env::var("UCF_RUNTIME_MODE") {
            Ok(raw) => RuntimeMode::parse(&raw).ok_or_else(|| ComputeError::InvalidInput {
                reason: format!("invalid UCF_RUNTIME_MODE={raw}"),
            })?,
            Err(_) => match cfg.kind {
                ComputeBackendKind::Burn | ComputeBackendKind::Worker => RuntimeMode::Production,
                ComputeBackendKind::Stub | ComputeBackendKind::Candle => RuntimeMode::Test,
            },
        };

        let deployment = match std::env::var("UCF_DEPLOYMENT_PROFILE") {
            Ok(raw) => {
                DeploymentProfile::parse(&raw).ok_or_else(|| ComputeError::InvalidInput {
                    reason: format!("invalid UCF_DEPLOYMENT_PROFILE={raw}"),
                })?
            }
            Err(_) => {
                if cfg.kind == ComputeBackendKind::Worker {
                    DeploymentProfile::MultiWorker
                } else {
                    DeploymentProfile::LocalOnly
                }
            }
        };

        let slot_shadow_enabled = ModelSlot::all().into_iter().any(|slot| {
            let key = format!("UCF_SLOT_{}_MODE", slot.env_key());
            std::env::var(&key)
                .ok()
                .and_then(|value| SlotMode::parse(&value))
                .is_some_and(|mode| mode == SlotMode::Shadow)
        });
        let diagnostics = RuntimeDiagnosticFlags {
            compare_enabled: enablement.mode == RealEnablementMode::Compare,
            shadow_enabled: matches!(
                enablement.mode,
                RealEnablementMode::Shadow | RealEnablementMode::Compare
            ),
            slot_shadow_enabled,
        };

        if deployment == DeploymentProfile::MultiWorker && cfg.kind != ComputeBackendKind::Worker {
            return Err(ComputeError::InvalidInput {
                reason: format!(
                    "unsupported runtime profile: deployment={} requires UCF_COMPUTE_BACKEND=worker",
                    deployment.as_env_str()
                ),
            });
        }

        if deployment == DeploymentProfile::LocalOnly && cfg.kind == ComputeBackendKind::Worker {
            return Err(ComputeError::InvalidInput {
                reason:
                    "unsupported runtime profile: UCF_COMPUTE_BACKEND=worker requires deployment=multi_worker"
                        .to_string(),
            });
        }

        if mode == RuntimeMode::Production {
            if !matches!(
                cfg.kind,
                ComputeBackendKind::Burn | ComputeBackendKind::Worker
            ) {
                return Err(ComputeError::InvalidInput {
                    reason: format!(
                        "production runtime mode requires backend=burn|worker (got {})",
                        cfg.kind.as_env_str()
                    ),
                });
            }

            if diagnostics.shadow_enabled || diagnostics.compare_enabled {
                return Err(ComputeError::InvalidInput {
                    reason:
                        "diagnostic-only shadow/compare configuration requested for production mode"
                            .to_string(),
                });
            }
        }

        Ok(Self {
            mode,
            deployment,
            enablement,
            diagnostics,
        })
    }

    pub fn from_runtime_env() -> Result<Self, ComputeError> {
        let cfg = ComputeBackendConfig::from_env()?;
        Self::from_env(&cfg)
    }

    pub fn fallback_for_execution_path(path: crate::compute_service::JobExecutionPath) -> Self {
        let deployment = if path == crate::compute_service::JobExecutionPath::WorkerIpc {
            DeploymentProfile::MultiWorker
        } else {
            DeploymentProfile::LocalOnly
        };
        Self {
            mode: RuntimeMode::Test,
            deployment,
            enablement: EnablementConfig::default(),
            diagnostics: RuntimeDiagnosticFlags {
                compare_enabled: false,
                shadow_enabled: false,
                slot_shadow_enabled: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_track_backend_kind() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();

        let burn = RuntimeProfile::from_env(&ComputeBackendConfig {
            kind: ComputeBackendKind::Burn,
            ..ComputeBackendConfig::default()
        })
        .expect("burn runtime profile");
        assert_eq!(burn.mode, RuntimeMode::Production);
        assert_eq!(burn.deployment, DeploymentProfile::LocalOnly);

        let stub =
            RuntimeProfile::from_env(&ComputeBackendConfig::default()).expect("stub runtime");
        assert_eq!(stub.mode, RuntimeMode::Test);
        assert_eq!(stub.deployment, DeploymentProfile::LocalOnly);
    }

    #[test]
    fn production_rejects_diagnostic_enablement() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();

        std::env::set_var("UCF_REAL_ENABLEMENT_MODE", "shadow");
        let result = RuntimeProfile::from_env(&ComputeBackendConfig {
            kind: ComputeBackendKind::Burn,
            ..ComputeBackendConfig::default()
        });

        assert!(matches!(result, Err(ComputeError::InvalidInput { .. })));
    }

    #[test]
    fn production_allows_slot_shadow_flags_without_global_shadow_mode() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();

        std::env::set_var("UCF_SLOT_EBM_MODE", "shadow");
        let result = RuntimeProfile::from_env(&ComputeBackendConfig {
            kind: ComputeBackendKind::Burn,
            ..ComputeBackendConfig::default()
        });

        let profile = result.expect("slot-level shadow remains allowed in production mode");
        assert_eq!(profile.mode, RuntimeMode::Production);
        assert!(profile.diagnostics.slot_shadow_enabled);
        assert!(!profile.diagnostics.shadow_enabled);
    }

    #[test]
    fn rejects_unsupported_multi_worker_profile() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();

        std::env::set_var("UCF_DEPLOYMENT_PROFILE", "multi_worker");
        let result = RuntimeProfile::from_env(&ComputeBackendConfig {
            kind: ComputeBackendKind::Burn,
            ..ComputeBackendConfig::default()
        });

        assert!(matches!(result, Err(ComputeError::InvalidInput { .. })));
    }

    #[test]
    fn compare_runtime_alias_maps_to_diagnostic_mode() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();

        std::env::set_var("UCF_RUNTIME_MODE", "compare");
        let profile = RuntimeProfile::from_env(&ComputeBackendConfig {
            kind: ComputeBackendKind::Stub,
            ..ComputeBackendConfig::default()
        })
        .expect("compare alias maps to diagnostic runtime mode");
        assert_eq!(profile.mode, RuntimeMode::Diagnostic);
    }
}
