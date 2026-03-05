use sha2::{Digest, Sha256};
use std::time::Instant;

use crate::work_meter::WorkMeter;

pub const STAGE_CONTRACT_VERSION_V1: u16 = 1;
pub const MAX_TEXT_BYTES: usize = 128;
pub const MAX_META_ENTRIES: usize = 8;
pub const MAX_SAE_SPIKES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StageErrorCode {
    BackendDisabled = 1,
    Timeout = 2,
    BudgetExceeded = 3,
    ValidationFailed = 4,
    Internal = 5,
}

impl StageErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BackendDisabled => "BACKEND_DISABLED",
            Self::Timeout => "TIMEOUT",
            Self::BudgetExceeded => "BUDGET_EXCEEDED",
            Self::ValidationFailed => "VALIDATION_FAILED",
            Self::Internal => "INTERNAL",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageError {
    pub code: StageErrorCode,
    pub reason: &'static str,
}

impl StageError {
    pub const fn backend_disabled(reason: &'static str) -> Self {
        Self {
            code: StageErrorCode::BackendDisabled,
            reason,
        }
    }

    pub const fn timeout(reason: &'static str) -> Self {
        Self {
            code: StageErrorCode::Timeout,
            reason,
        }
    }

    pub const fn budget_exceeded(reason: &'static str) -> Self {
        Self {
            code: StageErrorCode::BudgetExceeded,
            reason,
        }
    }

    pub const fn validation_failed(reason: &'static str) -> Self {
        Self {
            code: StageErrorCode::ValidationFailed,
            reason,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmInputV1 {
    pub prompt_digest: [u8; 32],
    pub context_digest: [u8; 32],
    pub max_tokens: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOutputV1 {
    pub response_preview: String,
    pub token_count: u16,
    pub response_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldInputV1 {
    pub context_digest: [u8; 32],
    pub signal_q: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldOutputV1 {
    pub prediction_error_q: u16,
    pub prediction_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeInputV1 {
    pub context_digest: [u8; 32],
    pub prediction_digest: [u8; 32],
    pub top_k: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeSpikeV1 {
    pub feature_id: u16,
    pub magnitude_q: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaeOutputV1 {
    pub spikes: Vec<SaeSpikeV1>,
    pub spikes_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmInputV1 {
    pub context_digest: [u8; 32],
    pub spikes_digest: [u8; 32],
    pub previous_state_digest: [u8; 32],
    pub pressure_q: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmOutputV1 {
    pub pressure_q: u16,
    pub state_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LfmInputV1 {
    pub pressure_q: u16,
    pub surprise_q: u16,
    pub state_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LfmOutputV1 {
    pub uncertainty_q: u16,
    pub stability_q: u16,
    pub lfm_digest: [u8; 32],
}

pub trait LlmBackendV1 {
    fn contract_version(&self) -> u16;
    fn backend_id(&self) -> u16;
    fn infer(&self, input: &LlmInputV1) -> Result<LlmOutputV1, StageError>;
}

pub trait WorldPredictorV1 {
    fn contract_version(&self) -> u16;
    fn backend_id(&self) -> u16;
    fn step(&self, input: &WorldInputV1) -> Result<WorldOutputV1, StageError>;
}

pub trait SaeExtractorV1 {
    fn contract_version(&self) -> u16;
    fn backend_id(&self) -> u16;
    fn infer(&self, input: &SaeInputV1) -> Result<SaeOutputV1, StageError>;
}

pub trait SsmKernelV1 {
    fn contract_version(&self) -> u16;
    fn backend_id(&self) -> u16;
    fn step(&self, input: &SsmInputV1) -> Result<SsmOutputV1, StageError>;
}

pub trait LfmBackendV1 {
    fn contract_version(&self) -> u16;
    fn backend_id(&self) -> u16;
    fn step(&self, input: &LfmInputV1) -> Result<LfmOutputV1, StageError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuLlmStubV1;
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuWorldStubV1;
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuSaeStubV1;
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuSsmStubV1;
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuLfmStubV1;

impl LlmBackendV1 for CpuLlmStubV1 {
    fn contract_version(&self) -> u16 {
        STAGE_CONTRACT_VERSION_V1
    }

    fn backend_id(&self) -> u16 {
        101
    }

    fn infer(&self, input: &LlmInputV1) -> Result<LlmOutputV1, StageError> {
        let mut hasher = Sha256::new();
        hasher.update(input.prompt_digest);
        hasher.update(input.context_digest);
        hasher.update(input.max_tokens.to_le_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let mut preview = format!("stub:{:02x}{:02x}{:02x}", digest[0], digest[1], digest[2]);
        preview.truncate(MAX_TEXT_BYTES);
        Ok(LlmOutputV1 {
            response_preview: preview,
            token_count: input.max_tokens.min(32),
            response_digest: digest,
        })
    }
}

impl WorldPredictorV1 for CpuWorldStubV1 {
    fn contract_version(&self) -> u16 {
        STAGE_CONTRACT_VERSION_V1
    }

    fn backend_id(&self) -> u16 {
        102
    }

    fn step(&self, input: &WorldInputV1) -> Result<WorldOutputV1, StageError> {
        let mut hasher = Sha256::new();
        hasher.update(input.context_digest);
        hasher.update(input.signal_q.to_le_bytes());
        let prediction_digest: [u8; 32] = hasher.finalize().into();
        let prediction_error_q = u16::from_le_bytes([prediction_digest[0], prediction_digest[1]]);
        Ok(WorldOutputV1 {
            prediction_error_q,
            prediction_digest,
        })
    }
}

impl SaeExtractorV1 for CpuSaeStubV1 {
    fn contract_version(&self) -> u16 {
        STAGE_CONTRACT_VERSION_V1
    }

    fn backend_id(&self) -> u16 {
        103
    }

    fn infer(&self, input: &SaeInputV1) -> Result<SaeOutputV1, StageError> {
        let k = usize::from(input.top_k).min(MAX_SAE_SPIKES);
        let mut spikes = Vec::with_capacity(k);
        for i in 0..k {
            let b = input.context_digest[i % 32] ^ input.prediction_digest[(i * 7) % 32];
            spikes.push(SaeSpikeV1 {
                feature_id: u16::from(b) + (i as u16 * 13),
                magnitude_q: u16::from(b) << 8,
            });
        }
        spikes.sort_by_key(|s| s.feature_id);
        let spikes_digest = digest_spikes(&spikes);
        Ok(SaeOutputV1 {
            spikes,
            spikes_digest,
        })
    }
}

impl SsmKernelV1 for CpuSsmStubV1 {
    fn contract_version(&self) -> u16 {
        STAGE_CONTRACT_VERSION_V1
    }

    fn backend_id(&self) -> u16 {
        104
    }

    fn step(&self, input: &SsmInputV1) -> Result<SsmOutputV1, StageError> {
        let mut hasher = Sha256::new();
        hasher.update(input.context_digest);
        hasher.update(input.spikes_digest);
        hasher.update(input.previous_state_digest);
        hasher.update(input.pressure_q.to_le_bytes());
        let state_digest: [u8; 32] = hasher.finalize().into();
        let pressure_q = input
            .pressure_q
            .saturating_add(u16::from(state_digest[0]) << 4);
        Ok(SsmOutputV1 {
            pressure_q,
            state_digest,
        })
    }
}

impl LfmBackendV1 for CpuLfmStubV1 {
    fn contract_version(&self) -> u16 {
        STAGE_CONTRACT_VERSION_V1
    }

    fn backend_id(&self) -> u16 {
        105
    }

    fn step(&self, input: &LfmInputV1) -> Result<LfmOutputV1, StageError> {
        let uncertainty_q = input.pressure_q.saturating_add(input.surprise_q / 2);
        let stability_q = u16::MAX.saturating_sub(uncertainty_q);
        let mut hasher = Sha256::new();
        hasher.update(input.state_digest);
        hasher.update(uncertainty_q.to_le_bytes());
        hasher.update(stability_q.to_le_bytes());
        let lfm_digest: [u8; 32] = hasher.finalize().into();
        Ok(LfmOutputV1 {
            uncertainty_q,
            stability_q,
            lfm_digest,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageViolationRecord {
    pub stage: &'static str,
    pub code: StageErrorCode,
}

pub struct StageRunner;

impl StageRunner {
    pub fn run<T, F>(
        stage: &'static str,
        timeout_ms: u64,
        budget: &mut WorkMeter,
        work_units: u64,
        f: F,
        degraded: T,
    ) -> (T, Option<StageViolationRecord>)
    where
        F: FnOnce() -> Result<T, StageError>,
    {
        if budget.spend(work_units, stage).is_err() {
            return (
                degraded,
                Some(StageViolationRecord {
                    stage,
                    code: StageErrorCode::BudgetExceeded,
                }),
            );
        }
        let started = Instant::now();
        match f() {
            Ok(output) => {
                let elapsed = started.elapsed().as_millis() as u64;
                if elapsed > timeout_ms {
                    (
                        degraded,
                        Some(StageViolationRecord {
                            stage,
                            code: StageErrorCode::Timeout,
                        }),
                    )
                } else {
                    (output, None)
                }
            }
            Err(err) => (
                degraded,
                Some(StageViolationRecord {
                    stage,
                    code: err.code,
                }),
            ),
        }
    }
}

fn digest_spikes(spikes: &[SaeSpikeV1]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for spike in spikes {
        hasher.update(spike.feature_id.to_le_bytes());
        hasher.update(spike.magnitude_q.to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stubs_are_deterministic() {
        let world = CpuWorldStubV1;
        let input = WorldInputV1 {
            context_digest: [7; 32],
            signal_q: 123,
        };
        let a = world.step(&input).expect("world ok");
        let b = world.step(&input).expect("world ok");
        assert_eq!(a, b);
    }

    #[test]
    fn sae_spike_count_is_bounded() {
        let sae = CpuSaeStubV1;
        let out = sae
            .infer(&SaeInputV1 {
                context_digest: [1; 32],
                prediction_digest: [2; 32],
                top_k: 255,
            })
            .expect("sae");
        assert!(out.spikes.len() <= MAX_SAE_SPIKES);
    }

    #[test]
    fn stage_runner_maps_budget_to_violation() {
        let mut meter = WorkMeter::new(1);
        let (_out, violation) = StageRunner::run("world", 1, &mut meter, 5, || Ok(7u8), 0u8);
        assert_eq!(
            violation.expect("violation").code,
            StageErrorCode::BudgetExceeded
        );
    }

    #[test]
    fn error_codes_are_stable() {
        assert_eq!(StageErrorCode::BackendDisabled.as_str(), "BACKEND_DISABLED");
        assert_eq!(StageErrorCode::Timeout.as_str(), "TIMEOUT");
        assert_eq!(StageErrorCode::BudgetExceeded.as_str(), "BUDGET_EXCEEDED");
        assert_eq!(
            StageErrorCode::ValidationFailed.as_str(),
            "VALIDATION_FAILED"
        );
    }
}
