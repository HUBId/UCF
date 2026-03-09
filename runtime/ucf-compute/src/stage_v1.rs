use sha2::{Digest, Sha256};
use std::time::Instant;

use crate::work_meter::WorkMeter;

pub const STAGE_CONTRACT_VERSION_V1: u16 = 1;
pub const MAX_TEXT_BYTES: usize = 128;
pub const MAX_META_ENTRIES: usize = 8;
pub const MAX_SAE_SPIKES: usize = 16;
pub const MOCK_WORLD_DIM: usize = 16;

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
    pub previous_world_state_digest: Option<[u8; 32]>,
    pub signal_q: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldOutputV1 {
    pub prediction_q: [i16; MOCK_WORLD_DIM],
    pub prediction_error_q: u16,
    pub surprise_q: u16,
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
    pub spikes_digest: [u8; 32],
    pub spike_count: u16,
    pub previous_state_digest: [u8; 32],
    pub pressure_prev_q: u16,
    pub surprise_q: u16,
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
    pub risk_q: u16,
    pub previous_lfm_digest: [u8; 32],
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

pub type MockWorldV1 = CpuWorldStubV1;
pub type MockSaeV1 = CpuSaeStubV1;
pub type MockSsmV1 = CpuSsmStubV1;
pub type MockLfmV1 = CpuLfmStubV1;

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
        let prev = input.previous_world_state_digest.unwrap_or([0; 32]);
        let prediction_q = mock_prediction_vector(input.context_digest, prev, input.signal_q);
        let prediction_digest = digest_prediction(&prediction_q);

        let mut error_sum = 0u32;
        for idx in 0..MOCK_WORLD_DIM {
            let cur = i32::from(prediction_q[idx]);
            let prev_val = i32::from(i16::from_le_bytes([prev[idx * 2], prev[idx * 2 + 1]]));
            error_sum = error_sum.saturating_add(cur.abs_diff(prev_val));
        }
        let mean_abs = error_sum / MOCK_WORLD_DIM as u32;
        let prediction_error_q = mean_abs.min(u32::from(u16::MAX)) as u16;

        let novelty_q = novelty_from_context(input.context_digest);
        let surprise_q = mix_q(prediction_error_q, novelty_q, 45875, 19661);
        Ok(WorldOutputV1 {
            prediction_q,
            prediction_error_q,
            surprise_q,
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
        let mut seeds = Vec::with_capacity(32);
        for i in 0..32 {
            let b = input.context_digest[i] ^ input.prediction_digest[(i * 11) % 32];
            seeds.push((u16::from(b), b));
        }
        seeds.sort_by(|(fa, ba), (fb, bb)| fa.cmp(fb).then_with(|| ba.cmp(bb)));
        seeds.dedup_by_key(|(feature_id, _)| *feature_id);

        let mut spikes = Vec::with_capacity(k);
        for (feature_id, b) in seeds.into_iter().take(k) {
            spikes.push(SaeSpikeV1 {
                feature_id,
                magnitude_q: u16::from(b) * 257,
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
        hasher.update(input.previous_state_digest);
        hasher.update(input.spikes_digest);
        hasher.update(input.surprise_q.to_le_bytes());
        let state_digest: [u8; 32] = hasher.finalize().into();

        let spike_pressure_q = ((u32::from(input.spike_count.min(MAX_SAE_SPIKES as u16))
            * u32::from(u16::MAX))
            / MAX_SAE_SPIKES as u32) as u16;
        let pressure_q = mix_q(input.pressure_prev_q, input.surprise_q, 35389, 19661);
        let pressure_q = mix_q(pressure_q, spike_pressure_q, 58982, 6553);
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
        let uncertainty_q = weighted_sum3_q(
            input.pressure_q,
            input.surprise_q,
            input.risk_q,
            26214,
            19661,
            19661,
        );
        let stability_q =
            u16::MAX.saturating_sub(((u32::from(uncertainty_q) * 58982) / 65535) as u16);
        let mut hasher = Sha256::new();
        hasher.update(input.previous_lfm_digest);
        hasher.update(input.pressure_q.to_le_bytes());
        hasher.update(input.surprise_q.to_le_bytes());
        hasher.update(input.risk_q.to_le_bytes());
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

pub(crate) fn digest_prediction(prediction_q: &[i16; MOCK_WORLD_DIM]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for value in prediction_q {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

fn mock_prediction_vector(
    context_digest: [u8; 32],
    previous_world_state_digest: [u8; 32],
    signal_q: u16,
) -> [i16; MOCK_WORLD_DIM] {
    let mut out = [0i16; MOCK_WORLD_DIM];
    for i in 0..MOCK_WORLD_DIM {
        let ctx = i16::from(context_digest[i]) - 128;
        let prev = i16::from(previous_world_state_digest[MOCK_WORLD_DIM + i]) - 128;
        let seed = i16::from((signal_q & 0x00ff) as u8) - 128;
        out[i] = (ctx * 96 + prev * 16 + seed * 8).clamp(i16::MIN, i16::MAX);
    }
    out
}

pub(crate) fn novelty_from_context(context_digest: [u8; 32]) -> u16 {
    let mut changes = 0u32;
    for idx in 1..32 {
        if context_digest[idx] != context_digest[idx - 1] {
            changes += 1;
        }
    }
    ((changes * u32::from(u16::MAX)) / 31) as u16
}

pub(crate) fn mix_q(a_q: u16, b_q: u16, w_a_q: u16, w_b_q: u16) -> u16 {
    let lhs = u32::from(a_q) * u32::from(w_a_q);
    let rhs = u32::from(b_q) * u32::from(w_b_q);
    ((lhs.saturating_add(rhs).saturating_add(32767)) / 65535).min(u32::from(u16::MAX)) as u16
}

fn weighted_sum3_q(a_q: u16, b_q: u16, c_q: u16, w1_q: u16, w2_q: u16, w3_q: u16) -> u16 {
    let total = u32::from(a_q) * u32::from(w1_q)
        + u32::from(b_q) * u32::from(w2_q)
        + u32::from(c_q) * u32::from(w3_q);
    ((total.saturating_add(32767)) / 65535).min(u32::from(u16::MAX)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stubs_are_deterministic() {
        let world = CpuWorldStubV1;
        let input = WorldInputV1 {
            context_digest: [7; 32],
            previous_world_state_digest: None,
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

    #[test]
    fn deterministic_mock_chain_coupling() {
        let world = CpuWorldStubV1;
        let sae = CpuSaeStubV1;
        let ssm = CpuSsmStubV1;
        let lfm = CpuLfmStubV1;

        let world_out = world
            .step(&WorldInputV1 {
                context_digest: [9; 32],
                previous_world_state_digest: Some([3; 32]),
                signal_q: 77,
            })
            .expect("world");
        let sae_out = sae
            .infer(&SaeInputV1 {
                context_digest: [9; 32],
                prediction_digest: world_out.prediction_digest,
                top_k: 32,
            })
            .expect("sae");
        let ssm_out = ssm
            .step(&SsmInputV1 {
                spikes_digest: sae_out.spikes_digest,
                spike_count: sae_out.spikes.len() as u16,
                previous_state_digest: [1; 32],
                pressure_prev_q: 1234,
                surprise_q: world_out.surprise_q,
            })
            .expect("ssm");
        let lfm_out = lfm
            .step(&LfmInputV1 {
                pressure_q: ssm_out.pressure_q,
                surprise_q: world_out.surprise_q,
                risk_q: 2222,
                previous_lfm_digest: [2; 32],
            })
            .expect("lfm");

        assert!(sae_out.spikes.len() <= MAX_SAE_SPIKES);
        assert!(lfm_out.stability_q <= u16::MAX.saturating_sub(lfm_out.uncertainty_q / 2));

        let world_out_b = world
            .step(&WorldInputV1 {
                context_digest: [9; 32],
                previous_world_state_digest: Some([3; 32]),
                signal_q: 77,
            })
            .expect("world");
        assert_eq!(world_out, world_out_b);
    }
}
