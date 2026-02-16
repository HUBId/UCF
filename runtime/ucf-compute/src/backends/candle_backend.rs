use candle_core::{Device, Tensor};

use crate::capabilities::FeatureExtractor;
use crate::feature_extractor::{FeatureVector, SaeOutput, SAE_FEATURE_DIM, SAE_MAX_SPIKES};
use crate::world_model::WorldModelOutput;
use crate::{ComputeBudget, ComputeError, ComputeInput, Spike};

const IN_DIM: usize = 32;
const OUT_DIM: usize = SAE_FEATURE_DIM;
const SPIKE_TOP_K: usize = 16;
const SCALE: u64 = 8;

#[derive(Debug, Clone, Copy)]
pub struct CandleFeatureExtractor {
    seed: u64,
}

impl CandleFeatureExtractor {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn check_budget(work_units: u64, budget: ComputeBudget) -> Result<(), ComputeError> {
        let elapsed_micros = work_units / SCALE;
        if work_units > budget.max_micros.saturating_mul(SCALE) {
            return Err(ComputeError::BudgetExceeded {
                stage: "sae/extract",
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    fn input_vector(input: &ComputeInput, world: &WorldModelOutput, seed: u64) -> [f32; IN_DIM] {
        let mut x = [0.0_f32; IN_DIM];
        for (i, value) in x.iter_mut().enumerate() {
            let u = input.context_digest[i % 32] as f32;
            let w = world.prediction_digest[(i + (seed as usize % 7)) % 32] as f32;
            *value = ((0.8 * (u / 255.0)) + (0.2 * (w / 255.0))).clamp(0.0, 1.0);
        }
        x
    }
}

impl Default for CandleFeatureExtractor {
    fn default() -> Self {
        Self::new(ComputeBudget::default().seed)
    }
}

impl FeatureExtractor for CandleFeatureExtractor {
    fn name(&self) -> &'static str {
        "candle_feature_extractor_v0"
    }

    fn extract(
        &self,
        input: &ComputeInput,
        world: &WorldModelOutput,
        budget: ComputeBudget,
    ) -> Result<SaeOutput, ComputeError> {
        Self::check_budget(1, budget)?;
        Self::check_budget((IN_DIM * OUT_DIM) as u64, budget)?;

        let x = Self::input_vector(input, world, self.seed);

        let device = Device::Cpu;
        let w = Tensor::from_slice(&weights_flat(), (OUT_DIM, IN_DIM), &device).map_err(|e| {
            ComputeError::Internal {
                reason: e.to_string(),
            }
        })?;
        let b = Tensor::from_slice(&B, OUT_DIM, &device).map_err(|e| ComputeError::Internal {
            reason: e.to_string(),
        })?;
        let x = Tensor::from_slice(&x, IN_DIM, &device)
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?
            .reshape((1, IN_DIM))
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?;

        let y = w
            .broadcast_mul(&x)
            .and_then(|v| v.sum(1))
            .and_then(|v| v.broadcast_add(&b))
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?;

        let mut yv = y.to_vec1::<f32>().map_err(|e| ComputeError::Internal {
            reason: e.to_string(),
        })?;

        let max_abs = yv
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, |acc, value| acc.max(value))
            .max(1e-6);

        let mut idx: Vec<usize> = (0..OUT_DIM).collect();
        idx.sort_by(|&a, &b| yv[b].abs().total_cmp(&yv[a].abs()));
        let top_k = SPIKE_TOP_K.min(SAE_MAX_SPIKES).min(idx.len());
        let threshold = idx
            .get(top_k.saturating_sub(1))
            .map(|i| yv[*i].abs())
            .unwrap_or(0.0);
        for v in &mut yv {
            if v.abs() < threshold {
                *v = 0.0;
            }
            *v = v.clamp(-1.0, 1.0);
        }

        let spikes = yv
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .map(|(feature_idx, value)| Spike {
                feature_id: feature_idx as u32,
                magnitude: (value.abs() / max_abs).clamp(0.0, 1.0),
                timestamp: input.t,
            })
            .collect::<Vec<_>>();

        let zeros = yv.iter().filter(|v| **v == 0.0).count();
        let sparsity = (zeros as f32 / yv.len() as f32).clamp(0.0, 1.0);
        let energy = (yv.iter().map(|v| v.abs()).sum::<f32>() / yv.len() as f32).clamp(0.0, 1.0);

        Ok(SaeOutput {
            feature_vec: FeatureVector {
                features: yv,
                digest: WEIGHTS_DIGEST,
            },
            spikes,
            sparsity,
            energy,
        })
    }
}

const B: [f32; OUT_DIM] = [
    0.04, 0.01, 0.03, 0.02, 0.03, 0.02, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03, 0.04,
    0.04, 0.01, 0.03, 0.02, 0.03, 0.02, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03, 0.04,
    0.04, 0.01, 0.03, 0.02, 0.03, 0.02, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03, 0.04,
    0.04, 0.01, 0.03, 0.02, 0.03, 0.02, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03, 0.04,
];

fn weights_flat() -> [f32; OUT_DIM * IN_DIM] {
    let mut w = [0.0_f32; OUT_DIM * IN_DIM];
    let mut i = 0;
    while i < OUT_DIM * IN_DIM {
        let phase = (i % 7) as f32;
        w[i] = 0.011 + phase * 0.001;
        i += 1;
    }
    w
}

const WEIGHTS_DIGEST: [u8; 32] = [
    0x2f, 0x3b, 0x44, 0x4d, 0x52, 0x63, 0x71, 0x80, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xee, 0xfc, 0x01,
    0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78, 0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0xf0, 0x0f,
];

#[cfg(test)]
mod tests {
    use crate::capabilities::WorldModelPredictor;
    use crate::world_model::{obs_features_from_context, MockJepaPredictor, WorldModelInput};
    use crate::FrameId;

    use super::*;

    fn input() -> ComputeInput {
        ComputeInput {
            frame_id: FrameId(7),
            t: 9,
            context_digest: [0x2a; 32],
        }
    }

    fn world(input: &ComputeInput) -> WorldModelOutput {
        let mut predictor = MockJepaPredictor::default();
        predictor
            .step(
                &WorldModelInput {
                    t: input.t,
                    context_digest: input.context_digest,
                    obs_features: obs_features_from_context(input.context_digest),
                    seed: 77,
                },
                ComputeBudget::default(),
            )
            .expect("world")
    }

    #[test]
    fn deterministic_for_same_seed_and_input() {
        let backend = CandleFeatureExtractor::new(123);
        let input = input();
        let world = world(&input);
        let budget = ComputeBudget::default();
        let a = backend.extract(&input, &world, budget).expect("compute a");
        let b = backend.extract(&input, &world, budget).expect("compute b");
        assert_eq!(a, b);
    }

    #[test]
    fn bounded_outputs_respected() {
        let backend = CandleFeatureExtractor::default();
        let input = input();
        let out = backend
            .extract(&input, &world(&input), ComputeBudget::default())
            .expect("compute");
        assert!(out.spikes.len() <= SAE_MAX_SPIKES);
        assert!((0.0..=1.0).contains(&out.sparsity));
        assert!((0.0..=1.0).contains(&out.energy));
    }

    #[test]
    fn budget_enforced() {
        let backend = CandleFeatureExtractor::default();
        let input = input();
        let err = backend
            .extract(
                &input,
                &world(&input),
                ComputeBudget {
                    max_micros: 1,
                    hard_timeout_micros: 1,
                    seed: 0,
                },
            )
            .expect_err("should fail budget");
        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "sae/extract"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
