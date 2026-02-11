use sha2::{Digest, Sha256};

use crate::capabilities::FeatureExtractor;
use crate::world_model::WorldModelOutput;
use crate::{ComputeBudget, ComputeError, ComputeInput, Spike, SplitMix64};

pub const SAE_FEATURE_DIM: usize = 64;
pub const SAE_MAX_FEATURES: usize = 128;
pub const SAE_TOP_K: usize = 16;
pub const SAE_MAX_SPIKES: usize = 64;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FeatureVector {
    pub features: Vec<f32>,
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SaeOutput {
    pub feature_vec: FeatureVector,
    pub spikes: Vec<Spike>,
    pub sparsity: f32,
    pub energy: f32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MockSaeExtractor;

impl MockSaeExtractor {
    fn normalized_unit(bits: u64) -> f32 {
        let v = (bits >> 40) as u32;
        (v as f32) / (u32::MAX as f32)
    }

    fn centered_unit(bits: u64) -> f32 {
        Self::normalized_unit(bits) * 2.0 - 1.0
    }

    fn check_budget(work_units: u64, budget: ComputeBudget) -> Result<(), ComputeError> {
        const SCALE: u64 = 8;
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

    fn digest_features(features: &[f32], t: u64) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for value in features {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(t.to_le_bytes());
        let digest = hasher.finalize();
        let mut out = [0_u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    fn top_k_threshold(values: &[f32], k: usize) -> f32 {
        if values.is_empty() || k == 0 {
            return f32::INFINITY;
        }
        let mut magnitudes: Vec<f32> = values.iter().map(|v| v.abs()).collect();
        magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if k > magnitudes.len() {
            0.0
        } else {
            magnitudes[k - 1]
        }
    }
}

impl FeatureExtractor for MockSaeExtractor {
    fn name(&self) -> &'static str {
        "mock_sae_v0"
    }

    fn extract(
        &self,
        input: &ComputeInput,
        world: &WorldModelOutput,
        budget: ComputeBudget,
    ) -> Result<SaeOutput, ComputeError> {
        let mut work_units = 16_u64;
        Self::check_budget(work_units, budget)?;

        let mut seed_hasher = Sha256::new();
        seed_hasher.update(world.prediction.prediction_digest);
        seed_hasher.update(input.context_digest);
        seed_hasher.update(input.t.to_le_bytes());
        seed_hasher.update(b"mock_sae_v0");
        let seed_digest = seed_hasher.finalize();

        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&seed_digest[0..8]);
        let mut prng = SplitMix64::new(u64::from_le_bytes(seed_bytes) ^ budget.seed);

        let x: Vec<f32> = (0..SAE_FEATURE_DIM)
            .map(|idx| {
                let source = input.context_digest[idx % input.context_digest.len()] as f32 / 255.0;
                let jitter = 0.2 * Self::centered_unit(prng.next_u64());
                (source + jitter).clamp(-1.0, 1.0)
            })
            .collect();

        let mut y = vec![0_f32; SAE_FEATURE_DIM];
        for (i, y_i) in y.iter_mut().enumerate() {
            work_units = work_units.saturating_add(5);
            Self::check_budget(work_units, budget)?;

            let mut acc = 0.0_f32;
            for (j, x_j) in x.iter().enumerate() {
                let hash = (budget.seed ^ ((i as u64) << 32) ^ (j as u64)).rotate_left(17)
                    ^ prng.next_u64();
                let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
                let scale = 0.05 + 0.15 * Self::normalized_unit(hash.rotate_left(7));
                acc += sign * scale * *x_j;
            }
            work_units = work_units.saturating_add(4);
            let bias = 0.1 * Self::centered_unit(prng.next_u64() ^ i as u64);
            *y_i = (acc / SAE_FEATURE_DIM as f32 + bias).clamp(-1.0, 1.0);
            Self::check_budget(work_units, budget)?;
        }

        let threshold = Self::top_k_threshold(&y, SAE_TOP_K);
        for value in &mut y {
            if value.abs() < threshold {
                *value = 0.0;
            }
        }

        let max_abs = y
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, |acc, value| acc.max(value))
            .max(1e-6);

        let mut spikes: Vec<Spike> = y
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .map(|(idx, value)| Spike {
                feature_id: idx as u32,
                magnitude: (value.abs() / max_abs).clamp(0.0, 1.0),
                timestamp: input.t,
            })
            .collect();
        spikes.sort_by(|a, b| a.feature_id.cmp(&b.feature_id));
        spikes.truncate(SAE_MAX_SPIKES);

        let zeros = y.iter().filter(|v| **v == 0.0).count();
        let sparsity = (zeros as f32 / y.len() as f32).clamp(0.0, 1.0);
        let energy = (y.iter().map(|v| v.abs()).sum::<f32>() / y.len() as f32).clamp(0.0, 1.0);

        let digest = Self::digest_features(&y, input.t);

        Ok(SaeOutput {
            feature_vec: FeatureVector {
                features: y.into_iter().take(SAE_MAX_FEATURES).collect(),
                digest,
            },
            spikes,
            sparsity,
            energy,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::capabilities::WorldModelPredictor;
    use crate::world_model::MockJepaPredictor;
    use crate::FrameId;

    use super::*;

    fn input() -> ComputeInput {
        ComputeInput {
            frame_id: FrameId(77),
            t: 23,
            context_digest: [0x5A; 32],
        }
    }

    fn world(input: &ComputeInput) -> WorldModelOutput {
        let predictor = MockJepaPredictor;
        let state = predictor.init_state(input, 11);
        predictor
            .predict(&state, input, ComputeBudget::default())
            .expect("world")
    }

    #[test]
    fn extraction_is_deterministic() {
        let input = input();
        let world = world(&input);
        let sae = MockSaeExtractor;

        let a = sae
            .extract(&input, &world, ComputeBudget::default())
            .expect("sae");
        let b = sae
            .extract(&input, &world, ComputeBudget::default())
            .expect("sae");
        assert_eq!(a, b);
    }

    #[test]
    fn sparsity_and_spike_bounds_hold() {
        let input = input();
        let world = world(&input);
        let sae = MockSaeExtractor;
        let out = sae
            .extract(&input, &world, ComputeBudget::default())
            .expect("sae");

        assert!((0.0..=1.0).contains(&out.sparsity));
        assert!((0.0..=1.0).contains(&out.energy));
        assert!(out.spikes.len() <= SAE_MAX_SPIKES);
        assert_eq!(out.feature_vec.features.len(), SAE_FEATURE_DIM);
        assert!(out
            .spikes
            .windows(2)
            .all(|w| w[0].feature_id <= w[1].feature_id));
    }

    #[test]
    fn budget_exceeded_uses_expected_stage() {
        let input = input();
        let world = world(&input);
        let sae = MockSaeExtractor;
        let err = sae
            .extract(
                &input,
                &world,
                ComputeBudget {
                    max_micros: 2,
                    hard_timeout_micros: 2,
                    seed: 1,
                },
            )
            .expect_err("budget exceeded");

        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "sae/extract"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
