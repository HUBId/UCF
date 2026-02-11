use sha2::{Digest, Sha256};

use crate::capabilities::WorldModelPredictor;
use crate::{ComputeBudget, ComputeError, ComputeInput, SplitMix64};

pub const WORLD_MODEL_LATENT_DIM: usize = 32;
pub const WORLD_MODEL_MAX_LATENT: usize = 64;
pub const WORLD_MODEL_MAX_DEBUG: usize = 8;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorldState {
    pub latent: Vec<f32>,
    pub t: u64,
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Prediction {
    pub next_latent: Vec<f32>,
    pub next_t: u64,
    pub prediction_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PredictionError {
    pub l2: f32,
    pub normalized: f32,
    pub surprise: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorldModelOutput {
    pub prediction: Prediction,
    pub error: PredictionError,
    pub debug: Vec<String>,
}

impl WorldModelOutput {
    pub fn bounded(mut self) -> Self {
        if self.prediction.next_latent.len() > WORLD_MODEL_MAX_LATENT {
            self.prediction.next_latent.truncate(WORLD_MODEL_MAX_LATENT);
        }
        if self.debug.len() > WORLD_MODEL_MAX_DEBUG {
            self.debug.truncate(WORLD_MODEL_MAX_DEBUG);
        }
        self
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MockJepaPredictor;

impl MockJepaPredictor {
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
                stage: "world_model/predict",
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    fn digest_latent(latent: &[f32], t: u64) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for value in latent {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(t.to_le_bytes());
        let digest = hasher.finalize();
        let mut out = [0_u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    fn make_observation(input: &ComputeInput) -> [f32; 32] {
        let mut obs = [0_f32; 32];
        for (idx, byte) in input.context_digest.iter().enumerate() {
            obs[idx] = (*byte as f32) / 255.0;
        }
        obs
    }
}

impl WorldModelPredictor for MockJepaPredictor {
    fn name(&self) -> &'static str {
        "mock_jepa_v0"
    }

    fn init_state(&self, input: &ComputeInput, seed: u64) -> WorldState {
        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&input.context_digest[0..8]);
        let context_seed = u64::from_le_bytes(seed_bytes);

        let mut prng =
            SplitMix64::new(context_seed ^ seed ^ input.t.rotate_left(11) ^ 0x4A45_5041_7630_0001);
        let latent: Vec<f32> = (0..WORLD_MODEL_LATENT_DIM)
            .map(|idx| {
                let jitter = (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                Self::centered_unit(prng.next_u64() ^ jitter)
            })
            .collect();

        WorldState {
            digest: Self::digest_latent(&latent, input.t),
            latent,
            t: input.t,
        }
    }

    fn predict(
        &self,
        state: &WorldState,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<WorldModelOutput, ComputeError> {
        let mut work_units = 24_u64;
        Self::check_budget(work_units, budget)?;

        let obs = Self::make_observation(input);

        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&state.digest[0..8]);
        let mut prng = SplitMix64::new(
            u64::from_le_bytes(seed_bytes) ^ budget.seed ^ state.t.rotate_left(5) ^ input.t,
        );

        let a = 0.65 + 0.25 * Self::normalized_unit(prng.next_u64());
        let b = 0.15 + 0.2 * Self::normalized_unit(prng.next_u64());
        let bias = 0.05 * Self::centered_unit(prng.next_u64());

        let mut next_latent = Vec::with_capacity(WORLD_MODEL_LATENT_DIM);
        for (idx, obs_value) in obs.iter().enumerate().take(WORLD_MODEL_LATENT_DIM) {
            work_units = work_units.saturating_add(5);
            Self::check_budget(work_units, budget)?;
            let prev = state.latent.get(idx).copied().unwrap_or(0.0);
            let candidate = prev * a + *obs_value * b + bias;
            next_latent.push(candidate.clamp(-1.0, 1.0));
        }

        let l2 = next_latent
            .iter()
            .zip(state.latent.iter().copied().chain(std::iter::repeat(0.0)))
            .take(WORLD_MODEL_LATENT_DIM)
            .map(|(next, prev)| {
                let d = next - prev;
                d * d
            })
            .sum::<f32>()
            / (WORLD_MODEL_LATENT_DIM as f32);

        let normalized = (l2 * 6.0).clamp(0.0, 1.0);
        let surprise = normalized;

        let next_t = input.t.saturating_add(1);
        let prediction_digest = Self::digest_latent(&next_latent, next_t);

        Ok(WorldModelOutput {
            prediction: Prediction {
                next_latent,
                next_t,
                prediction_digest,
            },
            error: PredictionError {
                l2,
                normalized,
                surprise,
            },
            debug: vec![
                format!("predictor={}", self.name()),
                format!("state_t={}", state.t),
                format!("next_t={next_t}"),
            ],
        }
        .bounded())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameId;

    fn input() -> ComputeInput {
        ComputeInput {
            frame_id: FrameId(7),
            t: 12,
            context_digest: [0xAB; 32],
        }
    }

    #[test]
    fn init_state_and_predict_are_deterministic() {
        let predictor = MockJepaPredictor;
        let input = input();
        let budget = ComputeBudget::default();

        let state_a = predictor.init_state(&input, 123);
        let state_b = predictor.init_state(&input, 123);
        assert_eq!(state_a, state_b);

        let out_a = predictor
            .predict(&state_a, &input, budget)
            .expect("predict");
        let out_b = predictor
            .predict(&state_b, &input, budget)
            .expect("predict");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn bounded_shapes_and_debug() {
        let predictor = MockJepaPredictor;
        let state = predictor.init_state(&input(), 5);
        let out = predictor
            .predict(&state, &input(), ComputeBudget::default())
            .expect("predict");

        assert_eq!(state.latent.len(), WORLD_MODEL_LATENT_DIM);
        assert_eq!(out.prediction.next_latent.len(), WORLD_MODEL_LATENT_DIM);
        assert!(out.debug.len() <= WORLD_MODEL_MAX_DEBUG);
    }

    #[test]
    fn error_values_stay_in_range() {
        let predictor = MockJepaPredictor;
        let state = predictor.init_state(&input(), 42);
        let out = predictor
            .predict(&state, &input(), ComputeBudget::default())
            .expect("predict");

        assert!(out.error.l2 >= 0.0);
        assert!((0.0..=1.0).contains(&out.error.normalized));
        assert!((0.0..=1.0).contains(&out.error.surprise));
    }

    #[test]
    fn budget_exceeded_reports_world_model_stage() {
        let predictor = MockJepaPredictor;
        let state = predictor.init_state(&input(), 99);
        let err = predictor
            .predict(
                &state,
                &input(),
                ComputeBudget {
                    max_micros: 1,
                    hard_timeout_micros: 1,
                    seed: 99,
                    ..ComputeBudget::default()
                },
            )
            .expect_err("budget should exceed");

        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "world_model/predict"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
