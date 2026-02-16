use sha2::{Digest, Sha256};

use crate::capabilities::WorkingMemoryModel;
use crate::feature_extractor::SaeOutput;
use crate::world_model::WorldModelOutput;
use crate::{ComputeBudget, ComputeError, ComputeInput, SplitMix64};

pub const SSM_MEM_DIM: usize = 16;
pub const SSM_MAX_MEM: usize = 32;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SsmState {
    pub mem: Vec<f32>,
    pub t: u64,
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SsmOutput {
    pub next_state: SsmState,
    pub pressure: f32,
    pub readout: f32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MockSsmSelectiveScan;

impl MockSsmSelectiveScan {
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
                stage: "ssm/step",
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    fn digest_mem(mem: &[f32], t: u64) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for value in mem {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(t.to_le_bytes());
        let digest = hasher.finalize();
        let mut out = [0_u8; 32];
        out.copy_from_slice(&digest);
        out
    }
}

impl WorkingMemoryModel for MockSsmSelectiveScan {
    fn name(&self) -> &'static str {
        "mock_ssm_selective_scan_v0"
    }

    fn init(&self, input: &ComputeInput, seed: u64) -> SsmState {
        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&input.context_digest[0..8]);
        let mut prng =
            SplitMix64::new(u64::from_le_bytes(seed_bytes) ^ seed ^ input.t.rotate_left(13));
        let mem: Vec<f32> = (0..SSM_MEM_DIM)
            .map(|idx| {
                let base = (input.context_digest[idx % input.context_digest.len()] as f32) / 255.0;
                (0.6 * base + 0.4 * Self::normalized_unit(prng.next_u64())).clamp(0.0, 1.0)
            })
            .collect();
        let digest = Self::digest_mem(&mem, input.t);
        SsmState {
            mem,
            t: input.t,
            digest,
        }
    }

    fn step(
        &self,
        state: &SsmState,
        sae: &SaeOutput,
        world: &WorldModelOutput,
        budget: ComputeBudget,
    ) -> Result<SsmOutput, ComputeError> {
        let mut work_units = 20_u64;
        Self::check_budget(work_units, budget)?;

        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&state.digest[0..8]);
        let mut prng = SplitMix64::new(u64::from_le_bytes(seed_bytes) ^ budget.seed);

        let u0 = (0.5 * sae.energy + 0.5 * world.surprise).clamp(0.0, 1.0);

        let mut mem = Vec::with_capacity(state.mem.len().min(SSM_MAX_MEM));
        for prev in state.mem.iter().take(SSM_MAX_MEM) {
            work_units = work_units.saturating_add(4);
            Self::check_budget(work_units, budget)?;

            let a = (0.55 + 0.35 * Self::normalized_unit(prng.next_u64())).clamp(0.0, 1.0);
            let b = (0.10 + 0.30 * Self::normalized_unit(prng.next_u64())).clamp(0.0, 1.0);
            let jitter = 0.03 * Self::centered_unit(prng.next_u64());
            let next = (a * *prev + b * u0 + jitter).clamp(0.0, 1.0);
            mem.push(next);
        }

        let readout = (mem.iter().sum::<f32>() / mem.len().max(1) as f32).clamp(0.0, 1.0);
        let mean = readout;
        let variance = mem
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / mem.len().max(1) as f32;
        let pressure = (variance * 3.5 + u0 * 0.65).clamp(0.0, 1.0);

        let next_t = state.t.saturating_add(1);
        let digest = Self::digest_mem(&mem, next_t);

        Ok(SsmOutput {
            next_state: SsmState {
                mem,
                t: next_t,
                digest,
            },
            pressure,
            readout,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::capabilities::{FeatureExtractor, WorldModelPredictor};
    use crate::feature_extractor::MockSaeExtractor;
    use crate::world_model::{obs_features_from_context, MockJepaPredictor, WorldModelInput};
    use crate::FrameId;

    use super::*;

    fn input() -> ComputeInput {
        ComputeInput {
            frame_id: FrameId(91),
            t: 9,
            context_digest: [0x33; 32],
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
                    seed: 4,
                },
                ComputeBudget::default(),
            )
            .expect("world")
    }

    #[test]
    fn init_and_step_are_deterministic() {
        let input = input();
        let world = world(&input);
        let sae = MockSaeExtractor
            .extract(&input, &world, ComputeBudget::default())
            .expect("sae");

        let ssm = MockSsmSelectiveScan;
        let state_a = ssm.init(&input, 7);
        let state_b = ssm.init(&input, 7);
        assert_eq!(state_a, state_b);

        let out_a = ssm
            .step(&state_a, &sae, &world, ComputeBudget::default())
            .expect("step");
        let out_b = ssm
            .step(&state_b, &sae, &world, ComputeBudget::default())
            .expect("step");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn pressure_stays_bounded() {
        let input = input();
        let world = world(&input);
        let sae = MockSaeExtractor
            .extract(&input, &world, ComputeBudget::default())
            .expect("sae");
        let ssm = MockSsmSelectiveScan;
        let state = ssm.init(&input, 3);
        let out = ssm
            .step(&state, &sae, &world, ComputeBudget::default())
            .expect("step");

        assert!((0.0..=1.0).contains(&out.pressure));
        assert!((0.0..=1.0).contains(&out.readout));
    }

    #[test]
    fn budget_exceeded_uses_expected_stage() {
        let input = input();
        let world = world(&input);
        let sae = MockSaeExtractor
            .extract(&input, &world, ComputeBudget::default())
            .expect("sae");
        let ssm = MockSsmSelectiveScan;
        let state = ssm.init(&input, 9);
        let err = ssm
            .step(
                &state,
                &sae,
                &world,
                ComputeBudget {
                    max_micros: 1,
                    hard_timeout_micros: 1,
                    seed: 9,
                    ..ComputeBudget::default()
                },
            )
            .expect_err("budget exceeded");

        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "ssm/step"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
