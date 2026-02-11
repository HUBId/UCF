use std::sync::Arc;

use crate::capabilities::{FeatureExtractor, WorkingMemoryModel, WorldModelPredictor};
use crate::feature_extractor::{FeatureVector, MockSaeExtractor, SaeOutput, SAE_FEATURE_DIM};
use crate::ssm::{MockSsmSelectiveScan, SsmOutput};
use crate::world_model::MockJepaPredictor;
use crate::{
    fuse_signals, AiComputeBackend, ComputeBudget, ComputeError, ComputeInput, ComputeSignals,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionConfig {
    pub world_weight: f32,
    pub ssm_weight: f32,
    pub energy_weight: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            world_weight: 1.0,
            ssm_weight: 1.0,
            energy_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitsConfig {
    pub budget_scale: u64,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self { budget_scale: 8 }
    }
}

pub struct ComputePipelineBackend {
    backend_name: &'static str,
    world: Arc<dyn WorldModelPredictor + Send + Sync>,
    sae: Arc<dyn FeatureExtractor + Send + Sync>,
    ssm: Arc<dyn WorkingMemoryModel + Send + Sync>,
    fusion: FusionConfig,
    limits: LimitsConfig,
}

impl ComputePipelineBackend {
    pub fn new(
        backend_name: &'static str,
        world: Arc<dyn WorldModelPredictor + Send + Sync>,
        sae: Arc<dyn FeatureExtractor + Send + Sync>,
        ssm: Arc<dyn WorkingMemoryModel + Send + Sync>,
        fusion: FusionConfig,
        limits: LimitsConfig,
    ) -> Self {
        Self {
            backend_name,
            world,
            sae,
            ssm,
            fusion,
            limits,
        }
    }

    pub fn stub() -> Self {
        Self::new(
            "stub",
            Arc::new(MockJepaPredictor),
            Arc::new(MockSaeExtractor),
            Arc::new(MockSsmSelectiveScan),
            FusionConfig::default(),
            LimitsConfig::default(),
        )
    }

    pub(crate) fn stage_budget(total: ComputeBudget, num: u64, den: u64) -> ComputeBudget {
        let max_micros = ((total.max_micros.saturating_mul(num)) / den).max(1);
        let hard_timeout_micros = ((total.hard_timeout_micros.saturating_mul(num)) / den).max(1);
        ComputeBudget {
            max_micros,
            hard_timeout_micros,
            seed: total.seed,
        }
    }

    fn check_budget(
        &self,
        work_units: u64,
        stage: &'static str,
        budget: ComputeBudget,
    ) -> Result<(), ComputeError> {
        let elapsed_micros = work_units / self.limits.budget_scale.max(1);
        if work_units
            > budget
                .max_micros
                .saturating_mul(self.limits.budget_scale.max(1))
        {
            return Err(ComputeError::BudgetExceeded {
                stage,
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    pub(crate) fn empty_sae() -> SaeOutput {
        SaeOutput {
            feature_vec: FeatureVector {
                features: vec![0.0; SAE_FEATURE_DIM],
                digest: [0_u8; 32],
            },
            spikes: Vec::new(),
            sparsity: 1.0,
            energy: 0.0,
        }
    }
}

impl AiComputeBackend for ComputePipelineBackend {
    fn name(&self) -> &'static str {
        self.backend_name
    }

    fn compute(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<ComputeSignals, ComputeError> {
        if input.t == 0 {
            return Err(ComputeError::InvalidInput {
                reason: "t must be non-zero".to_string(),
            });
        }

        self.check_budget(1, "pipeline/start", budget)?;

        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&input.context_digest[0..8]);
        let context_seed = u64::from_le_bytes(seed_bytes);

        let world_budget = Self::stage_budget(budget, 4, 10);
        let sae_budget = Self::stage_budget(budget, 3, 10);
        let ssm_budget = Self::stage_budget(budget, 3, 10);

        let state = self.world.init_state(input, budget.seed);
        let world_model_out = self.world.predict(&state, input, world_budget)?;
        let surprise = world_model_out.error.surprise;

        let (sae_out, sae_degraded) = match self.sae.extract(input, &world_model_out, sae_budget) {
            Ok(output) => (output, false),
            Err(ComputeError::BudgetExceeded { .. }) => (Self::empty_sae(), true),
            Err(other) => return Err(other),
        };

        let ssm_state = self
            .ssm
            .init(input, budget.seed ^ context_seed.rotate_left(9));
        let (ssm_out, ssm_degraded) =
            match self
                .ssm
                .step(&ssm_state, &sae_out, &world_model_out, ssm_budget)
            {
                Ok(output) => (output, false),
                Err(ComputeError::BudgetExceeded { .. }) => {
                    let fallback_pressure = surprise.clamp(0.0, 1.0);
                    (
                        SsmOutput {
                            next_state: ssm_state,
                            pressure: fallback_pressure,
                            readout: fallback_pressure,
                        },
                        true,
                    )
                }
                Err(other) => return Err(other),
            };

        let pressure = ssm_out.pressure;
        let (risk, confidence) = fuse_signals(
            surprise * self.fusion.world_weight,
            pressure * self.fusion.ssm_weight,
            sae_out.energy * self.fusion.energy_weight,
        );

        let summary = ComputeSignals {
            surprise,
            pressure,
            risk,
            confidence,
            spikes: sae_out.spikes.clone(),
            notes: Vec::new(),
            sparsity: Some(sae_out.sparsity),
            energy: Some(sae_out.energy),
            ssm_readout: Some(ssm_out.readout),
            ssm_digest: Some(ssm_out.next_state.digest),
        }
        .summary(self.name());

        let mut notes = vec![
            format!("backend={}", self.name()),
            format!("frame={}", input.frame_id.0),
            format!("world_model={}", self.world.name()),
            format!("feature_extractor={}", self.sae.name()),
            format!("working_memory={}", self.ssm.name()),
            format!(
                "pred_digest={}",
                &hex::encode(world_model_out.prediction.prediction_digest)[..12]
            ),
            format!(
                "sae_digest={}",
                &hex::encode(sae_out.feature_vec.digest)[..12]
            ),
            format!(
                "ssm_digest={}",
                &hex::encode(ssm_out.next_state.digest)[..12]
            ),
            format!("spike_count={}", sae_out.spikes.len()),
            format!("sparsity={:.4}", sae_out.sparsity),
            format!("energy={:.4}", sae_out.energy),
            format!(
                "digest_prefix={:02x}{:02x}",
                input.context_digest[0], input.context_digest[1]
            ),
            format!(
                "spikes_digest={}",
                &hex::encode(summary.spikes_digest)[..12]
            ),
        ];
        if sae_degraded {
            notes.push("degraded:sae_budget_exceeded".to_string());
        }
        if ssm_degraded {
            notes.push("degraded:ssm_budget_exceeded".to_string());
        }
        notes.sort();

        Ok(ComputeSignals {
            surprise,
            pressure,
            risk,
            confidence,
            spikes: sae_out.spikes,
            notes,
            sparsity: Some(sae_out.sparsity),
            energy: Some(sae_out.energy),
            ssm_readout: Some(ssm_out.readout),
            ssm_digest: Some(ssm_out.next_state.digest),
        }
        .bounded())
    }
}

#[cfg(test)]
mod tests {
    use crate::feature_extractor::MockSaeExtractor;
    use crate::ssm::MockSsmSelectiveScan;
    use crate::world_model::MockJepaPredictor;
    use crate::FrameId;

    use super::*;

    fn input() -> ComputeInput {
        ComputeInput {
            frame_id: FrameId(42),
            t: 7,
            context_digest: [1_u8; 32],
        }
    }

    #[test]
    fn deterministic_for_same_input_and_seed() {
        let backend = ComputePipelineBackend::stub();
        let budget = ComputeBudget::default();
        let a = backend.compute(&input(), budget).expect("compute");
        let b = backend.compute(&input(), budget).expect("compute");
        assert_eq!(a, b);
    }

    #[test]
    fn keeps_expected_capability_notes() {
        let backend = ComputePipelineBackend::new(
            "stub",
            Arc::new(MockJepaPredictor),
            Arc::new(MockSaeExtractor),
            Arc::new(MockSsmSelectiveScan),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let out = backend
            .compute(&input(), ComputeBudget::default())
            .expect("compute");
        assert!(out.notes.iter().any(|n| n == "world_model=mock_jepa_v0"));
        assert!(out
            .notes
            .iter()
            .any(|n| n == "feature_extractor=mock_sae_v0"));
        assert!(out
            .notes
            .iter()
            .any(|n| n == "working_memory=mock_ssm_selective_scan_v0"));
    }
}
