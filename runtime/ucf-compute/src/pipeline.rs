use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};

use crate::capabilities::{SaeExtractor, WorkingMemoryModel, WorldModelPredictor};
use crate::feature_extractor::{SaeOutput, ToySaeExtractor};
use crate::ssm::{MockSsmSelectiveScan, SsmOutput};
use crate::work_meter::WorkMeter;
use crate::world_model::{
    obs_features_from_context, MockJepaPredictor, StageQuality, WorldModelInput, WorldModelOutput,
};
use crate::{
    clamp01, fuse_signals, validate_risk_signal, AiComputeBackend, BackendProfileId, ComputeBudget,
    ComputeError, ComputeInput, ComputeSignals, DegradePolicy, EvidenceRef, RiskSignal,
    SignalQuality,
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
    world: Arc<Mutex<dyn WorldModelPredictor + Send + Sync>>,
    sae: Arc<dyn SaeExtractor + Send + Sync>,
    ssm: Arc<dyn WorkingMemoryModel + Send + Sync>,
    fusion: FusionConfig,
    _limits: LimitsConfig,
}

impl ComputePipelineBackend {
    pub fn new(
        backend_name: &'static str,
        world: Arc<Mutex<dyn WorldModelPredictor + Send + Sync>>,
        sae: Arc<dyn SaeExtractor + Send + Sync>,
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
            _limits: limits,
        }
    }

    pub fn stub() -> Self {
        Self::new(
            "stub",
            Arc::new(Mutex::new(MockJepaPredictor::default())),
            Arc::new(ToySaeExtractor::default()),
            Arc::new(MockSsmSelectiveScan),
            FusionConfig::default(),
            LimitsConfig::default(),
        )
    }

    pub(crate) fn empty_sae() -> SaeOutput {
        SaeOutput {
            spikes: Vec::new(),
            spike_count: 0,
            sparsity: 1.0,
            energy: 0.0,
            spikes_digest: [0_u8; 32],
            quality: StageQuality::DegradedFallback,
            notes: crate::feature_extractor::SmallNotes(vec!["degraded:empty".to_string()]),
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

        let mut global_meter = WorkMeter::new(budget.global_work_units);
        let mut world_meter = WorkMeter::new(budget.world_units);
        let mut sae_meter = WorkMeter::new(budget.sae_units);
        let mut ssm_meter = WorkMeter::new(budget.ssm_units);

        let mut exceeded_stage: Option<&'static str> = None;

        global_meter.spend(40, "world_model/step")?;
        world_meter.spend(40, "world_model/step")?;
        let mut world = self.world.lock().map_err(|_| ComputeError::InvalidInput {
            reason: "world model mutex poisoned".to_string(),
        })?;
        let world_model_name = world.name();
        let world_input = WorldModelInput {
            t: input.t,
            context_digest: input.context_digest,
            obs_features: obs_features_from_context(input.context_digest),
            seed: budget.seed,
        };
        let world_model_out = match world.step(&world_input, budget) {
            Ok(output) => output,
            Err(ComputeError::BudgetExceeded { stage, .. }) => {
                exceeded_stage = Some(stage);
                if budget.degrade_policy == DegradePolicy::FailFast {
                    let mut unavailable = ComputeSignals::unavailable(input, budget, self.name());
                    unavailable.budget_exceeded_stage = Some(stage);
                    return Ok(unavailable);
                }
                WorldModelOutput::degraded_budget(stage)
            }
            Err(other) => return Err(other),
        };
        let span = tracing::info_span!("world_model.step", predictor = world_model_name, t = input.t, pred = %hex::encode(&world_model_out.prediction_digest[..4]));
        let _enter = span.enter();
        drop(world);
        let surprise = world_model_out.surprise;
        metrics::histogram!("ucf_world_prediction_error")
            .record(f64::from(world_model_out.prediction_error));
        metrics::histogram!("ucf_world_surprise").record(f64::from(world_model_out.surprise));
        if world_model_out.quality == StageQuality::DegradedFallback {
            metrics::counter!("ucf_world_degraded_total").increment(1);
        }

        global_meter.spend(220, "sae/extract")?;
        let evidence_seed: [u8; 32] = Sha256::digest(input.context_digest).into();
        let sae_input =
            ToySaeExtractor::make_input(input, &world_model_out, budget.seed, evidence_seed);
        let sae_span = tracing::info_span!("sae.extract", extractor = self.sae.name(), t = input.t);
        let _sae_enter = sae_span.enter();
        let (sae_out, sae_degraded) = match sae_meter.spend(220, "sae/extract") {
            Ok(()) => match self.sae.extract(&sae_input, budget) {
                Ok(output) => (output, false),
                Err(ComputeError::BudgetExceeded { stage, .. }) => {
                    exceeded_stage = Some(stage);
                    if budget.degrade_policy == DegradePolicy::FailFast {
                        return Ok(ComputeSignals::unavailable(input, budget, self.name()));
                    }
                    (Self::empty_sae(), true)
                }
                Err(other) => return Err(other),
            },
            Err(ComputeError::BudgetExceeded { stage, .. }) => {
                exceeded_stage = Some(stage);
                if budget.degrade_policy == DegradePolicy::FailFast {
                    return Ok(ComputeSignals::unavailable(input, budget, self.name()));
                }
                (Self::empty_sae(), true)
            }
            Err(other) => return Err(other),
        };

        metrics::histogram!("ucf_sae_spike_count").record(f64::from(sae_out.spike_count));
        metrics::gauge!("ucf_sae_sparsity").set(f64::from(sae_out.sparsity));
        metrics::histogram!("ucf_sae_energy").record(f64::from(sae_out.energy));
        if sae_out.quality == StageQuality::DegradedFallback {
            metrics::counter!("ucf_sae_degraded_total").increment(1);
        }

        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&input.context_digest[0..8]);
        let context_seed = u64::from_le_bytes(seed_bytes);
        let ssm_state = self
            .ssm
            .init(input, budget.seed ^ context_seed.rotate_left(9));

        global_meter.spend(220, "ssm/step")?;
        let (ssm_out, ssm_degraded) = match ssm_meter.spend(220, "ssm/step") {
            Ok(()) => match self
                .ssm
                .step(&ssm_state, &sae_out, &world_model_out, budget)
            {
                Ok(output) => (output, false),
                Err(ComputeError::BudgetExceeded { stage, .. }) => {
                    exceeded_stage = Some(stage);
                    if budget.degrade_policy == DegradePolicy::FailFast {
                        return Ok(ComputeSignals::unavailable(input, budget, self.name()));
                    }
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
            },
            Err(ComputeError::BudgetExceeded { stage, .. }) => {
                exceeded_stage = Some(stage);
                if budget.degrade_policy == DegradePolicy::FailFast {
                    return Ok(ComputeSignals::unavailable(input, budget, self.name()));
                }
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

        let spikes_digest_ref = sae_out.spikes_digest;

        let quality = if world_model_out.quality != StageQuality::Ok || sae_degraded || ssm_degraded
        {
            SignalQuality::DegradedFallback
        } else {
            SignalQuality::VerifiedPipeline
        };
        let evidence = EvidenceRef {
            context_digest: input.context_digest,
            world_digest: Some(world_model_out.prediction_digest),
            spikes_digest: Some(spikes_digest_ref),
            ssm_digest: if ssm_degraded {
                None
            } else {
                Some(ssm_out.next_state.digest)
            },
            backend_profile: BackendProfileId::from_backend_name(self.name()),
            seed: budget.seed,
            budget_profile_id: budget.profile_id,
        };
        let mut risk_signal = RiskSignal {
            risk: clamp01(risk),
            confidence: clamp01(confidence),
            quality,
            evidence,
            version: 1,
        };
        if validate_risk_signal(&risk_signal).is_err() {
            risk_signal.risk = 1.0;
            risk_signal.confidence = 0.0;
            risk_signal.quality = SignalQuality::Unavailable;
        }

        let summary = ComputeSignals {
            surprise,
            pressure,
            risk: risk_signal.risk,
            confidence: risk_signal.confidence,
            risk_signal,
            spikes: sae_out.spikes.clone(),
            notes: Vec::new(),
            sparsity: Some(sae_out.sparsity),
            energy: Some(sae_out.energy),
            ssm_readout: Some(ssm_out.readout),
            ssm_digest: if ssm_degraded {
                None
            } else {
                Some(ssm_out.next_state.digest)
            },
            world_digest: Some(world_model_out.prediction_digest),
            sae_quality: Some(sae_out.quality),
            budget_exceeded_stage: exceeded_stage,
        }
        .summary(self.name());

        let mut notes = vec![
            format!("backend={}", self.name()),
            format!("frame={}", input.frame_id.0),
            format!("budget_profile_id={}", budget.profile_id),
            format!("world_model={}", world_model_name),
            format!("feature_extractor={}", self.sae.name()),
            format!("working_memory={}", self.ssm.name()),
            format!(
                "pred_digest={}",
                &hex::encode(world_model_out.prediction_digest)[..12]
            ),
            format!("pred_error={:.4}", world_model_out.prediction_error),
            format!("world_quality={:?}", world_model_out.quality),
            format!("sae_quality={:?}", sae_out.quality),
            format!(
                "ssm_digest={}",
                &hex::encode(ssm_out.next_state.digest)[..12]
            ),
            format!("spike_count={}", sae_out.spike_count),
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

        if let Some(stage) = exceeded_stage {
            notes.push(format!("budget_exceeded_stage={stage}"));
        }
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
            risk: risk_signal.risk,
            confidence: risk_signal.confidence,
            risk_signal,
            spikes: sae_out.spikes,
            notes,
            sparsity: Some(sae_out.sparsity),
            energy: Some(sae_out.energy),
            ssm_readout: Some(ssm_out.readout),
            ssm_digest: if ssm_degraded {
                None
            } else {
                Some(ssm_out.next_state.digest)
            },
            world_digest: Some(world_model_out.prediction_digest),
            sae_quality: Some(sae_out.quality),
            budget_exceeded_stage: exceeded_stage,
        }
        .bounded())
    }
}

#[cfg(test)]
mod tests {
    use crate::feature_extractor::ToySaeExtractor;
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
        let budget = ComputeBudget::default();
        let a = ComputePipelineBackend::stub()
            .compute(&input(), budget)
            .expect("compute");
        let b = ComputePipelineBackend::stub()
            .compute(&input(), budget)
            .expect("compute");
        assert_eq!(a, b);
    }

    #[test]
    fn keeps_expected_capability_notes() {
        let backend = ComputePipelineBackend::new(
            "stub",
            Arc::new(Mutex::new(MockJepaPredictor::default())),
            Arc::new(ToySaeExtractor::default()),
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
            .any(|n| n == "feature_extractor=toy_sae_v0"));
        assert!(out
            .notes
            .iter()
            .any(|n| n == "working_memory=mock_ssm_selective_scan_v0"));
    }

    #[test]
    fn stress_profile_triggers_degraded_quality() {
        let backend = ComputePipelineBackend::stub();
        let budget = ComputeBudget {
            sae_units: 100,
            global_work_units: 900,
            profile_id: 3,
            ..ComputeBudget::default()
        };
        let out = backend.compute(&input(), budget).expect("compute");
        assert_eq!(out.risk_signal.quality, SignalQuality::DegradedFallback);
        assert_eq!(out.budget_exceeded_stage, Some("sae/extract"));
    }

    #[test]
    fn fail_fast_profile_yields_unavailable() {
        let backend = ComputePipelineBackend::stub();
        let budget = ComputeBudget {
            sae_units: 100,
            degrade_policy: DegradePolicy::FailFast,
            ..ComputeBudget::default()
        };
        let out = backend.compute(&input(), budget).expect("compute");
        assert_eq!(out.risk_signal.quality, SignalQuality::Unavailable);
        assert_eq!(out.risk, 1.0);
        assert_eq!(out.confidence, 0.0);
    }
}
