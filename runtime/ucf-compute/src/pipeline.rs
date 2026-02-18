use sha2::{Digest, Sha256};
use std::sync::Arc;

use crate::backend_pack::{BackendPack, BackendPackFactory};
use crate::contracts::{
    validate_evidence_chain_digest, ContractRegistry, LfmValidatorV1, SaeValidatorV1,
    SsmValidatorV1, StageContractRegistry, StageContractVersion, StageKind, ValidationStatus,
    ViolationCode, WorldValidatorV1,
};
use crate::feature_extractor::{SaeOutput, ToySaeExtractor};
use crate::lfm::{LfmInput, LfmOutput};
use crate::ssm::{SsmInput, SsmOutput};
use crate::work_meter::WorkMeter;
use crate::world_model::{
    obs_features_from_context, StageQuality, WorldModelInput, WorldModelOutput,
};
use crate::{
    clamp01, fuse_signals, validate_risk_signal, AiComputeBackend, BackendPackConfig,
    BackendProfileId, ComputeBudget, ComputeError, ComputeInput, ComputeSignals, DegradePolicy,
    EvidenceRef, RiskSignal, SignalQuality,
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
    pack: Arc<dyn BackendPack>,
    fusion: FusionConfig,
    _limits: LimitsConfig,
}

impl ComputePipelineBackend {
    pub fn new(pack: Arc<dyn BackendPack>, fusion: FusionConfig, limits: LimitsConfig) -> Self {
        Self {
            pack,
            fusion,
            _limits: limits,
        }
    }

    pub fn stub() -> Self {
        let pack = BackendPackFactory::build(BackendPackConfig::default())
            .expect("default backend pack must build");
        Self::new(pack, FusionConfig::default(), LimitsConfig::default())
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
        self.pack.meta().pack_name
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
        let mut lfm_meter = WorkMeter::new(budget.lfm_units);

        let mut exceeded_stage: Option<&'static str> = None;
        let pack_meta = self.pack.meta();
        let registry = StageContractRegistry;
        let requested = StageContractVersion::V1;

        global_meter.spend(40, "world_model/step")?;
        world_meter.spend(40, "world_model/step")?;
        let mut world = self
            .pack
            .world()
            .lock()
            .map_err(|_| ComputeError::InvalidInput {
                reason: "world model mutex poisoned".to_string(),
            })?;
        let world_model_name = world.name();
        if !registry.supports(
            StageKind::World,
            pack_meta.world_backend,
            world.contract_version(),
        ) || world.contract_version() != requested
        {
            let mut unavailable = ComputeSignals::unavailable(input, budget, self.name());
            unavailable.validation_status = ValidationStatus::Degraded;
            unavailable.violation_reason_mask =
                1_u32 << (ViolationCode::BackendContractMismatch as u32);
            unavailable.backend_id = pack_meta.world_backend as u16;
            return Ok(unavailable);
        }
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
        let mut validation_report = WorldValidatorV1::validate(&world_input, &world_model_out);
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
        if !registry.supports(
            StageKind::Sae,
            pack_meta.sae_backend,
            self.pack.sae().contract_version(),
        ) || self.pack.sae().contract_version() != requested
        {
            validation_report.add_hard(ViolationCode::BackendContractMismatch);
            let mut out = ComputeSignals::unavailable(input, budget, self.name());
            out.backend_id = pack_meta.sae_backend as u16;
            out.violation_reason_mask = validation_report.violation_mask;
            return Ok(out);
        }
        let sae_span = tracing::info_span!(
            "sae.extract",
            extractor = self.pack.sae().name(),
            t = input.t
        );
        let _sae_enter = sae_span.enter();
        let (sae_out, sae_degraded) = match sae_meter.spend(220, "sae/extract") {
            Ok(()) => match self.pack.sae().extract(&sae_input, budget) {
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
        validation_report = validation_report.merge(SaeValidatorV1::validate(&sae_input, &sae_out));
        metrics::gauge!("ucf_sae_sparsity").set(f64::from(sae_out.sparsity));
        metrics::histogram!("ucf_sae_energy").record(f64::from(sae_out.energy));
        if sae_out.quality == StageQuality::DegradedFallback {
            metrics::counter!("ucf_sae_degraded_total").increment(1);
        }

        let ssm_input = SsmInput {
            t: input.t,
            spikes_digest: sae_out.spikes_digest,
            spike_count: sae_out.spike_count,
            sae_energy: sae_out.energy,
            world_surprise: world_model_out.surprise,
            risk: 0.0,
            seed: budget.seed,
            context_digest: input.context_digest,
        };

        global_meter.spend(220, "ssm/step")?;
        if !registry.supports(
            StageKind::Ssm,
            pack_meta.ssm_backend,
            self.pack
                .ssm()
                .lock()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: "ssm mutex poisoned".to_string(),
                })?
                .contract_version(),
        ) {
            validation_report.add_hard(ViolationCode::BackendContractMismatch);
            let mut out = ComputeSignals::unavailable(input, budget, self.name());
            out.backend_id = pack_meta.ssm_backend as u16;
            out.violation_reason_mask = validation_report.violation_mask;
            return Ok(out);
        }
        let mut ssm = self
            .pack
            .ssm()
            .lock()
            .map_err(|_| ComputeError::InvalidInput {
                reason: "ssm mutex poisoned".to_string(),
            })?;
        let ssm_name = ssm.name();
        let ssm_span = tracing::info_span!("ssm.step", kernel = ssm_name, t = input.t);
        let _ssm_enter = ssm_span.enter();
        let (ssm_out, ssm_degraded) = match ssm_meter.spend(220, "ssm/step") {
            Ok(()) => match ssm.step(&ssm_input, budget) {
                Ok(output) => (output, false),
                Err(ComputeError::BudgetExceeded { stage, .. }) => {
                    exceeded_stage = Some(stage);
                    if budget.degrade_policy == DegradePolicy::FailFast {
                        return Ok(ComputeSignals::unavailable(input, budget, self.name()));
                    }
                    (SsmOutput::degraded("budget_exceeded"), true)
                }
                Err(other) => return Err(other),
            },
            Err(ComputeError::BudgetExceeded { stage, .. }) => {
                exceeded_stage = Some(stage);
                if budget.degrade_policy == DegradePolicy::FailFast {
                    return Ok(ComputeSignals::unavailable(input, budget, self.name()));
                }
                (SsmOutput::degraded("budget_exceeded"), true)
            }
            Err(other) => return Err(other),
        };

        validation_report =
            validation_report.merge(SsmValidatorV1::validate(&ssm_input, &ssm_out, None));
        metrics::histogram!("ucf_ssm_pressure").record(f64::from(ssm_out.pressure));
        metrics::gauge!("ucf_ssm_state_norm").set(f64::from(ssm_out.state_norm));
        if ssm_out.quality == StageQuality::DegradedFallback {
            metrics::counter!("ucf_ssm_degraded_total").increment(1);
        }

        let lfm_input = LfmInput {
            t: input.t,
            context_digest: input.context_digest,
            world_digest: world_model_out.prediction_digest,
            surprise: world_model_out.surprise,
            spikes_digest: sae_out.spikes_digest,
            spike_count: sae_out.spike_count,
            sae_energy: sae_out.energy,
            pressure: ssm_out.pressure,
            coherence: None,
            instability: None,
            hormone_stress: None,
            neuro_arousal: None,
            governor_tier: Some(budget.governor_tier),
            prediction_error: Some(world_model_out.prediction_error),
            seed: budget.seed,
        };

        global_meter.spend(220, "lfm/step")?;
        if !registry.supports(
            StageKind::Lfm,
            pack_meta.lfm_backend,
            self.pack
                .lfm()
                .lock()
                .map_err(|_| ComputeError::InvalidInput {
                    reason: "lfm mutex poisoned".to_string(),
                })?
                .contract_version(),
        ) {
            validation_report.add_hard(ViolationCode::BackendContractMismatch);
            let mut out = ComputeSignals::unavailable(input, budget, self.name());
            out.backend_id = pack_meta.lfm_backend as u16;
            out.violation_reason_mask = validation_report.violation_mask;
            return Ok(out);
        }
        let mut lfm = self
            .pack
            .lfm()
            .lock()
            .map_err(|_| ComputeError::InvalidInput {
                reason: "lfm mutex poisoned".to_string(),
            })?;
        let lfm_name = lfm.name();
        let lfm_span = tracing::info_span!("lfm.step", kernel = lfm_name, t = input.t);
        let _lfm_enter = lfm_span.enter();
        let (lfm_out, lfm_degraded) = match lfm_meter.spend(220, "lfm/step") {
            Ok(()) => match lfm.step(&lfm_input, budget) {
                Ok(output) => (output, false),
                Err(ComputeError::BudgetExceeded { stage, .. }) => {
                    exceeded_stage = Some(stage);
                    if budget.degrade_policy == DegradePolicy::FailFast {
                        return Ok(ComputeSignals::unavailable(input, budget, self.name()));
                    }
                    (LfmOutput::degraded("budget_exceeded"), true)
                }
                Err(other) => return Err(other),
            },
            Err(ComputeError::BudgetExceeded { stage, .. }) => {
                exceeded_stage = Some(stage);
                if budget.degrade_policy == DegradePolicy::FailFast {
                    return Ok(ComputeSignals::unavailable(input, budget, self.name()));
                }
                (LfmOutput::degraded("budget_exceeded"), true)
            }
            Err(other) => return Err(other),
        };
        let plasticity_record = lfm_out.plasticity.clone();

        let lfm_backend_label = if lfm_name.contains("candle") {
            "candle"
        } else if lfm_name.contains("burn") {
            "burn"
        } else {
            "toy"
        };
        metrics::counter!("ucf_lfm_step_total", "backend" => lfm_backend_label).increment(1);
        metrics::gauge!("ucf_lfm_uncertainty").set(f64::from(lfm_out.uncertainty));
        metrics::gauge!("ucf_lfm_stability").set(f64::from(lfm_out.stability));
        if lfm_out.quality == StageQuality::DegradedFallback {
            metrics::counter!("ucf_lfm_degraded_total", "backend" => lfm_backend_label)
                .increment(1);
            if lfm_name.contains("lnn") {
                metrics::counter!("ucf_lfm_lnn_degraded_total").increment(1);
            }
        }

        validation_report = validation_report.merge(LfmValidatorV1::validate(&lfm_input, &lfm_out));
        let pressure = ssm_out.pressure;
        let (base_risk, base_confidence) = fuse_signals(
            surprise * self.fusion.world_weight,
            pressure * self.fusion.ssm_weight,
            sae_out.energy * self.fusion.energy_weight,
        );
        let risk = clamp01(base_risk + 0.2 * lfm_out.uncertainty);
        let confidence = clamp01(base_confidence * lfm_out.stability);

        let spikes_digest_ref = sae_out.spikes_digest;

        let quality = if world_model_out.quality != StageQuality::Ok
            || sae_degraded
            || ssm_degraded
            || lfm_degraded
        {
            SignalQuality::DegradedFallback
        } else {
            SignalQuality::VerifiedPipeline
        };
        let evidence = EvidenceRef {
            context_digest: input.context_digest,
            world_digest: Some(world_model_out.prediction_digest),
            spikes_digest: if sae_degraded {
                None
            } else {
                Some(spikes_digest_ref)
            },
            ssm_digest: if ssm_degraded {
                None
            } else {
                Some(ssm_out.state_digest)
            },
            lfm_digest: if lfm_degraded {
                None
            } else {
                Some(lfm_out.liquid_state_digest)
            },
            backend_profile: BackendProfileId::from_backend_name(self.name()),
            backend_pack_id: pack_meta.pack_id,
            fixtures_digest: pack_meta.fixtures_digest,
            model_hashes_digest: pack_meta.model_hashes_digest,
            llm_backend: pack_meta.llm_backend,
            world_backend: pack_meta.world_backend,
            sae_backend: pack_meta.sae_backend,
            ssm_backend: pack_meta.ssm_backend,
            lfm_backend: pack_meta.lfm_backend,
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
                Some(ssm_out.state_digest)
            },
            world_digest: Some(world_model_out.prediction_digest),
            lfm_uncertainty: Some(lfm_out.uncertainty),
            lfm_stability: Some(lfm_out.stability),
            lfm_state_norm: Some(lfm_out.state_norm),
            lfm_deriv_norm: Some(lfm_out.deriv_norm),
            lfm_saturation_ratio: Some(lfm_out.saturation_ratio),
            lfm_nan_inf_detected: lfm_out.nan_inf_detected,
            lfm_digest: if lfm_degraded {
                None
            } else {
                Some(lfm_out.liquid_state_digest)
            },
            sae_quality: Some(sae_out.quality),
            ssm_quality: Some(ssm_out.quality),
            lfm_quality: Some(lfm_out.quality),
            plasticity_record: plasticity_record.clone(),
            budget_exceeded_stage: exceeded_stage,
            contract_version: StageContractVersion::V1,
            backend_id: pack_meta.pack_id.0 as u16,
            validation_status: validation_report.status,
            violation_reason_mask: validation_report.violation_mask,
        }
        .summary(self.name());

        let mut notes = vec![
            format!("backend={}", self.name()),
            format!("frame={}", input.frame_id.0),
            format!("budget_profile_id={}", budget.profile_id),
            format!("world_model={}", world_model_name),
            format!("feature_extractor={}", self.pack.sae().name()),
            format!("working_memory={}", ssm_name),
            format!("lfm={}", lfm_name),
            format!(
                "pred_digest={}",
                &hex::encode(world_model_out.prediction_digest)[..12]
            ),
            format!("pred_error={:.4}", world_model_out.prediction_error),
            format!("world_quality={:?}", world_model_out.quality),
            format!("sae_quality={:?}", sae_out.quality),
            format!("ssm_quality={:?}", ssm_out.quality),
            format!("lfm_quality={:?}", lfm_out.quality),
            format!("ssm_digest={}", &hex::encode(ssm_out.state_digest)[..12]),
            format!(
                "lfm_digest={}",
                &hex::encode(lfm_out.liquid_state_digest)[..12]
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

        if let Some(rec) = &plasticity_record {
            notes.push(format!("plasticity_enabled={}", rec.enabled));
            notes.push(format!("plasticity_updates={}", rec.param_deltas.len()));
            notes.push(format!(
                "plasticity_digest={}",
                &hex::encode(rec.params_digest_after)[..12]
            ));
        }

        if let Some(stage) = exceeded_stage {
            notes.push(format!("budget_exceeded_stage={stage}"));
        }
        if sae_degraded {
            notes.push("degraded:sae_budget_exceeded".to_string());
        }
        if ssm_degraded {
            notes.push("degraded:ssm_budget_exceeded".to_string());
        }
        if lfm_degraded {
            notes.push("degraded:lfm_budget_exceeded".to_string());
        }
        notes.sort();

        let chain_report =
            validate_evidence_chain_digest(&crate::evidence::EvidenceChain::from_compute(
                input,
                &sae_out.spikes,
                &risk_signal,
                Some(sae_out.quality),
                Some(ssm_out.quality),
                Some(lfm_out.quality),
            ));
        validation_report = validation_report.merge(chain_report);

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
                Some(ssm_out.state_digest)
            },
            world_digest: Some(world_model_out.prediction_digest),
            lfm_uncertainty: Some(lfm_out.uncertainty),
            lfm_stability: Some(lfm_out.stability),
            lfm_state_norm: Some(lfm_out.state_norm),
            lfm_deriv_norm: Some(lfm_out.deriv_norm),
            lfm_saturation_ratio: Some(lfm_out.saturation_ratio),
            lfm_nan_inf_detected: lfm_out.nan_inf_detected,
            lfm_digest: if lfm_degraded {
                None
            } else {
                Some(lfm_out.liquid_state_digest)
            },
            sae_quality: Some(sae_out.quality),
            ssm_quality: Some(ssm_out.quality),
            lfm_quality: Some(lfm_out.quality),
            plasticity_record: plasticity_record.clone(),
            budget_exceeded_stage: exceeded_stage,
            contract_version: StageContractVersion::V1,
            backend_id: pack_meta.pack_id.0 as u16,
            validation_status: validation_report.status,
            violation_reason_mask: validation_report.violation_mask,
        }
        .bounded())
    }
}

#[cfg(test)]
mod tests {
    use crate::FrameId;
    use crate::{BackendPackConfig, BackendPackFactory};

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
        let pack = BackendPackFactory::build(BackendPackConfig::default()).expect("pack");
        let backend =
            ComputePipelineBackend::new(pack, FusionConfig::default(), LimitsConfig::default());
        let out = backend
            .compute(&input(), ComputeBudget::default())
            .expect("compute");
        assert!(out
            .notes
            .iter()
            .any(|n| n == "feature_extractor=toy_sae_v0"));
        assert!(out.notes.iter().any(|n| n.starts_with("lfm=")));
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
