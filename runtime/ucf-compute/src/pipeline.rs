use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Instant;

use crate::backend_pack::{
    ArtifactFailureCode, BackendPack, BackendPackFactory, ModelSlotProvenance, SlotRuntimeStatus,
};
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
    obs_features_from_context, world_vljepa_shadow_step, StageQuality, WorldInputEncodingV1,
    WorldModelInput, WorldModelOutput,
};

use crate::world_vljepa_shadow::{record_shadow_sample, shadow_disabled};
use crate::{
    clamp01, fuse_signals, validate_risk_signal, AiComputeBackend, BackendPackConfig,
    BackendProfileId, ComputeBudget, ComputeError, ComputeInput, ComputeSignals, DegradePolicy,
    EvidenceRef, RiskSignal, SignalQuality,
};

pub const CANONICAL_STAGE_SEQUENCE: [CanonicalStageId; 4] = [
    CanonicalStageId::World,
    CanonicalStageId::Sae,
    CanonicalStageId::Ssm,
    CanonicalStageId::Lfm,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalStageId {
    World,
    Sae,
    Ssm,
    Lfm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalPipelineState {
    Ok,
    Degraded,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalFailureKind {
    InvalidInput,
    BackendDisabled,
    StageContractMismatch,
    ArtifactUnavailable,
    ArtifactVerificationFailed,
    ArtifactIncompatible,
    DegradedFallback,
    ValidationDegraded,
    BudgetExceeded,
    ExecutionError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPipelineFailure {
    pub kind: CanonicalFailureKind,
    pub stage: Option<CanonicalStageId>,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalBackendRoute {
    pub pack_id: u32,
    pub world_backend: u8,
    pub sae_backend: u8,
    pub ssm_backend: u8,
    pub lfm_backend: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalPipelineResult {
    pub request: ComputeInput,
    pub stage_order: [CanonicalStageId; 4],
    pub route: CanonicalBackendRoute,
    pub state: CanonicalPipelineState,
    pub failure: Option<CanonicalPipelineFailure>,
    pub validation_status: ValidationStatus,
    pub violation_reason_mask: u32,
    pub model_slots: Vec<ModelSlotProvenance>,
    pub signals: ComputeSignals,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPipelineRequest {
    pub input: ComputeInput,
    pub budget: ComputeBudget,
}

struct UnavailableResultContext {
    validation_status: ValidationStatus,
    violation_reason_mask: u32,
    failure: CanonicalPipelineFailure,
    backend_id: Option<u16>,
    budget_stage: Option<&'static str>,
}

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
        Ok(self
            .compute_canonical(CanonicalPipelineRequest {
                input: input.clone(),
                budget,
            })?
            .signals)
    }
}

impl ComputePipelineBackend {
    pub fn compute_canonical(
        &self,
        request: CanonicalPipelineRequest,
    ) -> Result<CanonicalPipelineResult, ComputeError> {
        let input = request.input;
        let budget = request.budget;
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
        let route = CanonicalBackendRoute {
            pack_id: pack_meta.pack_id.0,
            world_backend: pack_meta.world_backend as u8,
            sae_backend: pack_meta.sae_backend as u8,
            ssm_backend: pack_meta.ssm_backend as u8,
            lfm_backend: pack_meta.lfm_backend as u8,
        };
        let registry = StageContractRegistry;
        let requested = StageContractVersion::V1;
        if let Some(failure) = first_artifact_failure(self.pack.model_slot_provenance()) {
            return Ok(self.unavailable_result(
                &input,
                budget,
                route,
                UnavailableResultContext {
                    validation_status: ValidationStatus::Degraded,
                    violation_reason_mask: 0,
                    failure,
                    backend_id: None,
                    budget_stage: None,
                },
            ));
        }

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
            let kind = if pack_meta.world_backend == crate::BackendComponentId::Disabled {
                CanonicalFailureKind::BackendDisabled
            } else {
                CanonicalFailureKind::StageContractMismatch
            };
            return Ok(self.unavailable_result(
                &input,
                budget,
                route,
                UnavailableResultContext {
                    validation_status: ValidationStatus::Degraded,
                    violation_reason_mask: 1_u32 << (ViolationCode::BackendContractMismatch as u32),
                    failure: CanonicalPipelineFailure {
                        kind,
                        stage: Some(CanonicalStageId::World),
                        detail: format!(
                            "world backend {:?} contract {:?} unsupported",
                            pack_meta.world_backend, requested
                        ),
                    },
                    backend_id: Some(pack_meta.world_backend as u16),
                    budget_stage: None,
                },
            ));
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
                    return Ok(self.unavailable_result(
                        &input,
                        budget,
                        route,
                        UnavailableResultContext {
                            validation_status: ValidationStatus::Degraded,
                            violation_reason_mask: 0,
                            failure: CanonicalPipelineFailure {
                                kind: CanonicalFailureKind::BudgetExceeded,
                                stage: Some(CanonicalStageId::World),
                                detail: format!("world stage budget exceeded at {stage}"),
                            },
                            backend_id: None,
                            budget_stage: Some(stage),
                        },
                    ));
                }
                WorldModelOutput::degraded_budget(stage)
            }
            Err(other) => return Err(other),
        };
        let span = tracing::info_span!("world_model.step", predictor = world_model_name, t = input.t, pred = %hex::encode(&world_model_out.prediction_digest[..4]));
        let _enter = span.enter();
        drop(world);

        if std::env::var("UCF_SLOT_WORLD_VLJEPA_MODE").ok().as_deref() == Some("shadow")
            && !shadow_disabled()
        {
            let vljepa_in = WorldInputEncodingV1 {
                context_digest: input.context_digest,
                risk: world_model_out.prediction_error,
                pressure: world_model_out.state_norm,
                surprise: world_model_out.surprise,
                uncertainty: world_model_out.surprise,
                confidence: (1.0 - world_model_out.surprise).clamp(0.0, 1.0),
                coherence: (1.0 - world_model_out.surprise).clamp(0.0, 1.0),
                sae_spikes_digest_prefix: None,
                ssm_state_digest_prefix: None,
                lfm_state_digest_prefix: None,
                token_summary_digest_prefix: None,
            };
            let started = std::time::Instant::now();
            let shadow =
                world_vljepa_shadow_step(input.t, &vljepa_in, self.pack.meta().model_hashes_digest);
            let latency_ms = started.elapsed().as_secs_f32() * 1000.0;
            let baseline_error_q =
                ucf_types::UQ0_16::from_f32_clamped(world_model_out.prediction_error).raw();
            record_shadow_sample(latency_ms, baseline_error_q, &shadow);
            metrics::histogram!("ucf_world_vljepa_prediction_error_q")
                .record(f64::from(shadow.prediction_error_q));
            tracing::info!(
                target: "ucf.world_vljepa_shadow",
                t = shadow.t,
                prediction_error_q = shadow.prediction_error_q,
                baseline_error_q,
                saturation_clamp_count = shadow.saturation_clamp_count,
                invalid_output = shadow.invalid_output,
                prediction_digest = %hex::encode(shadow.prediction_digest_prefix),
                status = shadow.status,
                "world_vljepa shadow record"
            );
        }
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
            ToySaeExtractor::make_input(&input, &world_model_out, budget.seed, evidence_seed);
        if !registry.supports(
            StageKind::Sae,
            pack_meta.sae_backend,
            self.pack.sae().contract_version(),
        ) || self.pack.sae().contract_version() != requested
        {
            validation_report.add_hard(ViolationCode::BackendContractMismatch);
            let kind = if pack_meta.sae_backend == crate::BackendComponentId::Disabled {
                CanonicalFailureKind::BackendDisabled
            } else {
                CanonicalFailureKind::StageContractMismatch
            };
            return Ok(self.unavailable_result(
                &input,
                budget,
                route,
                UnavailableResultContext {
                    validation_status: validation_report.status,
                    violation_reason_mask: validation_report.violation_mask,
                    failure: CanonicalPipelineFailure {
                        kind,
                        stage: Some(CanonicalStageId::Sae),
                        detail: format!(
                            "sae backend {:?} contract {:?} unsupported",
                            pack_meta.sae_backend, requested
                        ),
                    },
                    backend_id: Some(pack_meta.sae_backend as u16),
                    budget_stage: None,
                },
            ));
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
                        return Ok(self.unavailable_result(
                            &input,
                            budget,
                            route,
                            UnavailableResultContext {
                                validation_status: ValidationStatus::Degraded,
                                violation_reason_mask: 0,
                                failure: CanonicalPipelineFailure {
                                    kind: CanonicalFailureKind::BudgetExceeded,
                                    stage: Some(CanonicalStageId::Sae),
                                    detail: format!("sae stage budget exceeded at {stage}"),
                                },
                                backend_id: None,
                                budget_stage: Some(stage),
                            },
                        ));
                    }
                    (Self::empty_sae(), true)
                }
                Err(other) => return Err(other),
            },
            Err(ComputeError::BudgetExceeded { stage, .. }) => {
                exceeded_stage = Some(stage);
                if budget.degrade_policy == DegradePolicy::FailFast {
                    return Ok(self.unavailable_result(
                        &input,
                        budget,
                        route,
                        UnavailableResultContext {
                            validation_status: ValidationStatus::Degraded,
                            violation_reason_mask: 0,
                            failure: CanonicalPipelineFailure {
                                kind: CanonicalFailureKind::BudgetExceeded,
                                stage: Some(CanonicalStageId::Sae),
                                detail: format!("sae stage budget exceeded at {stage}"),
                            },
                            backend_id: None,
                            budget_stage: Some(stage),
                        },
                    ));
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
            let kind = if pack_meta.ssm_backend == crate::BackendComponentId::Disabled {
                CanonicalFailureKind::BackendDisabled
            } else {
                CanonicalFailureKind::StageContractMismatch
            };
            return Ok(self.unavailable_result(
                &input,
                budget,
                route,
                UnavailableResultContext {
                    validation_status: validation_report.status,
                    violation_reason_mask: validation_report.violation_mask,
                    failure: CanonicalPipelineFailure {
                        kind,
                        stage: Some(CanonicalStageId::Ssm),
                        detail: format!(
                            "ssm backend {:?} contract {:?} unsupported",
                            pack_meta.ssm_backend, requested
                        ),
                    },
                    backend_id: Some(pack_meta.ssm_backend as u16),
                    budget_stage: None,
                },
            ));
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
                        return Ok(self.unavailable_result(
                            &input,
                            budget,
                            route,
                            UnavailableResultContext {
                                validation_status: ValidationStatus::Degraded,
                                violation_reason_mask: 0,
                                failure: CanonicalPipelineFailure {
                                    kind: CanonicalFailureKind::BudgetExceeded,
                                    stage: Some(CanonicalStageId::Ssm),
                                    detail: format!("ssm stage budget exceeded at {stage}"),
                                },
                                backend_id: None,
                                budget_stage: Some(stage),
                            },
                        ));
                    }
                    (SsmOutput::degraded("budget_exceeded"), true)
                }
                Err(other) => return Err(other),
            },
            Err(ComputeError::BudgetExceeded { stage, .. }) => {
                exceeded_stage = Some(stage);
                if budget.degrade_policy == DegradePolicy::FailFast {
                    return Ok(self.unavailable_result(
                        &input,
                        budget,
                        route,
                        UnavailableResultContext {
                            validation_status: ValidationStatus::Degraded,
                            violation_reason_mask: 0,
                            failure: CanonicalPipelineFailure {
                                kind: CanonicalFailureKind::BudgetExceeded,
                                stage: Some(CanonicalStageId::Ssm),
                                detail: format!("ssm stage budget exceeded at {stage}"),
                            },
                            backend_id: None,
                            budget_stage: Some(stage),
                        },
                    ));
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
            risk: Some(ssm_out.pressure.clamp(0.0, 1.0)),
            confidence: Some((1.0 - ssm_out.pressure).clamp(0.0, 1.0)),
            prior_uncertainty: Some((1.0 - ssm_out.readout).clamp(0.0, 1.0)),
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
            let kind = if pack_meta.lfm_backend == crate::BackendComponentId::Disabled {
                CanonicalFailureKind::BackendDisabled
            } else {
                CanonicalFailureKind::StageContractMismatch
            };
            return Ok(self.unavailable_result(
                &input,
                budget,
                route,
                UnavailableResultContext {
                    validation_status: validation_report.status,
                    violation_reason_mask: validation_report.violation_mask,
                    failure: CanonicalPipelineFailure {
                        kind,
                        stage: Some(CanonicalStageId::Lfm),
                        detail: format!(
                            "lfm backend {:?} contract {:?} unsupported",
                            pack_meta.lfm_backend, requested
                        ),
                    },
                    backend_id: Some(pack_meta.lfm_backend as u16),
                    budget_stage: None,
                },
            ));
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
        let lfm_started = Instant::now();
        let (lfm_out, lfm_degraded) = match lfm_meter.spend(220, "lfm/step") {
            Ok(()) => match lfm.step(&lfm_input, budget) {
                Ok(output) => (output, false),
                Err(ComputeError::BudgetExceeded { stage, .. }) => {
                    exceeded_stage = Some(stage);
                    if budget.degrade_policy == DegradePolicy::FailFast {
                        return Ok(self.unavailable_result(
                            &input,
                            budget,
                            route,
                            UnavailableResultContext {
                                validation_status: ValidationStatus::Degraded,
                                violation_reason_mask: 0,
                                failure: CanonicalPipelineFailure {
                                    kind: CanonicalFailureKind::BudgetExceeded,
                                    stage: Some(CanonicalStageId::Lfm),
                                    detail: format!("lfm stage budget exceeded at {stage}"),
                                },
                                backend_id: None,
                                budget_stage: Some(stage),
                            },
                        ));
                    }
                    (LfmOutput::degraded("budget_exceeded"), true)
                }
                Err(other) => return Err(other),
            },
            Err(ComputeError::BudgetExceeded { stage, .. }) => {
                exceeded_stage = Some(stage);
                if budget.degrade_policy == DegradePolicy::FailFast {
                    return Ok(self.unavailable_result(
                        &input,
                        budget,
                        route,
                        UnavailableResultContext {
                            validation_status: ValidationStatus::Degraded,
                            violation_reason_mask: 0,
                            failure: CanonicalPipelineFailure {
                                kind: CanonicalFailureKind::BudgetExceeded,
                                stage: Some(CanonicalStageId::Lfm),
                                detail: format!("lfm stage budget exceeded at {stage}"),
                            },
                            backend_id: None,
                            budget_stage: Some(stage),
                        },
                    ));
                }
                (LfmOutput::degraded("budget_exceeded"), true)
            }
            Err(other) => return Err(other),
        };
        let plasticity_record = lfm_out.plasticity.clone();
        metrics::histogram!("ucf_lfm_ode_step_micros")
            .record(lfm_started.elapsed().as_micros() as f64);

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
            signal_bundle_digest: None,
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
                &input,
                &sae_out.spikes,
                &risk_signal,
                Some(sae_out.quality),
                Some(ssm_out.quality),
                Some(lfm_out.quality),
            ));
        validation_report = validation_report.merge(chain_report);

        let mut state = CanonicalPipelineState::Ok;
        let mut failure = None;
        if validation_report.status == ValidationStatus::Degraded {
            state = CanonicalPipelineState::Degraded;
            failure = Some(CanonicalPipelineFailure {
                kind: CanonicalFailureKind::ValidationDegraded,
                stage: None,
                detail: "one or more stage validators degraded output".to_string(),
            });
        } else if quality == SignalQuality::DegradedFallback {
            state = CanonicalPipelineState::Degraded;
            failure = Some(CanonicalPipelineFailure {
                kind: CanonicalFailureKind::DegradedFallback,
                stage: None,
                detail: "degraded but usable fallback output".to_string(),
            });
        }

        let signals = ComputeSignals {
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
            signal_bundle_digest: None,
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
        .bounded();

        Ok(CanonicalPipelineResult {
            request: input,
            stage_order: CANONICAL_STAGE_SEQUENCE,
            route,
            state,
            failure,
            validation_status: signals.validation_status,
            violation_reason_mask: signals.violation_reason_mask,
            model_slots: self.pack.model_slot_provenance().to_vec(),
            signals,
        })
    }

    fn unavailable_result(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
        route: CanonicalBackendRoute,
        ctx: UnavailableResultContext,
    ) -> CanonicalPipelineResult {
        let mut signals = ComputeSignals::unavailable(input, budget, self.name());
        signals.validation_status = ctx.validation_status;
        signals.violation_reason_mask = ctx.violation_reason_mask;
        if let Some(id) = ctx.backend_id {
            signals.backend_id = id;
        }
        signals.budget_exceeded_stage = ctx.budget_stage;
        signals
            .notes
            .push(format!("pipeline_failure={:?}", ctx.failure.kind).to_ascii_lowercase());
        CanonicalPipelineResult {
            request: input.clone(),
            stage_order: CANONICAL_STAGE_SEQUENCE,
            route,
            state: CanonicalPipelineState::Unavailable,
            failure: Some(ctx.failure),
            validation_status: ctx.validation_status,
            violation_reason_mask: ctx.violation_reason_mask,
            model_slots: self.pack.model_slot_provenance().to_vec(),
            signals,
        }
    }
}

fn first_artifact_failure(slots: &[ModelSlotProvenance]) -> Option<CanonicalPipelineFailure> {
    slots.iter().find_map(|slot| {
        if !slot.required_for_pack {
            return None;
        }
        match slot.status {
            SlotRuntimeStatus::Used | SlotRuntimeStatus::Disabled => None,
            SlotRuntimeStatus::Unavailable
            | SlotRuntimeStatus::VerificationFailed
            | SlotRuntimeStatus::Incompatible => {
                let kind = match slot.code {
                    Some(ArtifactFailureCode::Disabled) => CanonicalFailureKind::BackendDisabled,
                    Some(ArtifactFailureCode::MissingPath)
                    | Some(ArtifactFailureCode::ArtifactUnavailable) => {
                        CanonicalFailureKind::ArtifactUnavailable
                    }
                    Some(ArtifactFailureCode::MissingExpectedHash)
                    | Some(ArtifactFailureCode::HashMismatch)
                    | Some(ArtifactFailureCode::Oversized)
                    | Some(ArtifactFailureCode::PathViolation)
                    | Some(ArtifactFailureCode::ArtifactVerificationFailed) => {
                        CanonicalFailureKind::ArtifactVerificationFailed
                    }
                    Some(ArtifactFailureCode::ArtifactIncompatible) => {
                        CanonicalFailureKind::ArtifactIncompatible
                    }
                    None => CanonicalFailureKind::ArtifactUnavailable,
                };
                Some(CanonicalPipelineFailure {
                    kind,
                    stage: Some(match slot.stage {
                        "world" => CanonicalStageId::World,
                        "sae" => CanonicalStageId::Sae,
                        "ssm" => CanonicalStageId::Ssm,
                        "lfm" => CanonicalStageId::Lfm,
                        _ => CanonicalStageId::World,
                    }),
                    detail: format!(
                        "slot {} status {:?}: {}",
                        slot.slot.as_str(),
                        slot.status,
                        slot.detail.as_deref().unwrap_or("n/a")
                    ),
                })
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::{BackendPackKind, FrameId};

    use crate::{
        ArtifactFailureCode, BackendPackConfig, BackendPackFactory, ModelSlot, ModelSlotProvenance,
        SlotRuntimeStatus,
    };

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
    fn canonical_pipeline_request_uses_world_sae_ssm_lfm_order() {
        let backend = ComputePipelineBackend::stub();
        let result = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget: ComputeBudget::default(),
            })
            .expect("canonical compute");
        assert_eq!(result.stage_order, CANONICAL_STAGE_SEQUENCE);
        assert_ne!(result.state, CanonicalPipelineState::Unavailable);
    }

    #[test]
    fn stub_pack_contract_mismatch_is_structured_unavailable() {
        let pack = BackendPackFactory::build(BackendPackConfig {
            pack: BackendPackKind::StubV0,
            ..BackendPackConfig::default()
        })
        .expect("stub pack");
        let backend =
            ComputePipelineBackend::new(pack, FusionConfig::default(), LimitsConfig::default());
        let result = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget: ComputeBudget::default(),
            })
            .expect("canonical compute");
        assert_eq!(result.state, CanonicalPipelineState::Unavailable);
        let failure = result.failure.expect("failure");
        assert_eq!(failure.kind, CanonicalFailureKind::StageContractMismatch);
        assert_eq!(failure.stage, Some(CanonicalStageId::World));
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
        let canonical = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget,
            })
            .expect("canonical compute");
        assert_eq!(canonical.state, CanonicalPipelineState::Degraded);
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
        let canonical = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget,
            })
            .expect("canonical compute");
        assert_eq!(canonical.state, CanonicalPipelineState::Unavailable);
        assert_eq!(
            canonical.failure.expect("failure").kind,
            CanonicalFailureKind::BudgetExceeded
        );
    }

    #[test]
    fn invalid_input_remains_hard_execution_error() {
        let backend = ComputePipelineBackend::stub();
        let invalid = ComputeInput {
            frame_id: FrameId(7),
            t: 0,
            context_digest: [9; 32],
        };
        let result = backend.compute_canonical(CanonicalPipelineRequest {
            input: invalid,
            budget: ComputeBudget::default(),
        });
        assert!(matches!(result, Err(ComputeError::InvalidInput { .. })));
    }

    #[test]
    fn artifact_failures_are_classified() {
        let slots = vec![ModelSlotProvenance {
            slot: ModelSlot::Ssm,
            stage: "ssm",
            required_for_pack: true,
            status: SlotRuntimeStatus::VerificationFailed,
            code: Some(ArtifactFailureCode::HashMismatch),
            detail: Some("model hash mismatch".to_string()),
            resolved_path: None,
            hash_prefix: None,
            contract_version: Some("v1".to_string()),
            format: None,
        }];
        let failure = first_artifact_failure(&slots).expect("failure");
        assert_eq!(
            failure.kind,
            CanonicalFailureKind::ArtifactVerificationFailed
        );
        assert_eq!(failure.stage, Some(CanonicalStageId::Ssm));
    }

    #[test]
    fn degraded_fallback_sets_explicit_failure_kind() {
        let backend = ComputePipelineBackend::stub();
        let budget = ComputeBudget {
            sae_units: 100,
            global_work_units: 900,
            profile_id: 3,
            ..ComputeBudget::default()
        };
        let canonical = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget,
            })
            .expect("canonical compute");
        assert_eq!(canonical.state, CanonicalPipelineState::Degraded);
        let failure_kind = canonical.failure.expect("failure").kind;
        assert!(matches!(
            failure_kind,
            CanonicalFailureKind::DegradedFallback | CanonicalFailureKind::ValidationDegraded
        ));
    }
}
