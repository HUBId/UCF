use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Instant;

use crate::backend_pack::{
    ArtifactFailureCode, BackendComponentId, BackendPack, BackendPackFactory, ModelSlotProvenance,
    SlotRuntimeStatus,
};
use crate::contracts::{
    validate_evidence_chain_digest, ContractRegistry, LfmValidatorV1, NsrContractVersion,
    NsrFailureKind, NsrRequest, NsrResult, SaeValidatorV1, SsmValidatorV1, StageContractRegistry,
    StageContractVersion, StageKind, ValidationStatus, ViolationCode, WorldValidatorV1,
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
use ucf_nsr::{
    ActionType, DecisionIntentSummary, NsrBudget, NsrContext, NsrDatalogLiteEngine, NsrError,
    NsrPolicyEcologyEngine, OutputClass, PolicyTag, ReasonCode,
};

pub const CANONICAL_STAGE_SEQUENCE: [CanonicalStageId; 4] = [
    CanonicalStageId::World,
    CanonicalStageId::Sae,
    CanonicalStageId::Ssm,
    CanonicalStageId::Lfm,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    ContractMismatch,
    StageContractMismatch,
    ArtifactUnavailable,
    ArtifactVerificationFailed,
    ArtifactIncompatible,
    StageUnavailable,
    DegradedFallback,
    ValidationDegraded,
    BudgetExceeded,
    Timeout,
    ExecutionError,
    NsrDisabled,
    NsrUnavailable,
    NsrArtifactVerificationFailed,
    NsrContractMismatch,
    NsrBackendUnavailable,
    NsrExecutionError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalFaultDomain {
    ArtifactModel,
    Stage,
    Backend,
    WorkerTransport,
    PlacementCapacity,
    RuntimeService,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalIsolationDisposition {
    LocallyIsolated,
    DegradedButServiceable,
    HardEscalationJobFailure,
    ServiceRuntimeImpact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalFailureClassification {
    pub domain: CanonicalFaultDomain,
    pub isolation: CanonicalIsolationDisposition,
    pub systemic: bool,
}

pub fn classify_failure_kind(kind: CanonicalFailureKind) -> CanonicalFailureClassification {
    use CanonicalFailureKind as K;
    match kind {
        K::ArtifactUnavailable
        | K::ArtifactVerificationFailed
        | K::ArtifactIncompatible
        | K::NsrArtifactVerificationFailed => CanonicalFailureClassification {
            domain: CanonicalFaultDomain::ArtifactModel,
            isolation: CanonicalIsolationDisposition::HardEscalationJobFailure,
            systemic: false,
        },
        K::StageContractMismatch | K::StageUnavailable | K::ValidationDegraded => {
            CanonicalFailureClassification {
                domain: CanonicalFaultDomain::Stage,
                isolation: CanonicalIsolationDisposition::HardEscalationJobFailure,
                systemic: false,
            }
        }
        K::DegradedFallback => CanonicalFailureClassification {
            domain: CanonicalFaultDomain::Stage,
            isolation: CanonicalIsolationDisposition::DegradedButServiceable,
            systemic: false,
        },
        K::BackendDisabled
        | K::ContractMismatch
        | K::NsrContractMismatch
        | K::NsrBackendUnavailable => CanonicalFailureClassification {
            domain: CanonicalFaultDomain::Backend,
            isolation: CanonicalIsolationDisposition::HardEscalationJobFailure,
            systemic: false,
        },
        K::ExecutionError | K::NsrExecutionError => CanonicalFailureClassification {
            domain: CanonicalFaultDomain::WorkerTransport,
            isolation: CanonicalIsolationDisposition::HardEscalationJobFailure,
            systemic: false,
        },
        K::BudgetExceeded | K::Timeout => CanonicalFailureClassification {
            domain: CanonicalFaultDomain::PlacementCapacity,
            isolation: CanonicalIsolationDisposition::HardEscalationJobFailure,
            systemic: false,
        },
        K::InvalidInput | K::NsrDisabled | K::NsrUnavailable => CanonicalFailureClassification {
            domain: CanonicalFaultDomain::RuntimeService,
            isolation: CanonicalIsolationDisposition::ServiceRuntimeImpact,
            systemic: true,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalValidationSummary {
    pub input: ValidationStatus,
    pub stage: ValidationStatus,
    pub artifacts: ValidationStatus,
    pub output: ValidationStatus,
    pub evidence: ValidationStatus,
    pub violation_reason_mask: u32,
}

impl CanonicalValidationSummary {
    fn unavailable() -> Self {
        Self {
            input: ValidationStatus::Ok,
            stage: ValidationStatus::Degraded,
            artifacts: ValidationStatus::Degraded,
            output: ValidationStatus::Degraded,
            evidence: ValidationStatus::Degraded,
            violation_reason_mask: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CanonicalTimingSummary {
    pub total_micros: u64,
    pub world_micros: Option<u64>,
    pub sae_micros: Option<u64>,
    pub ssm_micros: Option<u64>,
    pub lfm_micros: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalStageProfileState {
    Success,
    SlowSuccess,
    Degraded,
    Skipped,
    Unavailable,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalStageProfile {
    pub stage: CanonicalStageId,
    pub state: CanonicalStageProfileState,
    pub duration_micros: Option<u64>,
    pub remaining_work_units: Option<u64>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageCostSignalProvenance {
    MeasuredTiming,
    DerivedFromBudgetAndMeter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalStageCostPattern {
    SlowButHealthy,
    DominantCostDriver,
    DegradedPathDriver,
    SkippedOrFallback,
    HardFailure,
    Inactive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalStageCostAttribution {
    pub stage: CanonicalStageId,
    pub state: CanonicalStageProfileState,
    pub timing_micros: Option<u64>,
    pub timing_share_bps: Option<u16>,
    pub work_consumed_units: u64,
    pub work_share_bps: Option<u16>,
    pub pattern: CanonicalStageCostPattern,
    pub dominant_timing: bool,
    pub dominant_work: bool,
    pub timing_provenance: StageCostSignalProvenance,
    pub work_provenance: StageCostSignalProvenance,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalHotspotSummary {
    pub slowest_stage: Option<CanonicalStageId>,
    pub dominant_stage: Option<CanonicalStageId>,
    pub dominant_stage_share_bps: Option<u16>,
    pub dominant_work_stage: Option<CanonicalStageId>,
    pub dominant_work_stage_share_bps: Option<u16>,
    pub degraded_stage: Option<CanonicalStageId>,
    pub fallback_stage: Option<CanonicalStageId>,
    pub degraded_stage_count: u8,
    pub skipped_stage_count: u8,
    pub unavailable_stage_count: u8,
    pub failed_stage_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalWorkSummary {
    pub global_budget_units: u64,
    pub global_remaining_units: u64,
    pub world_remaining_units: u64,
    pub sae_remaining_units: u64,
    pub ssm_remaining_units: u64,
    pub lfm_remaining_units: u64,
    pub budget_exceeded_stage: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalRunDiagnostics {
    pub timing: CanonicalTimingSummary,
    pub work: CanonicalWorkSummary,
    pub stage_profiles: Vec<CanonicalStageProfile>,
    pub stage_cost_attribution: Vec<CanonicalStageCostAttribution>,
    pub hotspots: CanonicalHotspotSummary,
    pub evidence_chain_digest_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPipelineFailure {
    pub kind: CanonicalFailureKind,
    pub stage: Option<CanonicalStageId>,
    pub detail: String,
}

impl CanonicalPipelineFailure {
    pub fn classification(&self) -> CanonicalFailureClassification {
        classify_failure_kind(self.kind)
    }
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
    pub executed_stages: Vec<CanonicalStageId>,
    pub route: CanonicalBackendRoute,
    pub state: CanonicalPipelineState,
    pub failure: Option<CanonicalPipelineFailure>,
    pub validation_status: ValidationStatus,
    pub violation_reason_mask: u32,
    pub validation: CanonicalValidationSummary,
    pub diagnostics: CanonicalRunDiagnostics,
    pub world_stage: WorldStageStatus,
    pub lfm_stage: LfmStageStatus,
    pub nsr_stage: NsrStageStatus,
    pub model_slots: Vec<ModelSlotProvenance>,
    pub signals: ComputeSignals,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LfmStageReadiness {
    Scaffolded,
    ContractReady,
    ArtifactReady,
    RuntimePathReady,
    ProductionBlocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LfmStageState {
    Disabled,
    Used,
    Unavailable,
    VerificationFailed,
    Incompatible,
    ContractMismatch,
    BackendUnavailable,
    ExecutionError,
    DegradedBypass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LfmStageStatus {
    pub slot: Option<ModelSlotProvenance>,
    pub state: LfmStageState,
    pub used: bool,
    pub runtime: String,
    pub backend: u8,
    pub readiness: LfmStageReadiness,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NsrStageReadiness {
    Scaffolded,
    ContractReady,
    ArtifactReady,
    RuntimePathReady,
    ProductionBlocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NsrStageState {
    Disabled,
    Used,
    Unavailable,
    VerificationFailed,
    Incompatible,
    ContractMismatch,
    BackendUnavailable,
    ExecutionError,
    DegradedBypass,
}

impl NsrStageState {
    fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NsrStageStatus {
    pub slot: ModelSlotProvenance,
    pub mode: String,
    pub state: NsrStageState,
    pub used: bool,
    pub readiness: NsrStageReadiness,
    pub detail: String,
    pub reason_codes: Vec<String>,
    pub digest_prefix: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldStageReadiness {
    Scaffolded,
    ContractReady,
    ArtifactReady,
    RuntimePathReady,
    ProductionBlocked,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorldStageStatus {
    pub predictor: String,
    pub slot: Option<crate::ModelSlot>,
    pub slot_status: Option<SlotRuntimeStatus>,
    pub slot_code: Option<ArtifactFailureCode>,
    pub used: bool,
    pub readiness: WorldStageReadiness,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPipelineRequest {
    pub input: ComputeInput,
    pub budget: ComputeBudget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalAdmissionDecision {
    pub route: CanonicalBackendRoute,
    pub failure: Option<CanonicalPipelineFailure>,
}

struct UnavailableResultContext {
    validation_status: ValidationStatus,
    violation_reason_mask: u32,
    validation: Option<CanonicalValidationSummary>,
    failure: CanonicalPipelineFailure,
    world_stage: Option<WorldStageStatus>,
    lfm_stage: Option<LfmStageStatus>,
    nsr_stage: Option<NsrStageStatus>,
    backend_id: Option<u16>,
    budget_stage: Option<&'static str>,
    stage_profiles: Option<Vec<CanonicalStageProfile>>,
    diagnostics: Option<CanonicalRunDiagnostics>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendExecutionLane {
    Toy,
    Candle,
    Burn,
    Worker,
    Mixed,
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

    pub fn execution_lane(&self) -> BackendExecutionLane {
        let meta = self.pack.meta();
        if meta.pack_name == "worker_v1" {
            return BackendExecutionLane::Worker;
        }
        let components = [
            meta.world_backend,
            meta.sae_backend,
            meta.ssm_backend,
            meta.lfm_backend,
        ];
        let has_burn = components.iter().any(|id| {
            matches!(
                id,
                BackendComponentId::BurnJepaV1
                    | BackendComponentId::BurnSaeV1
                    | BackendComponentId::BurnSsmV1
                    | BackendComponentId::BurnLfmV1
            )
        });
        let has_candle = components.iter().any(|id| {
            matches!(
                id,
                BackendComponentId::CandleJepaV1
                    | BackendComponentId::CandleSaeV1
                    | BackendComponentId::CandleSsmV1
                    | BackendComponentId::CandleEbmV1
                    | BackendComponentId::CandleVljepaV1
            )
        });
        match (has_burn, has_candle) {
            (true, false) => BackendExecutionLane::Burn,
            (false, true) => BackendExecutionLane::Candle,
            (false, false) => BackendExecutionLane::Toy,
            (true, true) => BackendExecutionLane::Mixed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NsrMode {
    Disabled,
    BestEffort,
    Required,
}

impl NsrMode {
    fn from_env() -> Self {
        match std::env::var("UCF_NSR_MODE")
            .unwrap_or_else(|_| "disabled".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "required" => Self::Required,
            "best_effort" | "besteffort" | "enabled" => Self::BestEffort,
            _ => Self::Disabled,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::BestEffort => "best_effort",
            Self::Required => "required",
        }
    }
}

#[derive(Debug, Clone)]
struct NsrOutcome {
    status: NsrStageStatus,
    result: Result<NsrResult, NsrFailureKind>,
    required_failure: bool,
}

fn nsr_slot_provenance(slots: &[ModelSlotProvenance]) -> ModelSlotProvenance {
    slots
        .iter()
        .find(|slot| slot.slot == crate::ModelSlot::EbmReasoner)
        .cloned()
        .unwrap_or(ModelSlotProvenance {
            slot: crate::ModelSlot::EbmReasoner,
            stage: "ebm_reasoner",
            required_for_pack: false,
            status: SlotRuntimeStatus::Disabled,
            code: Some(ArtifactFailureCode::Disabled),
            detail: Some("slot missing from provenance".to_string()),
            resolved_path: None,
            hash_prefix: None,
            contract_version: None,
            format: None,
            gate: Default::default(),
            rollout: None,
        })
}

fn nsr_disabled_stage(slot: ModelSlotProvenance, mode: NsrMode) -> NsrStageStatus {
    NsrStageStatus {
        slot,
        mode: mode.as_str().to_string(),
        state: NsrStageState::Disabled,
        used: false,
        readiness: NsrStageReadiness::Scaffolded,
        detail: "nsr stage disabled".to_string(),
        reason_codes: Vec::new(),
        digest_prefix: None,
    }
}

fn run_nsr_stage(
    mode: NsrMode,
    slot: ModelSlotProvenance,
    req: &NsrRequest,
    fail_fast: bool,
) -> NsrOutcome {
    if mode == NsrMode::Disabled {
        return NsrOutcome {
            status: nsr_disabled_stage(slot, mode),
            result: Err(NsrFailureKind::Disabled),
            required_failure: false,
        };
    }

    if matches!(slot.status, SlotRuntimeStatus::Disabled) {
        return NsrOutcome {
            status: NsrStageStatus {
                slot,
                mode: mode.as_str().to_string(),
                state: NsrStageState::Disabled,
                used: false,
                readiness: NsrStageReadiness::Scaffolded,
                detail: "nsr slot disabled".to_string(),
                reason_codes: Vec::new(),
                digest_prefix: None,
            },
            result: Err(NsrFailureKind::Disabled),
            required_failure: mode == NsrMode::Required || fail_fast,
        };
    }

    let version = slot
        .contract_version
        .clone()
        .unwrap_or_else(|| "v1".to_string());
    if version != req.contract_version.as_str() {
        return NsrOutcome {
            status: NsrStageStatus {
                slot,
                mode: mode.as_str().to_string(),
                state: NsrStageState::ContractMismatch,
                used: false,
                readiness: NsrStageReadiness::ProductionBlocked,
                detail: format!(
                    "nsr contract mismatch: slot={version}, runtime={}",
                    req.contract_version.as_str()
                ),
                reason_codes: Vec::new(),
                digest_prefix: None,
            },
            result: Err(NsrFailureKind::ContractMismatch),
            required_failure: mode == NsrMode::Required || fail_fast,
        };
    }

    match slot.status {
        SlotRuntimeStatus::Unavailable => {
            return NsrOutcome {
                status: NsrStageStatus {
                    slot,
                    mode: mode.as_str().to_string(),
                    state: NsrStageState::Unavailable,
                    used: false,
                    readiness: NsrStageReadiness::ProductionBlocked,
                    detail: "nsr slot unavailable".to_string(),
                    reason_codes: Vec::new(),
                    digest_prefix: None,
                },
                result: Err(NsrFailureKind::Unavailable),
                required_failure: mode == NsrMode::Required || fail_fast,
            };
        }
        SlotRuntimeStatus::VerificationFailed => {
            return NsrOutcome {
                status: NsrStageStatus {
                    slot,
                    mode: mode.as_str().to_string(),
                    state: NsrStageState::VerificationFailed,
                    used: false,
                    readiness: NsrStageReadiness::ProductionBlocked,
                    detail: "nsr slot verification failed".to_string(),
                    reason_codes: Vec::new(),
                    digest_prefix: None,
                },
                result: Err(NsrFailureKind::ArtifactVerificationFailed),
                required_failure: mode == NsrMode::Required || fail_fast,
            };
        }
        SlotRuntimeStatus::Incompatible => {
            return NsrOutcome {
                status: NsrStageStatus {
                    slot,
                    mode: mode.as_str().to_string(),
                    state: NsrStageState::Incompatible,
                    used: false,
                    readiness: NsrStageReadiness::ProductionBlocked,
                    detail: "nsr slot incompatible".to_string(),
                    reason_codes: Vec::new(),
                    digest_prefix: None,
                },
                result: Err(NsrFailureKind::Unavailable),
                required_failure: mode == NsrMode::Required || fail_fast,
            };
        }
        SlotRuntimeStatus::Used | SlotRuntimeStatus::Disabled => {}
    }

    let engine = NsrDatalogLiteEngine::default();
    let ctx = NsrContext {
        risk: req.base_risk,
        confidence: req.base_confidence,
        coherence: None,
        instability: None,
        pressure: Some(req.pressure),
        surprise: Some(req.surprise),
        cortisol: None,
        arousal: None,
        has_capability_token: false,
        compute_degraded_ratio: Some(if req.compute_degraded { 1.0 } else { 0.0 }),
    };
    let intent = DecisionIntentSummary {
        action_type: ActionType::Answer,
        tool_kinds: Vec::new(),
        target_domain_hashes: Vec::new(),
        target_path_hashes: Vec::new(),
        output_class: if req.base_risk > 0.75 {
            OutputClass::ExecIntent
        } else {
            OutputClass::SafeText
        },
    };
    let policy_tags = if req.base_risk > 0.8 {
        vec![PolicyTag::Sensitive]
    } else {
        Vec::new()
    };
    let budget = NsrBudget::default();
    match engine.assess(&ctx, &intent, &policy_tags, budget) {
        Ok(assessment) => {
            let reason_codes = assessment
                .reasons
                .iter()
                .map(reason_code_token)
                .collect::<Vec<_>>();
            NsrOutcome {
                status: NsrStageStatus {
                    slot,
                    mode: mode.as_str().to_string(),
                    state: NsrStageState::Used,
                    used: true,
                    readiness: NsrStageReadiness::RuntimePathReady,
                    detail: "nsr post-inference hook executed".to_string(),
                    reason_codes: reason_codes.clone(),
                    digest_prefix: Some(hex::encode(&assessment.digest[..6])),
                },
                result: Ok(NsrResult {
                    risk: assessment.nsr_risk.clamp(0.0, 1.0),
                    confidence: assessment.nsr_confidence.clamp(0.0, 1.0),
                    reason_codes,
                    digest: assessment.digest,
                    engine_id: assessment.engine_id.to_string(),
                    contract_version: NsrContractVersion::V1,
                }),
                required_failure: false,
            }
        }
        Err(err) => {
            let (kind, state, detail) = match err {
                NsrError::BudgetExceeded => (
                    NsrFailureKind::ExecutionError,
                    NsrStageState::ExecutionError,
                    "nsr budget exceeded".to_string(),
                ),
                NsrError::BackendDisabled => (
                    NsrFailureKind::Disabled,
                    NsrStageState::Disabled,
                    "nsr backend disabled".to_string(),
                ),
                NsrError::NotImplemented => (
                    NsrFailureKind::BackendUnavailable,
                    NsrStageState::BackendUnavailable,
                    "nsr backend not implemented".to_string(),
                ),
                NsrError::Unavailable(reason) => (
                    NsrFailureKind::Unavailable,
                    NsrStageState::Unavailable,
                    format!("nsr unavailable: {reason}"),
                ),
            };
            NsrOutcome {
                status: NsrStageStatus {
                    slot,
                    mode: mode.as_str().to_string(),
                    state,
                    used: false,
                    readiness: NsrStageReadiness::ProductionBlocked,
                    detail,
                    reason_codes: Vec::new(),
                    digest_prefix: None,
                },
                result: Err(kind),
                required_failure: mode == NsrMode::Required || fail_fast,
            }
        }
    }
}

fn canonical_failure_for_nsr(kind: NsrFailureKind, detail: String) -> CanonicalPipelineFailure {
    let kind = match kind {
        NsrFailureKind::Disabled => CanonicalFailureKind::NsrDisabled,
        NsrFailureKind::Unavailable => CanonicalFailureKind::NsrUnavailable,
        NsrFailureKind::ArtifactVerificationFailed => {
            CanonicalFailureKind::NsrArtifactVerificationFailed
        }
        NsrFailureKind::ContractMismatch => CanonicalFailureKind::NsrContractMismatch,
        NsrFailureKind::BackendUnavailable => CanonicalFailureKind::NsrBackendUnavailable,
        NsrFailureKind::ExecutionError => CanonicalFailureKind::NsrExecutionError,
    };
    CanonicalPipelineFailure {
        kind,
        stage: None,
        detail,
    }
}

fn reason_code_token(reason: &ReasonCode) -> String {
    match reason {
        ReasonCode::ViolatesDenyByDefault => "violates_deny_by_default".to_string(),
        ReasonCode::CoherenceGateTriggered => "coherence_gate_triggered".to_string(),
        ReasonCode::HighRiskToolRequest => "high_risk_tool_request".to_string(),
        ReasonCode::UntrustedTarget => "untrusted_target".to_string(),
        ReasonCode::BudgetStress => "budget_stress".to_string(),
        ReasonCode::LowConfidenceContext => "low_confidence_context".to_string(),
        ReasonCode::SensitiveOutputClass => "sensitive_output_class".to_string(),
        ReasonCode::PolicyRuleHit(id) => format!("policy_rule_hit_{id}"),
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
    pub fn technical_admission(
        &self,
        request: &CanonicalPipelineRequest,
    ) -> CanonicalAdmissionDecision {
        let input = &request.input;
        let budget = request.budget;
        let pack_meta = self.pack.meta();
        let route = CanonicalBackendRoute {
            pack_id: pack_meta.pack_id.0,
            world_backend: pack_meta.world_backend as u8,
            sae_backend: pack_meta.sae_backend as u8,
            ssm_backend: pack_meta.ssm_backend as u8,
            lfm_backend: pack_meta.lfm_backend as u8,
        };

        if input.t == 0 {
            return CanonicalAdmissionDecision {
                route,
                failure: Some(CanonicalPipelineFailure {
                    kind: CanonicalFailureKind::InvalidInput,
                    stage: None,
                    detail: "invalid request: t must be non-zero".to_string(),
                }),
            };
        }
        if budget.max_micros == 0 || budget.hard_timeout_micros == 0 {
            return CanonicalAdmissionDecision {
                route,
                failure: Some(CanonicalPipelineFailure {
                    kind: CanonicalFailureKind::InvalidInput,
                    stage: None,
                    detail: "invalid budget: max_micros and hard_timeout_micros must be non-zero"
                        .to_string(),
                }),
            };
        }
        if budget.max_micros > budget.hard_timeout_micros {
            return CanonicalAdmissionDecision {
                route,
                failure: Some(CanonicalPipelineFailure {
                    kind: CanonicalFailureKind::ContractMismatch,
                    stage: None,
                    detail: format!(
                        "incompatible budget: max_micros {} exceeds hard_timeout_micros {}",
                        budget.max_micros, budget.hard_timeout_micros
                    ),
                }),
            };
        }
        for (label, units) in [
            ("global", budget.global_work_units),
            ("world", budget.world_units),
            ("sae", budget.sae_units),
            ("ssm", budget.ssm_units),
            ("lfm", budget.lfm_units),
        ] {
            if units == 0 {
                return CanonicalAdmissionDecision {
                    route,
                    failure: Some(CanonicalPipelineFailure {
                        kind: CanonicalFailureKind::BudgetExceeded,
                        stage: None,
                        detail: format!(
                            "request too large for configured work budget: {label}_units=0"
                        ),
                    }),
                };
            }
        }

        if let Some(failure) = first_artifact_failure(self.pack.model_slot_provenance()) {
            return CanonicalAdmissionDecision {
                route,
                failure: Some(failure),
            };
        }

        let registry = StageContractRegistry;
        let requested = StageContractVersion::V1;
        let checks = [
            (
                CanonicalStageId::World,
                StageKind::World,
                pack_meta.world_backend,
                self.pack
                    .world()
                    .lock()
                    .ok()
                    .map(|w| w.contract_version())
                    .unwrap_or(requested),
            ),
            (
                CanonicalStageId::Sae,
                StageKind::Sae,
                pack_meta.sae_backend,
                self.pack.sae().contract_version(),
            ),
            (
                CanonicalStageId::Ssm,
                StageKind::Ssm,
                pack_meta.ssm_backend,
                self.pack
                    .ssm()
                    .lock()
                    .ok()
                    .map(|s| s.contract_version())
                    .unwrap_or(requested),
            ),
            (
                CanonicalStageId::Lfm,
                StageKind::Lfm,
                pack_meta.lfm_backend,
                self.pack
                    .lfm()
                    .lock()
                    .ok()
                    .map(|l| l.contract_version())
                    .unwrap_or(requested),
            ),
        ];
        for (stage_id, stage_kind, backend_id, contract_version) in checks {
            if !registry.supports(stage_kind, backend_id, contract_version)
                || contract_version != requested
            {
                return CanonicalAdmissionDecision {
                    route,
                    failure: Some(CanonicalPipelineFailure {
                        kind: if backend_id == crate::BackendComponentId::Disabled {
                            CanonicalFailureKind::BackendDisabled
                        } else {
                            CanonicalFailureKind::StageContractMismatch
                        },
                        stage: Some(stage_id),
                        detail: format!(
                            "{stage_kind:?} backend {backend_id:?} contract {contract_version:?} unsupported"
                        ),
                    }),
                };
            }
        }

        CanonicalAdmissionDecision {
            route,
            failure: None,
        }
    }

    pub fn compute_canonical(
        &self,
        request: CanonicalPipelineRequest,
    ) -> Result<CanonicalPipelineResult, ComputeError> {
        let input = request.input;
        let budget = request.budget;
        let pack_meta = self.pack.meta();
        let route = CanonicalBackendRoute {
            pack_id: pack_meta.pack_id.0,
            world_backend: pack_meta.world_backend as u8,
            sae_backend: pack_meta.sae_backend as u8,
            ssm_backend: pack_meta.ssm_backend as u8,
            lfm_backend: pack_meta.lfm_backend as u8,
        };
        if input.t == 0 {
            return Ok(self.unavailable_result(
                &input,
                budget,
                route,
                UnavailableResultContext {
                    validation_status: ValidationStatus::Degraded,
                    violation_reason_mask: 0,
                    validation: Some(CanonicalValidationSummary {
                        input: ValidationStatus::Degraded,
                        stage: ValidationStatus::Degraded,
                        artifacts: ValidationStatus::Warned,
                        output: ValidationStatus::Degraded,
                        evidence: ValidationStatus::Warned,
                        violation_reason_mask: 0,
                    }),
                    failure: CanonicalPipelineFailure {
                        kind: CanonicalFailureKind::InvalidInput,
                        stage: None,
                        detail: "invalid request: t must be non-zero".to_string(),
                    },
                    world_stage: None,
                    lfm_stage: None,
                    nsr_stage: None,
                    backend_id: None,
                    budget_stage: None,
                    stage_profiles: None,
                    diagnostics: None,
                },
            ));
        }

        let started_total = Instant::now();
        let mut timing = CanonicalTimingSummary::default();
        let mut global_meter = WorkMeter::new(budget.global_work_units);
        let mut world_meter = WorkMeter::new(budget.world_units);
        let mut sae_meter = WorkMeter::new(budget.sae_units);
        let mut ssm_meter = WorkMeter::new(budget.ssm_units);
        let mut lfm_meter = WorkMeter::new(budget.lfm_units);

        let mut exceeded_stage: Option<&'static str> = None;
        let mut executed_stages = Vec::with_capacity(CANONICAL_STAGE_SEQUENCE.len());
        let registry = StageContractRegistry;
        let requested = StageContractVersion::V1;
        let nsr_mode = NsrMode::from_env();
        let nsr_slot = nsr_slot_provenance(self.pack.model_slot_provenance());
        if let Some(failure) = first_artifact_failure(self.pack.model_slot_provenance()) {
            let world_stage = world_stage_from_slots(self.pack.model_slot_provenance());
            return Ok(self.unavailable_result(
                &input,
                budget,
                route,
                UnavailableResultContext {
                    validation_status: ValidationStatus::Degraded,
                    violation_reason_mask: 0,
                    validation: Some(CanonicalValidationSummary {
                        input: ValidationStatus::Ok,
                        stage: ValidationStatus::Degraded,
                        artifacts: ValidationStatus::Degraded,
                        output: ValidationStatus::Degraded,
                        evidence: ValidationStatus::Warned,
                        violation_reason_mask: 0,
                    }),
                    failure,
                    world_stage: Some(world_stage),
                    lfm_stage: None,
                    nsr_stage: Some(nsr_disabled_stage(nsr_slot.clone(), nsr_mode)),
                    backend_id: None,
                    budget_stage: None,
                    stage_profiles: None,
                    diagnostics: None,
                },
            ));
        }

        let world_started = Instant::now();
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
        let world_slot = world.canonical_slot();
        let previous_state_digest = world.current_state_digest();
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
                    validation: None,
                    failure: CanonicalPipelineFailure {
                        kind,
                        stage: Some(CanonicalStageId::World),
                        detail: format!(
                            "world backend {:?} contract {:?} unsupported",
                            pack_meta.world_backend, requested
                        ),
                    },
                    world_stage: Some(WorldStageStatus {
                        predictor: world_model_name.to_string(),
                        slot: world_slot,
                        slot_status: slot_status_for(world_slot, self.pack.model_slot_provenance()),
                        slot_code: slot_code_for(world_slot, self.pack.model_slot_provenance()),
                        used: false,
                        readiness: readiness_for_unavailable_world(
                            world_slot,
                            self.pack.model_slot_provenance(),
                        ),
                        detail: Some("world contract mismatch".to_string()),
                    }),
                    lfm_stage: None,
                    nsr_stage: None,
                    backend_id: Some(pack_meta.world_backend as u16),
                    budget_stage: None,
                    stage_profiles: None,
                    diagnostics: None,
                },
            ));
        }
        let world_input = WorldModelInput {
            t: input.t,
            context_digest: input.context_digest,
            previous_state_digest,
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
                            validation: None,
                            failure: CanonicalPipelineFailure {
                                kind: budget_failure_kind(stage),
                                stage: Some(CanonicalStageId::World),
                                detail: format!("world stage budget exceeded at {stage}"),
                            },
                            world_stage: None,
                            lfm_stage: None,
                            nsr_stage: None,
                            backend_id: None,
                            budget_stage: Some(stage),
                            stage_profiles: None,
                            diagnostics: None,
                        },
                    ));
                }
                WorldModelOutput::degraded_budget(stage)
            }
            Err(other) => {
                return Ok(self.unavailable_result(
                    &input,
                    budget,
                    route,
                    UnavailableResultContext {
                        validation_status: ValidationStatus::Degraded,
                        violation_reason_mask: 0,
                        validation: None,
                        failure: classify_stage_execution_error(
                            CanonicalStageId::World,
                            other,
                            "world stage execution failed",
                        ),
                        world_stage: None,
                        lfm_stage: None,
                        nsr_stage: None,
                        backend_id: Some(pack_meta.world_backend as u16),
                        budget_stage: None,
                        stage_profiles: None,
                        diagnostics: None,
                    },
                ));
            }
        };
        let span = tracing::info_span!("world_model.step", predictor = world_model_name, t = input.t, pred = %hex::encode(&world_model_out.prediction_digest[..4]));
        let _enter = span.enter();
        let world_stage = WorldStageStatus {
            predictor: world_model_name.to_string(),
            slot: world_slot,
            slot_status: slot_status_for(world_slot, self.pack.model_slot_provenance()),
            slot_code: slot_code_for(world_slot, self.pack.model_slot_provenance()),
            used: true,
            readiness: readiness_from_runtime(
                world_model_out.quality,
                world_slot,
                self.pack.model_slot_provenance(),
            ),
            detail: world_model_out.notes.first().cloned(),
        };
        timing.world_micros = Some(world_started.elapsed().as_micros() as u64);
        drop(world);
        executed_stages.push(CanonicalStageId::World);

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

        let sae_started = Instant::now();
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
                    validation: None,
                    failure: CanonicalPipelineFailure {
                        kind,
                        stage: Some(CanonicalStageId::Sae),
                        detail: format!(
                            "sae backend {:?} contract {:?} unsupported",
                            pack_meta.sae_backend, requested
                        ),
                    },
                    world_stage: None,
                    lfm_stage: None,
                    nsr_stage: None,
                    backend_id: Some(pack_meta.sae_backend as u16),
                    budget_stage: None,
                    stage_profiles: None,
                    diagnostics: None,
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
                                validation: None,
                                failure: CanonicalPipelineFailure {
                                    kind: budget_failure_kind(stage),
                                    stage: Some(CanonicalStageId::Sae),
                                    detail: format!("sae stage budget exceeded at {stage}"),
                                },
                                world_stage: None,
                                lfm_stage: None,
                                nsr_stage: None,
                                backend_id: None,
                                budget_stage: Some(stage),
                                stage_profiles: None,
                                diagnostics: None,
                            },
                        ));
                    }
                    (Self::empty_sae(), true)
                }
                Err(other) => {
                    return Ok(self.unavailable_result(
                        &input,
                        budget,
                        route,
                        UnavailableResultContext {
                            validation_status: ValidationStatus::Degraded,
                            violation_reason_mask: validation_report.violation_mask,
                            validation: None,
                            failure: classify_stage_execution_error(
                                CanonicalStageId::Sae,
                                other,
                                "sae stage execution failed",
                            ),
                            world_stage: None,
                            lfm_stage: None,
                            nsr_stage: None,
                            backend_id: Some(pack_meta.sae_backend as u16),
                            budget_stage: None,
                            stage_profiles: None,
                            diagnostics: None,
                        },
                    ));
                }
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
                            validation: None,
                            failure: CanonicalPipelineFailure {
                                kind: budget_failure_kind(stage),
                                stage: Some(CanonicalStageId::Sae),
                                detail: format!("sae stage budget exceeded at {stage}"),
                            },
                            world_stage: None,
                            lfm_stage: None,
                            nsr_stage: None,
                            backend_id: None,
                            budget_stage: Some(stage),
                            stage_profiles: None,
                            diagnostics: None,
                        },
                    ));
                }
                (Self::empty_sae(), true)
            }
            Err(other) => return Err(other),
        };

        metrics::histogram!("ucf_sae_spike_count").record(f64::from(sae_out.spike_count));
        executed_stages.push(CanonicalStageId::Sae);
        timing.sae_micros = Some(sae_started.elapsed().as_micros() as u64);
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

        let ssm_started = Instant::now();
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
                    validation: None,
                    failure: CanonicalPipelineFailure {
                        kind,
                        stage: Some(CanonicalStageId::Ssm),
                        detail: format!(
                            "ssm backend {:?} contract {:?} unsupported",
                            pack_meta.ssm_backend, requested
                        ),
                    },
                    world_stage: None,
                    lfm_stage: None,
                    nsr_stage: None,
                    backend_id: Some(pack_meta.ssm_backend as u16),
                    budget_stage: None,
                    stage_profiles: None,
                    diagnostics: None,
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
                                validation: None,
                                failure: CanonicalPipelineFailure {
                                    kind: budget_failure_kind(stage),
                                    stage: Some(CanonicalStageId::Ssm),
                                    detail: format!("ssm stage budget exceeded at {stage}"),
                                },
                                world_stage: None,
                                lfm_stage: None,
                                nsr_stage: None,
                                backend_id: None,
                                budget_stage: Some(stage),
                                stage_profiles: None,
                                diagnostics: None,
                            },
                        ));
                    }
                    (SsmOutput::degraded("budget_exceeded"), true)
                }
                Err(other) => {
                    return Ok(self.unavailable_result(
                        &input,
                        budget,
                        route,
                        UnavailableResultContext {
                            validation_status: ValidationStatus::Degraded,
                            violation_reason_mask: validation_report.violation_mask,
                            validation: None,
                            failure: classify_stage_execution_error(
                                CanonicalStageId::Ssm,
                                other,
                                "ssm stage execution failed",
                            ),
                            world_stage: None,
                            lfm_stage: None,
                            nsr_stage: None,
                            backend_id: Some(pack_meta.ssm_backend as u16),
                            budget_stage: None,
                            stage_profiles: None,
                            diagnostics: None,
                        },
                    ));
                }
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
                            validation: None,
                            failure: CanonicalPipelineFailure {
                                kind: budget_failure_kind(stage),
                                stage: Some(CanonicalStageId::Ssm),
                                detail: format!("ssm stage budget exceeded at {stage}"),
                            },
                            world_stage: None,
                            lfm_stage: None,
                            nsr_stage: None,
                            backend_id: None,
                            budget_stage: Some(stage),
                            stage_profiles: None,
                            diagnostics: None,
                        },
                    ));
                }
                (SsmOutput::degraded("budget_exceeded"), true)
            }
            Err(other) => return Err(other),
        };

        validation_report =
            validation_report.merge(SsmValidatorV1::validate(&ssm_input, &ssm_out, None));
        executed_stages.push(CanonicalStageId::Ssm);
        timing.ssm_micros = Some(ssm_started.elapsed().as_micros() as u64);
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

        let lfm_started = Instant::now();
        global_meter.spend(220, "lfm/step")?;
        let lfm_stage_disabled = pack_meta.lfm_backend == crate::BackendComponentId::Disabled;
        if !lfm_stage_disabled
            && !registry.supports(
                StageKind::Lfm,
                pack_meta.lfm_backend,
                self.pack
                    .lfm()
                    .lock()
                    .map_err(|_| ComputeError::InvalidInput {
                        reason: "lfm mutex poisoned".to_string(),
                    })?
                    .contract_version(),
            )
        {
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
                    validation: None,
                    failure: CanonicalPipelineFailure {
                        kind,
                        stage: Some(CanonicalStageId::Lfm),
                        detail: format!(
                            "lfm backend {:?} contract {:?} unsupported",
                            pack_meta.lfm_backend, requested
                        ),
                    },
                    world_stage: None,
                    lfm_stage: None,
                    nsr_stage: None,
                    backend_id: Some(pack_meta.lfm_backend as u16),
                    budget_stage: None,
                    stage_profiles: None,
                    diagnostics: None,
                },
            ));
        }
        let (lfm_out, lfm_degraded, lfm_budget_degraded, lfm_name): (
            LfmOutput,
            bool,
            bool,
            String,
        ) = if lfm_stage_disabled {
            (
                LfmOutput::degraded("backend_disabled"),
                true,
                false,
                "disabled".to_string(),
            )
        } else {
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
            let lfm_stage_started = Instant::now();
            let result = match lfm_meter.spend(220, "lfm/step") {
                Ok(()) => match lfm.step(&lfm_input, budget) {
                    Ok(output) => (output, false, false),
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
                                    validation: None,
                                    failure: CanonicalPipelineFailure {
                                        kind: budget_failure_kind(stage),
                                        stage: Some(CanonicalStageId::Lfm),
                                        detail: format!("lfm stage budget exceeded at {stage}"),
                                    },
                                    world_stage: None,
                                    lfm_stage: None,
                                    nsr_stage: None,
                                    backend_id: None,
                                    budget_stage: Some(stage),
                                    stage_profiles: None,
                                    diagnostics: None,
                                },
                            ));
                        }
                        (LfmOutput::degraded("budget_exceeded"), true, true)
                    }
                    Err(other) => {
                        return Ok(self.unavailable_result(
                            &input,
                            budget,
                            route,
                            UnavailableResultContext {
                                validation_status: ValidationStatus::Degraded,
                                violation_reason_mask: validation_report.violation_mask,
                                validation: None,
                                failure: classify_stage_execution_error(
                                    CanonicalStageId::Lfm,
                                    other,
                                    "lfm stage execution failed",
                                ),
                                world_stage: None,
                                lfm_stage: None,
                                nsr_stage: None,
                                backend_id: Some(pack_meta.lfm_backend as u16),
                                budget_stage: None,
                                stage_profiles: None,
                                diagnostics: None,
                            },
                        ));
                    }
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
                                validation: None,
                                failure: CanonicalPipelineFailure {
                                    kind: budget_failure_kind(stage),
                                    stage: Some(CanonicalStageId::Lfm),
                                    detail: format!("lfm stage budget exceeded at {stage}"),
                                },
                                world_stage: None,
                                lfm_stage: None,
                                nsr_stage: None,
                                backend_id: None,
                                budget_stage: Some(stage),
                                stage_profiles: None,
                                diagnostics: None,
                            },
                        ));
                    }
                    (LfmOutput::degraded("budget_exceeded"), true, true)
                }
                Err(other) => return Err(other),
            };
            metrics::histogram!("ucf_lfm_ode_step_micros")
                .record(lfm_stage_started.elapsed().as_micros() as f64);
            (result.0, result.1, result.2, lfm_name.to_string())
        };
        let plasticity_record = lfm_out.plasticity.clone();
        if !lfm_stage_disabled {
            executed_stages.push(CanonicalStageId::Lfm);
        }
        timing.lfm_micros = Some(lfm_started.elapsed().as_micros() as u64);

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
        let lfm_slot = lfm_slot_provenance(self.pack.model_slot_provenance());
        let lfm_stage = if lfm_stage_disabled {
            LfmStageStatus {
                slot: lfm_slot.clone(),
                state: LfmStageState::Disabled,
                used: false,
                runtime: lfm_name.clone(),
                backend: pack_meta.lfm_backend as u8,
                readiness: LfmStageReadiness::Scaffolded,
                detail: "lfm backend disabled".to_string(),
            }
        } else if lfm_budget_degraded {
            LfmStageStatus {
                slot: lfm_slot.clone(),
                state: LfmStageState::DegradedBypass,
                used: true,
                runtime: lfm_name.clone(),
                backend: pack_meta.lfm_backend as u8,
                readiness: LfmStageReadiness::RuntimePathReady,
                detail: "lfm budget exceeded, degraded output used".to_string(),
            }
        } else {
            LfmStageStatus {
                slot: lfm_slot.clone(),
                state: LfmStageState::Used,
                used: true,
                runtime: lfm_name.clone(),
                backend: pack_meta.lfm_backend as u8,
                readiness: if lfm_out.quality == StageQuality::Ok {
                    LfmStageReadiness::RuntimePathReady
                } else {
                    LfmStageReadiness::ProductionBlocked
                },
                detail: format!("lfm quality={:?}", lfm_out.quality),
            }
        };
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
        let nsr_request = NsrRequest {
            base_risk: risk_signal.risk,
            base_confidence: risk_signal.confidence,
            pressure,
            surprise,
            compute_degraded: quality != SignalQuality::VerifiedPipeline,
            contract_version: NsrContractVersion::V1,
        };
        let nsr_outcome = run_nsr_stage(
            nsr_mode,
            nsr_slot.clone(),
            &nsr_request,
            budget.degrade_policy == DegradePolicy::FailFast,
        );
        if let Ok(result) = &nsr_outcome.result {
            risk_signal.risk = clamp01(risk_signal.risk.max(result.risk));
            risk_signal.confidence = clamp01(risk_signal.confidence.min(result.confidence));
        } else if let Err(kind) = &nsr_outcome.result {
            if nsr_outcome.required_failure {
                return Ok(self.unavailable_result(
                    &input,
                    budget,
                    route,
                    UnavailableResultContext {
                        validation_status: ValidationStatus::Degraded,
                        violation_reason_mask: validation_report.violation_mask,
                        validation: None,
                        failure: canonical_failure_for_nsr(
                            *kind,
                            format!("required nsr stage failed: {}", nsr_outcome.status.detail),
                        ),
                        world_stage: Some(world_stage.clone()),
                        lfm_stage: Some(lfm_stage.clone()),
                        nsr_stage: Some(nsr_outcome.status.clone()),
                        backend_id: None,
                        budget_stage: None,
                        stage_profiles: None,
                        diagnostics: None,
                    },
                ));
            }
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
            nsr_digest: nsr_outcome.result.as_ref().ok().map(|result| result.digest),
            nsr_status: nsr_outcome.status.state.as_u8(),
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
            format!("world_readiness={:?}", world_stage.readiness).to_ascii_lowercase(),
            format!("feature_extractor={}", self.pack.sae().name()),
            format!("working_memory={}", ssm_name),
            format!("lfm={}", lfm_name),
            format!("lfm_state={:?}", lfm_stage.state).to_ascii_lowercase(),
            format!("lfm_readiness={:?}", lfm_stage.readiness).to_ascii_lowercase(),
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
        if lfm_budget_degraded {
            notes.push("degraded:lfm_budget_exceeded".to_string());
        }
        if lfm_stage_disabled {
            notes.push("degraded:lfm_skipped_backend_disabled".to_string());
        }
        notes.push(format!("nsr_mode={}", nsr_outcome.status.mode));
        notes.push(format!("nsr_state={:?}", nsr_outcome.status.state).to_ascii_lowercase());
        if let Some(prefix) = &nsr_outcome.status.digest_prefix {
            notes.push(format!("nsr_digest={prefix}"));
        }
        let evidence_chain = crate::evidence::EvidenceChain::from_compute(
            &input,
            &sae_out.spikes,
            &risk_signal,
            nsr_outcome.result.as_ref().ok().map(|result| result.digest),
            nsr_outcome.status.state.as_u8(),
            Some(sae_out.quality),
            Some(ssm_out.quality),
            Some(lfm_out.quality),
        );
        let chain_report = validate_evidence_chain_digest(&evidence_chain);
        validation_report = validation_report.merge(chain_report);
        timing.total_micros = started_total.elapsed().as_micros() as u64;

        let validation = CanonicalValidationSummary {
            input: ValidationStatus::Ok,
            stage: validation_report.status,
            artifacts: if first_artifact_failure(self.pack.model_slot_provenance()).is_some() {
                ValidationStatus::Degraded
            } else {
                ValidationStatus::Ok
            },
            output: if quality == SignalQuality::DegradedFallback {
                ValidationStatus::Warned
            } else {
                ValidationStatus::Ok
            },
            evidence: chain_report.status,
            violation_reason_mask: validation_report.violation_mask,
        };

        let work = CanonicalWorkSummary {
            global_budget_units: budget.global_work_units,
            global_remaining_units: global_meter.remaining(),
            world_remaining_units: world_meter.remaining(),
            sae_remaining_units: sae_meter.remaining(),
            ssm_remaining_units: ssm_meter.remaining(),
            lfm_remaining_units: lfm_meter.remaining(),
            budget_exceeded_stage: exceeded_stage,
        };
        let diagnostics = CanonicalRunDiagnostics {
            timing,
            work,
            stage_profiles: build_stage_profiles(
                timing,
                work,
                &world_stage,
                &lfm_stage,
                &executed_stages,
                sae_degraded,
                ssm_degraded,
                lfm_budget_degraded,
                exceeded_stage,
                None,
            ),
            stage_cost_attribution: Vec::new(),
            hotspots: CanonicalHotspotSummary {
                slowest_stage: None,
                dominant_stage: None,
                dominant_stage_share_bps: None,
                dominant_work_stage: None,
                dominant_work_stage_share_bps: None,
                degraded_stage: None,
                fallback_stage: None,
                degraded_stage_count: 0,
                skipped_stage_count: 0,
                unavailable_stage_count: 0,
                failed_stage_count: 0,
            },
            evidence_chain_digest_prefix: Some(evidence_chain.digest_prefix_hex()),
        };
        let stage_cost_attribution =
            build_stage_cost_attribution(&diagnostics.stage_profiles, timing, work, budget);
        let hotspots = build_hotspot_summary(&diagnostics.stage_profiles, &stage_cost_attribution);
        let diagnostics = CanonicalRunDiagnostics {
            stage_cost_attribution,
            hotspots,
            ..diagnostics
        };

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
        if let Some(pipeline_failure) = &failure {
            let classification = pipeline_failure.classification();
            notes.push(format!("fault_domain={:?}", classification.domain).to_ascii_lowercase());
            notes.push(
                format!("fault_isolation={:?}", classification.isolation).to_ascii_lowercase(),
            );
            notes.push(format!("fault_systemic={}", classification.systemic));
        }
        notes.sort();

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
            nsr_digest: nsr_outcome.result.as_ref().ok().map(|result| result.digest),
            nsr_status: nsr_outcome.status.state.as_u8(),
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
            executed_stages,
            route,
            state,
            failure,
            validation_status: signals.validation_status,
            violation_reason_mask: signals.violation_reason_mask,
            validation,
            diagnostics,
            world_stage,
            lfm_stage,
            nsr_stage: nsr_outcome.status,
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
        let classification = ctx.failure.classification();
        signals
            .notes
            .push(format!("fault_domain={:?}", classification.domain).to_ascii_lowercase());
        signals
            .notes
            .push(format!("fault_isolation={:?}", classification.isolation).to_ascii_lowercase());
        signals
            .notes
            .push(format!("fault_systemic={}", classification.systemic));
        let validation = ctx
            .validation
            .unwrap_or_else(CanonicalValidationSummary::unavailable);
        let diagnostics = ctx.diagnostics.unwrap_or(CanonicalRunDiagnostics {
            timing: CanonicalTimingSummary::default(),
            work: CanonicalWorkSummary {
                global_budget_units: budget.global_work_units,
                global_remaining_units: budget.global_work_units,
                world_remaining_units: budget.world_units,
                sae_remaining_units: budget.sae_units,
                ssm_remaining_units: budget.ssm_units,
                lfm_remaining_units: budget.lfm_units,
                budget_exceeded_stage: ctx.budget_stage,
            },
            stage_profiles: ctx.stage_profiles.unwrap_or_else(|| {
                default_unavailable_stage_profiles(
                    ctx.failure.stage,
                    ctx.failure.kind == CanonicalFailureKind::ExecutionError
                        || ctx.failure.kind == CanonicalFailureKind::Timeout,
                )
            }),
            stage_cost_attribution: Vec::new(),
            hotspots: CanonicalHotspotSummary {
                slowest_stage: None,
                dominant_stage: None,
                dominant_stage_share_bps: None,
                dominant_work_stage: None,
                dominant_work_stage_share_bps: None,
                degraded_stage: None,
                fallback_stage: None,
                degraded_stage_count: 0,
                skipped_stage_count: 0,
                unavailable_stage_count: 0,
                failed_stage_count: 0,
            },
            evidence_chain_digest_prefix: None,
        });
        let stage_cost_attribution = if diagnostics.stage_cost_attribution.is_empty() {
            build_stage_cost_attribution(
                &diagnostics.stage_profiles,
                diagnostics.timing,
                diagnostics.work,
                budget,
            )
        } else {
            diagnostics.stage_cost_attribution.clone()
        };
        let diagnostics = CanonicalRunDiagnostics {
            stage_cost_attribution: stage_cost_attribution.clone(),
            hotspots: build_hotspot_summary(&diagnostics.stage_profiles, &stage_cost_attribution),
            ..diagnostics
        };
        CanonicalPipelineResult {
            request: input.clone(),
            stage_order: CANONICAL_STAGE_SEQUENCE,
            executed_stages: Vec::new(),
            route,
            state: CanonicalPipelineState::Unavailable,
            failure: Some(ctx.failure),
            validation_status: ctx.validation_status,
            violation_reason_mask: ctx.violation_reason_mask,
            validation,
            diagnostics,
            world_stage: ctx
                .world_stage
                .unwrap_or_else(|| world_stage_from_slots(self.pack.model_slot_provenance())),
            lfm_stage: ctx
                .lfm_stage
                .unwrap_or_else(|| lfm_stage_from_slots(self.pack.model_slot_provenance())),
            nsr_stage: ctx.nsr_stage.unwrap_or_else(|| {
                nsr_disabled_stage(
                    nsr_slot_provenance(self.pack.model_slot_provenance()),
                    NsrMode::Disabled,
                )
            }),
            model_slots: self.pack.model_slot_provenance().to_vec(),
            signals,
        }
    }
}

fn stage_timing_for(stage: CanonicalStageId, timing: CanonicalTimingSummary) -> Option<u64> {
    match stage {
        CanonicalStageId::World => timing.world_micros,
        CanonicalStageId::Sae => timing.sae_micros,
        CanonicalStageId::Ssm => timing.ssm_micros,
        CanonicalStageId::Lfm => timing.lfm_micros,
    }
}

fn default_unavailable_stage_profiles(
    failed_stage: Option<CanonicalStageId>,
    hard_failure: bool,
) -> Vec<CanonicalStageProfile> {
    CANONICAL_STAGE_SEQUENCE
        .iter()
        .map(|stage| CanonicalStageProfile {
            stage: *stage,
            state: if Some(*stage) == failed_stage {
                if hard_failure {
                    CanonicalStageProfileState::Failed
                } else {
                    CanonicalStageProfileState::Unavailable
                }
            } else {
                CanonicalStageProfileState::Unavailable
            },
            duration_micros: None,
            remaining_work_units: None,
            detail: None,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_stage_profiles(
    timing: CanonicalTimingSummary,
    work: CanonicalWorkSummary,
    world_stage: &WorldStageStatus,
    lfm_stage: &LfmStageStatus,
    executed_stages: &[CanonicalStageId],
    sae_degraded: bool,
    ssm_degraded: bool,
    lfm_budget_degraded: bool,
    exceeded_stage: Option<&'static str>,
    failure: Option<&CanonicalPipelineFailure>,
) -> Vec<CanonicalStageProfile> {
    let slow_threshold_micros = timing.total_micros.saturating_mul(60) / 100;
    CANONICAL_STAGE_SEQUENCE
        .iter()
        .map(|stage| {
            let duration = stage_timing_for(*stage, timing);
            let stage_executed = executed_stages.contains(stage);
            let mut state = if !stage_executed {
                CanonicalStageProfileState::Unavailable
            } else {
                CanonicalStageProfileState::Success
            };
            let mut detail = None;
            if let Some(run_failure) = failure {
                if run_failure.stage == Some(*stage) {
                    state = CanonicalStageProfileState::Failed;
                    detail = Some(run_failure.detail.clone());
                }
            }
            match stage {
                CanonicalStageId::World => {
                    if world_stage.used
                        && world_stage.readiness == WorldStageReadiness::ProductionBlocked
                    {
                        state = CanonicalStageProfileState::Degraded;
                        detail = world_stage.detail.clone();
                    }
                }
                CanonicalStageId::Sae => {
                    if sae_degraded {
                        state = CanonicalStageProfileState::Degraded;
                        detail = Some("sae budget degraded fallback".to_string());
                    }
                }
                CanonicalStageId::Ssm => {
                    if ssm_degraded {
                        state = CanonicalStageProfileState::Degraded;
                        detail = Some("ssm budget degraded fallback".to_string());
                    }
                }
                CanonicalStageId::Lfm => {
                    if lfm_stage.state == LfmStageState::Disabled {
                        state = CanonicalStageProfileState::Skipped;
                        detail = Some("lfm backend disabled".to_string());
                    } else if lfm_budget_degraded
                        || lfm_stage.state == LfmStageState::DegradedBypass
                    {
                        state = CanonicalStageProfileState::Degraded;
                        detail = Some("lfm budget degraded fallback".to_string());
                    }
                }
            }
            if let Some(micros) = duration {
                if state == CanonicalStageProfileState::Success && micros >= slow_threshold_micros {
                    state = CanonicalStageProfileState::SlowSuccess;
                }
            }
            if let Some(budget_stage) = exceeded_stage {
                if matches!(
                    (*stage, budget_stage),
                    (CanonicalStageId::World, "world_model/step")
                        | (CanonicalStageId::Sae, "sae/extract")
                        | (CanonicalStageId::Ssm, "ssm/step")
                        | (CanonicalStageId::Lfm, "lfm/step")
                ) && detail.is_none()
                {
                    detail = Some(format!("capacity pressure at {budget_stage}"));
                }
            }
            CanonicalStageProfile {
                stage: *stage,
                state,
                duration_micros: duration,
                remaining_work_units: Some(match stage {
                    CanonicalStageId::World => work.world_remaining_units,
                    CanonicalStageId::Sae => work.sae_remaining_units,
                    CanonicalStageId::Ssm => work.ssm_remaining_units,
                    CanonicalStageId::Lfm => work.lfm_remaining_units,
                }),
                detail,
            }
        })
        .collect()
}

fn build_hotspot_summary(
    stage_profiles: &[CanonicalStageProfile],
    stage_cost_attribution: &[CanonicalStageCostAttribution],
) -> CanonicalHotspotSummary {
    let mut slowest_stage = None;
    let mut slowest_micros = 0_u64;
    let mut dominant_stage = None;
    let mut dominant_share_bps = None;
    let mut dominant_work_stage = None;
    let mut dominant_work_stage_share_bps = None;
    let mut degraded_stage = None;
    let mut fallback_stage = None;
    let mut degraded_stage_count = 0_u8;
    let mut skipped_stage_count = 0_u8;
    let mut unavailable_stage_count = 0_u8;
    let mut failed_stage_count = 0_u8;
    for profile in stage_profiles {
        match profile.state {
            CanonicalStageProfileState::Degraded => degraded_stage_count += 1,
            CanonicalStageProfileState::Skipped => {
                skipped_stage_count += 1;
                if fallback_stage.is_none() {
                    fallback_stage = Some(profile.stage);
                }
            }
            CanonicalStageProfileState::Unavailable => unavailable_stage_count += 1,
            CanonicalStageProfileState::Failed => failed_stage_count += 1,
            CanonicalStageProfileState::Success | CanonicalStageProfileState::SlowSuccess => {}
        }
        if let Some(duration) = profile.duration_micros {
            if duration >= slowest_micros {
                slowest_micros = duration;
                slowest_stage = Some(profile.stage);
            }
        }
    }
    for attribution in stage_cost_attribution {
        if attribution.dominant_timing {
            dominant_stage = Some(attribution.stage);
            dominant_share_bps = attribution.timing_share_bps;
        }
        if attribution.dominant_work {
            dominant_work_stage = Some(attribution.stage);
            dominant_work_stage_share_bps = attribution.work_share_bps;
        }
        if degraded_stage.is_none()
            && attribution.pattern == CanonicalStageCostPattern::DegradedPathDriver
        {
            degraded_stage = Some(attribution.stage);
        }
        if fallback_stage.is_none()
            && attribution.pattern == CanonicalStageCostPattern::SkippedOrFallback
        {
            fallback_stage = Some(attribution.stage);
        }
    }
    if dominant_stage.is_none() && slowest_micros > 0 {
        dominant_stage = slowest_stage;
    }
    CanonicalHotspotSummary {
        slowest_stage,
        dominant_stage,
        dominant_stage_share_bps: dominant_share_bps,
        dominant_work_stage,
        dominant_work_stage_share_bps,
        degraded_stage,
        fallback_stage,
        degraded_stage_count,
        skipped_stage_count,
        unavailable_stage_count,
        failed_stage_count,
    }
}

fn stage_budget_for(stage: CanonicalStageId, budget: ComputeBudget) -> u64 {
    match stage {
        CanonicalStageId::World => budget.world_units,
        CanonicalStageId::Sae => budget.sae_units,
        CanonicalStageId::Ssm => budget.ssm_units,
        CanonicalStageId::Lfm => budget.lfm_units,
    }
}

fn build_stage_cost_attribution(
    stage_profiles: &[CanonicalStageProfile],
    timing: CanonicalTimingSummary,
    work: CanonicalWorkSummary,
    budget: ComputeBudget,
) -> Vec<CanonicalStageCostAttribution> {
    let mut dominant_timing_stage = None;
    let mut dominant_timing_micros = 0_u64;
    let mut dominant_work_stage = None;
    let mut dominant_work_units = 0_u64;

    let mut precomputed = Vec::with_capacity(stage_profiles.len());
    for profile in stage_profiles {
        let stage_budget_units = stage_budget_for(profile.stage, budget);
        let remaining_work_units = profile.remaining_work_units.unwrap_or(stage_budget_units);
        let consumed_work_units = stage_budget_units.saturating_sub(remaining_work_units);
        if profile.duration_micros.unwrap_or_default() >= dominant_timing_micros {
            dominant_timing_micros = profile.duration_micros.unwrap_or_default();
            dominant_timing_stage = Some(profile.stage);
        }
        if consumed_work_units >= dominant_work_units {
            dominant_work_units = consumed_work_units;
            dominant_work_stage = Some(profile.stage);
        }
        precomputed.push((profile, consumed_work_units));
    }

    precomputed
        .into_iter()
        .map(|(profile, consumed_work_units)| {
            let timing_share_bps = profile.duration_micros.and_then(|micros| {
                (timing.total_micros > 0).then_some(
                    (micros.saturating_mul(10_000) / timing.total_micros).min(10_000) as u16,
                )
            });
            let work_share_bps = (work.global_budget_units > 0).then_some(
                (consumed_work_units.saturating_mul(10_000) / work.global_budget_units).min(10_000)
                    as u16,
            );
            let dominant_timing = dominant_timing_stage == Some(profile.stage)
                && profile.duration_micros.unwrap_or_default() > 0;
            let dominant_work =
                dominant_work_stage == Some(profile.stage) && consumed_work_units > 0;
            let pattern = match profile.state {
                CanonicalStageProfileState::Failed => CanonicalStageCostPattern::HardFailure,
                CanonicalStageProfileState::Degraded => {
                    CanonicalStageCostPattern::DegradedPathDriver
                }
                CanonicalStageProfileState::Skipped | CanonicalStageProfileState::Unavailable => {
                    CanonicalStageCostPattern::SkippedOrFallback
                }
                CanonicalStageProfileState::SlowSuccess => {
                    CanonicalStageCostPattern::SlowButHealthy
                }
                CanonicalStageProfileState::Success => {
                    if dominant_timing || dominant_work {
                        CanonicalStageCostPattern::DominantCostDriver
                    } else {
                        CanonicalStageCostPattern::Inactive
                    }
                }
            };
            CanonicalStageCostAttribution {
                stage: profile.stage,
                state: profile.state,
                timing_micros: profile.duration_micros,
                timing_share_bps,
                work_consumed_units: consumed_work_units,
                work_share_bps,
                pattern,
                dominant_timing,
                dominant_work,
                timing_provenance: StageCostSignalProvenance::MeasuredTiming,
                work_provenance: StageCostSignalProvenance::DerivedFromBudgetAndMeter,
                detail: profile.detail.clone(),
            }
        })
        .collect()
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
                    Some(ArtifactFailureCode::ActivationBlocked) => {
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

fn slot_status_for(
    slot: Option<crate::ModelSlot>,
    slots: &[ModelSlotProvenance],
) -> Option<SlotRuntimeStatus> {
    slot.and_then(|target| {
        slots
            .iter()
            .find(|entry| entry.slot == target)
            .map(|entry| entry.status)
    })
}

fn slot_code_for(
    slot: Option<crate::ModelSlot>,
    slots: &[ModelSlotProvenance],
) -> Option<ArtifactFailureCode> {
    slot.and_then(|target| {
        slots
            .iter()
            .find(|entry| entry.slot == target)
            .and_then(|entry| entry.code)
    })
}

fn readiness_for_unavailable_world(
    slot: Option<crate::ModelSlot>,
    slots: &[ModelSlotProvenance],
) -> WorldStageReadiness {
    match slot_status_for(slot, slots) {
        Some(SlotRuntimeStatus::Used) => WorldStageReadiness::ContractReady,
        Some(SlotRuntimeStatus::Disabled) => WorldStageReadiness::Scaffolded,
        Some(SlotRuntimeStatus::Unavailable)
        | Some(SlotRuntimeStatus::VerificationFailed)
        | Some(SlotRuntimeStatus::Incompatible) => WorldStageReadiness::ProductionBlocked,
        None => WorldStageReadiness::Scaffolded,
    }
}

fn readiness_from_runtime(
    quality: StageQuality,
    slot: Option<crate::ModelSlot>,
    slots: &[ModelSlotProvenance],
) -> WorldStageReadiness {
    match quality {
        StageQuality::Ok => {
            if slot_status_for(slot, slots) == Some(SlotRuntimeStatus::Used) {
                WorldStageReadiness::RuntimePathReady
            } else {
                WorldStageReadiness::ContractReady
            }
        }
        StageQuality::DegradedFallback => WorldStageReadiness::ArtifactReady,
        StageQuality::Unavailable => WorldStageReadiness::ProductionBlocked,
    }
}

fn world_stage_from_slots(slots: &[ModelSlotProvenance]) -> WorldStageStatus {
    let world = slots
        .iter()
        .find(|entry| entry.slot == crate::ModelSlot::WorldJepa);
    let status = world.map(|entry| entry.status);
    WorldStageStatus {
        predictor: "unavailable".to_string(),
        slot: Some(crate::ModelSlot::WorldJepa),
        slot_status: status,
        slot_code: world.and_then(|entry| entry.code),
        used: false,
        readiness: match status {
            Some(SlotRuntimeStatus::Used) => WorldStageReadiness::ArtifactReady,
            Some(SlotRuntimeStatus::Disabled) => WorldStageReadiness::Scaffolded,
            Some(SlotRuntimeStatus::Unavailable)
            | Some(SlotRuntimeStatus::VerificationFailed)
            | Some(SlotRuntimeStatus::Incompatible)
            | None => WorldStageReadiness::ProductionBlocked,
        },
        detail: world.and_then(|entry| entry.detail.clone()),
    }
}

fn lfm_slot_provenance(slots: &[ModelSlotProvenance]) -> Option<ModelSlotProvenance> {
    slots
        .iter()
        .find(|entry| entry.slot == crate::ModelSlot::Lfm)
        .cloned()
}

fn lfm_stage_from_slots(slots: &[ModelSlotProvenance]) -> LfmStageStatus {
    let slot = lfm_slot_provenance(slots);
    let (state, readiness, detail) = match slot.as_ref().map(|entry| entry.status) {
        Some(SlotRuntimeStatus::Used) => (
            LfmStageState::BackendUnavailable,
            LfmStageReadiness::ArtifactReady,
            "lfm slot validated but runtime path not executed".to_string(),
        ),
        Some(SlotRuntimeStatus::Disabled) => (
            LfmStageState::Disabled,
            LfmStageReadiness::Scaffolded,
            "lfm slot disabled".to_string(),
        ),
        Some(SlotRuntimeStatus::Unavailable) => (
            LfmStageState::Unavailable,
            LfmStageReadiness::ProductionBlocked,
            "lfm slot unavailable".to_string(),
        ),
        Some(SlotRuntimeStatus::VerificationFailed) => (
            LfmStageState::VerificationFailed,
            LfmStageReadiness::ProductionBlocked,
            "lfm slot verification failed".to_string(),
        ),
        Some(SlotRuntimeStatus::Incompatible) => (
            LfmStageState::Incompatible,
            LfmStageReadiness::ProductionBlocked,
            "lfm slot incompatible".to_string(),
        ),
        None => (
            LfmStageState::Disabled,
            LfmStageReadiness::Scaffolded,
            "lfm slot missing from runtime provenance".to_string(),
        ),
    };
    LfmStageStatus {
        slot,
        state,
        used: false,
        runtime: "unavailable".to_string(),
        backend: crate::BackendComponentId::Disabled as u8,
        readiness,
        detail,
    }
}

fn classify_stage_execution_error(
    stage: CanonicalStageId,
    err: ComputeError,
    detail_prefix: &str,
) -> CanonicalPipelineFailure {
    let (kind, detail) = match err {
        ComputeError::BackendDisabled => (
            CanonicalFailureKind::BackendDisabled,
            "backend disabled".to_string(),
        ),
        ComputeError::Internal { reason } if reason.starts_with("candle_stage_unavailable:") => {
            (CanonicalFailureKind::StageUnavailable, reason)
        }
        other => (CanonicalFailureKind::ExecutionError, other.to_string()),
    };
    CanonicalPipelineFailure {
        kind,
        stage: Some(stage),
        detail: format!("{detail_prefix}: {detail}"),
    }
}

fn budget_failure_kind(stage: &'static str) -> CanonicalFailureKind {
    if stage.contains("timeout") {
        CanonicalFailureKind::Timeout
    } else {
        CanonicalFailureKind::BudgetExceeded
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "compute-burn")]
    use sha2::{Digest, Sha256};
    #[cfg(feature = "compute-burn")]
    use std::fs;

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
        assert_eq!(result.executed_stages, CANONICAL_STAGE_SEQUENCE.to_vec());
        assert_ne!(result.state, CanonicalPipelineState::Unavailable);
        assert_eq!(
            result.diagnostics.stage_profiles.len(),
            CANONICAL_STAGE_SEQUENCE.len()
        );
        assert_eq!(
            result.diagnostics.stage_cost_attribution.len(),
            CANONICAL_STAGE_SEQUENCE.len()
        );
        assert!(result.diagnostics.hotspots.slowest_stage.is_some());
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
        assert!(result.executed_stages.is_empty());
        let world = result
            .diagnostics
            .stage_profiles
            .iter()
            .find(|profile| profile.stage == CanonicalStageId::World)
            .expect("world profile");
        assert!(matches!(
            world.state,
            CanonicalStageProfileState::Unavailable | CanonicalStageProfileState::Failed
        ));
        assert!(result.diagnostics.hotspots.unavailable_stage_count >= 1);
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
    fn invalid_input_is_structured_unavailable() {
        let backend = ComputePipelineBackend::stub();
        let invalid = ComputeInput {
            frame_id: FrameId(7),
            t: 0,
            context_digest: [9; 32],
        };
        let result = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: invalid,
                budget: ComputeBudget::default(),
            })
            .expect("canonical compute");
        assert_eq!(result.state, CanonicalPipelineState::Unavailable);
        assert_eq!(
            result.failure.expect("failure").kind,
            CanonicalFailureKind::InvalidInput
        );
        assert_eq!(result.validation.input, ValidationStatus::Degraded);
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
            gate: Default::default(),
            rollout: None,
        }];
        let failure = first_artifact_failure(&slots).expect("failure");
        assert_eq!(
            failure.kind,
            CanonicalFailureKind::ArtifactVerificationFailed
        );
        assert_eq!(failure.stage, Some(CanonicalStageId::Ssm));
    }

    #[test]
    fn world_slot_unavailable_maps_to_world_artifact_failure() {
        let slots = vec![ModelSlotProvenance {
            slot: ModelSlot::WorldJepa,
            stage: "world",
            required_for_pack: true,
            status: SlotRuntimeStatus::Unavailable,
            code: Some(ArtifactFailureCode::MissingPath),
            detail: Some("slot missing path".to_string()),
            resolved_path: None,
            hash_prefix: None,
            contract_version: Some("v1".to_string()),
            format: None,
            gate: Default::default(),
            rollout: None,
        }];
        let failure = first_artifact_failure(&slots).expect("failure");
        assert_eq!(failure.kind, CanonicalFailureKind::ArtifactUnavailable);
        assert_eq!(failure.stage, Some(CanonicalStageId::World));
        assert!(failure.detail.contains("world_jepa"));
    }

    #[test]
    fn artifact_incompatible_is_classified() {
        let slots = vec![ModelSlotProvenance {
            slot: ModelSlot::Lfm,
            stage: "lfm",
            required_for_pack: true,
            status: SlotRuntimeStatus::Incompatible,
            code: Some(ArtifactFailureCode::ArtifactIncompatible),
            detail: Some("slot incompatible with runtime".to_string()),
            resolved_path: None,
            hash_prefix: None,
            contract_version: Some("v9".to_string()),
            format: None,
            gate: Default::default(),
            rollout: None,
        }];
        let failure = first_artifact_failure(&slots).expect("failure");
        assert_eq!(failure.kind, CanonicalFailureKind::ArtifactIncompatible);
        assert_eq!(failure.stage, Some(CanonicalStageId::Lfm));
    }

    #[test]
    fn canonical_result_contains_world_stage_provenance() {
        let backend = ComputePipelineBackend::stub();
        let result = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget: ComputeBudget::default(),
            })
            .expect("canonical compute");
        assert!(result.world_stage.used);
        assert_eq!(result.world_stage.slot, Some(ModelSlot::WorldJepa));
        assert_eq!(result.world_stage.predictor, "mock_jepa_v0");
        assert!(matches!(
            result.world_stage.readiness,
            WorldStageReadiness::ContractReady | WorldStageReadiness::RuntimePathReady
        ));
        assert!(result.diagnostics.timing.total_micros > 0);
        assert_eq!(result.diagnostics.stage_profiles.len(), 4);
        assert_eq!(result.diagnostics.stage_cost_attribution.len(), 4);
        assert!(result.diagnostics.hotspots.dominant_stage.is_some());
        assert!(result
            .diagnostics
            .stage_cost_attribution
            .iter()
            .any(|entry| entry.dominant_timing || entry.dominant_work));
        assert_eq!(
            result.validation.violation_reason_mask,
            result.violation_reason_mask
        );
        assert!(result.diagnostics.evidence_chain_digest_prefix.is_some());
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
        assert!(canonical
            .diagnostics
            .stage_profiles
            .iter()
            .any(|profile| profile.state == CanonicalStageProfileState::Degraded));
        assert!(canonical
            .diagnostics
            .stage_cost_attribution
            .iter()
            .any(|entry| entry.pattern == CanonicalStageCostPattern::DegradedPathDriver));
        assert!(canonical.diagnostics.hotspots.degraded_stage.is_some());
    }

    #[test]
    fn execution_error_classifier_distinguishes_backend_disabled() {
        let disabled = classify_stage_execution_error(
            CanonicalStageId::Ssm,
            ComputeError::BackendDisabled,
            "ssm stage execution failed",
        );
        assert_eq!(disabled.kind, CanonicalFailureKind::BackendDisabled);

        let unavailable = classify_stage_execution_error(
            CanonicalStageId::World,
            ComputeError::Internal {
                reason: "candle_stage_unavailable:world".to_string(),
            },
            "world stage execution failed",
        );
        assert_eq!(unavailable.kind, CanonicalFailureKind::StageUnavailable);

        let execution = classify_stage_execution_error(
            CanonicalStageId::Ssm,
            ComputeError::Internal {
                reason: "boom".to_string(),
            },
            "ssm stage execution failed",
        );
        assert_eq!(execution.kind, CanonicalFailureKind::ExecutionError);
    }

    #[test]
    fn failure_kind_maps_to_fault_domains_and_isolation() {
        let artifact = classify_failure_kind(CanonicalFailureKind::ArtifactVerificationFailed);
        assert_eq!(artifact.domain, CanonicalFaultDomain::ArtifactModel);
        assert_eq!(
            artifact.isolation,
            CanonicalIsolationDisposition::HardEscalationJobFailure
        );
        assert!(!artifact.systemic);

        let degraded = classify_failure_kind(CanonicalFailureKind::DegradedFallback);
        assert_eq!(degraded.domain, CanonicalFaultDomain::Stage);
        assert_eq!(
            degraded.isolation,
            CanonicalIsolationDisposition::DegradedButServiceable
        );
        assert!(!degraded.systemic);

        let runtime = classify_failure_kind(CanonicalFailureKind::InvalidInput);
        assert_eq!(runtime.domain, CanonicalFaultDomain::RuntimeService);
        assert_eq!(
            runtime.isolation,
            CanonicalIsolationDisposition::ServiceRuntimeImpact
        );
        assert!(runtime.systemic);
    }

    #[test]
    fn degraded_path_exposes_fault_domain_notes() {
        let backend = ComputePipelineBackend::stub();
        let budget = ComputeBudget {
            sae_units: 100,
            global_work_units: 900,
            profile_id: 3,
            ..ComputeBudget::default()
        };
        let result = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget,
            })
            .expect("canonical compute");
        assert_eq!(result.state, CanonicalPipelineState::Degraded);
        assert!(result
            .signals
            .notes
            .iter()
            .any(|note| note == "fault_domain=stage"));
        assert!(result.signals.notes.iter().any(|note| {
            note == "fault_isolation=degradedbutserviceable"
                || note == "fault_isolation=hardescalationjobfailure"
        }));
        assert!(result
            .signals
            .notes
            .iter()
            .any(|note| note == "fault_systemic=false"));
    }

    #[test]
    fn nsr_required_verification_failed_maps_to_structured_failure_kind() {
        let slot = ModelSlotProvenance {
            slot: ModelSlot::EbmReasoner,
            stage: "ebm_reasoner",
            required_for_pack: false,
            status: SlotRuntimeStatus::VerificationFailed,
            code: Some(ArtifactFailureCode::HashMismatch),
            detail: Some("hash mismatch".to_string()),
            resolved_path: None,
            hash_prefix: None,
            contract_version: Some("v1".to_string()),
            format: None,
            gate: Default::default(),
            rollout: None,
        };
        let req = NsrRequest {
            base_risk: 0.3,
            base_confidence: 0.6,
            pressure: 0.2,
            surprise: 0.2,
            compute_degraded: false,
            contract_version: NsrContractVersion::V1,
        };
        let out = run_nsr_stage(NsrMode::Required, slot, &req, false);
        assert!(matches!(
            out.result,
            Err(NsrFailureKind::ArtifactVerificationFailed)
        ));
        assert!(out.required_failure);
        let failure = canonical_failure_for_nsr(
            out.result.expect_err("expected nsr failure"),
            "verification failed".to_string(),
        );
        assert_eq!(
            failure.kind,
            CanonicalFailureKind::NsrArtifactVerificationFailed
        );
    }

    #[test]
    fn nsr_contract_mismatch_is_explicit() {
        let slot = ModelSlotProvenance {
            slot: ModelSlot::EbmReasoner,
            stage: "ebm_reasoner",
            required_for_pack: false,
            status: SlotRuntimeStatus::Used,
            code: None,
            detail: None,
            resolved_path: Some("models/ebm_reasoner/model.safetensors".to_string()),
            hash_prefix: Some("abcdef".to_string()),
            contract_version: Some("v0".to_string()),
            format: None,
            gate: Default::default(),
            rollout: None,
        };
        let req = NsrRequest {
            base_risk: 0.3,
            base_confidence: 0.6,
            pressure: 0.2,
            surprise: 0.2,
            compute_degraded: false,
            contract_version: NsrContractVersion::V1,
        };
        let out = run_nsr_stage(NsrMode::BestEffort, slot, &req, false);
        assert!(matches!(out.result, Err(NsrFailureKind::ContractMismatch)));
        assert_eq!(out.status.state, NsrStageState::ContractMismatch);
    }

    #[test]
    fn canonical_pipeline_exposes_nsr_disabled_status_by_default() {
        let _guard = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        std::env::remove_var("UCF_NSR_MODE");

        let backend = ComputePipelineBackend::stub();
        let result = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget: ComputeBudget::default(),
            })
            .expect("canonical compute");
        assert_eq!(result.nsr_stage.state, NsrStageState::Disabled);
        assert!(!result.nsr_stage.used);
        assert_eq!(result.signals.nsr_digest, None);
    }

    #[cfg(feature = "compute-burn")]
    #[test]
    fn burn_pack_runs_honest_core_e2e_with_lfm_runtime_path() {
        let _guard = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models dir");

        let mut manifest = format!("allowlist_root = '{}'\n", models.display());
        for slot in ["world_jepa", "sae", "lfm", "ssm"] {
            let bytes = format!("{slot}-weights").into_bytes();
            let hash = hex::encode(Sha256::digest(&bytes));
            let model_path = models
                .join("promoted")
                .join(slot)
                .join(&hash)
                .join("model.safetensors");
            fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
            fs::write(&model_path, &bytes).expect("write");
            manifest.push_str(&format!(
                "[slots.{slot}]\nenabled = true\nexpected_sha256 = \"{hash}\"\nactive_hash = \"{hash}\"\nformat = \"burn\"\ncontract_version = \"v1\"\n"
            ));
        }
        let manifest_path = temp.path().join("manifest.toml");
        fs::write(&manifest_path, manifest).expect("manifest");
        std::env::set_var("UCF_MODEL_MANIFEST", &manifest_path);

        let pack = BackendPackFactory::build(BackendPackConfig {
            pack: BackendPackKind::BurnToyV1,
            seed: 7,
        })
        .expect("burn pack");
        let backend =
            ComputePipelineBackend::new(pack, FusionConfig::default(), LimitsConfig::default());
        let result = backend
            .compute_canonical(CanonicalPipelineRequest {
                input: input(),
                budget: ComputeBudget::default(),
            })
            .expect("compute");

        assert_eq!(
            result.executed_stages,
            vec![
                CanonicalStageId::World,
                CanonicalStageId::Sae,
                CanonicalStageId::Ssm,
                CanonicalStageId::Lfm
            ]
        );
        assert_eq!(result.stage_order, CANONICAL_STAGE_SEQUENCE);
        assert_eq!(
            result.route.world_backend,
            crate::BackendComponentId::BurnJepaV1 as u8
        );
        assert_eq!(
            result.route.sae_backend,
            crate::BackendComponentId::BurnSaeV1 as u8
        );
        assert_eq!(
            result.route.ssm_backend,
            crate::BackendComponentId::BurnSsmV1 as u8
        );
        assert_eq!(
            result.route.lfm_backend,
            crate::BackendComponentId::BurnLfmV1 as u8
        );
        assert_eq!(result.lfm_stage.state, LfmStageState::Used);
        assert_eq!(
            result.lfm_stage.readiness,
            LfmStageReadiness::RuntimePathReady
        );
        assert_eq!(result.nsr_stage.state, NsrStageState::Disabled);
        assert!(!result.nsr_stage.used);
        assert!(result.model_slots.iter().any(
            |slot| slot.slot == ModelSlot::WorldJepa && slot.status == SlotRuntimeStatus::Used
        ));
        let world_detail = result
            .model_slots
            .iter()
            .find(|slot| slot.slot == ModelSlot::WorldJepa)
            .and_then(|slot| slot.detail.as_deref())
            .expect("world slot detail");
        assert!(world_detail.contains("state=active"));
        assert!(world_detail.contains("selector=active_hash"));
        assert!(result
            .model_slots
            .iter()
            .any(|slot| slot.slot == ModelSlot::Sae && slot.status == SlotRuntimeStatus::Used));
        assert!(result
            .model_slots
            .iter()
            .any(|slot| slot.slot == ModelSlot::Ssm && slot.status == SlotRuntimeStatus::Used));
        assert!(result
            .model_slots
            .iter()
            .any(|slot| slot.slot == ModelSlot::Lfm && slot.status == SlotRuntimeStatus::Used));
        assert_eq!(result.lfm_stage.runtime, "burn_lfm_liquid_scalar_v1");
        assert!(!result
            .signals
            .notes
            .iter()
            .any(|note| note == "degraded:lfm_skipped_backend_disabled"));
    }
}
