use crate::coherence::{CoherenceRuntime, InterestProfile, Subscriber, TickInput};
use crate::compute_economics::{
    estimate_input_tokens, BudgetPool, ComputeEconomicsProfile, ComputeEconomy, ComputeStage,
    CostSchedule,
};
#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::ebm::EBM_FEATURE_D_MAX;
use crate::ebm::{
    active_ebm_constraints, candidate_feature_from_decision, configure_ebm_constraints,
    ebm_parity_within_lsb, CandleEbmReasonerV1, ConstraintParams, ConstraintTermId,
    ConstraintTermKind, ConstraintTermSpec, CpuEbmStubV0, EbmConstraintLibrary, EbmEnablementMode,
    EbmInput, EbmReasoner, EbmSignal, EbmSignals, EbmStatus, GpuMode,
};
use crate::errors::RuntimeError;
use crate::evolution::{
    DeltaOp, DeltaScore, EvolutionBudget, EvolutionContext, EvolutionEngine, LiquidWindowStats,
    MockEvolutionEngineV0, ReasonCode, SmallKey, StructuralDelta, TunableSnapshot,
};
use crate::hooks::{
    ComputeMilestone, ComputeMilestoneAggregator, ConsolidationHook, GeistHook, GeistRejectReason,
    GeistStateUpdater, LiquidContextWindow, LiquidTimelineIndex,
};
use crate::nsr_v1::{
    eval_input_from_candidate, NsrEngineV1, NsrOutput as NsrV1Output, NsrStatus as NsrV1Status,
};
use crate::sandbox_fs::SandboxFs;
use crate::tool_plugins::{
    compact_tool_result_note, run_plugin_tool, CapabilityTokenBinding, SandboxEnv,
    ToolPluginRegistry,
};
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;
#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
use ucf_backends_gpu::{detect_gpu_capability, GpuResourceCaps};

use ucf_biophys::v0::{
    apply_coherence_feedback, classify, compute_integration, couple_pair, hormone_step, hpa_step,
    modulate_hh, neuro_step, osc_step, phase_bin, phase_lock, ttfs_from_strength, ttfs_phase,
    verify_graph, BiophysModulation, CausalGraph, CoherenceState, Edge, EventBus, FieldEvent,
    FieldEventKind, FieldUpdateCfg, GatingModulation, HhParams, HormoneCfg, HormoneInput,
    HormoneState, HormoneStateSummary, HpaCfg, HpaState, IITCfg as BiophysIITCfg, IITInputs,
    IITState as BiophysIITState, Microcircuit, ModulationCfg, NeuroCfg, NeuroInput,
    NeuroSpikeBatch, NeuroStateSummary, NeuromodulatorField as BiophysField, NeuronPopState, Osc,
    OscId, PhaseCfg, RuleCfg, SnnSpikeEvent, SpikeCodecCfg, SpikeKind, VerifyVerdict,
};
use ucf_cde::v0::{
    on_intervention, on_observation, tick_decay, CdeCfg, CdeState, CdeUpdateKind, Intervention,
    Observation, VarId,
};
use ucf_compute::capabilities::{
    build_llm_backend, FinishReason, LlmBackendConfig, LlmInference, LlmOutputClass, LlmRequest,
    LlmResponse, LlmStatus,
};
use ucf_compute::enablement::{SlotEnablement, SlotMode};
use ucf_compute::{
    build_backend, compute_input_from_control, AiComputeBackend, BackendPackConfig,
    BackendPackFactory, BackendPackKind, ComputeBackendConfig, ComputeBudget, ComputeError,
    CpuStubBackend,
};
use ucf_core::archive_log::ArchiveLog;
use ucf_core::storage::{ArchiveCfg, FlushPolicy, MemArchiveStore};
use ucf_dbm::chemistry::{chemistry_step, ChemistryCfg, NeuromodState};
use ucf_dbm::regions::{region_step, BrainRegion, RegionKind};
#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
use ucf_ess::v1::GpuResourceViolationRecord;
use ucf_ess::v1::{
    compute_content_digest, AuditCheckpointRecord, AuditPayload, BackendPackRecord,
    CandidateSetRecord, CandidateSummaryRecord, CapabilityIssuanceRecord,
    ComputeBudgetViolationRecord, ComputeBudgetWindowRecord, DeltaEvaluationRecord,
    DeltaProposalRecord, DeltaRecommendationRecord, EbmConstraintProvenanceRecord,
    EbmEnvelopeViolationRecord, EbmReasoningRecord, EmergencyReasonCode, EmergencyRecord,
    EmergencyStateCode, ExperienceKind, ExperienceRecord, ExperienceStore, GpuParityRecord,
    GpuUnavailableRecord, HormoneRecord, IdAllocator, InMemoryEss, LfmSummaryRecord,
    LfmWindowRecord, NeuroRecord, NsrRecord, OutputRecord, PayloadClassification,
    PolicyProvenanceRecord, SandboxCallRecord, SandboxReplyRecord, ThrottleRecord, ToolAuthRecord,
    ToolExecutionRecord, ToolIssueAuditRecord, ToolPlanAuditRecord, ToolRequestRecord,
};
use ucf_fep::{
    check_coherence_invariants, fep_step, homeostasis_step, CoherenceCfg, CoherenceSnapshot,
    FepCfg, FepInputs, FepOutputs, HomeoCfg, HomeoState,
};
use ucf_frames::v1::{
    quantize_avg_v_mv, quantize_hormone, ArchiveAppendFrame, BiophysFrame, BiophysHhParams,
    CdeFrame, ChemFrame, CoherenceFrame, ControlFrame, DecisionFrame, DigitalBrainFrame, FepFrame,
    IitFrame, MicrocircuitFrame, NcdeFrame, NeuromodulatorSnapshot, NsrFrame, OnnFrame, PhaseFrame,
    SleFrame, SnnFrame, SsmFrame, TcfFrame,
};
use ucf_iit_proxy::v0::{iit_push_and_eval, IitCfg, IitSample, IitState};
use ucf_ncde::v0::{ncde_step, NcdeCfg, NcdeInput, NcdeState};
use ucf_neuromod::v0::{NeuromodInputs, NeuromodScheduler, NeuromodulatorField};
use ucf_nsr::v0::{Claim, NsrEngine, Verdict};
use ucf_nsr::{
    ActionType, CapabilityKind, DecisionIntentSummary, NsrBudget, NsrContext, NsrDatalogLiteEngine,
    NsrPolicyEcologyEngine, OutputClass, PolicyHint, PolicyTag,
};
use ucf_onn::v0::{onn_step, OnnCfg, OnnCore, OnnInput, OnnNode, OnnState, PhaseDeg};
use ucf_policy::{
    adapter::ActionAdapter,
    candidate::{
        assess_candidate, select_candidate, CandidateGenerator, CandidatePolicyHint,
        DecisionBudget, DecisionContext, DefaultCandidateGeneratorV0,
        OutputClass as CandidateOutputClass,
    },
    ebm_constraints::load_ebm_constraints,
    gem::{
        issue_capabilities_governed, request_from, request_from_intent, AuthorizationOutcome,
        CapabilityIssuanceDecision, Gem, GovernanceSignals, PayloadHint as GemPayloadHint,
        ToolGate, ToolGovernor, ToolStatus,
    },
    pbm::Pbm,
    policy_bundle::verify_policy_bundle,
    policy_packs::load_and_merge_policy_graph,
    rate_limiter::RateLimiter,
    tool_plans::{build_plan, issue_for_plan, make_plan_record},
};
use ucf_sle::v0::{sle_step, SleCfg, SleReason, SleSignals, SleState};
use ucf_snn::v0::{encode, snn_emit, to_brainbus, FeatureEvent, SnnCfg, SnnEncodeCfg, SpikeSrc};
use ucf_spikes::{
    encode_ttfs_us, filter_phase_locked, PhaseLockCfg, Spike as BusSpike, SpikeBus,
    SpikeKind as BusSpikeKind,
};
use ucf_ssm::v0::{ssm_step, SsmCfg, SsmState};
use ucf_tcf::v0::{tcf_tick, TcfCfg, TcfState};

const OSC_SSM: u8 = 1;
const OSC_CDE: u8 = 2;
const OSC_NSR: u8 = 3;
const OSC_COHERENCE: u8 = 4;
const OSC_NSR_TCF_ENFORCE: u8 = 1;
const MOD_PBM: ucf_onn::v0::OscId = 13;

struct NotImplementedBackend {
    name: &'static str,
}

impl AiComputeBackend for NotImplementedBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn compute(
        &self,
        _input: &ucf_compute::ComputeInput,
        _budget: ComputeBudget,
    ) -> Result<ucf_compute::ComputeSignals, ComputeError> {
        Err(ComputeError::NotImplemented)
    }
}

fn canonical_backend_label(backend_name: &str) -> &'static str {
    match backend_name {
        "stub" | "cpu_stub" | "stub_v0" | "toy_v1" => "stub",
        name if name.starts_with("candle") => "candle",
        name if name.starts_with("burn") => "burn",
        name if name.starts_with("worker") => "worker",
        _ => "unknown",
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn fail_if_training_mode_enabled() {
    if env_flag("UCF_EBM_TRAINING_MODE") {
        panic!("runtime forbids training mode; use offline ucf-ebm-train operator workflow");
    }
}

fn map_constraint_kind(kind: &str) -> Option<ConstraintTermKind> {
    match kind {
        "ToolIntentPenalty" => Some(ConstraintTermKind::ToolIntentPenalty),
        "CapabilityForbidden" => Some(ConstraintTermKind::CapabilityForbidden),
        "CapabilityHighRisk" => Some(ConstraintTermKind::CapabilityHighRisk),
        "ContextRiskAmplifier" => Some(ConstraintTermKind::ContextRiskAmplifier),
        "EmergencyDenyAllBias" => Some(ConstraintTermKind::EmergencyDenyAllBias),
        "OutputClassMismatch" => Some(ConstraintTermKind::OutputClassMismatch),
        "BudgetExhaustedBias" => Some(ConstraintTermKind::BudgetExhaustedBias),
        "NsrRiskAmplifier" => Some(ConstraintTermKind::NsrRiskAmplifier),
        _ => None,
    }
}

fn map_candidate_kind(kind: Option<&str>) -> Option<crate::ebm::CandidateKind> {
    match kind {
        Some("SafeText") => Some(crate::ebm::CandidateKind::SafeText),
        Some("Json") => Some(crate::ebm::CandidateKind::Json),
        Some("ToolIntent") => Some(crate::ebm::CandidateKind::ToolIntent),
        Some("NoOp") => Some(crate::ebm::CandidateKind::NoOp),
        Some("Other") => Some(crate::ebm::CandidateKind::Other),
        _ => None,
    }
}

fn load_runtime_ebm_constraints(policy_hash: &str) -> EbmConstraintLibrary {
    let loaded = load_ebm_constraints(std::path::Path::new("policies"));
    let library = loaded
        .ok()
        .map(|bundle| {
            let mut terms = bundle
                .terms
                .iter()
                .filter_map(|term| {
                    let kind = map_constraint_kind(&term.kind)?;
                    Some(ConstraintTermSpec {
                        id: ConstraintTermId(term.id),
                        kind,
                        weight_q: ucf_types::UQ0_16::from_raw(term.weight_q),
                        params: ConstraintParams {
                            capability_class_id: term.capability_class_id,
                            threshold_q: term.threshold_q.map(ucf_types::UQ0_16::from_raw),
                            candidate_kind: map_candidate_kind(term.candidate_kind.as_deref()),
                            governor_tier_min: term.governor_tier_min,
                        },
                    })
                })
                .collect::<Vec<_>>();
            terms.sort_by(|a, b| a.id.0.cmp(&b.id.0));
            EbmConstraintLibrary {
                schema_version: bundle.schema_version,
                terms,
                fallback_used: false,
                constraints_digest: bundle.constraints_digest,
            }
        })
        .unwrap_or_else(EbmConstraintLibrary::fallback);
    if library.fallback_used {
        metrics::counter!("ucf_ebm_constraint_fallback_total").increment(1);
    }
    metrics::gauge!("ucf_ebm_constraint_terms_loaded").set(library.terms.len() as f64);
    tracing::info!(
        terms = library.terms.len(),
        fallback = library.fallback_used,
        policy_hash = %policy_hash,
        "ebm constraints configured"
    );
    library
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn quantize_unit_u16(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 10_000.0).round() as u16
}

fn prefix8(digest: [u8; 32]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

fn prefix8_hex(hex: &str) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (i, chunk) in hex.as_bytes().chunks(2).take(8).enumerate() {
        if chunk.len() == 2 {
            let hi = (chunk[0] as char).to_digit(16).unwrap_or(0) as u8;
            let lo = (chunk[1] as char).to_digit(16).unwrap_or(0) as u8;
            out[i] = (hi << 4) | lo;
        }
    }
    out
}

const OUTPUT_SCHEMA_VERSION: u16 = 2;
const MAX_OUTPUT_TEXT_CHARS: usize = 4096;
const MAX_LLM_PROMPT_BYTES: usize = 8 * 1024;
const MIN_UNCERTAINTY_TOKENS: u32 = 64;
const UNCERTAINTY_MAXTOKENS_FACTOR: f32 = 0.6;
const UNCERTAINTY_HIGH_THRESHOLD: f32 = 0.75;
const STABILITY_LOW_THRESHOLD: f32 = 0.35;
const MAX_OVERRIDE_REASONS: usize = 8;
const EMERGENCY_MAX_TOKENS: u32 = 64;
const EMERGENCY_COOLDOWN_TICKS: u16 = 32;
const EMERGENCY_ARM_TREND_TICKS: u8 = 2;
const EMERGENCY_DV_TREND_K: u8 = 3;
const LFM_V_K: f32 = 0.25;
const LFM_V_MAX: f32 = 1.10;
const LFM_DV_MAX: f32 = 0.06;
const LFM_SATURATION_MAX: f32 = 0.20;
const IIT_INCOHERENCE_GOV_INC_HIGH_Q: u16 = 39_321; // ~0.6
const IIT_INCOHERENCE_GOV_EXTRA_PENALTY_MAX_Q: u16 = 16_384; // ~0.25

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputOverrideCode {
    ForcedSafeOnly,
    ForcedShort,
}

impl OutputOverrideCode {
    fn as_u8(self) -> u8 {
        match self {
            Self::ForcedSafeOnly => 1,
            Self::ForcedShort => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OverrideReasonCode {
    NsrSafeOnly,
    NsrBlock,
    HighUncertainty,
    LowStability,
}

impl OverrideReasonCode {
    fn as_u16(self) -> u16 {
        match self {
            Self::NsrSafeOnly => 1,
            Self::NsrBlock => 2,
            Self::HighUncertainty => 3,
            Self::LowStability => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmergencyTriggerReason {
    RunawayV,
    TrendDV,
    Saturation,
    NanInf,
}

impl EmergencyTriggerReason {
    fn as_ess(self) -> EmergencyReasonCode {
        match self {
            Self::RunawayV => EmergencyReasonCode::RunawayV,
            Self::TrendDV => EmergencyReasonCode::TrendDV,
            Self::Saturation => EmergencyReasonCode::Saturation,
            Self::NanInf => EmergencyReasonCode::NanInf,
        }
    }

    fn as_label(self) -> &'static str {
        match self {
            Self::RunawayV => "runaway_v",
            Self::TrendDV => "trend_dv",
            Self::Saturation => "saturation",
            Self::NanInf => "nan_inf",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StabilitySnapshot {
    v: f32,
    dv: f32,
    state_norm: f32,
    deriv_norm: f32,
    saturation_ratio: f32,
    nan_inf_detected: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum EmergencyState {
    Inactive,
    Armed {
        since_t: u64,
        reason: EmergencyTriggerReason,
    },
    Active {
        since_t: u64,
        reason: EmergencyTriggerReason,
        cool_down_remaining: u16,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelGovernanceStatus {
    Healthy,
    Degraded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GovernanceSlot {
    Llm,
    World,
    Sae,
    Ssm,
    Lfm,
    Ebm,
}

impl GovernanceSlot {
    fn as_str(self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::World => "world",
            Self::Sae => "sae",
            Self::Ssm => "ssm",
            Self::Lfm => "lfm",
            Self::Ebm => "ebm",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ModelGovernancePolicy {
    max_timeout_rate_q: u16,
    max_invalid_rate_q: u16,
    max_envelope_violation_rate_q: u16,
    max_delta_uncertainty_q: u16,
    max_delta_pressure_q: u16,
}

impl Default for ModelGovernancePolicy {
    fn default() -> Self {
        Self {
            max_timeout_rate_q: 3276,
            max_invalid_rate_q: 655,
            max_envelope_violation_rate_q: 655,
            max_delta_uncertainty_q: 9830,
            max_delta_pressure_q: 9830,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SlotWindowAcc {
    samples: u64,
    timeout: u64,
    invalid: u64,
    envelope: u64,
    uncertainty_sum_q: u64,
    pressure_sum_q: u64,
}

#[derive(Debug)]
struct ModelGovernanceRuntime {
    window_ticks: u64,
    persistent_windows: u8,
    current_window_start: u64,
    status: BTreeMap<GovernanceSlot, ModelGovernanceStatus>,
    breaches_streak: BTreeMap<GovernanceSlot, u8>,
    thresholds: BTreeMap<GovernanceSlot, ModelGovernancePolicy>,
    baseline_uncertainty_q: u16,
    baseline_pressure_q: u16,
    accum: BTreeMap<GovernanceSlot, SlotWindowAcc>,
}

impl ModelGovernanceRuntime {
    fn from_policy(window_ticks: u64, thresholds: &BTreeMap<String, i64>) -> Self {
        let mut out = Self::new(window_ticks, 0, 0);
        for slot in [
            GovernanceSlot::Llm,
            GovernanceSlot::World,
            GovernanceSlot::Sae,
            GovernanceSlot::Ssm,
            GovernanceSlot::Lfm,
            GovernanceSlot::Ebm,
        ] {
            let prefix = format!("model_governance_{}", slot.as_str());
            let mut p = ModelGovernancePolicy::default();
            p.max_timeout_rate_q = threshold_u16(
                thresholds,
                &format!("{prefix}_max_timeout_rate_q"),
                p.max_timeout_rate_q,
            );
            p.max_invalid_rate_q = threshold_u16(
                thresholds,
                &format!("{prefix}_max_invalid_rate_q"),
                p.max_invalid_rate_q,
            );
            p.max_envelope_violation_rate_q = threshold_u16(
                thresholds,
                &format!("{prefix}_max_envelope_violation_rate_q"),
                p.max_envelope_violation_rate_q,
            );
            p.max_delta_uncertainty_q = threshold_u16(
                thresholds,
                &format!("{prefix}_max_delta_uncertainty_q"),
                p.max_delta_uncertainty_q,
            );
            p.max_delta_pressure_q = threshold_u16(
                thresholds,
                &format!("{prefix}_max_delta_pressure_q"),
                p.max_delta_pressure_q,
            );
            out.thresholds.insert(slot, p);
        }
        out
    }

    fn new(window_ticks: u64, baseline_uncertainty_q: u16, baseline_pressure_q: u16) -> Self {
        let mut status = BTreeMap::new();
        let mut breaches_streak = BTreeMap::new();
        let mut thresholds = BTreeMap::new();
        let mut accum = BTreeMap::new();
        for slot in [
            GovernanceSlot::Llm,
            GovernanceSlot::World,
            GovernanceSlot::Sae,
            GovernanceSlot::Ssm,
            GovernanceSlot::Lfm,
            GovernanceSlot::Ebm,
        ] {
            status.insert(slot, ModelGovernanceStatus::Healthy);
            breaches_streak.insert(slot, 0);
            thresholds.insert(slot, ModelGovernancePolicy::default());
            accum.insert(slot, SlotWindowAcc::default());
        }
        Self {
            window_ticks: window_ticks.max(64),
            persistent_windows: 2,
            current_window_start: 0,
            status,
            breaches_streak,
            thresholds,
            baseline_uncertainty_q,
            baseline_pressure_q,
            accum,
        }
    }

    fn any_critical_degraded(&self) -> bool {
        self.status.get(&GovernanceSlot::Llm) == Some(&ModelGovernanceStatus::Degraded)
            || self.status.get(&GovernanceSlot::Ebm) == Some(&ModelGovernanceStatus::Degraded)
    }
}

fn threshold_u16(thresholds: &BTreeMap<String, i64>, key: &str, fallback: u16) -> u16 {
    thresholds
        .get(key)
        .copied()
        .and_then(|v| u16::try_from(v).ok())
        .unwrap_or(fallback)
}

fn model_governance_digest(status: &BTreeMap<GovernanceSlot, ModelGovernanceStatus>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.runtime.model_governance.status.v1");
    for slot in [
        GovernanceSlot::Llm,
        GovernanceSlot::World,
        GovernanceSlot::Sae,
        GovernanceSlot::Ssm,
        GovernanceSlot::Lfm,
        GovernanceSlot::Ebm,
    ] {
        hasher.update(slot.as_str().as_bytes());
        let degraded = status.get(&slot) == Some(&ModelGovernanceStatus::Degraded);
        hasher.update(&[u8::from(degraded)]);
    }
    *hasher.finalize().as_bytes()
}

fn model_governance_reason_codes(
    status: &BTreeMap<GovernanceSlot, ModelGovernanceStatus>,
) -> Vec<String> {
    let mut out = Vec::new();
    for slot in [
        GovernanceSlot::Llm,
        GovernanceSlot::World,
        GovernanceSlot::Sae,
        GovernanceSlot::Ssm,
        GovernanceSlot::Lfm,
        GovernanceSlot::Ebm,
    ] {
        if status.get(&slot) == Some(&ModelGovernanceStatus::Degraded) {
            out.push(format!("slot_{}_degraded", slot.as_str()));
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct DecodingPolicy {
    max_tokens_eff: u32,
    output_class: CandidateOutputClass,
    output_override: Option<OutputOverrideCode>,
    override_reasons: [Option<OverrideReasonCode>; MAX_OVERRIDE_REASONS],
}

impl DecodingPolicy {
    fn reason_codes(&self) -> Vec<u16> {
        self.override_reasons
            .iter()
            .flatten()
            .map(|reason| reason.as_u16())
            .take(MAX_OVERRIDE_REASONS)
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct PromptConditioning {
    risk: Option<f32>,
    confidence: Option<f32>,
    surprise: f32,
    pressure: f32,
    uncertainty: Option<f32>,
    coherence: Option<f32>,
    instability: Option<f32>,
    evidence_chain_digest: [u8; 32],
    lfm_readout_digest: Option<[u8; 32]>,
}

fn output_status_code(status: LlmStatus) -> u8 {
    match status {
        LlmStatus::Ok => 0,
        LlmStatus::Truncated => 1,
        LlmStatus::Refused => 2,
        LlmStatus::Failed => 3,
    }
}

fn finish_reason_code(reason: FinishReason) -> u8 {
    match reason {
        FinishReason::Stop => 0,
        FinishReason::Length => 1,
        FinishReason::PolicyRefusal => 2,
        FinishReason::Error => 3,
    }
}

fn bounded_summary_line(input: &str) -> String {
    input.chars().take(160).collect()
}

fn map_output_class(output_class: CandidateOutputClass) -> LlmOutputClass {
    match output_class {
        CandidateOutputClass::SafeText => LlmOutputClass::SafeText,
        CandidateOutputClass::Code => LlmOutputClass::Code,
        CandidateOutputClass::ExternalIo => LlmOutputClass::ExternalIo,
        CandidateOutputClass::ExecIntent => LlmOutputClass::ExecIntent,
        CandidateOutputClass::Sensitive => LlmOutputClass::Sensitive,
    }
}

fn validate_output(output_class: CandidateOutputClass, text: &str) -> Result<(), ()> {
    match output_class {
        CandidateOutputClass::SafeText => {
            if text.contains("```") {
                return Err(());
            }
            Ok(())
        }
        CandidateOutputClass::Code => Ok(()),
        CandidateOutputClass::ExternalIo | CandidateOutputClass::ExecIntent => {
            if text.is_empty() {
                Ok(())
            } else {
                Err(())
            }
        }
        CandidateOutputClass::Sensitive => Err(()),
    }
}

fn fmt_signal(value: Option<f32>) -> String {
    match value {
        Some(v) => format!("{v:.3}"),
        None => "na".to_string(),
    }
}

fn digest_prefix(digest: [u8; 32]) -> String {
    hex::encode(digest)[..12].to_string()
}

fn backend_pack_digest(summary: ucf_compute::ComputeSignalsSummary) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&summary.backend_pack_id.to_be_bytes());
    hasher.update(&summary.fixtures_digest);
    hasher.update(summary.backend_profile.as_bytes());
    *hasher.finalize().as_bytes()
}

fn liquid_window_from_index(
    window: Option<LiquidContextWindow>,
) -> Option<ucf_policy::candidate::LiquidContextWindowSummary> {
    window.map(|w| ucf_policy::candidate::LiquidContextWindowSummary {
        sample_count: w.sample_count,
        mean_uncertainty: w.mean_uncertainty,
        max_uncertainty: w.max_uncertainty,
        mean_stability: w.mean_stability,
        rolling_digest: w.rolling_digest,
    })
}

fn output_class_instruction(output_class: CandidateOutputClass) -> &'static str {
    match output_class {
        CandidateOutputClass::SafeText => {
            "Output format: plain safe text, no code blocks, no tool instructions."
        }
        CandidateOutputClass::Code => {
            "Output format: concise deterministic code/text response, no tool execution directives."
        }
        CandidateOutputClass::ExternalIo | CandidateOutputClass::ExecIntent => {
            "Output format: refusal-safe placeholder only."
        }
        CandidateOutputClass::Sensitive => "Output format: refusal-safe text only.",
    }
}

fn build_prompt(
    control_frame_summary: &str,
    decision_summary: &str,
    signals: PromptConditioning,
    liquid_context: Option<LiquidContextWindow>,
    output_class: CandidateOutputClass,
) -> String {
    let mut prompt = format!(
        concat!(
            "System constraints: deterministic response only; no tools; bounded safe output.\n",
            "Context summary:\n",
            "- {}\n",
            "- {}\n",
            "- output_class={:?}\n",
            "Signals: risk={} confidence={} surprise={:.3} pressure={:.3} uncertainty={} coherence={} instability={}\n",
            "Digests: evidence_chain={} lfm_readout={} liquid_rolling={}\n",
            "Liquid scalars: mean_u={} max_u={} mean_stability={} samples={}\n",
            "{}\n",
            "Do: obey policy constraints, be concise, be auditable.\n",
            "Don't: reveal hidden state vectors, emit sensitive content, invoke tools."
        ),
        control_frame_summary,
        decision_summary,
        output_class,
        fmt_signal(signals.risk),
        fmt_signal(signals.confidence),
        signals.surprise,
        signals.pressure,
        fmt_signal(signals.uncertainty),
        fmt_signal(signals.coherence),
        fmt_signal(signals.instability),
        digest_prefix(signals.evidence_chain_digest),
        signals
            .lfm_readout_digest
            .map(digest_prefix)
            .unwrap_or_else(|| "na".to_string()),
        liquid_context
            .map(|w| digest_prefix(w.rolling_digest))
            .unwrap_or_else(|| "na".to_string()),
        liquid_context
            .map(|w| format!("{:.3}", w.mean_uncertainty))
            .unwrap_or_else(|| "na".to_string()),
        liquid_context
            .map(|w| format!("{:.3}", w.max_uncertainty))
            .unwrap_or_else(|| "na".to_string()),
        liquid_context
            .map(|w| format!("{:.3}", w.mean_stability))
            .unwrap_or_else(|| "na".to_string()),
        liquid_context
            .map(|w| w.sample_count.to_string())
            .unwrap_or_else(|| "0".to_string()),
        output_class_instruction(output_class),
    );

    if prompt.len() > MAX_LLM_PROMPT_BYTES {
        prompt.truncate(MAX_LLM_PROMPT_BYTES);
    }
    prompt
}

fn compute_max_tokens_eff(base: u32, uncertainty: Option<f32>) -> u32 {
    let u = uncertainty.unwrap_or(0.0).clamp(0.0, 1.0);
    let scaled = (base as f32 * (1.0 - UNCERTAINTY_MAXTOKENS_FACTOR * u)).round() as u32;
    scaled.clamp(MIN_UNCERTAINTY_TOKENS.min(base), base)
}

fn apply_decoding_policy(
    base_max_tokens: u32,
    selected_output_class: CandidateOutputClass,
    nsr_hint: PolicyHint,
    lfm_uncertainty: Option<f32>,
    lfm_stability: Option<f32>,
    hormone_stress_gate: Option<f32>,
    emergency_active: bool,
) -> DecodingPolicy {
    let mut override_reasons = [None; MAX_OVERRIDE_REASONS];
    let mut reason_count = 0usize;
    let mut push_reason = |reason: OverrideReasonCode| {
        if reason_count < MAX_OVERRIDE_REASONS {
            override_reasons[reason_count] = Some(reason);
            reason_count += 1;
        }
    };

    let mut output_class = selected_output_class;
    let mut output_override = None;

    if emergency_active {
        output_class = CandidateOutputClass::SafeText;
        output_override = Some(OutputOverrideCode::ForcedSafeOnly);
        return DecodingPolicy {
            max_tokens_eff: EMERGENCY_MAX_TOKENS.min(base_max_tokens),
            output_class,
            output_override,
            override_reasons,
        };
    }

    if matches!(nsr_hint, PolicyHint::Block) {
        output_class = CandidateOutputClass::SafeText;
        output_override = Some(OutputOverrideCode::ForcedSafeOnly);
        push_reason(OverrideReasonCode::NsrBlock);
    } else if matches!(nsr_hint, PolicyHint::SafeOnly) {
        output_class = CandidateOutputClass::SafeText;
        output_override = Some(OutputOverrideCode::ForcedSafeOnly);
        push_reason(OverrideReasonCode::NsrSafeOnly);
    }

    let uncertainty_high = lfm_uncertainty
        .map(|u| u.clamp(0.0, 1.0) > UNCERTAINTY_HIGH_THRESHOLD)
        .unwrap_or(false);
    let stability_low = lfm_stability
        .map(|s| s.clamp(0.0, 1.0) < STABILITY_LOW_THRESHOLD)
        .unwrap_or(false);

    if uncertainty_high {
        push_reason(OverrideReasonCode::HighUncertainty);
    }
    if stability_low {
        push_reason(OverrideReasonCode::LowStability);
    }

    if (uncertainty_high || stability_low)
        && !matches!(output_override, Some(OutputOverrideCode::ForcedSafeOnly))
    {
        output_override = Some(OutputOverrideCode::ForcedShort);
    }

    let max_tokens_eff = {
        let base = compute_max_tokens_eff(base_max_tokens, lfm_uncertainty);
        let stress_gate = hormone_stress_gate.unwrap_or(1.0).clamp(0.1, 1.0);
        let tightened = (base as f32 * stress_gate).round() as u32;
        tightened.clamp(MIN_UNCERTAINTY_TOKENS.min(base_max_tokens), base_max_tokens)
    };
    DecodingPolicy {
        max_tokens_eff,
        output_class,
        output_override,
        override_reasons,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PhaseBus {
    osc_jepa: Osc,
    osc_nsr: Osc,
    osc_microcircuit: Osc,
}

#[derive(Clone, Debug, Default)]
struct WorkingContext {
    ssm_y: Vec<f32>,
}

#[derive(Clone, Debug)]
struct DigitalBrainState {
    amygdala: BrainRegion,
    pfc: BrainRegion,
    chem: NeuromodState,
    chem_cfg: ChemistryCfg,
}

#[derive(Clone, Copy, Debug, Default)]
struct EmotionVector {
    valence: f32,
    arousal: f32,
}

#[derive(Clone, Debug)]
struct FepState {
    cfg: FepCfg,
    homeo_cfg: HomeoCfg,
    homeo_state: HomeoState,
    coh_cfg: CoherenceCfg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyDecisionKind {
    InternalRecursion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyDecision {
    pub kind: PolicyDecisionKind,
    pub allow: bool,
    pub max_depth: u8,
}

#[derive(Clone, Debug, PartialEq)]
struct MetaInput {
    tokens: Vec<u32>,
    weight: f32,
    depth: u8,
    reason: SleReason,
}

#[derive(Clone, Copy, Debug, Default)]
struct EvolutionTickSummary {
    risk: f32,
    confidence: f32,
    coherence: f32,
    instability: f32,
    pressure: f32,
    surprise: f32,
    uncertainty: f32,
    ebm_energy_mean_q: u16,
    governor_tier: u8,
    budget_exceeded: bool,
    denied_tool: bool,
    evidence_chain_digest: [u8; 32],
    backend_pack_digest: [u8; 32],
    nsr_available: bool,
}

pub struct RuntimeOrchestrator {
    pub ess: InMemoryEss,
    pub ids: IdAllocator,
    pub pbm: Pbm,
    pub gem: Gem,
    compute_backend: Box<dyn AiComputeBackend>,
    compute_budget: ComputeBudget,
    llm_backend: std::sync::Arc<dyn LlmInference + Send + Sync>,
    llm_cfg: LlmBackendConfig,
    neuromod_field: NeuromodulatorField,
    neuromod_scheduler: NeuromodScheduler,
    onn: OnnCore,
    snn_encode_cfg: SnnEncodeCfg,
    snn_cfg: SnnCfg,
    snn_tick_counter: u64,
    last_snn_spike_count: usize,
    last_spike_rate_0_1: f32,
    biophys_field: BiophysField,
    biophys_cfg: FieldUpdateCfg,
    biophys_mod_cfg: ModulationCfg,
    hpa_state: HpaState,
    hpa_cfg: HpaCfg,
    last_biophys_tick_ms: Option<u64>,
    last_biophys_frame: Option<BiophysFrame>,
    microcircuit: Microcircuit,
    last_microcircuit_frame: Option<MicrocircuitFrame>,
    phase_bus: PhaseBus,
    phase_cfg: PhaseCfg,
    spike_codec_cfg: SpikeCodecCfg,
    last_phase_frame: Option<PhaseFrame>,
    onn_state: OnnState,
    onn_cfg: OnnCfg,
    event_bus: EventBus,
    spike_bus: SpikeBus,
    spike_seq: u32,
    last_onn_frame: Option<OnnFrame>,
    tcf_cfg: TcfCfg,
    tcf_state: TcfState,
    last_tcf_frame: Option<TcfFrame>,
    forced_mean_lock_for_test: Option<f32>,
    forced_nsr_risk_for_test: Option<f32>,
    last_snn_frame: Option<SnnFrame>,
    last_spike_frames: Vec<ucf_frames::v1::SpikeFrame>,
    attention_event: bool,
    iit_cfg: BiophysIITCfg,
    iit_state: BiophysIITState,
    iit_proxy_cfg: IitCfg,
    iit_proxy_state: IitState,
    last_iit_frame: Option<IitFrame>,
    cde_graph: CausalGraph,
    cde_cfg: CdeCfg,
    cde_state: CdeState,
    nsr_cfg: RuleCfg,
    nsr_engine: NsrEngine,
    nsr_v1_engine: NsrEngineV1,
    last_cde_frame: Option<CdeFrame>,
    last_nsr_frame: Option<NsrFrame>,
    ssm_state: SsmState,
    ssm_cfg: SsmCfg,
    working_context: WorkingContext,
    last_ssm_frame: Option<SsmFrame>,
    sle_cfg: SleCfg,
    sle_state: SleState,
    last_sle_frame: Option<SleFrame>,
    last_meta_input: Option<MetaInput>,
    policy_internal_recursion_allow: bool,
    policy_internal_recursion_max_depth: u8,
    ssm_gate: f32,
    ncde_cfg: NcdeCfg,
    ncde_state: NcdeState,
    last_ncde_frame: Option<NcdeFrame>,
    last_ncde_spike_u4: f32,
    digital_brain: DigitalBrainState,
    last_chem_frame: Option<ChemFrame>,
    last_digital_brain_frame: Option<DigitalBrainFrame>,
    emotion: EmotionVector,
    archive: ArchiveLog<MemArchiveStore>,
    last_archive_append_frame: Option<ArchiveAppendFrame>,
    last_archive_payload_len: usize,
    fep_state: FepState,
    hormone_state: HormoneState,
    hormone_cfg: HormoneCfg,
    hormone_persist_every: u64,
    lfm_persist_every: u64,
    lfm_window_emit_every: u64,
    liquid_timeline_index: LiquidTimelineIndex,
    last_hormone_summary: Option<HormoneStateSummary>,
    last_gating_modulation: Option<GatingModulation>,
    hormone_degraded_total: u64,
    neuro_state: NeuronPopState,
    neuro_cfg: NeuroCfg,
    neuro_persist_every: u64,
    last_neuro_summary: Option<NeuroStateSummary>,
    last_neuro_modulation: Option<BiophysModulation>,
    last_neuro_spikes: Option<NeuroSpikeBatch>,
    neuro_degraded_total: u64,
    last_neuro_degraded: bool,
    forced_surprise_for_test: Option<f32>,
    forced_geist_drift_for_test: Option<f32>,
    forced_ess_pressure_for_test: Option<f32>,
    last_fep_frame: Option<FepFrame>,
    last_coherence_frame: Option<CoherenceFrame>,
    coherence_violation_count: u64,
    coherence_violation_flag: bool,
    last_fep_outputs: Option<FepOutputs>,
    risk_quality_counts: [u64; 3],
    backpressure: f32,
    compute_budget_exceeded_total: u64,
    backpressure_active_total: u64,
    fep_risk_penalty_applied_total: u64,
    coherence_gated_total: u64,
    coherence_runtime: CoherenceRuntime,
    consolidation_hook_enabled: bool,
    geist_hook_enabled: bool,
    consolidation_hook_errors_total: u64,
    geist_hook_errors_total: u64,
    consolidation_milestones_emitted_total: u64,
    geist_updates_accepted_total: u64,
    geist_updates_rejected_total: u64,
    geist_updates_rejected_not_enough_samples_total: u64,
    geist_updates_rejected_unstable_total: u64,
    geist_updates_rejected_degraded_total: u64,
    geist_updates_rejected_drift_total: u64,
    compute_milestone_aggregator: ComputeMilestoneAggregator,
    geist_state_updater: GeistStateUpdater,
    last_compute_milestone: Option<ComputeMilestone>,
    tool_gate: ToolGate,
    tool_governor: ToolGovernor,
    slot_enablement: SlotEnablement,
    model_governance: ModelGovernanceRuntime,
    spent_tool_tokens: HashSet<[u8; 32]>,
    last_nsr_risk: Option<f32>,
    last_nsr_penalty_q: u16,
    emergency_state: EmergencyState,
    emergency_last_v: Option<f32>,
    emergency_dv_trend_count: u8,
    emergency_last_reason: Option<EmergencyTriggerReason>,
    candidate_generator: DefaultCandidateGeneratorV0,
    ebm_reasoner: Box<dyn EbmReasoner>,
    ebm_mode: EbmEnablementMode,
    gpu_mode: GpuMode,
    gpu_parity_fail_streak: u8,
    gpu_disabled: bool,
    ebm_run_id: u64,
    audit_head_digest: [u8; 32],
    audit_chain_checkpoint_total: u64,
    evolution_enabled: bool,
    evolution_window_ticks: u64,
    evolution_summaries: Vec<EvolutionTickSummary>,
    evolution_engine: Box<dyn EvolutionEngine>,
    evolution_proposals_total: u64,
    evolution_accepted_total: u64,
    evolution_rejected_total: u64,
    evolution_budget_exceeded_total: u64,
    evolution_suppressed_total: u64,
    evolution_recommended_total: u64,
    policy_bundle_hash: String,
    policy_graph_digest: String,
    policy_graph_digest_prefix: [u8; 8],
    compute_economy: ComputeEconomy,
    compute_budget_window_t0: u64,
    compute_budget_window_t: u64,
    compute_tier_sum: u64,
    compute_tier_count: u64,
    compute_tier_max: u8,
}

impl RuntimeOrchestrator {
    fn governance_slot_mode(&self, slot: GovernanceSlot) -> SlotMode {
        match slot {
            GovernanceSlot::Llm => self.slot_enablement.llm,
            GovernanceSlot::World => self.slot_enablement.world_jepa,
            GovernanceSlot::Sae => self.slot_enablement.sae,
            GovernanceSlot::Ssm => self.slot_enablement.ssm,
            GovernanceSlot::Lfm => self.slot_enablement.lfm,
            GovernanceSlot::Ebm => self.slot_enablement.ebm,
        }
    }

    fn update_model_governance_tick(
        &mut self,
        now_t: u64,
        corr: ucf_frames::v1::CorrelationId,
        compute_summary: Option<ucf_frames::v1::ComputeSignalsSummary>,
        llm_timed_out: bool,
    ) -> Result<(), RuntimeError> {
        let pressure_q = compute_summary.map(|s| s.pressure_q).unwrap_or(0);
        let uncertainty_q = compute_summary
            .and_then(|s| s.lfm_uncertainty_q)
            .unwrap_or(0);
        for slot in [
            GovernanceSlot::Llm,
            GovernanceSlot::World,
            GovernanceSlot::Sae,
            GovernanceSlot::Ssm,
            GovernanceSlot::Lfm,
            GovernanceSlot::Ebm,
        ] {
            if let Some(acc) = self.model_governance.accum.get_mut(&slot) {
                acc.samples = acc.samples.saturating_add(1);
                acc.pressure_sum_q = acc.pressure_sum_q.saturating_add(u64::from(pressure_q));
                acc.uncertainty_sum_q = acc
                    .uncertainty_sum_q
                    .saturating_add(u64::from(uncertainty_q));
                if llm_timed_out && matches!(slot, GovernanceSlot::Llm) {
                    acc.timeout = acc.timeout.saturating_add(1);
                }
                if compute_summary
                    .and_then(|s| s.lfm_nan_inf_detected)
                    .unwrap_or(false)
                    && matches!(slot, GovernanceSlot::Lfm)
                {
                    acc.invalid = acc.invalid.saturating_add(1);
                }
            }
        }

        if !now_t.is_multiple_of(self.model_governance.window_ticks) {
            return Ok(());
        }

        let window_start = self.model_governance.current_window_start;
        self.model_governance.current_window_start = now_t;
        let mut degrade_critical = false;
        for slot in [
            GovernanceSlot::Llm,
            GovernanceSlot::World,
            GovernanceSlot::Sae,
            GovernanceSlot::Ssm,
            GovernanceSlot::Lfm,
            GovernanceSlot::Ebm,
        ] {
            let slot_mode = self.governance_slot_mode(slot);
            let Some(acc) = self.model_governance.accum.get_mut(&slot) else {
                continue;
            };
            let policy = self.model_governance.thresholds[&slot];
            let denom = acc.samples.max(1);
            let timeout_rate_q = ((acc.timeout.saturating_mul(u64::from(u16::MAX))) / denom) as u16;
            let invalid_rate_q = ((acc.invalid.saturating_mul(u64::from(u16::MAX))) / denom) as u16;
            let envelope_rate_q =
                ((acc.envelope.saturating_mul(u64::from(u16::MAX))) / denom) as u16;
            let mean_uncertainty_q = (acc.uncertainty_sum_q / denom) as u16;
            let mean_pressure_q = (acc.pressure_sum_q / denom) as u16;
            let delta_uncertainty_q =
                mean_uncertainty_q.abs_diff(self.model_governance.baseline_uncertainty_q);
            let delta_pressure_q =
                mean_pressure_q.abs_diff(self.model_governance.baseline_pressure_q);

            let mut reasons = Vec::new();
            if timeout_rate_q > policy.max_timeout_rate_q {
                reasons.push("timeout_rate".to_string());
            }
            if invalid_rate_q > policy.max_invalid_rate_q {
                reasons.push("invalid_rate".to_string());
            }
            if envelope_rate_q > policy.max_envelope_violation_rate_q {
                reasons.push("envelope_violation_rate".to_string());
            }
            if delta_uncertainty_q > policy.max_delta_uncertainty_q {
                reasons.push("delta_uncertainty".to_string());
            }
            if delta_pressure_q > policy.max_delta_pressure_q {
                reasons.push("delta_pressure".to_string());
            }

            let breached = !reasons.is_empty();
            let streak = self
                .model_governance
                .breaches_streak
                .entry(slot)
                .or_insert(0);
            if breached {
                *streak = streak.saturating_add(1);
                self.model_governance
                    .status
                    .insert(slot, ModelGovernanceStatus::Degraded);
                if matches!(slot, GovernanceSlot::Llm | GovernanceSlot::Ebm) {
                    degrade_critical = true;
                }
                let rec = ExperienceRecord::note(
                    self.ids.next(),
                    ucf_core::types::SimTime {
                        tick: ucf_core::types::Tick::new(now_t),
                        window: ucf_core::types::WindowId::new(0),
                    },
                    corr,
                    format!(
                        "model_governance_alarm slot={} window={}..{} reasons={} mode={:?}",
                        slot.as_str(),
                        window_start,
                        now_t,
                        reasons.join(","),
                        slot_mode
                    ),
                );
                self.ess.append(rec)?;
                if *streak >= self.model_governance.persistent_windows {
                    let rec = ExperienceRecord::note(
                        self.ids.next(),
                        ucf_core::types::SimTime {
                            tick: ucf_core::types::Tick::new(now_t),
                            window: ucf_core::types::WindowId::new(0),
                        },
                        corr,
                        format!(
                            "model_rollback_recommendation slot={} reasons={} human_required_prod=true",
                            slot.as_str(),
                            reasons.join(",")
                        ),
                    );
                    self.ess.append(rec)?;
                }
            } else {
                *streak = 0;
                self.model_governance
                    .status
                    .insert(slot, ModelGovernanceStatus::Healthy);
                if slot_mode == SlotMode::Toy {
                    self.model_governance.baseline_uncertainty_q = mean_uncertainty_q;
                    self.model_governance.baseline_pressure_q = mean_pressure_q;
                }
            }
            *acc = SlotWindowAcc::default();
        }
        if degrade_critical {
            self.compute_budget.governor_tier = self.compute_budget.governor_tier.max(2);
        }
        Ok(())
    }

    fn summarize_tick_for_evolution(
        &mut self,
        compute_summary: Option<ucf_frames::v1::ComputeSignalsSummary>,
        denied_tool: bool,
        governor_tier: u8,
        backend_pack_digest: [u8; 32],
        nsr_available: bool,
        ebm_energy_mean_q: u16,
    ) {
        let summary = EvolutionTickSummary {
            risk: compute_summary
                .map(|c| c.risk)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0),
            confidence: compute_summary
                .map(|c| c.confidence)
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            coherence: compute_summary
                .and_then(|c| c.coherence)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0),
            instability: compute_summary
                .and_then(|c| c.instability)
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            pressure: compute_summary
                .map(|c| c.pressure)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0),
            surprise: compute_summary
                .map(|c| c.surprise)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0),
            uncertainty: compute_summary
                .and_then(|c| c.lfm_uncertainty)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0),
            ebm_energy_mean_q,
            governor_tier: governor_tier.min(3),
            budget_exceeded: compute_summary
                .and_then(|c| c.budget_exceeded_stage)
                .is_some(),
            denied_tool,
            evidence_chain_digest: compute_summary
                .and_then(|c| c.compute_chain_digest)
                .unwrap_or([0; 32]),
            backend_pack_digest,
            nsr_available,
        };
        if summary.budget_exceeded {
            self.evolution_budget_exceeded_total =
                self.evolution_budget_exceeded_total.saturating_add(1);
            metrics::counter!("ucf_evolution_budget_exceeded_total").increment(1);
        }
        self.evolution_summaries.push(summary);
        let keep = self.evolution_window_ticks.max(8) as usize;
        if self.evolution_summaries.len() > keep {
            let drop_n = self.evolution_summaries.len().saturating_sub(keep);
            self.evolution_summaries.drain(0..drop_n);
        }
    }

    fn liquid_window_stats(&self, now_t: u64) -> Option<LiquidWindowStats> {
        if self.evolution_summaries.len() < 8 {
            return None;
        }
        let t0 = now_t.saturating_sub(self.evolution_window_ticks);
        let t1 = now_t;
        let n = self.evolution_summaries.len() as f32;
        let half = (self.evolution_summaries.len() / 2).max(1);
        let first = &self.evolution_summaries[..half];
        let second = &self.evolution_summaries[half..];

        let mean = |f: fn(&EvolutionTickSummary) -> f32| -> f32 {
            self.evolution_summaries.iter().map(f).sum::<f32>() / n
        };
        let max = |f: fn(&EvolutionTickSummary) -> f32| -> f32 {
            self.evolution_summaries
                .iter()
                .map(f)
                .fold(0.0_f32, f32::max)
        };
        let half_mean =
            |slice: &[EvolutionTickSummary], f: fn(&EvolutionTickSummary) -> f32| -> f32 {
                if slice.is_empty() {
                    return 0.0;
                }
                slice.iter().map(f).sum::<f32>() / slice.len() as f32
            };

        let mean_uncertainty = mean(|s| s.uncertainty).clamp(0.0, 1.0);
        let max_uncertainty = max(|s| s.uncertainty).clamp(0.0, 1.0);
        let mean_pressure = mean(|s| s.pressure).clamp(0.0, 1.0);
        let max_pressure = max(|s| s.pressure).clamp(0.0, 1.0);
        let mean_surprise = mean(|s| s.surprise).clamp(0.0, 1.0);
        let max_surprise = max(|s| s.surprise).clamp(0.0, 1.0);
        let mean_risk = mean(|s| s.risk).clamp(0.0, 1.0);
        let max_risk = max(|s| s.risk).clamp(0.0, 1.0);
        let mean_coherence_value = mean(|s| s.coherence).clamp(0.0, 1.0);
        let mean_coherence = Some(mean_coherence_value);
        let delta_mean_uncertainty = (half_mean(second, |s| s.uncertainty)
            - half_mean(first, |s| s.uncertainty))
        .clamp(-1.0, 1.0);
        let delta_mean_pressure =
            (half_mean(second, |s| s.pressure) - half_mean(first, |s| s.pressure)).clamp(-1.0, 1.0);
        let ebm_energy_mean_q = (self
            .evolution_summaries
            .iter()
            .map(|s| s.ebm_energy_mean_q as f32)
            .sum::<f32>()
            / n
            / 65535.0)
            .clamp(0.0, 1.0);
        let ebm_energy_trend_q = ((half_mean(second, |s| s.ebm_energy_mean_q as f32)
            - half_mean(first, |s| s.ebm_energy_mean_q as f32))
            / 65535.0)
            .clamp(-1.0, 1.0);

        let q = |v: f32| quantize_unit_u16(v) as u32;
        let q_signed = |v: f32| ((v.clamp(-1.0, 1.0) * 10_000.0).round() as i32).to_be_bytes();
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ucf.runtime.evolution.liquid_window.v0");
        hasher.update(&t0.to_be_bytes());
        hasher.update(&t1.to_be_bytes());
        hasher.update(&q(mean_uncertainty).to_be_bytes());
        hasher.update(&q(max_uncertainty).to_be_bytes());
        hasher.update(&q(mean_pressure).to_be_bytes());
        hasher.update(&q(max_pressure).to_be_bytes());
        hasher.update(&q(mean_surprise).to_be_bytes());
        hasher.update(&q(max_surprise).to_be_bytes());
        hasher.update(&q(mean_risk).to_be_bytes());
        hasher.update(&q(max_risk).to_be_bytes());
        hasher.update(&q(mean_coherence_value).to_be_bytes());
        hasher.update(&q_signed(delta_mean_uncertainty));
        hasher.update(&q_signed(delta_mean_pressure));
        hasher.update(&quantize_unit_u16(ebm_energy_mean_q).to_le_bytes());
        hasher.update(&q_signed(ebm_energy_trend_q));

        Some(LiquidWindowStats {
            window: (t0, t1),
            mean_uncertainty,
            max_uncertainty,
            mean_pressure,
            max_pressure,
            mean_surprise,
            max_surprise,
            mean_risk,
            max_risk,
            mean_coherence,
            delta_mean_uncertainty,
            delta_mean_pressure,
            ebm_energy_mean_q,
            ebm_energy_trend_q,
            digest: *hasher.finalize().as_bytes(),
        })
    }

    fn maybe_run_evolution(
        &mut self,
        time: ucf_core::types::SimTime,
        corr: ucf_frames::v1::CorrelationId,
    ) -> Result<(), RuntimeError> {
        if !self.evolution_enabled || self.evolution_summaries.len() < 8 {
            return Ok(());
        }
        if !time.tick.get().is_multiple_of(self.evolution_window_ticks) {
            return Ok(());
        }
        let n = self.evolution_summaries.len() as f32;
        let risk_mean = self.evolution_summaries.iter().map(|x| x.risk).sum::<f32>() / n;
        let confidence_mean = self
            .evolution_summaries
            .iter()
            .map(|x| x.confidence)
            .sum::<f32>()
            / n;
        let coherence_mean = self
            .evolution_summaries
            .iter()
            .map(|x| x.coherence)
            .sum::<f32>()
            / n;
        let instability_mean = self
            .evolution_summaries
            .iter()
            .map(|x| x.instability)
            .sum::<f32>()
            / n;
        let budget_exceeded_rate = self
            .evolution_summaries
            .iter()
            .filter(|x| x.budget_exceeded)
            .count() as f32
            / n;
        let denied_tool_rate = self
            .evolution_summaries
            .iter()
            .filter(|x| x.denied_tool)
            .count() as f32
            / n;
        let governor_tier_mean = self
            .evolution_summaries
            .iter()
            .map(|x| f32::from(x.governor_tier))
            .sum::<f32>()
            / n;
        let governor_tier_max = self
            .evolution_summaries
            .iter()
            .map(|x| x.governor_tier)
            .max()
            .unwrap_or(0);
        let evidence_chain_digest = self
            .evolution_summaries
            .last()
            .map(|s| s.evidence_chain_digest)
            .unwrap_or([0; 32]);
        let backend_pack_digest = self
            .evolution_summaries
            .last()
            .map(|s| s.backend_pack_digest)
            .unwrap_or([0; 32]);
        let nsr_available = self.evolution_summaries.iter().all(|s| s.nsr_available);
        let emergency_active = self.emergency_active();
        let liquid_window =
            self.liquid_window_stats(time.tick.get())
                .unwrap_or(LiquidWindowStats {
                    window: (
                        time.tick.get().saturating_sub(self.evolution_window_ticks),
                        time.tick.get(),
                    ),
                    mean_uncertainty: 1.0,
                    max_uncertainty: 1.0,
                    mean_pressure: 1.0,
                    max_pressure: 1.0,
                    mean_surprise: 1.0,
                    max_surprise: 1.0,
                    mean_risk: risk_mean,
                    max_risk: risk_mean,
                    mean_coherence: Some(coherence_mean),
                    delta_mean_uncertainty: 0.0,
                    delta_mean_pressure: 0.0,
                    ebm_energy_mean_q: 0.0,
                    ebm_energy_trend_q: 0.0,
                    digest: [0; 32],
                });
        if emergency_active {
            self.evolution_suppressed_total = self.evolution_suppressed_total.saturating_add(1);
            metrics::counter!("ucf_structural_delta_suppressed_total", "reason" => "emergency")
                .increment(1);
            self.ess.append(ExperienceRecord::note(
                self.ids.next(),
                time,
                corr,
                "evolution_engine_suppressed:emergency",
            ))?;
            return Ok(());
        }
        let ctx = EvolutionContext {
            t: time.tick.get(),
            source_window: (
                time.tick.get().saturating_sub(self.evolution_window_ticks),
                time.tick.get(),
            ),
            evidence_chain_digest,
            risk_mean,
            confidence_mean,
            coherence_mean,
            instability_mean,
            budget_exceeded_rate,
            denied_tool_rate,
            stress_index: self.last_hormone_summary.map(|s| s.stress_index),
            neuro_arousal: self.last_neuro_summary.map(|s| s.arousal),
            liquid_window,
            governor_tier_mean,
            governor_tier_max,
            emergency_active,
            backend_pack_digest,
            nsr_available,
            params: TunableSnapshot {
                beta_policy_risk: self.fep_state.cfg.beta_policy_risk,
                beta_coherence_lock: self.fep_state.cfg.beta_coherence_lock,
                structure_delta_cap: self.fep_state.cfg.structure_delta_cap,
                coherence_min_closed_loop_gain: self.fep_state.coh_cfg.min_closed_loop_gain,
                coherence_max_unchecked_drift: self.fep_state.coh_cfg.max_unchecked_drift,
                coherence_max_memory_pressure: self.fep_state.coh_cfg.max_memory_pressure,
                coherence_risk_inhibit_min: self.fep_state.coh_cfg.min_policy_inhibit_on_risk,
            },
        };
        if !ctx.nsr_available {
            self.evolution_suppressed_total = self.evolution_suppressed_total.saturating_add(1);
            metrics::counter!("ucf_structural_delta_suppressed_total", "reason" => "nsr_unavailable")
                .increment(1);
            self.ess.append(ExperienceRecord::note(
                self.ids.next(),
                time,
                corr,
                "evolution_engine_suppressed:nsr_unavailable",
            ))?;
        }
        let _span = tracing::info_span!(
            "evolution.step",
            window = self.evolution_window_ticks,
            t = time.tick.get()
        )
        .entered();
        let mut candidates = self
            .evolution_engine
            .propose(ctx, EvolutionBudget { work_units: 64 });
        if candidates.len() > 4 {
            candidates.truncate(4);
        }
        let mut scored = Vec::with_capacity(candidates.len());
        for delta in candidates {
            self.evolution_proposals_total = self.evolution_proposals_total.saturating_add(1);
            metrics::counter!("ucf_structural_delta_proposals_total").increment(1);
            let proposal = DeltaProposalRecord {
                schema_version: 1,
                delta_id: delta.delta_id,
                t: delta.t,
                target: delta.target as u8,
                ops_summary: ops_summary_bytes(&delta),
                digest: delta.digest,
                evidence_chain_digest,
                window_stats_digest: ctx.liquid_window.digest,
                backend_pack_digest: ctx.backend_pack_digest,
            };
            self.ess.append(ExperienceRecord::from_delta_proposal(
                self.ids.next(),
                time,
                corr,
                proposal,
            ))?;
            let score = self.evolution_engine.evaluate(&delta, &ctx);
            scored.push((delta, score));
        }
        let selection = self.evolution_engine.select(&scored);
        let accepted_id = selection.accepted.as_ref().map(|(d, _)| d.delta_id);
        if let Some(id) = accepted_id {
            tracing::info!(accepted_delta_id = %hex::encode(&id[..4]), "evolution accepted candidate");
        }
        for (delta, score) in scored {
            let accepted = accepted_id == Some(delta.delta_id);
            if accepted {
                self.evolution_accepted_total = self.evolution_accepted_total.saturating_add(1);
                metrics::counter!("ucf_structural_delta_recommended_total").increment(1);
            } else {
                self.evolution_rejected_total = self.evolution_rejected_total.saturating_add(1);
                metrics::counter!("ucf_structural_delta_rejected_total", "reason" => top_reason(&score))
                    .increment(1);
            }
            let eval = DeltaEvaluationRecord {
                schema_version: 1,
                delta_id: delta.delta_id,
                fitness_q: u16::from(quantize_unit(score.fitness)),
                risk_penalty_q: u16::from(quantize_unit(score.risk_penalty)),
                stability_penalty_q: u16::from(quantize_unit(score.stability_penalty)),
                budget_penalty_q: u16::from(quantize_unit(score.budget_penalty)),
                score_digest: score.digest,
                accepted,
                reason_codes: reason_codes(&score),
                evidence_chain_digest,
                window_stats_digest: ctx.liquid_window.digest,
                backend_pack_digest: ctx.backend_pack_digest,
                suppression_reason_code: suppression_reason_code(&ctx),
            };
            self.ess.append(ExperienceRecord::from_delta_evaluation(
                self.ids.next(),
                time,
                corr,
                eval,
            ))?;
            if accepted {
                self.evolution_recommended_total =
                    self.evolution_recommended_total.saturating_add(1);
                let rec = DeltaRecommendationRecord {
                    schema_version: 1,
                    delta_id: delta.delta_id,
                    recommended_ops: ops_summary_bytes(&delta),
                    safety_clamps: clamp_summary_bytes(),
                    requires_human_apply: true,
                    evidence_chain_digest,
                    window_stats_digest: ctx.liquid_window.digest,
                    backend_pack_digest: ctx.backend_pack_digest,
                };
                self.ess
                    .append(ExperienceRecord::from_delta_recommendation(
                        self.ids.next(),
                        time,
                        corr,
                        rec,
                    ))?;
            }
        }
        Ok(())
    }

    fn persist_issuance_records(
        &mut self,
        time: ucf_core::types::SimTime,
        corr: ucf_frames::v1::CorrelationId,
        decision_id: u64,
        candidate_id: u16,
        issuance: &CapabilityIssuanceDecision,
        effective_tier: u8,
    ) -> Result<(), RuntimeError> {
        let issuance_payload = AuditPayload::CapabilityIssuance(CapabilityIssuanceRecord {
            policy_bundle_hash: self.policy_bundle_hash.clone(),
            policy_graph_digest: self.policy_graph_digest.clone(),
            t: time.tick.get(),
            decision_id,
            candidate_id: Some(candidate_id),
            requested_kinds: issuance
                .requested_kinds
                .iter()
                .map(|k| k.as_tag().to_string())
                .take(12)
                .collect(),
            granted_kinds: issuance
                .granted_kinds
                .iter()
                .map(|k| k.as_tag().to_string())
                .take(12)
                .collect(),
            denied_kinds: issuance
                .denied
                .iter()
                .map(|x| (x.kind.as_tag().to_string(), x.reason_code.to_string()))
                .take(16)
                .collect(),
            tier: issuance.tier.as_u8(),
            effective_tier,
            emergency_override: effective_tier > issuance.tier.as_u8(),
            governor_score_q: issuance.governor_score_q,
            governor_score_before_q: issuance.governor_score_before_q,
            governor_score_after_q: issuance.governor_score_after_q,
            ebm_penalty_q: issuance.ebm_penalty_q,
            nsr_penalty_q: issuance.nsr_penalty_q,
            ebm_energy_used_q: issuance.ebm_energy_used_q,
            governance_signals_digest: issuance.governance_signals_digest,
            throttle_state_digest: issuance.throttle_state_digest,
            evidence_chain_digest: issuance.evidence_chain_digest,
            schema_version: issuance.schema_version,
        });
        let issuance_record = ExperienceRecord::audit(
            self.ids.next(),
            time,
            corr,
            ExperienceKind::CapabilityIssuance,
            issuance_payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = issuance_record
            .audit_digest
            .unwrap_or(self.audit_head_digest);
        self.ess.append(issuance_record)?;

        for (kind, slot) in self.tool_governor.snapshot() {
            if slot.cooldown_ticks > 0 {
                metrics::counter!("ucf_tool_cooldown_active_total", "kind" => kind.as_tag().to_string())
                    .increment(1);
            }
            let throttle_payload = AuditPayload::Throttle(ThrottleRecord {
                t: time.tick.get(),
                kind: kind.as_tag().to_string(),
                tokens_remaining: slot.tokens,
                cooldown_ticks: slot.cooldown_ticks,
                deny_count: slot.deny_count,
                digest: issuance.throttle_state_digest,
                schema_version: 1,
            });
            let rec = ExperienceRecord::audit(
                self.ids.next(),
                time,
                corr,
                ExperienceKind::Throttle,
                throttle_payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = rec.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(rec)?;
        }
        Ok(())
    }

    fn persist_compute_budget_violation(
        &mut self,
        time: ucf_core::types::SimTime,
        corr: ucf_frames::v1::CorrelationId,
        violation: &crate::compute_economics::ComputeBudgetViolationRecord,
    ) -> Result<(), RuntimeError> {
        let payload = AuditPayload::ComputeBudgetViolation(ComputeBudgetViolationRecord {
            t: time.tick.get(),
            stage: violation.stage.as_str().to_string(),
            pool: match violation.pool {
                BudgetPool::Primary => "primary",
                BudgetPool::Shadow => "shadow",
            }
            .to_string(),
            reason: violation.reason.to_string(),
            attempted_cost: violation.attempted_cost,
            available: violation.available,
            schema_version: 1,
        });
        let rec = ExperienceRecord::audit(
            self.ids.next(),
            time,
            corr,
            ExperienceKind::ComputeBudgetViolation,
            payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = rec.audit_digest.unwrap_or(self.audit_head_digest);
        self.ess.append(rec)?;
        metrics::counter!("ucf_budget_denials_total", "stage" => violation.stage.as_str().to_string())
            .increment(1);
        Ok(())
    }

    fn maybe_persist_compute_budget_window(
        &mut self,
        time: ucf_core::types::SimTime,
        corr: ucf_frames::v1::CorrelationId,
    ) -> Result<(), RuntimeError> {
        let window = time.window.get();
        if window == self.compute_budget_window_t {
            return Ok(());
        }
        if self.compute_tier_count > 0 {
            let mean =
                (self.compute_tier_sum as f32 / self.compute_tier_count as f32).clamp(0.0, 3.0);
            let snap = self.compute_economy.snapshot_window(
                self.compute_budget_window_t0,
                time.tick.get().saturating_sub(1),
                self.compute_budget_window_t,
                quantize_unit_u16(mean / 3.0),
                self.compute_tier_max,
                {
                    let mut out = [0_u8; 8];
                    let decoded = hex::decode(&self.policy_bundle_hash).unwrap_or_default();
                    if decoded.len() >= 8 {
                        out.copy_from_slice(&decoded[..8]);
                    }
                    out
                },
            );
            let spent = |st: ComputeStage| *snap.spent_per_stage.get(&st).unwrap_or(&0);
            let payload = AuditPayload::ComputeBudgetWindow(ComputeBudgetWindowRecord {
                t0: snap.t0,
                t1: snap.t1,
                window: snap.window,
                primary_available_start: snap.primary_start.available,
                primary_spent_start: snap.primary_start.spent,
                primary_available_end: snap.primary_end.available,
                primary_spent_end: snap.primary_end.spent,
                shadow_available_start: snap.shadow_start.available,
                shadow_spent_start: snap.shadow_start.spent,
                shadow_available_end: snap.shadow_end.available,
                shadow_spent_end: snap.shadow_end.spent,
                llm_spent: spent(ComputeStage::Llm),
                governor_spent: spent(ComputeStage::Governor),
                jepa_spent: spent(ComputeStage::Jepa),
                sae_spent: spent(ComputeStage::Sae),
                ssm_spent: spent(ComputeStage::Ssm),
                lfm_spent: spent(ComputeStage::Lfm),
                tool_spent: spent(ComputeStage::Tool),
                governor_tier_mean_q: snap.governor_tier_mean_q,
                governor_tier_max: snap.governor_tier_max,
                policy_hash_prefix: self.policy_graph_digest_prefix,
                schema_version: 1,
            });
            let rec = ExperienceRecord::audit(
                self.ids.next(),
                time,
                corr,
                ExperienceKind::ComputeBudgetWindow,
                payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = rec.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(rec)?;
        }
        self.compute_economy.reset_window();
        self.compute_budget_window_t = window;
        self.compute_budget_window_t0 = time.tick.get();
        self.compute_tier_sum = 0;
        self.compute_tier_count = 0;
        self.compute_tier_max = 0;
        Ok(())
    }

    fn q_u16(v: f32) -> u16 {
        (v.clamp(0.0, 1.0) * 65_535.0).round() as u16
    }

    fn q_signed_u16(v: f32) -> u16 {
        let clamped = v.clamp(-1.0, 1.0);
        (((clamped + 1.0) * 0.5) * 65_535.0).round() as u16
    }

    fn emergency_active(&self) -> bool {
        matches!(self.emergency_state, EmergencyState::Active { .. })
    }

    fn persist_emergency_transition(
        &mut self,
        time: ucf_core::types::SimTime,
        corr: ucf_frames::v1::CorrelationId,
        state: EmergencyStateCode,
        reason: EmergencyTriggerReason,
        stability: StabilitySnapshot,
        compute_summary: &ucf_compute::ComputeSignalsSummary,
    ) -> Result<(), RuntimeError> {
        let payload = AuditPayload::Emergency(EmergencyRecord {
            policy_bundle_hash: self.policy_bundle_hash.clone(),
            policy_graph_digest: self.policy_graph_digest.clone(),
            t: time.tick.get(),
            state,
            reason: reason.as_ess(),
            v_q: Self::q_u16(stability.v),
            dv_q: Self::q_signed_u16(stability.dv),
            state_norm_q: Self::q_u16(stability.state_norm),
            deriv_norm_q: Self::q_u16(stability.deriv_norm),
            lfm_digest: compute_summary.lfm_digest.unwrap_or([0; 32]),
            backend_pack_digest: backend_pack_digest(*compute_summary),
            evidence_chain_digest: compute_summary.compute_chain_digest,
            schema_version: 1,
        });
        let rec = ExperienceRecord::audit(
            self.ids.next(),
            time,
            corr,
            ExperienceKind::Emergency,
            payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = rec.audit_digest.unwrap_or(self.audit_head_digest);
        self.ess.append(rec)?;
        Ok(())
    }

    fn update_emergency_state(
        &mut self,
        time: ucf_core::types::SimTime,
        corr: ucf_frames::v1::CorrelationId,
        compute_summary: &ucf_compute::ComputeSignalsSummary,
    ) -> Result<StabilitySnapshot, RuntimeError> {
        let state_norm = compute_summary
            .lfm_state_norm
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let deriv_norm = compute_summary
            .lfm_deriv_norm
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let saturation_ratio = compute_summary
            .lfm_saturation_ratio
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let nan_inf_detected = compute_summary.lfm_nan_inf_detected;
        let v = (state_norm * state_norm + LFM_V_K * deriv_norm * deriv_norm).clamp(0.0, 2.0);
        let dv = self.emergency_last_v.map(|prev| v - prev).unwrap_or(0.0);
        self.emergency_last_v = Some(v);

        let reason = if nan_inf_detected {
            Some(EmergencyTriggerReason::NanInf)
        } else if v > LFM_V_MAX {
            Some(EmergencyTriggerReason::RunawayV)
        } else if saturation_ratio > LFM_SATURATION_MAX {
            Some(EmergencyTriggerReason::Saturation)
        } else if dv > LFM_DV_MAX {
            self.emergency_dv_trend_count = self.emergency_dv_trend_count.saturating_add(1);
            if self.emergency_dv_trend_count >= EMERGENCY_DV_TREND_K {
                Some(EmergencyTriggerReason::TrendDV)
            } else {
                None
            }
        } else {
            self.emergency_dv_trend_count = 0;
            None
        };

        let snapshot = StabilitySnapshot {
            v,
            dv,
            state_norm,
            deriv_norm,
            saturation_ratio,
            nan_inf_detected,
        };

        match self.emergency_state {
            EmergencyState::Inactive => {
                if let Some(r) = reason {
                    if matches!(
                        r,
                        EmergencyTriggerReason::RunawayV | EmergencyTriggerReason::NanInf
                    ) {
                        self.emergency_state = EmergencyState::Active {
                            since_t: time.tick.get(),
                            reason: r,
                            cool_down_remaining: EMERGENCY_COOLDOWN_TICKS,
                        };
                        self.persist_emergency_transition(
                            time,
                            corr,
                            EmergencyStateCode::Active,
                            r,
                            snapshot,
                            compute_summary,
                        )?;
                        self.emergency_last_reason = Some(r);
                    } else {
                        self.emergency_state = EmergencyState::Armed {
                            since_t: time.tick.get(),
                            reason: r,
                        };
                        self.persist_emergency_transition(
                            time,
                            corr,
                            EmergencyStateCode::Armed,
                            r,
                            snapshot,
                            compute_summary,
                        )?;
                    }
                }
            }
            EmergencyState::Armed {
                since_t,
                reason: armed_reason,
            } => {
                if let Some(r) = reason {
                    if time.tick.get().saturating_sub(since_t)
                        >= u64::from(EMERGENCY_ARM_TREND_TICKS)
                        || matches!(
                            r,
                            EmergencyTriggerReason::RunawayV | EmergencyTriggerReason::NanInf
                        )
                    {
                        self.emergency_state = EmergencyState::Active {
                            since_t: time.tick.get(),
                            reason: r,
                            cool_down_remaining: EMERGENCY_COOLDOWN_TICKS,
                        };
                        self.persist_emergency_transition(
                            time,
                            corr,
                            EmergencyStateCode::Active,
                            r,
                            snapshot,
                            compute_summary,
                        )?;
                        self.emergency_last_reason = Some(r);
                    }
                } else {
                    self.emergency_state = EmergencyState::Inactive;
                    self.persist_emergency_transition(
                        time,
                        corr,
                        EmergencyStateCode::Off,
                        armed_reason,
                        snapshot,
                        compute_summary,
                    )?;
                }
            }
            EmergencyState::Active {
                since_t,
                reason: active_reason,
                cool_down_remaining,
            } => {
                let _ = since_t;
                let remaining = cool_down_remaining.saturating_sub(1);
                self.emergency_state = EmergencyState::Active {
                    since_t: time.tick.get(),
                    reason: active_reason,
                    cool_down_remaining: remaining,
                };
                if remaining == 0 && reason.is_none() {
                    self.emergency_state = EmergencyState::Inactive;
                    self.persist_emergency_transition(
                        time,
                        corr,
                        EmergencyStateCode::Off,
                        active_reason,
                        snapshot,
                        compute_summary,
                    )?;
                }
            }
        }

        metrics::gauge!("ucf_emergency_active").set(if self.emergency_active() {
            1.0
        } else {
            0.0
        });
        let cooldown = match self.emergency_state {
            EmergencyState::Active {
                cool_down_remaining,
                ..
            } => cool_down_remaining,
            _ => 0,
        };
        metrics::gauge!("ucf_emergency_cooldown_ticks").set(f64::from(cooldown));

        if let Some(r) = reason {
            metrics::counter!("ucf_emergency_trigger_total", "reason" => r.as_label().to_string())
                .increment(1);
        }

        Ok(snapshot)
    }

    pub fn try_new_from_env() -> Result<Self, RuntimeError> {
        let cfg = ComputeBackendConfig::from_env()?;
        let llm_cfg = LlmBackendConfig::from_env()?;
        let mut orchestrator = Self::new();
        orchestrator.compute_budget = cfg.to_budget();
        orchestrator.compute_backend = match build_backend(&cfg) {
            Ok(backend) => backend,
            Err(ComputeError::NotImplemented) => Box::new(NotImplementedBackend {
                name: canonical_backend_label(cfg.kind.as_env_str()),
            }),
            Err(err) => return Err(err.into()),
        };
        orchestrator.llm_backend = build_llm_backend(llm_cfg)?;
        orchestrator.llm_cfg = llm_cfg;
        orchestrator.append_backend_pack_record(0, "startup")?;
        Ok(orchestrator)
    }

    fn append_backend_pack_record(&mut self, t: u64, reason: &str) -> Result<(), RuntimeError> {
        let pack =
            BackendPackFactory::build(BackendPackConfig::from_env().unwrap_or(BackendPackConfig {
                pack: BackendPackKind::ToyV1,
                seed: self.compute_budget.seed,
            }))
            .map_err(RuntimeError::from)?;
        let meta = pack.meta();
        let time = ucf_core::types::SimTime {
            tick: ucf_core::types::Tick::new(t),
            window: ucf_core::types::WindowId::new(0),
        };
        let corr = ucf_frames::v1::CorrelationId(0);
        self.ess.append(ExperienceRecord::from_backend_pack(
            self.ids.next(),
            time,
            corr,
            BackendPackRecord {
                schema_version: 1,
                t,
                pack_name: meta.pack_name.to_string(),
                pack_id: meta.pack_id.0,
                fixtures_digest: meta.fixtures_digest,
                model_hashes_digest: meta.model_hashes_digest,
                llm_backend: meta.llm_backend as u8,
                world_backend: meta.world_backend as u8,
                sae_backend: meta.sae_backend as u8,
                ssm_backend: meta.ssm_backend as u8,
                lfm_backend: meta.lfm_backend as u8,
                meta_digest: meta.digest,
                reason: reason.to_string(),
            },
        ))?;
        Ok(())
    }

    pub fn new() -> Self {
        fail_if_training_mode_enabled();
        let mut onn = OnnCore::new(1.0, 0.0);
        onn.register(MOD_PBM, PhaseDeg(0.0));

        let onn_cfg = OnnCfg::default_v0();
        let onn_nodes = [
            OnnNode::Global,
            OnnNode::Tcf,
            OnnNode::Ssm,
            OnnNode::Nsr,
            OnnNode::Cde,
            OnnNode::Iit,
            OnnNode::Sle,
            OnnNode::Ncde,
            OnnNode::Spikes,
        ];
        let onn_state = OnnState::new(&onn_cfg, &onn_nodes);

        let ssm_cfg = SsmCfg::default_small();
        let ssm_state = SsmState::new(&ssm_cfg, 0);
        let ncde_cfg = NcdeCfg::default_v0();
        let ncde_state = NcdeState::new(&ncde_cfg);

        let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let policy_root = workspace_root.join("policies");

        if std::env::var("UCF_POLICY_BUNDLE_SHA256").is_err() {
            if let Ok(manifest) = crate::io_caps::IoCaps::runtime_default()
                .read_to_string(std::path::Path::new("manifest.toml"))
            {
                if let Some(hash_line) = manifest.lines().find(|l| l.starts_with("bundle_sha256 ="))
                {
                    let hash = hash_line
                        .split('=')
                        .nth(1)
                        .map(str::trim)
                        .unwrap_or("")
                        .trim_matches('"')
                        .to_string();
                    if !hash.is_empty() {
                        std::env::set_var("UCF_POLICY_BUNDLE_SHA256", hash);
                    }
                }
            }
        }

        let policy_provenance = verify_policy_bundle(&policy_root)
            .unwrap_or_else(|e| panic!("policy bundle verification failed: {e}"));
        let policy_overlay = std::env::var("UCF_POLICY_OVERLAY")
            .ok()
            .map(|v| policy_root.join(format!("packs/overlays/{v}")));
        let policy_base = policy_root.join("packs/base_v1");
        let (policy_graph, policy_graph_prov) =
            load_and_merge_policy_graph(&policy_base, policy_overlay.as_deref())
                .unwrap_or_else(|e| panic!("policy graph load failed: {e}"));
        if let Ok(expected) = std::env::var("UCF_POLICY_GRAPH_DIGEST") {
            assert_eq!(
                expected, policy_graph_prov.policy_graph_digest,
                "policy graph digest mismatch"
            );
        }
        let slot_enablement = SlotEnablement::from_env().unwrap_or_default();
        let model_governance = ModelGovernanceRuntime::from_policy(512, &policy_graph.thresholds);
        let ebm_constraints = load_runtime_ebm_constraints(&policy_provenance.bundle_sha256);
        configure_ebm_constraints(ebm_constraints);
        metrics::gauge!("ucf_policy_bundle_verified").set(1.0);

        let mut out = Self {
            ess: InMemoryEss::new(),
            ids: IdAllocator::new(1),
            pbm: Pbm,
            gem: Gem,
            compute_backend: Box::new(CpuStubBackend),
            compute_budget: ComputeBudget::default(),
            llm_backend: build_llm_backend(LlmBackendConfig::from_env().unwrap_or_default())
                .unwrap_or_else(|_| std::sync::Arc::new(ucf_compute::capabilities::LlmStubBackend)),
            llm_cfg: LlmBackendConfig::from_env().unwrap_or_default(),
            neuromod_field: NeuromodulatorField::new_baseline(),
            neuromod_scheduler: NeuromodScheduler::new(1),
            onn,
            snn_encode_cfg: SnnEncodeCfg::default(),
            snn_cfg: SnnCfg::default_v0(),
            snn_tick_counter: 0,
            last_snn_spike_count: 0,
            last_spike_rate_0_1: 0.0,
            biophys_field: BiophysField::default(),
            biophys_cfg: FieldUpdateCfg::default(),
            biophys_mod_cfg: ModulationCfg::default(),
            hpa_state: HpaState::default(),
            hpa_cfg: HpaCfg::default(),
            last_biophys_tick_ms: None,
            last_biophys_frame: None,
            microcircuit: Microcircuit::new_ring(32),
            last_microcircuit_frame: None,
            phase_bus: PhaseBus::default(),
            phase_cfg: PhaseCfg::default(),
            spike_codec_cfg: SpikeCodecCfg::default(),
            last_phase_frame: None,
            onn_state,
            onn_cfg,
            event_bus: EventBus::default(),
            spike_bus: SpikeBus::default(),
            spike_seq: 0,
            last_onn_frame: None,
            tcf_cfg: TcfCfg::default_gamma40(),
            tcf_state: TcfState::new(&TcfCfg::default_gamma40(), 0),
            last_tcf_frame: None,
            forced_mean_lock_for_test: None,
            forced_nsr_risk_for_test: None,
            last_snn_frame: None,
            last_spike_frames: Vec::new(),
            attention_event: false,
            iit_cfg: BiophysIITCfg::default(),
            iit_state: BiophysIITState::default(),
            iit_proxy_cfg: IitCfg::default_v0(),
            iit_proxy_state: IitState::new(&IitCfg::default_v0()),
            last_iit_frame: None,
            cde_graph: CausalGraph::default(),
            cde_cfg: CdeCfg::default(),
            cde_state: CdeState::default(),
            nsr_cfg: RuleCfg::default(),
            nsr_engine: NsrEngine::with_default_rules(),
            nsr_v1_engine: NsrEngineV1::from_policy_file(std::path::Path::new(
                "policies/bundle_v1/nsr_rules_v1.dl",
            )),
            last_cde_frame: None,
            last_nsr_frame: None,
            ssm_state,
            ssm_cfg,
            working_context: WorkingContext::default(),
            last_ssm_frame: None,
            sle_cfg: SleCfg::default_v0(),
            sle_state: SleState::new(),
            last_sle_frame: None,
            last_meta_input: None,
            policy_internal_recursion_allow: true,
            policy_internal_recursion_max_depth: SleCfg::default_v0().max_depth,
            ssm_gate: 0.0,
            ncde_cfg,
            ncde_state,
            last_ncde_frame: None,
            last_ncde_spike_u4: 0.0,
            digital_brain: DigitalBrainState {
                amygdala: BrainRegion::new(RegionKind::Amygdala, 16),
                pfc: BrainRegion::new(RegionKind::Pfc, 16),
                chem: NeuromodState::baseline(),
                chem_cfg: ChemistryCfg::default_v0(),
            },
            last_chem_frame: None,
            last_digital_brain_frame: None,
            emotion: EmotionVector::default(),
            archive: ArchiveLog::new(MemArchiveStore::new(), ArchiveCfg::default()),
            last_archive_append_frame: None,
            last_archive_payload_len: 0,
            fep_state: FepState {
                cfg: FepCfg::default_v0(),
                homeo_cfg: HomeoCfg::default_v0(),
                homeo_state: HomeoState::new(),
                coh_cfg: CoherenceCfg::default_v0(),
            },
            hormone_state: HormoneState::default(),
            hormone_cfg: HormoneCfg::default(),
            hormone_persist_every: 10,
            lfm_persist_every: env_u64("UCF_LFM_PERSIST_EVERY", 2).max(1),
            lfm_window_emit_every: env_u64("UCF_LFM_WINDOW_EMIT_EVERY", 8).max(2),
            liquid_timeline_index: LiquidTimelineIndex::default(),
            last_hormone_summary: None,
            last_gating_modulation: None,
            hormone_degraded_total: 0,
            neuro_state: NeuronPopState::default(),
            neuro_cfg: NeuroCfg::default(),
            neuro_persist_every: 10,
            last_neuro_summary: None,
            last_neuro_modulation: None,
            last_neuro_spikes: None,
            neuro_degraded_total: 0,
            last_neuro_degraded: false,
            forced_surprise_for_test: None,
            forced_geist_drift_for_test: None,
            forced_ess_pressure_for_test: None,
            last_fep_frame: None,
            last_coherence_frame: None,
            coherence_violation_count: 0,
            coherence_violation_flag: false,
            last_fep_outputs: None,
            risk_quality_counts: [0; 3],
            backpressure: 0.0,
            compute_budget_exceeded_total: 0,
            backpressure_active_total: 0,
            fep_risk_penalty_applied_total: 0,
            coherence_gated_total: 0,
            coherence_runtime: {
                let mut runtime = CoherenceRuntime::new();
                runtime.register_subscriber(Subscriber {
                    module_id: 1,
                    name: "cde",
                    interest: InterestProfile::HashBuckets(vec![0, 1, 2]),
                });
                runtime.register_subscriber(Subscriber {
                    module_id: 2,
                    name: "nsr",
                    interest: InterestProfile::HashBuckets(vec![3, 4, 5]),
                });
                runtime.register_subscriber(Subscriber {
                    module_id: 3,
                    name: "ssm",
                    interest: InterestProfile::TopKFeatures(vec![1, 2, 3, 5, 8, 13]),
                });
                runtime
            },
            consolidation_hook_enabled: env_flag("UCF_ENABLE_CONSOLIDATION_HOOK"),
            geist_hook_enabled: env_flag("UCF_ENABLE_GEIST_HOOK"),
            consolidation_hook_errors_total: 0,
            geist_hook_errors_total: 0,
            consolidation_milestones_emitted_total: 0,
            geist_updates_accepted_total: 0,
            geist_updates_rejected_total: 0,
            geist_updates_rejected_not_enough_samples_total: 0,
            geist_updates_rejected_unstable_total: 0,
            geist_updates_rejected_degraded_total: 0,
            geist_updates_rejected_drift_total: 0,
            compute_milestone_aggregator: ComputeMilestoneAggregator::new(vec![60, 600, 3600], 8)
                .expect("valid default consolidation windows"),
            geist_state_updater: GeistStateUpdater::new(60, 0.9, 0.2, 0.25)
                .expect("valid default geist hook config"),
            last_compute_milestone: None,
            tool_gate: ToolGate::new(
                ucf_policy::capability::CapabilitySet::empty(),
                RateLimiter::new(1024),
                Some(policy_provenance.bundle_sha256.clone()),
            ),
            tool_governor: ToolGovernor::default(),
            slot_enablement,
            model_governance,
            spent_tool_tokens: HashSet::new(),
            last_nsr_risk: None,
            last_nsr_penalty_q: 0,
            emergency_state: EmergencyState::Inactive,
            emergency_last_v: None,
            emergency_dv_trend_count: 0,
            emergency_last_reason: None,
            candidate_generator: DefaultCandidateGeneratorV0,
            ebm_reasoner: if env_flag("UCF_EBM_CANDLE") {
                Box::new(CandleEbmReasonerV1::from_model_store(env_flag(
                    "UCF_EBM_SEARCH",
                )))
            } else {
                Box::new(CpuEbmStubV0)
            },
            ebm_mode: std::env::var("UCF_SLOT_EBM_MODE")
                .ok()
                .and_then(|v| EbmEnablementMode::parse(&v))
                .unwrap_or(EbmEnablementMode::Off),
            gpu_mode: GpuMode::from_env(),
            gpu_parity_fail_streak: 0,
            gpu_disabled: false,
            ebm_run_id: 1,
            audit_head_digest: [0; 32],
            audit_chain_checkpoint_total: 0,
            evolution_enabled: env_flag("UCF_ENABLE_EVOLUTION"),
            evolution_window_ticks: env_u64("UCF_EVOLUTION_WINDOW_TICKS", 64).max(8),
            evolution_summaries: Vec::new(),
            evolution_engine: Box::new(MockEvolutionEngineV0::new(0)),
            evolution_proposals_total: 0,
            evolution_accepted_total: 0,
            evolution_rejected_total: 0,
            evolution_budget_exceeded_total: 0,
            evolution_suppressed_total: 0,
            evolution_recommended_total: 0,
            policy_bundle_hash: policy_provenance.bundle_sha256.clone(),
            policy_graph_digest: policy_graph_prov.policy_graph_digest.clone(),
            policy_graph_digest_prefix: prefix8_hex(&policy_graph_prov.policy_graph_digest),
            compute_economy: ComputeEconomy::new(
                ComputeEconomicsProfile::from_env(),
                CostSchedule::default(),
            ),
            compute_budget_window_t0: 0,
            compute_budget_window_t: 0,
            compute_tier_sum: 0,
            compute_tier_count: 0,
            compute_tier_max: 0,
        };

        let _ = out.ess.append(ExperienceRecord::audit(
            out.ids.next(),
            ucf_core::types::SimTime {
                tick: ucf_core::types::Tick::new(0),
                window: ucf_core::types::WindowId::new(0),
            },
            ucf_frames::v1::CorrelationId(0),
            ExperienceKind::PolicyProvenance,
            AuditPayload::PolicyProvenance(PolicyProvenanceRecord {
                t: 0,
                run_id: policy_provenance.run_id,
                bundle_version: policy_provenance.version,
                bundle_hash: policy_provenance.bundle_sha256.clone(),
                policy_graph_digest: policy_graph_prov.policy_graph_digest.clone(),
                base_pack_digest: policy_graph_prov.base_pack_digest.clone(),
                overlay_pack_digest: policy_graph_prov.overlay_pack_digest.clone(),
                enabled_features: policy_provenance.enabled_features,
                schema_version: 1,
            }),
            out.audit_head_digest,
        ));
        let constraints = active_ebm_constraints();
        let _ = out.ess.append(ExperienceRecord::audit(
            out.ids.next(),
            ucf_core::types::SimTime {
                tick: ucf_core::types::Tick::new(0),
                window: ucf_core::types::WindowId::new(0),
            },
            ucf_frames::v1::CorrelationId(0),
            ExperienceKind::EbmConstraintProvenance,
            AuditPayload::EbmConstraintProvenance(EbmConstraintProvenanceRecord {
                t: 0,
                policy_hash_prefix: prefix8_hex(&policy_provenance.bundle_sha256),
                constraints_digest_prefix: prefix8(constraints.constraints_digest),
                term_count: constraints.terms.len().min(usize::from(u16::MAX)) as u16,
                schema_version: constraints.schema_version,
            }),
            out.audit_head_digest,
        ));
        out
    }

    pub fn last_snn_spike_count(&self) -> usize {
        self.last_snn_spike_count
    }

    pub fn last_biophys_frame(&self) -> Option<BiophysFrame> {
        self.last_biophys_frame
    }

    pub fn last_microcircuit_frame(&self) -> Option<MicrocircuitFrame> {
        self.last_microcircuit_frame
    }

    pub fn last_phase_frame(&self) -> Option<PhaseFrame> {
        self.last_phase_frame
    }

    pub fn last_iit_frame(&self) -> Option<IitFrame> {
        self.last_iit_frame
    }

    pub fn last_onn_frame(&self) -> Option<OnnFrame> {
        self.last_onn_frame
    }

    pub fn last_tcf_frame(&self) -> Option<TcfFrame> {
        self.last_tcf_frame
    }

    pub fn last_snn_frame(&self) -> Option<SnnFrame> {
        self.last_snn_frame
    }

    pub fn spike_rate_0_1(&self) -> f32 {
        self.last_spike_rate_0_1
    }

    pub fn set_onn_coupling_for_test(&mut self, coupling: f32) {
        self.onn_cfg.k_couple = coupling.clamp(0.0, 1.0);
    }

    pub fn force_mean_lock_for_test(&mut self, mean_lock: f32) {
        self.forced_mean_lock_for_test = Some(mean_lock.clamp(0.0, 1.0));
    }

    pub fn force_nsr_risk_for_test(&mut self, nsr_risk: f32) {
        self.forced_nsr_risk_for_test = Some(nsr_risk.clamp(0.0, 1.0));
    }

    pub fn force_surprise_for_test(&mut self, surprise: f32) {
        self.forced_surprise_for_test = Some(surprise.clamp(0.0, 1.0));
    }

    pub fn force_geist_drift_for_test(&mut self, drift: f32) {
        self.forced_geist_drift_for_test = Some(drift.clamp(0.0, 1.0));
    }

    pub fn force_ess_pressure_for_test(&mut self, pressure: f32) {
        self.forced_ess_pressure_for_test = Some(pressure.clamp(0.0, 1.0));
    }

    pub fn set_iit_proxy_cfg_for_test(&mut self, cfg: IitCfg) {
        self.iit_proxy_cfg = cfg.clone();
        self.iit_proxy_state = IitState::new(&cfg);
    }

    pub fn drain_event_bus_for_test(&mut self) -> Vec<SnnSpikeEvent> {
        self.event_bus.drain()
    }

    pub fn inject_spike_for_test(&mut self, spike: BusSpike) {
        self.spike_bus.push(spike);
    }

    pub fn last_cde_frame(&self) -> Option<CdeFrame> {
        self.last_cde_frame
    }

    pub fn last_nsr_frame(&self) -> Option<NsrFrame> {
        self.last_nsr_frame
    }

    pub fn last_ssm_frame(&self) -> Option<SsmFrame> {
        self.last_ssm_frame
    }

    pub fn last_sle_frame(&self) -> Option<SleFrame> {
        self.last_sle_frame
    }

    pub fn set_internal_recursion_policy_for_test(&mut self, allow: bool, max_depth: u8) {
        self.policy_internal_recursion_allow = allow;
        self.policy_internal_recursion_max_depth = max_depth.max(1);
    }

    pub fn set_sle_cfg_for_test(&mut self, cfg: SleCfg) {
        self.sle_cfg = cfg;
        self.policy_internal_recursion_max_depth = self
            .policy_internal_recursion_max_depth
            .min(self.sle_cfg.max_depth.max(1));
    }

    pub fn last_spike_frames(&self) -> &[ucf_frames::v1::SpikeFrame] {
        &self.last_spike_frames
    }

    pub fn attention_event(&self) -> bool {
        self.attention_event
    }

    pub fn last_archive_append_frame(&self) -> Option<ArchiveAppendFrame> {
        self.last_archive_append_frame
    }

    pub fn ssm_gate(&self) -> f32 {
        self.ssm_gate
    }

    pub fn working_context_ssm_y(&self) -> &[f32] {
        &self.working_context.ssm_y
    }

    pub fn rebuild_liquid_timeline_index(&mut self) {
        self.liquid_timeline_index.rebuild_from_ess(&self.ess);
    }

    pub fn last_ncde_frame(&self) -> Option<NcdeFrame> {
        self.last_ncde_frame
    }

    pub fn last_chem_frame(&self) -> Option<ChemFrame> {
        self.last_chem_frame
    }

    pub fn last_digital_brain_frame(&self) -> Option<DigitalBrainFrame> {
        self.last_digital_brain_frame
    }

    pub fn emotion_vector(&self) -> (f32, f32) {
        (self.emotion.valence, self.emotion.arousal)
    }

    pub fn ncde_l2_norm_0_1(&self) -> f32 {
        self.last_ncde_frame
            .map(|frame| f32::from(frame.l2_q) / 255.0)
            .unwrap_or(0.0)
    }

    pub fn last_ncde_spike_u4(&self) -> f32 {
        self.last_ncde_spike_u4
    }

    pub fn archive_last_seq(&self) -> u64 {
        self.archive.last_seq().unwrap_or(0)
    }

    pub fn last_archive_payload_len(&self) -> usize {
        self.last_archive_payload_len
    }

    pub fn last_fep_frame(&self) -> Option<FepFrame> {
        self.last_fep_frame
    }

    pub fn last_coherence_frame(&self) -> Option<CoherenceFrame> {
        self.last_coherence_frame
    }

    pub fn last_hormone_summary(&self) -> Option<HormoneStateSummary> {
        self.last_hormone_summary
    }

    pub fn last_gating_modulation(&self) -> Option<GatingModulation> {
        self.last_gating_modulation
    }

    pub fn hormone_degraded_total(&self) -> u64 {
        self.hormone_degraded_total
    }

    pub fn last_neuro_summary(&self) -> Option<NeuroStateSummary> {
        self.last_neuro_summary
    }

    pub fn neuro_degraded_total(&self) -> u64 {
        self.neuro_degraded_total
    }

    pub fn coherence_violation_count(&self) -> u64 {
        self.coherence_violation_count
    }

    pub fn set_consolidation_hook_enabled_for_test(&mut self, enabled: bool) {
        self.consolidation_hook_enabled = enabled;
    }

    pub fn set_geist_hook_enabled_for_test(&mut self, enabled: bool) {
        self.geist_hook_enabled = enabled;
    }

    pub fn consolidation_milestones_emitted_total(&self) -> u64 {
        self.consolidation_milestones_emitted_total
    }

    pub fn geist_updates_accepted_total(&self) -> u64 {
        self.geist_updates_accepted_total
    }

    pub fn geist_updates_rejected_total(&self) -> u64 {
        self.geist_updates_rejected_total
    }

    pub fn hook_errors_total(&self) -> (u64, u64) {
        (
            self.consolidation_hook_errors_total,
            self.geist_hook_errors_total,
        )
    }

    pub fn orchestrator_backpressure(&self) -> f32 {
        self.backpressure
    }

    pub fn compute_budget_exceeded_total(&self) -> u64 {
        self.compute_budget_exceeded_total
    }

    pub fn orchestrator_backpressure_active_total(&self) -> u64 {
        self.backpressure_active_total
    }

    pub fn last_compute_milestone(&self) -> Option<&ComputeMilestone> {
        self.last_compute_milestone.as_ref()
    }

    pub fn force_causal_cycle_for_test(&mut self, now_ms: u64) {
        self.cde_state.hyps.clear();
        for (src, dst) in [(1_u16, 2_u16), (2, 3), (3, 1)] {
            self.cde_state.hyps.push(ucf_cde::v0::Hypothesis {
                edge: ucf_cde::v0::Edge { src, dst },
                score: 0.9,
                conf: 0.9,
                last_update_ms: now_ms,
                seen_obs: 0,
                seen_int: 1,
            });
        }
        sync_graph_from_cde_state(&mut self.cde_graph, &self.cde_state);
        self.cde_graph
            .upsert_hypothesis(Edge { from: 1, to: 2 }, now_ms, 0.2);
        self.cde_graph
            .upsert_hypothesis(Edge { from: 2, to: 3 }, now_ms, 0.2);
        self.cde_graph
            .upsert_hypothesis(Edge { from: 3, to: 1 }, now_ms, 0.2);
    }

    pub fn feed_cde_intervention_for_test(
        &mut self,
        now_ms: u64,
        do_set: Vec<(VarId, f32)>,
        measured: Vec<(VarId, f32)>,
    ) -> CdeUpdateKind {
        on_intervention(
            &mut self.cde_state,
            self.cde_cfg,
            Intervention {
                now_ms,
                do_set,
                measured,
            },
        )
    }

    pub fn ingest_and_process<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        ctrl: ControlFrame,
    ) -> Result<DecisionFrame, RuntimeError> {
        self.update_biophys_tick(ctrl.time.tick.get());

        let inputs = NeuromodInputs::baseline();
        self.neuromod_scheduler
            .advance(ctrl.time.tick.get(), &mut self.neuromod_field, inputs);
        self.onn.step_ms(1);
        let snapshot = self.neuromod_field.snapshot();
        let phi = self.iit_phi_snapshot();

        let decision = Pbm::decide(&ctrl, Some(snapshot));
        self.ingest_with_decision_and_snapshot(adapter, ctrl, decision, snapshot, phi)
    }

    pub fn ingest_with_decision<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        ctrl: ControlFrame,
        decision: DecisionFrame,
    ) -> Result<DecisionFrame, RuntimeError> {
        self.update_biophys_tick(ctrl.time.tick.get());

        let inputs = NeuromodInputs::baseline();
        self.neuromod_scheduler
            .advance(ctrl.time.tick.get(), &mut self.neuromod_field, inputs);
        self.onn.step_ms(1);
        let snapshot = self.neuromod_field.snapshot();
        let phi = self.iit_phi_snapshot();
        self.ingest_with_decision_and_snapshot(adapter, ctrl, decision, snapshot, phi)
    }

    fn decide_internal_recursion(&self) -> PolicyDecision {
        PolicyDecision {
            kind: PolicyDecisionKind::InternalRecursion,
            allow: self.policy_internal_recursion_allow,
            max_depth: self
                .policy_internal_recursion_max_depth
                .min(self.sle_cfg.max_depth.max(1)),
        }
    }

    fn update_biophys_tick(&mut self, now_ms: u64) {
        let dt_ms = self
            .last_biophys_tick_ms
            .map(|last| now_ms.saturating_sub(last) as u32)
            .unwrap_or(1)
            .max(1);
        self.last_biophys_tick_ms = Some(now_ms);
        let dt_s = (dt_ms as f32 / 1000.0).max(0.001);

        self.phase_bus.osc_jepa = osc_step(self.phase_bus.osc_jepa, dt_s);
        self.phase_bus.osc_nsr = osc_step(self.phase_bus.osc_nsr, dt_s);
        self.phase_bus.osc_microcircuit = osc_step(self.phase_bus.osc_microcircuit, dt_s);

        (self.phase_bus.osc_microcircuit, self.phase_bus.osc_nsr) = couple_pair(
            self.phase_bus.osc_microcircuit,
            self.phase_bus.osc_nsr,
            dt_s,
            self.phase_cfg,
        );
        (self.phase_bus.osc_nsr, self.phase_bus.osc_jepa) = couple_pair(
            self.phase_bus.osc_nsr,
            self.phase_bus.osc_jepa,
            dt_s,
            self.phase_cfg,
        );

        let attention_level = if self.attention_event {
            1.0
        } else {
            self.last_nsr_frame
                .map(|frame| 1.0 - (f32::from(frame.verified_q) / 255.0))
                .unwrap_or(1.0)
                .clamp(0.0, 1.0)
        };
        let anchors = vec![(OnnNode::Global, self.current_tcf_phase_0_1(now_ms))];
        let onn_out = onn_step(
            &self.onn_cfg,
            &mut self.onn_state,
            &OnnInput {
                now_ms,
                anchors,
                gate: attention_level,
            },
        );
        self.last_onn_frame = Some(OnnFrame {
            now_ms,
            global_phase_q: quantize_unit(onn_out.global_phase_0_1),
            lock_nsr_cde_q: quantize_unit(onn_out.lock_nsr_cde),
            lock_nsr_ssm_q: quantize_unit(onn_out.lock_nsr_ssm),
        });

        let coupling_targets = if self.iit_proxy_state.enforce {
            vec![(OSC_NSR_TCF_ENFORCE, self.tcf_state.global_phase)]
        } else {
            Vec::new()
        };
        let mut tcf_out = tcf_tick(
            &self.tcf_cfg,
            &mut self.tcf_state,
            now_ms,
            &coupling_targets,
        );
        if let Some(forced) = self.forced_mean_lock_for_test {
            tcf_out.mean_lock = forced.clamp(0.0, 1.0);
        }
        self.last_tcf_frame = Some(TcfFrame::from_metrics(
            tcf_out.now_ms,
            tcf_out.global_phase,
            tcf_out.mean_lock,
            tcf_out.jitter,
            tcf_out.phase_spread,
        ));

        let baseline = BiophysField::default();
        self.biophys_field = self
            .biophys_field
            .decay_towards(baseline, dt_s, self.biophys_cfg);

        if now_ms.is_multiple_of(10) {
            self.biophys_field = self.biophys_field.apply_event(
                FieldEvent {
                    kind: FieldEventKind::Reward,
                    magnitude: ucf_biophys::v0::Unit01::new(0.4),
                },
                self.biophys_cfg,
            );
        }

        if now_ms.is_multiple_of(15) {
            self.biophys_field = self.biophys_field.apply_event(
                FieldEvent {
                    kind: FieldEventKind::Stress,
                    magnitude: ucf_biophys::v0::Unit01::new(0.3),
                },
                self.biophys_cfg,
            );
        }

        let stress = if now_ms.is_multiple_of(15) { 0.7 } else { 0.05 };
        self.hpa_state = hpa_step(self.hpa_state, stress, dt_s, self.hpa_cfg);

        let field = self.biophys_field.with_hpa(self.hpa_state);
        let hh = modulate_hh(HhParams::default(), field, self.biophys_mod_cfg);
        let micro = self.microcircuit.step(field, dt_s);

        let _ttfs_zero_phase = ttfs_phase(0, self.spike_codec_cfg);

        let lock_nsr_jepa = phase_lock(self.phase_bus.osc_nsr, self.phase_bus.osc_jepa);
        let lock_micro_nsr = phase_lock(self.phase_bus.osc_microcircuit, self.phase_bus.osc_nsr);

        let spike_rate_hz = micro.spikes.len() as f32 / dt_s;
        let ssm_gate = self
            .last_ssm_frame
            .map(|frame| f32::from(frame.gate_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let nsr_verified_ratio = self
            .last_nsr_frame
            .map(|frame| f32::from(frame.verified_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let archive_append_bytes_q =
            ((self.last_archive_payload_len as f32) / 1024.0).clamp(0.0, 1.0);

        let obs_update = on_observation(
            &mut self.cde_state,
            self.cde_cfg,
            Observation {
                now_ms,
                vars: vec![
                    (1, tcf_out.mean_lock.clamp(0.0, 1.0)),
                    (2, ssm_gate),
                    (3, nsr_verified_ratio),
                    (4, archive_append_bytes_q),
                ],
            },
        );
        let decay_update = tick_decay(&mut self.cde_state, self.cde_cfg, now_ms);
        sync_graph_from_cde_state(&mut self.cde_graph, &self.cde_state);

        let (legacy_verdict, _legacy_verified_ratio) = verify_graph(&self.cde_graph, self.nsr_cfg);
        let top_conf = self
            .cde_state
            .hyps
            .iter()
            .map(|h| h.conf)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        let planned_intent = self
            .last_ssm_frame
            .map(|frame| {
                if frame.gate_q > 128 {
                    "act".to_string()
                } else {
                    "noop".to_string()
                }
            })
            .unwrap_or_else(|| "noop".to_string());
        let claim = Claim {
            intent: planned_intent,
            tool: "mock".to_string(),
            channel: "terminal".to_string(),
            risk: if tcf_out.mean_lock < 0.25 {
                "high".to_string()
            } else {
                "low".to_string()
            },
            audience: "research".to_string(),
        };
        let nsr_result = self.nsr_engine.check(&claim);
        let verified_ratio = if nsr_result.total == 0 {
            0.0
        } else {
            f32::from(nsr_result.satisfied) / f32::from(nsr_result.total)
        };
        let mut nsr_risk = (1.0 - verified_ratio).clamp(0.0, 1.0);
        if let Some(forced) = self.forced_nsr_risk_for_test {
            nsr_risk = forced.clamp(0.0, 1.0);
        }
        let verified_q = (verified_ratio * 255.0).round() as u8;

        let reward = (1.0 - nsr_risk).clamp(0.0, 1.0);
        let stress = nsr_risk.clamp(0.0, 1.0);
        let onn_lock = self
            .last_onn_frame
            .map(|frame| f32::from(frame.lock_nsr_cde_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let mut surprise = self
            .last_ncde_frame
            .map(|frame| f32::from(frame.l2_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        if let Some(forced) = self.forced_surprise_for_test {
            surprise = forced.clamp(0.0, 1.0);
        }
        let safety = onn_lock;
        let pain = ((surprise - 0.5).max(0.0) * 2.0).clamp(0.0, 1.0);
        let observed_brain_rate = self
            .last_digital_brain_frame
            .map(|f| f32::from(f.amyg_spikes.saturating_add(f.pfc_spikes)) / 32.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let mut ess_pressure = (self.ess.len() as f32 / 256.0).clamp(0.0, 1.0);
        if let Some(forced) = self.forced_ess_pressure_for_test {
            ess_pressure = forced.clamp(0.0, 1.0);
        }
        let mut geist_drift = (1.0 - onn_lock) * 0.5;
        if let Some(forced) = self.forced_geist_drift_for_test {
            geist_drift = forced.clamp(0.0, 1.0);
        }
        let snn_event_rate = self.last_spike_rate_0_1.clamp(0.0, 1.0);
        let complexity = (0.5 * snn_event_rate + 0.5 * ess_pressure).clamp(0.0, 1.0);
        let homeo_err = homeostasis_step(
            &self.fep_state.homeo_cfg,
            &mut self.fep_state.homeo_state,
            dt_s,
            snn_event_rate,
            observed_brain_rate,
            ess_pressure,
        );

        let last_compute_summary = self
            .ess
            .get(self.ess.len().saturating_sub(1))
            .and_then(|r| r.compute_summary);
        let compute_risk = last_compute_summary
            .map(|c| c.risk)
            .unwrap_or(nsr_risk)
            .clamp(0.0, 1.0);
        let compute_confidence = last_compute_summary
            .map(|c| c.confidence)
            .unwrap_or(1.0 - compute_risk)
            .clamp(0.0, 1.0);

        let hormone_input = HormoneInput {
            t: now_ms,
            pressure: ess_pressure,
            surprise,
            risk: nsr_risk,
            confidence: compute_confidence,
            coherence: last_compute_summary.and_then(|c| c.coherence),
            instability: last_compute_summary.and_then(|c| c.instability),
            evidence_chain_digest: last_compute_summary
                .and_then(|c| c.compute_chain_digest)
                .unwrap_or([0; 32]),
        };
        let _biophys_span = tracing::info_span!(
            "biophys_runtime.step",
            stress_index = tracing::field::Empty,
            evidence_digest_prefix =
                tracing::field::display(hex::encode(&hormone_input.evidence_chain_digest[..4]))
        )
        .entered();
        let biophys_start = Instant::now();
        let mut hormone_out = hormone_step(&self.hormone_cfg, self.hormone_state, &hormone_input);
        metrics::histogram!("ucf_biophys_runtime_ode_micros")
            .record(biophys_start.elapsed().as_micros() as f64);
        if self.emergency_active() {
            hormone_out.state.cortisol = 1.0;
            hormone_out.state.dopamine = 0.0;
            hormone_out.state.norepinephrine = 1.0;
            hormone_out.state.serotonin = 0.0;
            hormone_out.state.acetylcholine = 0.0;
            hormone_out.state.drive = 0.0;
            hormone_out.summary.cortisol = 1.0;
            hormone_out.summary.dopamine = 0.0;
            hormone_out.summary.norepinephrine = 1.0;
            hormone_out.summary.serotonin = 0.0;
            hormone_out.summary.acetylcholine = 0.0;
            hormone_out.summary.drive = 0.0;
            hormone_out.summary.stress_index = 1.0;
            hormone_out.modulation.stress_gate = 0.1;
            hormone_out.modulation.plasticity_gate = 0.0;
            hormone_out.modulation.attention_gain = 0.2;
        }
        tracing::Span::current().record("stress_index", hormone_out.summary.stress_index);
        self.hormone_state = hormone_out.state;
        self.last_hormone_summary = Some(hormone_out.summary);
        self.last_gating_modulation = Some(hormone_out.modulation);
        if hormone_out.degraded {
            self.hormone_degraded_total = self.hormone_degraded_total.saturating_add(1);
            metrics::counter!("ucf_hormone_degraded_total").increment(1);
        }
        metrics::gauge!("ucf_hormone_cortisol").set(f64::from(hormone_out.summary.cortisol));
        metrics::gauge!("ucf_hormone_drive").set(f64::from(hormone_out.summary.drive));
        metrics::gauge!("ucf_hormone_stress_index")
            .set(f64::from(hormone_out.summary.stress_index));

        let neuro_input = NeuroInput {
            t: now_ms,
            pressure: ess_pressure,
            surprise,
            risk: nsr_risk,
            confidence: compute_confidence,
            cortisol: hormone_out.summary.cortisol,
            drive: hormone_out.summary.drive,
            evidence_chain_digest: hormone_input.evidence_chain_digest,
        };
        let neuro_out = {
            let _neuro_span = tracing::info_span!("biophys_neuro.step", t = now_ms).entered();
            neuro_step(&self.neuro_cfg, self.neuro_state, &neuro_input)
        };
        self.neuro_state = neuro_out.state;
        self.last_neuro_summary = Some(neuro_out.summary);
        self.last_neuro_modulation = Some(neuro_out.modulation);
        self.last_neuro_spikes = Some(neuro_out.spikes.clone());
        self.last_neuro_degraded = neuro_out.degraded;
        if neuro_out.degraded {
            self.neuro_degraded_total = self.neuro_degraded_total.saturating_add(1);
            metrics::counter!("ucf_neuro_degraded_total").increment(1);
        }
        metrics::gauge!("ucf_neuro_arousal").set(f64::from(neuro_out.summary.arousal));
        metrics::gauge!("ucf_neuro_attention_gain")
            .set(f64::from(neuro_out.summary.attention_gain));
        metrics::gauge!("ucf_neuro_excitability").set(f64::from(neuro_out.summary.excitability));
        metrics::gauge!("ucf_neuro_spike_rate").set(f64::from(neuro_out.summary.spike_rate));

        let fep_in = FepInputs {
            now_ms,
            ebm_energy_mean_topk_q: 0,
            dt_s,
            surprise,
            complexity,
            policy_risk: nsr_risk,
            compute_risk,
            compute_confidence: (compute_confidence * neuro_out.summary.attention_gain)
                .clamp(0.0, 1.0),
            onn_lock,
            snn_event_rate,
            ess_pressure,
            ssm_pressure: ssm_gate,
            geist_drift,
            hormone_risk_penalty_scale: hormone_out.modulation.risk_penalty_scale,
            hormone_exploration_bias_delta: (hormone_out.modulation.exploration_bias_delta
                + (neuro_out.summary.arousal - 0.5) * 0.2)
                .clamp(-0.5, 0.5),
            hormone_attention_gain: (hormone_out.modulation.attention_gain
                * (0.5 + 0.5 * neuro_out.summary.attention_gain))
                .clamp(0.0, 3.0),
            hormone_action_threshold_delta: (hormone_out.modulation.action_threshold_delta
                + 0.1 * neuro_out.summary.arousal
                - 0.05 * neuro_out.summary.excitability)
                .clamp(-0.5, 0.5),
            nsr_risk: None,
            nsr_confidence: None,
        };
        self.fep_risk_penalty_applied_total = self.fep_risk_penalty_applied_total.saturating_add(1);
        let mut fep_out = fep_step(&self.fep_state.cfg, &fep_in);

        chemistry_step(
            dt_s,
            &mut self.digital_brain.chem,
            reward,
            stress,
            safety,
            pain,
            homeo_err,
            &self.digital_brain.chem_cfg,
        );

        let amyg_drive = stress * 1.2 + (1.0 - safety) * 0.6;
        let pfc_drive = safety * 0.8 + reward * 0.4 - stress * 0.3;
        let (amyg_spikes, amyg_v) = region_step(
            now_ms,
            dt_ms as f32,
            &mut self.digital_brain.amygdala,
            &self.digital_brain.chem,
            amyg_drive,
        );
        let (pfc_spikes, pfc_v) = region_step(
            now_ms,
            dt_ms as f32,
            &mut self.digital_brain.pfc,
            &self.digital_brain.chem,
            pfc_drive,
        );

        self.last_chem_frame = Some(ChemFrame {
            now_ms,
            dopa_q: quantize_hormone(self.digital_brain.chem.dopa),
            s5ht_q: quantize_hormone(self.digital_brain.chem.serotonin),
            oxy_q: quantize_hormone(self.digital_brain.chem.oxytocin),
            end_q: quantize_hormone(self.digital_brain.chem.endorphin),
        });
        self.last_digital_brain_frame = Some(DigitalBrainFrame {
            now_ms,
            amyg_spikes: amyg_spikes.min(u16::MAX as u32) as u16,
            pfc_spikes: pfc_spikes.min(u16::MAX as u32) as u16,
            amyg_avg_v_q: quantize_avg_v_mv(amyg_v),
            pfc_avg_v_q: quantize_avg_v_mv(pfc_v),
        });
        self.emotion.valence = (self.digital_brain.chem.dopa - stress).clamp(-1.0, 1.0);
        self.emotion.arousal = (stress - self.digital_brain.chem.serotonin * 0.5).clamp(-1.0, 1.0);

        let snapshot = CoherenceSnapshot {
            surprise,
            ess_pressure,
            ssm_pressure: ssm_gate,
            onn_lock,
            policy_risk: nsr_risk,
            geist_drift,
            attention_gain: fep_out.attention_gain,
            learn_gate: fep_out.learn_gate,
            memory_priority: fep_out.memory_priority,
            action_inhibit: fep_out.action_inhibit,
            homeo_err,
            chem_dopa: self.digital_brain.chem.dopa,
            chem_5ht: self.digital_brain.chem.serotonin,
            chem_oxy: self.digital_brain.chem.oxytocin,
            chem_end: self.digital_brain.chem.endorphin,
            brain_amyg_spikes: amyg_spikes as f32,
            brain_pfc_spikes: pfc_spikes as f32,
        };
        self.coherence_violation_flag = false;
        if check_coherence_invariants(&self.fep_state.coh_cfg, &snapshot).is_err() {
            self.coherence_violation_count = self.coherence_violation_count.saturating_add(1);
            self.coherence_violation_flag = true;
            fep_out.action_inhibit = (fep_out.action_inhibit + 0.1).clamp(0.0, 1.0);
        }
        metrics::gauge!("ucf_fep_free_energy_proxy_q").set(f64::from(fep_out.free_energy_proxy_q));
        self.last_fep_outputs = Some(fep_out.clone());
        self.last_fep_frame = Some(FepFrame {
            now_ms,
            attention_q: quantize_unit(fep_out.attention_gain),
            learn_gate_q: quantize_unit(fep_out.learn_gate),
            memprio_q: quantize_unit(fep_out.memory_priority),
            inhibit_q: quantize_unit(fep_out.action_inhibit),
            confidence_q: quantize_unit(fep_out.confidence),
            homeo_err_q: quantize_unit(homeo_err),
        });
        let coupling = (0.25 * fep_out.attention_gain
            + 0.25 * fep_out.memory_priority
            + 0.25 * fep_out.action_inhibit
            + 0.25 * (1.0 - homeo_err.clamp(0.0, 1.0)))
        .clamp(0.0, 1.0);
        self.last_coherence_frame = Some(CoherenceFrame {
            now_ms,
            coupling_q: quantize_unit(coupling),
            drift_q: quantize_unit(geist_drift),
            risk_q: quantize_unit(nsr_risk),
            lock_q: quantize_unit(onn_lock),
        });

        let mut integration = compute_integration(
            IITInputs {
                lock_nsr_jepa,
                lock_micro_nsr,
                spike_rate_hz,
            },
            self.iit_cfg,
        );

        if legacy_verdict == VerifyVerdict::Verified && verified_ratio >= 0.85 {
            integration = (integration + 0.05).clamp(0.0, 1.0);
        }

        let alpha = self.iit_cfg.ema_alpha.clamp(0.0, 1.0);
        self.iit_state.integration_ema =
            (1.0 - alpha) * self.iit_state.integration_ema + alpha * integration;

        let mut coherence_state = classify(self.iit_state.integration_ema, self.iit_cfg);
        if nsr_result.verdict == Verdict::Block {
            coherence_state = CoherenceState::Fragmenting;
        }

        apply_coherence_feedback(
            &mut self.biophys_field,
            self.iit_state.integration_ema,
            coherence_state,
        );
        let top_conf_from_cde = top_conf.clamp(0.0, 1.0);
        let archive_append_bytes_q =
            ((self.last_archive_payload_len as f32) / 1024.0).clamp(0.0, 1.0);

        let spike_novelty = self
            .last_spike_frames
            .iter()
            .filter(|spike| spike.kind == 1)
            .map(|spike| f32::from(spike.strength_q) / 255.0)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let sle_sig = SleSignals {
            now_ms,
            nsr_risk,
            cde_conf: top_conf_from_cde,
            phi_proxy: self.iit_proxy_state.phi_proxy.clamp(0.0, 1.0),
            spike_novelty,
            attention_event: self.attention_event,
        };
        let sle_out = sle_step(&self.sle_cfg, &mut self.sle_state, &sle_sig);
        let policy = self.decide_internal_recursion();
        let mut meta_weight = 0.0_f32;
        let mut meta_reason_norm = 0.0_f32;
        let mut sle_frame = SleFrame {
            now_ms,
            fired: 0,
            reason: 0,
            depth: 0,
            weight_q: 0,
            tok_n: 0,
        };
        self.last_meta_input = None;
        if let Some(meta) = sle_out {
            sle_frame.fired = 1;
            let reason_id = match meta.reason {
                SleReason::HighRisk => 1,
                SleReason::LowIntegration => 2,
                SleReason::NoveltyShock => 3,
                SleReason::Conflict => 4,
            };
            sle_frame.reason = reason_id;
            sle_frame.tok_n = meta.tokens.len().min(255) as u8;
            let mut depth = meta.depth.min(policy.max_depth.max(1));
            if depth == 0 {
                depth = 1;
            }
            sle_frame.depth = depth;
            if policy.allow {
                let weight = meta.weight.clamp(0.0, 1.0);
                meta_weight = weight;
                meta_reason_norm = (reason_id as f32 / 4.0).clamp(0.0, 1.0);
                sle_frame.weight_q = (weight * 255.0).round() as u8;
                self.last_meta_input = Some(MetaInput {
                    tokens: meta.tokens,
                    weight,
                    depth,
                    reason: meta.reason,
                });
            }
        }
        self.last_sle_frame = Some(sle_frame);

        let input = [
            tcf_out.mean_lock.clamp(0.0, 1.0),
            verified_ratio.clamp(0.0, 1.0),
            top_conf_from_cde,
            archive_append_bytes_q,
            if self.attention_event { 1.0 } else { 0.0 },
            meta_weight,
            meta_reason_norm,
            0.0,
        ];
        let gate = tcf_out.mean_lock.clamp(0.0, 1.0);
        let mut ssm_out = ssm_step(&self.ssm_cfg, &mut self.ssm_state, now_ms, &input, gate);
        self.working_context.ssm_y = ssm_out.y.clone();
        let ssm_energy = if ssm_out.y.is_empty() {
            0.0
        } else {
            ssm_out.y.iter().map(|v| v.abs()).sum::<f32>() / (ssm_out.y.len() as f32)
        };

        let tcf_phase_bin = self
            .last_tcf_frame
            .map(|frame| frame.phase_bin)
            .unwrap_or_else(|| phase_bin(tcf_out.global_phase, 255));

        if ssm_out.gate > 0.6 {
            let strength = ((ssm_out.gate - 0.6) / 0.4).clamp(0.0, 1.0);
            self.spike_bus.push(BusSpike {
                now_ms,
                kind: BusSpikeKind::Novelty,
                chan: OSC_SSM,
                phase: tcf_phase_bin,
                strength,
                ttfs_us: encode_ttfs_us(strength),
            });
        }

        if matches!(obs_update, CdeUpdateKind::Updated { .. }) {
            let strength = top_conf.clamp(0.0, 1.0);
            self.spike_bus.push(BusSpike {
                now_ms,
                kind: BusSpikeKind::CausalHit,
                chan: OSC_CDE,
                phase: tcf_phase_bin,
                strength,
                ttfs_us: encode_ttfs_us(strength),
            });
        }

        if matches!(nsr_result.verdict, Verdict::Allow | Verdict::Block) {
            let strength = verified_ratio.clamp(0.0, 1.0);
            self.spike_bus.push(BusSpike {
                now_ms,
                kind: BusSpikeKind::Verify,
                chan: OSC_NSR,
                phase: tcf_phase_bin,
                strength,
                ttfs_us: encode_ttfs_us(strength),
            });
        }

        if tcf_out.mean_lock > 0.7 {
            let strength = tcf_out.mean_lock.clamp(0.0, 1.0);
            self.spike_bus.push(BusSpike {
                now_ms,
                kind: BusSpikeKind::AttentionShift,
                chan: OSC_COHERENCE,
                phase: tcf_phase_bin,
                strength,
                ttfs_us: encode_ttfs_us(strength),
            });
        }

        let tick_spikes = filter_phase_locked(
            &PhaseLockCfg {
                max_dist: 24,
                attenuate: true,
            },
            tcf_phase_bin,
            &self.spike_bus.drain(),
        );

        self.attention_event = tick_spikes.iter().any(|spike| {
            spike.strength > 0.6
                && matches!(
                    spike.kind,
                    BusSpikeKind::Novelty | BusSpikeKind::AttentionShift
                )
        });

        self.last_spike_frames = tick_spikes
            .iter()
            .map(|spike| ucf_frames::v1::SpikeFrame {
                now_ms: spike.now_ms,
                kind: match spike.kind {
                    BusSpikeKind::Novelty => 1,
                    BusSpikeKind::Verify => 2,
                    BusSpikeKind::CausalHit => 3,
                    BusSpikeKind::MemoryMark => 4,
                    BusSpikeKind::AttentionShift => 5,
                },
                chan: spike.chan,
                phase: spike.phase,
                strength_q: (spike.strength.clamp(0.0, 1.0) * 255.0).round() as u8,
                ttfs_q: ((spike.ttfs_us as f32 / 5000.0).clamp(0.0, 1.0) * 255.0).round() as u8,
            })
            .collect();

        for spike in &tick_spikes {
            let mapped = match spike.kind {
                BusSpikeKind::Novelty => SpikeKind::Feature,
                BusSpikeKind::Verify => SpikeKind::Verify,
                BusSpikeKind::CausalHit => SpikeKind::Causal,
                BusSpikeKind::MemoryMark => continue,
                BusSpikeKind::AttentionShift => SpikeKind::Attention,
            };
            let delivery = self.make_spike(
                spike.now_ms,
                spike.chan.into(),
                mapped,
                spike.phase,
                spike.strength,
                Some(((spike.ttfs_us as f32 / 5000.0).clamp(0.0, 1.0) * 255.0).round() as u8),
            );
            self.event_bus.push(delivery);
        }

        let onn_phase_0_1 = self
            .last_onn_frame
            .map(|frame| f32::from(frame.global_phase_q) / 255.0)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let cde_candidates: Vec<(u32, f32)> = self
            .cde_state
            .hyps
            .iter()
            .enumerate()
            .map(|(idx, hyp)| (1000 + (idx as u32), hyp.conf.clamp(0.0, 1.0)))
            .collect();
        let nsr_candidates = if nsr_risk > 0.0 {
            vec![(2000_u32, nsr_risk.clamp(0.0, 1.0))]
        } else {
            Vec::new()
        };
        let ssm_candidates = if ssm_out.gate > 0.0 {
            vec![(3000_u32, ssm_out.gate.clamp(0.0, 1.0))]
        } else {
            Vec::new()
        };

        let snn_from_cde = snn_emit(
            &self.snn_cfg,
            now_ms,
            onn_phase_0_1,
            SpikeSrc::Cde,
            &cde_candidates,
        );
        let snn_from_nsr = snn_emit(
            &self.snn_cfg,
            now_ms,
            onn_phase_0_1,
            SpikeSrc::Nsr,
            &nsr_candidates,
        );
        let snn_from_ssm = snn_emit(
            &self.snn_cfg,
            now_ms,
            onn_phase_0_1,
            SpikeSrc::Ssm,
            &ssm_candidates,
        );

        let fired = snn_from_cde
            .fired_count
            .saturating_add(snn_from_nsr.fired_count)
            .saturating_add(snn_from_ssm.fired_count);
        let suppressed = snn_from_cde
            .suppressed_count
            .saturating_add(snn_from_nsr.suppressed_count)
            .saturating_add(snn_from_ssm.suppressed_count);
        let max_amp_q = snn_from_cde
            .emitted
            .iter()
            .chain(snn_from_nsr.emitted.iter())
            .chain(snn_from_ssm.emitted.iter())
            .map(|event| event.amp_q)
            .max()
            .unwrap_or(0);

        self.last_snn_frame = Some(SnnFrame {
            now_ms,
            fired,
            suppressed,
            max_amp_q,
        });

        self.last_biophys_frame = Some(BiophysFrame {
            now_ms,
            field: ucf_biophys::v0::summarize(field),
            hh_params: BiophysHhParams {
                g_na: hh.g_na,
                g_k: hh.g_k,
                g_l: hh.g_l,
                threshold_shift_mv: hh.threshold_shift_mv,
                max_firing_hz: hh.max_firing_hz,
            },
            hpa_cortisol: self.hpa_state.cortisol,
        });
        self.last_microcircuit_frame = Some(MicrocircuitFrame {
            now_ms,
            n: self.microcircuit.neurons.len() as u32,
            spike_count: micro.spikes.len() as u32,
            avg_v: micro.avg_v,
        });
        self.last_phase_frame = Some(PhaseFrame {
            now_ms,
            jepa_phase: self.phase_bus.osc_jepa.phase,
            nsr_phase: self.phase_bus.osc_nsr.phase,
            micro_phase: self.phase_bus.osc_microcircuit.phase,
            lock_nsr_jepa,
            lock_micro_nsr,
        });
        let spike_rate = if self.snn_cfg.max_events_per_tick == 0 {
            0.0
        } else {
            f32::from(fired) / (self.snn_cfg.max_events_per_tick as f32)
        }
        .clamp(0.0, 1.0);
        self.last_spike_rate_0_1 = spike_rate;
        let tcf_phase_norm = self.current_tcf_phase_0_1(now_ms);
        let ncde_inp = NcdeInput {
            u: vec![
                (1.0 - nsr_risk).clamp(0.0, 1.0),
                top_conf.clamp(0.0, 1.0),
                self.iit_proxy_state.phi_proxy.clamp(0.0, 1.0),
                ssm_out.gate.clamp(0.0, 1.0),
                spike_rate.clamp(0.0, 1.0),
                meta_weight.clamp(0.0, 1.0),
            ],
            phase: tcf_phase_norm,
        };
        self.last_ncde_spike_u4 = ncde_inp.u[4];
        let ncde_tick = ncde_step(&self.ncde_cfg, &mut self.ncde_state, now_ms, &ncde_inp);
        self.last_ncde_frame = Some(NcdeFrame {
            now_ms: ncde_tick.now_ms,
            l2_q: ncde_tick.l2_q,
            phase_q: ncde_tick.phase_q,
        });

        let iit_tick = iit_push_and_eval(
            &self.iit_proxy_cfg,
            &mut self.iit_proxy_state,
            IitSample {
                now_ms,
                tcf_lock: tcf_out.mean_lock.clamp(0.0, 1.0),
                ssm_gate: ssm_out.gate.clamp(0.0, 1.0),
                nsr_risk,
                cde_conf: top_conf.clamp(0.0, 1.0),
                spike_rate,
            },
        );
        if iit_tick.enforce {
            let _nsr_risk_floor = nsr_risk.max(0.8);
            ssm_out.gate = (ssm_out.gate * 0.5).clamp(0.0, 1.0);
        }

        self.ssm_gate = ssm_out.gate;
        self.last_ssm_frame = Some(SsmFrame {
            now_ms,
            gate_q: quantize_unit(ssm_out.gate),
            energy_q: quantize_unit(ssm_energy),
        });

        self.last_iit_frame = Some(IitFrame {
            now_ms,
            phi_q: quantize_unit(iit_tick.phi_proxy),
            coh_q: quantize_unit(iit_tick.coherence),
            flow_q: quantize_unit(iit_tick.flow),
            enforce: u8::from(iit_tick.enforce),
        });
        let changed = match obs_update {
            CdeUpdateKind::Updated { changed, .. } => changed as u16,
            _ => 0,
        };
        let pruned = match decay_update {
            CdeUpdateKind::Pruned { pruned } => pruned as u16,
            _ => 0,
        };
        self.last_cde_frame = Some(CdeFrame {
            now_ms,
            hyps: self.cde_state.hyps.len() as u16,
            changed,
            pruned,
            top_conf_q: (top_conf.clamp(0.0, 1.0) * 255.0).round() as u8,
        });
        self.last_nsr_frame = Some(NsrFrame {
            now_ms,
            verdict: match nsr_result.verdict {
                Verdict::Allow => 1,
                Verdict::Block => 2,
                Verdict::Unknown => 0,
            },
            satisfied: nsr_result.satisfied,
            total: nsr_result.total,
            verified_q,
        });

        let payload =
            tick_summary_payload(now_ms, coherence_state, tcf_out.mean_lock, ssm_out.gate);
        let flushed = match self.archive.cfg.flush {
            FlushPolicy::EveryAppend => true,
            FlushPolicy::IntervalMs(interval) => {
                now_ms.saturating_sub(self.archive.last_flush_ms) >= interval
            }
            FlushPolicy::Manual => false,
        };

        let append = self
            .archive
            .append(now_ms, &payload)
            .expect("append archive tick summary");
        self.last_archive_payload_len = payload.len();
        self.last_archive_append_frame = Some(ArchiveAppendFrame {
            now_ms,
            seq: append.seq,
            bytes: append.bytes as u32,
            flushed,
        });
    }

    fn current_tcf_phase_0_1(&self, now_ms: u64) -> f32 {
        self.last_onn_frame
            .map(|frame| f32::from(frame.global_phase_q) / 255.0)
            .or_else(|| {
                self.last_tcf_frame
                    .map(|frame| f32::from(frame.phase_bin) / 255.0)
            })
            // Fallback if no TCF frame is available yet: deterministic sawtooth over 1s.
            .unwrap_or_else(|| ((now_ms % 1_000) as f32) / 1_000.0)
            .clamp(0.0, 1.0)
    }

    fn make_spike(
        &mut self,
        now_ms: u64,
        src: OscId,
        kind: SpikeKind,
        phase_bin: u8,
        magnitude: f32,
        ttfs_override: Option<u8>,
    ) -> SnnSpikeEvent {
        self.spike_seq = self.spike_seq.wrapping_add(1);
        SnnSpikeEvent {
            now_ms,
            src,
            kind,
            spike_id: self.spike_seq,
            phase_bin,
            ttfs_code: ttfs_override.unwrap_or_else(|| ttfs_from_strength(magnitude)),
            magnitude,
        }
    }

    fn deterministic_features(&mut self) -> [FeatureEvent; 3] {
        let k = self.snn_tick_counter as f32;
        self.snn_tick_counter = self.snn_tick_counter.wrapping_add(1);

        [
            FeatureEvent {
                chan: 10,
                intensity: 0.9,
                novelty: 0.85,
            },
            FeatureEvent {
                chan: 11,
                intensity: (k % 5.0) / 4.0,
                novelty: 0.15,
            },
            FeatureEvent {
                chan: 12,
                intensity: ((k + 1.0) % 4.0) / 3.0,
                novelty: 0.65,
            },
        ]
    }

    fn iit_phi_snapshot(&self) -> ucf_frames::v1::PhiProxySnapshot {
        ucf_frames::v1::PhiProxySnapshot {
            phi: self.iit_proxy_state.phi_proxy,
            coherence_mean: self.iit_proxy_state.coherence,
            coherence_min: self.iit_proxy_state.flow,
            n_pairs: self.iit_proxy_state.filled as u16,
        }
    }

    fn emit_snn_signals<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        now_ms: u64,
    ) -> Result<(), RuntimeError> {
        let phase = self.onn.phase(MOD_PBM);
        let features = self.deterministic_features();
        let snn_spikes = encode(now_ms, phase, self.snn_encode_cfg, &features);
        self.last_snn_spike_count = snn_spikes.len();

        let brainbus_spikes = to_brainbus(&snn_spikes);
        adapter.emit_brain_spikes(brainbus_spikes)?;
        let _ = adapter.take_brain_spike_meta();
        Ok(())
    }

    fn ingest_with_decision_and_snapshot<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        ctrl: ControlFrame,
        decision: DecisionFrame,
        snapshot: NeuromodulatorSnapshot,
        phi: ucf_frames::v1::PhiProxySnapshot,
    ) -> Result<DecisionFrame, RuntimeError> {
        self.emit_snn_signals(adapter, ctrl.time.tick.get())?;

        let mut decision = decision;

        let compute_input = compute_input_from_control(&ctrl);
        let mut compute_budget = self.compute_budget;
        if self.emergency_active() {
            compute_budget.governor_tier = 3;
        }
        self.maybe_persist_compute_budget_window(ctrl.time, ctrl.corr)?;
        self.compute_economy
            .begin_tick(compute_budget.governor_tier, self.emergency_active());
        self.compute_tier_sum = self
            .compute_tier_sum
            .saturating_add(u64::from(compute_budget.governor_tier.min(3)));
        self.compute_tier_count = self.compute_tier_count.saturating_add(1);
        self.compute_tier_max = self
            .compute_tier_max
            .max(compute_budget.governor_tier.min(3));
        if let Err(v) = self.compute_economy.try_charge(
            BudgetPool::Primary,
            ComputeStage::Governor,
            self.compute_economy
                .schedule
                .stage_cost(ComputeStage::Governor, 1),
            ctrl.time.tick.get(),
        ) {
            self.persist_compute_budget_violation(ctrl.time, ctrl.corr, &v)?;
        }

        let mut compute_signals = if let Err(v) = self.compute_economy.try_charge(
            BudgetPool::Primary,
            ComputeStage::Jepa,
            self.compute_economy
                .schedule
                .stage_cost(ComputeStage::Jepa, 64),
            ctrl.time.tick.get(),
        ) {
            self.persist_compute_budget_violation(ctrl.time, ctrl.corr, &v)?;
            let mut unavailable = ucf_compute::ComputeSignals::unavailable(
                &compute_input,
                compute_budget,
                self.compute_backend.name(),
            );
            unavailable.budget_exceeded_stage = Some("compute_tokens/jepa");
            unavailable
        } else {
            match self.compute_backend.compute(&compute_input, compute_budget) {
                Ok(signals) => signals,
                Err(_) => ucf_compute::ComputeSignals::unavailable(
                    &compute_input,
                    compute_budget,
                    self.compute_backend.name(),
                ),
            }
        };
        if ucf_compute::validate_risk_signal(&compute_signals.risk_signal).is_err() {
            compute_signals = ucf_compute::ComputeSignals::unavailable(
                &compute_input,
                compute_budget,
                self.compute_backend.name(),
            );
        }
        let compute_summary =
            compute_signals.summary(canonical_backend_label(self.compute_backend.name()));
        for (stage, dim) in [
            (ComputeStage::Sae, 128_u32),
            (ComputeStage::Ssm, 64_u32),
            (ComputeStage::Lfm, 64_u32),
        ] {
            if let Err(v) = self.compute_economy.try_charge(
                BudgetPool::Primary,
                stage,
                self.compute_economy.schedule.stage_cost(stage, dim),
                ctrl.time.tick.get(),
            ) {
                self.persist_compute_budget_violation(ctrl.time, ctrl.corr, &v)?;
            }
        }
        let _stability_snapshot =
            self.update_emergency_state(ctrl.time, ctrl.corr, &compute_summary)?;
        let emergency_active = self.emergency_active();
        if emergency_active {
            if let Some(rec) = compute_signals.plasticity_record.as_mut() {
                rec.enabled = false;
                rec.emergency_disabled = true;
            }
        }
        let nsr_v0_assessment = NsrDatalogLiteEngine::default()
            .assess(
                &NsrContext {
                    risk: compute_summary.risk,
                    confidence: compute_summary.confidence,
                    coherence: decision.compute_summary.and_then(|s| s.coherence),
                    instability: decision.compute_summary.and_then(|s| s.instability),
                    pressure: Some(compute_summary.pressure),
                    surprise: Some(compute_summary.surprise),
                    cortisol: self.last_hormone_summary.map(|h| h.cortisol),
                    arousal: self.last_neuro_summary.map(|n| n.arousal),
                    has_capability_token: !self.tool_gate.capabilities.tokens.is_empty(),
                    compute_degraded_ratio: Some(if compute_summary.risk_quality == 2 {
                        1.0
                    } else {
                        0.0
                    }),
                },
                &DecisionIntentSummary {
                    action_type: ActionType::ToolUse,
                    tool_kinds: vec![CapabilityKind::NetHttp],
                    target_domain_hashes: vec![
                        blake3::hash(compute_summary.backend.as_bytes()).as_bytes()[0] as u64,
                    ],
                    target_path_hashes: vec![
                        blake3::hash(compute_summary.backend_profile.as_bytes()).as_bytes()[0]
                            as u64,
                    ],
                    output_class: if compute_summary.risk > 0.75 {
                        OutputClass::ExecIntent
                    } else {
                        OutputClass::SafeText
                    },
                },
                &[PolicyTag::Network],
                NsrBudget::default(),
            )
            .unwrap_or_else(|_| ucf_nsr::fallback_assessment_fail_open());
        let quality_idx = usize::from(compute_summary.risk_quality.min(2));
        self.risk_quality_counts[quality_idx] =
            self.risk_quality_counts[quality_idx].saturating_add(1);
        if compute_summary.budget_exceeded_stage.is_some() {
            self.compute_budget_exceeded_total =
                self.compute_budget_exceeded_total.saturating_add(1);
        }
        let quality_penalty = if compute_summary.risk_quality == 0 {
            0.0
        } else {
            0.2
        };
        let bp_input = (compute_summary.pressure + quality_penalty).clamp(0.0, 1.0);
        self.backpressure = (0.6 * self.backpressure + 0.4 * bp_input).clamp(0.0, 1.0);
        if compute_summary.budget_exceeded_stage.is_some() {
            self.backpressure = self.backpressure.max(0.85);
        }
        if self.backpressure > 0.8 {
            self.backpressure_active_total = self.backpressure_active_total.saturating_add(1);
        }
        let primary_remaining = self.compute_economy.remaining(BudgetPool::Primary);
        let shadow_remaining = self.compute_economy.remaining(BudgetPool::Shadow);
        metrics::gauge!("ucf_compute_tokens_remaining", "pool" => "primary")
            .set(primary_remaining as f64);
        metrics::gauge!("ucf_compute_tokens_remaining", "pool" => "shadow")
            .set(shadow_remaining as f64);
        metrics::counter!("ucf_compute_tokens_spent_total", "stage" => "governor").increment(0);
        if primary_remaining < 300 {
            self.backpressure = self.backpressure.max(0.9);
        }
        decision = decision.with_compute_summary(ucf_frames::v1::ComputeSignalsSummary {
            backend: compute_summary.backend,
            surprise: compute_summary.surprise,
            pressure: compute_summary.pressure,
            risk: compute_summary.risk,
            confidence: compute_summary.confidence,
            surprise_q: compute_summary.surprise_q,
            pressure_q: compute_summary.pressure_q,
            risk_q: compute_summary.risk_q,
            confidence_q: compute_summary.confidence_q,
            spike_count: compute_summary.spike_count,
            spikes_digest: compute_summary.spikes_digest,
            sparsity: compute_summary.sparsity,
            energy: compute_summary.energy,
            ssm_readout: compute_summary.ssm_readout,
            ssm_digest: compute_summary.ssm_digest,
            world_digest: compute_summary.world_digest,
            risk_quality: Some(compute_summary.risk_quality),
            evidence_context_digest: Some(compute_summary.evidence_context_digest),
            evidence_world_digest: compute_summary.evidence_world_digest,
            evidence_spikes_digest: compute_summary.evidence_spikes_digest,
            evidence_ssm_digest: compute_summary.evidence_ssm_digest,
            evidence_lfm_digest: compute_summary.evidence_lfm_digest,
            backend_profile: Some(compute_summary.backend_profile),
            backend_pack_id: Some(compute_summary.backend_pack_id),
            fixtures_digest: Some(compute_summary.fixtures_digest),
            llm_backend: Some(compute_summary.llm_backend),
            world_backend: Some(compute_summary.world_backend),
            sae_backend: Some(compute_summary.sae_backend),
            ssm_backend: Some(compute_summary.ssm_backend),
            lfm_backend: Some(compute_summary.lfm_backend),
            lfm_uncertainty: compute_summary.lfm_uncertainty,
            lfm_stability: compute_summary.lfm_stability,
            lfm_uncertainty_q: compute_summary.lfm_uncertainty_q,
            lfm_stability_q: compute_summary.lfm_stability_q,
            lfm_state_norm: compute_summary.lfm_state_norm,
            lfm_deriv_norm: compute_summary.lfm_deriv_norm,
            lfm_saturation_ratio: compute_summary.lfm_saturation_ratio,
            lfm_nan_inf_detected: Some(compute_summary.lfm_nan_inf_detected),
            lfm_digest: compute_summary.lfm_digest,
            budget_profile_id: Some(compute_summary.budget_profile_id),
            seed: Some(compute_summary.seed),
            risk_contract_version: Some(compute_summary.risk_contract_version),
            compute_schema_version: Some(compute_summary.compute_schema_version),
            compute_chain_digest: Some(compute_summary.compute_chain_digest),
            compute_code_version: Some(compute_summary.compute_code_version),
            budget_exceeded_stage: compute_summary.budget_exceeded_stage,
            contract_version: Some(compute_summary.contract_version),
            backend_id: Some(compute_summary.backend_id),
            validation_status: Some(compute_summary.validation_status as u8),
            violation_reason_mask: Some(compute_summary.violation_reason_mask),
            lfm_quality: compute_summary.lfm_quality.map(|q| q as u8),
            coherence: None,
            instability: None,
            coherence_q: None,
            instability_q: None,
            phi_proxy: None,
            coherence_digest: None,
            iit_coherence_q: None,
            iit_incoherence_q: None,
            iit_reason_codes: None,
            stage_allow_mask: None,
            free_energy_proxy_q: self
                .last_fep_outputs
                .as_ref()
                .map(|f| f.free_energy_proxy_q),
            ebm_energy_mean_topk_q: self
                .last_fep_outputs
                .as_ref()
                .map(|f| f.ebm_energy_mean_topk_q),
            ebm_w_q: self.last_fep_outputs.as_ref().map(|f| f.w_ebm_q),
            fep_coupling_version: self.last_fep_outputs.as_ref().map(|f| f.coupling_version),
        });

        let (
            routing,
            windows,
            schedule,
            coherence_metrics,
            iit_monitor,
            phase_window,
            coherence_gate,
        ) = self.coherence_runtime.tick(
            &compute_signals.spikes,
            TickInput {
                t: ctrl.time.tick.get(),
                source_digest: compute_summary.spikes_digest,
                pressure: compute_summary.pressure,
                surprise: compute_summary.surprise,
                risk: compute_summary.risk,
                confidence: compute_summary.confidence,
                budget_limit: 8,
            },
        );
        let _ = (routing, windows, schedule);

        self.ess.append(ExperienceRecord::note(
            self.ids.next(),
            ctrl.time,
            ctrl.corr,
            format!(
                "iit_monitor:coh_q={} incoh_q={} digest={} reasons={:?}",
                iit_monitor.coherence_q.raw(),
                iit_monitor.incoherence_q.raw(),
                hex::encode(&iit_monitor.monitor_digest[..8]),
                iit_monitor.reason_codes
            ),
        ))?;
        self.ess.append(ExperienceRecord::note(
            self.ids.next(),
            ctrl.time,
            ctrl.corr,
            format!(
                "phase_window:t={} reason={} llm={} tool={} lfm={} nsr={} ebm={} gov={}",
                phase_window.t,
                phase_window.reason,
                phase_window.allow_mask.llm_generation,
                phase_window.allow_mask.tool_planning,
                phase_window.allow_mask.lfm,
                phase_window.allow_mask.nsr,
                phase_window.allow_mask.ebm_constraints,
                phase_window.allow_mask.governor
            ),
        ))?;
        let stage_allow_mask = ucf_frames::v1::StageAllowMask {
            lfm: phase_window.allow_mask.lfm,
            nsr: phase_window.allow_mask.nsr,
            ebm_constraints: phase_window.allow_mask.ebm_constraints,
            governor: phase_window.allow_mask.governor,
            llm_generation: phase_window.allow_mask.llm_generation,
            tool_planning: phase_window.allow_mask.tool_planning,
        };

        if let Some(mods) = self.last_gating_modulation {
            if mods.stress_gate <= 0.35 {
                if let Err(err) = self.ess.append(ExperienceRecord::note(
                    self.ids.next(),
                    ctrl.time,
                    ctrl.corr,
                    "hormone_modulation:HighCortisol:tighten_budget",
                )) {
                    tracing::warn!(?err, "failed to persist hormone modulation note");
                }
            }
        }

        if let Some(summary) = decision.compute_summary {
            decision = decision.with_compute_summary(ucf_frames::v1::ComputeSignalsSummary {
                coherence: Some(coherence_metrics.coherence),
                instability: Some(coherence_metrics.instability),
                coherence_q: Some(Self::q_u16(coherence_metrics.coherence)),
                instability_q: Some(Self::q_u16(coherence_metrics.instability)),
                phi_proxy: Some(coherence_metrics.phi_proxy),
                coherence_digest: Some(iit_monitor.monitor_digest),
                iit_coherence_q: Some(iit_monitor.coherence_q.raw()),
                iit_incoherence_q: Some(iit_monitor.incoherence_q.raw()),
                iit_reason_codes: {
                    let mut codes = [0u8; 4];
                    for (idx, reason) in iit_monitor.reason_codes.iter().take(4).enumerate() {
                        codes[idx] = *reason as u8;
                    }
                    Some(codes)
                },
                stage_allow_mask: Some(stage_allow_mask),
                ..summary
            });
        }
        decision = decision.with_gating_reason(coherence_gate);
        if matches!(nsr_v0_assessment.policy_hint, PolicyHint::Block) {
            decision = decision.with_gating_reason(Some("nsr_block"));
        } else if matches!(nsr_v0_assessment.policy_hint, PolicyHint::SafeOnly)
            && decision.gating_reason.is_none()
        {
            decision = decision.with_gating_reason(Some("nsr_safe_only"));
        }
        if coherence_gate.is_some() {
            self.coherence_gated_total = self.coherence_gated_total.saturating_add(1);
        }

        if self.backpressure > 0.8 && decision.decision == ucf_frames::v1::DecisionCode::Allow {
            let mut next = DecisionFrame::defer_with_reason(
                decision.time,
                decision.corr,
                decision.intent,
                ucf_frames::v1::ReasonCode("compute_backpressure"),
                "compute_backpressure",
            )
            .with_meta(decision.meta);
            if let Some(summary) = decision.compute_summary {
                next = next.with_compute_summary(summary);
            }
            decision = next;
        }

        if compute_summary.risk_quality == 2
            && decision.decision == ucf_frames::v1::DecisionCode::Allow
        {
            let mut next = DecisionFrame::defer_with_reason(
                decision.time,
                decision.corr,
                decision.intent,
                ucf_frames::v1::ReasonCode("compute_unavailable"),
                "compute_unavailable",
            )
            .with_meta(decision.meta);
            if let Some(summary) = decision.compute_summary {
                next = next.with_compute_summary(summary);
            }
            decision = next;
        }

        decision = if let Some(nsr_frame) = self.last_nsr_frame {
            match nsr_frame.verdict {
                2 => {
                    let mut next = DecisionFrame::deny_with_reason(
                        decision.time,
                        decision.corr,
                        decision.intent,
                        ucf_frames::v1::ReasonCode("deny_nsr_block"),
                        ucf_frames::v1::DenyReasonCode::PolicyViolation,
                        "deny_nsr_block",
                    )
                    .with_meta(decision.meta);
                    if let Some(summary) = decision.compute_summary {
                        next = next.with_compute_summary(summary);
                    }
                    next
                }
                0 | 1 => decision,
                _ => decision,
            }
        } else {
            decision
        };

        if let Some(fep) = &self.last_fep_outputs {
            if fep.action_inhibit >= 0.5 && decision.decision == ucf_frames::v1::DecisionCode::Allow
            {
                let mut next = DecisionFrame::defer_with_reason(
                    decision.time,
                    decision.corr,
                    decision.intent,
                    ucf_frames::v1::ReasonCode("fep_inhibit_high"),
                    "fep_inhibit_high",
                )
                .with_meta(decision.meta);
                if let Some(summary) = decision.compute_summary {
                    next = next.with_compute_summary(summary);
                }
                decision = next;
            }
        }

        if coherence_gate.is_some() && decision.decision == ucf_frames::v1::DecisionCode::Allow {
            let mut next = DecisionFrame::defer_with_reason(
                decision.time,
                decision.corr,
                decision.intent,
                ucf_frames::v1::ReasonCode("coherence_gate"),
                "coherence_gate",
            )
            .with_meta(decision.meta)
            .with_gating_reason(coherence_gate);
            if let Some(summary) = decision.compute_summary {
                next = next.with_compute_summary(summary);
            }
            decision = next;
        }

        let liquid_context = self.liquid_timeline_index.context_window(16);
        if let Some(window) = liquid_context {
            metrics::gauge!("ucf_lfm_uncertainty_mean_recent")
                .set(f64::from(window.mean_uncertainty));
        }
        let candidate_ctx = DecisionContext {
            now_t: ctrl.time.tick.get(),
            risk: compute_summary.risk,
            confidence: compute_summary.confidence,
            evidence_chain_digest: compute_summary.compute_chain_digest,
            planning_allowed: !matches!(decision.decision, ucf_frames::v1::DecisionCode::Deny)
                && !emergency_active,
            liquid_context: liquid_window_from_index(liquid_context),
        };
        let demo_toolread = matches!(
            &ctrl.payload,
            ucf_frames::v1::ControlPayload::Text(text) if text.contains("tool_demo_file_read")
        );
        let candidates =
            self.candidate_generator
                .generate(&ctrl, &candidate_ctx, DecisionBudget::default());
        metrics::counter!("ucf_candidates_generated_total").increment(candidates.len() as u64);
        let nsr_block = matches!(nsr_v0_assessment.policy_hint, PolicyHint::Block);
        let nsr_safe_only = matches!(nsr_v0_assessment.policy_hint, PolicyHint::SafeOnly);
        let assessments: Vec<_> = candidates
            .iter()
            .map(|candidate| assess_candidate(candidate, &decision, nsr_block, nsr_safe_only))
            .collect();
        let nsr_v1_outputs: Vec<NsrV1Output> = candidates
            .iter()
            .take(32)
            .map(|candidate| {
                self.nsr_v1_engine.evaluate(eval_input_from_candidate(
                    candidate,
                    compute_summary.risk,
                    emergency_active,
                    self.compute_budget.governor_tier,
                ))
            })
            .collect();
        let nsr_for_ebm = nsr_v1_outputs
            .iter()
            .map(|o| o.nsr_risk_q)
            .max_by_key(|q| q.raw());
        let allowed_count = assessments.iter().filter(|a| a.allowed).count() as u64;
        metrics::counter!("ucf_candidates_allowed_total").increment(allowed_count);
        metrics::counter!("ucf_candidates_blocked_total")
            .increment(candidates.len() as u64 - allowed_count);

        let mut selected = select_candidate(&candidates, &assessments).or_else(|| {
            candidates
                .iter()
                .zip(assessments.iter())
                .find(|(c, _)| c.is_noop())
                .map(|(c, a)| (c.clone(), a.clone()))
        });
        let mut ebm_output = None;
        let mut ebm_signal = None;
        let mut ebm_suppressed_by_emergency = false;
        if !matches!(self.ebm_mode, EbmEnablementMode::Off) {
            if emergency_active {
                ebm_suppressed_by_emergency = true;
            }
            let ebm_input = EbmInput {
                t: ctrl.time.tick.get(),
                governor_tier: self.compute_budget.governor_tier,
                emergency_active,
                context_digest: compute_input.context_digest,
                signals: EbmSignals {
                    risk_q: ucf_types::UQ0_16::from_f32_clamped(compute_summary.risk),
                    confidence_q: ucf_types::UQ0_16::from_f32_clamped(compute_summary.confidence),
                    pressure_q: ucf_types::UQ0_16::from_f32_clamped(compute_summary.pressure),
                    surprise_q: ucf_types::UQ0_16::from_f32_clamped(compute_summary.surprise),
                    uncertainty_q: ucf_types::UQ0_16::from_f32_clamped(
                        compute_summary.lfm_uncertainty.unwrap_or(1.0),
                    ),
                    coherence_q: None,
                    nsr_risk_q: nsr_for_ebm,
                },
                candidates: candidates
                    .iter()
                    .map(candidate_feature_from_decision)
                    .collect(),
            };

            let mut cpu_ref_reasoner = CpuEbmStubV0;
            let mut cpu_budget = ucf_compute::work_meter::WorkMeter::new(64);
            let cpu_out = cpu_ref_reasoner.score_candidates(ebm_input.clone(), &mut cpu_budget);

            let mut final_out = cpu_out.clone();
            let requested_gpu = !matches!(self.gpu_mode, GpuMode::Off);
            let mut gpu_allowed = requested_gpu && !self.gpu_disabled;

            #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
            if requested_gpu {
                gpu_allowed = false;
                self.gpu_disabled = true;
                let _ = self.ess.append(ExperienceRecord::from_gpu_unavailable(
                    self.ids.next(),
                    ctrl.time,
                    ctrl.corr,
                    GpuUnavailableRecord {
                        schema_version: 1,
                        t: ctrl.time.tick.get(),
                        stage: "ebm".to_string(),
                        reason: "gpu_feature_disabled".to_string(),
                    },
                ));
            }

            #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
            {
                if requested_gpu {
                    let caps = detect_gpu_capability();
                    if !caps.available {
                        gpu_allowed = false;
                        self.gpu_disabled = true;
                        let _ = self.ess.append(ExperienceRecord::from_gpu_unavailable(
                            self.ids.next(),
                            ctrl.time,
                            ctrl.corr,
                            GpuUnavailableRecord {
                                schema_version: 1,
                                t: ctrl.time.tick.get(),
                                stage: "ebm".to_string(),
                                reason: "gpu_unavailable".to_string(),
                            },
                        ));
                    }
                }
                if gpu_allowed {
                    let caps = GpuResourceCaps::from_env();
                    let estimated_vram_bytes = (ebm_input.candidates.len() as u64)
                        .saturating_mul(4)
                        .saturating_mul(EBM_FEATURE_D_MAX as u64)
                        .saturating_mul(4);
                    let estimated_kernel_micros = 50_000_u64;
                    if !caps.within_caps(estimated_vram_bytes, estimated_kernel_micros) {
                        gpu_allowed = false;
                        self.gpu_disabled = true;
                        let _ = self
                            .ess
                            .append(ExperienceRecord::from_gpu_resource_violation(
                                self.ids.next(),
                                ctrl.time,
                                ctrl.corr,
                                GpuResourceViolationRecord {
                                    schema_version: 1,
                                    t: ctrl.time.tick.get(),
                                    stage: "ebm".to_string(),
                                    estimated_vram_bytes,
                                    estimated_kernel_micros,
                                    action: "gpu_disabled".to_string(),
                                },
                            ));
                    }
                }
            }

            if gpu_allowed {
                let mut ebm_budget = ucf_compute::work_meter::WorkMeter::new(64);
                let gpu_out = self
                    .ebm_reasoner
                    .score_candidates(ebm_input.clone(), &mut ebm_budget);
                let parity = ebm_parity_within_lsb(&cpu_out, &gpu_out, 1);
                let parity_fail = parity.mismatch_count > 0;
                if parity_fail {
                    self.gpu_parity_fail_streak = self.gpu_parity_fail_streak.saturating_add(1);
                } else {
                    self.gpu_parity_fail_streak = 0;
                }
                if self.gpu_parity_fail_streak >= 3 {
                    self.gpu_disabled = true;
                }
                let _ = self.ess.append(ExperienceRecord::from_gpu_parity(
                    self.ids.next(),
                    ctrl.time,
                    ctrl.corr,
                    GpuParityRecord {
                        schema_version: 1,
                        t: ctrl.time.tick.get(),
                        stage: parity.stage.to_string(),
                        mismatch_count: parity.mismatch_count,
                        compared_count: parity.compared_count,
                        max_scalar_delta_lsb: parity.max_scalar_delta_lsb,
                        action: if parity_fail {
                            "fallback_cpu".to_string()
                        } else {
                            "ok".to_string()
                        },
                    },
                ));

                match self.gpu_mode {
                    GpuMode::Shadow => {
                        final_out = cpu_out.clone();
                    }
                    GpuMode::Active => {
                        if !parity_fail && !self.gpu_disabled {
                            final_out = gpu_out;
                        }
                    }
                    GpuMode::Off => {
                        final_out = cpu_out.clone();
                    }
                }
            }

            if matches!(self.ebm_mode, EbmEnablementMode::Active) {
                selected = final_out
                    .best_indices
                    .iter()
                    .filter_map(|idx| {
                        candidates
                            .get(usize::from(*idx))
                            .zip(assessments.get(usize::from(*idx)))
                    })
                    .find(|(_, assessment)| assessment.allowed)
                    .map(|(candidate, assessment)| (candidate.clone(), assessment.clone()))
                    .or(selected);
            }
            ebm_signal = Some(EbmSignal::from_output(&final_out));
            ebm_output = Some(final_out);
        }
        if let (Some(sig), Some(summary)) = (ebm_signal, decision.compute_summary) {
            decision = decision.with_compute_summary(ucf_frames::v1::ComputeSignalsSummary {
                ebm_energy_mean_topk_q: Some(sig.energy_mean_topk_q.raw()),
                ..summary
            });
            metrics::gauge!("ucf_ebm_energy_mean_topk_q")
                .set(f64::from(sig.energy_mean_topk_q.raw()));
        }
        if matches!(nsr_v0_assessment.policy_hint, PolicyHint::Block) || emergency_active {
            selected = candidates
                .iter()
                .zip(assessments.iter())
                .find(|(c, _)| c.is_noop())
                .map(|(c, a)| (c.clone(), a.clone()));
        }
        if demo_toolread {
            if let Some((candidate, assessment)) = candidates
                .iter()
                .zip(assessments.iter())
                .filter(|(c, _)| !c.tool_intents.is_empty())
                .min_by_key(|(c, _)| c.candidate_id)
            {
                let mut forced = assessment.clone();
                forced.allowed = true;
                selected = Some((candidate.clone(), forced));
            }
        }
        let (selected_candidate, selected_assessment) = selected.unwrap_or_else(|| {
            let c = candidates.first().cloned().expect("candidate exists");
            let a = assessments.first().cloned().expect("assessment exists");
            (c, a)
        });
        let selected_nsr_v1 = candidates
            .iter()
            .position(|c| c.candidate_id == selected_candidate.candidate_id)
            .and_then(|idx| nsr_v1_outputs.get(idx).cloned())
            .unwrap_or_else(|| NsrV1Output {
                status: NsrV1Status::ParseErrorFallback,
                nsr_risk_q: ucf_types::UQ0_16::ONE,
                top_reasons: vec![],
                rules_digest: [0; 32],
            });
        metrics::counter!(
            "ucf_candidate_selected_total",
            "intent_kind" => format!("{:?}", selected_candidate.intent_kind),
            "output_class" => format!("{:?}", selected_candidate.output_class)
        )
        .increment(1);
        metrics::counter!("ucf_tool_intents_total", "kind" => "all".to_string())
            .increment(selected_candidate.tool_intents.len() as u64);
        tracing::info_span!(
            "decision.candidates",
            selected_id = selected_candidate.candidate_id,
            digest_prefix = format!(
                "{:02x}{:02x}{:02x}{:02x}",
                selected_candidate.digest[0],
                selected_candidate.digest[1],
                selected_candidate.digest[2],
                selected_candidate.digest[3]
            ),
        );

        let eid1 = self.ids.next();
        self.ess.append(
            ExperienceRecord::from_control(eid1, ctrl.clone())
                .with_neuromod(snapshot)
                .with_iit_phi(phi),
        )?;

        let eid2 = self.ids.next();
        let decision_record = ExperienceRecord::from_decision(eid2, decision.clone())
            .with_neuromod(snapshot)
            .with_iit_phi(phi);
        self.ess.append(decision_record.clone())?;

        if ctrl.time.tick.get().is_multiple_of(self.lfm_persist_every) {
            if let (Some(liquid_state_digest), Some(uncertainty), Some(stability)) = (
                compute_summary.lfm_digest,
                compute_summary.lfm_uncertainty,
                compute_summary.lfm_stability,
            ) {
                let evidence_chain_digest = compute_summary.compute_chain_digest;
                let summary = LfmSummaryRecord {
                    t: ctrl.time.tick.get(),
                    decision_id: Some(eid2.0),
                    evidence_chain_digest,
                    backend_pack_digest: backend_pack_digest(compute_summary),
                    liquid_state_digest,
                    liquid_readout_digest: liquid_state_digest,
                    uncertainty,
                    stability,
                    schema_version: 2,
                    digest: [0; 32],
                }
                .with_digest();
                self.liquid_timeline_index.append(summary);
                self.ess.append(ExperienceRecord::from_lfm_summary(
                    self.ids.next(),
                    decision.time,
                    decision.corr,
                    summary,
                ))?;
                metrics::counter!("ucf_lfm_records_appended_total").increment(1);

                if ctrl
                    .time
                    .tick
                    .get()
                    .is_multiple_of(self.lfm_window_emit_every)
                {
                    let window = self.liquid_timeline_index.get_last(32);
                    if !window.is_empty() {
                        let mut sum_u = 0.0f32;
                        let mut sum_s = 0.0f32;
                        let mut hasher = blake3::Hasher::new();
                        for item in &window {
                            sum_u += item.uncertainty.clamp(0.0, 1.0);
                            sum_s += item.stability.clamp(0.0, 1.0);
                            hasher.update(&item.digest);
                        }
                        let denom = window.len() as f32;
                        let record = LfmWindowRecord {
                            t0: window.first().map(|s| s.t).unwrap_or(ctrl.time.tick.get()),
                            t1: window.last().map(|s| s.t).unwrap_or(ctrl.time.tick.get()),
                            sample_count: window.len().min(usize::from(u16::MAX)) as u16,
                            mean_uncertainty: (sum_u / denom).clamp(0.0, 1.0),
                            mean_stability: (sum_s / denom).clamp(0.0, 1.0),
                            rolling_digest: *hasher.finalize().as_bytes(),
                            schema_version: 2,
                            digest: [0; 32],
                        }
                        .with_digest();
                        self.ess.append(ExperienceRecord::from_lfm_window(
                            self.ids.next(),
                            decision.time,
                            decision.corr,
                            record,
                        ))?;
                    }
                }
            }
        }

        let candidate_summaries: Vec<CandidateSummaryRecord> = candidates
            .iter()
            .zip(assessments.iter())
            .take(8)
            .map(|(candidate, assessment)| CandidateSummaryRecord {
                candidate_id: candidate.candidate_id,
                digest: candidate.digest,
                intent_kind: candidate.intent_kind as u8,
                output_class: candidate.output_class as u8,
                tool_intent_count: candidate.tool_intents.len().min(8) as u8,
                allowed: assessment.allowed,
                policy_hint: match assessment.policy_hint {
                    CandidatePolicyHint::Block => 2,
                    CandidatePolicyHint::SafeOnly => 1,
                    CandidatePolicyHint::Normal => 0,
                },
            })
            .collect();
        let candidate_set_record = CandidateSetRecord {
            schema_version: 1,
            decision_id: eid2.0,
            t: ctrl.time.tick.get(),
            selected_candidate_id: selected_candidate.candidate_id,
            selected_candidate_digest: selected_candidate.digest,
            summaries: candidate_summaries,
        };
        self.ess.append(ExperienceRecord::from_candidate_set(
            self.ids.next(),
            decision.time,
            decision.corr,
            candidate_set_record,
        ))?;

        if let Some(out) = ebm_output {
            let mut top_energies_q = Vec::new();
            let mut top_candidate_ids = Vec::new();
            for idx in out.best_indices.iter().take(8) {
                let i = usize::from(*idx);
                if let (Some(energy), Some(candidate)) = (out.energies_q.get(i), candidates.get(i))
                {
                    top_energies_q.push(energy.raw());
                    top_candidate_ids.push(candidate.candidate_id);
                }
            }
            let record = EbmReasoningRecord {
                suppressed_by_emergency: ebm_suppressed_by_emergency,
                schema_version: 2,
                t: ctrl.time.tick.get(),
                run_id: self.ebm_run_id,
                decision_id: eid2.0,
                backend_pack_digest_prefix: prefix8(backend_pack_digest(compute_summary)),
                ebm_backend_id: self.ebm_reasoner.backend_id() as u8,
                ebm_model_digest_prefix: out.model_digest_prefix,
                contract_version: self.ebm_reasoner.contract_version().as_u16(),
                enablement_mode: self.ebm_mode.as_u8(),
                risk_q: ucf_types::UQ0_16::from_f32_clamped(compute_summary.risk).raw(),
                pressure_q: ucf_types::UQ0_16::from_f32_clamped(compute_summary.pressure).raw(),
                surprise_q: ucf_types::UQ0_16::from_f32_clamped(compute_summary.surprise).raw(),
                uncertainty_q: ucf_types::UQ0_16::from_f32_clamped(
                    compute_summary.lfm_uncertainty.unwrap_or(1.0),
                )
                .raw(),
                aggregate_energy_q: out.aggregate_energy_q.raw(),
                base_energy_q: out
                    .base_energies_q
                    .get(usize::from(out.best_indices.first().copied().unwrap_or(0)))
                    .copied()
                    .unwrap_or(out.aggregate_energy_q)
                    .raw(),
                top_energies_q,
                top_candidate_ids,
                ebm_digest_prefix: prefix8(out.ebm_digest),
                constraints_digest_prefix: out.constraints_digest_prefix,
                top_term_contributions: out
                    .selected_term_contributions
                    .iter()
                    .take(8)
                    .map(|t| (t.id, t.contrib_q.raw()))
                    .collect(),
                search_enabled: out.search_enabled,
                search_steps_used: out.search_steps_used,
                evidence_chain_digest_prefix: prefix8(compute_summary.compute_chain_digest),
                status: out.status.as_u8(),
                reason_code: match out.status {
                    EbmStatus::Ok => 0,
                    EbmStatus::BudgetExceeded => 1,
                    EbmStatus::DegradedFallback => 2,
                    EbmStatus::Disabled => 3,
                    EbmStatus::Error => 4,
                },
            };
            self.ess.append(ExperienceRecord::from_ebm_reasoning(
                self.ids.next(),
                decision.time,
                decision.corr,
                record,
            ))?;
            if !matches!(out.status, EbmStatus::Ok) {
                self.ess
                    .append(ExperienceRecord::from_ebm_envelope_violation(
                        self.ids.next(),
                        decision.time,
                        decision.corr,
                        EbmEnvelopeViolationRecord {
                            schema_version: 1,
                            t: ctrl.time.tick.get(),
                            decision_id: eid2.0,
                            violation_code: out.status.as_u8(),
                            details: format!("status={:?}", out.status),
                        },
                    ))?;
            }
        }

        let decoding_policy = apply_decoding_policy(
            self.llm_cfg.max_tokens,
            selected_candidate.output_class,
            nsr_v0_assessment.policy_hint,
            compute_summary.lfm_uncertainty,
            compute_summary.lfm_stability,
            self.last_gating_modulation.map(|m| m.stress_gate),
            emergency_active,
        );
        metrics::histogram!("ucf_llm_max_tokens_eff").record(decoding_policy.max_tokens_eff as f64);
        if matches!(
            decoding_policy.output_override,
            Some(OutputOverrideCode::ForcedSafeOnly)
        ) {
            metrics::counter!("ucf_llm_forced_safeonly_total").increment(1);
        }
        if matches!(
            decoding_policy.output_override,
            Some(OutputOverrideCode::ForcedShort)
        ) {
            metrics::counter!("ucf_llm_forced_short_total").increment(1);
        }

        let output_record = if stage_allow_mask.llm_generation
            && matches!(
                decoding_policy.output_class,
                CandidateOutputClass::SafeText | CandidateOutputClass::Code
            ) {
            metrics::counter!("ucf_lfm_conditioning_used_total").increment(1);
            let rationale = selected_candidate
                .rationale
                .lines
                .iter()
                .map(|line| bounded_summary_line(line))
                .collect::<Vec<_>>()
                .join(" | ");
            let control_summary = bounded_summary_line(ctrl.intent.summary.as_ref());
            let decision_summary = bounded_summary_line(&format!(
                "decision={} candidate={} class={:?} rationale={}",
                eid2.0, selected_candidate.candidate_id, decoding_policy.output_class, rationale
            ));
            let prompt = build_prompt(
                &control_summary,
                &decision_summary,
                PromptConditioning {
                    risk: Some(compute_summary.risk),
                    confidence: Some(compute_summary.confidence),
                    surprise: compute_summary.surprise,
                    pressure: compute_summary.pressure,
                    uncertainty: compute_summary.lfm_uncertainty,
                    coherence: None,
                    instability: None,
                    evidence_chain_digest: compute_summary.compute_chain_digest,
                    lfm_readout_digest: compute_summary.lfm_digest,
                },
                self.liquid_timeline_index.context_window(16),
                decoding_policy.output_class,
            );
            let llm_req = LlmRequest {
                schema_version: 1,
                t: ctrl.time.tick.get(),
                decision_id: eid2.0,
                candidate_id: selected_candidate.candidate_id,
                output_class: map_output_class(decoding_policy.output_class),
                prompt,
                context_digest: compute_input.context_digest,
                evidence_chain_digest: compute_summary.compute_chain_digest,
                lfm_readout_digest: compute_summary.lfm_digest,
                lfm_uncertainty: compute_summary.lfm_uncertainty,
                lfm_stability: compute_summary.lfm_stability,
                coherence: None,
                instability: None,
                risk: Some(compute_summary.risk),
                confidence: Some(compute_summary.confidence),
                seed: self.llm_cfg.seed,
                max_tokens: decoding_policy.max_tokens_eff,
                temperature: 0.0,
                top_p: 1.0,
                sampling_enabled: false,
            }
            .bounded();
            let llm_request_digest = llm_req.digest();
            let in_tokens = estimate_input_tokens(&llm_req.prompt);
            let llm_reservation = self.compute_economy.reserve_llm(
                BudgetPool::Primary,
                in_tokens,
                llm_req.max_tokens,
                ctrl.time.tick.get(),
            );
            let mut llm_resp = if let Err(v) = &llm_reservation {
                self.persist_compute_budget_violation(ctrl.time, ctrl.corr, v)?;
                LlmResponse {
                    status: LlmStatus::Refused,
                    text: "refused: compute token budget exhausted".to_string(),
                    token_count: 0,
                    finish_reason: FinishReason::PolicyRefusal,
                    digest: [0; 32],
                }
            } else {
                {
                    let mode = crate::stage_isolation::StageIsolationMode::from_env();
                    let isolated = mode.isolate_llm();
                    let infer = if isolated {
                        crate::stage_isolation::infer_llm_isolated(&llm_req).map_err(|_| {
                            ucf_compute::ComputeError::Internal {
                                reason: "stage_worker".to_string(),
                            }
                        })
                    } else {
                        self.llm_backend.infer(&llm_req, self.compute_budget)
                    };
                    infer.unwrap_or_else(|err| {
                        let text =
                            if matches!(err, ucf_compute::ComputeError::BudgetExceeded { .. }) {
                                "System busy; try again.".to_string()
                            } else {
                                "refused: llm backend unavailable".to_string()
                            };
                        LlmResponse {
                            status: LlmStatus::Failed,
                            text,
                            token_count: 0,
                            finish_reason: FinishReason::Error,
                            digest: [0; 32],
                        }
                    })
                }
            };
            if let Ok(res) = llm_reservation {
                if let Some(v) = self.compute_economy.settle_llm(
                    res,
                    in_tokens,
                    llm_resp.token_count,
                    ctrl.time.tick.get(),
                ) {
                    self.persist_compute_budget_violation(ctrl.time, ctrl.corr, &v)?;
                }
            }
            if llm_resp.digest == [0; 32] {
                llm_resp = ucf_compute::capabilities::LlmResponse::new(
                    llm_resp.status,
                    llm_resp.text,
                    llm_resp.token_count,
                    llm_resp.finish_reason,
                );
            }
            metrics::counter!("ucf_llm_infer_total", "backend" => self.llm_backend.name().to_string()).increment(1);
            if matches!(llm_resp.status, LlmStatus::Refused) {
                metrics::counter!("ucf_llm_refused_total").increment(1);
            }
            if matches!(llm_resp.status, LlmStatus::Truncated) {
                metrics::counter!("ucf_llm_truncated_total").increment(1);
            }
            let mut text = llm_resp
                .text
                .chars()
                .take(MAX_OUTPUT_TEXT_CHARS)
                .collect::<String>();
            if validate_output(decoding_policy.output_class, &text).is_err() {
                text = "refused: output validation failed".to_string();
                llm_resp = ucf_compute::capabilities::LlmResponse::new(
                    LlmStatus::Refused,
                    text.clone(),
                    0,
                    FinishReason::PolicyRefusal,
                );
            }
            OutputRecord {
                schema_version: OUTPUT_SCHEMA_VERSION,
                decision_id: eid2.0,
                candidate_id: selected_candidate.candidate_id,
                t: ctrl.time.tick.get(),
                output_class: decoding_policy.output_class as u8,
                llm_backend_name: self.llm_backend.name().to_string(),
                llm_request_digest,
                llm_response_digest: llm_resp.digest,
                token_count: llm_resp.token_count,
                status: output_status_code(llm_resp.status),
                finish_reason: finish_reason_code(llm_resp.finish_reason),
                content_digest: compute_content_digest(&text),
                text: Some(text),
                redacted: false,
                payload_len: None,
                payload_classification: Some(PayloadClassification::Safe),
                redaction_policy_marker: None,
                evidence_chain_digest: compute_summary.compute_chain_digest,
                lfm_readout_digest: compute_summary.lfm_digest,
                lfm_uncertainty: compute_summary.lfm_uncertainty,
                lfm_stability: compute_summary.lfm_stability,
                max_tokens_eff: decoding_policy.max_tokens_eff,
                output_override: decoding_policy
                    .output_override
                    .map(OutputOverrideCode::as_u8),
                override_reasons: decoding_policy.reason_codes(),
            }
        } else {
            let summary = selected_candidate
                .tool_intents
                .iter()
                .take(4)
                .map(|intent| format!("{:?}", intent.kind))
                .collect::<Vec<_>>()
                .join(",");
            OutputRecord {
                schema_version: OUTPUT_SCHEMA_VERSION,
                decision_id: eid2.0,
                candidate_id: selected_candidate.candidate_id,
                t: ctrl.time.tick.get(),
                output_class: selected_candidate.output_class as u8,
                llm_backend_name: "plan-only".to_string(),
                llm_request_digest: [0; 32],
                llm_response_digest: [0; 32],
                token_count: 0,
                status: output_status_code(LlmStatus::Refused),
                finish_reason: finish_reason_code(FinishReason::PolicyRefusal),
                content_digest: compute_content_digest(&format!(
                    "tool intents require gate: {summary}"
                )),
                text: Some(format!("tool intents require gate: {summary}")),
                redacted: false,
                payload_len: None,
                payload_classification: Some(PayloadClassification::Safe),
                redaction_policy_marker: None,
                evidence_chain_digest: compute_summary.compute_chain_digest,
                lfm_readout_digest: compute_summary.lfm_digest,
                lfm_uncertainty: compute_summary.lfm_uncertainty,
                lfm_stability: compute_summary.lfm_stability,
                max_tokens_eff: decoding_policy.max_tokens_eff,
                output_override: decoding_policy
                    .output_override
                    .map(OutputOverrideCode::as_u8),
                override_reasons: decoding_policy.reason_codes(),
            }
        };
        metrics::counter!("ucf_output_records_total", "class" => format!("{:?}", selected_candidate.output_class)).increment(1);
        self.ess.append(ExperienceRecord::from_output(
            self.ids.next(),
            decision.time,
            decision.corr,
            output_record,
        ))?;
        if ctrl
            .time
            .tick
            .get()
            .is_multiple_of(self.hormone_persist_every)
        {
            if let Some(summary) = self.last_hormone_summary {
                let hormone_record = HormoneRecord {
                    t: summary.t,
                    cortisol_q: u16::from(quantize_unit(summary.cortisol)),
                    dopamine_q: u16::from(quantize_unit(summary.dopamine)),
                    norepinephrine_q: u16::from(quantize_unit(summary.norepinephrine)),
                    serotonin_q: u16::from(quantize_unit(summary.serotonin)),
                    acetylcholine_q: u16::from(quantize_unit(summary.acetylcholine)),
                    drive_q: u16::from(quantize_unit(summary.drive)),
                    stress_index_q: u16::from(quantize_unit(summary.stress_index)),
                    attention_gain_q: self
                        .last_gating_modulation
                        .map(|m| u16::from(quantize_unit(m.attention_gain))),
                    plasticity_gate_q: self
                        .last_gating_modulation
                        .map(|m| u16::from(quantize_unit(m.plasticity_gate))),
                    stress_gate_q: self
                        .last_gating_modulation
                        .map(|m| u16::from(quantize_unit(m.stress_gate))),
                    hormone_digest: summary.digest,
                    evidence_chain_digest: summary.evidence_chain_digest,
                    modulation_digest: self.last_gating_modulation.map(|m| m.digest),
                    schema_version: 2,
                };
                self.ess.append(ExperienceRecord::from_hormone(
                    self.ids.next(),
                    ctrl.time,
                    ctrl.corr,
                    hormone_record,
                ))?;
            }
        }
        if ctrl
            .time
            .tick
            .get()
            .is_multiple_of(self.neuro_persist_every)
        {
            if let Some(summary) = self.last_neuro_summary {
                let spikes = self.last_neuro_spikes.as_ref();
                let neuro_record = NeuroRecord {
                    t: summary.t,
                    arousal_q: u16::from(quantize_unit(summary.arousal)),
                    attention_gain_q: u16::from(quantize_unit(summary.attention_gain)),
                    excitability_q: u16::from(quantize_unit(summary.excitability)),
                    spike_rate_q: u16::from(quantize_unit(summary.spike_rate)),
                    summary_digest: summary.digest,
                    evidence_chain_digest: summary.evidence_chain_digest,
                    hormone_digest: self.last_hormone_summary.map(|h| h.digest),
                    spikes_digest: spikes.map(|s| s.digest),
                    spike_count: spikes
                        .map(|s| s.spikes.len().min(u16::MAX as usize) as u16)
                        .unwrap_or(0),
                    degraded: self.last_neuro_degraded,
                    schema_version: 1,
                };
                self.ess.append(ExperienceRecord::from_neuro(
                    self.ids.next(),
                    ctrl.time,
                    ctrl.corr,
                    neuro_record,
                ))?;
            }
        }

        if self.consolidation_hook_enabled {
            {
                match self
                    .compute_milestone_aggregator
                    .on_append(&decision_record)
                {
                    Ok(milestones) => {
                        for milestone in milestones {
                            self.consolidation_milestones_emitted_total = self
                                .consolidation_milestones_emitted_total
                                .saturating_add(1);
                            self.last_compute_milestone = Some(milestone.clone());
                            if self.geist_hook_enabled {
                                match self.geist_state_updater.on_milestone(&milestone) {
                                    Ok(Some(_)) => {
                                        self.geist_updates_accepted_total =
                                            self.geist_updates_accepted_total.saturating_add(1);
                                    }
                                    Ok(None) => {
                                        self.geist_updates_rejected_total =
                                            self.geist_updates_rejected_total.saturating_add(1);
                                        match self.geist_state_updater.last_reject_reason() {
                                            Some(GeistRejectReason::NotEnoughSamples) => {
                                                self.geist_updates_rejected_not_enough_samples_total = self
                                                    .geist_updates_rejected_not_enough_samples_total
                                                    .saturating_add(1);
                                            }
                                            Some(GeistRejectReason::Unstable) => {
                                                self.geist_updates_rejected_unstable_total = self
                                                    .geist_updates_rejected_unstable_total
                                                    .saturating_add(1);
                                            }
                                            Some(GeistRejectReason::Degraded) => {
                                                self.geist_updates_rejected_degraded_total = self
                                                    .geist_updates_rejected_degraded_total
                                                    .saturating_add(1);
                                            }
                                            Some(GeistRejectReason::Drift) => {
                                                self.geist_updates_rejected_drift_total = self
                                                    .geist_updates_rejected_drift_total
                                                    .saturating_add(1);
                                            }
                                            None => {}
                                        }
                                    }
                                    Err(_error) => {
                                        self.geist_hook_errors_total =
                                            self.geist_hook_errors_total.saturating_add(1);
                                    }
                                }
                            }
                        }
                    }
                    Err(_error) => {
                        self.consolidation_hook_errors_total =
                            self.consolidation_hook_errors_total.saturating_add(1);
                    }
                }
            }
        }

        if let Some(fep) = &self.last_fep_outputs {
            if fep.memory_priority >= 0.45 {
                let eid = self.ids.next();
                self.ess.append(ExperienceRecord::note(
                    eid,
                    decision.time,
                    decision.corr,
                    "consolidate:high_mem_priority",
                ))?;
            }
        }

        let denied_tool = matches!(decision.decision, ucf_frames::v1::DecisionCode::Deny);
        self.summarize_tick_for_evolution(
            decision.compute_summary,
            denied_tool,
            self.compute_budget.governor_tier,
            backend_pack_digest(compute_summary),
            true,
            decision
                .compute_summary
                .and_then(|s| s.ebm_energy_mean_topk_q)
                .unwrap_or(0),
        );
        self.maybe_run_evolution(decision.time, decision.corr)?;

        let nsr_record = NsrRecord {
            t: ctrl.time.tick.get(),
            decision_id: eid2.0,
            evidence_chain_digest: compute_summary.compute_chain_digest,
            ruleset_id: 1,
            engine_id: "nsr_v1_datalog_lite",
            schema_version: 1,
            nsr_risk_q: selected_nsr_v1.nsr_risk_q.raw(),
            nsr_status: selected_nsr_v1.status as u8,
            nsr_confidence_q: quantize_unit_u16(nsr_v0_assessment.nsr_confidence),
            rules_digest_prefix: prefix8(selected_nsr_v1.rules_digest),
            policy_hint: match nsr_v0_assessment.policy_hint {
                PolicyHint::Block => 2,
                PolicyHint::SafeOnly => 1,
                PolicyHint::Normal => 0,
            },
            reasons: selected_nsr_v1
                .top_reasons
                .iter()
                .map(|r| r.term_id)
                .take(8)
                .collect(),
            facts_digest: nsr_v0_assessment.facts_digest,
            assessment_digest: nsr_v0_assessment.digest,
        };
        self.ess.append(ExperienceRecord::from_nsr(
            self.ids.next(),
            ctrl.time,
            ctrl.corr,
            nsr_record,
        ))?;

        self.last_nsr_risk = Some(selected_nsr_v1.nsr_risk_q.to_f32());

        let mut governance_signals = GovernanceSignals::from_inputs(
            Some(&decision),
            decision.time.tick.get(),
            self.last_nsr_risk,
            self.last_hormone_summary.map(|s| s.stress_index),
        );
        self.update_model_governance_tick(
            decision.time.tick.get(),
            decision.corr,
            decision.compute_summary,
            false,
        )?;
        if emergency_active {
            governance_signals.ebm_energy_mean_topk_q = None;
        }
        let (issued_caps, issuance_decision) = issue_capabilities_governed(
            Some(&decision),
            decision.time.tick.get(),
            governance_signals,
            &mut self.tool_governor,
        );
        metrics::counter!("ucf_governor_ebm_penalty_total")
            .increment(u64::from(issuance_decision.ebm_penalty_q));
        metrics::counter!("ucf_governor_nsr_penalty_total")
            .increment(u64::from(issuance_decision.nsr_penalty_q));
        self.last_nsr_penalty_q = issuance_decision.nsr_penalty_q;
        let incoherence_q = decision
            .compute_summary
            .and_then(|s| s.iit_incoherence_q)
            .unwrap_or(0);
        let incoherence_extra_penalty_q = if incoherence_q > IIT_INCOHERENCE_GOV_INC_HIGH_Q {
            let span = u32::from(u16::MAX - IIT_INCOHERENCE_GOV_INC_HIGH_Q);
            let delta = u32::from(incoherence_q - IIT_INCOHERENCE_GOV_INC_HIGH_Q);
            ((delta * u32::from(IIT_INCOHERENCE_GOV_EXTRA_PENALTY_MAX_Q)) / span) as u16
        } else {
            0
        };
        let (issued_caps, mut effective_tier) = if emergency_active {
            metrics::counter!("ucf_emergency_forced_deny_tools_total").increment(1);
            (ucf_policy::capability::CapabilitySet::empty(), 3_u8)
        } else {
            (issued_caps, issuance_decision.tier.as_u8())
        };
        let mut effective_caps = issued_caps;
        if incoherence_extra_penalty_q > 0 {
            effective_tier = effective_tier.max(2);
            metrics::counter!("ucf_governor_incoherence_penalty_total")
                .increment(u64::from(incoherence_extra_penalty_q));
            self.ess.append(ExperienceRecord::note(
                self.ids.next(),
                decision.time,
                decision.corr,
                format!("governor_incoherence_tighten:q={incoherence_extra_penalty_q}"),
            ))?;
        }
        if self.model_governance.any_critical_degraded() {
            effective_tier = 3;
            effective_caps = ucf_policy::capability::CapabilitySet::empty();
            self.ess.append(ExperienceRecord::note(
                self.ids.next(),
                decision.time,
                decision.corr,
                "model_governance_critical_degraded:tool_issuance_denied",
            ))?;
        }
        self.tool_gate.capabilities = effective_caps;
        self.persist_issuance_records(
            decision.time,
            decision.corr,
            eid2.0,
            selected_candidate.candidate_id,
            &issuance_decision,
            effective_tier,
        )?;
        let mut requests = Vec::new();
        if selected_assessment.allowed {
            for intent in selected_candidate.tool_intents.iter().take(8) {
                if self
                    .tool_gate
                    .capabilities
                    .tokens
                    .iter()
                    .any(|token| token.kind == intent.kind)
                {
                    let target = intent
                        .target
                        .preview
                        .clone()
                        .unwrap_or_else(|| format!("h64:{:016x}", intent.target.hash64));
                    requests.push(request_from_intent(
                        &decision,
                        eid2.0,
                        ctrl.corr.0,
                        intent.kind.clone(),
                        target,
                        GemPayloadHint {
                            bytes_out: intent.payload_hint.bytes_out,
                            bytes_in: intent.payload_hint.bytes_in,
                        },
                        selected_candidate.candidate_id,
                        intent.intent_digest,
                    ));
                } else {
                    let eid = self.ids.next();
                    self.ess.append(ExperienceRecord::note(
                        eid,
                        decision.time,
                        decision.corr,
                        "candidate_tool_missing_capability",
                    ))?;
                }
            }
        }
        if requests.is_empty() && selected_candidate.output_class == CandidateOutputClass::SafeText
        {
            let mut req = request_from(&ctrl, &decision, eid2.0);
            req.candidate_id = Some(selected_candidate.candidate_id);
            req.tool_intent_digest = Some(selected_candidate.digest);
            requests.push(req);
        }

        if !stage_allow_mask.tool_planning && !demo_toolread {
            self.ess.append(ExperienceRecord::note(
                self.ids.next(),
                decision.time,
                decision.corr,
                "phase_window_denied_tool_planning",
            ))?;
            requests.clear();
        }

        for request in requests {
            metrics::counter!("ucf_tool_requests_created_total").increment(1);
            if let Err(v) = self.compute_economy.try_charge(
                BudgetPool::Primary,
                ComputeStage::Tool,
                self.compute_economy
                    .schedule
                    .stage_cost(ComputeStage::Tool, 1),
                decision.time.tick.get(),
            ) {
                self.persist_compute_budget_violation(decision.time, decision.corr, &v)?;
                continue;
            }

            let Some(intent) = selected_candidate.tool_intents.iter().find(|intent| {
                request
                    .tool_intent_digest
                    .is_some_and(|digest| digest == intent.intent_digest)
            }) else {
                let eid = self.ids.next();
                self.ess.append(ExperienceRecord::note(
                    eid,
                    decision.time,
                    decision.corr,
                    "tool_plan_missing_intent",
                ))?;
                continue;
            };

            let policy_graph_bytes = hex::decode(&self.policy_graph_digest).unwrap_or_default();
            let mut policy_graph_digest = [0u8; 32];
            if policy_graph_bytes.len() >= 32 {
                policy_graph_digest.copy_from_slice(&policy_graph_bytes[..32]);
            }
            let plan = build_plan(
                intent,
                &request,
                selected_candidate.candidate_id,
                selected_candidate.digest,
                policy_graph_digest,
            );
            let plan_record = make_plan_record(
                &plan,
                governance_signals.ebm_energy_mean_topk_q.map(|q| q.raw()),
                Some(selected_nsr_v1.nsr_risk_q.raw()),
            );
            let eid_plan = self.ids.next();
            let plan_payload = AuditPayload::ToolPlan(ToolPlanAuditRecord {
                plan_digest_prefix: plan_record.plan_digest_prefix,
                tool_id: plan_record.tool_id.clone(),
                tool_class_id: plan_record.tool_class_id.clone(),
                args_digest_prefix: plan_record.args_digest_prefix,
                required_caps: plan_record.required_caps.clone(),
                ebm_energy_q: plan_record.ebm_energy_q,
                nsr_risk_q: plan_record.nsr_risk_q,
            });
            let plan_audit = ExperienceRecord::audit(
                eid_plan,
                decision.time,
                decision.corr,
                ExperienceKind::ToolPlan,
                plan_payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = plan_audit.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(plan_audit)?;

            let capability_token = self
                .tool_gate
                .capabilities
                .allows(&request, request.requested_at_t)
                .ok();
            let (issue_decision, issue_record) = issue_for_plan(
                &plan,
                &request,
                capability_token,
                issuance_decision.tier,
                governance_signals,
                policy_graph_digest,
            );
            let eid_issue = self.ids.next();
            let model_governance_digest = model_governance_digest(&self.model_governance.status);
            let issue_payload = AuditPayload::ToolIssue(ToolIssueAuditRecord {
                plan_digest_prefix: issue_record.plan_digest_prefix,
                issued: issue_decision.issued,
                issued_caps: issue_record.issued_caps.clone(),
                deny_reasons: issue_record.deny_reasons.clone(),
                model_governance_digest_prefix: Some(prefix8(model_governance_digest)),
                model_governance_reason_codes: model_governance_reason_codes(
                    &self.model_governance.status,
                ),
                policy_graph_digest_prefix: issue_record.policy_graph_digest_prefix,
                security_chain_digest_prefix: issue_record.security_chain_digest_prefix,
            });
            let issue_audit = ExperienceRecord::audit(
                eid_issue,
                decision.time,
                decision.corr,
                ExperienceKind::ToolIssue,
                issue_payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = issue_audit.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(issue_audit)?;

            if !issue_decision.issued {
                let eid_exec_denied = self.ids.next();
                let exec_denied = ExperienceRecord::audit(
                    eid_exec_denied,
                    decision.time,
                    decision.corr,
                    ExperienceKind::ToolExecution,
                    AuditPayload::ToolExecution(ToolExecutionRecord {
                        tool_request_id: request.id,
                        status: "Denied".to_string(),
                        bytes_out: request.payload_hint.bytes_out,
                        bytes_in: request.payload_hint.bytes_in,
                        error_code: Some(issue_decision.deny_reasons.join(",")),
                    }),
                    self.audit_head_digest,
                );
                self.audit_head_digest = exec_denied.audit_digest.unwrap_or(self.audit_head_digest);
                self.ess.append(exec_denied)?;
                continue;
            }

            if let Some(token) = issue_decision.issued_token.as_ref() {
                if self.spent_tool_tokens.contains(&token.token_digest) {
                    let eid_exec_denied = self.ids.next();
                    let exec_denied = ExperienceRecord::audit(
                        eid_exec_denied,
                        decision.time,
                        decision.corr,
                        ExperienceKind::ToolExecution,
                        AuditPayload::ToolExecution(ToolExecutionRecord {
                            tool_request_id: request.id,
                            status: "Denied".to_string(),
                            bytes_out: request.payload_hint.bytes_out,
                            bytes_in: request.payload_hint.bytes_in,
                            error_code: Some("token_replay".to_string()),
                        }),
                        self.audit_head_digest,
                    );
                    self.audit_head_digest =
                        exec_denied.audit_digest.unwrap_or(self.audit_head_digest);
                    self.ess.append(exec_denied)?;
                    continue;
                }
            }

            let Some(issued_token) = issue_decision.issued_token.clone() else {
                let eid_exec_denied = self.ids.next();
                let exec_denied = ExperienceRecord::audit(
                    eid_exec_denied,
                    decision.time,
                    decision.corr,
                    ExperienceKind::ToolExecution,
                    AuditPayload::ToolExecution(ToolExecutionRecord {
                        tool_request_id: request.id,
                        status: "Denied".to_string(),
                        bytes_out: request.payload_hint.bytes_out,
                        bytes_in: request.payload_hint.bytes_in,
                        error_code: Some("missing_issued_token".to_string()),
                    }),
                    self.audit_head_digest,
                );
                self.audit_head_digest = exec_denied.audit_digest.unwrap_or(self.audit_head_digest);
                self.ess.append(exec_denied)?;
                continue;
            };

            let registry = ToolPluginRegistry::with_builtin_stubs();
            let fs_env = SandboxFs::new(vec![
                (
                    "demo_root".to_string(),
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../../fixtures/demo_root"),
                ),
                (
                    "models_root".to_string(),
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"),
                ),
            ]);
            let plugin_audit = run_plugin_tool(
                &registry,
                &request,
                crate::tool_plugins::PluginDispatchSpec {
                    tool_id: plan.tool_id.clone(),
                    tool_class_id: plan.tool_class_id.clone(),
                    plan_digest: plan.plan_digest,
                    args: plan.args_canonical.clone(),
                    binding: CapabilityTokenBinding {
                        token: issued_token.clone(),
                        plan_digest: plan.plan_digest,
                    },
                },
                &SandboxEnv { fs: Some(&fs_env) },
            );
            let module = format!("plugin:{}", plan.tool_id);
            let op = format!("class:{}", plan.tool_class_id);
            let audit = crate::sandbox::SandboxToolExecution {
                request: request.clone(),
                auth: plugin_audit.auth,
                result: plugin_audit.result.clone(),
                call_digest: plan.plan_digest,
                reply_digest: plugin_audit.result_digest,
                capability_summary: crate::sandbox::CapabilitySetSummary::default(),
                module: module.clone(),
                op: op.clone(),
            };

            let req_payload = AuditPayload::ToolRequest(ToolRequestRecord {
                tool_request_id: request.id,
                capability_kind: request.kind.as_tag().to_string(),
                target: request.target.clone(),
                decision_id: request.decision_id,
                evidence_chain_digest: request.evidence_chain_digest,
                candidate_id: request.candidate_id,
                tool_intent_digest: request.tool_intent_digest,
            });
            let eid_req = self.ids.next();
            let req_record = ExperienceRecord::audit(
                eid_req,
                decision.time,
                decision.corr,
                ExperienceKind::ToolRequest,
                req_payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = req_record.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(req_record)?;

            let call_payload = AuditPayload::SandboxCall(SandboxCallRecord {
                tool_request_id: request.id,
                call_digest: audit.call_digest,
                module,
                op,
                evidence_chain_digest: request.evidence_chain_digest,
                capability_count: audit.capability_summary.items.len() as u32,
                isolation_runtime: Some("plugin_builtin_v1".to_string()),
                wasm_module_digest: None,
                fuel_used: None,
            });
            let eid_call = self.ids.next();
            let call_record = ExperienceRecord::audit(
                eid_call,
                decision.time,
                decision.corr,
                ExperienceKind::SandboxCall,
                call_payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = call_record.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(call_record)?;

            let (allowed, reason, token_digest) = match audit.auth {
                AuthorizationOutcome::Allowed { token_digest } => {
                    self.spent_tool_tokens.insert(token_digest);
                    (true, "allowed".to_string(), Some(token_digest))
                }
                AuthorizationOutcome::Denied { reason } => (false, format!("{reason:?}"), None),
                AuthorizationOutcome::RateLimited { retry_after_ticks } => {
                    (false, format!("rate_limited:{retry_after_ticks}"), None)
                }
            };
            let auth_payload = AuditPayload::ToolAuth(ToolAuthRecord {
                tool_request_id: request.id,
                allowed,
                reason,
                token_digest,
            });
            let eid_auth = self.ids.next();
            let auth_record = ExperienceRecord::audit(
                eid_auth,
                decision.time,
                decision.corr,
                ExperienceKind::ToolAuth,
                auth_payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = auth_record.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(auth_record)?;

            let exec_payload = AuditPayload::ToolExecution(ToolExecutionRecord {
                tool_request_id: request.id,
                status: format!("{:?}", audit.result.status),
                bytes_out: audit.result.bytes_out,
                bytes_in: audit.result.bytes_in,
                error_code: Some(compact_tool_result_note(
                    plugin_audit.result_digest,
                    &plugin_audit.preview,
                )),
            });
            let eid_exec = self.ids.next();
            let exec_record = ExperienceRecord::audit(
                eid_exec,
                decision.time,
                decision.corr,
                ExperienceKind::ToolExecution,
                exec_payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = exec_record.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(exec_record)?;

            let reply_payload = AuditPayload::SandboxReply(SandboxReplyRecord {
                tool_request_id: request.id,
                reply_digest: audit.reply_digest,
                status: format!("{:?}", audit.result.status),
                bytes_out: audit.result.bytes_out.unwrap_or(0),
                bytes_in: audit.result.bytes_in.unwrap_or(0),
                token_digest,
            });
            let eid_reply = self.ids.next();
            let reply_record = ExperienceRecord::audit(
                eid_reply,
                decision.time,
                decision.corr,
                ExperienceKind::SandboxReply,
                reply_payload,
                self.audit_head_digest,
            );
            self.audit_head_digest = reply_record.audit_digest.unwrap_or(self.audit_head_digest);
            self.ess.append(reply_record)?;

            if matches!(audit.result.status, ToolStatus::Failed) {
                let mut note = format!(
                    "gem_error:{}",
                    audit
                        .result
                        .error_code
                        .unwrap_or_else(|| "unknown".to_string())
                );
                if note.chars().count() > 120 {
                    note = note.chars().take(120).collect();
                }
                let eid3 = self.ids.next();
                self.ess.append(ExperienceRecord::note(
                    eid3,
                    decision.time,
                    decision.corr,
                    note,
                ))?;
                return Err(ucf_policy::errors::PolicyError::AdapterError("tool_failed").into());
            }
        }

        let checkpoint_payload = AuditPayload::AuditCheckpoint(AuditCheckpointRecord {
            head_digest: self.audit_head_digest,
        });
        let eid_cp = self.ids.next();
        let checkpoint_record = ExperienceRecord::audit(
            eid_cp,
            decision.time,
            decision.corr,
            ExperienceKind::AuditCheckpoint,
            checkpoint_payload,
            self.audit_head_digest,
        );
        self.audit_head_digest = checkpoint_record
            .audit_digest
            .unwrap_or(self.audit_head_digest);
        self.audit_chain_checkpoint_total = self.audit_chain_checkpoint_total.saturating_add(1);
        self.ess.append(checkpoint_record)?;

        if let Some((count, target)) = adapter.take_brain_spike_meta() {
            let mut note = format!("brain_spikes:n={count},dst={target}");
            if note.chars().count() > 120 {
                note = note.chars().take(120).collect();
            }

            let eid3 = self.ids.next();
            self.ess.append(ExperienceRecord::note(
                eid3,
                decision.time,
                decision.corr,
                note,
            ))?;
        }

        let decision =
            decision.with_policy_graph_digest_prefix(Some(self.policy_graph_digest_prefix));
        Ok(decision)
    }
}

fn tick_summary_payload(
    now_ms: u64,
    coherence_state: CoherenceState,
    mean_lock: f32,
    gate: f32,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(11);
    payload.extend_from_slice(&now_ms.to_le_bytes());
    payload.push(match coherence_state {
        CoherenceState::Stable => 0,
        CoherenceState::Drifting => 1,
        CoherenceState::Fragmenting => 2,
    });
    payload.push(quantize_unit(mean_lock));
    payload.push(quantize_unit(gate));
    payload
}

fn quantize_unit(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn encode_key(key: SmallKey) -> u8 {
    key as u8
}

fn encode_reason(reason: ReasonCode) -> u8 {
    reason as u8
}

fn reason_codes(score: &DeltaScore) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (idx, reason) in score.audit.reasons.iter().copied().take(8).enumerate() {
        out[idx] = encode_reason(reason);
    }
    out
}

fn top_reason(score: &DeltaScore) -> &'static str {
    match score.audit.reasons.first().copied() {
        Some(ReasonCode::ImprovesStability) => "improves_stability",
        Some(ReasonCode::ReducesBudgetExceed) => "reduces_budget_exceed",
        Some(ReasonCode::DegradesConfidence) => "degrades_confidence",
        Some(ReasonCode::ViolatesClamp) => "violates_clamp",
        Some(ReasonCode::TooAggressiveChange) => "too_aggressive_change",
        Some(ReasonCode::IncreasesRisk) => "increases_risk",
        Some(ReasonCode::WeakSafetyMargin) => "weak_safety_margin",
        Some(ReasonCode::HeuristicAssumption) => "heuristic_assumption",
        Some(ReasonCode::SafetyDominance) => "safety_dominance",
        Some(ReasonCode::EbmEnergyHigh) => "ebm_energy_high",
        Some(ReasonCode::EbmTrendUp) => "ebm_trend_up",
        None => "none",
    }
}

fn suppression_reason_code(ctx: &EvolutionContext) -> u8 {
    if ctx.emergency_active {
        return 1;
    }
    if !ctx.nsr_available {
        return 2;
    }
    if ctx.governor_tier_max >= 3 {
        return 3;
    }
    0
}

fn ops_summary_bytes(delta: &StructuralDelta) -> [u8; 128] {
    let mut out = [0u8; 128];
    let mut i = 0usize;
    for op in delta.ops.iter().take(8) {
        if i + 16 > out.len() {
            break;
        }
        match op {
            DeltaOp::Set { key, value } => {
                out[i] = 0;
                out[i + 1] = encode_key(*key);
                out[i + 4..i + 8].copy_from_slice(&value.to_le_bytes());
            }
            DeltaOp::Add { key, delta } => {
                out[i] = 1;
                out[i + 1] = encode_key(*key);
                out[i + 4..i + 8].copy_from_slice(&delta.to_le_bytes());
            }
            DeltaOp::Clamp { key, min, max } => {
                out[i] = 2;
                out[i + 1] = encode_key(*key);
                out[i + 4..i + 8].copy_from_slice(&min.to_le_bytes());
                out[i + 8..i + 12].copy_from_slice(&max.to_le_bytes());
            }
        }
        i += 16;
    }
    out
}

fn clamp_summary_bytes() -> [u8; 64] {
    let mut out = [0u8; 64];
    out[0] = 1;
    out
}

impl Default for RuntimeOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

fn sync_graph_from_cde_state(graph: &mut CausalGraph, state: &CdeState) {
    graph.vars.clear();
    graph.hyps.clear();
    for hyp in &state.hyps {
        graph.upsert_var(u32::from(hyp.edge.src));
        graph.upsert_var(u32::from(hyp.edge.dst));
        graph.upsert_hypothesis(
            Edge {
                from: u32::from(hyp.edge.src),
                to: u32::from(hyp.edge.dst),
            },
            hyp.last_update_ms,
            hyp.conf.clamp(0.0, 1.0),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_summary_fixture() -> ucf_compute::ComputeSignalsSummary {
        ucf_compute::ComputeSignalsSummary {
            backend: "stub",
            surprise: 0.2,
            pressure: 0.2,
            risk: 0.2,
            confidence: 0.8,
            surprise_q: 0,
            pressure_q: 0,
            risk_q: 0,
            confidence_q: 0,
            spike_count: 0,
            spikes_digest: [1; 32],
            sparsity: Some(0.1),
            energy: Some(0.1),
            ssm_readout: Some(0.1),
            ssm_digest: Some([2; 32]),
            world_digest: Some([3; 32]),
            risk_quality: 0,
            evidence_context_digest: [4; 32],
            evidence_world_digest: Some([5; 32]),
            evidence_spikes_digest: Some([6; 32]),
            evidence_ssm_digest: Some([7; 32]),
            evidence_lfm_digest: Some([8; 32]),
            ssm_quality: Some(ucf_compute::world_model::StageQuality::Ok),
            lfm_quality: Some(ucf_compute::world_model::StageQuality::Ok),
            backend_profile: "stub",
            backend_pack_id: 1,
            fixtures_digest: [9; 32],
            model_hashes_digest: [11; 32],
            llm_backend: 0,
            world_backend: 0,
            sae_backend: 0,
            ssm_backend: 0,
            lfm_backend: 0,
            lfm_uncertainty: Some(0.1),
            lfm_stability: Some(0.9),
            lfm_uncertainty_q: Some(0),
            lfm_stability_q: Some(0),
            lfm_state_norm: Some(0.1),
            lfm_deriv_norm: Some(0.1),
            lfm_saturation_ratio: Some(0.0),
            lfm_nan_inf_detected: false,
            lfm_digest: Some([10; 32]),
            budget_profile_id: 1,
            seed: 1,
            risk_contract_version: 1,
            compute_schema_version: 1,
            compute_chain_digest: [11; 32],
            compute_code_version: "v1",
            budget_exceeded_stage: None,
            contract_version: 1,
            backend_id: 0,
            validation_status: ucf_compute::ValidationStatus::Ok,
            violation_reason_mask: 0,
        }
    }

    #[test]
    fn prompt_builder_is_deterministic_and_bounded() {
        let signals = PromptConditioning {
            risk: Some(0.2),
            confidence: Some(0.8),
            surprise: 0.1,
            pressure: 0.3,
            uncertainty: Some(0.4),
            coherence: Some(0.7),
            instability: Some(0.2),
            evidence_chain_digest: [9; 32],
            lfm_readout_digest: Some([3; 32]),
        };
        let a = build_prompt(
            &"x".repeat(4000),
            &"y".repeat(4000),
            signals,
            None,
            CandidateOutputClass::SafeText,
        );
        let b = build_prompt(
            &"x".repeat(4000),
            &"y".repeat(4000),
            signals,
            None,
            CandidateOutputClass::SafeText,
        );
        assert_eq!(a, b);
        assert!(a.len() <= MAX_LLM_PROMPT_BYTES);
        assert!(a.contains("Signals: risk=0.200 confidence=0.800"));
    }

    #[test]
    fn max_tokens_eff_decreases_with_uncertainty() {
        let base = 256;
        let low = compute_max_tokens_eff(base, Some(0.1));
        let high = compute_max_tokens_eff(base, Some(0.9));
        assert!(high <= low);
        assert!(low <= base);
        assert!(high >= MIN_UNCERTAINTY_TOKENS.min(base));
    }

    #[test]
    fn override_policy_is_auditable() {
        let out = apply_decoding_policy(
            200,
            CandidateOutputClass::Code,
            PolicyHint::SafeOnly,
            Some(0.95),
            Some(0.2),
            Some(1.0),
            false,
        );
        assert_eq!(out.output_class, CandidateOutputClass::SafeText);
        assert_eq!(
            out.output_override,
            Some(OutputOverrideCode::ForcedSafeOnly)
        );
        let reasons = out.reason_codes();
        assert!(reasons.contains(&OverrideReasonCode::NsrSafeOnly.as_u16()));
        assert!(reasons.contains(&OverrideReasonCode::HighUncertainty.as_u16()));
        assert!(reasons.contains(&OverrideReasonCode::LowStability.as_u16()));
    }
    #[test]
    fn fails_fast_on_bad_policy_bundle_hash() {
        let original = std::env::var("UCF_POLICY_BUNDLE_SHA256").ok();
        std::env::set_var("UCF_POLICY_BUNDLE_SHA256", "deadbeef");
        let outcome = std::panic::catch_unwind(RuntimeOrchestrator::new);
        assert!(outcome.is_err());
        if let Some(v) = original {
            std::env::set_var("UCF_POLICY_BUNDLE_SHA256", v);
        } else {
            std::env::remove_var("UCF_POLICY_BUNDLE_SHA256");
        }
    }

    #[test]
    fn hormone_stress_gate_tightens_llm_budget() {
        let low_stress = apply_decoding_policy(
            256,
            CandidateOutputClass::SafeText,
            PolicyHint::Normal,
            Some(0.2),
            Some(0.8),
            Some(1.0),
            false,
        );
        let high_stress = apply_decoding_policy(
            256,
            CandidateOutputClass::SafeText,
            PolicyHint::Normal,
            Some(0.2),
            Some(0.8),
            Some(0.3),
            false,
        );
        assert!(high_stress.max_tokens_eff <= low_stress.max_tokens_eff);
    }

    #[test]
    fn emergency_decoding_policy_forces_safe_short_output() {
        let out = apply_decoding_policy(
            512,
            CandidateOutputClass::Code,
            PolicyHint::Normal,
            Some(0.1),
            Some(0.9),
            Some(1.0),
            true,
        );
        assert_eq!(out.output_class, CandidateOutputClass::SafeText);
        assert_eq!(out.max_tokens_eff, EMERGENCY_MAX_TOKENS);
    }

    #[test]
    fn emergency_state_transitions_on_runaway_and_cooldown() {
        if !std::path::Path::new("policies/manifest.toml").exists() {
            return;
        }
        if std::env::var("UCF_POLICY_BUNDLE_SHA256").is_err() {
            if let Ok(manifest) = crate::io_caps::IoCaps::runtime_default()
                .read_to_string(std::path::Path::new("manifest.toml"))
            {
                if let Some(hash_line) = manifest.lines().find(|l| l.starts_with("bundle_sha256 ="))
                {
                    let hash = hash_line
                        .split('=')
                        .nth(1)
                        .map(str::trim)
                        .unwrap_or("")
                        .trim_matches('"')
                        .to_string();
                    if !hash.is_empty() {
                        std::env::set_var("UCF_POLICY_BUNDLE_SHA256", hash);
                    }
                }
            }
        }
        let mut orchestrator = RuntimeOrchestrator::new();
        let time = ucf_core::types::SimTime {
            tick: ucf_core::types::Tick::new(10),
            window: ucf_core::types::WindowId::new(0),
        };
        let corr = ucf_frames::v1::CorrelationId(7);
        let mut summary = compute_summary_fixture();
        summary.lfm_state_norm = Some(1.0);
        summary.lfm_deriv_norm = Some(1.0);
        summary.lfm_saturation_ratio = Some(0.0);
        summary.lfm_nan_inf_detected = false;
        let _ = orchestrator
            .update_emergency_state(time, corr, &summary)
            .expect("emergency update");
        assert!(orchestrator.emergency_active());

        let mut stable = summary;
        stable.lfm_state_norm = Some(0.1);
        stable.lfm_deriv_norm = Some(0.0);
        for step in 0..=EMERGENCY_COOLDOWN_TICKS {
            let t = ucf_core::types::SimTime {
                tick: ucf_core::types::Tick::new(11 + u64::from(step)),
                window: ucf_core::types::WindowId::new(0),
            };
            let _ = orchestrator
                .update_emergency_state(t, corr, &stable)
                .expect("update");
        }
        assert!(!orchestrator.emergency_active());
    }

    #[test]
    fn runtime_panics_if_training_mode_env_is_enabled() {
        let original = std::env::var("UCF_EBM_TRAINING_MODE").ok();
        std::env::set_var("UCF_EBM_TRAINING_MODE", "1");
        let outcome = std::panic::catch_unwind(RuntimeOrchestrator::new);
        assert!(outcome.is_err());
        if let Some(v) = original {
            std::env::set_var("UCF_EBM_TRAINING_MODE", v);
        } else {
            std::env::remove_var("UCF_EBM_TRAINING_MODE");
        }
    }
}
