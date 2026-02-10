#![forbid(unsafe_code)]

use std::fmt;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use blake3::Hasher;
use ucf::boundary::{self, v1::WorkspaceBroadcastV1, v1::WorkspaceSignalV1};
use ucf_ai_host::{AiHost, AiInputFrame, ChannelLabel, MockAiHost};
use ucf_ai_port::{
    AiInference, AiOutput, AiPort, AiPortWorker, OutputChannel, OutputSuppressed,
    OutputSuppressionSink, SpeechGate,
};
use ucf_archive::{build_compact_record, ExperienceAppender};
use ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind, RecordMeta};
use ucf_attn_controller::{
    AttentionEventSink, AttentionUpdated, AttentionWeights, AttnController, AttnInputs,
    FocusChannel,
};
use ucf_bluebrain_port::{BlueBrainPort, NeuromodDelta};
use ucf_brain_mapper::map_to_stimulus;
use ucf_cde::{
    apply_edge_thresh_delta, apply_score_step_delta, derive_observation_commit,
    CausalEdge as CdeV1Edge, CausalEngine, CdeCore as CdeV1Core, CdeInputs as CdeV1Inputs,
    CdeOutputs as CdeV1Outputs, CdeParams, Edge as CdeGraphEdge, ObservationKey, VarId as CdeVarId,
};
use ucf_cde_scm::{
    CausalReport, CdeEngine, CdeInputs, CdeNodeId, CdeOutputs, CounterfactualResult,
};
use ucf_commit::canonical_control_frame_len;
use ucf_consistency_engine::{
    ConsistencyAction, ConsistencyActionKind, ConsistencyEngine, ConsistencyInputs,
    ConsistencyReport, DriftBand,
};
use ucf_coupling::{CouplingCore, CouplingInputs, CouplingOutputs, SignalId, SignalSample};
use ucf_digital_brain::DigitalBrainPort;
use ucf_feature_spiker::{
    apply_feature_thresh_delta, apply_threat_thresh_delta, FeatureSpikeParams,
};
use ucf_feature_translator::{
    ActivationView, LensPort as FeatureLensPort, LensSelection, MockLensPort, MockSaePort,
    SaePort as FeatureSaePort,
};
use ucf_geist::{SelfState, SelfStateBuilder};
use ucf_iit::{IitCore, IitInputs, IitOutput, IitParams, IntegrationMonitor};
use ucf_iit_monitor::{
    actions_for_phi, report_for_phi, IitAction, IitActionKind, IitBand, IitReport,
};
use ucf_influence::{InfluenceGraphV2, InfluenceInputs, InfluenceNodeId, InfluenceOutputs};
use ucf_ism::IsmStore;
use ucf_jepa::{JepaCore, JepaInputs, JepaOutputs, WorldModel as JepaWorldModel};
use ucf_ncde::{
    apply_gain_phase_delta, apply_gain_spike_delta, apply_leak_delta, ContinuousDynamics, NcdeCore,
    NcdeInputs, NcdeOutputs, NcdeParams,
};
use ucf_nsr::{Fact, NeuroSymbolicReasoner, NsrCore, NsrInputs, NsrOutputs, RuleHit, RuleSeverity};
use ucf_nsr_port::{light_report, ActionIntent, NsrInput, NsrPort, NsrReport, NsrVerdict};
use ucf_onn::{
    apply_coupling_delta, apply_lock_window_delta, OnnCore, OnnInputs, OnnParams, PhaseBus,
    PhaseLockDecision, PhaseProvider,
};
use ucf_output_router::{
    GateBundle, NsrSummary, OutputRouter, OutputRouterEvent, RouterConfig, SandboxVerdict,
};
use ucf_params_registry::{commit_snapshot_chain, ParamSnapshot};
use ucf_policy_ecology::RiskGateResult;
use ucf_policy_gateway::PolicyEvaluator;
use ucf_predictive_coding::{
    band_for_score, error, surprise, Observation, PredictionError, SurpriseBand, SurpriseSignal,
    SurpriseUpdated, WorldModel, WorldStateVec,
};
use ucf_recursion_controller::{RecursionBudget, RecursionController, RecursionInputs};
use ucf_risk_gate::{digest_reasons, RiskGate};
use ucf_rsa::v0 as rsa_v0;
use ucf_rsa::v0::{ParamTarget, RsaCore, RsaInputs, RsaOutputs};
use ucf_rsa_hooks::{MockRsaHook, RsaContext, RsaHook, RsaProposal};
use ucf_sandbox::{
    AiCallRequest, ControlFrameNormalized, IntentSummary, MockWasmSandbox, SandboxBudget,
    SandboxCaps, SandboxPort, SandboxReport,
};
use ucf_sle::{SelfReflex, SleCore, SleEngine, SleInputs, SleOutputs, StrangeLoop};
use ucf_spikebus::{
    producers::{MockLensProducer, MockSaeProducer, SpikeProducer},
    ModuleId, Spike, SpikeBus, SpikeInputs, SpikeKind, SpikeOutputs, SpikeParams, SpikeRouter,
};
use ucf_ssm::{SsmCore, SsmInputs, SsmOutputs, SsmParams, WorkingMemory};
use ucf_ssm_port::SsmState;
use ucf_structural_store::{
    OnnKnobs, SnnKnobs, StructuralCycleStats, StructuralDeltaProposal, StructuralParams,
    StructuralStore,
};
use ucf_tcf::{TcfCore, TcfInputs, TcfPlan, TemporalCoordinator};
use ucf_tcf_port::{
    ai_mode_for_pulse, apply_attn_k_delta, apply_energy_k_delta, apply_replay_k_delta,
    idle_attention, CyclePlan, CyclePlanned, DeterministicTcf, PulseKind, TcfConfig, TcfPort,
};
use ucf_tom_port::{IntentType, TomPort};
use ucf_types::v1::spec::{ControlFrame, DecisionKind, Digest, ExperienceRecord, PolicyDecision};
use ucf_types::{AlgoId, Digest32, EvidenceId, GainBudget, LearningSignal, StructuralDelta};
use ucf_workspace::{
    output_event_commit, NsrHitSummary, NsrTraceSummary, SignalKind, SleOutputsSnapshot, Workspace,
    WorkspaceConfig, WorkspaceSignal, WorkspaceSnapshot,
};

const ISM_ANCHOR_TOP_K: usize = 4;
const FEATURE_SIGNAL_PRIORITY: u16 = 3200;
const FEATURE_RECORD_KIND: u16 = 42;
const SANDBOX_DENIED_RECORD_KIND: u16 = 73;
const CAUSAL_REPORT_RECORD_KIND: u16 = 91;
const CDE_OUTPUT_RECORD_KIND: u16 = 92;
const AI_HOST_MODULE_ID: u8 = 11;
const PHASE_FRAME_RECORD_KIND: u16 = 118;
const SPIKE_RECORD_KIND: u16 = 131;
const NCDE_RECORD_KIND: u16 = 142;
const SSM_RECORD_KIND: u16 = 149;
const UPDATE_MODE_RECORD_KIND: u16 = 156;
const JEPA_RECORD_KIND: u16 = 160;
const NSR_HIT_SUMMARY_MAX: usize = 8;
const ONN_COHERENCE_THROTTLE: u16 = 2000;
const ONN_COHERENCE_THROTTLE_RESTRICT: u16 = 3500;
const LOCK_MIN_SPEAK: u16 = 3000;
const PHI_OUTPUT_THRESHOLD: u16 = 3200;
const COHERENCE_LAG_LEN: usize = 4;
const CDE_SURPRISE_HIGH_THRESHOLD: u16 = 8_000;
const CDE_EDGE_WEIGHT_POSITIVE: i16 = 6_000;
const CDE_EDGE_WEIGHT_NEGATIVE: i16 = -6_000;
const GAIN_BUDGET_MAX: u16 = 10_000;
const GAIN_BUDGET_RELAX_STEP: u16 = 200;
const GAIN_BUDGET_RELAX_WINDOW: u8 = 4;
const GAIN_BUDGET_STABLE_MIN: u8 = 12;
const LOW_PLV_THRESHOLD: u16 = 2500;
const HIGH_NOVELTY_THRESHOLD: u16 = 8000;
const LEARNING_SURPRISE_HIGH: u16 = 7000;
const TRIGGER_VIOLATION: u8 = 1;
const TRIGGER_LOW_PLV: u8 = 1 << 1;
const TRIGGER_HIGH_NOVELTY: u8 = 1 << 2;
const COHERENCE_RISK_HIGH: u16 = 7000;
const COHERENCE_PLV_LOW: u16 = 3200;
const RSA_REASON_MODE_BLOCK: u32 = 1 << 4;
const CAUSAL_REPORT_FLAG_LIGHT: u16 = 0b1000;
const SELF_CONSISTENCY_OK_THRESHOLD: u16 = 5000;
const NSR_POLICY_COMMIT_DOMAIN: &[u8] = b"ucf.nsr.policy.commit.v1";

pub const PIPELINE: &[&str] = &[
    "onn", "spikebus", "coupling", "jepa", "iit", "tcf", "nsr", "sle", "ncde", "ssm", "cde",
    "output", "archive",
];

#[derive(Debug)]
pub enum RouterError {
    PolicyDenied(i32),
}

pub struct RuntimeModules {
    pub phase: Box<dyn PhaseProvider + Send>,
    pub spikes: Box<dyn SpikeRouter + Send + Sync>,
    pub world: Box<dyn JepaWorldModel + Send>,
    pub ssm: Box<dyn WorkingMemory + Send>,
    pub ncde: Box<dyn ContinuousDynamics + Send>,
    pub cde: Box<dyn CausalEngine + Send>,
    pub nsr: Box<dyn NeuroSymbolicReasoner + Send + Sync>,
    pub tcf: Box<dyn TemporalCoordinator + Send>,
    pub iit: Box<dyn IntegrationMonitor + Send>,
    pub sle: Box<dyn StrangeLoop + Send>,
    pub ai_host: Box<dyn AiHost + Send + Sync>,
}

impl RuntimeModules {
    pub fn v1_defaults() -> Self {
        Self {
            phase: Box::new(OnnCore::default()),
            spikes: Box::new(SpikeBus::default()),
            world: Box::new(JepaCore::default()),
            ssm: Box::new(SsmCore::new(SsmParams::default())),
            ncde: Box::new(NcdeCore::new(NcdeParams::default())),
            cde: Box::new(CdeV1Core::new()),
            nsr: Box::new(NsrCore::default()),
            tcf: Box::new(TcfCore::default()),
            iit: Box::new(IitCore::default()),
            sle: Box::new(SleCore::default()),
            ai_host: Box::new(MockAiHost::default()),
        }
    }
}

impl fmt::Display for RouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RouterError::PolicyDenied(kind) => {
                write!(f, "policy decision denied routing (kind={kind})")
            }
        }
    }
}

impl std::error::Error for RouterError {}

pub struct Router {
    policy: Arc<dyn PolicyEvaluator + Send + Sync>,
    archive: Arc<dyn ExperienceAppender + Send + Sync>,
    archive_store: Arc<dyn ArchiveStore + Send + Sync>,
    archive_appender: Mutex<ArchiveAppender>,
    digital_brain: Option<Arc<dyn DigitalBrainPort + Send + Sync>>,
    bluebrain_port: Mutex<Option<Box<dyn BlueBrainPort + Send + Sync>>>,
    ai_port: Arc<dyn AiPort + Send + Sync>,
    sandbox_port: Mutex<Box<dyn SandboxPort + Send + Sync>>,
    sandbox_inference_cache: Arc<Mutex<Option<AiInference>>>,
    feature_sae: Arc<dyn FeatureSaePort + Send + Sync>,
    feature_lens: Arc<dyn FeatureLensPort + Send + Sync>,
    speech_gate: Arc<dyn SpeechGate + Send + Sync>,
    risk_gate: Arc<dyn RiskGate + Send + Sync>,
    nsr_port: Arc<NsrPort>,
    tom_port: Arc<dyn TomPort + Send + Sync>,
    output_suppression_sink: Option<Arc<dyn OutputSuppressionSink + Send + Sync>>,
    attention_controller: Option<AttnController>,
    attention_sink: Option<Arc<dyn AttentionEventSink + Send + Sync>>,
    output_router: Mutex<OutputRouter>,
    output_router_base: RouterConfig,
    workspace: Arc<Mutex<Workspace>>,
    workspace_base: WorkspaceConfig,
    cycle_counter: AtomicU64,
    force_stabilize_cycles: AtomicU16,
    tcf_port: Mutex<Box<dyn TcfPort + Send + Sync>>,
    last_tcf_plan: Mutex<Option<TcfPlan>>,
    last_phase_bus: Mutex<Option<PhaseBus>>,
    last_phase_lock: Mutex<Option<PhaseLockDecision>>,
    feature_params: Mutex<FeatureSpikeParams>,
    last_attention: Mutex<AttentionWeights>,
    last_surprise: Mutex<Option<SurpriseSignal>>,
    last_jepa_output: Mutex<Option<JepaOutputs>>,
    stage_trace: Option<Arc<dyn StageTrace + Send + Sync>>,
    world_model: WorldModel,
    world_state: Mutex<Option<WorldStateVec>>,
    sle_engine: Arc<SleEngine>,
    consistency_engine: ConsistencyEngine,
    ism_store: Arc<Mutex<IsmStore>>,
    last_iit_hints: Mutex<IitHintState>,
    recursion_controller: RecursionController,
    rsa_hooks: Vec<Arc<dyn RsaHook + Send + Sync>>,
    structural_store: Mutex<StructuralStore>,
    nsr_warn_streak: Mutex<u16>,
    last_self_state: Mutex<Option<SelfState>>,
    last_workspace_snapshot: Mutex<Option<WorkspaceSnapshot>>,
    last_recursion_budget: Mutex<Option<RecursionBudget>>,
    last_nsr_report: Mutex<Option<NsrReport>>,
    pending_neuromod_delta: Mutex<Option<NeuromodDelta>>,
    last_brain_response_commit: Mutex<Option<Digest32>>,
    last_brain_arousal: Mutex<u16>,
    cde_engine: Mutex<CdeEngine>,
    last_cde_output: Mutex<Option<CdeOutputs>>,
    last_cde_v1_output: Mutex<Option<CdeV1Outputs>>,
    influence_state: Mutex<InfluenceGraphV2>,
    last_influence_outputs: Mutex<Option<InfluenceOutputs>>,
    last_influence_root_commit: Mutex<Option<Digest32>>,
    coupling_core: Mutex<CouplingCore>,
    last_coupling_outputs: Mutex<Option<CouplingOutputs>>,
    coherence_lag: Mutex<CoherenceLag>,
    last_update_mode: Mutex<UpdateMode>,
    gain_budget_state: Mutex<BudgetState>,
    gain_budget_stable_cycles: Mutex<u8>,
    gain_budget_last_violation_count: Mutex<u16>,
    last_ssm_output: Mutex<Option<SsmOutputs>>,
    last_ncde_output: Mutex<Option<NcdeOutputs>>,
    runtime_modules: Mutex<RuntimeModules>,
    pending_ai_spikes: Mutex<Vec<Spike>>,
    rsa_core: Mutex<RsaCore>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterOutcome {
    pub evidence_id: EvidenceId,
    pub decision_kind: DecisionKind,
    pub speech_outputs: Vec<AiOutput>,
    pub integration_score: Option<u16>,
    pub workspace_snapshot_commit: Option<Digest32>,
    pub surprise_signal: Option<SurpriseSignal>,
    pub structural_stats: Option<StructuralCycleStats>,
    pub structural_proposal: Option<StructuralDeltaProposal>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OutputSuppressionInfo {
    channel: OutputChannel,
    reason_digest: Digest32,
    risk: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IitActionEffects {
    integration_bias: i16,
    broadcast_cap: usize,
    max_thought_frames_per_cycle: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IitHintState {
    tighten_sync: bool,
    damp_output: bool,
    damp_learning: bool,
    request_replay: bool,
    hints_commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConsistencyActionEffects {
    max_thought_frames_per_cycle: u16,
    noise_boost: u16,
    replay_boost: u16,
}

struct AttentionContext<'a> {
    policy_class: u16,
    risk_score: u16,
    integration_score: u16,
    integration_bias: i16,
    consistency_instability: u16,
    consistency_effects: Option<ConsistencyActionEffects>,
    tom_report: &'a ucf_tom_port::TomReport,
    surprise_score: u16,
    influence: Option<&'a InfluenceOutputs>,
    ssm_attention_gain: Option<u16>,
    lagged_plv: Option<u16>,
}

pub trait StageTrace {
    fn record(&self, stage: PulseKind);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UpdateMode {
    Conservative = 0,
    Normal = 1,
    Exploratory = 2,
    Stabilize = 3,
}

impl UpdateMode {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Conservative,
            1 => Self::Normal,
            2 => Self::Exploratory,
            _ => Self::Stabilize,
        }
    }

    fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug)]
struct CoherenceLag {
    phase_commit: [Digest32; COHERENCE_LAG_LEN],
    ssm_commit: [Digest32; COHERENCE_LAG_LEN],
    iit_commit: [Digest32; COHERENCE_LAG_LEN],
    nsr_verdict: [u8; COHERENCE_LAG_LEN],
    novelty: [u16; COHERENCE_LAG_LEN],
    salience: [u16; COHERENCE_LAG_LEN],
    plv: [u16; COHERENCE_LAG_LEN],
    commit: Digest32,
}

impl CoherenceLag {
    fn new() -> Self {
        let mut lag = Self {
            phase_commit: [Digest32::new([0u8; 32]); COHERENCE_LAG_LEN],
            ssm_commit: [Digest32::new([0u8; 32]); COHERENCE_LAG_LEN],
            iit_commit: [Digest32::new([0u8; 32]); COHERENCE_LAG_LEN],
            nsr_verdict: [0u8; COHERENCE_LAG_LEN],
            novelty: [0u16; COHERENCE_LAG_LEN],
            salience: [0u16; COHERENCE_LAG_LEN],
            plv: [0u16; COHERENCE_LAG_LEN],
            commit: Digest32::new([0u8; 32]),
        };
        lag.commit = commit_coherence_lag(&lag);
        lag
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        phase_commit: Digest32,
        ssm_commit: Digest32,
        iit_commit: Digest32,
        nsr_verdict: u8,
        novelty: u16,
        salience: u16,
        plv: u16,
    ) {
        self.phase_commit.rotate_right(1);
        self.ssm_commit.rotate_right(1);
        self.iit_commit.rotate_right(1);
        self.nsr_verdict.rotate_right(1);
        self.novelty.rotate_right(1);
        self.salience.rotate_right(1);
        self.plv.rotate_right(1);
        self.phase_commit[0] = phase_commit;
        self.ssm_commit[0] = ssm_commit;
        self.iit_commit[0] = iit_commit;
        self.nsr_verdict[0] = nsr_verdict;
        self.novelty[0] = novelty.min(10_000);
        self.salience[0] = salience.min(10_000);
        self.plv[0] = plv.min(10_000);
        self.commit = commit_coherence_lag(self);
    }

    fn avg_salience(&self) -> u16 {
        avg_u16(&self.salience)
    }

    fn avg_plv(&self) -> u16 {
        avg_u16(&self.plv)
    }

    fn novelty_trend_up(&self) -> bool {
        avg_recent(&self.novelty) > avg_prior(&self.novelty)
    }
}

fn commit_gain_budget(budget: &GainBudget) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.gain.budget.v1");
    hasher.update(&budget.master.to_be_bytes());
    hasher.update(&budget.coupling.to_be_bytes());
    hasher.update(&budget.ssm_update.to_be_bytes());
    hasher.update(&budget.ncde.to_be_bytes());
    hasher.update(&budget.tcf_attention.to_be_bytes());
    hasher.update(&budget.tcf_learning.to_be_bytes());
    hasher.update(&budget.onn_coupling.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn finalize_gain_budget(mut budget: GainBudget) -> GainBudget {
    budget.master = budget.master.min(GAIN_BUDGET_MAX);
    budget.coupling = budget.coupling.min(GAIN_BUDGET_MAX);
    budget.ssm_update = budget.ssm_update.min(GAIN_BUDGET_MAX);
    budget.ncde = budget.ncde.min(GAIN_BUDGET_MAX);
    budget.tcf_attention = budget.tcf_attention.min(GAIN_BUDGET_MAX);
    budget.tcf_learning = budget.tcf_learning.min(GAIN_BUDGET_MAX);
    budget.onn_coupling = budget.onn_coupling.min(GAIN_BUDGET_MAX);
    budget.commit = commit_gain_budget(&budget);
    budget
}

fn commit_budget_state(state: &BudgetState) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.gain.budget.state.v1");
    hasher.update(state.current.commit.as_bytes());
    hasher.update(&[
        state.low_plv_streak,
        state.high_novelty_streak,
        state.violation_streak,
        state.adapt_cooldown,
        state.spike_threshold_cooldown,
        state.tcf_learning_cooldown,
    ]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn default_budget_state() -> BudgetState {
    let budget = finalize_gain_budget(GainBudget::default());
    let mut state = BudgetState {
        current: budget,
        low_plv_streak: 0,
        high_novelty_streak: 0,
        violation_streak: 0,
        adapt_cooldown: 0,
        spike_threshold_cooldown: 0,
        tcf_learning_cooldown: 0,
        commit: Digest32::new([0u8; 32]),
    };
    state.commit = commit_budget_state(&state);
    state
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BudgetState {
    current: GainBudget,
    low_plv_streak: u8,
    high_novelty_streak: u8,
    violation_streak: u8,
    adapt_cooldown: u8,
    spike_threshold_cooldown: u8,
    tcf_learning_cooldown: u8,
    commit: Digest32,
}

struct StageContext {
    decision: Option<PolicyDecision>,
    decision_kind: DecisionKind,
    inference: Option<AiInference>,
    sandbox_report: Option<SandboxReport>,
    sandbox_verdict: Option<SandboxVerdict>,
    tom_report: Option<ucf_tom_port::TomReport>,
    nsr_report: Option<NsrReport>,
    nsr_output: Option<NsrOutputs>,
    causal_report: Option<CausalReport>,
    attention_risk: u16,
    thought_outputs: Vec<AiOutput>,
    speech_outputs: Vec<AiOutput>,
    output_intent: bool,
    suppressions: Vec<OutputSuppressionInfo>,
    integration_score: Option<u16>,
    integration_bias: i16,
    predictive_result: Option<(PredictionError, SurpriseSignal)>,
    jepa_outputs: Option<JepaOutputs>,
    attention_weights: Option<AttentionWeights>,
    lens_selection: Option<LensSelection>,
    evidence_id: Option<EvidenceId>,
    workspace_snapshot_commit: Option<Digest32>,
    self_state: Option<SelfState>,
    sle_reflex: Option<SelfReflex>,
    sle_outputs: Option<SleOutputs>,
    consistency_report: Option<ConsistencyReport>,
    consistency_actions: Vec<ConsistencyAction>,
    consistency_effects: Option<ConsistencyActionEffects>,
    iit_report: Option<IitReport>,
    iit_actions: Vec<IitAction>,
    iit_output: Option<IitOutput>,
    nsr_warn_streak: Option<u16>,
    recursion_budget: Option<RecursionBudget>,
    phase_commit: Option<Digest32>,
    phase_bus: Option<PhaseBus>,
    phase_lock: Option<PhaseLockDecision>,
    percept_commit: Option<Digest32>,
    percept_energy: Option<u16>,
    coherence_plv: Option<u16>,
    onn_outputs: Option<ucf_onn::OnnOutputs>,
    spike_outputs: Option<SpikeOutputs>,
    replay_pressure: Option<u16>,
    drift_score: Option<u16>,
    surprise_score: Option<u16>,
    tcf_energy_smooth: Option<u16>,
    tcf_plan: Option<TcfPlan>,
    influence_outputs: Option<InfluenceOutputs>,
    coupling_outputs: Option<CouplingOutputs>,
    structural_stats: Option<StructuralCycleStats>,
    structural_proposal: Option<StructuralDeltaProposal>,
    cde_output: Option<CdeOutputs>,
    cde_v1_output: Option<CdeV1Outputs>,
    ncde_output: Option<NcdeOutputs>,
    ssm_output: Option<SsmOutputs>,
    update_mode: Option<UpdateMode>,
    coherence_request_replay: bool,
}

impl StageContext {
    fn new() -> Self {
        Self {
            decision: None,
            decision_kind: DecisionKind::DecisionKindUnspecified,
            inference: None,
            sandbox_report: None,
            sandbox_verdict: None,
            tom_report: None,
            nsr_report: None,
            nsr_output: None,
            causal_report: None,
            attention_risk: 0,
            thought_outputs: Vec::new(),
            speech_outputs: Vec::new(),
            output_intent: false,
            suppressions: Vec::new(),
            integration_score: None,
            integration_bias: 0,
            predictive_result: None,
            jepa_outputs: None,
            attention_weights: None,
            lens_selection: None,
            evidence_id: None,
            workspace_snapshot_commit: None,
            self_state: None,
            sle_reflex: None,
            sle_outputs: None,
            consistency_report: None,
            consistency_actions: Vec::new(),
            consistency_effects: None,
            iit_report: None,
            iit_actions: Vec::new(),
            iit_output: None,
            nsr_warn_streak: None,
            recursion_budget: None,
            phase_commit: None,
            phase_bus: None,
            phase_lock: None,
            percept_commit: None,
            percept_energy: None,
            coherence_plv: None,
            onn_outputs: None,
            spike_outputs: None,
            replay_pressure: None,
            drift_score: None,
            surprise_score: None,
            tcf_energy_smooth: None,
            tcf_plan: None,
            influence_outputs: None,
            coupling_outputs: None,
            structural_stats: None,
            structural_proposal: None,
            cde_output: None,
            cde_v1_output: None,
            ncde_output: None,
            ssm_output: None,
            update_mode: None,
            coherence_request_replay: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BudgetCycle {
    budget: GainBudget,
    low_plv_streak: u8,
    high_novelty_streak: u8,
    violation_streak: u8,
    triggers: u8,
    request_replay: bool,
    stabilize_cycles: u16,
    learning_signal: LearningSignal,
    structural_delta: StructuralDelta,
    spike_novelty_threshold_bump: u16,
}

fn relax_budget(value: u16) -> u16 {
    value
        .saturating_add(GAIN_BUDGET_RELAX_STEP)
        .min(GAIN_BUDGET_MAX)
}

#[allow(clippy::too_many_arguments)]
fn commit_learning_signal(
    cycle_id: u64,
    learn_rate: u16,
    update_mass: u16,
    mode: u8,
    attention_gain: u16,
    novelty: u16,
    salience: u16,
    surprise: u16,
    plv: u16,
    nsr_trace_root: Digest32,
    ssm_commit: Digest32,
    jepa_commit: Digest32,
    onn_phase_commit: Digest32,
    cde_graph_commit: Digest32,
    tcf_attention_cap: u16,
    tcf_learning_cap: u16,
) -> LearningSignal {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.learning.signal.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&learn_rate.to_be_bytes());
    hasher.update(&update_mass.to_be_bytes());
    hasher.update(&[mode]);
    hasher.update(&attention_gain.to_be_bytes());
    hasher.update(&novelty.to_be_bytes());
    hasher.update(&salience.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    hasher.update(&plv.to_be_bytes());
    hasher.update(&tcf_attention_cap.to_be_bytes());
    hasher.update(&tcf_learning_cap.to_be_bytes());
    hasher.update(ssm_commit.as_bytes());
    hasher.update(jepa_commit.as_bytes());
    hasher.update(onn_phase_commit.as_bytes());
    hasher.update(nsr_trace_root.as_bytes());
    hasher.update(cde_graph_commit.as_bytes());
    LearningSignal {
        cycle_id,
        learn_rate,
        update_mass,
        mode,
        commit: Digest32::new(*hasher.finalize().as_bytes()),
    }
}

#[allow(clippy::too_many_arguments)]
fn commit_structural_delta(
    cycle_id: u64,
    learning: LearningSignal,
    ssm_state_digest: Digest32,
    world_state: Digest32,
    cde_graph_commit: Digest32,
    nsr_trace_root: Digest32,
    phase_commit: Digest32,
    plv: u16,
    novelty: u16,
    surprise: u16,
    nsr_warn_or_block: bool,
) -> StructuralDelta {
    let mut root_hasher = Hasher::new();
    root_hasher.update(b"ucf.structural.delta.root.v1");
    root_hasher.update(learning.commit.as_bytes());
    root_hasher.update(ssm_state_digest.as_bytes());
    root_hasher.update(world_state.as_bytes());
    root_hasher.update(cde_graph_commit.as_bytes());
    root_hasher.update(nsr_trace_root.as_bytes());
    root_hasher.update(phase_commit.as_bytes());
    let delta_root = Digest32::new(*root_hasher.finalize().as_bytes());

    let mut targets = [0u16; 4];
    targets[0] = if plv < 2500 { 1 } else { 0 };
    targets[1] = if novelty > 7000 { 2 } else { 0 };
    targets[2] = if surprise > 7000 { 3 } else { 0 };
    targets[3] = if nsr_warn_or_block { 4 } else { 0 };
    let delta_mass = learning.update_mass.min(learning.learn_rate);

    let mut hasher = Hasher::new();
    hasher.update(b"ucf.structural.delta.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(delta_root.as_bytes());
    hasher.update(&delta_mass.to_be_bytes());
    for target in targets {
        hasher.update(&target.to_be_bytes());
    }
    hasher.update(learning.commit.as_bytes());
    StructuralDelta {
        cycle_id,
        delta_root,
        delta_mass,
        targets,
        commit: Digest32::new(*hasher.finalize().as_bytes()),
    }
}

#[allow(clippy::too_many_arguments)]
fn update_budget_state(
    mut state: BudgetState,
    mut stable_cycles: u8,
    mut last_violation_count: u16,
    cycle_id: u64,
    attention_gain: u16,
    ssm_novelty: u16,
    ssm_salience: u16,
    surprise: u16,
    global_plv: u16,
    nsr_rule_hits: &[(u16, u16, u16)],
    tcf_attention_cap: u16,
    tcf_learning_cap: u16,
    ssm_commit: Digest32,
    jepa_commit: Digest32,
    onn_phase_commit: Digest32,
    nsr_trace_root: Digest32,
    cde_graph_commit: Digest32,
    ssm_state_digest: Digest32,
    world_state: Digest32,
    phase_commit: Digest32,
    leakage_violations: u16,
) -> (BudgetState, u8, u16, BudgetCycle) {
    let low_plv = global_plv < LOW_PLV_THRESHOLD;
    let high_novelty = ssm_novelty > HIGH_NOVELTY_THRESHOLD;
    let violation_increased = leakage_violations > last_violation_count;
    last_violation_count = leakage_violations;
    let nsr_has_block = nsr_rule_hits.iter().any(|(_, _, severity)| *severity >= 2);
    let nsr_has_warn = nsr_rule_hits.iter().any(|(_, _, severity)| *severity == 1);

    let base = i32::from(
        (attention_gain / 2)
            .saturating_add(ssm_novelty / 4)
            .saturating_add(ssm_salience / 4),
    );
    let mut coherence_penalty = 0i32;
    if global_plv < 2500 {
        coherence_penalty -= 2000;
    }
    if global_plv < 1500 {
        coherence_penalty -= 2000;
    }
    let nsr_penalty = if nsr_has_block {
        -3000
    } else if nsr_has_warn {
        -1000
    } else {
        0
    };
    let surprise_boost = if surprise > LEARNING_SURPRISE_HIGH {
        1500
    } else {
        0
    };
    let learn_rate =
        (base + coherence_penalty + nsr_penalty + surprise_boost).clamp(0, 10_000) as u16;
    let update_mass = ((u32::from(ssm_novelty) + u32::from(ssm_salience)) / 2).min(10_000) as u16;
    let mode = if learn_rate < 2500 {
        0
    } else if surprise > LEARNING_SURPRISE_HIGH || ssm_novelty > LEARNING_SURPRISE_HIGH {
        1
    } else {
        2
    };
    let learning_signal = commit_learning_signal(
        cycle_id,
        learn_rate,
        update_mass,
        mode,
        attention_gain,
        ssm_novelty,
        ssm_salience,
        surprise,
        global_plv,
        nsr_trace_root,
        ssm_commit,
        jepa_commit,
        onn_phase_commit,
        cde_graph_commit,
        tcf_attention_cap,
        tcf_learning_cap,
    );

    let structural_delta = commit_structural_delta(
        cycle_id,
        learning_signal,
        ssm_state_digest,
        world_state,
        cde_graph_commit,
        nsr_trace_root,
        phase_commit,
        global_plv,
        ssm_novelty,
        surprise,
        nsr_has_warn || nsr_has_block,
    );

    let low_plv_streak = if low_plv {
        state.low_plv_streak.saturating_add(1)
    } else {
        state.low_plv_streak.saturating_sub(1)
    };
    let high_novelty_streak = if high_novelty {
        state.high_novelty_streak.saturating_add(1)
    } else {
        state.high_novelty_streak.saturating_sub(1)
    };
    let violation_streak = if violation_increased {
        state.violation_streak.saturating_add(1)
    } else {
        state.violation_streak.saturating_sub(1)
    };

    let mut triggers = 0u8;
    let mut request_replay = false;
    let mut stabilize_cycles = 0u16;

    let mut budget = state.current;
    if violation_increased {
        triggers |= TRIGGER_VIOLATION;
    }
    if violation_streak >= 1 {
        budget.master = budget.master.min(6000);
        budget.coupling = budget.coupling.min(5000);
        budget.tcf_learning = budget.tcf_learning.min(4000);
        stabilize_cycles = stabilize_cycles.saturating_add(1);
    }
    if low_plv_streak >= 6 {
        triggers |= TRIGGER_LOW_PLV;
        budget.master = budget.master.min(7000);
        budget.onn_coupling = budget.onn_coupling.min(5000);
        request_replay = true;
        stabilize_cycles = stabilize_cycles.saturating_add(8);
    }
    if high_novelty_streak >= 6 {
        triggers |= TRIGGER_HIGH_NOVELTY;
        budget.tcf_attention = budget.tcf_attention.min(5000);
        budget.ssm_update = budget.ssm_update.min(6000);
    }

    if state.adapt_cooldown > 0 {
        state.adapt_cooldown = state.adapt_cooldown.saturating_sub(1);
    }
    if state.spike_threshold_cooldown > 0 {
        state.spike_threshold_cooldown = state.spike_threshold_cooldown.saturating_sub(1);
    }
    if state.tcf_learning_cooldown > 0 {
        state.tcf_learning_cooldown = state.tcf_learning_cooldown.saturating_sub(1);
    }

    if learning_signal.mode == 1 && structural_delta.delta_mass > 6000 {
        budget.master = budget.master.saturating_sub(500).max(6000);
        state.adapt_cooldown = 4;
    }
    if learning_signal.mode == 2 && global_plv > 8500 && !nsr_has_warn && !nsr_has_block {
        budget.master = budget.master.saturating_add(200).min(10_000);
    }
    if structural_delta.targets.contains(&1) && structural_delta.delta_mass > 6000 {
        budget.onn_coupling = budget.onn_coupling.saturating_sub(500).max(3000);
    }
    if structural_delta.targets.contains(&4) && nsr_has_warn {
        budget.tcf_learning = budget.tcf_learning.saturating_sub(500);
        state.tcf_learning_cooldown = 4;
    }

    if low_plv_streak == 0 && high_novelty_streak == 0 && violation_streak == 0 {
        stable_cycles = stable_cycles.saturating_add(1);
        if stable_cycles >= GAIN_BUDGET_STABLE_MIN
            && stable_cycles.is_multiple_of(GAIN_BUDGET_RELAX_WINDOW)
        {
            budget.master = relax_budget(budget.master);
            budget.coupling = relax_budget(budget.coupling);
            budget.ssm_update = relax_budget(budget.ssm_update);
            budget.ncde = relax_budget(budget.ncde);
            budget.tcf_attention = relax_budget(budget.tcf_attention);
            budget.tcf_learning = relax_budget(budget.tcf_learning);
            budget.onn_coupling = relax_budget(budget.onn_coupling);
        }
    } else {
        stable_cycles = 0;
    }

    budget = finalize_gain_budget(budget);
    state = BudgetState {
        current: budget,
        low_plv_streak,
        high_novelty_streak,
        violation_streak,
        adapt_cooldown: state.adapt_cooldown,
        spike_threshold_cooldown: state.spike_threshold_cooldown,
        tcf_learning_cooldown: state.tcf_learning_cooldown,
        commit: Digest32::new([0u8; 32]),
    };
    state.commit = commit_budget_state(&state);

    let cycle = BudgetCycle {
        budget,
        low_plv_streak,
        high_novelty_streak,
        violation_streak,
        triggers,
        request_replay,
        stabilize_cycles,
        learning_signal,
        structural_delta,
        spike_novelty_threshold_bump: if structural_delta.targets.contains(&3) {
            500
        } else {
            0
        },
    };
    (state, stable_cycles, last_violation_count, cycle)
}

impl Router {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        policy: Arc<dyn PolicyEvaluator + Send + Sync>,
        archive: Arc<dyn ExperienceAppender + Send + Sync>,
        archive_store: Arc<dyn ArchiveStore + Send + Sync>,
        digital_brain: Option<Arc<dyn DigitalBrainPort + Send + Sync>>,
        ai_port: Arc<dyn AiPort + Send + Sync>,
        speech_gate: Arc<dyn SpeechGate + Send + Sync>,
        risk_gate: Arc<dyn RiskGate + Send + Sync>,
        tom_port: Arc<dyn TomPort + Send + Sync>,
        output_suppression_sink: Option<Arc<dyn OutputSuppressionSink + Send + Sync>>,
    ) -> Self {
        let sandbox_worker = AiPortWorker::new(ai_port.clone());
        let sandbox_inference_cache = sandbox_worker.inference_cache();
        let sandbox_port: Box<dyn SandboxPort + Send + Sync> = Box::new(MockWasmSandbox::new(
            Box::new(sandbox_worker),
            SandboxCaps::default(),
        ));
        let output_router_base = RouterConfig {
            thought_capacity: 128,
            max_thought_frames_per_cycle: 32,
            external_enabled: true,
        };
        let workspace_base = WorkspaceConfig {
            cap: 64,
            broadcast_cap: 8,
        };
        Self {
            policy,
            archive,
            archive_store,
            archive_appender: Mutex::new(ArchiveAppender::new()),
            digital_brain,
            bluebrain_port: Mutex::new(None),
            ai_port,
            sandbox_port: Mutex::new(sandbox_port),
            sandbox_inference_cache,
            feature_sae: Arc::new(MockSaePort::new()),
            feature_lens: Arc::new(MockLensPort::new()),
            speech_gate,
            risk_gate,
            nsr_port: Arc::new(NsrPort::default()),
            tom_port,
            output_suppression_sink,
            attention_controller: Some(AttnController),
            attention_sink: None,
            output_router_base: output_router_base.clone(),
            output_router: Mutex::new(OutputRouter::new(output_router_base)),
            workspace_base,
            workspace: Arc::new(Mutex::new(Workspace::new(workspace_base))),
            cycle_counter: AtomicU64::new(0),
            force_stabilize_cycles: AtomicU16::new(0),
            tcf_port: Mutex::new(Box::new(DeterministicTcf::default())),
            last_tcf_plan: Mutex::new(None),
            last_phase_bus: Mutex::new(None),
            last_phase_lock: Mutex::new(None),
            feature_params: Mutex::new(FeatureSpikeParams::default()),
            last_attention: Mutex::new(idle_attention()),
            last_surprise: Mutex::new(None),
            last_jepa_output: Mutex::new(None),
            stage_trace: None,
            world_model: WorldModel::default(),
            world_state: Mutex::new(None),
            sle_engine: Arc::new(SleEngine::new(6)),
            consistency_engine: ConsistencyEngine,
            ism_store: Arc::new(Mutex::new(IsmStore::new(64))),
            last_iit_hints: Mutex::new(IitHintState {
                tighten_sync: false,
                damp_output: false,
                damp_learning: false,
                request_replay: false,
                hints_commit: Digest32::new([0u8; 32]),
            }),
            recursion_controller: RecursionController::default(),
            rsa_hooks: vec![Arc::new(MockRsaHook::new())],
            structural_store: Mutex::new(StructuralStore::default()),
            nsr_warn_streak: Mutex::new(0),
            last_self_state: Mutex::new(None),
            last_workspace_snapshot: Mutex::new(None),
            last_recursion_budget: Mutex::new(None),
            last_nsr_report: Mutex::new(None),
            pending_neuromod_delta: Mutex::new(None),
            last_brain_response_commit: Mutex::new(None),
            last_brain_arousal: Mutex::new(0),
            cde_engine: Mutex::new(CdeEngine::new()),
            last_cde_output: Mutex::new(None),
            last_cde_v1_output: Mutex::new(None),
            influence_state: Mutex::new(InfluenceGraphV2::new_default()),
            last_influence_outputs: Mutex::new(None),
            last_influence_root_commit: Mutex::new(None),
            coupling_core: Mutex::new(CouplingCore::new_default()),
            last_coupling_outputs: Mutex::new(None),
            coherence_lag: Mutex::new(CoherenceLag::new()),
            last_update_mode: Mutex::new(UpdateMode::Normal),
            gain_budget_state: Mutex::new(default_budget_state()),
            gain_budget_stable_cycles: Mutex::new(0),
            gain_budget_last_violation_count: Mutex::new(0),
            last_ssm_output: Mutex::new(None),
            last_ncde_output: Mutex::new(None),
            runtime_modules: Mutex::new(RuntimeModules::v1_defaults()),
            pending_ai_spikes: Mutex::new(Vec::new()),
            rsa_core: Mutex::new(RsaCore::default()),
        }
    }

    pub fn with_attention_sink(mut self, sink: Arc<dyn AttentionEventSink + Send + Sync>) -> Self {
        self.attention_sink = Some(sink);
        self
    }

    pub fn with_sandbox_port(mut self, port: Box<dyn SandboxPort + Send + Sync>) -> Self {
        self.sandbox_port = Mutex::new(port);
        self.sandbox_inference_cache = Arc::new(Mutex::new(None));
        self
    }

    pub fn with_nsr_port(mut self, port: Arc<NsrPort>) -> Self {
        self.nsr_port = port;
        self
    }

    pub fn with_tcf_port(mut self, port: Box<dyn TcfPort + Send + Sync>) -> Self {
        self.tcf_port = Mutex::new(port);
        self
    }

    pub fn with_bluebrain_port(mut self, port: Box<dyn BlueBrainPort + Send + Sync>) -> Self {
        self.bluebrain_port = Mutex::new(Some(port));
        self
    }

    pub fn with_stage_trace(mut self, trace: Arc<dyn StageTrace + Send + Sync>) -> Self {
        self.stage_trace = Some(trace);
        self
    }

    pub fn disable_attention(mut self) -> Self {
        self.attention_controller = None;
        self
    }

    pub fn workspace_handle(&self) -> Arc<Mutex<Workspace>> {
        Arc::clone(&self.workspace)
    }

    pub fn last_workspace_snapshot(&self) -> Option<WorkspaceSnapshot> {
        self.last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }

    pub fn force_stabilize_cycles(&self) -> u16 {
        self.force_stabilize_cycles.load(Ordering::Relaxed)
    }

    pub fn pending_neuromod_delta(&self) -> Option<NeuromodDelta> {
        self.pending_neuromod_delta
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }

    fn current_gain_budget(&self) -> GainBudget {
        self.gain_budget_state
            .lock()
            .map(|state| state.current)
            .unwrap_or_else(|err| err.into_inner().current)
    }

    fn refresh_gain_budget(&self, cycle_id: u64, ctx: Option<&StageContext>) -> BudgetCycle {
        let snapshot = self.last_workspace_snapshot();
        let global_plv = ctx
            .and_then(|c| c.coherence_plv)
            .or_else(|| snapshot.as_ref().map(|snap| snap.onn_global_plv))
            .unwrap_or(0);
        let ssm_novelty = ctx
            .and_then(|c| c.ssm_output.as_ref().map(|o| o.ssm_novelty))
            .or_else(|| snapshot.as_ref().map(|snap| snap.ssm_novelty))
            .unwrap_or(0);
        let ssm_salience = ctx
            .and_then(|c| c.ssm_output.as_ref().map(|o| o.ssm_salience))
            .or_else(|| snapshot.as_ref().map(|snap| snap.ssm_salience))
            .unwrap_or(0);
        let attention_gain = ctx
            .and_then(|c| c.ssm_output.as_ref().map(|o| o.ssm_attention_gain))
            .or_else(|| snapshot.as_ref().map(|snap| snap.ssm_attention_gain))
            .unwrap_or(0);
        let surprise = ctx
            .and_then(|c| c.jepa_outputs.as_ref().map(|o| o.surprise))
            .or_else(|| snapshot.as_ref().map(|snap| snap.jepa_surprise))
            .unwrap_or(0);
        let tcf_attention_cap = snapshot
            .as_ref()
            .map(|snap| snap.tcf_attention_gain_cap)
            .unwrap_or(10_000);
        let tcf_learning_cap = snapshot
            .as_ref()
            .map(|snap| snap.tcf_learning_gain_cap)
            .unwrap_or(10_000);
        let ssm_commit = ctx
            .and_then(|c| c.ssm_output.as_ref().map(|o| o.commit))
            .or_else(|| snapshot.as_ref().map(|snap| snap.ssm_commit))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let jepa_commit = ctx
            .and_then(|c| c.jepa_outputs.as_ref().map(|o| o.commit))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let onn_phase_commit = snapshot
            .as_ref()
            .map(|snap| snap.onn_phase_commit)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let nsr_trace_root = ctx
            .and_then(|c| c.nsr_output.as_ref().map(|o| o.trace_root))
            .or_else(|| snapshot.as_ref().and_then(|snap| snap.nsr_trace_root))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let cde_graph_commit = snapshot
            .as_ref()
            .map(|snap| snap.cde_graph_commit)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let ssm_state_digest = ctx
            .and_then(|c| c.ssm_output.as_ref().map(|o| o.ssm_state_digest))
            .or_else(|| snapshot.as_ref().map(|snap| snap.ssm_state_digest))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let world_state = ctx
            .and_then(|c| c.jepa_outputs.as_ref().map(|o| o.world_state))
            .or_else(|| snapshot.as_ref().map(|snap| snap.jepa_world_state))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let phase_commit = snapshot
            .as_ref()
            .map(|snap| snap.onn_phase_commit)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let nsr_rule_hits: Vec<(u16, u16, u16)> = ctx
            .and_then(|c| {
                c.nsr_report.as_ref().map(|r| {
                    r.violations
                        .iter()
                        .map(|v| {
                            (
                                0u16,
                                0u16,
                                if v.severity >= 2 {
                                    2
                                } else if v.severity >= 1 {
                                    1
                                } else {
                                    0
                                },
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .unwrap_or_default();
        let leakage_violations = self
            .last_nsr_report
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|report| report.violations.len()))
            .unwrap_or(0)
            .min(usize::from(u16::MAX)) as u16;

        let mut state_guard = self
            .gain_budget_state
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut stable_guard = self
            .gain_budget_stable_cycles
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut violation_guard = self
            .gain_budget_last_violation_count
            .lock()
            .unwrap_or_else(|err| err.into_inner());

        let (next_state, next_stable, next_violation_count, cycle) = update_budget_state(
            *state_guard,
            *stable_guard,
            *violation_guard,
            cycle_id,
            attention_gain,
            ssm_novelty,
            ssm_salience,
            surprise,
            global_plv,
            &nsr_rule_hits,
            tcf_attention_cap,
            tcf_learning_cap,
            ssm_commit,
            jepa_commit,
            onn_phase_commit,
            nsr_trace_root,
            cde_graph_commit,
            ssm_state_digest,
            world_state,
            phase_commit,
            leakage_violations,
        );
        *state_guard = next_state;
        *stable_guard = next_stable;
        *violation_guard = next_violation_count;

        if cycle.stabilize_cycles > 0 {
            self.force_stabilize_cycles
                .fetch_add(cycle.stabilize_cycles, Ordering::Relaxed);
        }
        if let Ok(mut workspace) = self.workspace.lock() {
            workspace.set_gain_budget_state(
                cycle.budget.commit,
                cycle.low_plv_streak,
                cycle.high_novelty_streak,
                cycle.violation_streak,
                cycle.triggers,
            );
            workspace.set_learning_signal(cycle.learning_signal);
            workspace.set_structural_delta(cycle.structural_delta);
        }
        cycle
    }

    fn latest_workspace_snapshot(&self, cycle_id: u64) -> WorkspaceSnapshot {
        self.last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
            .unwrap_or(WorkspaceSnapshot {
                cycle_id,
                broadcast: Vec::new(),
                recursion_used: 0,
                spike_accepted_root: Digest32::new([0u8; 32]),
                spike_counts: Vec::new(),
                spike_max_intensity: 0,
                ncde_commit: Digest32::new([0u8; 32]),
                ncde_state_digest: Digest32::new([0u8; 32]),
                ncde_energy: 0,
                replay_pressure_hint: 0,
                cde_commit: Digest32::new([0u8; 32]),
                cde_graph_commit: Digest32::new([0u8; 32]),
                cde_top_edges: Vec::new(),
                cde_top_edge_commits: Vec::new(),
                cde_intervention_commit: None,
                cde_observation_commit: Digest32::new([0u8; 32]),
                cde_last_query_result: None,
                ssm_commit: Digest32::new([0u8; 32]),
                ssm_state_commit: Digest32::new([0u8; 32]),
                ssm_state_digest: Digest32::new([0u8; 32]),
                ssm_salience: 0,
                ssm_novelty: 0,
                ssm_attention_gain: 0,
                jepa_world_state: Digest32::new([0u8; 32]),
                jepa_prediction: Digest32::new([0u8; 32]),
                jepa_surprise: 0,
                learning_signal_commit: Digest32::new([0u8; 32]),
                learning_signal_learn_rate: 0,
                learning_signal_mode: 0,
                structural_delta_commit: Digest32::new([0u8; 32]),
                structural_delta_mass: 0,
                structural_delta_targets: [0; 4],
                influence_v2_commit: Digest32::new([0u8; 32]),
                influence_pulses_root: Digest32::new([0u8; 32]),
                influence_node_values: Vec::new(),
                coupling_influences_root: Digest32::new([0u8; 32]),
                coupling_top_influences: Vec::new(),
                coupling_lag_commits: Vec::new(),
                gain_budget_commit: Digest32::new([0u8; 32]),
                budget_low_plv_streak: 0,
                budget_high_novelty_streak: 0,
                budget_violation_streak: 0,
                budget_triggers: 0,
                tcf_plan_commit: Digest32::new([0u8; 32]),
                tcf_attention_gain_cap: 0,
                tcf_learning_gain_cap: 0,
                tcf_output_gain_cap: 0,
                tcf_sleep_active: false,
                tcf_replay_active: false,
                tcf_lock_window_buckets: 0,
                lock_window_source_cycle: 0,
                coherence_lag_commit: Digest32::new([0u8; 32]),
                update_mode: 0,
                onn_phase_commit: Digest32::new([0u8; 32]),
                onn_gamma_bucket: 0,
                onn_global_plv: 0,
                iit_output: None,
                nsr_trace_root: None,
                nsr_prev_commit: None,
                nsr_verdict: None,
                nsr_triggered_rules_root: None,
                nsr_derived_facts_root: None,
                nsr_fact_flags: 0,
                nsr_hit_counts: [0u16; 3],
                nsr_hit_summaries: Vec::new(),
                rsa_commit: Digest32::new([0u8; 32]),
                rsa_proposal_commit: None,
                rsa_decision_apply: false,
                rsa_apply_allowed: false,
                rsa_apply_commit: Digest32::new([0u8; 32]),
                rsa_reason_mask: 0,
                rsa_applied_params_root: Digest32::new([0u8; 32]),
                rsa_snapshot_chain_commit: Digest32::new([0u8; 32]),
                sle_commit: Digest32::new([0u8; 32]),
                sle_reflection_commit: Digest32::new([0u8; 32]),
                sle_reflection_class: 0,
                sle_reflection_intensity: 0,
                sle_thought_only_root: Digest32::new([0u8; 32]),
                sle_ssm_bias: 0,
                sle_cde_bias: 0,
                sle_request_replay: false,
                internal_utterances: Vec::new(),
                commit: Digest32::new([0u8; 32]),
            })
    }

    pub fn handle_control_frame(
        &self,
        cf: ControlFrameNormalized,
    ) -> Result<RouterOutcome, RouterError> {
        let _cycle_seed = self.cycle_counter.fetch_add(1, Ordering::SeqCst);
        let plan_attn = self
            .last_attention
            .lock()
            .map(|attn| attn.clone())
            .unwrap_or_else(|_| idle_attention());
        let tighten_sync = self
            .last_iit_hints
            .lock()
            .map(|hints| hints.tighten_sync)
            .unwrap_or(false);
        let plan_surprise = self
            .last_surprise
            .lock()
            .ok()
            .and_then(|guard| guard.clone());

        let (cycle_plan, tcf_energy_smooth) = {
            let mut tcf = self.tcf_port.lock().expect("tcf lock");
            tcf.apply_sync_hint(tighten_sync);
            let plan = tcf.step(&plan_attn, plan_surprise.as_ref());
            let energy = tcf.state().energy;
            (plan, energy)
        };
        let planned = DeterministicTcf::planned_event(&cycle_plan);
        self.append_cycle_plan_record(&cycle_plan, &planned);

        let cycle_id = cycle_plan.cycle_id;
        let mut ctx = StageContext::new();
        ctx.tcf_energy_smooth = Some(tcf_energy_smooth);
        let budget_cycle = self.refresh_gain_budget(cycle_id, Some(&ctx));

        for pulse in &cycle_plan.pulses {
            self.emit_stage_trace(pulse.kind);
            match pulse.kind {
                PulseKind::Sense => {
                    let decision = self.policy.evaluate(cf.as_ref().clone());
                    self.publish_workspace_signal(WorkspaceSignal::from_policy_decision(
                        &decision,
                        None,
                        Some(pulse.slot),
                    ));
                    self.ensure_allowed(&decision)?;
                    ctx.decision_kind = DecisionKind::try_from(decision.kind)
                        .unwrap_or(DecisionKind::DecisionKindUnspecified);
                    ctx.decision = Some(decision);
                }
                PulseKind::Think => {
                    let mode = ai_mode_for_pulse(pulse.kind);
                    self.run_think_stage(&cf, &mut ctx, cycle_id, pulse.slot, mode);
                }
                PulseKind::Verify => {
                    if ctx.decision.is_none() {
                        continue;
                    }
                    if ctx.inference.is_none() {
                        let mode = ai_mode_for_pulse(PulseKind::Think);
                        self.run_think_stage(&cf, &mut ctx, cycle_id, pulse.slot, mode);
                    }
                    let decision = ctx.decision.clone().expect("decision available");
                    let Some(inference) = ctx.inference.clone() else {
                        continue;
                    };
                    let workspace_snapshot = self.latest_workspace_snapshot(cycle_id);
                    let intent = self.build_intent_summary();
                    let surprise_score = self.surprise_score_from_ctx(&ctx);
                    let drift_score = drift_score_from_snapshot(&workspace_snapshot);
                    let causal_attention_risk = intent.risk;
                    let attention_weights = self
                        .last_attention
                        .lock()
                        .map(|attn| attn.clone())
                        .unwrap_or_else(|_| idle_attention());
                    ctx.attention_weights = Some(attention_weights.clone());
                    let percept_commit = cf.commitment().digest;
                    let percept_energy =
                        canonical_control_frame_len(cf.as_ref()).min(10_000) as u16;
                    ctx.percept_commit = Some(percept_commit);
                    ctx.percept_energy = Some(percept_energy);
                    let lens_selection = ctx.inference.as_ref().and_then(|inference| {
                        self.translate_features(
                            inference.activation_view.as_ref(),
                            &attention_weights,
                            cycle_id,
                            pulse.slot,
                        )
                    });
                    ctx.lens_selection = lens_selection.clone();
                    ctx.surprise_score = Some(surprise_score);
                    let lock_window_buckets = self.lagged_lock_window_buckets();
                    let onn_outputs = self.tick_onn_phase(
                        cycle_id,
                        causal_attention_risk,
                        drift_score,
                        surprise_score,
                        lock_window_buckets,
                        pulse.slot,
                    );
                    let phase_bus = onn_outputs.phase_bus;
                    let phase_lock = onn_outputs.lock;
                    ctx.phase_commit = Some(phase_bus.commit);
                    ctx.phase_bus = Some(phase_bus);
                    ctx.phase_lock = Some(phase_lock);
                    ctx.coherence_plv = Some(onn_outputs.phase_bus.global_plv);
                    ctx.onn_outputs = Some(onn_outputs);
                    let snn_knobs = self.current_snn_knobs();
                    let spike_params = spike_params_from_knobs(&snn_knobs);
                    let mut candidates: Vec<Spike> = Vec::new();
                    if let Ok(mut pending) = self.pending_ai_spikes.lock() {
                        candidates.extend(pending.drain(..));
                    }
                    #[cfg(feature = "mock-spike-producers")]
                    {
                        let sae = MockSaeProducer::new(Digest32::new([1u8; 32]));
                        let lens = MockLensProducer::new(Digest32::new([2u8; 32]));
                        candidates.extend(sae.produce(cycle_id, phase_bus.gamma_bucket));
                        candidates.extend(lens.produce(cycle_id, phase_bus.gamma_bucket));
                    }
                    if let Some(sle_outputs) = ctx.sle_outputs.as_ref() {
                        if sle_outputs.thought_only_root != Digest32::new([0u8; 32]) {
                            candidates.push(Spike::new(
                                cycle_id,
                                SpikeKind::ThoughtOnly,
                                sle_outputs.reflection.intensity,
                                phase_bus.gamma_bucket,
                                ModuleId::Sle,
                                sle_outputs.thought_only_root,
                            ));
                        }
                    }
                    if let Some(iit_output) = ctx.iit_output.as_ref() {
                        if iit_output.request_replay {
                            candidates.push(Spike::new(
                                cycle_id,
                                SpikeKind::ReplayHint,
                                iit_output.phi_proxy,
                                phase_bus.gamma_bucket,
                                ModuleId::Iit,
                                iit_output.hints_commit,
                            ));
                        }
                    }
                    if let Some(decision) = ctx.decision.as_ref() {
                        let intensity = decision.confidence_bp.min(u32::from(u16::MAX)) as u16;
                        candidates.push(Spike::new(
                            cycle_id,
                            SpikeKind::PolicySignal,
                            intensity,
                            phase_bus.gamma_bucket,
                            ModuleId::Policy,
                            digest_policy_commit(decision),
                        ));
                    }
                    let spike_inputs = SpikeInputs::new(cycle_id, phase_lock, candidates);
                    let spike_outputs = {
                        let mut modules =
                            self.runtime_modules.lock().expect("runtime modules lock");
                        modules.spikes.set_params(spike_params);
                        modules.spikes.tick(&spike_inputs)
                    };
                    if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.set_spike_outputs(spike_outputs.clone());
                    }
                    ctx.spike_outputs = Some(spike_outputs.clone());
                    self.append_spike_bus_record(cycle_id, &spike_outputs);
                    let nsr_full = !spike_outputs.accepted.is_empty();
                    let cde_full = nsr_full;
                    let causal_report = {
                        let last_cde = self
                            .last_cde_output
                            .lock()
                            .ok()
                            .and_then(|guard| guard.clone());
                        if let Some(output) = last_cde.as_ref() {
                            if cde_full {
                                cde_report_from_outputs(output)
                            } else {
                                CausalReport::new(
                                    output.graph_commit,
                                    Vec::new(),
                                    CAUSAL_REPORT_FLAG_LIGHT,
                                )
                            }
                        } else {
                            CausalReport::new(
                                Digest32::new([0u8; 32]),
                                Vec::new(),
                                CAUSAL_REPORT_FLAG_LIGHT,
                            )
                        }
                    };
                    let summary = format!(
                        "CDE ok cf={} dag={}",
                        causal_report.counterfactuals.len(),
                        short_digest(causal_report.dag_commit)
                    );
                    self.publish_workspace_signal(WorkspaceSignal {
                        kind: SignalKind::Integration,
                        priority: 3100,
                        digest: causal_report.commit,
                        summary,
                        slot: pulse.slot,
                    });
                    self.append_causal_report_record(cycle_id, &causal_report);
                    ctx.causal_report = Some(causal_report.clone());
                    let nsr_input = self.build_nsr_input(
                        cycle_id,
                        decision.kind as u16,
                        &inference.outputs,
                        &workspace_snapshot,
                        intent,
                        Some(&causal_report),
                    );
                    let nsr_report = if nsr_full {
                        self.nsr_port.evaluate(&nsr_input)
                    } else {
                        light_report(&nsr_input)
                    };
                    ctx.nsr_report = Some(nsr_report.clone());
                    if let Ok(mut guard) = self.last_nsr_report.lock() {
                        *guard = Some(nsr_report.clone());
                    }
                    let nsr_summary = format!(
                        "NSR={} v={} causal={}",
                        nsr_verdict_token(nsr_report.verdict),
                        nsr_report.violations.len(),
                        nsr_verdict_token_lower(nsr_report.causal_verdict())
                    );
                    self.publish_workspace_signal(WorkspaceSignal {
                        kind: SignalKind::Risk,
                        priority: nsr_signal_priority(nsr_report.verdict),
                        digest: nsr_report.commit,
                        summary: nsr_summary,
                        slot: pulse.slot,
                    });
                    self.append_nsr_report_record(cycle_id, &nsr_report);
                    if nsr_report.verdict == NsrVerdict::Deny {
                        self.append_nsr_audit_notice(cycle_id, &nsr_report);
                    }
                    if nsr_report.causal_verdict() == NsrVerdict::Deny {
                        self.append_causal_audit_notice(cycle_id, &nsr_report);
                    }
                    if ctx.phase_bus.is_none() {
                        let phase_bus = self.latest_phase_bus(cycle_id);
                        ctx.phase_commit = Some(phase_bus.commit);
                        ctx.phase_bus = Some(phase_bus);
                        ctx.coherence_plv = Some(phase_bus.global_plv);
                    }
                    let mut attention_risk = 0u16;
                    let mut outputs = inference.outputs.clone();
                    ctx.output_intent = outputs
                        .iter()
                        .any(|output| matches!(output.channel, OutputChannel::Speech));
                    let spike_thought_only = ctx
                        .spike_outputs
                        .as_ref()
                        .and_then(|outputs| {
                            outputs
                                .counts
                                .iter()
                                .find(|(kind, _)| *kind == SpikeKind::ThoughtOnly)
                                .map(|(_, count)| *count > 0)
                        })
                        .unwrap_or(false);
                    let thought_only = spike_thought_only
                        || (!outputs.is_empty()
                            && outputs
                                .iter()
                                .all(|output| matches!(output.channel, OutputChannel::Thought)));
                    let tool_req = false;
                    let mut risk_results = Vec::with_capacity(outputs.len());
                    let mut speech_gate_results = Vec::with_capacity(outputs.len());
                    let tom_report = ctx.tom_report.as_ref().expect("tom report available");
                    for output in outputs.iter() {
                        let gate_result = self.risk_gate.evaluate(
                            ctx.nsr_report.as_ref(),
                            inference.scm_dag.as_ref(),
                            output,
                            &cf,
                            Some(tom_report),
                            inference.cde_confidence,
                        );
                        attention_risk = attention_risk.max(gate_result.risk);
                        risk_results.push(gate_result);
                        speech_gate_results.push(self.speech_gate.allow_speech(&cf, output));
                    }
                    ctx.attention_risk = attention_risk;
                    self.publish_workspace_signals(risk_results.iter().map(|result| {
                        WorkspaceSignal::from_risk_result(result, None, Some(pulse.slot))
                    }));

                    let spike_outputs = ctx
                        .spike_outputs
                        .clone()
                        .unwrap_or_else(|| empty_spike_outputs(cycle_id));
                    let spike_counts = spike_outputs.counts.clone();
                    ctx.replay_pressure = Some(replay_pressure_from_spikes(&spike_counts));
                    let sle_request_replay = self
                        .last_workspace_snapshot
                        .lock()
                        .ok()
                        .and_then(|snapshot| {
                            snapshot
                                .as_ref()
                                .map(|snapshot| snapshot.sle_request_replay)
                        })
                        .unwrap_or(false);
                    if sle_request_replay {
                        let current = ctx.replay_pressure.unwrap_or(0);
                        ctx.replay_pressure = Some(current.max(5_000));
                    }
                    self.tick_coupling(&mut ctx, cycle_id);
                    let phase_bus = ctx
                        .phase_bus
                        .unwrap_or_else(|| self.latest_phase_bus(cycle_id));
                    self.tick_jepa(&mut ctx, cycle_id, &phase_bus);
                    let surprise_score = self.surprise_score_from_ctx(&ctx);
                    let iit_output = {
                        let ncde_output = ctx
                            .ncde_output
                            .or_else(|| self.last_ncde_output.lock().ok().and_then(|g| *g));
                        let ssm_output = ctx
                            .ssm_output
                            .clone()
                            .or_else(|| self.last_ssm_output.lock().ok().and_then(|g| g.clone()));
                        let coupling_outputs = ctx.coupling_outputs.clone().or_else(|| {
                            self.last_coupling_outputs
                                .lock()
                                .ok()
                                .and_then(|g| g.clone())
                        });
                        let cde_commit = ctx
                            .cde_output
                            .as_ref()
                            .map(|output| output.commit)
                            .or_else(|| {
                                self.last_cde_output
                                    .lock()
                                    .ok()
                                    .and_then(|guard| guard.as_ref().map(|out| out.commit))
                            })
                            .unwrap_or_else(|| Digest32::new([0u8; 32]));
                        let phase_bus = ctx
                            .phase_bus
                            .unwrap_or_else(|| self.latest_phase_bus(cycle_id));
                        let inputs = IitInputs::new(
                            cycle_id,
                            phase_bus.commit,
                            phase_bus.gamma_bucket,
                            phase_bus.global_plv,
                            ssm_output
                                .as_ref()
                                .map(|output| output.ssm_state_commit)
                                .unwrap_or_else(|| Digest32::new([0u8; 32])),
                            ncde_output
                                .as_ref()
                                .map(|output| output.ncde_state_digest)
                                .unwrap_or_else(|| Digest32::new([0u8; 32])),
                            cde_commit,
                            workspace_snapshot
                                .nsr_trace_root
                                .unwrap_or_else(|| Digest32::new([0u8; 32])),
                            coupling_outputs
                                .as_ref()
                                .map(|output| output.influences_root)
                                .unwrap_or_else(|| Digest32::new([0u8; 32])),
                            attention_risk,
                            drift_score,
                            surprise_score,
                        );
                        let mut modules =
                            self.runtime_modules.lock().expect("runtime modules lock");
                        modules.iit.tick(&inputs)
                    };
                    let iit_report = report_for_phi(iit_output.phi_proxy);
                    let iit_actions = actions_for_phi(iit_output.phi_proxy, attention_risk);
                    ctx.iit_output = Some(iit_output);
                    if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.set_iit_output(iit_output);
                    }
                    if let Ok(mut guard) = self.last_iit_hints.lock() {
                        *guard = IitHintState {
                            tighten_sync: iit_output.tighten_sync,
                            damp_output: iit_output.damp_output,
                            damp_learning: iit_output.damp_learning,
                            request_replay: iit_output.request_replay,
                            hints_commit: iit_output.hints_commit,
                        };
                    }
                    if iit_output.request_replay {
                        let current = ctx.replay_pressure.unwrap_or(0);
                        ctx.replay_pressure = Some(current.saturating_add(800).min(10_000));
                    }
                    self.append_iit_output_record(cycle_id, &iit_output);
                    let mode_lag_snapshot = self.coherence_lag_snapshot();
                    let nsr_verdict = ctx
                        .nsr_report
                        .as_ref()
                        .map(|report| report.verdict)
                        .unwrap_or(NsrVerdict::Allow);
                    let update_mode = self.compute_update_mode(
                        &mode_lag_snapshot,
                        nsr_verdict,
                        attention_risk,
                        surprise_score,
                        drift_score,
                        iit_output.phi_proxy,
                    );
                    if update_mode == UpdateMode::Stabilize {
                        self.force_stabilize_cycles.fetch_add(1, Ordering::Relaxed);
                    }
                    ctx.update_mode = Some(update_mode);
                    ctx.coherence_request_replay = self.coherence_request_replay(
                        update_mode,
                        surprise_score,
                        iit_output.phi_proxy,
                    );
                    if ctx.coherence_request_replay {
                        let current = ctx.replay_pressure.unwrap_or(0);
                        ctx.replay_pressure = Some(current.max(5_000));
                    }
                    if budget_cycle.request_replay {
                        ctx.coherence_request_replay = true;
                        let current = ctx.replay_pressure.unwrap_or(0);
                        ctx.replay_pressure = Some(current.max(5_000));
                    }
                    if let Ok(mut guard) = self.last_update_mode.lock() {
                        *guard = update_mode;
                    }
                    let attention_weights =
                        ctx.attention_weights.clone().unwrap_or_else(idle_attention);
                    // Bluebrain stimulation occurs during the Verify pulse using the latest
                    // workspace snapshot, attention, and surprise context.
                    let surprise_signal = ctx.predictive_result.as_ref().map(|(_, signal)| signal);
                    let lens_selection = ctx.lens_selection.clone();
                    self.stimulate_bluebrain_port(
                        &cf,
                        &workspace_snapshot,
                        &attention_weights,
                        surprise_signal,
                        lens_selection.as_ref(),
                        pulse.slot,
                    );
                    let recursion_inputs = RecursionInputs {
                        phi: iit_output.phi_proxy,
                        drift_score,
                        surprise: surprise_score,
                        risk: attention_risk,
                        attn_gain: attention_weights.gain,
                        focus: focus_channel_score(attention_weights.channel),
                    };
                    let recursion_budget = self.recursion_controller.compute(&recursion_inputs);
                    self.sle_engine.set_max_level(recursion_budget.max_depth);
                    self.publish_workspace_signal(WorkspaceSignal::from_recursion_budget(
                        recursion_budget.max_depth,
                        recursion_budget.per_cycle_steps,
                        recursion_budget.commit,
                        None,
                        Some(pulse.slot),
                    ));
                    self.archive
                        .append(self.build_recursion_budget_record(cycle_id, &recursion_budget));
                    ctx.recursion_budget = Some(recursion_budget);
                    if let Ok(mut guard) = self.last_recursion_budget.lock() {
                        *guard = Some(recursion_budget);
                    }
                    let risk_commit = digest_risk_results(&risk_results);
                    let ssm_commit = inference
                        .ssm_state
                        .as_ref()
                        .map(|state| state.commit)
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let attn_commit = self
                        .last_attention
                        .lock()
                        .map(|attn| attn.commit)
                        .unwrap_or_else(|_| Digest32::new([0u8; 32]));
                    let ncde_commit = ctx
                        .ncde_output
                        .as_ref()
                        .map(|output| output.commit)
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let consistency = consistency_score_from_nsr(ctx.nsr_report.as_ref());
                    let self_state = SelfStateBuilder::new(cycle_id)
                        .ssm_commit(ssm_commit)
                        .workspace_commit(workspace_snapshot.commit)
                        .risk_commit(risk_commit)
                        .attn_commit(attn_commit)
                        .ncde_commit(ncde_commit)
                        .consistency(consistency)
                        .build();
                    let previous_state = self
                        .last_self_state
                        .lock()
                        .ok()
                        .and_then(|state| *state)
                        .unwrap_or(self_state);
                    let sle_reflex = self
                        .sle_engine
                        .evaluate(&workspace_snapshot, &previous_state);
                    self.publish_workspace_signal(WorkspaceSignal::from_sle_reflex(
                        sle_reflex.loop_level,
                        sle_reflex.delta,
                        sle_reflex.commit,
                        None,
                        Some(pulse.slot),
                    ));
                    self.append_self_state_record(&self_state);
                    self.archive
                        .append(self.build_sle_reflex_record(cycle_id, &sle_reflex));
                    if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.record_recursion_used(u16::from(sle_reflex.loop_level));
                    }
                    if let Ok(mut guard) = self.last_self_state.lock() {
                        *guard = Some(self_state);
                    }
                    ctx.self_state = Some(self_state);
                    ctx.sle_reflex = Some(sle_reflex);
                    let influence_outputs = ctx.influence_outputs.clone().or_else(|| {
                        self.last_influence_outputs
                            .lock()
                            .ok()
                            .and_then(|guard| guard.clone())
                    });
                    let spike_outputs = ctx
                        .spike_outputs
                        .clone()
                        .unwrap_or_else(|| empty_spike_outputs(cycle_id));
                    let spike_counts = spike_outputs.counts.clone();
                    let spike_root_commit = spike_outputs.accepted_root;
                    let ncde_snapshot = ctx
                        .ncde_output
                        .or_else(|| self.last_ncde_output.lock().ok().and_then(|g| *g));
                    let cde_output = self.tick_cde(
                        cycle_id,
                        &phase_bus,
                        spike_root_commit,
                        spike_counts.clone(),
                        self.world_state_commit_from_ctx(&ctx),
                        influence_outputs.as_ref(),
                        &iit_output,
                        ctx.ssm_output.as_ref(),
                        ncde_snapshot.as_ref(),
                        drift_score,
                        surprise_score,
                        attention_risk,
                    );
                    if let Some(output) = cde_output.clone() {
                        ctx.cde_output = Some(output.clone());
                        if let Ok(mut guard) = self.last_cde_output.lock() {
                            *guard = Some(output.clone());
                        }
                    }
                    let replay_pressure = ctx.replay_pressure.unwrap_or(0);
                    let base_learning_cap = self
                        .tcf_plan_for(Some(&ctx))
                        .map(|plan| plan.learning_gain_cap)
                        .unwrap_or(10_000);
                    let update_mode = ctx.update_mode.unwrap_or(UpdateMode::Normal);
                    let nsr_verdict = ctx
                        .nsr_report
                        .as_ref()
                        .map(|report| report.verdict)
                        .unwrap_or(NsrVerdict::Allow);
                    let phi_high = self.iit_params().phi_high;
                    let learning_gain_cap = self.adjust_learning_cap(
                        update_mode,
                        base_learning_cap,
                        nsr_verdict,
                        iit_output.phi_proxy,
                        phi_high,
                    );
                    let jepa_surprise = ctx
                        .jepa_outputs
                        .as_ref()
                        .map(|output| output.surprise)
                        .unwrap_or(surprise_score);
                    let ssm_state_digest =
                        ctx.ssm_output
                            .as_ref()
                            .map(|output| output.ssm_state_digest)
                            .or_else(|| {
                                self.last_ssm_output.lock().ok().and_then(|guard| {
                                    guard.as_ref().map(|out| out.ssm_state_digest)
                                })
                            })
                            .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let world_state_commit = self.world_state_commit_from_ctx(&ctx);
                    let observation_commit = derive_observation_commit(
                        world_state_commit,
                        ssm_state_digest,
                        spike_root_commit,
                        phase_bus.commit,
                    );
                    let nsr_trace_root = ctx
                        .nsr_output
                        .as_ref()
                        .map(|output| output.trace_root)
                        .or_else(|| {
                            self.last_workspace_snapshot.lock().ok().and_then(|guard| {
                                guard.as_ref().and_then(|snap| snap.nsr_trace_root)
                            })
                        })
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let observation_key = ObservationKey::new(
                        cycle_id,
                        observation_commit,
                        world_state_commit,
                        ssm_state_digest,
                        spike_root_commit,
                        phase_bus.commit,
                        nsr_trace_root,
                        jepa_surprise,
                    );
                    let policy_restrict = matches!(
                        ctx.decision_kind,
                        DecisionKind::DecisionKindDeny
                            | DecisionKind::DecisionKindEscalate
                            | DecisionKind::DecisionKindObserve
                    );
                    let cde_v1_output = self.tick_cde_v1(
                        cycle_id,
                        &phase_bus,
                        spike_root_commit,
                        observation_commit,
                        observation_key,
                        ctx.ssm_output.as_ref(),
                        ncde_snapshot.as_ref(),
                        &iit_output,
                        influence_outputs.as_ref(),
                        replay_pressure,
                        drift_score,
                        surprise_score,
                        attention_risk,
                        self.tcf_plan_for(Some(&ctx)),
                        learning_gain_cap,
                        &spike_counts,
                        policy_restrict,
                    );
                    if let Some((output, graph_commit)) = cde_v1_output.clone() {
                        ctx.cde_v1_output = Some(output.clone());
                        if let Ok(mut guard) = self.last_cde_v1_output.lock() {
                            *guard = Some(output.clone());
                        }
                        if let Ok(mut workspace) = self.workspace.lock() {
                            workspace.set_cde_output(
                                output.summary_commit,
                                graph_commit,
                                compress_cde_v1_edges(&output.top_edges),
                                collect_cde_v1_edge_commits(&output.top_edges),
                                output.intervention.as_ref().map(|item| item.commit),
                                observation_key.commit,
                                None,
                            );
                        }
                        self.append_cde_output_record(
                            cycle_id,
                            &output,
                            graph_commit,
                            observation_key.commit,
                            None,
                        );
                    }
                    let influence_for_nsr = ctx.influence_outputs.clone().or_else(|| {
                        self.last_influence_outputs
                            .lock()
                            .ok()
                            .and_then(|guard| guard.clone())
                    });
                    let policy_ok = decision.kind == DecisionKind::DecisionKindAllow as i32;
                    let policy_commit = digest_policy_commit(&decision);
                    let phase_bus = ctx
                        .phase_bus
                        .unwrap_or_else(|| self.latest_phase_bus(cycle_id));
                    let nsr_inputs = self.build_nsr_inputs(
                        cycle_id,
                        &phase_bus,
                        influence_for_nsr.as_ref(),
                        ctx.cde_v1_output.as_ref(),
                        ctx.cde_output.as_ref(),
                        ctx.ssm_output.as_ref(),
                        ctx.ncde_output.as_ref(),
                        &iit_output,
                        ctx.self_state.as_ref(),
                        spike_counts.clone(),
                        spike_outputs.max_intensity,
                        ctx.replay_pressure.unwrap_or(0),
                        drift_score,
                        surprise_score,
                        jepa_surprise,
                        attention_risk,
                        ctx.output_intent,
                        thought_only,
                        tool_req,
                        policy_commit,
                        policy_ok,
                        matches!(pulse.kind, PulseKind::Sleep),
                        self.tcf_plan_for(Some(&ctx)),
                    );
                    let (nsr_output, nsr_trace) = self
                        .runtime_modules
                        .lock()
                        .expect("runtime modules lock")
                        .nsr
                        .tick_with_trace(&nsr_inputs);
                    let (nsr_hit_counts, nsr_hit_summaries, nsr_warned) =
                        summarize_nsr_hits(&nsr_trace.hits);
                    ctx.nsr_output = Some(nsr_output.clone());
                    if let Ok(mut workspace) = self.workspace.lock() {
                        let nsr_fact_flags =
                            nsr_fact_flags(phase_bus.global_plv >= 7_000, surprise_score >= 7_000);
                        workspace.set_nsr_trace(NsrTraceSummary {
                            trace_root: nsr_output.trace_root,
                            prev_commit: None,
                            verdict: nsr_output.verdict.as_u8(),
                            derived_facts_root: None,
                            triggered_rules_root: None,
                            fact_flags: nsr_fact_flags,
                            hit_counts: nsr_hit_counts,
                            hit_summaries: nsr_hit_summaries,
                        });
                    }
                    self.append_nsr_output_record(cycle_id, &nsr_output);
                    let nsr_warn_streak = self.update_nsr_warn_streak(nsr_output.verdict);
                    ctx.nsr_warn_streak = Some(nsr_warn_streak);
                    if nsr_output.verdict != NsrVerdict::Allow || nsr_warned {
                        ctx.update_mode = Some(UpdateMode::Stabilize);
                    }
                    if let Some(mode) = ctx.update_mode {
                        ctx.coherence_request_replay = self.coherence_request_replay(
                            mode,
                            surprise_score,
                            iit_output.phi_proxy,
                        );
                        if ctx.coherence_request_replay {
                            let current = ctx.replay_pressure.unwrap_or(0);
                            ctx.replay_pressure = Some(current.max(5_000));
                        }
                        if let Ok(mut guard) = self.last_update_mode.lock() {
                            *guard = mode;
                        }
                    }

                    let ai_host_outputs = {
                        let policy_snapshot_commit = ctx
                            .decision
                            .as_ref()
                            .map(digest_policy_commit)
                            .unwrap_or_else(|| Digest32::new([0u8; 32]));
                        let nsr_verdict = ctx
                            .nsr_report
                            .as_ref()
                            .map(|report| report.verdict)
                            .unwrap_or(nsr_output.verdict);
                        let external_commit = digest_ai_external_commit(
                            ctx.percept_commit
                                .unwrap_or_else(|| Digest32::new([0u8; 32])),
                            nsr_verdict,
                        );
                        let (attention_cap, learning_cap) = self
                            .tcf_plan_for(Some(&ctx))
                            .map(|plan| (plan.attention_gain_cap, plan.learning_gain_cap))
                            .unwrap_or((10_000, 10_000));
                        let ai_input = AiInputFrame::new(
                            cycle_id,
                            external_commit,
                            phase_bus.commit,
                            phase_bus.gamma_bucket,
                            phase_bus.global_plv,
                            ssm_state_digest,
                            world_state_commit,
                            policy_snapshot_commit,
                            attention_cap,
                            learning_cap,
                        );
                        let mut modules =
                            self.runtime_modules.lock().expect("runtime modules lock");
                        modules.ai_host.tick(&ai_input)
                    };

                    let mut ai_spike_candidates = Vec::new();
                    for event in &ai_host_outputs.features {
                        ai_spike_candidates.push(Spike::new(
                            cycle_id,
                            event.kind,
                            event.intensity,
                            event.bucket,
                            ModuleId::Other(AI_HOST_MODULE_ID),
                            event.commit,
                        ));
                    }
                    for thought in &ai_host_outputs.internal_thoughts {
                        ai_spike_candidates.push(Spike::new(
                            cycle_id,
                            SpikeKind::ThoughtOnly,
                            thought.salience,
                            phase_bus.gamma_bucket,
                            ModuleId::Other(AI_HOST_MODULE_ID),
                            thought.thought_root,
                        ));
                    }
                    if !ai_spike_candidates.is_empty() {
                        if let Ok(mut pending) = self.pending_ai_spikes.lock() {
                            pending.extend(ai_spike_candidates);
                        }
                    }

                    let base_output_len = outputs.len();
                    let nsr_allows_output = ctx
                        .nsr_report
                        .as_ref()
                        .map(|report| report.verdict == NsrVerdict::Allow)
                        .unwrap_or(nsr_output.verdict == NsrVerdict::Allow);
                    if nsr_allows_output {
                        for candidate in &ai_host_outputs.outputs {
                            if candidate.label == ChannelLabel::External {
                                outputs.push(ai_output_from_candidate(candidate));
                            }
                        }
                    }
                    if outputs.len() > base_output_len {
                        let tom_report = ctx.tom_report.as_ref().expect("tom report available");
                        let base_risk_len = risk_results.len();
                        for output in outputs.iter().skip(base_output_len) {
                            let gate_result = self.risk_gate.evaluate(
                                ctx.nsr_report.as_ref(),
                                inference.scm_dag.as_ref(),
                                output,
                                &cf,
                                Some(tom_report),
                                inference.cde_confidence,
                            );
                            attention_risk = attention_risk.max(gate_result.risk);
                            risk_results.push(gate_result);
                            speech_gate_results.push(self.speech_gate.allow_speech(&cf, output));
                        }
                        self.publish_workspace_signals(
                            risk_results.iter().skip(base_risk_len).map(|result| {
                                WorkspaceSignal::from_risk_result(result, None, Some(pulse.slot))
                            }),
                        );
                    }

                    let influence_for_output = self
                        .last_influence_outputs
                        .lock()
                        .ok()
                        .and_then(|guard| guard.clone());
                    apply_influence_output_suppression(
                        &mut speech_gate_results,
                        &outputs,
                        influence_for_output.as_ref(),
                    );

                    let effects = iit_action_effects(
                        self.workspace_base,
                        &self.output_router_base,
                        &iit_actions,
                    );
                    ctx.integration_score = Some(iit_output.phi_proxy);
                    ctx.integration_bias = effects.integration_bias;
                    ctx.iit_actions = iit_actions.clone();
                    self.apply_iit_effects(effects);
                    self.publish_workspace_signal(WorkspaceSignal::from_integration_score(
                        iit_output.phi_proxy,
                        None,
                        Some(pulse.slot),
                    ));
                    self.append_iit_report_record(cycle_id, &iit_report);
                    for action in &iit_actions {
                        self.archive
                            .append(self.build_iit_action_record(cycle_id, action));
                    }
                    ctx.iit_report = Some(iit_report);

                    let surprise_band = self
                        .last_surprise
                        .lock()
                        .ok()
                        .and_then(|guard| guard.as_ref().map(|signal| band_for_score(signal.score)))
                        .unwrap_or(SurpriseBand::Low);
                    let suppression_count = workspace_suppression_count(&workspace_snapshot);
                    let (ism_root, anchors) = self
                        .ism_store
                        .lock()
                        .map(|store| {
                            let anchors = store
                                .anchors()
                                .iter()
                                .rev()
                                .take(ISM_ANCHOR_TOP_K)
                                .copied()
                                .collect::<Vec<_>>();
                            (store.root_commit(), anchors)
                        })
                        .unwrap_or_else(|_| (Digest32::new([0u8; 32]), Vec::new()));
                    let sle_reflex = ctx.sle_reflex.clone().expect("sle reflex available");
                    let self_state = ctx.self_state.expect("self state available");
                    let consistency_inputs = ConsistencyInputs {
                        self_state: &self_state,
                        self_symbol: sle_reflex.self_symbol,
                        ism_root,
                        anchors: &anchors,
                        suppression_count,
                        policy_class: decision.kind as u16,
                        policy_stable: decision.kind == DecisionKind::DecisionKindAllow as i32,
                        risk_score: attention_risk,
                        surprise_band,
                        phi: iit_report.phi,
                    };
                    let (consistency_report, consistency_actions) =
                        self.consistency_engine.evaluate(&consistency_inputs);
                    self.publish_workspace_signal(WorkspaceSignal::from_consistency_drift(
                        &consistency_report,
                        None,
                        Some(pulse.slot),
                    ));
                    self.append_consistency_report_record(cycle_id, &consistency_report);
                    for action in &consistency_actions {
                        self.archive
                            .append(self.build_consistency_action_record(cycle_id, action));
                    }
                    let consistency_effects = self
                        .output_router
                        .lock()
                        .map(|mut output_router| {
                            let effects = consistency_action_effects(
                                output_router.max_thought_frames_per_cycle(),
                                &consistency_actions,
                            );
                            output_router.set_max_thought_frames_per_cycle(
                                effects.max_thought_frames_per_cycle,
                            );
                            effects
                        })
                        .ok();
                    ctx.consistency_report = Some(consistency_report);
                    ctx.consistency_actions = consistency_actions;
                    ctx.consistency_effects = consistency_effects;

                    if let Ok(workspace) = self.workspace.lock() {
                        ctx.spike_outputs = Some(workspace.spike_summary());
                    }

                    let policy_ok = decision.kind == DecisionKind::DecisionKindAllow as i32;
                    let nsr_verdict = ctx
                        .nsr_output
                        .as_ref()
                        .map(|output| output.verdict)
                        .unwrap_or(NsrVerdict::Allow);
                    let consistency_ok = !matches!(
                        ctx.consistency_report,
                        Some(ref report) if report.band == DriftBand::Critical
                    );
                    let coherence_plv = ctx.coherence_plv.unwrap_or(0);
                    let structural_stats = StructuralCycleStats::new(
                        coherence_plv,
                        iit_output.phi_proxy,
                        drift_score,
                        surprise_score,
                        nsr_verdict.as_u8(),
                        policy_ok,
                        consistency_ok,
                    );
                    ctx.drift_score = Some(drift_score);
                    ctx.surprise_score = Some(surprise_score);
                    ctx.structural_stats = Some(structural_stats);

                    let nsr_summary = NsrSummary {
                        verdict: nsr_verdict,
                        violations_digest: ctx
                            .nsr_output
                            .as_ref()
                            .map(|output| output.trace_root)
                            .unwrap_or_else(|| digest_reasons(&[])),
                    };
                    let coherence_threshold = if nsr_verdict == NsrVerdict::Restrict {
                        ONN_COHERENCE_THROTTLE_RESTRICT
                    } else {
                        ONN_COHERENCE_THROTTLE
                    };
                    let output_lock = coherence_plv;
                    let output_gain_cap = self
                        .tcf_plan_for(Some(&ctx))
                        .map(|plan| plan.output_gain_cap)
                        .unwrap_or(10_000);
                    let gates = GateBundle {
                        policy_decision: decision.clone(),
                        sandbox: ctx.sandbox_verdict.clone().unwrap_or(SandboxVerdict::Allow),
                        risk_results,
                        nsr_summary,
                        speech_gate: speech_gate_results,
                        coherence_plv,
                        coherence_threshold,
                        phi_proxy: iit_output.phi_proxy,
                        phi_threshold: PHI_OUTPUT_THRESHOLD,
                        speak_lock: output_lock,
                        speak_lock_min: LOCK_MIN_SPEAK,
                        damp_output: iit_output.damp_output,
                        output_gain_cap,
                    };
                    let mut output_router = self.output_router.lock().expect("output router lock");
                    if let Some(budget) = ctx.recursion_budget.as_ref() {
                        output_router.apply_recursion_budget(budget);
                    }
                    output_router.apply_coherence(coherence_plv, coherence_threshold);
                    let decisions = output_router.route(&cf, outputs.clone(), &gates);
                    let events = output_router.drain_events();
                    self.publish_workspace_signals(
                        decisions
                            .iter()
                            .map(|decision| {
                                WorkspaceSignal::from_output_decision(
                                    decision,
                                    None,
                                    Some(pulse.slot),
                                )
                            })
                            .chain(events.iter().map(|event| {
                                WorkspaceSignal::from_output_event(event, None, Some(pulse.slot))
                            })),
                    );
                    for event in &events {
                        self.append_output_event_record(cycle_id, event);
                    }

                    for (idx, output) in outputs.iter().enumerate() {
                        match output.channel {
                            OutputChannel::Thought => ctx.thought_outputs.push(output.clone()),
                            OutputChannel::Speech => {
                                if decisions
                                    .get(idx)
                                    .map(|decision| decision.permitted)
                                    .unwrap_or(false)
                                {
                                    ctx.speech_outputs.push(output.clone());
                                } else if let Some(result) = gates.risk_results.get(idx) {
                                    let reason = decisions
                                        .get(idx)
                                        .map(|decision| decision.reason_code.clone())
                                        .unwrap_or_else(|| "risk_denied".to_string());
                                    let reason_digest = digest_reasons(&[reason]);
                                    ctx.suppressions.push(OutputSuppressionInfo {
                                        channel: OutputChannel::Speech,
                                        reason_digest,
                                        risk: result.risk,
                                    });
                                    if let Some(sink) = &self.output_suppression_sink {
                                        sink.publish(OutputSuppressed {
                                            channel: OutputChannel::Speech,
                                            reason_digest,
                                            risk: result.risk,
                                        });
                                    }
                                }
                            }
                        }
                    }

                    if ctx.integration_score.is_none() {
                        ctx.integration_score = ctx
                            .thought_outputs
                            .iter()
                            .find_map(|output| output.integration_score);
                    }
                    ctx.attention_risk = attention_risk;

                    if let Some(state) = inference.ssm_state.as_ref() {
                        self.publish_workspace_signal(WorkspaceSignal::from_world_state(
                            state.commit,
                            Some(pulse.slot),
                        ));
                    }
                    let observation = inference
                        .ssm_state
                        .as_ref()
                        .map(observation_from_ssm_state)
                        .unwrap_or_else(|| observation_from_frame(&cf));
                    let predictive_result = self.update_predictive_coding(&observation);
                    if let Some((_, surprise_signal)) = predictive_result.as_ref() {
                        let update = SurpriseUpdated::from(surprise_signal);
                        self.publish_workspace_signal(WorkspaceSignal::from_surprise_update(
                            &update,
                            Some(pulse.slot),
                        ));
                    }
                    ctx.predictive_result = predictive_result;
                }
                PulseKind::Consolidate => {
                    if let (Some(score), None) = (ctx.integration_score, ctx.iit_report.as_ref()) {
                        self.publish_workspace_signal(WorkspaceSignal::from_integration_score(
                            score,
                            None,
                            Some(pulse.slot),
                        ));
                    }
                    let Some(decision) = ctx.decision.as_ref() else {
                        continue;
                    };
                    let Some(tom_report) = ctx.tom_report.as_ref() else {
                        continue;
                    };
                    let surprise_score = self.surprise_score_from_ctx(&ctx);
                    let drift_score = ctx
                        .consistency_report
                        .as_ref()
                        .map(|report| report.drift_score)
                        .unwrap_or(0);
                    let phase_commit = ctx.phase_commit.unwrap_or(Digest32::new([0u8; 32]));
                    let attn_gain = self
                        .last_attention
                        .lock()
                        .map(|attn| attn.gain)
                        .unwrap_or(0);
                    let ssm_salience = ctx
                        .ssm_output
                        .as_ref()
                        .map(|output| output.ssm_salience)
                        .unwrap_or(0);
                    let ssm_novelty = ctx
                        .ssm_output
                        .as_ref()
                        .map(|output| output.ssm_novelty)
                        .unwrap_or(0);
                    let ncde_output = ctx
                        .ncde_output
                        .or_else(|| self.last_ncde_output.lock().ok().and_then(|g| *g));
                    let ncde_energy = ncde_output.map(|output| output.ncde_energy).unwrap_or(0);
                    let ncde_commit = ncde_output
                        .map(|output| output.commit)
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let ncde_state_digest = ncde_output
                        .map(|output| output.ncde_state_digest)
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let nsr_verdict = ctx
                        .nsr_output
                        .as_ref()
                        .map(|output| output.verdict.as_u8())
                        .unwrap_or(0);
                    let coherence_plv = ctx.coherence_plv.unwrap_or(0);
                    let phi_proxy = ctx.integration_score.unwrap_or(0);
                    let cde_commit = ctx
                        .cde_v1_output
                        .as_ref()
                        .map(|output| output.summary_commit);
                    let (workspace_sle_reflection, rsa_applied) = self
                        .workspace
                        .lock()
                        .ok()
                        .map(|workspace| {
                            (
                                Some(workspace.sle_reflection_commit()),
                                workspace.rsa_applied(),
                            )
                        })
                        .unwrap_or((None, false));
                    let sle_self_symbol = ctx
                        .sle_outputs
                        .as_ref()
                        .map(|output| output.reflection.commit)
                        .or(workspace_sle_reflection);
                    let influence_inputs = InfluenceInputs {
                        cycle_id,
                        phase_commit,
                        coherence_plv,
                        phi_proxy,
                        ssm_salience,
                        ssm_novelty,
                        ncde_energy,
                        ncde_commit,
                        ncde_state_digest,
                        nsr_verdict,
                        risk: ctx.attention_risk,
                        drift: drift_score,
                        surprise: surprise_score,
                        cde_commit,
                        sle_self_symbol,
                        rsa_applied,
                        commit: influence_inputs_commit(
                            cycle_id,
                            phase_commit,
                            coherence_plv,
                            phi_proxy,
                            ssm_salience,
                            ssm_novelty,
                            ncde_energy,
                            ncde_commit,
                            ncde_state_digest,
                            nsr_verdict,
                            ctx.attention_risk,
                            drift_score,
                            surprise_score,
                            cde_commit,
                            sle_self_symbol,
                            rsa_applied,
                        ),
                    };
                    let influence_result = self.influence_state.lock().ok().map(|mut state| {
                        let outputs = state.tick(&influence_inputs);
                        (state.commit, outputs)
                    });
                    if let Some((graph_commit, outputs)) = influence_result.clone() {
                        ctx.influence_outputs = Some(outputs.clone());
                        if let Ok(mut guard) = self.last_influence_outputs.lock() {
                            *guard = Some(outputs.clone());
                        }
                        if let Ok(mut guard) = self.last_influence_root_commit.lock() {
                            *guard = Some(graph_commit);
                        }
                        if let Ok(mut workspace) = self.workspace.lock() {
                            let compact_nodes = outputs
                                .node_values
                                .iter()
                                .map(|(node, value)| (node.to_u16(), *value))
                                .collect();
                            workspace.set_influence_snapshot(
                                graph_commit,
                                outputs.pulses_root,
                                compact_nodes,
                            );
                        }
                        let signal = WorkspaceSignal::from_influence_update(
                            outputs.node_values.len(),
                            graph_commit,
                            outputs.pulses_root,
                            outputs.commit,
                            Some(attn_gain),
                            Some(pulse.slot),
                        );
                        self.publish_workspace_signal(signal);
                        self.append_influence_record(
                            cycle_id,
                            graph_commit,
                            outputs.pulses_root,
                            outputs.commit,
                            outputs.pulses.len(),
                            outputs.node_values.len(),
                        );
                    }
                    if let Some(outputs) = influence_result.as_ref().map(|(_, outputs)| outputs) {
                        let base_pressure = ctx.replay_pressure.unwrap_or(0);
                        let influenced = apply_influence_replay_pressure(base_pressure, outputs);
                        let coupling_pressure = self.coupling_influence(SignalId::ReplayPressure);
                        ctx.replay_pressure =
                            Some(apply_coupling_bias(influenced, coupling_pressure, 2000));
                    }
                    let lag_snapshot = self.coherence_lag_snapshot();
                    let attention_ctx = AttentionContext {
                        policy_class: decision.kind as u16,
                        risk_score: ctx.attention_risk,
                        integration_score: ctx.integration_score.unwrap_or(0),
                        integration_bias: ctx.integration_bias,
                        consistency_instability: ctx
                            .consistency_report
                            .as_ref()
                            .map(|report| report.drift_score)
                            .unwrap_or(0),
                        consistency_effects: ctx.consistency_effects,
                        tom_report,
                        surprise_score,
                        influence: influence_result.as_ref().map(|(_, outputs)| outputs),
                        ssm_attention_gain: ctx
                            .ssm_output
                            .as_ref()
                            .map(|output| output.ssm_attention_gain)
                            .or_else(|| {
                                self.last_ssm_output.lock().ok().and_then(|guard| {
                                    guard.as_ref().map(|output| output.ssm_attention_gain)
                                })
                            }),
                        lagged_plv: Some(lag_snapshot.avg_plv()),
                    };
                    let attention_weights = self.compute_attention(attention_ctx);
                    if let Some(weights) = attention_weights.as_ref() {
                        self.ai_port.update_attention(weights);
                        let update = AttentionUpdated {
                            channel: weights.channel,
                            gain: weights.gain,
                            replay_bias: weights.replay_bias,
                            wm_commit: ctx.ssm_output.as_ref().map(|output| output.commit),
                            commit: weights.commit,
                        };
                        self.publish_workspace_signal(WorkspaceSignal::from_attention_update(
                            &update,
                            Some(pulse.slot),
                        ));
                        if let Some(sink) = &self.attention_sink {
                            sink.publish(update);
                        }
                        if let Ok(mut guard) = self.last_attention.lock() {
                            *guard = weights.clone();
                        }
                    }
                    let attention_gain = attention_weights
                        .as_ref()
                        .map(|weights| weights.gain)
                        .unwrap_or(0);
                    let phase_bus = ctx
                        .phase_bus
                        .unwrap_or_else(|| self.latest_phase_bus(cycle_id));
                    let spike_outputs = ctx
                        .spike_outputs
                        .clone()
                        .unwrap_or_else(|| empty_spike_outputs(cycle_id));
                    let spike_root_commit = spike_outputs.accepted_root;
                    let spike_counts_ssm = spike_outputs.counts.clone();
                    let percept_commit = ctx
                        .percept_commit
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let percept_energy = ctx.percept_energy.unwrap_or(0);
                    let prior_ncde = ctx
                        .ncde_output
                        .or_else(|| self.last_ncde_output.lock().ok().and_then(|g| *g));
                    let ncde_energy = prior_ncde.map(|output| output.ncde_energy).unwrap_or(0);
                    let b_q15_bias = self.coherence_b_q15_bias(&lag_snapshot);
                    if let Some(ssm_output) = self.tick_ssm(
                        &phase_bus,
                        percept_commit,
                        percept_energy,
                        ncde_energy,
                        spike_root_commit,
                        spike_counts_ssm,
                        b_q15_bias,
                        drift_score,
                        surprise_score,
                        ctx.attention_risk,
                    ) {
                        ctx.ssm_output = Some(ssm_output.clone());
                        if let Ok(mut guard) = self.last_ssm_output.lock() {
                            *guard = Some(ssm_output.clone());
                        }
                        if let Ok(mut workspace) = self.workspace.lock() {
                            workspace.set_ssm_snapshot(
                                ssm_output.commit,
                                ssm_output.ssm_state_commit,
                                ssm_output.ssm_state_digest,
                                ssm_output.ssm_salience,
                                ssm_output.ssm_novelty,
                                ssm_output.ssm_attention_gain,
                            );
                        }
                        self.append_ssm_output_record(cycle_id, &ssm_output);
                    }
                    if let Ok(mut guard) = self.last_surprise.lock() {
                        *guard = ctx
                            .predictive_result
                            .as_ref()
                            .map(|(_, signal)| signal.clone());
                    }
                    if let Some((error, surprise_signal)) = ctx.predictive_result.as_ref() {
                        let record =
                            self.build_predictive_record(cf.as_ref(), error, surprise_signal);
                        self.archive.append(record);
                    }
                    let record = self.build_experience_record(
                        cf.as_ref(),
                        decision,
                        &ctx.thought_outputs,
                        &ctx.suppressions,
                        Some(tom_summary(tom_report)),
                        attention_weights.as_ref(),
                    );
                    let evidence_id = self.archive.append(record.clone());
                    if let Some(brain) = &self.digital_brain {
                        brain.ingest(record);
                    }
                    ctx.attention_weights = attention_weights;
                    let policy_ok = matches!(
                        ctx.decision_kind,
                        DecisionKind::DecisionKindAllow | DecisionKind::DecisionKindUnspecified
                    );
                    let phase_bus = ctx
                        .phase_bus
                        .unwrap_or_else(|| self.latest_phase_bus(cycle_id));
                    if let Some(iit_output) = ctx.iit_output.as_ref() {
                        if let Some(plan) = self.tick_tcf_plan(
                            &ctx,
                            cycle_id,
                            &phase_bus,
                            iit_output,
                            policy_ok,
                            ctx.coherence_request_replay,
                        ) {
                            ctx.tcf_plan = Some(plan);
                            if let Ok(mut guard) = self.last_tcf_plan.lock() {
                                *guard = Some(plan);
                            }
                            if let Ok(mut workspace) = self.workspace.lock() {
                                workspace.set_tcf_plan(
                                    plan.commit,
                                    plan.attention_gain_cap,
                                    plan.learning_gain_cap,
                                    plan.output_gain_cap,
                                    plan.sleep_active,
                                    plan.replay_active,
                                    plan.lock_window_buckets,
                                );
                            }
                        }
                    }
                    let coupling_snapshot = ctx.coupling_outputs.clone().or_else(|| {
                        self.last_coupling_outputs
                            .lock()
                            .ok()
                            .and_then(|g| g.clone())
                    });
                    let ssm_snapshot = ctx
                        .ssm_output
                        .clone()
                        .or_else(|| self.last_ssm_output.lock().ok().and_then(|g| g.clone()));
                    let spike_counts_ncde = spike_outputs.counts.clone();
                    let learning_gain_cap =
                        ctx.tcf_plan
                            .map(|plan| plan.learning_gain_cap)
                            .or_else(|| {
                                self.last_tcf_plan.lock().ok().and_then(|plan| {
                                    plan.as_ref().map(|plan| plan.learning_gain_cap)
                                })
                            })
                            .unwrap_or(10_000);
                    let update_mode = ctx.update_mode.unwrap_or(UpdateMode::Normal);
                    let nsr_verdict = ctx
                        .nsr_output
                        .as_ref()
                        .map(|output| output.verdict)
                        .or_else(|| ctx.nsr_report.as_ref().map(|report| report.verdict))
                        .unwrap_or(NsrVerdict::Allow);
                    let phi_proxy = ctx.integration_score.unwrap_or(0);
                    let phi_high = self.iit_params().phi_high;
                    let learning_gain_cap = self.adjust_learning_cap(
                        update_mode,
                        learning_gain_cap,
                        nsr_verdict,
                        phi_proxy,
                        phi_high,
                    );
                    if let Some(ncde_output) = self.tick_ncde(
                        cycle_id,
                        &phase_bus,
                        attention_gain,
                        coupling_snapshot.as_ref(),
                        ssm_snapshot.as_ref(),
                        spike_root_commit,
                        spike_counts_ncde,
                        ctx.attention_risk,
                        drift_score,
                        surprise_score,
                        learning_gain_cap,
                    ) {
                        ctx.ncde_output = Some(ncde_output);
                        if let Ok(mut guard) = self.last_ncde_output.lock() {
                            *guard = Some(ncde_output);
                        }
                        if let Ok(mut workspace) = self.workspace.lock() {
                            workspace.set_ncde_snapshot(
                                ncde_output.commit,
                                ncde_output.ncde_state_digest,
                                ncde_output.ncde_energy,
                                ncde_output.replay_pressure_hint,
                            );
                        }
                        self.append_ncde_output_record(cycle_id, &ncde_output);
                        let base_pressure = ctx.replay_pressure.unwrap_or(0);
                        let combined = base_pressure
                            .saturating_add(ncde_output.replay_pressure_hint)
                            .min(10_000);
                        ctx.replay_pressure = Some(combined);
                    }
                    let phase_commit = ctx
                        .phase_commit
                        .unwrap_or_else(|| self.latest_phase_bus(cycle_id).commit);
                    let ssm_snapshot = ctx.ssm_output.as_ref().cloned().or_else(|| {
                        self.last_ssm_output
                            .lock()
                            .ok()
                            .and_then(|guard| guard.clone())
                    });
                    let ssm_commit = ssm_snapshot
                        .as_ref()
                        .map(|output| output.commit)
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let ssm_novelty = ssm_snapshot
                        .as_ref()
                        .map(|output| output.ssm_novelty)
                        .unwrap_or(0);
                    let ssm_salience = ssm_snapshot
                        .as_ref()
                        .map(|output| output.ssm_salience)
                        .unwrap_or(0);
                    let iit_commit = ctx
                        .iit_output
                        .as_ref()
                        .map(|output| output.commit)
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let nsr_verdict = ctx
                        .nsr_output
                        .as_ref()
                        .map(|output| output.verdict.as_u8())
                        .or_else(|| ctx.nsr_report.as_ref().map(|report| report.verdict.as_u8()))
                        .unwrap_or(0);
                    let plv = ctx
                        .coherence_plv
                        .unwrap_or_else(|| self.latest_phase_bus(cycle_id).global_plv);
                    let lag_commit = self.update_coherence_lag(
                        phase_commit,
                        ssm_commit,
                        iit_commit,
                        nsr_verdict,
                        ssm_novelty,
                        ssm_salience,
                        plv,
                    );
                    let update_mode = ctx.update_mode.unwrap_or_else(|| {
                        self.last_update_mode
                            .lock()
                            .map(|mode| *mode)
                            .unwrap_or(UpdateMode::Normal)
                    });
                    if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.set_coherence_state(lag_commit, update_mode.as_u8());
                    }
                    self.append_update_mode_record(cycle_id, update_mode, lag_commit);
                    ctx.evidence_id = Some(evidence_id);
                }
                PulseKind::Broadcast => {
                    let snapshot = self.arbitrate_workspace(cycle_id);
                    let sle_outputs = self.tick_sle(
                        cycle_id,
                        snapshot.onn_phase_commit,
                        snapshot.onn_gamma_bucket,
                        snapshot.iit_output.as_ref(),
                        ctx.drift_score.unwrap_or(0),
                        ctx.surprise_score.unwrap_or(0),
                        ctx.attention_risk,
                        ctx.ncde_output.as_ref(),
                        &snapshot,
                    );
                    if let Some(outputs) = sle_outputs {
                        ctx.sle_outputs = Some(outputs);
                    }
                    self.append_workspace_snapshot_record(&snapshot);
                    if let Ok(mut guard) = self.last_workspace_snapshot.lock() {
                        *guard = Some(snapshot.clone());
                    }
                    ctx.workspace_snapshot_commit = Some(snapshot.commit);
                }
                PulseKind::Sleep => {
                    let policy_ok = matches!(
                        ctx.decision_kind,
                        DecisionKind::DecisionKindAllow | DecisionKind::DecisionKindUnspecified
                    );
                    let influence_outputs = ctx.influence_outputs.clone().or_else(|| {
                        self.last_influence_outputs
                            .lock()
                            .ok()
                            .and_then(|guard| guard.clone())
                    });
                    let replay_pressure = ctx.replay_pressure.unwrap_or(0);
                    let phi_proxy = ctx.integration_score.unwrap_or(0);
                    let ncde_energy = ctx
                        .ncde_output
                        .as_ref()
                        .map(|output| output.ncde_energy)
                        .unwrap_or(0);
                    let sleep_drive = influence_outputs
                        .as_ref()
                        .map(|outputs| outputs.node_value(InfluenceNodeId::SleepDrive))
                        .unwrap_or(0);
                    let coupling_sleep = self.coupling_influence(SignalId::SleepDrive);
                    let sleep_drive = apply_coupling_bias_i16(sleep_drive, coupling_sleep, 2000);
                    let params = self
                        .structural_store
                        .lock()
                        .map(|store| store.current.clone())
                        .unwrap_or_else(|_| StructuralStore::default().current);
                    let (sleep_active, replay_active) =
                        self.tcf_sleep_replay(Some(&ctx)).unwrap_or_else(|| {
                            let sleep_active = derive_sleep_active(
                                matches!(pulse.kind, PulseKind::Sleep),
                                replay_pressure,
                                phi_proxy,
                                sleep_drive,
                                ncde_energy,
                                &params,
                            );
                            let replay_active = replay_pressure >= 5_000;
                            (sleep_active, replay_active)
                        });
                    let prev_snapshot = self
                        .last_workspace_snapshot
                        .lock()
                        .ok()
                        .and_then(|guard| guard.clone())
                        .unwrap_or_else(|| self.latest_workspace_snapshot(cycle_id));
                    let prev_phi = prev_snapshot
                        .iit_output
                        .as_ref()
                        .map(|output| output.phi_proxy)
                        .unwrap_or(0);
                    let prev_plv = prev_snapshot.onn_global_plv;
                    let prev_drift = drift_score_from_snapshot(&prev_snapshot);
                    let (onn_params_commit, ncde_params_commit, cde_params_commit) = self
                        .runtime_modules
                        .lock()
                        .map(|modules| {
                            (
                                modules.phase.params().commit,
                                modules.ncde.params().commit,
                                modules.cde.params().commit,
                            )
                        })
                        .unwrap_or_else(|_| {
                            (
                                OnnParams::default().commit,
                                NcdeParams::default().commit,
                                CdeParams::default().commit,
                            )
                        });
                    let tcf_params_commit = self
                        .tcf_port
                        .lock()
                        .map(|tcf| tcf.params().commit)
                        .unwrap_or_else(|_| TcfConfig::default().commit);
                    let feature_params_commit = self
                        .feature_params
                        .lock()
                        .map(|params| params.commit)
                        .unwrap_or_else(|_| FeatureSpikeParams::default().commit);
                    let nsr_verdict = ctx
                        .nsr_output
                        .as_ref()
                        .map(|output| output.verdict.as_u8())
                        .unwrap_or(0);
                    let inputs = RsaInputs::new(
                        cycle_id,
                        sleep_active,
                        replay_active,
                        nsr_verdict,
                        policy_ok,
                        phi_proxy,
                        ctx.coherence_plv.unwrap_or(0),
                        ctx.drift_score.unwrap_or(0),
                        ctx.surprise_score.unwrap_or(0),
                        ctx.attention_risk,
                        prev_phi,
                        prev_plv,
                        prev_drift,
                        self.coupling_influence(SignalId::RsaProposalStrength),
                        onn_params_commit,
                        tcf_params_commit,
                        ncde_params_commit,
                        cde_params_commit,
                        feature_params_commit,
                    );
                    let mut rsa_core = self.rsa_core.lock().unwrap_or_else(|err| err.into_inner());
                    let mut outputs = rsa_core.tick(&inputs);
                    let proposal_commit = outputs.proposal.as_ref().map(|proposal| proposal.commit);
                    let nsr_apply_output = if outputs.proposal.is_some() {
                        let phase_bus = ctx
                            .phase_bus
                            .unwrap_or_else(|| self.latest_phase_bus(cycle_id));
                        let policy_commit = ctx
                            .decision
                            .as_ref()
                            .map(digest_policy_commit)
                            .unwrap_or_else(|| Digest32::new([0u8; 32]));
                        let fallback_iit = IitOutput {
                            cycle_id,
                            phi_proxy: 0,
                            tighten_sync: false,
                            damp_output: false,
                            damp_learning: false,
                            request_replay: false,
                            hints_commit: Digest32::new([0u8; 32]),
                            commit: Digest32::new([0u8; 32]),
                        };
                        let iit_output = ctx.iit_output.as_ref().unwrap_or(&fallback_iit);
                        let spike_outputs = ctx
                            .spike_outputs
                            .clone()
                            .unwrap_or_else(|| empty_spike_outputs(cycle_id));
                        let jepa_surprise = ctx
                            .jepa_outputs
                            .as_ref()
                            .map(|output| output.surprise)
                            .or(ctx.surprise_score)
                            .unwrap_or(0);
                        let nsr_inputs = self.build_nsr_inputs(
                            cycle_id,
                            &phase_bus,
                            influence_outputs.as_ref(),
                            ctx.cde_v1_output.as_ref(),
                            ctx.cde_output.as_ref(),
                            ctx.ssm_output.as_ref(),
                            ctx.ncde_output.as_ref(),
                            iit_output,
                            ctx.self_state.as_ref(),
                            spike_outputs.counts.clone(),
                            spike_outputs.max_intensity,
                            replay_pressure,
                            ctx.drift_score.unwrap_or(0),
                            ctx.surprise_score.unwrap_or(0),
                            jepa_surprise,
                            ctx.attention_risk,
                            false,
                            false,
                            false,
                            policy_commit,
                            policy_ok,
                            matches!(pulse.kind, PulseKind::Sleep),
                            self.tcf_plan_for(Some(&ctx)),
                        );
                        self.runtime_modules
                            .lock()
                            .ok()
                            .map(|modules| modules.nsr.tick(&nsr_inputs))
                    } else {
                        None
                    };
                    if let Some(output) = nsr_apply_output.as_ref() {
                        if output.verdict != NsrVerdict::Allow {
                            outputs.decision =
                                rsa_v0::RsaDecision::new(false, outputs.decision.reason_mask | 2);
                        }
                    }
                    let update_mode = ctx.update_mode.unwrap_or_else(|| {
                        self.last_update_mode
                            .lock()
                            .map(|mode| *mode)
                            .unwrap_or(UpdateMode::Normal)
                    });
                    let tcf_budget = self
                        .tcf_plan_for(Some(&ctx))
                        .map(|plan| plan.learning_gain_cap)
                        .unwrap_or(0);
                    let phi_high = self.iit_params().phi_high;
                    let nsr_verdict = ctx
                        .nsr_output
                        .as_ref()
                        .map(|output| output.verdict)
                        .unwrap_or(NsrVerdict::Allow);
                    let (rsa_apply_allowed, rsa_apply_commit) = self.rsa_apply_gate(
                        update_mode,
                        nsr_verdict,
                        phi_proxy,
                        phi_high,
                        tcf_budget,
                    );
                    if !rsa_apply_allowed {
                        outputs.decision = rsa_v0::RsaDecision::new(
                            false,
                            outputs.decision.reason_mask | RSA_REASON_MODE_BLOCK,
                        );
                    }
                    if outputs.decision.apply {
                        if let Some(proposal) = outputs.proposal.clone() {
                            let (onn_params, ncde_params, cde_params) = self
                                .runtime_modules
                                .lock()
                                .map(|modules| {
                                    (
                                        modules.phase.params(),
                                        modules.ncde.params(),
                                        modules.cde.params(),
                                    )
                                })
                                .unwrap_or_else(|_| {
                                    (
                                        OnnParams::default(),
                                        NcdeParams::default(),
                                        CdeParams::default(),
                                    )
                                });
                            let tcf_params = self
                                .tcf_port
                                .lock()
                                .map(|tcf| tcf.params())
                                .unwrap_or_else(|_| TcfConfig::default());
                            let feature_params = self
                                .feature_params
                                .lock()
                                .map(|params| *params)
                                .unwrap_or_else(|_| FeatureSpikeParams::default());
                            let (next_onn, next_tcf, next_ncde, next_cde, next_feature) =
                                apply_rsa_deltas(
                                    &proposal,
                                    onn_params,
                                    tcf_params,
                                    ncde_params,
                                    cde_params,
                                    feature_params,
                                );
                            let applied_root = rsa_v0::params_root(
                                next_onn.commit,
                                next_tcf.commit,
                                next_ncde.commit,
                                next_cde.commit,
                                next_feature.commit,
                            );
                            let snapshot = ParamSnapshot::new(
                                cycle_id,
                                inputs.onn_params_commit,
                                inputs.tcf_params_commit,
                                inputs.ncde_params_commit,
                                inputs.cde_params_commit,
                                inputs.feature_params_commit,
                                rsa_v0::snapshot_deltas_from_proposal(&proposal),
                                applied_root,
                            );
                            let snapshot_chain_commit = commit_snapshot_chain(
                                rsa_core.last_snapshot_commit,
                                snapshot.commit,
                            );

                            let next_onn_for_store = next_onn;
                            if let Ok(mut modules) = self.runtime_modules.lock() {
                                modules.phase.set_params(next_onn);
                                modules.ncde.set_params(next_ncde);
                                modules.cde.apply_params(next_cde);
                            }
                            if let Ok(mut tcf) = self.tcf_port.lock() {
                                tcf.set_params(next_tcf);
                            }
                            if let Ok(mut params) = self.feature_params.lock() {
                                *params = next_feature;
                            }
                            if let Ok(mut store) = self.structural_store.lock() {
                                let mut current = store.current.clone();
                                current.onn = OnnKnobs::new(
                                    next_onn_for_store.k_couple,
                                    next_onn_for_store.k_dither,
                                    next_onn_for_store.couple_clamp_q12,
                                );
                                current.ncde = next_ncde;
                                let updated = StructuralParams::new(
                                    current.onn,
                                    current.snn,
                                    current.nsr,
                                    current.replay,
                                    current.ssm,
                                    current.ncde,
                                    current.rsa,
                                );
                                store.apply_params(updated);
                            }
                            outputs.applied_params_root = applied_root;
                            outputs.snapshot_chain_commit = snapshot_chain_commit;
                            rsa_core.record_snapshot_chain(snapshot_chain_commit);
                            rsa_core.record_apply(&proposal.deltas, applied_root, cycle_id);
                        }
                    }
                    outputs.recompute_commit();
                    if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.set_rsa_output(
                            outputs.commit,
                            proposal_commit,
                            outputs.decision.apply,
                            rsa_apply_allowed,
                            rsa_apply_commit,
                            outputs.decision.reason_mask,
                            outputs.applied_params_root,
                            outputs.snapshot_chain_commit,
                        );
                    }
                    self.append_rsa_outputs_record(cycle_id, &outputs);
                    let phi = ctx.integration_score.unwrap_or(0);
                    let surprise_score = self.surprise_score_from_ctx(&ctx);
                    let workspace_commit = ctx
                        .workspace_snapshot_commit
                        .unwrap_or_else(|| self.latest_workspace_snapshot(cycle_id).commit);
                    let context = RsaContext {
                        cycle_id,
                        pulse_kind: PulseKind::Sleep,
                        phi,
                        surprise_score,
                        workspace_commit,
                    };
                    let mut proposals: Vec<RsaProposal> = Vec::new();
                    for hook in &self.rsa_hooks {
                        proposals.extend(hook.propose(&context));
                    }
                    let proposal_digest = digest_rsa_proposals(&proposals);
                    self.publish_workspace_signal(WorkspaceSignal::from_sleep_proposals(
                        proposals.len(),
                        proposal_digest,
                        None,
                        Some(pulse.slot),
                    ));
                    if !proposals.is_empty() {
                        self.archive.append(self.build_rsa_proposals_record(
                            cycle_id,
                            &proposals,
                            proposal_digest,
                        ));
                    }
                }
            }
        }

        let evidence_id = ctx
            .evidence_id
            .unwrap_or_else(|| EvidenceId::new(format!("cycle-{cycle_id}")));
        Ok(RouterOutcome {
            evidence_id,
            decision_kind: ctx.decision_kind,
            speech_outputs: ctx.speech_outputs,
            integration_score: ctx.integration_score,
            workspace_snapshot_commit: ctx.workspace_snapshot_commit,
            surprise_signal: ctx.predictive_result.map(|(_, signal)| signal),
            structural_stats: ctx.structural_stats,
            structural_proposal: ctx.structural_proposal,
        })
    }

    fn ensure_allowed(&self, decision: &PolicyDecision) -> Result<(), RouterError> {
        match decision.kind {
            kind if kind == DecisionKind::DecisionKindUnspecified as i32 => Ok(()),
            kind if kind == DecisionKind::DecisionKindAllow as i32 => Ok(()),
            kind if kind == DecisionKind::DecisionKindDeny as i32 => {
                Err(RouterError::PolicyDenied(kind))
            }
            kind => Err(RouterError::PolicyDenied(kind)),
        }
    }

    fn build_experience_record(
        &self,
        cf: &ControlFrame,
        decision: &PolicyDecision,
        thought_outputs: &[AiOutput],
        suppressions: &[OutputSuppressionInfo],
        tom_summary: Option<String>,
        attention: Option<&AttentionWeights>,
    ) -> ExperienceRecord {
        let record_id = format!("exp-{}", cf.frame_id);
        let mut payload = format!(
            "frame_id={};policy_id={};decision_kind={};decision_action={}",
            cf.frame_id, cf.policy_id, decision.kind, decision.action
        )
        .into_bytes();

        if !thought_outputs.is_empty() {
            let thoughts = thought_outputs
                .iter()
                .map(|output| output.content.as_str())
                .collect::<Vec<_>>()
                .join("|");
            let notes = format!(";ai_thoughts={thoughts}");
            payload.extend_from_slice(notes.as_bytes());
        }
        if let Some(score) = thought_outputs
            .iter()
            .find_map(|output| output.integration_score)
        {
            let notes = format!(";integration_score={score}");
            payload.extend_from_slice(notes.as_bytes());
        }
        if !suppressions.is_empty() {
            let details = suppressions
                .iter()
                .map(|suppression| {
                    let channel = match suppression.channel {
                        OutputChannel::Thought => "thought",
                        OutputChannel::Speech => "speech",
                    };
                    format!(
                        "{channel}:{risk}:{reason}",
                        risk = suppression.risk,
                        reason = suppression.reason_digest
                    )
                })
                .collect::<Vec<_>>()
                .join("|");
            let notes = format!(";output_suppressed={details}");
            payload.extend_from_slice(notes.as_bytes());
        }
        if let Some(summary) = tom_summary {
            let notes = format!(";tom_summary={summary}");
            payload.extend_from_slice(notes.as_bytes());
        }
        if let Some(attn) = attention {
            let notes = format!(
                ";attn_channel={};attn_gain={};attn_replay_bias={};attn_commit={}",
                attn.channel.as_str(),
                attn.gain,
                attn.replay_bias,
                attn.commit
            );
            payload.extend_from_slice(notes.as_bytes());
        }

        ExperienceRecord {
            record_id,
            observed_at_ms: cf.issued_at_ms,
            subject_id: cf.policy_id.clone(),
            payload,
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        }
    }

    fn append_archive_record(&self, kind: RecordKind, payload_commit: Digest32, meta: RecordMeta) {
        let mut appender = self.archive_appender.lock().expect("archive appender lock");
        let record = appender.build_record_with_commit(kind, payload_commit, meta);
        self.archive_store.append(record);
    }

    fn append_workspace_snapshot_record(&self, snapshot: &WorkspaceSnapshot) {
        let tier = snapshot.broadcast.len().min(u8::MAX as usize) as u8;
        let boundary_commit = boundary_workspace_broadcast(snapshot);
        let meta = RecordMeta {
            cycle_id: snapshot.cycle_id,
            tier,
            flags: snapshot.recursion_used,
            boundary_commit,
        };
        self.append_archive_record(RecordKind::WorkspaceSnapshot, snapshot.commit, meta);
    }

    fn append_rsa_outputs_record(&self, cycle_id: u64, outputs: &RsaOutputs) {
        let proposal_flag = outputs.proposal.is_some() as u8;
        let payload = format!(
            "rsa_commit={};decision_commit={};applied_params_root={};proposal={}",
            outputs.commit, outputs.decision.commit, outputs.applied_params_root, proposal_flag
        )
        .into_bytes();
        let record_id = format!("rsa-{}", cycle_id);
        let record = build_compact_record(record_id, cycle_id, "rsa", payload);
        self.archive.append(record);
    }

    fn append_cycle_plan_record(&self, plan: &CyclePlan, planned: &CyclePlanned) {
        let meta = RecordMeta {
            cycle_id: planned.cycle_id,
            tier: planned.pulse_count,
            flags: 0,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        self.append_archive_record(RecordKind::CyclePlan, plan.commit, meta);
    }

    fn append_phase_frame_record(&self, cycle_id: u64, phase_bus: &PhaseBus) {
        let meta = RecordMeta {
            cycle_id,
            tier: phase_bus.gamma_bucket,
            flags: phase_bus.global_plv,
            boundary_commit: phase_bus.phase_commit,
        };
        self.append_archive_record(
            RecordKind::Other(PHASE_FRAME_RECORD_KIND),
            phase_bus.commit,
            meta,
        );
    }

    fn append_self_state_record(&self, state: &SelfState) {
        let meta = RecordMeta {
            cycle_id: state.cycle_id,
            tier: 0,
            flags: state.consistency,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        self.append_archive_record(RecordKind::SelfState, state.commit, meta);
    }

    fn append_iit_output_record(&self, cycle_id: u64, output: &IitOutput) {
        let payload = format!(
            "commit={};phi_proxy={};hints_commit={}",
            output.commit, output.phi_proxy, output.hints_commit
        )
        .into_bytes();
        let record_id = format!("iit-{cycle_id}-{}", hex::encode(output.commit.as_bytes()));
        let record = build_compact_record(record_id, cycle_id, "iit", payload);
        self.archive.append(record);
    }

    fn append_iit_report_record(&self, cycle_id: u64, report: &IitReport) {
        let meta = RecordMeta {
            cycle_id,
            tier: iit_band_tier(report.band),
            flags: report.phi,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        self.append_archive_record(RecordKind::IitReport, report.commit, meta);
    }

    fn append_consistency_report_record(&self, cycle_id: u64, report: &ConsistencyReport) {
        let meta = RecordMeta {
            cycle_id,
            tier: drift_band_tier(report.band),
            flags: report.drift_score,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        self.append_archive_record(RecordKind::ConsistencyReport, report.commit, meta);
    }

    fn append_output_event_record(&self, cycle_id: u64, event: &OutputRouterEvent) {
        let (payload_commit, tier, flags) = match event {
            OutputRouterEvent::ThoughtBuffered { frame } => (
                output_event_commit(b"thought_buffered", frame.commit, None, 0),
                1,
                0,
            ),
            OutputRouterEvent::SpeechEmitted { frame } => (
                output_event_commit(b"speech_emitted", frame.commit, None, 0),
                2,
                0,
            ),
            OutputRouterEvent::OutputSuppressed {
                frame,
                evidence,
                risk,
                ..
            } => (
                output_event_commit(b"output_suppressed", frame.commit, Some(*evidence), *risk),
                3,
                *risk,
            ),
        };
        let meta = RecordMeta {
            cycle_id,
            tier,
            flags,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        self.append_archive_record(RecordKind::OutputEvent, payload_commit, meta);
    }

    fn surprise_score_from_ctx(&self, ctx: &StageContext) -> u16 {
        if let Some(output) = ctx.jepa_outputs.as_ref() {
            return output.surprise;
        }
        if let Ok(guard) = self.last_jepa_output.lock() {
            if let Some(output) = guard.as_ref() {
                return output.surprise;
            }
        }
        if let Some(score) = ctx.surprise_score {
            return score;
        }
        if let Some((_, signal)) = ctx.predictive_result.as_ref() {
            return signal.score;
        }
        self.last_surprise
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|signal| signal.score))
            .unwrap_or(0)
    }

    fn world_state_commit_from_ctx(&self, ctx: &StageContext) -> Digest32 {
        ctx.jepa_outputs
            .as_ref()
            .map(|output| output.world_state)
            .or_else(|| {
                self.last_jepa_output
                    .lock()
                    .ok()
                    .and_then(|guard| guard.as_ref().map(|output| output.world_state))
            })
            .unwrap_or_else(|| Digest32::new([0u8; 32]))
    }

    #[allow(clippy::too_many_arguments)]
    fn tick_onn_phase(
        &self,
        cycle_id: u64,
        risk: u16,
        drift_score: u16,
        surprise_score: u16,
        lock_window_buckets: u8,
        slot: u8,
    ) -> ucf_onn::OnnOutputs {
        self.sync_onn_params();
        let ssm_state_commit = self
            .last_ssm_output
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|out| out.ssm_state_commit))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let ncde_state_digest = self
            .last_ncde_output
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|out| out.ncde_state_digest))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let cde_commit = self
            .last_cde_output
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|out| out.commit))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let nsr_trace_root = self
            .last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|snap| snap.nsr_trace_root))
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let iit_hints_commit = self
            .last_iit_hints
            .lock()
            .ok()
            .map(|hints| hints.hints_commit)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let inputs = OnnInputs::new(
            cycle_id,
            ssm_state_commit,
            ncde_state_digest,
            cde_commit,
            nsr_trace_root,
            iit_hints_commit,
            lock_window_buckets,
            risk,
            drift_score,
            surprise_score,
        );
        let outputs = {
            let budget = self.current_gain_budget();
            let mut modules = self.runtime_modules.lock().expect("runtime modules lock");
            modules.phase.tick_with_budget(&inputs, &budget)
        };
        if let Ok(mut guard) = self.last_phase_bus.lock() {
            *guard = Some(outputs.phase_bus);
        }
        if let Ok(mut guard) = self.last_phase_lock.lock() {
            *guard = Some(outputs.lock);
        }
        let priority = 3200u16
            .saturating_add(outputs.phase_bus.global_plv / 5)
            .min(10_000);
        let summary = format!(
            "PHASE bucket={} PLV={}",
            outputs.phase_bus.gamma_bucket, outputs.phase_bus.global_plv
        );
        self.publish_workspace_signal(WorkspaceSignal {
            kind: SignalKind::Brain,
            priority,
            digest: outputs.commit,
            summary,
            slot,
        });
        if let Ok(mut workspace) = self.workspace.lock() {
            let lock_window_source_cycle = cycle_id.saturating_sub(1);
            workspace.set_onn_snapshot(
                outputs.phase_bus.phase_commit,
                outputs.phase_bus.gamma_bucket,
                outputs.phase_bus.global_plv,
                lock_window_source_cycle,
            );
        }
        self.append_phase_frame_record(cycle_id, &outputs.phase_bus);
        outputs
    }

    fn tick_jepa(&self, ctx: &mut StageContext, cycle_id: u64, phase_bus: &PhaseBus) {
        let percept_commit = ctx
            .percept_commit
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let percept_energy = ctx.percept_energy.unwrap_or(0);
        let ssm_state_digest = ctx
            .ssm_output
            .as_ref()
            .map(|output| output.ssm_state_digest)
            .or_else(|| {
                self.last_ssm_output
                    .lock()
                    .ok()
                    .and_then(|guard| guard.as_ref().map(|output| output.ssm_state_digest))
            })
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let coupling_root = ctx
            .coupling_outputs
            .as_ref()
            .map(|outputs| outputs.influences_root)
            .or_else(|| {
                self.last_coupling_outputs
                    .lock()
                    .ok()
                    .and_then(|guard| guard.as_ref().map(|outputs| outputs.influences_root))
            })
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let outputs = {
            let mut modules = self.runtime_modules.lock().expect("runtime modules lock");
            let last_world_state = modules.world.last_world_state();
            let inputs = JepaInputs::new(
                cycle_id,
                percept_commit,
                percept_energy,
                ssm_state_digest,
                phase_bus.commit,
                phase_bus.gamma_bucket,
                last_world_state,
                coupling_root,
            );
            modules.world.tick(&inputs)
        };
        ctx.surprise_score = Some(outputs.surprise);
        ctx.jepa_outputs = Some(outputs);
        if let Ok(mut guard) = self.last_jepa_output.lock() {
            *guard = Some(outputs);
        }
        if let Ok(mut workspace) = self.workspace.lock() {
            workspace.set_jepa_snapshot(outputs.world_state, outputs.prediction, outputs.surprise);
        }
        self.append_jepa_output_record(cycle_id, &outputs);
    }

    fn build_sle_reflex_record(&self, cycle_id: u64, reflex: &SelfReflex) -> ExperienceRecord {
        let record_id = format!(
            "sle-reflex-{}-{}",
            cycle_id,
            hex::encode(reflex.self_symbol.as_bytes())
        );
        let payload = format!(
            "self_symbol={};loop_level={};delta={}",
            reflex.self_symbol, reflex.loop_level, reflex.delta
        )
        .into_bytes();
        ExperienceRecord {
            record_id,
            observed_at_ms: cycle_id,
            subject_id: "sle".to_string(),
            payload,
            digest: Some(digest32_to_proto(reflex.self_symbol)),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tick_sle(
        &self,
        cycle_id: u64,
        phase_commit: Digest32,
        phase_bucket: u8,
        iit_output: Option<&IitOutput>,
        drift_score: u16,
        surprise_score: u16,
        attention_risk: u16,
        ncde_output: Option<&NcdeOutputs>,
        workspace_snapshot: &WorkspaceSnapshot,
    ) -> Option<SleOutputs> {
        let phi_proxy = iit_output.map(|output| output.phi_proxy).unwrap_or(0);
        let nsr_trace_root = workspace_snapshot
            .nsr_trace_root
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let nsr_verdict = workspace_snapshot.nsr_verdict.unwrap_or(0);
        let ncde_state_digest = ncde_output
            .map(|output| output.ncde_state_digest)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let ncde_energy = ncde_output
            .map(|output| output.ncde_energy)
            .unwrap_or(workspace_snapshot.ncde_energy);
        let cde_commit = workspace_snapshot.cde_commit;
        let inputs = SleInputs::new(
            cycle_id,
            phase_commit,
            phase_bucket,
            workspace_snapshot.ssm_state_commit,
            workspace_snapshot.ssm_salience,
            workspace_snapshot.ssm_novelty,
            ncde_state_digest,
            ncde_energy,
            cde_commit,
            nsr_verdict,
            nsr_trace_root,
            phi_proxy,
            workspace_snapshot.onn_global_plv,
            workspace_snapshot.tcf_sleep_active,
            workspace_snapshot.tcf_replay_active,
            attention_risk,
            drift_score,
            surprise_score,
        );

        let outputs = self
            .runtime_modules
            .lock()
            .ok()
            .map(|mut modules| modules.sle.tick(&inputs))?;

        if let Ok(mut workspace) = self.workspace.lock() {
            workspace.set_sle_outputs(SleOutputsSnapshot {
                sle_commit: outputs.commit,
                reflection_commit: outputs.reflection.commit,
                reflection_class: outputs.reflection.class as u8,
                reflection_intensity: outputs.reflection.intensity,
                thought_only_root: outputs.thought_only_root,
                ssm_bias: outputs.ssm_bias,
                cde_bias: outputs.cde_bias,
                request_replay: outputs.request_replay,
            });
        }

        self.append_sle_outputs_record(cycle_id, &outputs);
        Some(outputs)
    }

    fn append_sle_outputs_record(&self, cycle_id: u64, outputs: &SleOutputs) {
        let record_id = format!("sle-{}", cycle_id);
        let mut payload = Vec::with_capacity(Digest32::LEN * 2 + 3);
        payload.extend_from_slice(outputs.reflection.commit.as_bytes());
        payload.push(outputs.reflection.class as u8);
        payload.extend_from_slice(&outputs.reflection.intensity.to_be_bytes());
        payload.extend_from_slice(outputs.thought_only_root.as_bytes());
        let record = build_compact_record(record_id, cycle_id, "sle", payload);
        self.archive.append(record);
    }

    fn build_iit_action_record(&self, cycle_id: u64, action: &IitAction) -> ExperienceRecord {
        let record_id = format!(
            "iit-action-{}-{}",
            cycle_id,
            hex::encode(action.commit.as_bytes())
        );
        let kind = match action.kind {
            IitActionKind::Fusion => "FUSION",
            IitActionKind::Isolate => "ISOLATE",
            IitActionKind::ReplayBias => "REPLAY_BIAS",
            IitActionKind::Throttle => "THROTTLE",
        };
        let payload = format!("kind={kind};intensity={}", action.intensity).into_bytes();
        ExperienceRecord {
            record_id,
            observed_at_ms: cycle_id,
            subject_id: "iit".to_string(),
            payload,
            digest: Some(digest32_to_proto(action.commit)),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    fn build_recursion_budget_record(
        &self,
        cycle_id: u64,
        budget: &RecursionBudget,
    ) -> ExperienceRecord {
        let record_id = format!(
            "rdc-budget-{}-{}",
            cycle_id,
            hex::encode(budget.commit.as_bytes())
        );
        let payload = format!(
            "depth={};steps={};decay={};commit={}",
            budget.max_depth, budget.per_cycle_steps, budget.level_decay, budget.commit
        )
        .into_bytes();
        ExperienceRecord {
            record_id,
            observed_at_ms: cycle_id,
            subject_id: "rdc".to_string(),
            payload,
            digest: Some(digest32_to_proto(budget.commit)),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    fn build_consistency_action_record(
        &self,
        cycle_id: u64,
        action: &ConsistencyAction,
    ) -> ExperienceRecord {
        let record_id = format!(
            "consistency-action-{}-{}",
            cycle_id,
            hex::encode(action.commit.as_bytes())
        );
        let kind = match action.kind {
            ConsistencyActionKind::DampNoise => "DAMP_NOISE",
            ConsistencyActionKind::ReduceRecursion => "REDUCE_RECURSION",
            ConsistencyActionKind::IncreaseReplay => "INCREASE_REPLAY",
            ConsistencyActionKind::ThrottleOutput => "THROTTLE_OUTPUT",
        };
        let payload = format!("kind={kind};intensity={}", action.intensity).into_bytes();
        ExperienceRecord {
            record_id,
            observed_at_ms: cycle_id,
            subject_id: "consistency".to_string(),
            payload,
            digest: Some(digest32_to_proto(action.commit)),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    fn build_rsa_proposals_record(
        &self,
        cycle_id: u64,
        proposals: &[RsaProposal],
        digest: Digest32,
    ) -> ExperienceRecord {
        let record_id = format!(
            "rsa-proposals-{}-{}",
            cycle_id,
            hex::encode(digest.as_bytes())
        );
        let payload = proposals
            .iter()
            .map(|proposal| {
                format!(
                    "{}|{}|{}|{}",
                    proposal.id, proposal.target, proposal.expected_gain, proposal.risks
                )
            })
            .collect::<Vec<_>>()
            .join(";")
            .into_bytes();
        ExperienceRecord {
            record_id,
            observed_at_ms: cycle_id,
            subject_id: "rsa".to_string(),
            payload,
            digest: Some(digest32_to_proto(digest)),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    fn compute_attention(&self, ctx: AttentionContext<'_>) -> Option<AttentionWeights> {
        let controller = self.attention_controller.as_ref()?;
        let integration_score = apply_integration_bias(ctx.integration_score, ctx.integration_bias);
        let inputs = AttnInputs {
            policy_class: ctx.policy_class,
            risk_score: ctx.risk_score,
            integration_score,
            consistency_instability: ctx.consistency_instability,
            intent_type: intent_type_code(ctx.tom_report.intent.intent),
            surprise_score: ctx.surprise_score,
        };
        let weights = controller.compute(&inputs);
        let weights = apply_consistency_effects(weights, ctx.consistency_effects);
        let weights = apply_influence_effects(weights, ctx.influence);
        let weights = self.apply_ncde_attention_bias(weights);
        let weights = self.apply_ssm_attention_bias(weights, ctx.ssm_attention_gain);
        let weights = self.apply_coupling_attention_bias(weights);
        let weights = self.enforce_ssm_attention_dominance(weights, ctx.ssm_attention_gain);
        let tcf_cap = self.tcf_attention_cap();
        let memory_cap =
            self.attention_cap_from_memory(ctx.ssm_attention_gain, ctx.lagged_plv, tcf_cap);
        Some(self.apply_tcf_attention_cap(weights, tcf_cap, Some(memory_cap)))
    }

    fn apply_coupling_attention_bias(&self, mut weights: AttentionWeights) -> AttentionWeights {
        let influence = self.coupling_influence(SignalId::AttentionFinalGain);
        if influence == 0 {
            return weights;
        }
        let gain = apply_coupling_bias(weights.gain, influence, 2000);
        if gain != weights.gain {
            weights.gain = gain;
            weights.commit = commit_attention_override(&weights);
        }
        weights
    }

    fn apply_tcf_attention_cap(
        &self,
        mut weights: AttentionWeights,
        tcf_cap: u16,
        memory_cap: Option<u16>,
    ) -> AttentionWeights {
        let cap = memory_cap.unwrap_or(tcf_cap).min(tcf_cap);
        if cap == 0 {
            weights.gain = 0;
            weights.commit = commit_attention_override(&weights);
            return weights;
        }
        if weights.gain > cap {
            weights.gain = cap;
            weights.commit = commit_attention_override(&weights);
        }
        weights
    }

    fn tcf_attention_cap(&self) -> u16 {
        self.last_tcf_plan
            .lock()
            .ok()
            .and_then(|plan| plan.as_ref().map(|plan| plan.attention_gain_cap))
            .unwrap_or(10_000)
    }

    fn attention_cap_from_memory(
        &self,
        ssm_attention_gain: Option<u16>,
        lagged_plv: Option<u16>,
        tcf_cap: u16,
    ) -> u16 {
        let mut cap = ssm_attention_gain.unwrap_or(tcf_cap).min(tcf_cap);
        if let Some(plv) = lagged_plv {
            if plv < COHERENCE_PLV_LOW {
                let plv_penalty_cap = cap.saturating_sub(cap / 5);
                cap = cap.max(plv_penalty_cap);
            }
        }
        cap
    }

    fn tcf_plan_for(&self, ctx: Option<&StageContext>) -> Option<TcfPlan> {
        ctx.and_then(|ctx| ctx.tcf_plan)
            .or_else(|| self.last_tcf_plan.lock().ok().and_then(|plan| *plan))
    }

    fn lagged_lock_window_buckets(&self) -> u8 {
        self.last_tcf_plan
            .lock()
            .ok()
            .and_then(|plan| plan.as_ref().map(|plan| plan.lock_window_buckets))
            .unwrap_or(1)
    }

    fn tcf_sleep_replay(&self, ctx: Option<&StageContext>) -> Option<(bool, bool)> {
        self.tcf_plan_for(ctx)
            .map(|plan| (plan.sleep_active, plan.replay_active))
    }

    fn iit_params(&self) -> IitParams {
        self.runtime_modules
            .lock()
            .map(|modules| modules.iit.params())
            .unwrap_or_else(|_| IitParams::default())
    }

    fn coherence_lag_snapshot(&self) -> CoherenceLag {
        self.coherence_lag
            .lock()
            .map(|lag| *lag)
            .unwrap_or_else(|_| CoherenceLag::new())
    }

    #[allow(clippy::too_many_arguments)]
    fn update_coherence_lag(
        &self,
        phase_commit: Digest32,
        ssm_commit: Digest32,
        iit_commit: Digest32,
        nsr_verdict: u8,
        novelty: u16,
        salience: u16,
        plv: u16,
    ) -> Digest32 {
        let mut lag = self
            .coherence_lag
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        lag.push(
            phase_commit,
            ssm_commit,
            iit_commit,
            nsr_verdict,
            novelty,
            salience,
            plv,
        );
        lag.commit
    }

    fn consume_force_stabilize(&self) -> bool {
        let mut current = self.force_stabilize_cycles.load(Ordering::Relaxed);
        while current > 0 {
            match self.force_stabilize_cycles.compare_exchange(
                current,
                current - 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
        false
    }

    fn compute_update_mode(
        &self,
        lag: &CoherenceLag,
        nsr_verdict: NsrVerdict,
        risk: u16,
        surprise: u16,
        drift: u16,
        phi: u16,
    ) -> UpdateMode {
        let base = update_mode_seed(lag.phase_commit[0], lag.ssm_commit[0], lag.iit_commit[0]);
        let params = self.iit_params();
        if self.consume_force_stabilize() {
            return UpdateMode::Stabilize;
        }
        if nsr_verdict != NsrVerdict::Allow {
            return UpdateMode::Stabilize;
        }
        if risk >= COHERENCE_RISK_HIGH {
            return UpdateMode::Conservative;
        }
        if surprise >= params.surprise_hi && phi >= params.phi_high {
            return UpdateMode::Exploratory;
        }
        if drift >= params.drift_hi || phi < params.phi_low {
            return UpdateMode::Stabilize;
        }
        base
    }

    fn coherence_b_q15_bias(&self, lag: &CoherenceLag) -> i16 {
        let mut bias = 0i16;
        if lag.novelty_trend_up() {
            bias = bias.saturating_add(64);
        }
        if lag.avg_salience() < 2000 {
            bias = bias.saturating_sub(64);
        }
        bias
    }

    fn adjust_learning_cap(
        &self,
        mode: UpdateMode,
        base_cap: u16,
        nsr_verdict: NsrVerdict,
        phi: u16,
        phi_high: u16,
    ) -> u16 {
        let mut cap = base_cap;
        match mode {
            UpdateMode::Conservative => {
                cap = ((u32::from(base_cap) * 70) / 100).min(u32::from(u16::MAX)) as u16;
            }
            UpdateMode::Exploratory => {
                if nsr_verdict == NsrVerdict::Allow && phi >= phi_high {
                    cap = ((u32::from(base_cap) * 120) / 100).min(10_000u32) as u16;
                }
            }
            UpdateMode::Stabilize => {
                cap = cap.min(3000);
            }
            UpdateMode::Normal => {}
        }
        cap
    }

    fn coherence_request_replay(&self, mode: UpdateMode, surprise: u16, phi: u16) -> bool {
        let params = self.iit_params();
        (mode == UpdateMode::Stabilize && surprise >= params.surprise_hi) || phi < params.phi_low
    }

    fn append_update_mode_record(&self, cycle_id: u64, mode: UpdateMode, lag_commit: Digest32) {
        let meta = RecordMeta {
            cycle_id,
            tier: mode.as_u8(),
            flags: 0,
            boundary_commit: lag_commit,
        };
        let payload_commit = update_mode_commit(mode, lag_commit);
        self.append_archive_record(
            RecordKind::Other(UPDATE_MODE_RECORD_KIND),
            payload_commit,
            meta,
        );
    }

    fn rsa_apply_gate(
        &self,
        mode: UpdateMode,
        nsr_verdict: NsrVerdict,
        phi_proxy: u16,
        phi_high: u16,
        tcf_budget: u16,
    ) -> (bool, Digest32) {
        let allowed = mode != UpdateMode::Stabilize
            && nsr_verdict == NsrVerdict::Allow
            && phi_proxy >= phi_high
            && tcf_budget > 0;
        let commit = commit_rsa_apply_gate(mode, nsr_verdict, phi_proxy, phi_high, tcf_budget);
        (allowed, commit)
    }

    fn apply_ncde_attention_bias(&self, mut weights: AttentionWeights) -> AttentionWeights {
        let energy = self
            .last_ncde_output
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|output| output.ncde_energy));
        let Some(energy) = energy else {
            return weights;
        };
        let bias = (energy / 100).min(200);
        if bias > 0 {
            weights.gain = clamp_u16(u32::from(weights.gain) + u32::from(bias));
            weights.commit = commit_attention_override(&weights);
        }
        weights
    }

    fn coupling_influence(&self, signal: SignalId) -> i16 {
        self.last_coupling_outputs
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .as_ref()
                    .map(|outputs| coupling_influence_value(Some(outputs), signal))
            })
            .unwrap_or(0)
    }

    fn tick_coupling(&self, ctx: &mut StageContext, cycle_id: u64) {
        let phase_bus = ctx
            .phase_bus
            .unwrap_or_else(|| self.latest_phase_bus(cycle_id));
        let samples = self.collect_coupling_samples(ctx, cycle_id);
        if samples.is_empty() {
            return;
        }
        let inputs =
            CouplingInputs::new(cycle_id, phase_bus.commit, phase_bus.gamma_bucket, samples);
        let budget = self.current_gain_budget();
        let coupling_result = self.coupling_core.lock().ok().map(|mut core| {
            let outputs = core.tick(&inputs, &budget);
            let buffer_commits = core.buffer_commits();
            (outputs, buffer_commits)
        });
        if let Some((outputs, buffer_commits)) = coupling_result {
            ctx.coupling_outputs = Some(outputs.clone());
            if let Ok(mut guard) = self.last_coupling_outputs.lock() {
                *guard = Some(outputs.clone());
            }
            let top_influences = top_coupling_influences(&outputs.influences, 6);
            let lag_commits = buffer_commits
                .into_iter()
                .map(|(id, commit)| (id.as_u16(), commit))
                .collect::<Vec<_>>();
            if let Ok(mut workspace) = self.workspace.lock() {
                workspace.set_coupling_snapshot(
                    outputs.influences_root,
                    top_influences,
                    lag_commits,
                );
            }
            self.append_coupling_record(cycle_id, &outputs);
        }
    }

    fn tick_tcf_plan(
        &self,
        ctx: &StageContext,
        cycle_id: u64,
        phase_bus: &PhaseBus,
        iit_output: &IitOutput,
        policy_ok: bool,
        request_replay_override: bool,
    ) -> Option<TcfPlan> {
        let coupling_root = ctx
            .coupling_outputs
            .as_ref()
            .map(|outputs| outputs.influences_root)
            .or_else(|| {
                self.last_coupling_outputs
                    .lock()
                    .ok()
                    .and_then(|guard| guard.as_ref().map(|outputs| outputs.influences_root))
            })
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let nsr_verdict = ctx
            .nsr_output
            .as_ref()
            .map(|output| output.verdict.as_u8())
            .or_else(|| ctx.nsr_report.as_ref().map(|report| report.verdict.as_u8()))
            .unwrap_or(0);
        let flow_energy_hint = ctx
            .ncde_output
            .as_ref()
            .map(|output| output.flow_energy)
            .or_else(|| {
                self.last_ncde_output
                    .lock()
                    .ok()
                    .and_then(|g| g.map(|o| o.flow_energy))
            })
            .unwrap_or(0);
        let attention_gain =
            ctx.ssm_output
                .as_ref()
                .map(|output| output.ssm_attention_gain)
                .or_else(|| {
                    self.last_workspace_snapshot.lock().ok().and_then(|guard| {
                        guard.as_ref().map(|snapshot| snapshot.ssm_attention_gain)
                    })
                })
                .unwrap_or(0);
        let inputs = TcfInputs::new(
            cycle_id,
            phase_bus.commit,
            phase_bus.gamma_bucket,
            ctx.coherence_plv.unwrap_or(0),
            iit_output.phi_proxy,
            ctx.attention_risk,
            ctx.drift_score.unwrap_or(0),
            ctx.surprise_score.unwrap_or(0),
            attention_gain,
            flow_energy_hint,
            iit_output.hints_commit,
            iit_output.tighten_sync,
            iit_output.damp_output,
            iit_output.damp_learning,
            iit_output.request_replay || request_replay_override,
            coupling_root,
            nsr_verdict,
            policy_ok,
        );
        let budget = self.current_gain_budget();
        let mut modules = self.runtime_modules.lock().ok()?;
        Some(modules.tcf.tick_with_budget(&inputs, &budget))
    }

    fn collect_coupling_samples(&self, ctx: &StageContext, cycle_id: u64) -> Vec<SignalSample> {
        let mut samples = Vec::new();
        let push_i16 = |id: SignalId, value: i16, samples: &mut Vec<SignalSample>| {
            samples.push(SignalSample::new(cycle_id, id, value));
        };
        let push_u16 = |id: SignalId, value: u16, samples: &mut Vec<SignalSample>| {
            let value = value.min(i16::MAX as u16) as i16;
            samples.push(SignalSample::new(cycle_id, id, value));
        };

        if let Some(value) = ctx.percept_energy {
            push_u16(SignalId::PerceptEnergy, value, &mut samples);
        }
        let ssm_output = ctx.ssm_output.as_ref().cloned().or_else(|| {
            self.last_ssm_output
                .lock()
                .ok()
                .and_then(|guard| guard.clone())
        });
        if let Some(output) = ssm_output.as_ref() {
            push_u16(SignalId::SsmSalience, output.ssm_salience, &mut samples);
            push_u16(SignalId::SsmNovelty, output.ssm_novelty, &mut samples);
            push_u16(
                SignalId::SsmAttentionGain,
                output.ssm_attention_gain,
                &mut samples,
            );
        }
        let attention_weights = ctx
            .attention_weights
            .as_ref()
            .cloned()
            .or_else(|| self.last_attention.lock().ok().map(|attn| attn.clone()));
        if let Some(weights) = attention_weights.as_ref() {
            push_u16(SignalId::AttentionFinalGain, weights.gain, &mut samples);
        }
        let ncde_output = ctx
            .ncde_output
            .as_ref()
            .cloned()
            .or_else(|| self.last_ncde_output.lock().ok().and_then(|guard| *guard));
        if let Some(output) = ncde_output.as_ref() {
            push_u16(SignalId::NcdeEnergy, output.ncde_energy, &mut samples);
        }
        let phi = ctx.integration_score.or_else(|| {
            self.last_workspace_snapshot
                .lock()
                .ok()
                .and_then(|snapshot| {
                    snapshot
                        .as_ref()
                        .and_then(|snapshot| snapshot.iit_output.as_ref().map(|out| out.phi_proxy))
                })
        });
        if let Some(phi) = phi {
            push_u16(SignalId::PhiProxy, phi, &mut samples);
        }
        if let Some(plv) = ctx.coherence_plv {
            push_u16(SignalId::GlobalPlv, plv, &mut samples);
        }
        push_u16(SignalId::Risk, ctx.attention_risk, &mut samples);
        if let Some(drift) = ctx.drift_score {
            push_u16(SignalId::Drift, drift, &mut samples);
        }
        if let Some(surprise) = ctx.surprise_score {
            push_u16(SignalId::Surprise, surprise, &mut samples);
        }
        if let Some(replay_pressure) = ctx.replay_pressure {
            push_u16(SignalId::ReplayPressure, replay_pressure, &mut samples);
        }
        let sleep_drive = ctx
            .influence_outputs
            .as_ref()
            .map(|outputs| outputs.node_value(InfluenceNodeId::SleepDrive))
            .or_else(|| {
                self.last_influence_outputs.lock().ok().and_then(|guard| {
                    guard
                        .as_ref()
                        .map(|outputs| outputs.node_value(InfluenceNodeId::SleepDrive))
                })
            })
            .unwrap_or(0);
        push_i16(SignalId::SleepDrive, sleep_drive, &mut samples);
        let nsr_verdict = ctx
            .nsr_output
            .as_ref()
            .map(|output| i16::from(output.verdict.as_u8()))
            .or_else(|| {
                ctx.nsr_report
                    .as_ref()
                    .map(|report| i16::from(report.verdict.as_u8()))
            })
            .unwrap_or(0);
        push_i16(SignalId::NsrVerdict, nsr_verdict, &mut samples);

        let last_coupling = self
            .last_coupling_outputs
            .lock()
            .ok()
            .and_then(|guard| guard.clone());
        let learning_hint =
            coupling_influence_value(last_coupling.as_ref(), SignalId::LearningHint);
        let rsa_strength =
            coupling_influence_value(last_coupling.as_ref(), SignalId::RsaProposalStrength);
        push_i16(SignalId::LearningHint, learning_hint, &mut samples);
        push_i16(SignalId::RsaProposalStrength, rsa_strength, &mut samples);

        samples
    }

    fn append_coupling_record(&self, cycle_id: u64, outputs: &CouplingOutputs) {
        let checksum = coupling_checksum(&outputs.influences);
        let count = u16::try_from(outputs.influences.len()).unwrap_or(u16::MAX);
        let mut payload = Vec::with_capacity(Digest32::LEN * 2 + 2);
        payload.extend_from_slice(outputs.influences_root.as_bytes());
        payload.extend_from_slice(&count.to_be_bytes());
        payload.extend_from_slice(checksum.as_bytes());
        let record_id = format!(
            "coupling-{cycle_id}-{}",
            hex::encode(outputs.influences_root.as_bytes())
        );
        let record = build_compact_record(record_id, cycle_id, "coupling", payload);
        self.archive.append(record);
    }

    fn apply_ssm_attention_bias(
        &self,
        mut weights: AttentionWeights,
        ssm_attention_gain: Option<u16>,
    ) -> AttentionWeights {
        let Some(ssm_attention_gain) = ssm_attention_gain else {
            return weights;
        };
        if ssm_attention_gain == 0 {
            return weights;
        }
        let combined = weights.gain.max(ssm_attention_gain);
        if combined != weights.gain {
            weights.gain = combined;
            weights.commit = commit_attention_override(&weights);
        }
        weights
    }

    fn enforce_ssm_attention_dominance(
        &self,
        mut weights: AttentionWeights,
        ssm_attention_gain: Option<u16>,
    ) -> AttentionWeights {
        let Some(ssm_attention_gain) = ssm_attention_gain else {
            return weights;
        };
        if ssm_attention_gain == 0 {
            return weights;
        }
        let dominant = ssm_attention_gain.max(weights.gain / 4);
        if dominant != weights.gain {
            weights.gain = dominant;
            weights.commit = commit_attention_override(&weights);
        }
        weights
    }

    fn apply_iit_effects(&self, effects: IitActionEffects) {
        if let Ok(mut workspace) = self.workspace.lock() {
            workspace.set_broadcast_cap(effects.broadcast_cap);
        }
        if let Ok(mut output_router) = self.output_router.lock() {
            output_router.set_max_thought_frames_per_cycle(effects.max_thought_frames_per_cycle);
        }
    }

    fn update_predictive_coding(
        &self,
        observation: &Observation,
    ) -> Option<(PredictionError, SurpriseSignal)> {
        let mut guard = self.world_state.lock().ok()?;
        let previous = guard.clone();
        *guard = Some(observation.state.clone());
        drop(guard);
        let previous = previous?;
        let prediction = self.world_model.predict(&previous);
        let error = error(&prediction, observation);
        let surprise_signal = surprise(&error);
        Some((error, surprise_signal))
    }

    fn build_predictive_record(
        &self,
        cf: &ControlFrame,
        error: &PredictionError,
        surprise_signal: &SurpriseSignal,
    ) -> ExperienceRecord {
        let record_id = format!(
            "predictive-{}-{}",
            cf.frame_id,
            hex::encode(error.commit.as_bytes())
        );
        let payload = format!(
            "pred_error={};surprise={}",
            error.commit, surprise_signal.commit
        )
        .into_bytes();

        ExperienceRecord {
            record_id,
            observed_at_ms: cf.issued_at_ms,
            subject_id: cf.policy_id.clone(),
            payload,
            digest: None,
            vrf_tag: None,
            proof_ref: None,
        }
    }

    fn publish_workspace_signal(&self, signal: WorkspaceSignal) {
        if let Ok(mut workspace) = self.workspace.lock() {
            workspace.publish(signal);
        }
    }

    fn publish_workspace_signals<I>(&self, signals: I)
    where
        I: IntoIterator<Item = WorkspaceSignal>,
    {
        if let Ok(mut workspace) = self.workspace.lock() {
            for signal in signals {
                workspace.publish(signal);
            }
        }
    }

    fn translate_features(
        &self,
        activation_view: Option<&ActivationView>,
        attention: &AttentionWeights,
        cycle_id: u64,
        slot: u8,
    ) -> Option<LensSelection> {
        let activation_view = activation_view?;
        let set = self.feature_sae.encode(activation_view);
        let selection = self.feature_lens.select(&set, attention);
        let summary = format!(
            "FEATURES topk={} commit={}",
            selection.topk.len(),
            selection.commit
        );
        self.publish_workspace_signal(WorkspaceSignal {
            kind: SignalKind::Integration,
            priority: FEATURE_SIGNAL_PRIORITY,
            digest: selection.commit,
            summary,
            slot,
        });
        self.append_feature_translation_record(
            cycle_id,
            activation_view.commit,
            selection.commit,
            selection.topk.len(),
        );
        Some(selection)
    }

    fn append_feature_translation_record(
        &self,
        cycle_id: u64,
        activation_commit: Digest32,
        selection_commit: Digest32,
        topk: usize,
    ) {
        self.append_feature_translation_archive_record(
            cycle_id,
            activation_commit,
            selection_commit,
            topk,
        );
    }

    fn stimulate_bluebrain_port(
        &self,
        cf: &ControlFrameNormalized,
        workspace_snapshot: &WorkspaceSnapshot,
        attention: &AttentionWeights,
        surprise: Option<&SurpriseSignal>,
        lens_selection: Option<&LensSelection>,
        slot: u8,
    ) {
        let mut guard = match self.bluebrain_port.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        let Some(port) = guard.as_mut() else {
            return;
        };
        let stimulus = map_to_stimulus(cf, workspace_snapshot, attention, surprise, lens_selection);
        let response = port.stimulate(&stimulus);
        self.publish_workspace_signal(WorkspaceSignal::from_brain_stimulated(
            stimulus.commit,
            Some(slot),
        ));
        self.publish_workspace_signal(WorkspaceSignal::from_brain_responded(
            response.commit,
            response.arousal,
            response.valence,
            Some(slot),
        ));
        if let Ok(mut delta_guard) = self.pending_neuromod_delta.lock() {
            *delta_guard = Some(response.delta);
        }
        if let Ok(mut guard) = self.last_brain_response_commit.lock() {
            *guard = Some(response.commit);
        }
        if let Ok(mut guard) = self.last_brain_arousal.lock() {
            *guard = response.arousal;
        }
    }

    fn append_feature_translation_archive_record(
        &self,
        cycle_id: u64,
        activation_commit: Digest32,
        selection_commit: Digest32,
        topk: usize,
    ) {
        let payload_commit = feature_translation_commit(activation_commit, selection_commit);
        let meta = RecordMeta {
            cycle_id,
            tier: topk.min(u8::MAX as usize) as u8,
            flags: 0,
            boundary_commit: activation_commit,
        };
        self.append_archive_record(RecordKind::Other(FEATURE_RECORD_KIND), payload_commit, meta);
    }

    fn append_spike_bus_record(&self, cycle_id: u64, outputs: &SpikeOutputs) {
        let counts = summarize_spike_counts(&outputs.counts);
        let payload_commit = spike_record_commit(
            outputs.accepted_root,
            counts.total,
            counts.top_kinds.as_slice(),
            outputs.max_intensity,
        );
        let meta = RecordMeta {
            cycle_id,
            tier: counts.total.min(u8::MAX as u16) as u8,
            flags: outputs.max_intensity,
            boundary_commit: outputs.commit,
        };
        self.append_archive_record(RecordKind::Other(SPIKE_RECORD_KIND), payload_commit, meta);
    }

    fn append_ncde_output_record(&self, cycle_id: u64, output: &NcdeOutputs) {
        let payload = format!(
            "commit={};energy={};state_digest={};replay_hint={}",
            output.commit,
            output.ncde_energy,
            output.ncde_state_digest,
            output.replay_pressure_hint
        )
        .into_bytes();
        let record_id = format!("ncde-{cycle_id}-{}", hex::encode(output.commit.as_bytes()));
        let record = build_compact_record(record_id, cycle_id, "ncde", payload);
        self.archive.append(record);

        let meta = RecordMeta {
            cycle_id,
            tier: 0,
            flags: 0,
            boundary_commit: output.commit,
        };
        self.append_archive_record(RecordKind::Other(NCDE_RECORD_KIND), output.commit, meta);
    }

    fn append_cde_output_record(
        &self,
        cycle_id: u64,
        output: &CdeV1Outputs,
        graph_commit: Digest32,
        observation_commit: Digest32,
        last_query_result: Option<Digest32>,
    ) {
        let top_edge_count = output.top_edges.len().min(u16::MAX as usize) as u16;
        let intervention_flag = output.intervention.is_some();
        let last_query_flag = last_query_result.is_some();
        let last_query_commit = last_query_result.unwrap_or_else(|| Digest32::new([0u8; 32]));
        let payload = format!(
            "commit={};dag={};graph={};summary={};observation={};top_edges={top_edge_count};intervention={intervention_flag};last_query={last_query_flag};last_query_commit={}",
            output.commit,
            output.dag_commit,
            graph_commit,
            output.summary_commit,
            observation_commit,
            last_query_commit
        )
        .into_bytes();
        let record_id = format!(
            "cde-output-{cycle_id}-{}",
            hex::encode(output.commit.as_bytes())
        );
        let record = build_compact_record(record_id, cycle_id, "cde", payload);
        self.archive.append(record);

        let meta = RecordMeta {
            cycle_id,
            tier: top_edge_count.min(u8::MAX as u16) as u8,
            flags: u16::from(intervention_flag),
            boundary_commit: output.commit,
        };
        self.append_archive_record(
            RecordKind::Other(CDE_OUTPUT_RECORD_KIND),
            output.commit,
            meta,
        );
    }

    fn append_ssm_output_record(&self, cycle_id: u64, output: &SsmOutputs) {
        let payload = format!(
            "commit={};state_commit={};salience={};novelty={};attn_gain={}",
            output.commit,
            output.ssm_state_commit,
            output.ssm_salience,
            output.ssm_novelty,
            output.ssm_attention_gain
        )
        .into_bytes();
        let record_id = format!("ssm-{cycle_id}-{}", hex::encode(output.commit.as_bytes()));
        let record = build_compact_record(record_id, cycle_id, "ssm", payload);
        self.archive.append(record);

        let meta = RecordMeta {
            cycle_id,
            tier: 0,
            flags: output.ssm_salience,
            boundary_commit: output.ssm_state_commit,
        };
        self.append_archive_record(RecordKind::Other(SSM_RECORD_KIND), output.commit, meta);
    }

    fn append_jepa_output_record(&self, cycle_id: u64, output: &JepaOutputs) {
        let payload = format!(
            "commit={};world_state={};prediction={};surprise={}",
            output.commit, output.world_state, output.prediction, output.surprise
        )
        .into_bytes();
        let record_id = format!("jepa-{cycle_id}-{}", hex::encode(output.commit.as_bytes()));
        let record = build_compact_record(record_id, cycle_id, "jepa", payload);
        self.archive.append(record);

        let meta = RecordMeta {
            cycle_id,
            tier: 0,
            flags: output.surprise,
            boundary_commit: output.world_state,
        };
        self.append_archive_record(RecordKind::Other(JEPA_RECORD_KIND), output.commit, meta);
    }

    #[allow(clippy::too_many_arguments)]
    fn tick_cde_v1(
        &self,
        cycle_id: u64,
        phase_bus: &PhaseBus,
        spike_accepted_root: Digest32,
        observation_commit: Digest32,
        observation_key: ObservationKey,
        ssm_output: Option<&SsmOutputs>,
        ncde_output: Option<&NcdeOutputs>,
        iit_output: &IitOutput,
        influence_outputs: Option<&InfluenceOutputs>,
        replay_pressure: u16,
        drift: u16,
        surprise: u16,
        risk: u16,
        tcf_plan: Option<TcfPlan>,
        learning_gain_cap: u16,
        spike_counts: &[(SpikeKind, u16)],
        policy_restrict: bool,
    ) -> Option<(CdeV1Outputs, Digest32)> {
        let attention_gain = self
            .last_attention
            .lock()
            .map(|attn| attn.gain)
            .unwrap_or(0);
        let learning_rate = learning_gain_cap.min(10_000);
        let sleep_drive_raw = influence_outputs
            .map(|outputs| outputs.node_value(InfluenceNodeId::SleepDrive))
            .unwrap_or(0);
        let sleep_drive = sleep_drive_raw.clamp(0, 10_000) as u16;
        let ncde_energy = ncde_output.map(|output| output.ncde_energy).unwrap_or(0);
        let params = self
            .structural_store
            .lock()
            .map(|store| store.current.clone())
            .unwrap_or_else(|_| StructuralStore::default().current);
        let (sleep_active, replay_active) = tcf_plan
            .map(|plan| (plan.sleep_active, plan.replay_active))
            .or_else(|| self.tcf_sleep_replay(None))
            .unwrap_or_else(|| {
                let sleep_active = derive_sleep_active(
                    false,
                    replay_pressure,
                    iit_output.phi_proxy,
                    sleep_drive_raw,
                    ncde_energy,
                    &params,
                );
                let replay_active = replay_pressure >= 5_000;
                (sleep_active, replay_active)
            });
        let sle_cde_bias = self
            .last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|snapshot| snapshot.as_ref().map(|snapshot| snapshot.sle_cde_bias))
            .unwrap_or(0);
        let inputs = CdeV1Inputs::new(
            cycle_id,
            phase_bus.commit,
            phase_bus.gamma_bucket,
            ssm_output.map(|output| output.ssm_salience).unwrap_or(0),
            ssm_output.map(|output| output.ssm_novelty).unwrap_or(0),
            sle_cde_bias,
            attention_gain,
            learning_rate,
            replay_pressure,
            sleep_drive,
            ncde_energy,
            phase_bus.global_plv,
            iit_output.phi_proxy,
            risk,
            drift,
            surprise,
            sleep_active,
            replay_active,
            spike_accepted_root,
            observation_commit,
        );
        let mut modules = self.runtime_modules.lock().ok()?;
        modules.cde.register_observation(observation_key);
        let surprise_high = surprise >= CDE_SURPRISE_HIGH_THRESHOLD;
        let threat_spike = spike_counts
            .iter()
            .any(|(kind, count)| *kind == SpikeKind::Threat && *count > 0);
        if surprise_high && threat_spike {
            modules.cde.propose_edge(CdeGraphEdge::new(
                CdeVarId::WORLD_STATE,
                CdeVarId::SPIKE_ROOT,
                CDE_EDGE_WEIGHT_POSITIVE,
            ));
        }
        if policy_restrict {
            modules.cde.propose_edge(CdeGraphEdge::new(
                CdeVarId::NSR_TRACE_ROOT,
                CdeVarId::TCF_ATTENTION_CAP,
                CDE_EDGE_WEIGHT_NEGATIVE,
            ));
        }
        let outputs = modules.cde.tick(&inputs);
        let graph_commit = modules.cde.graph_commit();
        Some((outputs, graph_commit))
    }

    #[allow(clippy::too_many_arguments)]
    fn tick_cde(
        &self,
        cycle_id: u64,
        phase_bus: &PhaseBus,
        spike_root_commit: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        world_state_commit: Digest32,
        influence_outputs: Option<&InfluenceOutputs>,
        iit_output: &IitOutput,
        ssm_output: Option<&SsmOutputs>,
        ncde_output: Option<&NcdeOutputs>,
        drift: u16,
        surprise: u16,
        risk: u16,
    ) -> Option<CdeOutputs> {
        let influence_commit = influence_outputs
            .map(|outputs| outputs.commit)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let influence_node_in = influence_outputs
            .map(|outputs| outputs.node_values.clone())
            .unwrap_or_default();
        let global_phase = u16::from(phase_bus.gamma_bucket) * 256;
        let inputs = CdeInputs::new(
            cycle_id,
            phase_bus.commit,
            phase_bus.gamma_bucket,
            global_phase,
            phase_bus.global_plv,
            iit_output.phi_proxy,
            spike_root_commit,
            spike_counts,
            world_state_commit,
            influence_commit,
            influence_node_in,
            ssm_output.map(|output| output.ssm_salience).unwrap_or(0),
            ncde_output.map(|output| output.ncde_energy).unwrap_or(0),
            drift,
            surprise,
            risk,
        );
        let mut engine = self.cde_engine.lock().ok()?;
        Some(engine.tick(&inputs))
    }

    #[allow(clippy::too_many_arguments)]
    fn tick_ncde(
        &self,
        cycle_id: u64,
        phase_bus: &PhaseBus,
        attention_gain: u16,
        coupling_outputs: Option<&CouplingOutputs>,
        ssm_output: Option<&SsmOutputs>,
        spike_root_commit: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        risk: u16,
        drift: u16,
        surprise: u16,
        learning_gain_cap: u16,
    ) -> Option<NcdeOutputs> {
        self.sync_ncde_params();
        let mut modules = self.runtime_modules.lock().ok()?;
        let _ = coupling_outputs;
        let _ = ssm_output;
        let _ = risk;
        let _ = drift;
        let spike_counts = ucf_ncde::SpikeCountsSummary::from_counts(&spike_counts);
        let budget_commit = self.current_gain_budget().commit;
        let inputs = NcdeInputs::new(
            cycle_id,
            phase_bus.commit,
            phase_bus.gamma_bucket,
            phase_bus.global_plv,
            spike_root_commit,
            spike_counts,
            attention_gain.min(learning_gain_cap),
            surprise,
            budget_commit,
        );
        let budget = self.current_gain_budget();
        Some(modules.ncde.tick_with_budget(&inputs, &budget))
    }

    #[allow(clippy::too_many_arguments)]
    fn tick_ssm(
        &self,
        phase_bus: &PhaseBus,
        percept_commit: Digest32,
        percept_energy: u16,
        ncde_energy: u16,
        spike_root_commit: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        b_q15_bias: i16,
        drift: u16,
        surprise: u16,
        risk: u16,
    ) -> Option<SsmOutputs> {
        self.sync_ssm_params();
        let mut modules = self.runtime_modules.lock().ok()?;
        let coupling_outputs = self
            .last_coupling_outputs
            .lock()
            .ok()
            .and_then(|guard| guard.clone());
        let (coupling_root, coupling_influences) = coupling_outputs
            .map(|outputs| (outputs.influences_root, outputs.influences.clone()))
            .unwrap_or_else(|| (Digest32::new([0u8; 32]), Vec::new()));
        let workspace_snapshot = self
            .last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|snapshot| snapshot.as_ref().cloned());
        let spike_max_intensity = workspace_snapshot
            .as_ref()
            .map(|snapshot| snapshot.spike_max_intensity)
            .unwrap_or(0);
        let jepa_surprise = workspace_snapshot
            .as_ref()
            .map(|snapshot| snapshot.jepa_surprise)
            .unwrap_or(0);
        let nsr_trace_root = workspace_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.nsr_trace_root)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let sle_bias = workspace_snapshot
            .as_ref()
            .map(|snapshot| snapshot.sle_ssm_bias)
            .unwrap_or(0);
        let tcf_attention_cap = self
            .last_tcf_plan
            .lock()
            .ok()
            .and_then(|plan| plan.as_ref().map(|plan| plan.attention_gain_cap))
            .unwrap_or(10_000);
        let tcf_learning_cap = self
            .last_tcf_plan
            .lock()
            .ok()
            .and_then(|plan| plan.as_ref().map(|plan| plan.learning_gain_cap))
            .unwrap_or(10_000);
        let gain_budget_commit = self
            .last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|snapshot| {
                snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.gain_budget_commit)
            })
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let input = SsmInputs::new(
            phase_bus.cycle_id,
            phase_bus.commit,
            phase_bus.gamma_bucket,
            percept_commit,
            percept_energy,
            spike_root_commit,
            spike_max_intensity,
            spike_counts,
            coupling_root,
            coupling_influences,
            tcf_attention_cap,
            tcf_learning_cap,
            gain_budget_commit,
            b_q15_bias,
            sle_bias,
            ncde_energy,
            risk,
            drift,
            surprise,
            jepa_surprise,
            nsr_trace_root,
        );
        let budget = self.current_gain_budget();
        Some(modules.ssm.tick_with_budget(&input, &budget))
    }

    fn append_sandbox_denied_record(&self, cycle_id: u64, report: &SandboxReport, reason: &str) {
        let payload_commit = sandbox_denied_commit(reason, report.commit);
        let meta = RecordMeta {
            cycle_id,
            tier: 0,
            flags: 0,
            boundary_commit: report.commit,
        };
        self.append_archive_record(
            RecordKind::Other(SANDBOX_DENIED_RECORD_KIND),
            payload_commit,
            meta,
        );
    }

    fn latest_phase_bus(&self, cycle_id: u64) -> PhaseBus {
        self.last_phase_bus
            .lock()
            .ok()
            .and_then(|guard| *guard)
            .unwrap_or(PhaseBus {
                cycle_id,
                gamma_bucket: 0,
                global_plv: 0,
                osc_buckets: [0u8; 16],
                phase_commit: Digest32::new([0u8; 32]),
                commit: Digest32::new([0u8; 32]),
            })
    }

    fn sync_onn_params(&self) {
        let knobs = match self.structural_store.lock() {
            Ok(store) => store.current.onn.clone(),
            Err(_) => return,
        };
        if let Ok(mut modules) = self.runtime_modules.lock() {
            let params = modules.phase.params();
            if params.commit != knobs.commit {
                let updated = ucf_onn::OnnParams::new(
                    params.n,
                    params.omega_q12,
                    knobs.k_couple,
                    knobs.k_dither,
                    params.buckets,
                    knobs.couple_clamp_q12,
                );
                modules.phase.set_params(updated);
            }
        }
    }

    fn sync_ssm_params(&self) {
        let params = match self.structural_store.lock() {
            Ok(store) => store.current.ssm,
            Err(_) => return,
        };
        if let Ok(mut modules) = self.runtime_modules.lock() {
            let current = modules.ssm.params();
            if current.commit != params.commit {
                modules.ssm.set_params(params);
                modules.ssm.reset_if_dim_mismatch(&params);
            }
        }
    }

    fn sync_ncde_params(&self) {
        let params = match self.structural_store.lock() {
            Ok(store) => store.current.ncde,
            Err(_) => return,
        };
        if let Ok(mut modules) = self.runtime_modules.lock() {
            let current = modules.ncde.params();
            if current.commit != params.commit {
                modules.ncde.set_params(params);
            }
        }
    }

    fn current_snn_knobs(&self) -> SnnKnobs {
        self.structural_store
            .lock()
            .map(|store| store.current.snn.clone())
            .unwrap_or_else(|_| SnnKnobs::default())
    }

    fn update_nsr_warn_streak(&self, verdict: NsrVerdict) -> u16 {
        let mut streak = self
            .nsr_warn_streak
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        if verdict == NsrVerdict::Restrict {
            *streak = streak.saturating_add(1);
        } else {
            *streak = 0;
        }
        *streak
    }

    fn arbitrate_workspace(&self, cycle_id: u64) -> WorkspaceSnapshot {
        let mut workspace = self.workspace.lock().expect("workspace lock");
        workspace.arbitrate(cycle_id)
    }

    fn emit_stage_trace(&self, stage: PulseKind) {
        if let Some(trace) = self.stage_trace.as_ref() {
            trace.record(stage);
        }
    }

    fn take_sandbox_inference(&self) -> Option<AiInference> {
        self.sandbox_inference_cache
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    fn build_intent_summary(&self) -> IntentSummary {
        let attention = self
            .last_attention
            .lock()
            .map(|attn| attn.clone())
            .unwrap_or_else(|_| idle_attention());
        let stability = self
            .last_self_state
            .lock()
            .ok()
            .and_then(|state| *state)
            .map(|state| state.stability_score())
            .unwrap_or(0);
        let drift = self
            .last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|snapshot| snapshot.as_ref().map(drift_score_from_snapshot))
            .unwrap_or(0);
        let intent = focus_channel_score(attention.channel);
        let risk = drift.max(10_000u16.saturating_sub(stability));
        IntentSummary::new(intent, risk)
    }

    fn build_nsr_input(
        &self,
        cycle_id: u64,
        policy_class: u16,
        outputs: &[AiOutput],
        workspace_snapshot: &WorkspaceSnapshot,
        intent: IntentSummary,
        causal_report: Option<&CausalReport>,
    ) -> NsrInput {
        let proposed_actions = action_intents_from_outputs(outputs);
        let causal_report_commit = causal_report
            .map(|report| report.commit)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let counterfactuals = causal_report
            .map(|report| report.counterfactuals.clone())
            .unwrap_or_default();
        let thresholds = self
            .structural_store
            .lock()
            .map(|store| store.current.nsr.clone())
            .ok();
        let input = NsrInput::new(
            cycle_id,
            intent,
            policy_class,
            proposed_actions,
            workspace_snapshot.commit,
            causal_report_commit,
            counterfactuals,
        );
        if let Some(thresholds) = thresholds {
            input.with_nsr_thresholds(thresholds.warn, thresholds.deny)
        } else {
            input
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_nsr_inputs(
        &self,
        cycle_id: u64,
        phase_bus: &PhaseBus,
        influence_outputs: Option<&InfluenceOutputs>,
        cde_v1_output: Option<&CdeV1Outputs>,
        cde_output: Option<&CdeOutputs>,
        ssm_output: Option<&SsmOutputs>,
        ncde_output: Option<&NcdeOutputs>,
        iit_output: &IitOutput,
        self_state: Option<&SelfState>,
        spike_counts: Vec<(SpikeKind, u16)>,
        spike_max_intensity: u16,
        replay_pressure: u16,
        drift_score: u16,
        surprise_score: u16,
        jepa_surprise: u16,
        risk_score: u16,
        _output_intent: bool,
        thought_only: bool,
        tool_req: bool,
        policy_commit: Digest32,
        _policy_ok: bool,
        sleep_pulse: bool,
        tcf_plan: Option<TcfPlan>,
    ) -> NsrInputs {
        let sleep_drive = influence_outputs
            .map(|outputs| outputs.node_value(InfluenceNodeId::SleepDrive))
            .unwrap_or(0);
        let params = self
            .structural_store
            .lock()
            .map(|store| store.current.clone())
            .unwrap_or_else(|_| StructuralStore::default().current);
        let phi_proxy = iit_output.phi_proxy;
        let ncde_energy = ncde_output.map(|output| output.ncde_energy).unwrap_or(0);
        let (sleep_active, replay_active) = tcf_plan
            .map(|plan| (plan.sleep_active, plan.replay_active))
            .or_else(|| self.tcf_sleep_replay(None))
            .unwrap_or_else(|| {
                let sleep_active = derive_sleep_active(
                    sleep_pulse,
                    replay_pressure,
                    phi_proxy,
                    sleep_drive,
                    ncde_energy,
                    &params,
                );
                let replay_active = replay_pressure >= 5_000;
                (sleep_active, replay_active)
            });
        let geist_consistency =
            self_state.map(|state| state.consistency < SELF_CONSISTENCY_OK_THRESHOLD);
        let mut facts = vec![
            Fact::Phi(iit_output.phi_proxy),
            Fact::Plv(phase_bus.global_plv),
            Fact::Drift(drift_score),
            Fact::Surprise(surprise_score),
            Fact::Risk(risk_score),
            Fact::OnnPhase {
                gamma_bucket: phase_bus.gamma_bucket,
            },
        ];
        let lock_window_buckets = tcf_plan
            .as_ref()
            .map(|plan| plan.lock_window_buckets)
            .unwrap_or(1);
        facts.push(Fact::OnnLocked {
            global_plv: phase_bus.global_plv,
            lock_window_buckets,
        });
        let mut total_spikes = 0u16;
        let mut threat_spikes = 0u16;
        let mut thought_only_spikes = 0u16;
        for (kind, count) in &spike_counts {
            total_spikes = total_spikes.saturating_add(*count);
            if *kind == SpikeKind::Threat {
                threat_spikes = threat_spikes.saturating_add(*count);
            }
            if *kind == SpikeKind::ThoughtOnly {
                thought_only_spikes = thought_only_spikes.saturating_add(*count);
            }
        }
        facts.push(Fact::SpikeSummary {
            total: total_spikes,
            threat: threat_spikes,
            thought_only: thought_only_spikes,
        });
        facts.push(Fact::SpikeMaxIntensity(spike_max_intensity));
        facts.push(Fact::JepaSurprise(jepa_surprise));
        let phase_locked = phase_bus.global_plv >= 7_000;
        let high_surprise = surprise_score >= 7_000;
        let spike_threat_present = threat_spikes > 0;
        let thought_only_present = thought_only_spikes > 0;
        if phase_locked {
            facts.push(Fact::PhaseLocked);
        }
        if high_surprise {
            facts.push(Fact::HighSurprise);
        }
        if spike_threat_present {
            facts.push(Fact::SpikeThreatPresent);
        }
        if thought_only_present {
            facts.push(Fact::ThoughtOnlyPresent);
        }
        if sleep_active {
            facts.push(Fact::TcfSleepActive);
        }
        if replay_active {
            facts.push(Fact::TcfReplayActive);
        }
        if let Some(output) = cde_output {
            if !output.counterfactual_delta.is_empty() {
                facts.push(Fact::CdeCounterfactualOk {
                    commit: output.commit,
                });
            }
        }
        if let Some(output) = ssm_output {
            facts.push(Fact::SsmNovelty(output.ssm_novelty));
            facts.push(Fact::SsmSalience(output.ssm_salience));
        }
        if let Some(output) = ncde_output {
            facts.push(Fact::NcdeEnergy(output.ncde_energy));
        }
        let _ = geist_consistency;
        facts.push(Fact::IitHints {
            tighten_sync: iit_output.tighten_sync,
            damp_output: iit_output.damp_output,
            damp_learning: iit_output.damp_learning,
            request_replay: iit_output.request_replay,
        });
        facts.push(Fact::PolicyCommit {
            commit: policy_commit,
        });
        if tool_req {
            facts.push(Fact::ToolCallRequested);
        }
        if thought_only {
            facts.push(Fact::ThoughtOnlyRequested);
        }
        if let Some(output) = cde_v1_output {
            let mut edges = output.top_edges.clone();
            edges.sort_by(|left, right| left.commit.as_bytes().cmp(right.commit.as_bytes()));
            for edge in edges {
                let score = edge.score.unsigned_abs();
                facts.push(Fact::CdeEdge {
                    edge_commit: edge.commit,
                    score,
                });
            }
        }
        NsrInputs::new(cycle_id, phase_bus.commit, policy_commit, facts)
    }

    fn build_sandbox_budget(&self, mode: u16) -> SandboxBudget {
        let attention = self
            .last_attention
            .lock()
            .map(|attn| attn.clone())
            .unwrap_or_else(|_| idle_attention());
        let recursion = self
            .last_recursion_budget
            .lock()
            .ok()
            .and_then(|budget| *budget);
        let stability = self
            .last_self_state
            .lock()
            .ok()
            .and_then(|state| *state)
            .map(|state| state.stability_score())
            .unwrap_or(0);
        let drift = self
            .last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|snapshot| snapshot.as_ref().map(drift_score_from_snapshot))
            .unwrap_or(0);

        let base_steps = recursion.map(|budget| budget.per_cycle_steps).unwrap_or(24);
        let base_depth = recursion.map(|budget| budget.max_depth).unwrap_or(2);
        let mut ops = u64::from(base_steps).saturating_mul(300);
        ops = ops.saturating_add(u64::from(attention.gain));
        let mut max_frames = u16::from(base_depth).saturating_add(2);
        let mut max_output_chars = 240usize.saturating_add(attention.gain as usize / 6);

        if drift >= 7000 {
            ops = ops.saturating_sub(600);
            max_frames = max_frames.saturating_sub(1);
        }
        if drift >= 8500 {
            ops = ops.saturating_sub(600);
            max_frames = max_frames.saturating_sub(1);
        }
        if stability >= 8000 {
            ops = ops.saturating_add(700);
            max_frames = max_frames.saturating_add(1);
        }
        if matches!(attention.channel, FocusChannel::Threat) {
            ops = ops.saturating_sub(600);
            max_frames = max_frames.saturating_sub(1);
        }
        if self
            .last_nsr_report
            .lock()
            .ok()
            .and_then(|report| report.as_ref().map(|report| report.verdict))
            == Some(NsrVerdict::Deny)
        {
            ops = ops.saturating_mul(60) / 100;
            max_frames = max_frames.saturating_sub(1);
            max_output_chars = max_output_chars.saturating_sub(120);
        }

        if mode == ucf_sandbox::AI_MODE_THOUGHT {
            ops = ops.saturating_mul(70) / 100;
            max_frames = max_frames.saturating_sub(1);
            max_output_chars = max_output_chars.saturating_sub(80);
        }

        ops = ops.clamp(200, 12_000);
        max_frames = max_frames.clamp(1, 20);
        max_output_chars = max_output_chars.clamp(80, 2000);

        SandboxBudget {
            ops,
            max_output_chars,
            max_frames,
        }
    }

    fn append_nsr_report_record(&self, cycle_id: u64, report: &NsrReport) {
        let codes = report
            .violations
            .iter()
            .map(|violation| violation.code.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let causal_codes = report
            .violations
            .iter()
            .filter(|violation| is_causal_violation_code(&violation.code))
            .map(|violation| violation.code.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let payload = format!(
            "commit={};verdict={};causal_report={};codes={};causal_codes={}",
            report.commit,
            nsr_verdict_token(report.verdict),
            report.causal_report_commit,
            codes,
            causal_codes
        )
        .into_bytes();
        let record_id = format!(
            "nsr-report-{cycle_id}-{}",
            hex::encode(report.commit.as_bytes())
        );
        let record = build_compact_record(record_id, cycle_id, "nsr", payload);
        self.archive.append(record);
    }

    fn append_nsr_output_record(&self, cycle_id: u64, output: &NsrOutputs) {
        let payload = format!(
            "commit={};verdict={};trace_root={}",
            output.commit,
            nsr_verdict_token(output.verdict),
            output.trace_root
        )
        .into_bytes();
        let record_id = format!(
            "nsr-v1-{cycle_id}-{}",
            hex::encode(output.commit.as_bytes())
        );
        let record = build_compact_record(record_id, cycle_id, "nsr-v1", payload);
        self.archive.append(record);
    }

    fn append_causal_report_record(&self, cycle_id: u64, report: &CausalReport) {
        let payload = format!(
            "commit={};dag={};cf={};flags={}",
            report.commit,
            report.dag_commit,
            report.counterfactuals.len(),
            report.flags
        )
        .into_bytes();
        let record_id = format!(
            "causal-report-{cycle_id}-{}",
            hex::encode(report.commit.as_bytes())
        );
        let record = build_compact_record(record_id, cycle_id, "causal", payload);
        self.archive.append(record);
        let meta = RecordMeta {
            cycle_id,
            tier: report.counterfactuals.len().min(u8::MAX as usize) as u8,
            flags: report.flags,
            boundary_commit: report.commit,
        };
        self.append_archive_record(
            RecordKind::Other(CAUSAL_REPORT_RECORD_KIND),
            report.commit,
            meta,
        );
    }

    fn append_influence_record(
        &self,
        cycle_id: u64,
        graph_commit: Digest32,
        pulses_root: Digest32,
        outputs_commit: Digest32,
        pulse_count: usize,
        node_count: usize,
    ) {
        let payload = format!(
            "graph={};pulses={};outputs={};pulse_count={pulse_count};nodes={node_count}",
            graph_commit, pulses_root, outputs_commit
        )
        .into_bytes();
        let record_id = format!(
            "influence-{cycle_id}-{}",
            hex::encode(outputs_commit.as_bytes())
        );
        let record = build_compact_record(record_id, cycle_id, "influence", payload);
        self.archive.append(record);
    }

    fn append_nsr_audit_notice(&self, cycle_id: u64, report: &NsrReport) {
        let notice = ucf::boundary::v1::AuditNoticeV1 {
            event_kind: 1,
            evidence_digest: ucf::boundary::Digest32::new(*report.commit.as_bytes()),
            reason_code: 1,
        };
        let digest = notice.digest();
        let record_id = format!("audit-notice-{cycle_id}-{}", hex::encode(digest.as_bytes()));
        let payload = digest.as_bytes().to_vec();
        let record = build_compact_record(record_id, cycle_id, "audit", payload);
        self.archive.append(record);
    }

    fn append_causal_audit_notice(&self, cycle_id: u64, report: &NsrReport) {
        let notice = ucf::boundary::v1::AuditNoticeV1 {
            event_kind: 2,
            evidence_digest: ucf::boundary::Digest32::new(*report.causal_report_commit.as_bytes()),
            reason_code: 2,
        };
        let digest = notice.digest();
        let record_id = format!("audit-notice-{cycle_id}-{}", hex::encode(digest.as_bytes()));
        let payload = digest.as_bytes().to_vec();
        let record = build_compact_record(record_id, cycle_id, "audit", payload);
        self.archive.append(record);
    }

    fn handle_sandbox_denied(
        &self,
        ctx: &mut StageContext,
        report: &SandboxReport,
        cf: &ControlFrameNormalized,
        cycle_id: u64,
        slot: u8,
    ) {
        let reason = match &report.verdict {
            SandboxVerdict::Allow => "ALLOW",
            SandboxVerdict::Deny { reason } => reason.as_str(),
        };
        let summary = format!("SANDBOX=DENY {}", reason);
        self.publish_workspace_signal(WorkspaceSignal {
            kind: SignalKind::Risk,
            priority: 9500,
            digest: report.commit,
            summary,
            slot,
        });
        self.append_sandbox_denied_record(cycle_id, report, reason);
        ctx.sandbox_verdict = Some(report.verdict.clone());
        let inference = AiInference::new(Vec::new());
        let tom_report = self.tom_port.analyze(cf, &inference.outputs);
        ctx.inference = Some(inference);
        ctx.tom_report = Some(tom_report);
    }

    fn run_think_stage(
        &self,
        cf: &ControlFrameNormalized,
        ctx: &mut StageContext,
        cycle_id: u64,
        slot: u8,
        mode: u16,
    ) {
        if ctx.inference.is_some() {
            return;
        }
        if let Ok(mut guard) = self.sandbox_inference_cache.lock() {
            *guard = None;
        }
        let intent = self.build_intent_summary();
        let budget = self.build_sandbox_budget(mode);
        let request = AiCallRequest::new(cycle_id, cf.commitment().digest, mode, budget);
        let mut sandbox = self.sandbox_port.lock().expect("sandbox lock");
        let report = sandbox.evaluate_call(cf, &intent, &request);
        ctx.sandbox_report = Some(report.clone());
        if !report.verdict.is_allow() {
            self.handle_sandbox_denied(ctx, &report, cf, cycle_id, slot);
            return;
        }
        match sandbox.run_ai(&request) {
            Ok(call_result) => {
                let inference = self
                    .take_sandbox_inference()
                    .unwrap_or_else(|| AiInference::new(call_result.outputs.clone()));
                let tom_report = self.tom_port.analyze(cf, &inference.outputs);
                ctx.inference = Some(inference);
                ctx.tom_report = Some(tom_report);
                ctx.sandbox_verdict = Some(SandboxVerdict::Allow);
            }
            Err(report) => {
                ctx.sandbox_report = Some(report.clone());
                self.handle_sandbox_denied(ctx, &report, cf, cycle_id, slot);
            }
        }
    }
}

fn tom_summary(report: &ucf_tom_port::TomReport) -> String {
    let intent = match report.intent.intent {
        IntentType::AskInfo => "ask_info",
        IntentType::Negotiate => "negotiate",
        IntentType::RequestAction => "request_action",
        IntentType::SocialBond => "social_bond",
        IntentType::Unknown => "unknown",
    };
    let bucket = risk_bucket(report.risk.overall);
    format!("intent={intent},overall={bucket}")
}

fn observation_from_frame(cf: &ControlFrameNormalized) -> Observation {
    let digest = cf.commitment().digest;
    let mut data = Vec::with_capacity(Digest32::LEN / 2);
    for chunk in digest.as_bytes().chunks_exact(2) {
        let pair = [chunk[0], chunk[1]];
        data.push(i16::from_be_bytes(pair));
    }
    let dims = u16::try_from(data.len()).unwrap_or(0);
    Observation::new(WorldStateVec::new(dims, data))
}

fn observation_from_ssm_state(state: &SsmState) -> Observation {
    let mut data = Vec::with_capacity(state.s.len());
    for value in &state.s {
        data.push(clamp_i16(i64::from(*value)));
    }
    let dims = u16::try_from(data.len()).unwrap_or(0);
    Observation::new(WorldStateVec::new(dims, data))
}

fn clamp_i16(value: i64) -> i16 {
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

fn clamp_u16(value: u32) -> u16 {
    value.min(10_000) as u16
}

fn risk_bucket(overall: u16) -> &'static str {
    match overall {
        0..=3333 => "low",
        3334..=6666 => "med",
        _ => "high",
    }
}

fn intent_type_code(intent: IntentType) -> u16 {
    match intent {
        IntentType::AskInfo => AttnController::INTENT_ASK_INFO,
        IntentType::Negotiate => AttnController::INTENT_NEGOTIATE,
        IntentType::RequestAction => AttnController::INTENT_REQUEST_ACTION,
        IntentType::SocialBond => AttnController::INTENT_SOCIAL_BOND,
        IntentType::Unknown => AttnController::INTENT_UNKNOWN,
    }
}

fn apply_integration_bias(score: u16, bias: i16) -> u16 {
    if bias < 0 {
        score.saturating_sub(bias.unsigned_abs())
    } else {
        score.saturating_add(bias as u16).min(10_000)
    }
}

fn boundary_workspace_broadcast(snapshot: &WorkspaceSnapshot) -> Digest32 {
    let top_signals = snapshot
        .broadcast
        .iter()
        .map(|signal| WorkspaceSignalV1 {
            kind: signal.kind as u16,
            digest: boundary_digest32(&signal.digest),
            priority: signal.priority,
        })
        .collect();
    let message = WorkspaceBroadcastV1 {
        snapshot_commit: boundary_digest32(&snapshot.commit),
        top_signals,
    };
    boundary_to_types(message.digest())
}

fn boundary_digest32(digest: &Digest32) -> boundary::Digest32 {
    boundary::Digest32::new(*digest.as_bytes())
}

fn boundary_to_types(digest: boundary::Digest32) -> Digest32 {
    Digest32::new(*digest.as_bytes())
}

fn compress_cde_v1_edges(edges: &[CdeV1Edge]) -> Vec<(u16, u16, u16, u8)> {
    edges
        .iter()
        .take(8)
        .map(|edge| {
            let score = i32::from(edge.score).abs().min(i32::from(u16::MAX)) as u16;
            (edge.from.to_u16(), edge.to.to_u16(), score, edge.lag)
        })
        .collect()
}

fn collect_cde_v1_edge_commits(edges: &[CdeV1Edge]) -> Vec<Digest32> {
    edges.iter().take(8).map(|edge| edge.commit).collect()
}

fn cde_report_from_outputs(output: &CdeOutputs) -> CausalReport {
    let counterfactuals = output
        .counterfactual_delta
        .iter()
        .map(|(node, delta)| {
            let confidence = output
                .top_edges
                .iter()
                .filter(|(from, to, _, _)| *from == *node || *to == *node)
                .map(|(_, _, conf, _)| *conf)
                .max()
                .unwrap_or(0);
            let seed = cde_counterfactual_seed(*node, output.commit);
            CounterfactualResult::new(*delta, confidence, seed)
        })
        .collect();
    CausalReport::new(output.graph_commit, counterfactuals, 0)
}

fn cde_counterfactual_seed(node: CdeNodeId, output_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.counterfactual.seed.v1");
    hasher.update(&node.to_u16().to_be_bytes());
    hasher.update(output_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_policy_commit(decision: &PolicyDecision) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_POLICY_COMMIT_DOMAIN);
    hasher.update(&decision.kind.to_be_bytes());
    hasher.update(&decision.action.to_be_bytes());
    hasher.update(&decision.confidence_bp.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn nsr_fact_flags(phase_locked: bool, high_surprise: bool) -> u8 {
    (phase_locked as u8) | ((high_surprise as u8) << 1)
}

fn summarize_nsr_hits(hits: &[RuleHit]) -> ([u16; 3], Vec<NsrHitSummary>, bool) {
    let mut counts = [0u16; 3];
    let mut summaries = Vec::new();
    let mut warned = false;
    for hit in hits {
        match hit.severity {
            RuleSeverity::Info => counts[0] = counts[0].saturating_add(1),
            RuleSeverity::Warn => {
                counts[1] = counts[1].saturating_add(1);
                warned = true;
            }
            RuleSeverity::Block => counts[2] = counts[2].saturating_add(1),
        }
        if summaries.len() < NSR_HIT_SUMMARY_MAX {
            summaries.push(NsrHitSummary {
                rule_id: hit.id.0,
                severity: hit.severity as u8,
                reason: hit.reason,
                commit: hit.commit,
            });
        }
    }
    (counts, summaries, warned)
}

struct SpikeCountSummary {
    total: u16,
    top_kinds: Vec<SpikeKind>,
}

fn summarize_spike_counts(counts: &[(SpikeKind, u16)]) -> SpikeCountSummary {
    if counts.is_empty() {
        return SpikeCountSummary {
            total: 0,
            top_kinds: Vec::new(),
        };
    }
    let total = counts
        .iter()
        .map(|(_, count)| *count as u32)
        .sum::<u32>()
        .min(u32::from(u16::MAX)) as u16;
    let mut sorted = counts.to_vec();
    sorted.sort_by(|(kind_a, count_a), (kind_b, count_b)| {
        count_b.cmp(count_a).then_with(|| kind_a.cmp(kind_b))
    });
    let top_kinds = sorted.iter().take(3).map(|(kind, _)| *kind).collect();
    SpikeCountSummary { total, top_kinds }
}

fn empty_spike_outputs(cycle_id: u64) -> SpikeOutputs {
    SpikeOutputs {
        cycle_id,
        accepted_root: Digest32::new([0u8; 32]),
        accepted: Vec::new(),
        counts: Vec::new(),
        max_intensity: 0,
        commit: Digest32::new([0u8; 32]),
    }
}

fn spike_params_from_knobs(knobs: &SnnKnobs) -> SpikeParams {
    SpikeParams::new(
        knobs.threshold_for(SpikeKind::Feature),
        knobs.threshold_for(SpikeKind::Novelty),
        knobs.threshold_for(SpikeKind::Threat),
        knobs.threshold_for(SpikeKind::Reward),
        knobs.threshold_for(SpikeKind::CausalLink),
        knobs.threshold_for(SpikeKind::PolicySignal),
        knobs.threshold_for(SpikeKind::ThoughtOnly),
        knobs.threshold_for(SpikeKind::ReplayHint),
        usize::from(knobs.verify_limit.max(1)),
    )
}

fn replay_pressure_from_spikes(spike_counts: &[(SpikeKind, u16)]) -> u16 {
    spike_counts
        .iter()
        .find_map(|(kind, count)| (*kind == SpikeKind::ReplayHint).then_some(*count))
        .unwrap_or(0)
}

fn apply_influence_replay_pressure(base: u16, influence: &InfluenceOutputs) -> u16 {
    let pressure = influence.node_value(InfluenceNodeId::ReplayPressure);
    if pressure <= 0 {
        return base;
    }
    let boost = (i32::from(pressure) / 2).clamp(0, 5000) as u16;
    base.saturating_add(boost).min(10_000)
}

fn coupling_influence_value(outputs: Option<&CouplingOutputs>, signal: SignalId) -> i16 {
    outputs
        .and_then(|outputs| {
            outputs
                .influences
                .iter()
                .find(|(id, _)| *id == signal)
                .map(|(_, value)| *value)
        })
        .unwrap_or(0)
}

fn apply_coupling_bias(base: u16, influence: i16, cap: u16) -> u16 {
    if influence == 0 {
        return base;
    }
    let cap_i16 = cap.min(10_000) as i16;
    let bias = influence.clamp(-cap_i16, cap_i16);
    if bias < 0 {
        base.saturating_sub(bias.unsigned_abs())
    } else {
        base.saturating_add(bias as u16).min(10_000)
    }
}

fn apply_coupling_bias_i16(base: i16, influence: i16, cap: i16) -> i16 {
    if influence == 0 {
        return base;
    }
    let cap = cap.abs();
    let bias = influence.clamp(-cap, cap);
    base.saturating_add(bias)
}

fn top_coupling_influences(influences: &[(SignalId, i16)], limit: usize) -> Vec<(u16, i16)> {
    let mut ranked = influences.to_vec();
    ranked.sort_by(|(id_a, value_a), (id_b, value_b)| {
        let mag_a = i32::from(*value_a).abs();
        let mag_b = i32::from(*value_b).abs();
        mag_b.cmp(&mag_a).then_with(|| id_a.cmp(id_b))
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(id, value)| (id.as_u16(), value))
        .collect()
}

fn coupling_checksum(influences: &[(SignalId, i16)]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.coupling.checksum.v1");
    hasher.update(
        &u32::try_from(influences.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for (id, value) in influences {
        hasher.update(&id.as_u16().to_be_bytes());
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn derive_sleep_active(
    sleep_pulse: bool,
    replay_pressure: u16,
    phi_proxy: u16,
    sleep_drive: i16,
    ncde_energy: u16,
    params: &StructuralParams,
) -> bool {
    if sleep_pulse {
        return true;
    }
    let phi_low = params.rsa.phi_min_apply.saturating_sub(400);
    let drive = sleep_drive.max(0) as u16;
    let energy_drive = ncde_energy / 2;
    let composite_drive = drive.saturating_add(energy_drive).min(10_000);
    replay_pressure >= 5000 || phi_proxy < phi_low || composite_drive >= 2500
}

fn spike_record_commit(
    accepted_root: Digest32,
    count: u16,
    top_kinds: &[SpikeKind],
    max_intensity: u16,
) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.spikebus.record.v2");
    hasher.update(accepted_root.as_bytes());
    hasher.update(&count.to_be_bytes());
    hasher.update(&max_intensity.to_be_bytes());
    hasher.update(&(top_kinds.len() as u16).to_be_bytes());
    for kind in top_kinds {
        hasher.update(&kind.as_u16().to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn workspace_suppression_count(snapshot: &WorkspaceSnapshot) -> u16 {
    snapshot
        .broadcast
        .iter()
        .filter(|signal| {
            matches!(signal.kind, SignalKind::Output) && signal.summary.contains("OUTPUT=SUPPRESS")
        })
        .count()
        .min(u16::MAX as usize) as u16
}

fn drift_score_from_snapshot(snapshot: &WorkspaceSnapshot) -> u16 {
    snapshot
        .broadcast
        .iter()
        .find_map(|signal| {
            if !matches!(signal.kind, SignalKind::Consistency) {
                return None;
            }
            if !signal.summary.contains("DRIFT=") {
                return None;
            }
            signal
                .summary
                .split_whitespace()
                .find_map(|token| token.strip_prefix("SCORE="))
                .and_then(|value| value.parse::<u16>().ok())
        })
        .unwrap_or(0)
}

fn short_digest(digest: Digest32) -> String {
    hex::encode(digest.as_bytes()).chars().take(8).collect()
}

fn iit_band_tier(band: IitBand) -> u8 {
    match band {
        IitBand::Low => 1,
        IitBand::Medium => 2,
        IitBand::High => 3,
    }
}

fn drift_band_tier(band: DriftBand) -> u8 {
    match band {
        DriftBand::Low => 1,
        DriftBand::Medium => 2,
        DriftBand::High => 3,
        DriftBand::Critical => 4,
    }
}

fn focus_channel_score(channel: FocusChannel) -> u16 {
    match channel {
        FocusChannel::Threat => 9000,
        FocusChannel::Task => 7000,
        FocusChannel::Exploration => 6500,
        FocusChannel::Memory => 5000,
        FocusChannel::Social => 4500,
        FocusChannel::Idle => 2000,
    }
}

fn iit_action_effects(
    base_workspace: WorkspaceConfig,
    base_router: &RouterConfig,
    actions: &[IitAction],
) -> IitActionEffects {
    let mut integration_bias: i16 = 0;
    let mut broadcast_cap = base_workspace.broadcast_cap;
    let mut max_thought_frames_per_cycle = base_router.max_thought_frames_per_cycle;

    for action in actions {
        match action.kind {
            IitActionKind::Fusion => {
                let bump = (action.intensity / 1000).max(1) as usize;
                broadcast_cap = broadcast_cap.saturating_add(bump);
                integration_bias = integration_bias.saturating_sub((action.intensity / 3) as i16);
            }
            IitActionKind::ReplayBias => {
                integration_bias = integration_bias.saturating_sub((action.intensity / 2) as i16);
            }
            IitActionKind::Isolate => {
                let reduction = action.intensity / 500 + 1;
                max_thought_frames_per_cycle =
                    max_thought_frames_per_cycle.saturating_sub(reduction);
            }
            IitActionKind::Throttle => {
                let reduction = action.intensity / 1000 + 1;
                max_thought_frames_per_cycle =
                    max_thought_frames_per_cycle.saturating_sub(reduction);
            }
        }
    }

    let broadcast_cap_max = base_workspace.broadcast_cap.saturating_add(8);
    let broadcast_cap = broadcast_cap.clamp(1, broadcast_cap_max.max(1));
    let max_thought_frames_per_cycle = max_thought_frames_per_cycle.max(4);

    IitActionEffects {
        integration_bias,
        broadcast_cap,
        max_thought_frames_per_cycle,
    }
}

fn consistency_action_effects(
    base_max_thought_frames: u16,
    actions: &[ConsistencyAction],
) -> ConsistencyActionEffects {
    let mut max_thought_frames_per_cycle = base_max_thought_frames;
    let mut noise_boost = 0u16;
    let mut replay_boost = 0u16;

    for action in actions {
        match action.kind {
            ConsistencyActionKind::ReduceRecursion => {
                let reduction = action.intensity / 2000 + 1;
                max_thought_frames_per_cycle =
                    max_thought_frames_per_cycle.saturating_sub(reduction);
            }
            ConsistencyActionKind::ThrottleOutput => {
                let reduction = action.intensity / 2500 + 1;
                max_thought_frames_per_cycle =
                    max_thought_frames_per_cycle.saturating_sub(reduction);
            }
            ConsistencyActionKind::DampNoise => {
                noise_boost = noise_boost.saturating_add(action.intensity / 2);
            }
            ConsistencyActionKind::IncreaseReplay => {
                replay_boost = replay_boost.saturating_add(action.intensity / 2);
            }
        }
    }

    ConsistencyActionEffects {
        max_thought_frames_per_cycle: max_thought_frames_per_cycle.max(1),
        noise_boost,
        replay_boost,
    }
}

fn apply_consistency_effects(
    mut weights: AttentionWeights,
    effects: Option<ConsistencyActionEffects>,
) -> AttentionWeights {
    let Some(effects) = effects else {
        return weights;
    };
    let mut changed = false;
    if effects.noise_boost > 0 {
        weights.noise_suppress =
            clamp_u16(u32::from(weights.noise_suppress) + u32::from(effects.noise_boost));
        changed = true;
    }
    if effects.replay_boost > 0 {
        weights.replay_bias =
            clamp_u16(u32::from(weights.replay_bias) + u32::from(effects.replay_boost));
        changed = true;
    }
    if changed {
        weights.commit = commit_attention_override(&weights);
    }
    weights
}

fn apply_influence_effects(
    mut weights: AttentionWeights,
    influence: Option<&InfluenceOutputs>,
) -> AttentionWeights {
    let Some(influence) = influence else {
        return weights;
    };
    let attention_in = influence.node_value(InfluenceNodeId::AttentionGain);
    let memory_in = influence.node_value(InfluenceNodeId::WorkingMemory);
    let mut changed = false;
    let delta = (i32::from(attention_in) + i32::from(memory_in)) / 4;
    if delta != 0 {
        let adjusted = (i32::from(weights.gain) + delta).clamp(0, 10_000);
        weights.gain = adjusted as u16;
        changed = true;
    }
    if attention_in <= -2000 {
        if weights.channel != FocusChannel::Threat {
            weights.channel = FocusChannel::Threat;
            changed = true;
        }
    } else if (attention_in >= 2000 || memory_in >= 2000)
        && weights.channel != FocusChannel::Threat
        && weights.channel != FocusChannel::Exploration
    {
        weights.channel = FocusChannel::Exploration;
        changed = true;
    }
    if changed {
        weights.commit = commit_attention_override(&weights);
    }
    weights
}

fn apply_influence_output_suppression(
    speech_gate: &mut [bool],
    outputs: &[AiOutput],
    influence: Option<&InfluenceOutputs>,
) {
    let Some(influence) = influence else {
        return;
    };
    let suppression = influence.node_value(InfluenceNodeId::OutputSuppression);
    if suppression < 1500 {
        return;
    }
    for (idx, output) in outputs.iter().enumerate() {
        if output.channel == OutputChannel::Speech {
            if let Some(entry) = speech_gate.get_mut(idx) {
                *entry = false;
            }
        }
    }
}

fn action_intents_from_outputs(outputs: &[AiOutput]) -> Vec<ActionIntent> {
    outputs
        .iter()
        .map(|output| {
            let tag = match output.channel {
                OutputChannel::Speech => "external_effect",
                OutputChannel::Thought => "internal_thought",
            };
            ActionIntent::new(tag)
        })
        .collect()
}

fn nsr_verdict_token(verdict: NsrVerdict) -> &'static str {
    match verdict {
        NsrVerdict::Allow => "Allow",
        NsrVerdict::Restrict => "Restrict",
        NsrVerdict::Deny => "Deny",
    }
}

fn nsr_verdict_token_lower(verdict: NsrVerdict) -> &'static str {
    match verdict {
        NsrVerdict::Allow => "allow",
        NsrVerdict::Restrict => "restrict",
        NsrVerdict::Deny => "deny",
    }
}

fn nsr_signal_priority(verdict: NsrVerdict) -> u16 {
    match verdict {
        NsrVerdict::Allow => 4200,
        NsrVerdict::Restrict => 7600,
        NsrVerdict::Deny => 9500,
    }
}

fn is_causal_violation_code(code: &str) -> bool {
    matches!(
        code,
        "NSR_CAUSAL_RISK_INCREASE"
            | "NSR_CAUSAL_CONFIDENCE_HIGH_DENY"
            | "NSR_CAUSAL_UNCERTAIN_WARN"
    )
}

fn consistency_score_from_nsr(report: Option<&NsrReport>) -> u16 {
    match report {
        Some(report) if report.verdict == NsrVerdict::Allow => 10_000,
        Some(report) if report.verdict == NsrVerdict::Restrict => 4500,
        Some(_) => 2000,
        None => 5000,
    }
}

fn commit_attention_override(weights: &AttentionWeights) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.attn.override.v1");
    hasher.update(weights.channel.as_str().as_bytes());
    hasher.update(&weights.gain.to_be_bytes());
    hasher.update(&weights.noise_suppress.to_be_bytes());
    hasher.update(&weights.replay_bias.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn avg_u16(values: &[u16; COHERENCE_LAG_LEN]) -> u16 {
    let sum: u32 = values.iter().map(|value| u32::from(*value)).sum();
    let avg = sum / values.len().max(1) as u32;
    u16::try_from(avg.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
}

fn avg_recent(values: &[u16; COHERENCE_LAG_LEN]) -> u16 {
    let sum = u32::from(values[0]).saturating_add(u32::from(values[1]));
    u16::try_from(sum / 2).unwrap_or(0)
}

fn avg_prior(values: &[u16; COHERENCE_LAG_LEN]) -> u16 {
    let sum = u32::from(values[2]).saturating_add(u32::from(values[3]));
    u16::try_from(sum / 2).unwrap_or(0)
}

fn commit_coherence_lag(lag: &CoherenceLag) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.router.coherence.lag.v1");
    for commit in &lag.phase_commit {
        hasher.update(commit.as_bytes());
    }
    for commit in &lag.ssm_commit {
        hasher.update(commit.as_bytes());
    }
    for commit in &lag.iit_commit {
        hasher.update(commit.as_bytes());
    }
    hasher.update(&lag.nsr_verdict);
    for value in &lag.novelty {
        hasher.update(&value.to_be_bytes());
    }
    for value in &lag.salience {
        hasher.update(&value.to_be_bytes());
    }
    for value in &lag.plv {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn update_mode_seed(
    phase_commit: Digest32,
    ssm_commit: Digest32,
    iit_commit: Digest32,
) -> UpdateMode {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.router.update.mode.v1");
    hasher.update(phase_commit.as_bytes());
    hasher.update(ssm_commit.as_bytes());
    hasher.update(iit_commit.as_bytes());
    let bytes = hasher.finalize();
    let raw = u32::from_be_bytes([
        bytes.as_bytes()[0],
        bytes.as_bytes()[1],
        bytes.as_bytes()[2],
        bytes.as_bytes()[3],
    ]);
    UpdateMode::from_u8((raw % 4) as u8)
}

fn update_mode_commit(mode: UpdateMode, lag_commit: Digest32) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.router.update.mode.commit.v1");
    hasher.update(&[mode.as_u8()]);
    hasher.update(lag_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_rsa_apply_gate(
    mode: UpdateMode,
    nsr_verdict: NsrVerdict,
    phi_proxy: u16,
    phi_high: u16,
    tcf_budget: u16,
) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.router.rsa.apply.gate.v1");
    hasher.update(&[mode.as_u8()]);
    hasher.update(&[nsr_verdict.as_u8()]);
    hasher.update(&phi_proxy.to_be_bytes());
    hasher.update(&phi_high.to_be_bytes());
    hasher.update(&tcf_budget.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn influence_inputs_commit(
    cycle_id: u64,
    phase_commit: Digest32,
    coherence_plv: u16,
    phi_proxy: u16,
    ssm_salience: u16,
    ssm_novelty: u16,
    ncde_energy: u16,
    ncde_commit: Digest32,
    ncde_state_digest: Digest32,
    nsr_verdict: u8,
    risk: u16,
    drift: u16,
    surprise: u16,
    cde_commit: Option<Digest32>,
    sle_self_symbol: Option<Digest32>,
    rsa_applied: bool,
) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.influence.inputs.v2");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(&coherence_plv.to_be_bytes());
    hasher.update(&phi_proxy.to_be_bytes());
    hasher.update(&ssm_salience.to_be_bytes());
    hasher.update(&ssm_novelty.to_be_bytes());
    hasher.update(&ncde_energy.to_be_bytes());
    hasher.update(ncde_commit.as_bytes());
    hasher.update(ncde_state_digest.as_bytes());
    hasher.update(&[nsr_verdict]);
    hasher.update(&risk.to_be_bytes());
    hasher.update(&drift.to_be_bytes());
    hasher.update(&surprise.to_be_bytes());
    if let Some(commit) = cde_commit {
        hasher.update(commit.as_bytes());
    }
    if let Some(commit) = sle_self_symbol {
        hasher.update(commit.as_bytes());
    }
    hasher.update(&[rsa_applied as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn feature_translation_commit(activation_commit: Digest32, selection_commit: Digest32) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.router.feature_translation.v1");
    hasher.update(activation_commit.as_bytes());
    hasher.update(selection_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn sandbox_denied_commit(reason: &str, report_commit: Digest32) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.router.sandbox_denied.v1");
    hasher.update(report_commit.as_bytes());
    hasher.update(reason.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_risk_results(results: &[RiskGateResult]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.router.risk_commit.v1");
    hasher.update(&u64::try_from(results.len()).unwrap_or(0).to_be_bytes());
    for result in results {
        hasher.update(&[result.decision as u8]);
        hasher.update(&result.risk.to_be_bytes());
        let reasons_digest = digest_reasons(&result.reasons);
        hasher.update(reasons_digest.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_rsa_proposals(proposals: &[RsaProposal]) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.router.rsa_proposals.v1");
    hasher.update(&u64::try_from(proposals.len()).unwrap_or(0).to_be_bytes());
    for proposal in proposals {
        hasher.update(proposal.id.as_bytes());
        hasher.update(&proposal.expected_gain.to_be_bytes());
        hasher.update(&proposal.risks.to_be_bytes());
        hasher.update(
            &u16::try_from(proposal.target.len())
                .unwrap_or(0)
                .to_be_bytes(),
        );
        hasher.update(proposal.target.as_bytes());
        hasher.update(proposal.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn apply_rsa_deltas(
    proposal: &rsa_v0::RsaProposal,
    mut onn_params: OnnParams,
    mut tcf_params: TcfConfig,
    mut ncde_params: NcdeParams,
    mut cde_params: CdeParams,
    mut feature_params: FeatureSpikeParams,
) -> (
    OnnParams,
    TcfConfig,
    NcdeParams,
    CdeParams,
    FeatureSpikeParams,
) {
    for delta in &proposal.deltas {
        match delta.target {
            ParamTarget::OnnCoupling => {
                onn_params = apply_coupling_delta(&onn_params, delta.delta);
            }
            ParamTarget::OnnLockWindow => {
                onn_params = apply_lock_window_delta(&onn_params, delta.delta);
            }
            ParamTarget::TcfAttK => {
                tcf_params = apply_attn_k_delta(&tcf_params, delta.delta);
            }
            ParamTarget::TcfReplayK => {
                tcf_params = apply_replay_k_delta(&tcf_params, delta.delta);
            }
            ParamTarget::TcfEnergyK => {
                tcf_params = apply_energy_k_delta(&tcf_params, delta.delta);
            }
            ParamTarget::NcdeGainPhase => {
                ncde_params = apply_gain_phase_delta(&ncde_params, delta.delta);
            }
            ParamTarget::NcdeGainSpike => {
                ncde_params = apply_gain_spike_delta(&ncde_params, delta.delta);
            }
            ParamTarget::NcdeLeak => {
                ncde_params = apply_leak_delta(&ncde_params, delta.delta);
            }
            ParamTarget::CdeScoreStep => {
                cde_params = apply_score_step_delta(&cde_params, delta.delta);
            }
            ParamTarget::CdeEdgeThresh => {
                cde_params = apply_edge_thresh_delta(&cde_params, delta.delta);
            }
            ParamTarget::FeatureSpikeThresh => {
                feature_params = apply_feature_thresh_delta(&feature_params, delta.delta);
            }
            ParamTarget::ThreatSpikeThresh => {
                feature_params = apply_threat_thresh_delta(&feature_params, delta.delta);
            }
            ParamTarget::Unknown(_) => {}
        }
    }
    (
        onn_params,
        tcf_params,
        ncde_params,
        cde_params,
        feature_params,
    )
}

fn digest32_to_proto(digest: Digest32) -> Digest {
    Digest {
        algorithm: AlgoId::Blake3_256.to_string(),
        value: Vec::new(),
        algo_id: Some(AlgoId::BLAKE3_256_ID as u32),
        domain: None,
        value_32: Some(digest.as_bytes().to_vec()),
    }
}

fn digest_ai_external_commit(percept_commit: Digest32, verdict: NsrVerdict) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ai.host.external.v1");
    hasher.update(percept_commit.as_bytes());
    hasher.update(&[verdict.as_u8()]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn ai_output_from_candidate(candidate: &ucf_ai_host::AiOutputCandidate) -> AiOutput {
    let payload = hex::encode(candidate.payload_commit.as_bytes());
    AiOutput {
        channel: OutputChannel::Speech,
        content: format!("ai-host:{payload}"),
        confidence: candidate.confidence,
        rationale_commit: Some(candidate.payload_commit),
        integration_score: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ucf_ai_host::MockAiHost;
    use ucf_ai_port::{MockAiPort, PolicySpeechGate};
    use ucf_archive::InMemoryArchive;
    use ucf_archive_store::InMemoryArchiveStore;
    use ucf_cde::MockCausalEngine;
    use ucf_iit::MockIntegrationMonitor;
    use ucf_jepa::MockWorldModel;
    use ucf_ncde::MockContinuousDynamics;
    use ucf_nsr::MockNeuroSymbolicReasoner;
    use ucf_onn::MockPhaseProvider;
    use ucf_policy_ecology::PolicyEcology;
    use ucf_policy_gateway::NoOpPolicyEvaluator;
    use ucf_risk_gate::PolicyRiskGate;
    use ucf_sle::MockStrangeLoop;
    use ucf_spikebus::MockSpikeRouter;
    use ucf_ssm::MockWorkingMemory;
    use ucf_structural_store::{OnnKnobs, SnnKnobs, StructuralParams, StructuralStore};
    use ucf_tcf::MockTemporalCoordinator;
    use ucf_tom_port::MockTomPort;

    fn build_router() -> Router {
        let policy = Arc::new(NoOpPolicyEvaluator::new());
        let archive = Arc::new(InMemoryArchive::new());
        let archive_store = Arc::new(InMemoryArchiveStore::new());
        let ai_port = Arc::new(MockAiPort::default());
        let policy_ecology = PolicyEcology::allow_all();
        let speech_gate = Arc::new(PolicySpeechGate::new(policy_ecology.clone()));
        let risk_gate = Arc::new(PolicyRiskGate::new(policy_ecology));
        let tom_port = Arc::new(MockTomPort::new());

        Router::new(
            policy,
            archive,
            archive_store,
            None,
            ai_port,
            speech_gate,
            risk_gate,
            tom_port,
            None,
        )
    }

    fn test_phase_bus(cycle_id: u64, gamma_bucket: u8, global_plv: u16, seed: u8) -> PhaseBus {
        let mut osc_buckets = [0u8; 16];
        osc_buckets[0] = gamma_bucket;
        PhaseBus {
            cycle_id,
            gamma_bucket,
            global_plv,
            osc_buckets,
            phase_commit: Digest32::new([seed; 32]),
            commit: Digest32::new([seed.wrapping_add(1); 32]),
        }
    }

    fn test_control_frame(frame_id: &str) -> ControlFrameNormalized {
        let decision = PolicyDecision {
            kind: DecisionKind::DecisionKindAllow as i32,
            action: ucf_types::v1::spec::ActionCode::ActionCodeContinue as i32,
            rationale: "ok".to_string(),
            confidence_bp: 10_000,
            constraint_ids: Vec::new(),
        };
        let cf = ControlFrame {
            frame_id: frame_id.to_string(),
            issued_at_ms: 0,
            decision: Some(decision),
            evidence_ids: Vec::new(),
            policy_id: "policy-allow".to_string(),
        };
        ucf_sandbox::normalize(cf)
    }

    fn run_runtime_cycle(modules: &mut RuntimeModules, cycle_id: u64) {
        let budget = GainBudget::default();
        let onn_inputs = OnnInputs::new(
            cycle_id,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            2,
            1200,
            900,
            700,
        );
        let onn_outputs = modules.phase.tick_with_budget(&onn_inputs, &budget);
        let spike_inputs = SpikeInputs::new(cycle_id, onn_outputs.lock, Vec::new());
        modules.spikes.set_params(SpikeParams::default());
        let spike_outputs = modules.spikes.tick(&spike_inputs);

        let last_world_state = modules.world.last_world_state();
        let jepa_inputs = JepaInputs::new(
            cycle_id,
            Digest32::new([6u8; 32]),
            1100,
            Digest32::new([7u8; 32]),
            onn_outputs.phase_bus.commit,
            onn_outputs.phase_bus.gamma_bucket,
            last_world_state,
            Digest32::new([8u8; 32]),
        );
        let jepa_outputs = modules.world.tick(&jepa_inputs);

        let ssm_inputs = SsmInputs::new(
            cycle_id,
            onn_outputs.phase_bus.commit,
            onn_outputs.phase_bus.gamma_bucket,
            Digest32::new([9u8; 32]),
            900,
            spike_outputs.accepted_root,
            spike_outputs.max_intensity,
            spike_outputs.counts.clone(),
            Digest32::new([10u8; 32]),
            Vec::new(),
            10_000,
            10_000,
            Digest32::new([15u8; 32]),
            0,
            0,
            1200,
            800,
            600,
            400,
            700,
            Digest32::new([16u8; 32]),
        );
        let ssm_outputs = modules.ssm.tick_with_budget(&ssm_inputs, &budget);

        let ncde_inputs = NcdeInputs::new(
            cycle_id,
            onn_outputs.phase_bus.commit,
            onn_outputs.phase_bus.gamma_bucket,
            onn_outputs.phase_bus.global_plv,
            spike_outputs.accepted_root,
            ucf_ncde::SpikeCountsSummary::from_counts(&spike_outputs.counts),
            1000,
            800,
            Digest32::new([11u8; 32]),
        );
        let ncde_outputs = modules.ncde.tick_with_budget(&ncde_inputs, &budget);

        let cde_inputs = CdeV1Inputs::new(
            cycle_id,
            onn_outputs.phase_bus.commit,
            onn_outputs.phase_bus.gamma_bucket,
            ssm_outputs.ssm_salience,
            ssm_outputs.ssm_novelty,
            0,
            1000,
            900,
            800,
            1200,
            ncde_outputs.ncde_energy,
            onn_outputs.phase_bus.global_plv,
            1500,
            700,
            600,
            500,
            false,
            false,
            spike_outputs.accepted_root,
            Digest32::new([12u8; 32]),
        );
        let cde_outputs = modules.cde.tick(&cde_inputs);

        let iit_inputs = IitInputs::new(
            cycle_id,
            onn_outputs.phase_bus.commit,
            onn_outputs.phase_bus.gamma_bucket,
            onn_outputs.phase_bus.global_plv,
            ssm_outputs.ssm_state_commit,
            ncde_outputs.ncde_state_digest,
            cde_outputs.commit,
            Digest32::new([13u8; 32]),
            Digest32::new([14u8; 32]),
            800,
            700,
            600,
        );
        let iit_outputs = modules.iit.tick(&iit_inputs);

        let tcf_inputs = TcfInputs::new(
            cycle_id,
            onn_outputs.phase_bus.commit,
            onn_outputs.phase_bus.gamma_bucket,
            onn_outputs.phase_bus.global_plv,
            iit_outputs.phi_proxy,
            800,
            700,
            600,
            ssm_outputs.ssm_attention_gain,
            ncde_outputs.flow_energy,
            iit_outputs.hints_commit,
            iit_outputs.tighten_sync,
            iit_outputs.damp_output,
            iit_outputs.damp_learning,
            iit_outputs.request_replay,
            Digest32::new([15u8; 32]),
            0,
            true,
        );
        let tcf_plan = modules.tcf.tick_with_budget(&tcf_inputs, &budget);

        let nsr_inputs = NsrInputs::new(
            cycle_id,
            onn_outputs.phase_bus.commit,
            Digest32::new([16u8; 32]),
            Vec::new(),
        );
        let nsr_outputs = modules.nsr.tick(&nsr_inputs);

        let sle_inputs = SleInputs::new(
            cycle_id,
            onn_outputs.phase_bus.commit,
            onn_outputs.phase_bus.gamma_bucket,
            ssm_outputs.ssm_state_commit,
            ssm_outputs.ssm_salience,
            ssm_outputs.ssm_novelty,
            ncde_outputs.ncde_state_digest,
            ncde_outputs.ncde_energy,
            cde_outputs.commit,
            nsr_outputs.verdict.as_u8(),
            nsr_outputs.trace_root,
            iit_outputs.phi_proxy,
            onn_outputs.phase_bus.global_plv,
            tcf_plan.sleep_active,
            tcf_plan.replay_active,
            700,
            600,
            500,
        );
        let _ = modules.sle.tick(&sle_inputs);

        let _ = jepa_outputs;
    }

    #[test]
    fn predictive_observation_changes_with_ssm_state() {
        let state_a = SsmState::new(vec![1, -2, 3]);
        let state_b = SsmState::new(vec![2, -2, 3]);

        let obs_a = observation_from_ssm_state(&state_a);
        let obs_b = observation_from_ssm_state(&state_b);

        assert_ne!(obs_a.commit, obs_b.commit);
    }

    #[test]
    fn fusion_action_biases_attention_replay() {
        let action = IitAction {
            kind: IitActionKind::Fusion,
            intensity: 2000,
            commit: Digest32::new([9u8; 32]),
        };
        let effects = iit_action_effects(
            WorkspaceConfig {
                cap: 64,
                broadcast_cap: 8,
            },
            &RouterConfig {
                thought_capacity: 64,
                max_thought_frames_per_cycle: 10,
                external_enabled: true,
            },
            &[action],
        );
        let controller = AttnController;
        let base_inputs = AttnInputs {
            policy_class: 1,
            risk_score: 1000,
            integration_score: 6000,
            consistency_instability: 0,
            intent_type: AttnController::INTENT_ASK_INFO,
            surprise_score: 0,
        };
        let base_weights = controller.compute(&base_inputs);
        let biased_score =
            apply_integration_bias(base_inputs.integration_score, effects.integration_bias);
        let biased_inputs = AttnInputs {
            integration_score: biased_score,
            ..base_inputs
        };
        let biased_weights = controller.compute(&biased_inputs);

        assert!(biased_weights.replay_bias >= base_weights.replay_bias);
    }

    #[test]
    fn high_drift_reduces_output_router_thought_budget() {
        let engine = ConsistencyEngine;
        let anchor =
            ucf_ism::IsmAnchor::new(Digest32::new([1u8; 32]), Digest32::new([2u8; 32]), 1, 1);
        let self_state = SelfState {
            cycle_id: 1,
            ssm_commit: Digest32::new([1u8; 32]),
            workspace_commit: Digest32::new([2u8; 32]),
            risk_commit: Digest32::new([3u8; 32]),
            attn_commit: Digest32::new([4u8; 32]),
            ncde_commit: Digest32::new([6u8; 32]),
            consistency: 0,
            commit: Digest32::new([5u8; 32]),
        };
        let anchors = [anchor];
        let inputs = ConsistencyInputs {
            self_state: &self_state,
            self_symbol: Digest32::new([255u8; 32]),
            ism_root: Digest32::new([0u8; 32]),
            anchors: &anchors,
            suppression_count: 3,
            policy_class: 2,
            policy_stable: false,
            risk_score: 9000,
            surprise_band: SurpriseBand::Critical,
            phi: 1000,
        };
        let (report, actions) = engine.evaluate(&inputs);
        assert!(matches!(report.band, DriftBand::High | DriftBand::Critical));

        let mut output_router = OutputRouter::new(RouterConfig {
            thought_capacity: 64,
            max_thought_frames_per_cycle: 10,
            external_enabled: true,
        });
        let effects =
            consistency_action_effects(output_router.max_thought_frames_per_cycle(), &actions);
        output_router.set_max_thought_frames_per_cycle(effects.max_thought_frames_per_cycle);
        assert!(output_router.max_thought_frames_per_cycle() < 10);
    }

    #[test]
    fn high_drift_increases_replay_bias() {
        let action = ConsistencyAction {
            kind: ConsistencyActionKind::IncreaseReplay,
            intensity: 8000,
            commit: Digest32::new([8u8; 32]),
        };
        let effects = consistency_action_effects(10, &[action]);
        let controller = AttnController;
        let inputs = AttnInputs {
            policy_class: 1,
            risk_score: 1000,
            integration_score: 6000,
            consistency_instability: 0,
            intent_type: AttnController::INTENT_ASK_INFO,
            surprise_score: 0,
        };
        let base_weights = controller.compute(&inputs);
        let boosted = apply_consistency_effects(base_weights.clone(), Some(effects));

        assert!(boosted.replay_bias > base_weights.replay_bias);
    }

    #[test]
    fn influence_adjusts_attention_gain_and_channel() {
        let base = AttentionWeights {
            channel: FocusChannel::Task,
            gain: 3000,
            noise_suppress: 1200,
            replay_bias: 1500,
            commit: Digest32::new([1u8; 32]),
        };
        let influence = InfluenceOutputs {
            cycle_id: 1,
            pulses_root: Digest32::new([9u8; 32]),
            pulses: Vec::new(),
            node_values: vec![
                (InfluenceNodeId::AttentionGain, 4200),
                (InfluenceNodeId::WorkingMemory, 800),
            ],
            commit: Digest32::new([2u8; 32]),
        };
        let updated = apply_influence_effects(base.clone(), Some(&influence));

        assert!(updated.gain > base.gain);
        assert_eq!(updated.channel, FocusChannel::Exploration);
    }

    #[test]
    fn replay_pressure_influence_boosts_pressure() {
        let base = 1200;
        let influence = InfluenceOutputs {
            cycle_id: 2,
            pulses_root: Digest32::new([5u8; 32]),
            pulses: Vec::new(),
            node_values: vec![(InfluenceNodeId::ReplayPressure, 4200)],
            commit: Digest32::new([6u8; 32]),
        };
        let boosted = apply_influence_replay_pressure(base, &influence);
        assert!(boosted > base);
    }

    #[test]
    fn ncde_commit_flows_into_workspace_and_self_state() {
        let router = build_router();
        let phase_bus = test_phase_bus(1, 0, 6000, 9);
        let output = router
            .tick_ncde(
                1,
                &phase_bus,
                4500,
                None,
                None,
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Novelty, 2)],
                1500,
                1000,
                2000,
                10_000,
            )
            .expect("ncde output");
        {
            let mut workspace = router.workspace.lock().expect("workspace lock");
            workspace.set_ncde_snapshot(
                output.commit,
                output.ncde_state_digest,
                output.ncde_energy,
                output.replay_pressure_hint,
            );
        }
        let snapshot = router.arbitrate_workspace(1);
        assert_eq!(snapshot.ncde_commit, output.commit);
        assert_eq!(snapshot.ncde_state_digest, output.ncde_state_digest);
        assert_eq!(snapshot.ncde_energy, output.ncde_energy);

        let base = AttentionWeights {
            channel: FocusChannel::Task,
            gain: 3000,
            noise_suppress: 1200,
            replay_bias: 1500,
            commit: Digest32::new([3u8; 32]),
        };
        {
            let mut guard = router.last_ncde_output.lock().expect("ncde lock");
            *guard = Some(output);
        }
        let biased = router.apply_ncde_attention_bias(base.clone());
        assert!(biased.gain >= base.gain);

        let state = SelfStateBuilder::new(1).ncde_commit(output.commit).build();
        assert_eq!(state.ncde_commit, output.commit);
    }

    #[test]
    fn ssm_commit_flows_into_workspace_and_attention() {
        let router = build_router();
        let phase_bus = test_phase_bus(1, 4, 7000, 4);
        let ncde_output = router
            .tick_ncde(
                1,
                &phase_bus,
                4500,
                None,
                None,
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Novelty, 2)],
                1500,
                1000,
                2000,
                10_000,
            )
            .expect("ncde output");
        let ssm_output = router
            .tick_ssm(
                &phase_bus,
                Digest32::new([4u8; 32]),
                1200,
                ncde_output.ncde_energy,
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Threat, 3)],
                0,
                1000,
                2000,
                1500,
            )
            .expect("ssm output");
        {
            let mut workspace = router.workspace.lock().expect("workspace lock");
            workspace.set_ssm_snapshot(
                ssm_output.commit,
                ssm_output.ssm_state_commit,
                ssm_output.ssm_state_digest,
                ssm_output.ssm_salience,
                ssm_output.ssm_novelty,
                ssm_output.ssm_attention_gain,
            );
        }
        let snapshot = router.arbitrate_workspace(1);
        assert_eq!(snapshot.ssm_commit, ssm_output.commit);
        assert_eq!(snapshot.ssm_state_commit, ssm_output.ssm_state_commit);

        let base = AttentionWeights {
            channel: FocusChannel::Task,
            gain: 3000,
            noise_suppress: 1200,
            replay_bias: 1500,
            commit: Digest32::new([4u8; 32]),
        };
        let biased =
            router.apply_ssm_attention_bias(base.clone(), Some(ssm_output.ssm_attention_gain));
        assert!(biased.gain >= base.gain);
    }

    #[test]
    fn sle_bias_applies_to_next_ssm_cycle() {
        let router_low = build_router();
        let router_high = build_router();
        let phase_bus = test_phase_bus(1, 4, 7000, 4);
        let sle_commit = Digest32::new([5u8; 32]);
        let reflection_commit = Digest32::new([6u8; 32]);
        {
            let mut workspace = router_low.workspace.lock().expect("workspace lock");
            workspace.set_sle_outputs(SleOutputsSnapshot {
                sle_commit,
                reflection_commit,
                reflection_class: 0,
                reflection_intensity: 0,
                thought_only_root: Digest32::new([0u8; 32]),
                ssm_bias: 0,
                cde_bias: 0,
                request_replay: false,
            });
        }
        let snapshot_low = router_low.arbitrate_workspace(1);
        if let Ok(mut guard) = router_low.last_workspace_snapshot.lock() {
            *guard = Some(snapshot_low);
        }
        {
            let mut workspace = router_high.workspace.lock().expect("workspace lock");
            workspace.set_sle_outputs(SleOutputsSnapshot {
                sle_commit,
                reflection_commit,
                reflection_class: 0,
                reflection_intensity: 0,
                thought_only_root: Digest32::new([0u8; 32]),
                ssm_bias: 1500,
                cde_bias: 0,
                request_replay: false,
            });
        }
        let snapshot_high = router_high.arbitrate_workspace(1);
        if let Ok(mut guard) = router_high.last_workspace_snapshot.lock() {
            *guard = Some(snapshot_high);
        }

        let output_low = router_low
            .tick_ssm(
                &phase_bus,
                Digest32::new([4u8; 32]),
                1200,
                2200,
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Threat, 3)],
                0,
                1000,
                2000,
                1500,
            )
            .expect("ssm output low");
        let output_high = router_high
            .tick_ssm(
                &phase_bus,
                Digest32::new([4u8; 32]),
                1200,
                2200,
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Threat, 3)],
                0,
                1000,
                2000,
                1500,
            )
            .expect("ssm output high");

        assert_ne!(output_low.ssm_state_commit, output_high.ssm_state_commit);
    }

    #[test]
    fn attention_cap_does_not_increase_when_ssm_gain_drops() {
        let router = build_router();
        let tcf_cap = 8000;
        let cap_high = router.attention_cap_from_memory(Some(7000), Some(7000), tcf_cap);
        let cap_low = router.attention_cap_from_memory(Some(3000), Some(7000), tcf_cap);
        assert!(cap_low <= cap_high);
    }

    #[test]
    fn nsr_restrict_forces_stabilize_and_blocks_rsa() {
        let router = build_router();
        let lag = CoherenceLag::new();
        let mode = router.compute_update_mode(&lag, NsrVerdict::Restrict, 1000, 500, 500, 8000);
        assert_eq!(mode, UpdateMode::Stabilize);
        let phi_high = router.iit_params().phi_high;
        let (allowed, _) =
            router.rsa_apply_gate(mode, NsrVerdict::Restrict, phi_high, phi_high, 5000);
        assert!(!allowed);
    }

    #[test]
    fn structural_params_drive_onn_window_and_snn_verify_limit() {
        let router = build_router();
        let base = StructuralStore::default_params();
        let onn = OnnKnobs::new(base.onn.k_couple, 8000, base.onn.couple_clamp_q12);
        let snn = SnnKnobs::new(base.snn.kind_thresholds.clone(), 12);
        let params = StructuralParams::new(
            onn,
            snn,
            base.nsr,
            base.replay,
            base.ssm,
            base.ncde,
            base.rsa,
        );
        {
            let mut store = router
                .structural_store
                .lock()
                .expect("structural store lock");
            *store = StructuralStore::new(params);
        }

        router.sync_onn_params();
        let onn_dither = router
            .runtime_modules
            .lock()
            .map(|modules| modules.phase.params().k_dither)
            .unwrap_or(0);
        assert_eq!(onn_dither, 8000);
        assert_eq!(router.current_snn_knobs().verify_limit, 12);
    }

    #[test]
    fn violation_shrinks_budget_deterministically() {
        let state = default_budget_state();
        let (next_state, _, _, cycle) = update_budget_state(
            state,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            5000,
            &[],
            10_000,
            10_000,
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            1,
        );
        assert!(cycle.triggers & TRIGGER_VIOLATION != 0);
        assert_eq!(next_state.current.master, 6000);
        assert_eq!(next_state.current.coupling, 5000);
        assert_eq!(next_state.current.tcf_learning, 4000);
    }

    #[test]
    fn low_plv_triggers_replay_and_stabilize() {
        let mut state = default_budget_state();
        let mut stable = 0;
        let mut last_violation = 0;
        let mut cycle = BudgetCycle {
            budget: state.current,
            low_plv_streak: 0,
            high_novelty_streak: 0,
            violation_streak: 0,
            triggers: 0,
            request_replay: false,
            stabilize_cycles: 0,
            learning_signal: LearningSignal {
                cycle_id: 0,
                learn_rate: 0,
                update_mass: 0,
                mode: 0,
                commit: Digest32::new([0u8; 32]),
            },
            structural_delta: StructuralDelta {
                cycle_id: 0,
                delta_root: Digest32::new([0u8; 32]),
                delta_mass: 0,
                targets: [0; 4],
                commit: Digest32::new([0u8; 32]),
            },
            spike_novelty_threshold_bump: 0,
        };

        for _ in 0..6 {
            let (next_state, next_stable, next_violation, next_cycle) = update_budget_state(
                state,
                stable,
                last_violation,
                1,
                0,
                0,
                0,
                0,
                2000,
                &[],
                10_000,
                10_000,
                Digest32::new([0u8; 32]),
                Digest32::new([0u8; 32]),
                Digest32::new([0u8; 32]),
                Digest32::new([0u8; 32]),
                Digest32::new([0u8; 32]),
                Digest32::new([0u8; 32]),
                Digest32::new([0u8; 32]),
                Digest32::new([0u8; 32]),
                0,
            );
            state = next_state;
            stable = next_stable;
            last_violation = next_violation;
            cycle = next_cycle;
        }

        assert!(cycle.triggers & TRIGGER_LOW_PLV != 0);
        assert!(cycle.request_replay);
        assert!(cycle.stabilize_cycles >= 8);
        assert_eq!(state.current.onn_coupling, 5000);
        assert_eq!(state.current.master, 7000);
    }

    #[test]
    fn low_plv_reduces_learning_rate() {
        let state = default_budget_state();
        let (_, _, _, low) = update_budget_state(
            state,
            0,
            0,
            1,
            4000,
            6000,
            6000,
            2000,
            1200,
            &[],
            10_000,
            10_000,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            Digest32::new([6u8; 32]),
            Digest32::new([7u8; 32]),
            Digest32::new([8u8; 32]),
            0,
        );
        let (_, _, _, high) = update_budget_state(
            state,
            0,
            0,
            1,
            4000,
            6000,
            6000,
            2000,
            9000,
            &[],
            10_000,
            10_000,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            Digest32::new([6u8; 32]),
            Digest32::new([7u8; 32]),
            Digest32::new([8u8; 32]),
            0,
        );
        assert!(low.learning_signal.learn_rate < high.learning_signal.learn_rate);
    }

    #[test]
    fn high_surprise_increases_learning_unless_nsr_block() {
        let state = default_budget_state();
        let (_, _, _, boosted) = update_budget_state(
            state,
            0,
            0,
            1,
            4000,
            5000,
            5000,
            8001,
            9000,
            &[],
            10_000,
            10_000,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            Digest32::new([6u8; 32]),
            Digest32::new([7u8; 32]),
            Digest32::new([8u8; 32]),
            0,
        );
        let (_, _, _, blocked) = update_budget_state(
            state,
            0,
            0,
            1,
            4000,
            5000,
            5000,
            8001,
            9000,
            &[(1, 1, 2)],
            10_000,
            10_000,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            Digest32::new([6u8; 32]),
            Digest32::new([7u8; 32]),
            Digest32::new([8u8; 32]),
            1,
        );
        assert!(blocked.learning_signal.learn_rate < boosted.learning_signal.learn_rate);
    }

    #[test]
    fn structural_delta_is_deterministic() {
        let learning = LearningSignal {
            cycle_id: 9,
            learn_rate: 7000,
            update_mass: 6500,
            mode: 1,
            commit: Digest32::new([9u8; 32]),
        };
        let a = commit_structural_delta(
            9,
            learning,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            2000,
            8000,
            8001,
            true,
        );
        let b = commit_structural_delta(
            9,
            learning,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            2000,
            8000,
            8001,
            true,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn runtime_modules_v1_defaults_run_cycles() {
        let mut modules = RuntimeModules::v1_defaults();
        run_runtime_cycle(&mut modules, 1);
        run_runtime_cycle(&mut modules, 2);
    }

    #[test]
    fn runtime_modules_mock_bundle_compiles() {
        let mut modules = RuntimeModules {
            phase: Box::new(MockPhaseProvider::default()),
            spikes: Box::new(MockSpikeRouter::default()),
            world: Box::new(MockWorldModel::default()),
            ssm: Box::new(MockWorkingMemory::default()),
            ncde: Box::new(MockContinuousDynamics::default()),
            cde: Box::new(MockCausalEngine::default()),
            nsr: Box::new(MockNeuroSymbolicReasoner::default()),
            tcf: Box::new(MockTemporalCoordinator::default()),
            iit: Box::new(MockIntegrationMonitor::default()),
            sle: Box::new(MockStrangeLoop::default()),
            ai_host: Box::new(MockAiHost::default()),
        };
        run_runtime_cycle(&mut modules, 42);
    }

    #[test]
    fn ai_host_internal_thoughts_are_stored_without_speech_output() {
        let router = build_router();
        let cf = test_control_frame("frame-ai-host-internal");
        let outcome = router
            .handle_control_frame(cf)
            .expect("router should accept control frame");
        assert!(outcome.speech_outputs.is_empty());
        let pending = router
            .pending_ai_spikes
            .lock()
            .expect("pending ai spikes lock");
        assert!(pending
            .iter()
            .any(|spike| spike.kind == SpikeKind::ThoughtOnly));
    }

    #[test]
    fn relaxation_requires_stable_window() {
        let budget = finalize_gain_budget(GainBudget {
            master: 9000,
            coupling: 9000,
            ..GainBudget::default()
        });
        let mut state = BudgetState {
            current: budget,
            low_plv_streak: 0,
            high_novelty_streak: 0,
            violation_streak: 0,
            adapt_cooldown: 0,
            spike_threshold_cooldown: 0,
            tcf_learning_cooldown: 0,
            commit: Digest32::new([0u8; 32]),
        };
        state.commit = commit_budget_state(&state);

        let (state_no_relax, stable_no_relax, _, _) = update_budget_state(
            state,
            10,
            0,
            1,
            0,
            0,
            0,
            0,
            5000,
            &[],
            10_000,
            10_000,
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            0,
        );
        assert_eq!(stable_no_relax, 11);
        assert_eq!(state_no_relax.current.master, 9000);

        let mut state_relax = BudgetState {
            current: budget,
            low_plv_streak: 0,
            high_novelty_streak: 0,
            violation_streak: 0,
            adapt_cooldown: 0,
            spike_threshold_cooldown: 0,
            tcf_learning_cooldown: 0,
            commit: Digest32::new([0u8; 32]),
        };
        state_relax.commit = commit_budget_state(&state_relax);
        let (state_relax, stable_relax, _, _) = update_budget_state(
            state_relax,
            11,
            0,
            1,
            0,
            0,
            0,
            0,
            5000,
            &[],
            10_000,
            10_000,
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            Digest32::new([0u8; 32]),
            0,
        );
        assert_eq!(stable_relax, 12);
        assert_eq!(state_relax.current.master, 9200);
    }
}
