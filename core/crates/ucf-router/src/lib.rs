#![forbid(unsafe_code)]

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use blake3::Hasher;
use ucf::boundary::{self, v1::WorkspaceBroadcastV1, v1::WorkspaceSignalV1};
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
use ucf_cde_scm::{
    edge_key, CausalReport, CdeEngine, CdeInputs, CdeNodeId, CdeOutputs, CounterfactualResult,
    Intervention, NSR_ATOM_MIN,
};
use ucf_commit::commit_policy_decision;
use ucf_consistency_engine::{
    ConsistencyAction, ConsistencyActionKind, ConsistencyEngine, ConsistencyInputs,
    ConsistencyReport, DriftBand,
};
use ucf_digital_brain::DigitalBrainPort;
use ucf_feature_spiker::{build_feature_spike_batch, FeatureSpikerInputs};
use ucf_feature_translator::{
    ActivationView, LensPort as FeatureLensPort, LensSelection, MockLensPort, MockSaePort,
    SaePort as FeatureSaePort,
};
use ucf_geist::{SelfState, SelfStateBuilder};
use ucf_iit::{IitInputs, IitMonitor as IitProxyMonitor, IitOutput};
use ucf_iit_monitor::{
    actions_for_phi, report_for_phi, IitAction, IitActionKind, IitBand, IitReport,
};
use ucf_influence::{InfluenceGraphV2, InfluenceInputs, InfluenceNodeId, InfluenceOutputs};
use ucf_ism::IsmStore;
use ucf_ncde::{NcdeCore, NcdeInputs, NcdeOutputs, NcdeParams};
use ucf_nsr::{ConstraintEngine, NsrEngine, NsrInputs, ReasoningTrace};
use ucf_nsr_port::{light_report, ActionIntent, NsrInput, NsrPort, NsrReport, NsrVerdict};
use ucf_onn::{OnnCore, OnnInputs, OscId, PhaseFrame};
use ucf_output_router::{
    GateBundle, NsrSummary, OutputRouter, OutputRouterEvent, RouterConfig, SandboxVerdict,
};
use ucf_policy_ecology::RiskGateResult;
use ucf_policy_gateway::PolicyEvaluator;
use ucf_predictive_coding::{
    band_for_score, error, surprise, Observation, PredictionError, SurpriseBand, SurpriseSignal,
    SurpriseUpdated, WorldModel, WorldStateVec,
};
use ucf_recursion_controller::{RecursionBudget, RecursionController, RecursionInputs};
use ucf_risk_gate::{digest_reasons, RiskGate};
use ucf_rsa::{rsa_gates_allow, RsaCore, RsaInputs, RsaOutputs};
use ucf_rsa_hooks::{MockRsaHook, RsaContext, RsaHook, RsaProposal};
use ucf_sandbox::{
    AiCallRequest, ControlFrameNormalized, IntentSummary, MockWasmSandbox, SandboxBudget,
    SandboxCaps, SandboxPort, SandboxReport,
};
use ucf_sle::{SelfReflex, SleCore, SleEngine, SleInputs, SleOutputs, SleStimulusKind};
use ucf_spike_encoder::encode_spike_with_window;
use ucf_spikebus::{SpikeBatch, SpikeBusSummary, SpikeEvent, SpikeKind, SpikeSuppression};
use ucf_ssm::{SsmCore, SsmInput as WmSsmInput, SsmOutput as WmSsmOutput, SsmParams};
use ucf_ssm_port::SsmState;
use ucf_structural_store::{
    NsrThresholds, ParamDeltaRef, SnnKnobs, StructuralCycleStats, StructuralDeltaProposal,
    StructuralParams, StructuralStore,
};
use ucf_tcf_port::{
    ai_mode_for_pulse, idle_attention, CyclePlan, CyclePlanned, DeterministicTcf, PulseKind,
    TcfPort,
};
use ucf_tom_port::{IntentType, TomPort};
use ucf_types::v1::spec::{ControlFrame, DecisionKind, Digest, ExperienceRecord, PolicyDecision};
use ucf_types::{AlgoId, Digest32, EvidenceId};
use ucf_workspace::{
    output_event_commit, InternalUtterance, SignalKind, Workspace, WorkspaceConfig,
    WorkspaceSignal, WorkspaceSnapshot,
};

const ISM_ANCHOR_TOP_K: usize = 4;
const FEATURE_SIGNAL_PRIORITY: u16 = 3200;
const FEATURE_RECORD_KIND: u16 = 42;
const SANDBOX_DENIED_RECORD_KIND: u16 = 73;
const CAUSAL_REPORT_RECORD_KIND: u16 = 91;
const CDE_OUTPUT_RECORD_KIND: u16 = 92;
const PHASE_FRAME_RECORD_KIND: u16 = 118;
const SPIKE_RECORD_KIND: u16 = 131;
const NCDE_RECORD_KIND: u16 = 142;
const SSM_RECORD_KIND: u16 = 149;
const ONN_COHERENCE_THROTTLE: u16 = 2000;
const ONN_COHERENCE_THROTTLE_WARN: u16 = 3500;
const LOCK_MIN_CAUSAL: u16 = 5200;
const LOCK_MIN_CONSIST: u16 = 5600;
const LOCK_MIN_SPEAK: u16 = 3000;
const PHI_OUTPUT_THRESHOLD: u16 = 3200;
const CAUSAL_REPORT_FLAG_LIGHT: u16 = 0b1000;
const SELF_CONSISTENCY_OK_THRESHOLD: u16 = 5000;

#[derive(Debug)]
pub enum RouterError {
    PolicyDenied(i32),
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
    nsr_engine: Mutex<NsrEngine>,
    tom_port: Arc<dyn TomPort + Send + Sync>,
    output_suppression_sink: Option<Arc<dyn OutputSuppressionSink + Send + Sync>>,
    attention_controller: Option<AttnController>,
    attention_sink: Option<Arc<dyn AttentionEventSink + Send + Sync>>,
    output_router: Mutex<OutputRouter>,
    output_router_base: RouterConfig,
    workspace: Arc<Mutex<Workspace>>,
    workspace_base: WorkspaceConfig,
    cycle_counter: AtomicU64,
    tcf_port: Mutex<Box<dyn TcfPort + Send + Sync>>,
    onn_core: Mutex<OnnCore>,
    last_attention: Mutex<AttentionWeights>,
    last_surprise: Mutex<Option<SurpriseSignal>>,
    stage_trace: Option<Arc<dyn StageTrace + Send + Sync>>,
    world_model: WorldModel,
    world_state: Mutex<Option<WorldStateVec>>,
    sle_engine: Arc<SleEngine>,
    sle_core: Mutex<SleCore>,
    consistency_engine: ConsistencyEngine,
    ism_store: Arc<Mutex<IsmStore>>,
    iit_monitor: Mutex<IitProxyMonitor>,
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
    influence_state: Mutex<InfluenceGraphV2>,
    last_influence_outputs: Mutex<Option<InfluenceOutputs>>,
    last_influence_root_commit: Mutex<Option<Digest32>>,
    ssm_core: Mutex<SsmCore>,
    last_ssm_output: Mutex<Option<WmSsmOutput>>,
    ncde_core: Mutex<NcdeCore>,
    last_ncde_output: Mutex<Option<NcdeOutputs>>,
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
    ssm_salience: Option<u16>,
}

pub trait StageTrace {
    fn record(&self, stage: PulseKind);
}

struct StageContext {
    decision: Option<PolicyDecision>,
    decision_kind: DecisionKind,
    inference: Option<AiInference>,
    sandbox_report: Option<SandboxReport>,
    sandbox_verdict: Option<SandboxVerdict>,
    tom_report: Option<ucf_tom_port::TomReport>,
    nsr_report: Option<NsrReport>,
    nsr_trace: Option<ReasoningTrace>,
    causal_report: Option<CausalReport>,
    attention_risk: u16,
    thought_outputs: Vec<AiOutput>,
    speech_outputs: Vec<AiOutput>,
    suppressions: Vec<OutputSuppressionInfo>,
    integration_score: Option<u16>,
    integration_bias: i16,
    predictive_result: Option<(PredictionError, SurpriseSignal)>,
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
    nsr_warn_streak: Option<u16>,
    recursion_budget: Option<RecursionBudget>,
    phase_commit: Option<Digest32>,
    coherence_plv: Option<u16>,
    onn_outputs: Option<ucf_onn::OnnOutputs>,
    spike_summary: Option<SpikeBusSummary>,
    replay_pressure: Option<u16>,
    drift_score: Option<u16>,
    surprise_score: Option<u16>,
    tcf_energy_smooth: Option<u16>,
    influence_outputs: Option<InfluenceOutputs>,
    structural_stats: Option<StructuralCycleStats>,
    structural_proposal: Option<StructuralDeltaProposal>,
    cde_output: Option<CdeOutputs>,
    ncde_output: Option<NcdeOutputs>,
    ssm_output: Option<WmSsmOutput>,
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
            nsr_trace: None,
            causal_report: None,
            attention_risk: 0,
            thought_outputs: Vec::new(),
            speech_outputs: Vec::new(),
            suppressions: Vec::new(),
            integration_score: None,
            integration_bias: 0,
            predictive_result: None,
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
            nsr_warn_streak: None,
            recursion_budget: None,
            phase_commit: None,
            coherence_plv: None,
            onn_outputs: None,
            spike_summary: None,
            replay_pressure: None,
            drift_score: None,
            surprise_score: None,
            tcf_energy_smooth: None,
            influence_outputs: None,
            structural_stats: None,
            structural_proposal: None,
            cde_output: None,
            ncde_output: None,
            ssm_output: None,
        }
    }
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
            nsr_engine: Mutex::new(NsrEngine::new(NsrThresholds::default())),
            tom_port,
            output_suppression_sink,
            attention_controller: Some(AttnController),
            attention_sink: None,
            output_router_base: output_router_base.clone(),
            output_router: Mutex::new(OutputRouter::new(output_router_base)),
            workspace_base,
            workspace: Arc::new(Mutex::new(Workspace::new(workspace_base))),
            cycle_counter: AtomicU64::new(0),
            tcf_port: Mutex::new(Box::new(DeterministicTcf::default())),
            onn_core: Mutex::new(OnnCore::default()),
            last_attention: Mutex::new(idle_attention()),
            last_surprise: Mutex::new(None),
            stage_trace: None,
            world_model: WorldModel::default(),
            world_state: Mutex::new(None),
            sle_engine: Arc::new(SleEngine::new(6)),
            sle_core: Mutex::new(SleCore::default()),
            consistency_engine: ConsistencyEngine,
            ism_store: Arc::new(Mutex::new(IsmStore::new(64))),
            iit_monitor: Mutex::new(IitProxyMonitor::new()),
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
            influence_state: Mutex::new(InfluenceGraphV2::new_default()),
            last_influence_outputs: Mutex::new(None),
            last_influence_root_commit: Mutex::new(None),
            ssm_core: Mutex::new(SsmCore::new(SsmParams::default())),
            last_ssm_output: Mutex::new(None),
            ncde_core: Mutex::new(NcdeCore::new(NcdeParams::default())),
            last_ncde_output: Mutex::new(None),
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

    pub fn pending_neuromod_delta(&self) -> Option<NeuromodDelta> {
        self.pending_neuromod_delta
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
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
                spike_seen_root: Digest32::new([0u8; 32]),
                spike_accepted_root: Digest32::new([0u8; 32]),
                spike_counts: Vec::new(),
                spike_causal_link_count: 0,
                spike_consistency_alert_count: 0,
                spike_thought_only_count: 0,
                spike_output_intent_count: 0,
                spike_cap_hit: false,
                ncde_commit: Digest32::new([0u8; 32]),
                ncde_state_commit: Digest32::new([0u8; 32]),
                ncde_energy: 0,
                cde_commit: Digest32::new([0u8; 32]),
                cde_graph_commit: Digest32::new([0u8; 32]),
                cde_top_edges: Vec::new(),
                ssm_commit: Digest32::new([0u8; 32]),
                ssm_state_commit: Digest32::new([0u8; 32]),
                influence_v2_commit: Digest32::new([0u8; 32]),
                influence_pulses_root: Digest32::new([0u8; 32]),
                influence_node_values: Vec::new(),
                onn_states_commit: Digest32::new([0u8; 32]),
                onn_global_plv: 0,
                onn_pair_locks_commit: Digest32::new([0u8; 32]),
                onn_phase_frame_commit: Digest32::new([0u8; 32]),
                iit_output: None,
                nsr_trace_root: None,
                nsr_prev_commit: None,
                nsr_verdict: None,
                rsa_commit: Digest32::new([0u8; 32]),
                rsa_chosen: None,
                rsa_applied: false,
                rsa_new_params_commit: None,
                sle_commit: Digest32::new([0u8; 32]),
                sle_self_symbol_commit: Digest32::new([0u8; 32]),
                sle_rate_limited: false,
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
        let plan_surprise = self
            .last_surprise
            .lock()
            .ok()
            .and_then(|guard| guard.clone());

        let (cycle_plan, tcf_energy_smooth) = {
            let mut tcf = self.tcf_port.lock().expect("tcf lock");
            let plan = tcf.step(&plan_attn, plan_surprise.as_ref());
            let energy = tcf.state().energy;
            (plan, energy)
        };
        let planned = DeterministicTcf::planned_event(&cycle_plan);
        self.append_cycle_plan_record(&cycle_plan, &planned);

        let cycle_id = cycle_plan.cycle_id;
        let mut ctx = StageContext::new();
        ctx.tcf_energy_smooth = Some(tcf_energy_smooth);

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
                    let surprise_score = ctx
                        .predictive_result
                        .as_ref()
                        .map(|(_, signal)| signal.score)
                        .unwrap_or(0);
                    let drift_score = drift_score_from_snapshot(&workspace_snapshot);
                    let causal_attention_risk = intent.risk;
                    let attention_weights = self
                        .last_attention
                        .lock()
                        .map(|attn| attn.clone())
                        .unwrap_or_else(|_| idle_attention());
                    ctx.attention_weights = Some(attention_weights.clone());
                    let lens_selection = ctx.inference.as_ref().and_then(|inference| {
                        self.translate_features(
                            inference.activation_view.as_ref(),
                            &attention_weights,
                            cycle_id,
                            pulse.slot,
                        )
                    });
                    ctx.lens_selection = lens_selection.clone();
                    let phase_frame = self.latest_phase_frame(cycle_id);
                    ctx.phase_commit = Some(phase_frame.commit);
                    let snn_knobs = self.current_snn_knobs();
                    let phase_window = self.onn_phase_window();
                    let ssm_output = ctx
                        .ssm_output
                        .clone()
                        .or_else(|| self.last_ssm_output.lock().ok().and_then(|g| g.clone()));
                    let (wm_salience, wm_novelty) = ssm_output
                        .as_ref()
                        .map(|output| (output.wm_salience, output.wm_novelty))
                        .unwrap_or((0, 0));
                    let feature_inputs = FeatureSpikerInputs {
                        ssm_salience: wm_salience,
                        ssm_novelty: wm_novelty,
                        wm_salience,
                        wm_novelty,
                        risk: causal_attention_risk,
                        surprise: surprise_score,
                        cde_top_conf: workspace_snapshot
                            .cde_top_edges
                            .iter()
                            .map(|(_, _, conf, _)| *conf)
                            .max(),
                    };
                    let (feature_batch, _) = build_feature_spike_batch(
                        cycle_id,
                        &phase_frame,
                        phase_window,
                        feature_inputs,
                    );
                    let feature_batch_len = feature_batch.events.len();
                    self.append_spike_batch(
                        cycle_id,
                        feature_batch,
                        vec![SpikeSuppression::default(); feature_batch_len],
                        &mut ctx,
                    );
                    let verify_limit = usize::from(snn_knobs.verify_limit.max(1));
                    let nsr_spikes = if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.drain_spikes_for(OscId::Nsr, cycle_id, verify_limit)
                    } else {
                        Vec::new()
                    };
                    let cde_spikes = if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.drain_spikes_for(OscId::Cde, cycle_id, verify_limit)
                    } else {
                        Vec::new()
                    };
                    let nsr_full =
                        spikes_in_phase(&nsr_spikes, phase_frame.global_phase, phase_window);
                    let cde_full =
                        spikes_in_phase(&cde_spikes, phase_frame.global_phase, phase_window);
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
                    let (influence_pulses_root, influence_nodes) = self
                        .last_influence_outputs
                        .lock()
                        .ok()
                        .and_then(|guard| {
                            guard
                                .as_ref()
                                .map(|out| (out.pulses_root, out.node_values.clone()))
                        })
                        .unwrap_or_else(|| (Digest32::new([0u8; 32]), Vec::new()));
                    let phase_frame = self.tick_onn_phase(
                        cycle_id,
                        influence_pulses_root,
                        influence_nodes,
                        ctx.integration_score.unwrap_or(0),
                        ctx.attention_risk,
                        drift_score,
                        plan_surprise
                            .as_ref()
                            .map(|signal| signal.score)
                            .unwrap_or(0),
                        plan_attn.gain,
                        nsr_report.verdict,
                        pulse.slot,
                    );
                    ctx.phase_commit = Some(phase_frame.commit);
                    ctx.coherence_plv = Some(phase_frame.coherence_plv);
                    ctx.onn_outputs = Some(onn_outputs_from_phase(&phase_frame));
                    let mut attention_risk = 0u16;
                    let outputs = &inference.outputs;
                    let mut risk_results = Vec::with_capacity(outputs.len());
                    let mut speech_gate_results = Vec::with_capacity(outputs.len());
                    let tom_report = ctx.tom_report.as_ref().expect("tom report available");
                    for output in outputs {
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
                    let influence_for_output = self
                        .last_influence_outputs
                        .lock()
                        .ok()
                        .and_then(|guard| guard.clone());
                    apply_influence_output_suppression(
                        &mut speech_gate_results,
                        outputs,
                        influence_for_output.as_ref(),
                    );
                    ctx.attention_risk = attention_risk;
                    self.publish_workspace_signals(risk_results.iter().map(|result| {
                        WorkspaceSignal::from_risk_result(result, None, Some(pulse.slot))
                    }));

                    let spike_summary = ctx
                        .spike_summary
                        .clone()
                        .unwrap_or_else(empty_spike_summary);
                    let spike_root_commit = spike_summary.accepted_root;
                    let spike_counts = spike_summary.counts.clone();
                    let spike_counts_iit = spike_counts.clone();
                    let spike_counts_ssm = spike_counts.clone();
                    ctx.replay_pressure = Some(replay_pressure_from_spikes(&spike_counts));
                    let tcf_energy_smooth = ctx.tcf_energy_smooth.unwrap_or(0);
                    if let Some(ncde_output) = self.tick_ncde(
                        cycle_id,
                        &phase_frame,
                        tcf_energy_smooth,
                        influence_pulses_root,
                        spike_root_commit,
                        spike_counts,
                        drift_score,
                        surprise_score,
                        attention_risk,
                    ) {
                        ctx.ncde_output = Some(ncde_output);
                        if let Ok(mut guard) = self.last_ncde_output.lock() {
                            *guard = Some(ncde_output);
                        }
                        if let Ok(mut workspace) = self.workspace.lock() {
                            workspace.set_ncde_snapshot(
                                ncde_output.commit,
                                ncde_output.state_commit,
                                ncde_output.energy,
                            );
                        }
                        self.append_ncde_output_record(cycle_id, &ncde_output);
                        if let Some(ssm_output) = self.tick_ssm(
                            &phase_frame,
                            &ncde_output,
                            influence_pulses_root,
                            spike_root_commit,
                            spike_counts_ssm,
                            drift_score,
                            surprise_score,
                            attention_risk,
                        ) {
                            ctx.ssm_output = Some(ssm_output.clone());
                            if let Ok(mut guard) = self.last_ssm_output.lock() {
                                *guard = Some(ssm_output.clone());
                            }
                            if let Ok(mut workspace) = self.workspace.lock() {
                                workspace.set_ssm_commits(ssm_output.commit, ssm_output.x_commit);
                            }
                            self.append_ssm_output_record(cycle_id, &ssm_output);
                        }
                    }

                    let iit_output = {
                        let ssm_output = ctx
                            .ssm_output
                            .clone()
                            .or_else(|| self.last_ssm_output.lock().ok().and_then(|g| g.clone()));
                        let ncde_output = ctx
                            .ncde_output
                            .or_else(|| self.last_ncde_output.lock().ok().and_then(|g| *g));
                        let inputs = IitInputs::new(
                            cycle_id,
                            phase_frame.commit,
                            phase_frame.coherence_plv,
                            spike_root_commit,
                            spike_counts_iit.clone(),
                            ssm_output
                                .as_ref()
                                .map(|output| output.commit)
                                .unwrap_or_else(|| Digest32::new([0u8; 32])),
                            ssm_output
                                .as_ref()
                                .map(|output| output.wm_salience)
                                .unwrap_or(0),
                            ssm_output
                                .as_ref()
                                .map(|output| output.wm_novelty)
                                .unwrap_or(0),
                            ncde_output
                                .as_ref()
                                .map(|output| output.commit)
                                .unwrap_or_else(|| Digest32::new([0u8; 32])),
                            ncde_output
                                .as_ref()
                                .map(|output| output.energy)
                                .unwrap_or(0),
                            workspace_snapshot.nsr_trace_root,
                            workspace_snapshot.nsr_verdict,
                            self.last_cde_output
                                .lock()
                                .ok()
                                .and_then(|guard| guard.as_ref().map(|out| out.commit)),
                            drift_score,
                            surprise_score,
                            attention_risk,
                        );
                        let mut monitor = self.iit_monitor.lock().expect("iit monitor lock");
                        monitor.tick(&inputs)
                    };
                    let iit_report = report_for_phi(iit_output.phi_proxy);
                    let iit_actions = actions_for_phi(iit_output.phi_proxy, attention_risk);
                    if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.set_iit_output(iit_output.clone());
                    }
                    self.append_iit_output_record(cycle_id, &iit_output);
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
                        .map(|output| output.state_commit)
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
                    let spike_summary = ctx
                        .spike_summary
                        .clone()
                        .unwrap_or_else(empty_spike_summary);
                    let spike_root_commit = spike_summary.accepted_root;
                    let cde_output = self.tick_cde(
                        cycle_id,
                        &phase_frame,
                        spike_root_commit,
                        spike_counts_iit.clone(),
                        influence_outputs.as_ref(),
                        &iit_output,
                        ctx.ssm_output.as_ref(),
                        ctx.ncde_output.as_ref(),
                        drift_score,
                        surprise_score,
                        attention_risk,
                    );
                    if let Some(output) = cde_output.clone() {
                        ctx.cde_output = Some(output.clone());
                        if let Ok(mut guard) = self.last_cde_output.lock() {
                            *guard = Some(output.clone());
                        }
                        if let Ok(mut workspace) = self.workspace.lock() {
                            workspace.set_cde_output(
                                output.commit,
                                output.graph_commit,
                                compress_cde_edges(&output.top_edges),
                            );
                        }
                        self.append_cde_output_record(cycle_id, &output);
                        if output.emit_spikes {
                            self.queue_cde_spikes(
                                cycle_id,
                                &phase_frame,
                                &output,
                                &attention_weights,
                                &mut ctx,
                            );
                        }
                    }
                    let cde_atoms = cde_atoms_from_outputs(cde_output.as_ref());
                    let nsr_thresholds = self
                        .structural_store
                        .lock()
                        .map(|store| store.current.nsr.clone())
                        .unwrap_or_else(|_| NsrThresholds::default());
                    let nsr_inputs = self.build_nsr_trace_inputs(
                        cycle_id,
                        &phase_frame,
                        &iit_output,
                        drift_score,
                        surprise_score,
                        attention_risk,
                        spike_counts_iit.clone(),
                        ctx.ssm_output.as_ref(),
                        ctx.ncde_output.as_ref(),
                        &decision,
                        ctx.self_state.as_ref(),
                        cde_atoms,
                    );
                    let nsr_trace = {
                        let mut engine = self.nsr_engine.lock().expect("nsr engine lock");
                        engine.set_thresholds(nsr_thresholds);
                        engine.evaluate(&nsr_inputs)
                    };
                    ctx.nsr_trace = Some(nsr_trace.clone());
                    if let Ok(mut workspace) = self.workspace.lock() {
                        workspace.set_nsr_trace(
                            nsr_trace.trace_root,
                            nsr_trace.prev_commit,
                            nsr_trace.verdict.as_u8(),
                        );
                    }
                    self.append_nsr_trace_record(cycle_id, &nsr_trace);
                    let nsr_warn_streak = self.update_nsr_warn_streak(nsr_trace.verdict);
                    ctx.nsr_warn_streak = Some(nsr_warn_streak);

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
                    if consistency_report.drift_score > 0 {
                        let phase_window = self.onn_phase_window();
                        let spikes = vec![
                            encode_spike_with_window(
                                cycle_id,
                                SpikeKind::ConsistencyAlert,
                                OscId::Geist,
                                OscId::Geist,
                                consistency_report.drift_score,
                                phase_frame.commit,
                                consistency_report.commit,
                                phase_window,
                            ),
                            encode_spike_with_window(
                                cycle_id,
                                SpikeKind::ConsistencyAlert,
                                OscId::Geist,
                                OscId::Nsr,
                                consistency_report.drift_score,
                                phase_frame.commit,
                                consistency_report.commit,
                                phase_window,
                            ),
                        ];
                        self.append_spike_events(
                            cycle_id,
                            &phase_frame,
                            spikes,
                            true,
                            NsrVerdict::Allow,
                            &mut ctx,
                        );
                    }
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

                    let ism_anchor_commit = anchors.first().map(|anchor| anchor.commit);
                    if let Some(sle_outputs) = self.tick_sle(
                        cycle_id,
                        &phase_frame,
                        &iit_output,
                        drift_score,
                        surprise_score,
                        attention_risk,
                        &decision,
                        ctx.ssm_output.as_ref(),
                        ctx.ncde_output.as_ref(),
                        ctx.self_state.as_ref(),
                        ctx.nsr_trace.as_ref(),
                        ism_anchor_commit,
                    ) {
                        ctx.sle_outputs = Some(sle_outputs);
                    }
                    if let Ok(workspace) = self.workspace.lock() {
                        ctx.spike_summary = Some(workspace.spike_summary());
                    }

                    let policy_ok = decision.kind == DecisionKind::DecisionKindAllow as i32;
                    let nsr_verdict = ctx
                        .nsr_trace
                        .as_ref()
                        .map(|trace| trace.verdict)
                        .unwrap_or(NsrVerdict::Allow);
                    let consistency_ok = !matches!(
                        ctx.consistency_report,
                        Some(ref report) if report.band == DriftBand::Critical
                    );
                    let structural_stats = StructuralCycleStats::new(
                        phase_frame.coherence_plv,
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
                            .nsr_trace
                            .as_ref()
                            .map(|trace| {
                                let codes = trace
                                    .rule_hits
                                    .iter()
                                    .map(|(_, reason, _)| reason.token())
                                    .collect::<Vec<_>>();
                                digest_reasons(&codes)
                            })
                            .unwrap_or_else(|| digest_reasons(&[])),
                    };
                    let coherence_threshold = if nsr_verdict == NsrVerdict::Warn {
                        ONN_COHERENCE_THROTTLE_WARN
                    } else {
                        ONN_COHERENCE_THROTTLE
                    };
                    let output_lock =
                        pair_lock_for(&phase_frame, OscId::Output, OscId::Nsr).unwrap_or(0);
                    let gates = GateBundle {
                        policy_decision: decision.clone(),
                        sandbox: ctx.sandbox_verdict.clone().unwrap_or(SandboxVerdict::Allow),
                        risk_results,
                        nsr_summary,
                        speech_gate: speech_gate_results,
                        coherence_plv: phase_frame.coherence_plv,
                        coherence_threshold,
                        phi_proxy: iit_output.phi_proxy,
                        phi_threshold: PHI_OUTPUT_THRESHOLD,
                        speak_lock: output_lock,
                        speak_lock_min: LOCK_MIN_SPEAK,
                    };
                    let mut output_router = self.output_router.lock().expect("output router lock");
                    if let Some(budget) = ctx.recursion_budget.as_ref() {
                        output_router.apply_recursion_budget(budget);
                    }
                    output_router.apply_coherence(phase_frame.coherence_plv, coherence_threshold);
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

                    let mut output_intent_events = Vec::new();
                    let mut output_intent_suppressions = Vec::new();
                    for (idx, output) in outputs.iter().enumerate() {
                        match output.channel {
                            OutputChannel::Thought => ctx.thought_outputs.push(output.clone()),
                            OutputChannel::Speech => {
                                if decisions
                                    .get(idx)
                                    .map(|decision| decision.permitted)
                                    .unwrap_or(false)
                                {
                                    let strength = gates
                                        .risk_results
                                        .get(idx)
                                        .map(|result| result.risk)
                                        .unwrap_or(0);
                                    let spike = encode_spike_with_window(
                                        cycle_id,
                                        SpikeKind::OutputIntent,
                                        OscId::Output,
                                        OscId::Nsr,
                                        strength,
                                        phase_frame.commit,
                                        output
                                            .rationale_commit
                                            .unwrap_or_else(|| Digest32::new([0u8; 32])),
                                        self.onn_phase_window(),
                                    );
                                    let suppression = self.suppression_for_event(
                                        &phase_frame,
                                        &spike,
                                        policy_ok,
                                        nsr_verdict,
                                    );
                                    output_intent_events.push(spike);
                                    output_intent_suppressions.push(suppression);
                                    if suppression.is_suppressed() {
                                        if let Some(result) = gates.risk_results.get(idx) {
                                            let reason_digest = digest_reasons(&[
                                                "output_intent_suppressed".to_string(),
                                            ]);
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
                                    } else {
                                        ctx.speech_outputs.push(output.clone());
                                    }
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
                    if !output_intent_events.is_empty() {
                        let batch =
                            SpikeBatch::new(cycle_id, phase_frame.commit, output_intent_events);
                        let summary = if let Ok(mut workspace) = self.workspace.lock() {
                            workspace.append_spike_batch(batch.clone(), output_intent_suppressions)
                        } else {
                            empty_spike_summary()
                        };
                        ctx.spike_summary = Some(summary.clone());
                        self.append_spike_bus_record(cycle_id, &summary, batch.commit);
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
                    let surprise_score = ctx
                        .predictive_result
                        .as_ref()
                        .map(|(_, signal)| signal.score)
                        .unwrap_or(0);
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
                        .map(|output| output.wm_salience)
                        .unwrap_or(0);
                    let ssm_novelty = ctx
                        .ssm_output
                        .as_ref()
                        .map(|output| output.wm_novelty)
                        .unwrap_or(0);
                    let ncde_energy = ctx
                        .ncde_output
                        .as_ref()
                        .map(|output| output.energy)
                        .unwrap_or(0);
                    let ncde_commit = ctx
                        .ncde_output
                        .as_ref()
                        .map(|output| output.state_commit)
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let ncde_state_digest = ctx
                        .ncde_output
                        .as_ref()
                        .map(|output| output.state_digest)
                        .unwrap_or_else(|| Digest32::new([0u8; 32]));
                    let nsr_verdict = ctx
                        .nsr_report
                        .as_ref()
                        .map(|report| report.verdict.as_u8())
                        .unwrap_or(0);
                    let coherence_plv = ctx.coherence_plv.unwrap_or(0);
                    let phi_proxy = ctx.integration_score.unwrap_or(0);
                    let cde_commit = ctx.cde_output.as_ref().map(|output| output.commit);
                    let sle_self_symbol = ctx
                        .sle_outputs
                        .as_ref()
                        .map(|output| output.self_symbol_commit);
                    let rsa_applied = self
                        .workspace
                        .lock()
                        .ok()
                        .map(|workspace| workspace.rsa_applied())
                        .unwrap_or(false);
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
                        ctx.replay_pressure =
                            Some(apply_influence_replay_pressure(base_pressure, outputs));
                    }
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
                        ssm_salience: ctx.ssm_output.as_ref().map(|output| output.wm_salience),
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
                    ctx.evidence_id = Some(evidence_id);
                }
                PulseKind::Broadcast => {
                    let snapshot = self.arbitrate_workspace(cycle_id);
                    self.append_workspace_snapshot_record(&snapshot);
                    if let Ok(mut guard) = self.last_workspace_snapshot.lock() {
                        *guard = Some(snapshot.clone());
                    }
                    ctx.workspace_snapshot_commit = Some(snapshot.commit);
                }
                PulseKind::Sleep => {
                    if let Ok(mut store) = self.structural_store.lock() {
                        let params = store.current.clone();
                        let nsr_verdict = ctx
                            .nsr_trace
                            .as_ref()
                            .map(|trace| trace.verdict.as_u8())
                            .unwrap_or(0);
                        let nsr_warn_streak = ctx.nsr_warn_streak.unwrap_or(0).min(63) as u8;
                        let nsr_encoded = (nsr_warn_streak << 2) | (nsr_verdict & 0b11);
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
                        let sleep_drive = influence_outputs
                            .as_ref()
                            .map(|outputs| outputs.node_value(InfluenceNodeId::SleepDrive))
                            .unwrap_or(0);
                        let phi_proxy = ctx.integration_score.unwrap_or(0);
                        let ncde_energy = ctx
                            .ncde_output
                            .as_ref()
                            .map(|output| output.energy)
                            .unwrap_or(0);
                        let sleep_active = derive_sleep_active(
                            matches!(pulse.kind, PulseKind::Sleep),
                            replay_pressure,
                            phi_proxy,
                            sleep_drive,
                            ncde_energy,
                            &params,
                        );
                        let inputs = RsaInputs::new(
                            cycle_id,
                            sleep_active,
                            ctx.phase_commit.unwrap_or_else(|| Digest32::new([0u8; 32])),
                            ctx.coherence_plv.unwrap_or(0),
                            phi_proxy,
                            nsr_encoded,
                            ctx.nsr_trace
                                .as_ref()
                                .map(|trace| trace.trace_root)
                                .unwrap_or_else(|| Digest32::new([0u8; 32])),
                            policy_ok,
                            ctx.attention_risk,
                            ctx.drift_score.unwrap_or(0),
                            ctx.surprise_score.unwrap_or(0),
                            replay_pressure,
                            ctx.ssm_output
                                .as_ref()
                                .map(|output| output.wm_salience)
                                .unwrap_or(0),
                            ncde_energy,
                            params.commit,
                        );
                        let mut rsa_core =
                            self.rsa_core.lock().unwrap_or_else(|err| err.into_inner());
                        let mut outputs = rsa_core.tick_with_params(&inputs, &params);
                        if let Some(influence) = influence_outputs.as_ref() {
                            let bias =
                                influence.node_value(InfluenceNodeId::StructuralPlasticity) / 4;
                            let chosen = choose_best_proposal_with_bias(&outputs.proposals, bias);
                            outputs.chosen = chosen;
                            outputs.recompute_commit();
                        }
                        let mut applied = false;
                        let mut new_params_commit = None;
                        if let Some(chosen_commit) = outputs.chosen {
                            if let Some(chosen) = outputs
                                .proposals
                                .iter()
                                .find(|proposal| proposal.commit == chosen_commit)
                            {
                                let score =
                                    i32::from(chosen.expected_gain) - i32::from(chosen.risk_cost);
                                if score > 0
                                    && rsa_core.can_apply(cycle_id)
                                    && rsa_gates_allow(&inputs, &params)
                                {
                                    let deltas = chosen
                                        .deltas
                                        .iter()
                                        .map(|delta| ParamDeltaRef {
                                            key: delta.key,
                                            from: delta.from,
                                            to: delta.to,
                                        })
                                        .collect::<Vec<_>>();
                                    if let Some(updated) = store.apply_deltas(&deltas) {
                                        let commit = updated.commit;
                                        store.apply_params(updated);
                                        applied = true;
                                        new_params_commit = Some(commit);
                                        rsa_core.mark_applied(cycle_id);
                                    }
                                }
                            }
                        }
                        outputs.applied = applied;
                        outputs.new_params_commit = new_params_commit;
                        outputs.recompute_commit();
                        if let Ok(mut workspace) = self.workspace.lock() {
                            workspace.set_rsa_output(
                                outputs.commit,
                                outputs.chosen,
                                outputs.applied,
                                outputs.new_params_commit,
                            );
                        }
                        self.append_rsa_outputs_record(cycle_id, &outputs);
                    }
                    let phi = ctx.integration_score.unwrap_or(0);
                    let surprise_score = ctx
                        .predictive_result
                        .as_ref()
                        .map(|(_, signal)| signal.score)
                        .unwrap_or(0);
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
        let chosen = outputs
            .chosen
            .map(|commit| commit.to_string())
            .unwrap_or_else(|| "none".to_string());
        let new_params = outputs
            .new_params_commit
            .map(|commit| commit.to_string())
            .unwrap_or_else(|| "none".to_string());
        let payload = format!(
            "rsa_commit={};proposal_count={};chosen={};applied={};new_params_commit={}",
            outputs.commit,
            outputs.proposals.len(),
            chosen,
            outputs.applied as u8,
            new_params
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

    fn append_phase_frame_record(&self, cycle_id: u64, frame: &PhaseFrame) {
        let locked_pairs = frame
            .pair_locks
            .iter()
            .filter(|pair| pair.lock >= LOCK_MIN_CAUSAL)
            .count();
        let meta = RecordMeta {
            cycle_id,
            tier: locked_pairs.min(u8::MAX as usize) as u8,
            flags: frame.coherence_plv,
            boundary_commit: frame.phase_frame_commit,
        };
        self.append_archive_record(
            RecordKind::Other(PHASE_FRAME_RECORD_KIND),
            frame.commit,
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
            "commit={};phi_proxy={};coupling_proxy={};coherence={}",
            output.commit, output.phi_proxy, output.coupling_proxy, output.coherence
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

    #[allow(clippy::too_many_arguments)]
    fn tick_onn_phase(
        &self,
        cycle_id: u64,
        influence_pulses_root: Digest32,
        influence_nodes: Vec<(InfluenceNodeId, i16)>,
        coherence_hint: u16,
        risk: u16,
        drift_score: u16,
        surprise_score: u16,
        attn_gain: u16,
        nsr_verdict: NsrVerdict,
        slot: u8,
    ) -> PhaseFrame {
        let brain_arousal = self
            .last_brain_arousal
            .lock()
            .ok()
            .map(|value| *value)
            .unwrap_or(0);
        self.sync_onn_params();
        let mut nodes = influence_nodes;
        nodes.push((InfluenceNodeId::BlueBrainArousal, brain_arousal as i16));
        nodes.push((InfluenceNodeId::AttentionGain, attn_gain as i16));
        nodes.push((InfluenceNodeId::Surprise, surprise_score as i16));
        let inputs = OnnInputs::new(
            cycle_id,
            influence_pulses_root,
            nodes,
            coherence_hint,
            risk,
            drift_score,
            nsr_verdict.as_u8(),
        );
        let outputs = {
            let mut onn = self.onn_core.lock().expect("onn core lock");
            onn.tick(&inputs)
        };
        let module_phase = self
            .onn_core
            .lock()
            .map(|core| core.states.iter().map(|s| (s.id, s.phase)).collect())
            .unwrap_or_else(|_| Vec::new());
        let global_phase = module_phase
            .iter()
            .find_map(|(id, phase)| (*id == OscId::Iit).then_some(*phase))
            .unwrap_or(0);
        let frame = PhaseFrame {
            cycle_id,
            global_phase,
            module_phase,
            coherence_plv: outputs.global_plv,
            pair_locks: outputs.pair_locks.clone(),
            states_commit: outputs.states_commit,
            phase_frame_commit: outputs.phase_frame_commit,
            commit: outputs.commit,
        };
        let priority = 3200u16.saturating_add(frame.coherence_plv / 5).min(10_000);
        let summary = format!(
            "PHASE COH={} PLV={}",
            frame.coherence_plv, outputs.global_plv
        );
        self.publish_workspace_signal(WorkspaceSignal {
            kind: SignalKind::Brain,
            priority,
            digest: frame.commit,
            summary,
            slot,
        });
        if let Ok(mut workspace) = self.workspace.lock() {
            let pair_commit = pair_locks_commit(&frame.pair_locks);
            workspace.set_onn_snapshot(
                frame.states_commit,
                frame.coherence_plv,
                pair_commit,
                frame.phase_frame_commit,
            );
        }
        self.append_phase_frame_record(cycle_id, &frame);
        frame
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
        phase_frame: &PhaseFrame,
        iit_output: &IitOutput,
        drift_score: u16,
        surprise_score: u16,
        attention_risk: u16,
        decision: &PolicyDecision,
        ssm_output: Option<&WmSsmOutput>,
        ncde_output: Option<&NcdeOutputs>,
        self_state: Option<&SelfState>,
        nsr_trace: Option<&ReasoningTrace>,
        ism_anchor_commit: Option<Digest32>,
    ) -> Option<SleOutputs> {
        let ssm_commit = ssm_output
            .map(|output| output.commit)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let wm_salience = ssm_output.map(|output| output.wm_salience).unwrap_or(0);
        let wm_novelty = ssm_output.map(|output| output.wm_novelty).unwrap_or(0);
        let ncde_commit = ncde_output
            .map(|output| output.state_commit)
            .unwrap_or_else(|| Digest32::new([0u8; 32]));
        let ncde_energy = ncde_output.map(|output| output.energy).unwrap_or(0);
        let geist_commit = self_state.map(|state| state.commit);
        let geist_consistency_ok =
            self_state.map(|state| state.consistency >= SELF_CONSISTENCY_OK_THRESHOLD);
        let nsr_trace_root = nsr_trace.map(|trace| trace.trace_root);
        let nsr_verdict = nsr_trace.map(|trace| trace.verdict.as_u8());
        let policy_decision_commit = if decision.kind == DecisionKind::DecisionKindDeny as i32 {
            Some(commit_policy_decision(decision).digest)
        } else {
            None
        };

        let inputs = SleInputs::new(
            cycle_id,
            phase_frame.commit,
            phase_frame.coherence_plv,
            iit_output.phi_proxy,
            ssm_commit,
            wm_salience,
            wm_novelty,
            ncde_commit,
            ncde_energy,
            geist_commit,
            geist_consistency_ok,
            ism_anchor_commit,
            nsr_trace_root,
            nsr_verdict,
            policy_decision_commit,
            attention_risk,
            drift_score,
            surprise_score,
        );

        let outputs = self
            .sle_core
            .lock()
            .map(|mut core| core.tick(&inputs))
            .ok()?;

        let mut spike_record = None;
        if let Ok(mut workspace) = self.workspace.lock() {
            workspace.set_sle_outputs(
                outputs.commit,
                outputs.self_symbol_commit,
                outputs.rate_limited,
            );
            if let Some(pulse) = outputs
                .stimuli
                .iter()
                .find(|stim| matches!(stim.kind, SleStimulusKind::ThoughtOnlyPulse))
            {
                let severity = pulse.value.unsigned_abs();
                workspace.push_internal_utterance(InternalUtterance::new(
                    outputs.self_symbol_commit,
                    severity,
                ));
                let amplitude = severity.min(10_000);
                let phase_window = self.onn_phase_window();
                let spikes = vec![
                    encode_spike_with_window(
                        cycle_id,
                        SpikeKind::ThoughtOnly,
                        OscId::Jepa,
                        OscId::BlueBrain,
                        amplitude,
                        phase_frame.commit,
                        pulse.commit,
                        phase_window,
                    ),
                    encode_spike_with_window(
                        cycle_id,
                        SpikeKind::ThoughtOnly,
                        OscId::Jepa,
                        OscId::Geist,
                        amplitude,
                        phase_frame.commit,
                        pulse.commit,
                        phase_window,
                    ),
                ];
                let batch = SpikeBatch::new(cycle_id, phase_frame.commit, spikes);
                let summary = workspace.append_spike_batch(
                    batch.clone(),
                    vec![SpikeSuppression::default(); batch.events.len()],
                );
                spike_record = Some((summary, batch.commit));
            }
        }
        if let Some((summary, batch_commit)) = spike_record {
            self.append_spike_bus_record(cycle_id, &summary, batch_commit);
        }

        self.append_sle_outputs_record(cycle_id, &outputs);
        Some(outputs)
    }

    fn append_sle_outputs_record(&self, cycle_id: u64, outputs: &SleOutputs) {
        let record_id = format!("sle-{}", cycle_id);
        let mut payload = Vec::with_capacity(Digest32::LEN * 2 + 2 + 1);
        payload.extend_from_slice(outputs.commit.as_bytes());
        payload.extend_from_slice(outputs.self_symbol_commit.as_bytes());
        payload.extend_from_slice(
            &u16::try_from(outputs.stimuli.len())
                .unwrap_or(u16::MAX)
                .to_be_bytes(),
        );
        payload.push(outputs.rate_limited as u8);
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
        Some(self.apply_ssm_attention_bias(weights, ctx.ssm_salience))
    }

    fn apply_ncde_attention_bias(&self, mut weights: AttentionWeights) -> AttentionWeights {
        let energy = self
            .last_ncde_output
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|output| output.energy));
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

    fn apply_ssm_attention_bias(
        &self,
        mut weights: AttentionWeights,
        wm_salience: Option<u16>,
    ) -> AttentionWeights {
        let Some(wm_salience) = wm_salience else {
            return weights;
        };
        if wm_salience == 0 {
            return weights;
        }
        let multiplier = (10_000u32 + u32::from(wm_salience) / 5).min(12_000);
        let scaled = (u32::from(weights.gain).saturating_mul(multiplier)) / 10_000;
        let updated = clamp_u16(scaled);
        if updated != weights.gain {
            weights.gain = updated;
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

    fn append_spike_bus_record(
        &self,
        cycle_id: u64,
        summary: &SpikeBusSummary,
        batch_commit: Digest32,
    ) {
        let counts = summarize_spike_counts(&summary.counts);
        let payload_commit = spike_record_commit(
            summary.accepted_root,
            summary.seen_root,
            counts.total,
            counts.top_kinds.as_slice(),
            summary.cap_hit,
        );
        let meta = RecordMeta {
            cycle_id,
            tier: counts.total.min(u8::MAX as u16) as u8,
            flags: summary.cap_hit as u16,
            boundary_commit: batch_commit,
        };
        self.append_archive_record(RecordKind::Other(SPIKE_RECORD_KIND), payload_commit, meta);
    }

    fn append_spike_batch(
        &self,
        cycle_id: u64,
        batch: SpikeBatch,
        suppressions: Vec<SpikeSuppression>,
        ctx: &mut StageContext,
    ) -> SpikeBusSummary {
        let summary = if let Ok(mut workspace) = self.workspace.lock() {
            workspace.append_spike_batch(batch.clone(), suppressions)
        } else {
            empty_spike_summary()
        };
        ctx.spike_summary = Some(summary.clone());
        self.append_spike_bus_record(cycle_id, &summary, batch.commit);
        summary
    }

    fn append_spike_events(
        &self,
        cycle_id: u64,
        phase_frame: &PhaseFrame,
        events: Vec<SpikeEvent>,
        policy_ok: bool,
        nsr_verdict: NsrVerdict,
        ctx: &mut StageContext,
    ) -> SpikeBusSummary {
        if events.is_empty() {
            return ctx
                .spike_summary
                .clone()
                .unwrap_or_else(empty_spike_summary);
        }
        let suppressions = events
            .iter()
            .map(|event| self.suppression_for_event(phase_frame, event, policy_ok, nsr_verdict))
            .collect::<Vec<_>>();
        let batch = SpikeBatch::new(cycle_id, phase_frame.commit, events);
        self.append_spike_batch(cycle_id, batch, suppressions, ctx)
    }

    fn suppression_for_event(
        &self,
        phase_frame: &PhaseFrame,
        event: &SpikeEvent,
        policy_ok: bool,
        nsr_verdict: NsrVerdict,
    ) -> SpikeSuppression {
        let mut suppression = SpikeSuppression::default();
        match event.kind {
            SpikeKind::CausalLink if event.dst == OscId::Nsr => {
                let lock = pair_lock_for(phase_frame, OscId::Cde, OscId::Nsr).unwrap_or(0);
                suppression.suppressed_by_onn = lock < LOCK_MIN_CAUSAL;
            }
            SpikeKind::ConsistencyAlert if event.dst == OscId::Nsr => {
                let lock = pair_lock_for(phase_frame, OscId::Geist, OscId::Iit).unwrap_or(0);
                suppression.suppressed_by_onn = lock < LOCK_MIN_CONSIST;
            }
            SpikeKind::OutputIntent => {
                let lock = pair_lock_for(phase_frame, OscId::Output, OscId::Nsr).unwrap_or(0);
                suppression.suppressed_by_onn = lock < LOCK_MIN_SPEAK;
                suppression.suppressed_by_policy =
                    !policy_ok || !matches!(nsr_verdict, NsrVerdict::Allow);
            }
            _ => {}
        }
        suppression
    }

    fn append_ncde_output_record(&self, cycle_id: u64, output: &NcdeOutputs) {
        let payload = format!(
            "commit={};energy={};state_digest={}",
            output.commit, output.energy, output.state_digest
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

    fn append_cde_output_record(&self, cycle_id: u64, output: &CdeOutputs) {
        let stable_edges = output
            .top_edges
            .iter()
            .filter(|(_, _, conf, _)| *conf >= NSR_ATOM_MIN)
            .count()
            .min(u16::MAX as usize) as u16;
        let payload = format!(
            "commit={};graph={};stable_edges={};interventions={}",
            output.commit,
            output.graph_commit,
            stable_edges,
            output.interventions.len()
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
            tier: stable_edges.min(u8::MAX as u16) as u8,
            flags: output.interventions.len().min(u16::MAX as usize) as u16,
            boundary_commit: output.commit,
        };
        self.append_archive_record(
            RecordKind::Other(CDE_OUTPUT_RECORD_KIND),
            output.commit,
            meta,
        );
    }

    fn append_ssm_output_record(&self, cycle_id: u64, output: &WmSsmOutput) {
        let payload = format!(
            "commit={};x_commit={};salience={};novelty={}",
            output.commit, output.x_commit, output.wm_salience, output.wm_novelty
        )
        .into_bytes();
        let record_id = format!("ssm-{cycle_id}-{}", hex::encode(output.commit.as_bytes()));
        let record = build_compact_record(record_id, cycle_id, "ssm", payload);
        self.archive.append(record);

        let meta = RecordMeta {
            cycle_id,
            tier: 0,
            flags: output.wm_salience,
            boundary_commit: output.x_commit,
        };
        self.append_archive_record(RecordKind::Other(SSM_RECORD_KIND), output.commit, meta);
    }

    #[allow(clippy::too_many_arguments)]
    fn tick_cde(
        &self,
        cycle_id: u64,
        phase_frame: &PhaseFrame,
        spike_root_commit: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        influence_outputs: Option<&InfluenceOutputs>,
        iit_output: &IitOutput,
        ssm_output: Option<&WmSsmOutput>,
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
        let inputs = CdeInputs::new(
            cycle_id,
            phase_frame.commit,
            phase_frame.global_phase,
            phase_frame.coherence_plv,
            iit_output.phi_proxy,
            spike_root_commit,
            spike_counts,
            influence_commit,
            influence_node_in,
            ssm_output.map(|output| output.wm_salience).unwrap_or(0),
            ncde_output.map(|output| output.energy).unwrap_or(0),
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
        phase_frame: &PhaseFrame,
        tcf_energy_smooth: u16,
        influence_pulses_root: Digest32,
        spike_root_commit: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        drift: u16,
        surprise: u16,
        risk: u16,
    ) -> Option<NcdeOutputs> {
        self.sync_ncde_params();
        let mut core = self.ncde_core.lock().ok()?;
        let inputs = NcdeInputs::new(
            cycle_id,
            phase_frame.commit,
            phase_frame.global_phase,
            tcf_energy_smooth,
            spike_root_commit,
            spike_counts,
            influence_pulses_root,
            phase_frame.coherence_plv,
            risk,
            drift,
            surprise,
        );
        Some(core.tick(&inputs))
    }

    #[allow(clippy::too_many_arguments)]
    fn tick_ssm(
        &self,
        phase_frame: &PhaseFrame,
        ncde_output: &NcdeOutputs,
        influence_pulses_root: Digest32,
        spike_root_commit: Digest32,
        spike_counts: Vec<(SpikeKind, u16)>,
        drift: u16,
        surprise: u16,
        risk: u16,
    ) -> Option<WmSsmOutput> {
        self.sync_ssm_params();
        let mut core = self.ssm_core.lock().ok()?;
        let ncde_summary = ncde_summary_from_digest(ncde_output.state_digest);
        let input = WmSsmInput::new(
            ncde_output.cycle_id,
            phase_frame.commit,
            phase_frame.global_phase,
            phase_frame.coherence_plv,
            influence_pulses_root,
            ncde_output.state_commit,
            ncde_output.energy,
            ncde_summary,
            spike_root_commit,
            spike_counts,
            drift,
            surprise,
            risk,
        );
        Some(core.tick(&input))
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

    fn latest_phase_frame(&self, cycle_id: u64) -> PhaseFrame {
        self.onn_core
            .lock()
            .ok()
            .map(|core| {
                let module_phase = core
                    .states
                    .iter()
                    .map(|state| (state.id, state.phase))
                    .collect::<Vec<_>>();
                let global_phase = module_phase
                    .iter()
                    .find_map(|(id, phase)| (*id == OscId::Iit).then_some(*phase))
                    .unwrap_or(0);
                PhaseFrame {
                    cycle_id,
                    global_phase,
                    module_phase,
                    coherence_plv: 0,
                    pair_locks: Vec::new(),
                    states_commit: Digest32::new([0u8; 32]),
                    phase_frame_commit: Digest32::new([0u8; 32]),
                    commit: Digest32::new([0u8; 32]),
                }
            })
            .unwrap_or(PhaseFrame {
                cycle_id,
                global_phase: 0,
                module_phase: Vec::new(),
                coherence_plv: 0,
                pair_locks: Vec::new(),
                states_commit: Digest32::new([0u8; 32]),
                phase_frame_commit: Digest32::new([0u8; 32]),
                commit: Digest32::new([0u8; 32]),
            })
    }

    fn onn_phase_window(&self) -> u16 {
        self.sync_onn_params();
        self.onn_core
            .lock()
            .map(|core| core.params.lock_window)
            .unwrap_or(0)
    }

    fn sync_onn_params(&self) {
        let knobs = match self.structural_store.lock() {
            Ok(store) => store.current.onn.clone(),
            Err(_) => return,
        };
        if let Ok(mut core) = self.onn_core.lock() {
            if core.params.commit != knobs.commit {
                core.params = ucf_onn::OnnParams::new(
                    knobs.base_step,
                    knobs.coupling,
                    knobs.max_delta,
                    knobs.lock_window,
                );
            }
        }
    }

    fn sync_ssm_params(&self) {
        let params = match self.structural_store.lock() {
            Ok(store) => store.current.ssm,
            Err(_) => return,
        };
        if let Ok(mut core) = self.ssm_core.lock() {
            if core.params.commit != params.commit {
                let updated = params;
                core.params = updated;
                core.state.reset_if_dim_mismatch(&updated);
            }
        }
    }

    fn sync_ncde_params(&self) {
        let params = match self.structural_store.lock() {
            Ok(store) => store.current.ncde,
            Err(_) => return,
        };
        if let Ok(mut core) = self.ncde_core.lock() {
            if core.params.commit != params.commit {
                *core = NcdeCore::new(params);
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
        if verdict == NsrVerdict::Warn {
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
    fn build_nsr_trace_inputs(
        &self,
        cycle_id: u64,
        phase_frame: &PhaseFrame,
        iit_output: &IitOutput,
        drift_score: u16,
        surprise_score: u16,
        risk_score: u16,
        spike_counts: Vec<(SpikeKind, u16)>,
        ssm_output: Option<&WmSsmOutput>,
        ncde_output: Option<&NcdeOutputs>,
        decision: &PolicyDecision,
        self_state: Option<&SelfState>,
        reasoning_atoms: Vec<(u16, i16)>,
    ) -> NsrInputs {
        let policy_decision_commit = if decision.kind == DecisionKind::DecisionKindDeny as i32 {
            Some(commit_policy_decision(decision).digest)
        } else {
            None
        };
        let self_consistency_ok =
            self_state.map(|state| state.consistency >= SELF_CONSISTENCY_OK_THRESHOLD);
        NsrInputs::new(
            cycle_id,
            phase_frame.commit,
            phase_frame.coherence_plv,
            iit_output.phi_proxy,
            risk_score,
            drift_score,
            surprise_score,
            ssm_output.map(|output| output.wm_salience).unwrap_or(0),
            ncde_output.map(|output| output.energy).unwrap_or(0),
            spike_counts,
            policy_decision_commit,
            self_consistency_ok,
            reasoning_atoms,
        )
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

    fn append_nsr_trace_record(&self, cycle_id: u64, trace: &ReasoningTrace) {
        let payload = format!(
            "trace_root={};verdict={};hits={}",
            trace.trace_root,
            nsr_verdict_token(trace.verdict),
            trace.rule_hits.len()
        )
        .into_bytes();
        let record_id = format!(
            "nsr-trace-{cycle_id}-{}",
            hex::encode(trace.trace_root.as_bytes())
        );
        let record = build_compact_record(record_id, cycle_id, "nsr-trace", payload);
        self.archive.append(record);
    }

    fn queue_cde_spikes(
        &self,
        cycle_id: u64,
        phase_frame: &PhaseFrame,
        output: &CdeOutputs,
        _attention_weights: &AttentionWeights,
        ctx: &mut StageContext,
    ) {
        let phase_window = self.onn_phase_window();
        let spikes = cde_spikes_from_outputs(output, phase_frame, phase_window);
        if spikes.is_empty() {
            return;
        }
        self.append_spike_events(cycle_id, phase_frame, spikes, true, NsrVerdict::Allow, ctx);
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

fn ncde_summary_from_digest(digest: Digest32) -> Vec<i16> {
    digest
        .as_bytes()
        .chunks_exact(2)
        .take(4)
        .map(|chunk| i16::from_be_bytes([chunk[0], chunk[1]]))
        .collect()
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

fn compress_cde_edges(edges: &[(CdeNodeId, CdeNodeId, u16, u8)]) -> Vec<(u16, u16, u16, u8)> {
    edges
        .iter()
        .take(8)
        .map(|(from, to, conf, lag)| (from.to_u16(), to.to_u16(), *conf, *lag))
        .collect()
}

fn cde_atoms_from_outputs(output: Option<&CdeOutputs>) -> Vec<(u16, i16)> {
    let Some(output) = output else {
        return Vec::new();
    };
    output
        .top_edges
        .iter()
        .filter(|(_, _, conf, _)| *conf >= NSR_ATOM_MIN)
        .take(8)
        .map(|(from, to, conf, _)| {
            let value = (*conf).min(i16::MAX as u16) as i16;
            (edge_key(*from, *to), value)
        })
        .collect()
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

fn cde_spikes_from_outputs(
    output: &CdeOutputs,
    phase_frame: &PhaseFrame,
    phase_window: u16,
) -> Vec<SpikeEvent> {
    let mut spikes = Vec::new();
    for edge in &output.top_edges {
        if !cde_edge_is_spike_candidate(edge) {
            continue;
        }
        let payload_commit = cde_spike_payload_commit(edge, &output.interventions, output.commit);
        let amplitude = edge.2;
        for dst in [OscId::Cde, OscId::Nsr] {
            spikes.push(encode_spike_with_window(
                output.cycle_id,
                SpikeKind::CausalLink,
                OscId::Cde,
                dst,
                amplitude,
                phase_frame.commit,
                payload_commit,
                phase_window,
            ));
        }
    }
    spikes
}

fn cde_edge_is_spike_candidate(edge: &(CdeNodeId, CdeNodeId, u16, u8)) -> bool {
    let (from, to, conf, _) = *edge;
    conf >= ucf_cde_scm::CONF_SPIKE
        && ((from == CdeNodeId::Risk && to == CdeNodeId::OutputSuppression)
            || (from == CdeNodeId::Drift && to == CdeNodeId::ReplayPressure))
}

fn cde_spike_payload_commit(
    edge: &(CdeNodeId, CdeNodeId, u16, u8),
    interventions: &[Intervention],
    output_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.cde.spike.payload.v1");
    hasher.update(output_commit.as_bytes());
    hasher.update(&edge.0.to_u16().to_be_bytes());
    hasher.update(&edge.1.to_u16().to_be_bytes());
    hasher.update(&edge.2.to_be_bytes());
    hasher.update(&[edge.3]);
    hasher.update(
        &u64::try_from(interventions.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for intervention in interventions {
        hasher.update(intervention.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn spikes_in_phase(spikes: &[SpikeEvent], global_phase: u16, phase_window: u16) -> bool {
    spikes
        .iter()
        .any(|spike| in_phase(spike.ttfs, global_phase, phase_window))
}

fn in_phase(ttfs_code: u16, global_phase: u16, phase_window: u16) -> bool {
    let diff = u32::from(ttfs_code.abs_diff(global_phase));
    let wrap = 65_536u32.saturating_sub(diff);
    let distance = diff.min(wrap);
    distance <= u32::from(phase_window)
}

fn onn_outputs_from_phase(phase_frame: &PhaseFrame) -> ucf_onn::OnnOutputs {
    ucf_onn::OnnOutputs {
        cycle_id: phase_frame.cycle_id,
        states_commit: phase_frame.states_commit,
        global_plv: phase_frame.coherence_plv,
        pair_locks: phase_frame.pair_locks.clone(),
        phase_frame_commit: phase_frame.phase_frame_commit,
        commit: phase_frame.commit,
    }
}

fn pair_lock_for(phase_frame: &PhaseFrame, a: OscId, b: OscId) -> Option<u16> {
    phase_frame.pair_locks.iter().find_map(|pair| {
        ((pair.a == a && pair.b == b) || (pair.a == b && pair.b == a)).then_some(pair.lock)
    })
}

fn pair_locks_commit(pairs: &[ucf_onn::PairLock]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.onn.pair_locks.root.v2");
    hasher.update(&u32::try_from(pairs.len()).unwrap_or(u32::MAX).to_be_bytes());
    for pair in pairs {
        hasher.update(pair.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
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

fn empty_spike_summary() -> SpikeBusSummary {
    SpikeBusSummary {
        seen_root: Digest32::new([0u8; 32]),
        accepted_root: Digest32::new([0u8; 32]),
        counts: Vec::new(),
        causal_link_count: 0,
        consistency_alert_count: 0,
        thought_only_count: 0,
        output_intent_count: 0,
        cap_hit: false,
    }
}

fn replay_pressure_from_spikes(spike_counts: &[(SpikeKind, u16)]) -> u16 {
    spike_counts
        .iter()
        .find_map(|(kind, count)| (*kind == SpikeKind::ReplayCue).then_some(*count))
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
    seen_root: Digest32,
    count: u16,
    top_kinds: &[SpikeKind],
    cap_hit: bool,
) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ucf.spikebus.record.v1");
    hasher.update(accepted_root.as_bytes());
    hasher.update(seen_root.as_bytes());
    hasher.update(&count.to_be_bytes());
    hasher.update(&[cap_hit as u8]);
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

fn choose_best_proposal_with_bias(
    proposals: &[ucf_rsa::StructuralProposal],
    bias: i16,
) -> Option<Digest32> {
    if proposals.is_empty() {
        return None;
    }
    let bias = bias.clamp(-2000, 2000);
    proposals
        .iter()
        .max_by(|left, right| {
            let left_score =
                i32::from(left.expected_gain) + i32::from(bias) - i32::from(left.risk_cost);
            let right_score =
                i32::from(right.expected_gain) + i32::from(bias) - i32::from(right.risk_cost);
            left_score
                .cmp(&right_score)
                .then_with(|| left.commit.as_bytes().cmp(right.commit.as_bytes()))
        })
        .map(|proposal| proposal.commit)
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
        NsrVerdict::Warn => "Warn",
        NsrVerdict::Deny => "Deny",
    }
}

fn nsr_verdict_token_lower(verdict: NsrVerdict) -> &'static str {
    match verdict {
        NsrVerdict::Allow => "allow",
        NsrVerdict::Warn => "warn",
        NsrVerdict::Deny => "deny",
    }
}

fn nsr_signal_priority(verdict: NsrVerdict) -> u16 {
    match verdict {
        NsrVerdict::Allow => 4200,
        NsrVerdict::Warn => 7600,
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
        Some(report) if report.verdict == NsrVerdict::Warn => 4500,
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

fn digest32_to_proto(digest: Digest32) -> Digest {
    Digest {
        algorithm: AlgoId::Blake3_256.to_string(),
        value: Vec::new(),
        algo_id: Some(AlgoId::BLAKE3_256_ID as u32),
        domain: None,
        value_32: Some(digest.as_bytes().to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ucf_ai_port::{MockAiPort, PolicySpeechGate};
    use ucf_archive::InMemoryArchive;
    use ucf_archive_store::InMemoryArchiveStore;
    use ucf_policy_ecology::PolicyEcology;
    use ucf_policy_gateway::NoOpPolicyEvaluator;
    use ucf_risk_gate::PolicyRiskGate;
    use ucf_structural_store::{OnnKnobs, SnnKnobs, StructuralParams, StructuralStore};
    use ucf_tom_port::MockTomPort;
    use ucf_types::v1::spec::{ActionCode, DecisionKind, PolicyDecision};

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
        let phase_frame = PhaseFrame {
            cycle_id: 1,
            global_phase: 12_000,
            module_phase: Vec::new(),
            coherence_plv: 6000,
            pair_locks: Vec::new(),
            states_commit: Digest32::new([0u8; 32]),
            phase_frame_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([1u8; 32]),
        };
        let output = router
            .tick_ncde(
                1,
                &phase_frame,
                4500,
                Digest32::new([9u8; 32]),
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Novelty, 2)],
                1000,
                2000,
                1500,
            )
            .expect("ncde output");
        {
            let mut workspace = router.workspace.lock().expect("workspace lock");
            workspace.set_ncde_snapshot(output.commit, output.state_commit, output.energy);
        }
        let snapshot = router.arbitrate_workspace(1);
        assert_eq!(snapshot.ncde_commit, output.commit);
        assert_eq!(snapshot.ncde_state_commit, output.state_commit);
        assert_eq!(snapshot.ncde_energy, output.energy);

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

        let state = SelfStateBuilder::new(1)
            .ncde_commit(output.state_commit)
            .build();
        assert_eq!(state.ncde_commit, output.state_commit);
    }

    #[test]
    fn cde_spikes_emit_for_stable_edge() {
        let router = build_router();
        let phase_frame = PhaseFrame {
            cycle_id: 1,
            global_phase: 12_000,
            module_phase: Vec::new(),
            coherence_plv: 7000,
            pair_locks: vec![ucf_onn::PairLock {
                a: OscId::Cde,
                b: OscId::Nsr,
                lock: LOCK_MIN_CAUSAL,
                phase_diff: 0,
                commit: Digest32::new([10u8; 32]),
            }],
            states_commit: Digest32::new([0u8; 32]),
            phase_frame_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([3u8; 32]),
        };
        let output = CdeOutputs::new(
            1,
            Digest32::new([4u8; 32]),
            vec![(
                CdeNodeId::Risk,
                CdeNodeId::OutputSuppression,
                ucf_cde_scm::CONF_SPIKE,
                0,
            )],
            Vec::new(),
            Vec::new(),
            true,
        );
        let attention = AttentionWeights {
            channel: FocusChannel::Task,
            gain: 3000,
            noise_suppress: 1200,
            replay_bias: 1500,
            commit: Digest32::new([5u8; 32]),
        };
        if output.emit_spikes {
            let mut ctx = StageContext::new();
            router.queue_cde_spikes(1, &phase_frame, &output, &attention, &mut ctx);
        }
        let spikes = router
            .workspace
            .lock()
            .expect("workspace lock")
            .drain_spikes_for(OscId::Nsr, 1, 10);
        assert!(!spikes.is_empty());
        assert!(spikes
            .iter()
            .all(|spike| spike.kind == SpikeKind::CausalLink));
    }

    #[test]
    fn cde_spikes_skip_when_not_emitting() {
        let router = build_router();
        let phase_frame = PhaseFrame {
            cycle_id: 2,
            global_phase: 12_500,
            module_phase: Vec::new(),
            coherence_plv: 2000,
            pair_locks: Vec::new(),
            states_commit: Digest32::new([0u8; 32]),
            phase_frame_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([6u8; 32]),
        };
        let output = CdeOutputs::new(
            2,
            Digest32::new([7u8; 32]),
            vec![(
                CdeNodeId::Risk,
                CdeNodeId::OutputSuppression,
                ucf_cde_scm::CONF_SPIKE,
                0,
            )],
            Vec::new(),
            Vec::new(),
            false,
        );
        let attention = AttentionWeights {
            channel: FocusChannel::Task,
            gain: 2500,
            noise_suppress: 1100,
            replay_bias: 1400,
            commit: Digest32::new([8u8; 32]),
        };
        if output.emit_spikes {
            let mut ctx = StageContext::new();
            router.queue_cde_spikes(2, &phase_frame, &output, &attention, &mut ctx);
        }
        let spikes = router
            .workspace
            .lock()
            .expect("workspace lock")
            .drain_spikes_for(OscId::Nsr, 2, 10);
        assert!(spikes.is_empty());
    }

    #[test]
    fn ssm_commit_flows_into_workspace_and_attention() {
        let router = build_router();
        let phase_frame = PhaseFrame {
            cycle_id: 1,
            global_phase: 12_000,
            module_phase: Vec::new(),
            coherence_plv: 7000,
            pair_locks: Vec::new(),
            states_commit: Digest32::new([0u8; 32]),
            phase_frame_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([1u8; 32]),
        };
        let ncde_output = router
            .tick_ncde(
                1,
                &phase_frame,
                4500,
                Digest32::new([9u8; 32]),
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Novelty, 2)],
                1000,
                2000,
                1500,
            )
            .expect("ncde output");
        let ssm_output = router
            .tick_ssm(
                &phase_frame,
                &ncde_output,
                Digest32::new([9u8; 32]),
                Digest32::new([2u8; 32]),
                vec![(SpikeKind::Threat, 3)],
                1000,
                2000,
                1500,
            )
            .expect("ssm output");
        {
            let mut workspace = router.workspace.lock().expect("workspace lock");
            workspace.set_ssm_commits(ssm_output.commit, ssm_output.x_commit);
        }
        let snapshot = router.arbitrate_workspace(1);
        assert_eq!(snapshot.ssm_commit, ssm_output.commit);
        assert_eq!(snapshot.ssm_state_commit, ssm_output.x_commit);

        let base = AttentionWeights {
            channel: FocusChannel::Task,
            gain: 3000,
            noise_suppress: 1200,
            replay_bias: 1500,
            commit: Digest32::new([4u8; 32]),
        };
        let biased = router.apply_ssm_attention_bias(base.clone(), Some(ssm_output.wm_salience));
        assert!(biased.gain >= base.gain);
    }

    #[test]
    fn structural_params_drive_onn_window_and_snn_verify_limit() {
        let router = build_router();
        let base = StructuralStore::default_params();
        let onn = OnnKnobs::new(
            base.onn.base_step,
            base.onn.coupling,
            base.onn.max_delta,
            12_000,
        );
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

        assert_eq!(router.onn_phase_window(), 12_000);
        assert_eq!(router.current_snn_knobs().verify_limit, 12);
    }

    #[test]
    fn spikes_in_phase_trigger_full_checks() {
        let spike = SpikeEvent::new(
            1,
            SpikeKind::Threat,
            OscId::Jepa,
            OscId::Nsr,
            1020,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
        );
        assert!(spikes_in_phase(&[spike], 1000, 50));
    }

    #[test]
    fn spikes_out_of_phase_do_not_trigger_full_checks() {
        let spike = SpikeEvent::new(
            1,
            SpikeKind::Threat,
            OscId::Jepa,
            OscId::Nsr,
            2000,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
        );
        assert!(!spikes_in_phase(&[spike], 1000, 50));
    }

    #[test]
    fn causal_spikes_suppressed_when_lock_low() {
        let router = build_router();
        let phase_frame = PhaseFrame {
            cycle_id: 1,
            global_phase: 0,
            module_phase: Vec::new(),
            coherence_plv: 0,
            pair_locks: vec![ucf_onn::PairLock {
                a: OscId::Cde,
                b: OscId::Nsr,
                lock: LOCK_MIN_CAUSAL.saturating_sub(1),
                phase_diff: 0,
                commit: Digest32::new([1u8; 32]),
            }],
            states_commit: Digest32::new([0u8; 32]),
            phase_frame_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([2u8; 32]),
        };
        let spike = SpikeEvent::new(
            1,
            SpikeKind::CausalLink,
            OscId::Cde,
            OscId::Nsr,
            1200,
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
        );
        let mut ctx = StageContext::new();
        let summary = router.append_spike_events(
            1,
            &phase_frame,
            vec![spike],
            true,
            NsrVerdict::Allow,
            &mut ctx,
        );
        assert_ne!(summary.seen_root, summary.accepted_root);
    }

    #[test]
    fn output_intent_suppressed_when_policy_denied() {
        let router = build_router();
        let phase_frame = PhaseFrame {
            cycle_id: 1,
            global_phase: 0,
            module_phase: Vec::new(),
            coherence_plv: 0,
            pair_locks: vec![ucf_onn::PairLock {
                a: OscId::Output,
                b: OscId::Nsr,
                lock: LOCK_MIN_SPEAK,
                phase_diff: 0,
                commit: Digest32::new([4u8; 32]),
            }],
            states_commit: Digest32::new([0u8; 32]),
            phase_frame_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([5u8; 32]),
        };
        let spike = SpikeEvent::new(
            1,
            SpikeKind::OutputIntent,
            OscId::Output,
            OscId::Nsr,
            500,
            Digest32::new([6u8; 32]),
            Digest32::new([7u8; 32]),
        );
        let suppression =
            router.suppression_for_event(&phase_frame, &spike, false, NsrVerdict::Allow);
        assert!(suppression.suppressed_by_policy);
    }

    #[test]
    fn sle_thought_pulse_routes_to_internal_spikes_only() {
        let router = build_router();
        let phase_frame = PhaseFrame {
            cycle_id: 3,
            global_phase: 12_000,
            module_phase: Vec::new(),
            coherence_plv: 6500,
            pair_locks: Vec::new(),
            states_commit: Digest32::new([0u8; 32]),
            phase_frame_commit: Digest32::new([0u8; 32]),
            commit: Digest32::new([1u8; 32]),
        };
        let iit_output = IitOutput {
            cycle_id: 3,
            phi_proxy: 5200,
            coupling_proxy: 4000,
            coherence: 6000,
            warnings: 0,
            commit: Digest32::new([2u8; 32]),
        };
        let decision = PolicyDecision {
            kind: DecisionKind::DecisionKindAllow as i32,
            action: ActionCode::ActionCodeContinue as i32,
            rationale: "ok".to_string(),
            confidence_bp: 1000,
            constraint_ids: Vec::new(),
        };

        let outputs = router
            .tick_sle(
                3,
                &phase_frame,
                &iit_output,
                1200,
                800,
                1500,
                &decision,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("sle outputs");

        assert!(outputs
            .stimuli
            .iter()
            .any(|stim| matches!(stim.kind, SleStimulusKind::ThoughtOnlyPulse)));

        let snapshot = router.arbitrate_workspace(3);
        assert!(!snapshot.internal_utterances.is_empty());

        let spikes = router
            .workspace
            .lock()
            .expect("workspace lock")
            .drain_spikes_for(OscId::BlueBrain, 3, 8);
        assert!(spikes
            .iter()
            .any(|spike| spike.kind == SpikeKind::ThoughtOnly));

        let events = router
            .output_router
            .lock()
            .expect("output router lock")
            .drain_events();
        assert!(events.is_empty());
    }
}
