#![forbid(unsafe_code)]

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ucf::boundary::{self, v1::WorkspaceBroadcastV1, v1::WorkspaceSignalV1};
use ucf_ai_port::{
    AiInference, AiOutput, AiPort, AiPortWorker, OutputChannel, OutputSuppressed,
    OutputSuppressionSink, SpeechGate,
};
use ucf_archive::ExperienceAppender;
use ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind, RecordMeta};
use ucf_attn_controller::{
    AttentionEventSink, AttentionUpdated, AttentionWeights, AttnController, AttnInputs,
    FocusChannel,
};
use ucf_bluebrain_port::{BlueBrainPort, NeuromodDelta};
use ucf_brain_mapper::map_to_stimulus;
use ucf_consistency_engine::{
    ConsistencyAction, ConsistencyActionKind, ConsistencyEngine, ConsistencyInputs,
    ConsistencyReport, DriftBand,
};
use ucf_digital_brain::DigitalBrainPort;
use ucf_feature_translator::{
    ActivationView, LensPort as FeatureLensPort, LensSelection, MockLensPort, MockSaePort,
    SaePort as FeatureSaePort,
};
use ucf_geist::{SelfState, SelfStateBuilder};
use ucf_iit_monitor::{IitAction, IitActionKind, IitBand, IitMonitor, IitReport};
use ucf_ism::IsmStore;
use ucf_nsr_port::NsrReport;
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
use ucf_rsa_hooks::{MockRsaHook, RsaContext, RsaHook, RsaProposal};
use ucf_sandbox::{
    AiCallRequest, ControlFrameNormalized, IntentSummary, MockWasmSandbox, SandboxBudget,
    SandboxCaps, SandboxPort, SandboxReport,
};
use ucf_sle::{SelfReflex, SleEngine};
use ucf_ssm_port::SsmState;
use ucf_tcf_port::{
    ai_mode_for_pulse, idle_attention, CyclePlan, CyclePlanned, DeterministicTcf, PulseKind,
    TcfPort,
};
use ucf_tom_port::{IntentType, TomPort};
use ucf_types::v1::spec::{ControlFrame, DecisionKind, Digest, ExperienceRecord, PolicyDecision};
use ucf_types::{AlgoId, Digest32, EvidenceId};
use ucf_workspace::{
    output_event_commit, SignalKind, Workspace, WorkspaceConfig, WorkspaceSignal, WorkspaceSnapshot,
};

const ISM_ANCHOR_TOP_K: usize = 4;
const FEATURE_SIGNAL_PRIORITY: u16 = 3200;
const FEATURE_RECORD_KIND: u16 = 42;
const SANDBOX_DENIED_RECORD_KIND: u16 = 73;

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
    last_attention: Mutex<AttentionWeights>,
    last_surprise: Mutex<Option<SurpriseSignal>>,
    stage_trace: Option<Arc<dyn StageTrace + Send + Sync>>,
    world_model: WorldModel,
    world_state: Mutex<Option<WorldStateVec>>,
    sle_engine: Arc<SleEngine>,
    consistency_engine: ConsistencyEngine,
    ism_store: Arc<Mutex<IsmStore>>,
    iit_monitor: Mutex<IitMonitor>,
    recursion_controller: RecursionController,
    rsa_hooks: Vec<Arc<dyn RsaHook + Send + Sync>>,
    last_self_state: Mutex<Option<SelfState>>,
    last_workspace_snapshot: Mutex<Option<WorkspaceSnapshot>>,
    last_recursion_budget: Mutex<Option<RecursionBudget>>,
    pending_neuromod_delta: Mutex<Option<NeuromodDelta>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterOutcome {
    pub evidence_id: EvidenceId,
    pub decision_kind: DecisionKind,
    pub speech_outputs: Vec<AiOutput>,
    pub integration_score: Option<u16>,
    pub workspace_snapshot_commit: Option<Digest32>,
    pub surprise_signal: Option<SurpriseSignal>,
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
    attention_risk: u16,
    thought_outputs: Vec<AiOutput>,
    speech_outputs: Vec<AiOutput>,
    suppressions: Vec<OutputSuppressionInfo>,
    integration_score: Option<u16>,
    integration_bias: i16,
    predictive_result: Option<(PredictionError, SurpriseSignal)>,
    attention_weights: Option<AttentionWeights>,
    evidence_id: Option<EvidenceId>,
    workspace_snapshot_commit: Option<Digest32>,
    self_state: Option<SelfState>,
    sle_reflex: Option<SelfReflex>,
    consistency_report: Option<ConsistencyReport>,
    consistency_actions: Vec<ConsistencyAction>,
    consistency_effects: Option<ConsistencyActionEffects>,
    iit_report: Option<IitReport>,
    iit_actions: Vec<IitAction>,
    recursion_budget: Option<RecursionBudget>,
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
            attention_risk: 0,
            thought_outputs: Vec::new(),
            speech_outputs: Vec::new(),
            suppressions: Vec::new(),
            integration_score: None,
            integration_bias: 0,
            predictive_result: None,
            attention_weights: None,
            evidence_id: None,
            workspace_snapshot_commit: None,
            self_state: None,
            sle_reflex: None,
            consistency_report: None,
            consistency_actions: Vec::new(),
            consistency_effects: None,
            iit_report: None,
            iit_actions: Vec::new(),
            recursion_budget: None,
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
            last_attention: Mutex::new(idle_attention()),
            last_surprise: Mutex::new(None),
            stage_trace: None,
            world_model: WorldModel::default(),
            world_state: Mutex::new(None),
            sle_engine: Arc::new(SleEngine::new(6)),
            consistency_engine: ConsistencyEngine,
            ism_store: Arc::new(Mutex::new(IsmStore::new(64))),
            iit_monitor: Mutex::new(IitMonitor::new(4)),
            recursion_controller: RecursionController::default(),
            rsa_hooks: vec![Arc::new(MockRsaHook::new())],
            last_self_state: Mutex::new(None),
            last_workspace_snapshot: Mutex::new(None),
            last_recursion_budget: Mutex::new(None),
            pending_neuromod_delta: Mutex::new(None),
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

        let cycle_plan = {
            let mut tcf = self.tcf_port.lock().expect("tcf lock");
            tcf.step(&plan_attn, plan_surprise.as_ref())
        };
        let planned = DeterministicTcf::planned_event(&cycle_plan);
        self.append_cycle_plan_record(&cycle_plan, &planned);

        let cycle_id = cycle_plan.cycle_id;
        let mut ctx = StageContext::new();

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
                    let decision = ctx.decision.as_ref().expect("decision available");
                    let Some(inference) = ctx.inference.as_ref() else {
                        continue;
                    };
                    let mut attention_risk = 0u16;
                    let outputs = &inference.outputs;
                    let mut risk_results = Vec::with_capacity(outputs.len());
                    let mut speech_gate_results = Vec::with_capacity(outputs.len());
                    let tom_report = ctx.tom_report.as_ref().expect("tom report available");
                    for output in outputs {
                        let gate_result = self.risk_gate.evaluate(
                            inference.nsr_report.as_ref(),
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
                    self.publish_workspace_signals(risk_results.iter().map(|result| {
                        WorkspaceSignal::from_risk_result(result, None, Some(pulse.slot))
                    }));

                    let workspace_snapshot = self.latest_workspace_snapshot(cycle_id);
                    let (iit_report, iit_actions) = {
                        let mut monitor = self.iit_monitor.lock().expect("iit monitor lock");
                        monitor.evaluate(&workspace_snapshot, attention_risk)
                    };
                    let surprise_score = ctx
                        .predictive_result
                        .as_ref()
                        .map(|(_, signal)| signal.score)
                        .unwrap_or(0);
                    let attention_weights = self
                        .last_attention
                        .lock()
                        .map(|attn| attn.clone())
                        .unwrap_or_else(|_| idle_attention());
                    // Bluebrain stimulation occurs during the Verify pulse using the latest
                    // workspace snapshot, attention, and surprise context.
                    let surprise_signal = ctx.predictive_result.as_ref().map(|(_, signal)| signal);
                    let lens_selection = ctx.inference.as_ref().and_then(|inference| {
                        self.translate_features(
                            inference.activation_view.as_ref(),
                            &attention_weights,
                            cycle_id,
                            pulse.slot,
                        )
                    });
                    self.stimulate_bluebrain_port(
                        &cf,
                        &workspace_snapshot,
                        &attention_weights,
                        surprise_signal,
                        lens_selection.as_ref(),
                        pulse.slot,
                    );
                    let recursion_inputs = RecursionInputs {
                        phi: iit_report.phi,
                        drift_score: drift_score_from_snapshot(&workspace_snapshot),
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
                    let consistency = consistency_score_from_nsr(inference.nsr_report.as_ref());
                    let self_state = SelfStateBuilder::new(cycle_id)
                        .ssm_commit(ssm_commit)
                        .workspace_commit(workspace_snapshot.commit)
                        .risk_commit(risk_commit)
                        .attn_commit(attn_commit)
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

                    let effects = iit_action_effects(
                        self.workspace_base,
                        &self.output_router_base,
                        &iit_actions,
                    );
                    ctx.integration_score = Some(iit_report.phi);
                    ctx.integration_bias = effects.integration_bias;
                    ctx.iit_actions = iit_actions.clone();
                    self.apply_iit_effects(effects);
                    self.publish_workspace_signal(WorkspaceSignal::from_integration_score(
                        iit_report.phi,
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

                    let nsr_summary = NsrSummary {
                        ok: inference
                            .nsr_report
                            .as_ref()
                            .map(|report| report.ok)
                            .unwrap_or(true),
                        violations_digest: inference
                            .nsr_report
                            .as_ref()
                            .map(|report| digest_reasons(&report.violations))
                            .unwrap_or_else(|| digest_reasons(&[])),
                    };
                    let gates = GateBundle {
                        policy_decision: decision.clone(),
                        sandbox: ctx.sandbox_verdict.clone().unwrap_or(SandboxVerdict::Allow),
                        risk_results,
                        nsr_summary,
                        speech_gate: speech_gate_results,
                    };
                    let mut output_router = self.output_router.lock().expect("output router lock");
                    if let Some(budget) = ctx.recursion_budget.as_ref() {
                        output_router.apply_recursion_budget(budget);
                    }
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
                    let surprise_score = ctx
                        .predictive_result
                        .as_ref()
                        .map(|(_, signal)| signal.score)
                        .unwrap_or(0);
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
                    };
                    let attention_weights = self.compute_attention(attention_ctx);
                    if let Some(weights) = attention_weights.as_ref() {
                        self.ai_port.update_attention(weights);
                        let update = AttentionUpdated {
                            channel: weights.channel,
                            gain: weights.gain,
                            replay_bias: weights.replay_bias,
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

    fn append_cycle_plan_record(&self, plan: &CyclePlan, planned: &CyclePlanned) {
        let meta = RecordMeta {
            cycle_id: planned.cycle_id,
            tier: planned.pulse_count,
            flags: 0,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        self.append_archive_record(RecordKind::CyclePlan, plan.commit, meta);
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
        Some(apply_consistency_effects(weights, ctx.consistency_effects))
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
    }

    fn append_feature_translation_record(
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

fn consistency_score_from_nsr(report: Option<&NsrReport>) -> u16 {
    match report {
        Some(report) if report.ok => 10_000,
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
}
