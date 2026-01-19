#![forbid(unsafe_code)]

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ucf_ai_port::{
    AiInference, AiOutput, AiPort, OutputChannel, OutputSuppressed, OutputSuppressionSink,
    SpeechGate,
};
use ucf_archive::ExperienceAppender;
use ucf_attn_controller::{
    AttentionEventSink, AttentionUpdated, AttentionWeights, AttnController, AttnInputs,
};
use ucf_digital_brain::DigitalBrainPort;
use ucf_geist::{encode_self_state, SelfState, SelfStateBuilder};
use ucf_iit_monitor::{IitAction, IitActionKind, IitBand, IitMonitor, IitReport};
use ucf_nsr_port::NsrReport;
use ucf_output_router::{GateBundle, NsrSummary, OutputRouter, RouterConfig, SandboxVerdict};
use ucf_policy_ecology::RiskGateResult;
use ucf_policy_gateway::PolicyEvaluator;
use ucf_predictive_coding::{
    error, surprise, Observation, PredictionError, SurpriseSignal, SurpriseUpdated, WorldModel,
    WorldStateVec,
};
use ucf_risk_gate::{digest_reasons, RiskGate};
use ucf_rsa_hooks::{MockRsaHook, RsaContext, RsaHook, RsaProposal};
use ucf_sandbox::ControlFrameNormalized;
use ucf_sle::{SelfReflex, SleEngine};
use ucf_ssm_port::SsmState;
use ucf_tcf_port::{idle_attention, CyclePlan, CyclePlanned, DeterministicTcf, PulseKind, TcfPort};
use ucf_tom_port::{IntentType, TomPort};
use ucf_types::v1::spec::{ControlFrame, DecisionKind, Digest, ExperienceRecord, PolicyDecision};
use ucf_types::{AlgoId, Digest32, EvidenceId};
use ucf_workspace::{
    encode_workspace_snapshot, Workspace, WorkspaceConfig, WorkspaceSignal, WorkspaceSnapshot,
};

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
    digital_brain: Option<Arc<dyn DigitalBrainPort + Send + Sync>>,
    ai_port: Arc<dyn AiPort + Send + Sync>,
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
    iit_monitor: Mutex<IitMonitor>,
    rsa_hooks: Vec<Arc<dyn RsaHook + Send + Sync>>,
    last_self_state: Mutex<Option<SelfState>>,
    last_workspace_snapshot: Mutex<Option<WorkspaceSnapshot>>,
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

pub trait StageTrace {
    fn record(&self, stage: PulseKind);
}

struct StageContext {
    decision: Option<PolicyDecision>,
    decision_kind: DecisionKind,
    inference: Option<AiInference>,
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
    iit_report: Option<IitReport>,
    iit_actions: Vec<IitAction>,
}

impl StageContext {
    fn new() -> Self {
        Self {
            decision: None,
            decision_kind: DecisionKind::DecisionKindUnspecified,
            inference: None,
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
            iit_report: None,
            iit_actions: Vec::new(),
        }
    }
}

impl Router {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        policy: Arc<dyn PolicyEvaluator + Send + Sync>,
        archive: Arc<dyn ExperienceAppender + Send + Sync>,
        digital_brain: Option<Arc<dyn DigitalBrainPort + Send + Sync>>,
        ai_port: Arc<dyn AiPort + Send + Sync>,
        speech_gate: Arc<dyn SpeechGate + Send + Sync>,
        risk_gate: Arc<dyn RiskGate + Send + Sync>,
        tom_port: Arc<dyn TomPort + Send + Sync>,
        output_suppression_sink: Option<Arc<dyn OutputSuppressionSink + Send + Sync>>,
    ) -> Self {
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
            digital_brain,
            ai_port,
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
            iit_monitor: Mutex::new(IitMonitor::new(4)),
            rsa_hooks: vec![Arc::new(MockRsaHook::new())],
            last_self_state: Mutex::new(None),
            last_workspace_snapshot: Mutex::new(None),
        }
    }

    pub fn with_attention_sink(mut self, sink: Arc<dyn AttentionEventSink + Send + Sync>) -> Self {
        self.attention_sink = Some(sink);
        self
    }

    pub fn with_tcf_port(mut self, port: Box<dyn TcfPort + Send + Sync>) -> Self {
        self.tcf_port = Mutex::new(port);
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

    fn latest_workspace_snapshot(&self, cycle_id: u64) -> WorkspaceSnapshot {
        self.last_workspace_snapshot
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
            .unwrap_or(WorkspaceSnapshot {
                cycle_id,
                broadcast: Vec::new(),
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
        self.archive
            .append(self.build_cycle_plan_record(&cycle_plan, &planned));

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
                    self.run_think_stage(&cf, &mut ctx);
                }
                PulseKind::Verify => {
                    if ctx.decision.is_none() {
                        continue;
                    }
                    if ctx.inference.is_none() {
                        self.run_think_stage(&cf, &mut ctx);
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
                    self.archive
                        .append(self.build_self_state_record(&self_state));
                    self.archive
                        .append(self.build_sle_reflex_record(cycle_id, &sle_reflex));
                    if let Ok(mut guard) = self.last_self_state.lock() {
                        *guard = Some(self_state);
                    }
                    ctx.self_state = Some(self_state);
                    ctx.sle_reflex = Some(sle_reflex);

                    let (iit_report, iit_actions) = {
                        let mut monitor = self.iit_monitor.lock().expect("iit monitor lock");
                        monitor.evaluate(&workspace_snapshot, attention_risk)
                    };
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
                    self.archive
                        .append(self.build_iit_report_record(cycle_id, &iit_report));
                    for action in &iit_actions {
                        self.archive
                            .append(self.build_iit_action_record(cycle_id, action));
                    }
                    ctx.iit_report = Some(iit_report);

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
                        sandbox: SandboxVerdict::Ok,
                        risk_results,
                        nsr_summary,
                        speech_gate: speech_gate_results,
                    };
                    let mut output_router = self.output_router.lock().expect("output router lock");
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
                    let attention_weights = self.compute_attention(
                        decision.kind as u16,
                        ctx.attention_risk,
                        ctx.integration_score.unwrap_or(0),
                        ctx.integration_bias,
                        tom_report,
                        surprise_score,
                    );
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
                    let workspace_record = self.build_workspace_record(&snapshot);
                    self.archive.append(workspace_record);
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

    fn build_workspace_record(&self, snapshot: &WorkspaceSnapshot) -> ExperienceRecord {
        let record_id = format!("workspace-{}", hex::encode(snapshot.commit.as_bytes()));
        let payload = encode_workspace_snapshot(snapshot);

        ExperienceRecord {
            record_id,
            observed_at_ms: snapshot.cycle_id,
            subject_id: "workspace".to_string(),
            payload,
            digest: Some(digest32_to_proto(snapshot.commit)),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    fn build_cycle_plan_record(
        &self,
        plan: &CyclePlan,
        planned: &CyclePlanned,
    ) -> ExperienceRecord {
        let record_id = format!(
            "cycle-plan-{}-{}",
            planned.cycle_id,
            hex::encode(plan.commit.as_bytes())
        );
        let payload = ucf_tcf_port::encode_cycle_plan(plan);

        ExperienceRecord {
            record_id,
            observed_at_ms: planned.cycle_id,
            subject_id: "tcf".to_string(),
            payload,
            digest: Some(digest32_to_proto(planned.commit)),
            vrf_tag: None,
            proof_ref: None,
        }
    }

    fn build_self_state_record(&self, state: &SelfState) -> ExperienceRecord {
        let record_id = format!(
            "self-state-{}-{}",
            state.cycle_id,
            hex::encode(state.commit.as_bytes())
        );
        let payload = encode_self_state(state);
        ExperienceRecord {
            record_id,
            observed_at_ms: state.cycle_id,
            subject_id: "geist".to_string(),
            payload,
            digest: Some(digest32_to_proto(state.commit)),
            vrf_tag: None,
            proof_ref: None,
        }
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

    fn build_iit_report_record(&self, cycle_id: u64, report: &IitReport) -> ExperienceRecord {
        let record_id = format!(
            "iit-report-{}-{}",
            cycle_id,
            hex::encode(report.commit.as_bytes())
        );
        let band = match report.band {
            IitBand::Low => "LOW",
            IitBand::Medium => "MED",
            IitBand::High => "HIGH",
        };
        let payload = format!("phi={};band={band}", report.phi).into_bytes();
        ExperienceRecord {
            record_id,
            observed_at_ms: cycle_id,
            subject_id: "iit".to_string(),
            payload,
            digest: Some(digest32_to_proto(report.commit)),
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

    fn compute_attention(
        &self,
        policy_class: u16,
        risk_score: u16,
        integration_score: u16,
        integration_bias: i16,
        tom_report: &ucf_tom_port::TomReport,
        surprise_score: u16,
    ) -> Option<AttentionWeights> {
        let controller = self.attention_controller.as_ref()?;
        let integration_score = apply_integration_bias(integration_score, integration_bias);
        let inputs = AttnInputs {
            policy_class,
            risk_score,
            integration_score,
            consistency_instability: 0,
            intent_type: intent_type_code(tom_report.intent.intent),
            surprise_score,
        };
        Some(controller.compute(&inputs))
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

    fn arbitrate_workspace(&self, cycle_id: u64) -> WorkspaceSnapshot {
        let mut workspace = self.workspace.lock().expect("workspace lock");
        workspace.arbitrate(cycle_id)
    }

    fn emit_stage_trace(&self, stage: PulseKind) {
        if let Some(trace) = self.stage_trace.as_ref() {
            trace.record(stage);
        }
    }

    fn run_think_stage(&self, cf: &ControlFrameNormalized, ctx: &mut StageContext) {
        if ctx.inference.is_some() {
            return;
        }
        let inference = self.ai_port.infer_with_context(cf);
        let tom_report = self.tom_port.analyze(cf, &inference.outputs);
        ctx.inference = Some(inference);
        ctx.tom_report = Some(tom_report);
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

fn consistency_score_from_nsr(report: Option<&NsrReport>) -> u16 {
    match report {
        Some(report) if report.ok => 10_000,
        Some(_) => 2000,
        None => 5000,
    }
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
}
