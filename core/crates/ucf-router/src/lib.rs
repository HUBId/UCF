#![forbid(unsafe_code)]

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ucf_ai_port::{
    AiOutput, AiPort, OutputChannel, OutputSuppressed, OutputSuppressionSink, SpeechGate,
};
use ucf_archive::ExperienceAppender;
use ucf_attn_controller::{
    AttentionEventSink, AttentionUpdated, AttentionWeights, AttnController, AttnInputs,
};
use ucf_digital_brain::DigitalBrainPort;
use ucf_output_router::{GateBundle, NsrSummary, OutputRouter, RouterConfig, SandboxVerdict};
use ucf_policy_gateway::PolicyEvaluator;
use ucf_predictive_coding::{
    error, surprise, Observation, PredictionError, SurpriseSignal, SurpriseUpdated, WorldModel,
    WorldStateVec,
};
use ucf_risk_gate::{digest_reasons, RiskGate};
use ucf_sandbox::ControlFrameNormalized;
use ucf_ssm_port::SsmState;
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
    workspace: Arc<Mutex<Workspace>>,
    cycle_counter: AtomicU64,
    world_model: WorldModel,
    world_state: Mutex<Option<WorldStateVec>>,
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
            output_router: Mutex::new(OutputRouter::new(RouterConfig {
                thought_capacity: 128,
                max_thought_frames_per_cycle: 32,
                external_enabled: true,
            })),
            workspace: Arc::new(Mutex::new(Workspace::new(WorkspaceConfig {
                cap: 64,
                broadcast_cap: 8,
            }))),
            cycle_counter: AtomicU64::new(0),
            world_model: WorldModel::default(),
            world_state: Mutex::new(None),
        }
    }

    pub fn with_attention_sink(mut self, sink: Arc<dyn AttentionEventSink + Send + Sync>) -> Self {
        self.attention_sink = Some(sink);
        self
    }

    pub fn disable_attention(mut self) -> Self {
        self.attention_controller = None;
        self
    }

    pub fn workspace_handle(&self) -> Arc<Mutex<Workspace>> {
        Arc::clone(&self.workspace)
    }

    pub fn handle_control_frame(
        &self,
        cf: ControlFrameNormalized,
    ) -> Result<RouterOutcome, RouterError> {
        let cycle_id = self.cycle_counter.fetch_add(1, Ordering::SeqCst);
        let decision = self.policy.evaluate(cf.as_ref().clone());
        self.publish_workspace_signal(WorkspaceSignal::from_policy_decision(&decision, None));
        self.ensure_allowed(&decision)?;
        let decision_kind =
            DecisionKind::try_from(decision.kind).unwrap_or(DecisionKind::DecisionKindUnspecified);

        let inference = self.ai_port.infer_with_context(&cf);
        let tom_report = self.tom_port.analyze(&cf, &inference.outputs);
        let mut thought_outputs = Vec::new();
        let mut speech_outputs = Vec::new();
        let mut suppressions = Vec::new();
        let mut attention_risk = 0u16;
        let outputs = inference.outputs;
        let mut risk_results = Vec::with_capacity(outputs.len());
        let mut speech_gate_results = Vec::with_capacity(outputs.len());
        for output in &outputs {
            let gate_result = self.risk_gate.evaluate(
                inference.nsr_report.as_ref(),
                inference.scm_dag.as_ref(),
                output,
                &cf,
                Some(&tom_report),
                inference.cde_confidence,
            );
            attention_risk = attention_risk.max(gate_result.risk);
            risk_results.push(gate_result);
            speech_gate_results.push(self.speech_gate.allow_speech(&cf, output));
        }
        self.publish_workspace_signals(
            risk_results
                .iter()
                .map(|result| WorkspaceSignal::from_risk_result(result, None)),
        );

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
                .map(|decision| WorkspaceSignal::from_output_decision(decision, None))
                .chain(
                    events
                        .iter()
                        .map(|event| WorkspaceSignal::from_output_event(event, None)),
                ),
        );

        for (idx, output) in outputs.iter().enumerate() {
            match output.channel {
                OutputChannel::Thought => thought_outputs.push(output.clone()),
                OutputChannel::Speech => {
                    if decisions
                        .get(idx)
                        .map(|decision| decision.permitted)
                        .unwrap_or(false)
                    {
                        speech_outputs.push(output.clone());
                    } else if let Some(result) = gates.risk_results.get(idx) {
                        let reason = decisions
                            .get(idx)
                            .map(|decision| decision.reason_code.clone())
                            .unwrap_or_else(|| "risk_denied".to_string());
                        let reason_digest = digest_reasons(&[reason]);
                        suppressions.push(OutputSuppressionInfo {
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

        let integration_score = thought_outputs
            .iter()
            .find_map(|output| output.integration_score);
        if let Some(score) = integration_score {
            self.publish_workspace_signal(WorkspaceSignal::from_integration_score(score, None));
        }
        if let Some(state) = inference.ssm_state.as_ref() {
            self.publish_workspace_signal(WorkspaceSignal::from_world_state(state.commit));
        }
        let observation = inference
            .ssm_state
            .as_ref()
            .map(observation_from_ssm_state)
            .unwrap_or_else(|| observation_from_frame(&cf));
        let predictive_result = self.update_predictive_coding(&observation);
        if let Some((error, surprise_signal)) = predictive_result.as_ref() {
            let update = SurpriseUpdated::from(surprise_signal);
            self.publish_workspace_signal(WorkspaceSignal::from_surprise_update(&update));
            let record = self.build_predictive_record(cf.as_ref(), error, surprise_signal);
            self.archive.append(record);
        }
        let attention_weights = self.compute_attention(
            decision.kind as u16,
            attention_risk,
            integration_score.unwrap_or(0),
            &tom_report,
            predictive_result
                .as_ref()
                .map(|(_, signal)| signal.score)
                .unwrap_or(0),
        );
        if let Some(weights) = attention_weights.as_ref() {
            self.ai_port.update_attention(weights);
            let update = AttentionUpdated {
                channel: weights.channel,
                gain: weights.gain,
                replay_bias: weights.replay_bias,
                commit: weights.commit,
            };
            self.publish_workspace_signal(WorkspaceSignal::from_attention_update(&update));
            if let Some(sink) = &self.attention_sink {
                sink.publish(update);
            }
        }

        let record = self.build_experience_record(
            cf.as_ref(),
            &decision,
            &thought_outputs,
            &suppressions,
            Some(tom_summary(&tom_report)),
            attention_weights.as_ref(),
        );
        let evidence_id = self.archive.append(record.clone());

        if let Some(brain) = &self.digital_brain {
            brain.ingest(record);
        }

        let snapshot = self.arbitrate_workspace(cycle_id);
        let workspace_record = self.build_workspace_record(&snapshot);
        self.archive.append(workspace_record);

        Ok(RouterOutcome {
            evidence_id,
            decision_kind,
            speech_outputs,
            integration_score,
            workspace_snapshot_commit: Some(snapshot.commit),
            surprise_signal: predictive_result.map(|(_, signal)| signal),
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

    fn compute_attention(
        &self,
        policy_class: u16,
        risk_score: u16,
        integration_score: u16,
        tom_report: &ucf_tom_port::TomReport,
        surprise_score: u16,
    ) -> Option<AttentionWeights> {
        let controller = self.attention_controller.as_ref()?;
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
}
