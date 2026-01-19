#![forbid(unsafe_code)]

use std::fmt;
use std::sync::Arc;

use ucf_ai_port::{
    AiOutput, AiPort, OutputChannel, OutputSuppressed, OutputSuppressionSink, SpeechGate,
};
use ucf_archive::ExperienceAppender;
use ucf_digital_brain::DigitalBrainPort;
use ucf_policy_ecology::RiskDecision;
use ucf_policy_gateway::PolicyEvaluator;
use ucf_risk_gate::{digest_reasons, RiskGate};
use ucf_sandbox::ControlFrameNormalized;
use ucf_tom_port::{IntentType, TomPort};
use ucf_types::v1::spec::{ControlFrame, DecisionKind, ExperienceRecord, PolicyDecision};
use ucf_types::{Digest32, EvidenceId};

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterOutcome {
    pub evidence_id: EvidenceId,
    pub decision_kind: DecisionKind,
    pub speech_outputs: Vec<AiOutput>,
    pub integration_score: Option<u16>,
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
        }
    }

    pub fn handle_control_frame(
        &self,
        cf: ControlFrameNormalized,
    ) -> Result<RouterOutcome, RouterError> {
        let decision = self.policy.evaluate(cf.as_ref().clone());
        self.ensure_allowed(&decision)?;
        let decision_kind =
            DecisionKind::try_from(decision.kind).unwrap_or(DecisionKind::DecisionKindUnspecified);

        let inference = self.ai_port.infer_with_context(&cf);
        let tom_report = self.tom_port.analyze(&cf, &inference.outputs);
        let mut thought_outputs = Vec::new();
        let mut speech_outputs = Vec::new();
        let mut suppressions = Vec::new();
        for output in inference.outputs {
            let gate_result = self.risk_gate.evaluate(
                inference.nsr_report.as_ref(),
                inference.scm_dag.as_ref(),
                &output,
                &cf,
                Some(&tom_report),
                inference.cde_confidence,
            );
            match output.channel {
                OutputChannel::Thought => thought_outputs.push(output),
                OutputChannel::Speech => {
                    let mut reasons = gate_result.reasons.clone();
                    let allow_speech = self.speech_gate.allow_speech(&cf, &output);
                    if !allow_speech {
                        reasons.push("speech_gate_denied".to_string());
                    }
                    let allow_risk = matches!(gate_result.decision, RiskDecision::Permit);
                    if allow_speech && allow_risk {
                        speech_outputs.push(output);
                    } else {
                        let reason_digest = digest_reasons(&reasons);
                        suppressions.push(OutputSuppressionInfo {
                            channel: OutputChannel::Speech,
                            reason_digest,
                            risk: gate_result.risk,
                        });
                        if let Some(sink) = &self.output_suppression_sink {
                            sink.publish(OutputSuppressed {
                                channel: OutputChannel::Speech,
                                reason_digest,
                                risk: gate_result.risk,
                            });
                        }
                    }
                }
            }
        }

        let record = self.build_experience_record(
            cf.as_ref(),
            &decision,
            &thought_outputs,
            &suppressions,
            Some(tom_summary(&tom_report)),
        );
        let evidence_id = self.archive.append(record.clone());
        let integration_score = thought_outputs
            .iter()
            .find_map(|output| output.integration_score);

        if let Some(brain) = &self.digital_brain {
            brain.ingest(record);
        }

        Ok(RouterOutcome {
            evidence_id,
            decision_kind,
            speech_outputs,
            integration_score,
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

fn risk_bucket(overall: u16) -> &'static str {
    match overall {
        0..=3333 => "low",
        3334..=6666 => "med",
        _ => "high",
    }
}
