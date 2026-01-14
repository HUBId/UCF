#![forbid(unsafe_code)]

use std::fmt;
use std::sync::Arc;

use ucf_ai_port::{AiOutput, AiPort, OutputChannel, SpeechGate};
use ucf_archive::ExperienceAppender;
use ucf_digital_brain::DigitalBrainPort;
use ucf_policy_gateway::PolicyEvaluator;
use ucf_sandbox::ControlFrameNormalized;
use ucf_types::v1::spec::{ControlFrame, DecisionKind, ExperienceRecord, PolicyDecision};
use ucf_types::EvidenceId;

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterOutcome {
    pub evidence_id: EvidenceId,
    pub decision_kind: DecisionKind,
    pub speech_outputs: Vec<AiOutput>,
}

impl Router {
    pub fn new(
        policy: Arc<dyn PolicyEvaluator + Send + Sync>,
        archive: Arc<dyn ExperienceAppender + Send + Sync>,
        digital_brain: Option<Arc<dyn DigitalBrainPort + Send + Sync>>,
        ai_port: Arc<dyn AiPort + Send + Sync>,
        speech_gate: Arc<dyn SpeechGate + Send + Sync>,
    ) -> Self {
        Self {
            policy,
            archive,
            digital_brain,
            ai_port,
            speech_gate,
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

        let outputs = self.ai_port.infer(&cf);
        let mut thought_outputs = Vec::new();
        let mut speech_outputs = Vec::new();
        for output in outputs {
            match output.channel {
                OutputChannel::Thought => thought_outputs.push(output),
                OutputChannel::Speech => {
                    if self.speech_gate.allow_speech(&cf, &output) {
                        speech_outputs.push(output);
                    }
                }
            }
        }

        let record = self.build_experience_record(cf.as_ref(), &decision, &thought_outputs);
        let evidence_id = self.archive.append(record.clone());

        if let Some(brain) = &self.digital_brain {
            brain.ingest(record);
        }

        Ok(RouterOutcome {
            evidence_id,
            decision_kind,
            speech_outputs,
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
