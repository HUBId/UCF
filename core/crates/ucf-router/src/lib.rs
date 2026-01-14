#![forbid(unsafe_code)]

use std::fmt;
use std::sync::Arc;

use ucf_archive::ExperienceAppender;
use ucf_digital_brain::DigitalBrainPort;
use ucf_policy_gateway::PolicyEvaluator;
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterOutcome {
    pub evidence_id: EvidenceId,
    pub decision_kind: DecisionKind,
}

impl Router {
    pub fn new(
        policy: Arc<dyn PolicyEvaluator + Send + Sync>,
        archive: Arc<dyn ExperienceAppender + Send + Sync>,
        digital_brain: Option<Arc<dyn DigitalBrainPort + Send + Sync>>,
    ) -> Self {
        Self {
            policy,
            archive,
            digital_brain,
        }
    }

    pub fn handle_control_frame(&self, cf: ControlFrame) -> Result<RouterOutcome, RouterError> {
        let decision = self.policy.evaluate(cf.clone());
        self.ensure_allowed(&decision)?;
        let decision_kind =
            DecisionKind::try_from(decision.kind).unwrap_or(DecisionKind::DecisionKindUnspecified);

        let record = self.build_experience_record(&cf, &decision);
        let evidence_id = self.archive.append(record.clone());

        if let Some(brain) = &self.digital_brain {
            brain.ingest(record);
        }

        Ok(RouterOutcome {
            evidence_id,
            decision_kind,
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
    ) -> ExperienceRecord {
        let record_id = format!("exp-{}", cf.frame_id);
        let payload = format!(
            "frame_id={};policy_id={};decision_kind={};decision_action={}",
            cf.frame_id, cf.policy_id, decision.kind, decision.action
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
}
