#![forbid(unsafe_code)]

use ucf_sandbox::SandboxErrorCode;
use ucf_types::v1::spec::DecisionKind;
use ucf_types::{Digest32, EvidenceId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeEvent {
    pub evidence_id: EvidenceId,
    pub decision_kind: DecisionKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxRejectEvent {
    pub code: SandboxErrorCode,
    pub message: String,
    pub field: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechEvent {
    pub evidence_id: EvidenceId,
    pub content: String,
    pub confidence: u16,
    pub rationale_commit: Option<Digest32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceBroadcast {
    pub snapshot_commit: Digest32,
}
