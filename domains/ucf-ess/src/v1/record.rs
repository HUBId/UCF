use std::sync::Arc;

use sha2::{Digest, Sha256};
use ucf_core::types::SimTime;
use ucf_frames::v1::{
    BrainFrame, ComputeSignalsSummary, ControlFrame, CorrelationId, DecisionFrame, DecisionMeta,
    NeuromodulatorSnapshot, PhiProxySnapshot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExperienceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperienceKind {
    ControlIn,
    DecisionOut,
    BrainOut,
    Note,
    ToolRequest,
    ToolAuth,
    ToolExecution,
    SandboxCall,
    SandboxReply,
    AuditCheckpoint,
    Hormone,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperienceRecord {
    pub id: ExperienceId,
    pub time: SimTime,
    pub corr: CorrelationId,
    pub kind: ExperienceKind,
    pub payload: ExperiencePayload,
    pub neuromod: Option<NeuromodulatorSnapshot>,
    pub iit_phi: Option<PhiProxySnapshot>,
    pub decision_meta: Option<DecisionMeta>,
    pub compute_summary: Option<ComputeSignalsSummary>,
    pub hormone_record: Option<HormoneRecord>,
    pub audit_prev_digest: Option<[u8; 32]>,
    pub audit_digest: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExperiencePayload {
    Control(ControlFrame),
    Decision(Box<DecisionFrame>),
    Brain(BrainFrame),
    Text(Arc<str>),
    Audit(AuditPayload),
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditPayload {
    ToolRequest(ToolRequestRecord),
    ToolAuth(ToolAuthRecord),
    ToolExecution(ToolExecutionRecord),
    SandboxCall(SandboxCallRecord),
    SandboxReply(SandboxReplyRecord),
    AuditCheckpoint(AuditCheckpointRecord),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRequestRecord {
    pub tool_request_id: u64,
    pub capability_kind: String,
    pub target: String,
    pub decision_id: u64,
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolAuthRecord {
    pub tool_request_id: u64,
    pub allowed: bool,
    pub reason: String,
    pub token_digest: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolExecutionRecord {
    pub tool_request_id: u64,
    pub status: String,
    pub bytes_out: Option<u32>,
    pub bytes_in: Option<u32>,
    pub error_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxCallRecord {
    pub tool_request_id: u64,
    pub call_digest: [u8; 32],
    pub module: String,
    pub op: String,
    pub evidence_chain_digest: [u8; 32],
    pub capability_count: u32,
    pub isolation_runtime: Option<String>,
    pub wasm_module_digest: Option<[u8; 32]>,
    pub fuel_used: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxReplyRecord {
    pub tool_request_id: u64,
    pub reply_digest: [u8; 32],
    pub status: String,
    pub bytes_out: u32,
    pub bytes_in: u32,
    pub token_digest: Option<[u8; 32]>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditCheckpointRecord {
    pub head_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HormoneRecord {
    pub t: u64,
    pub cortisol_q: u16,
    pub drive_q: u16,
    pub stress_index_q: u16,
    pub hormone_digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
    pub modulation_digest: Option<[u8; 32]>,
    pub schema_version: u16,
}

impl ExperienceRecord {
    pub fn from_control(id: ExperienceId, ctrl: ControlFrame) -> Self {
        Self {
            id,
            time: ctrl.time,
            corr: ctrl.corr,
            kind: ExperienceKind::ControlIn,
            payload: ExperiencePayload::Control(ctrl),
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_decision(id: ExperienceId, decision: DecisionFrame) -> Self {
        Self {
            id,
            time: decision.time,
            corr: decision.corr,
            kind: ExperienceKind::DecisionOut,
            payload: ExperiencePayload::Decision(Box::new(decision.clone())),
            neuromod: None,
            iit_phi: None,
            decision_meta: Some(decision.meta),
            compute_summary: decision.compute_summary,
            hormone_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_brain(id: ExperienceId, brain: BrainFrame) -> Self {
        Self {
            id,
            time: brain.time,
            corr: brain.corr,
            kind: ExperienceKind::BrainOut,
            payload: ExperiencePayload::Brain(brain),
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn note(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        text: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::Note,
            payload: ExperiencePayload::Text(text.into()),
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_hormone(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        hormone_record: HormoneRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::Hormone,
            payload: ExperiencePayload::Empty,
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: Some(hormone_record),
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn audit(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        kind: ExperienceKind,
        payload: AuditPayload,
        prev_digest: [u8; 32],
    ) -> Self {
        let canonical = format!("{:?}|{:?}|{}|{}", kind, payload, time.tick.get(), corr.0);
        let mut hasher = Sha256::new();
        hasher.update(prev_digest);
        hasher.update(canonical.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        Self {
            id,
            time,
            corr,
            kind,
            payload: ExperiencePayload::Audit(payload),
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            audit_prev_digest: Some(prev_digest),
            audit_digest: Some(digest),
        }
    }

    pub fn with_neuromod(mut self, neuromod: NeuromodulatorSnapshot) -> Self {
        self.neuromod = Some(neuromod);
        self
    }
}

impl ExperienceRecord {
    pub fn with_iit_phi(mut self, iit_phi: PhiProxySnapshot) -> Self {
        self.iit_phi = Some(iit_phi);
        self
    }
}
