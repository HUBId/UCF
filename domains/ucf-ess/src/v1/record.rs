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
    Neuro,
    DeltaProposal,
    DeltaEvaluation,
    DeltaRecommendation,
    Nsr,
    CandidateSet,
    Output,
    BackendPack,
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
    pub neuro_record: Option<NeuroRecord>,
    pub delta_proposal_record: Option<DeltaProposalRecord>,
    pub delta_evaluation_record: Option<DeltaEvaluationRecord>,
    pub delta_recommendation_record: Option<DeltaRecommendationRecord>,
    pub nsr_record: Option<NsrRecord>,
    pub backend_pack_record: Option<BackendPackRecord>,
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
    CandidateSet(CandidateSetRecord),
    Output(OutputRecord),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRequestRecord {
    pub tool_request_id: u64,
    pub capability_kind: String,
    pub target: String,
    pub decision_id: u64,
    pub evidence_chain_digest: [u8; 32],
    pub candidate_id: Option<u16>,
    pub tool_intent_digest: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSummaryRecord {
    pub candidate_id: u16,
    pub digest: [u8; 32],
    pub intent_kind: u8,
    pub output_class: u8,
    pub tool_intent_count: u8,
    pub allowed: bool,
    pub policy_hint: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSetRecord {
    pub schema_version: u16,
    pub decision_id: u64,
    pub t: u64,
    pub selected_candidate_id: u16,
    pub selected_candidate_digest: [u8; 32],
    pub summaries: Vec<CandidateSummaryRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputRecord {
    pub schema_version: u16,
    pub decision_id: u64,
    pub candidate_id: u16,
    pub t: u64,
    pub output_class: u8,
    pub llm_backend_name: String,
    pub llm_request_digest: [u8; 32],
    pub llm_response_digest: [u8; 32],
    pub token_count: u32,
    pub status: u8,
    pub finish_reason: u8,
    pub text: Option<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuroRecord {
    pub t: u64,
    pub arousal_q: u16,
    pub attention_gain_q: u16,
    pub excitability_q: u16,
    pub spike_rate_q: u16,
    pub summary_digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
    pub hormone_digest: Option<[u8; 32]>,
    pub spikes_digest: Option<[u8; 32]>,
    pub spike_count: u16,
    pub degraded: bool,
    pub schema_version: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaProposalRecord {
    pub schema_version: u16,
    pub delta_id: [u8; 32],
    pub t: u64,
    pub target: u8,
    pub ops_summary: [u8; 128],
    pub digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaEvaluationRecord {
    pub schema_version: u16,
    pub delta_id: [u8; 32],
    pub fitness_q: u16,
    pub risk_penalty_q: u16,
    pub stability_penalty_q: u16,
    pub budget_penalty_q: u16,
    pub score_digest: [u8; 32],
    pub accepted: bool,
    pub reason_codes: [u8; 8],
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaRecommendationRecord {
    pub schema_version: u16,
    pub delta_id: [u8; 32],
    pub recommended_ops: [u8; 128],
    pub safety_clamps: [u8; 64],
    pub requires_human_apply: bool,
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendPackRecord {
    pub schema_version: u16,
    pub t: u64,
    pub pack_name: String,
    pub pack_id: u32,
    pub fixtures_digest: [u8; 32],
    pub llm_backend: u8,
    pub world_backend: u8,
    pub sae_backend: u8,
    pub ssm_backend: u8,
    pub lfm_backend: u8,
    pub meta_digest: [u8; 32],
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NsrRecord {
    pub t: u64,
    pub decision_id: u64,
    pub evidence_chain_digest: [u8; 32],
    pub ruleset_id: u32,
    pub engine_id: &'static str,
    pub schema_version: u16,
    pub nsr_risk_q: u16,
    pub nsr_confidence_q: u16,
    pub policy_hint: u8,
    pub reasons: Vec<u16>,
    pub facts_digest: [u8; 32],
    pub assessment_digest: [u8; 32],
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
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
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
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
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
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
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
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_backend_pack(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        backend_pack_record: BackendPackRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::BackendPack,
            payload: ExperiencePayload::Empty,
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: Some(backend_pack_record),
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
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_neuro(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        neuro_record: NeuroRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::Neuro,
            payload: ExperiencePayload::Empty,
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: Some(neuro_record),
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_delta_proposal(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        delta_proposal_record: DeltaProposalRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::DeltaProposal,
            payload: ExperiencePayload::Empty,
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: None,
            delta_proposal_record: Some(delta_proposal_record),
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_delta_evaluation(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        delta_evaluation_record: DeltaEvaluationRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::DeltaEvaluation,
            payload: ExperiencePayload::Empty,
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: Some(delta_evaluation_record),
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_delta_recommendation(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        delta_recommendation_record: DeltaRecommendationRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::DeltaRecommendation,
            payload: ExperiencePayload::Empty,
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: Some(delta_recommendation_record),
            nsr_record: None,
            backend_pack_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_nsr(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        nsr_record: NsrRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::Nsr,
            payload: ExperiencePayload::Empty,
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: Some(nsr_record),
            backend_pack_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_candidate_set(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        candidate_set: CandidateSetRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::CandidateSet,
            payload: ExperiencePayload::Audit(AuditPayload::CandidateSet(candidate_set)),
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_output(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        output: OutputRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::Output,
            payload: ExperiencePayload::Audit(AuditPayload::Output(output)),
            neuromod: None,
            iit_phi: None,
            decision_meta: None,
            compute_summary: None,
            hormone_record: None,
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
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
            neuro_record: None,
            delta_proposal_record: None,
            delta_evaluation_record: None,
            delta_recommendation_record: None,
            nsr_record: None,
            backend_pack_record: None,
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
