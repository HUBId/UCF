use std::sync::Arc;

use sha2::{Digest, Sha256};
use ucf_core::types::SimTime;
use ucf_frames::v1::{
    BrainFrame, ComputeSignalsSummary, ControlFrame, CorrelationId, DecisionFrame, DecisionMeta,
    NeuromodulatorSnapshot, PhiProxySnapshot,
};
use ucf_types::{quantize_unit, CANONICAL_UNIT_QUANT_MAX};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExperienceId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperienceKind {
    ControlIn,
    DecisionOut,
    BrainOut,
    Note,
    ToolRequest,
    ToolPlan,
    ToolIssue,
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
    EbmReasoning,
    EbmEnvelopeViolation,
    Output,
    BackendPack,
    LfmSummary,
    LfmWindow,
    CapabilityIssuance,
    Throttle,
    Emergency,
    PolicyProvenance,
    EbmConstraintProvenance,
    RemoteCall,
    RemoteCallDenied,
    ComputeBudgetWindow,
    ComputeBudgetViolation,
    RetrievalDecision,
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
    pub lfm_summary_record: Option<LfmSummaryRecord>,
    pub lfm_window_record: Option<LfmWindowRecord>,
    pub ebm_tag: Option<ExperienceEbmTagRecord>,
    pub audit_prev_digest: Option<[u8; 32]>,
    pub audit_digest: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperienceEbmTagRecord {
    pub decision_id: u64,
    pub evidence_chain_digest: [u8; 32],
    pub ebm_energy_min_q: u16,
    pub ebm_energy_mean_topk_q: u16,
    pub ebm_constraints_digest_prefix: [u8; 8],
    pub ebm_top_terms: Vec<(u16, u16)>,
    pub ebm_reasoning_digest_prefix: [u8; 8],
}

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ExperiencePayload {
    Control(ControlFrame),
    Decision(Box<DecisionFrame>),
    Brain(BrainFrame),
    Text(Arc<str>),
    Audit(AuditPayload),
    Empty,
}

impl ExperienceEbmTagRecord {
    pub const MAX_TOP_TERMS: usize = 4;

    pub fn clamp_bounds(mut self) -> Self {
        self.ebm_top_terms
            .sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        self.ebm_top_terms.truncate(Self::MAX_TOP_TERMS);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AuditPayload {
    ToolRequest(ToolRequestRecord),
    ToolPlan(ToolPlanAuditRecord),
    ToolIssue(ToolIssueAuditRecord),
    ToolAuth(ToolAuthRecord),
    ToolExecution(ToolExecutionRecord),
    SandboxCall(SandboxCallRecord),
    SandboxReply(SandboxReplyRecord),
    AuditCheckpoint(AuditCheckpointRecord),
    CandidateSet(CandidateSetRecord),
    EbmReasoning(EbmReasoningRecord),
    EbmEnvelopeViolation(EbmEnvelopeViolationRecord),
    Output(OutputRecord),
    CapabilityIssuance(CapabilityIssuanceRecord),
    Throttle(ThrottleRecord),
    Emergency(EmergencyRecord),
    PolicyProvenance(PolicyProvenanceRecord),
    EbmConstraintProvenance(EbmConstraintProvenanceRecord),
    RemoteCall(RemoteCallRecord),
    RemoteCallDenied(RemoteCallDeniedRecord),
    ComputeBudgetWindow(ComputeBudgetWindowRecord),
    ComputeBudgetViolation(ComputeBudgetViolationRecord),
    RetrievalDecision(RetrievalDecisionRecord),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievedExperienceRole {
    PrecedentSafe,
    Template,
    AvoidExample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalReasonCode {
    EbmBiasApplied,
    AvoidExamplesIncluded,
    HighRiskContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalSelectionRecord {
    pub experience_id: ExperienceId,
    pub experience_digest_prefix: [u8; 8],
    pub role: RetrievedExperienceRole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalDecisionRecord {
    pub schema_version: u16,
    pub t: u64,
    pub query_digest_prefix: [u8; 8],
    pub selected: Vec<RetrievalSelectionRecord>,
    pub low_energy_threshold_q: u16,
    pub high_energy_threshold_q: u16,
    pub policy_hash_prefix: [u8; 8],
    pub evidence_chain_digest_prefix: [u8; 8],
    pub reason_codes: Vec<RetrievalReasonCode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteCallRecord {
    pub t: u64,
    pub stage: String,
    pub endpoint_id: String,
    pub request_digest_prefix: [u8; 8],
    pub status: u16,
    pub elapsed_ms: u64,
    pub bytes_in: u32,
    pub bytes_out: u32,
    pub governor_tier: u8,
    pub policy_hash_prefix: [u8; 8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteCallDeniedRecord {
    pub t: u64,
    pub stage: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeBudgetWindowRecord {
    pub t0: u64,
    pub t1: u64,
    pub window: u64,
    pub primary_available_start: u64,
    pub primary_spent_start: u64,
    pub primary_available_end: u64,
    pub primary_spent_end: u64,
    pub shadow_available_start: u64,
    pub shadow_spent_start: u64,
    pub shadow_available_end: u64,
    pub shadow_spent_end: u64,
    pub llm_spent: u64,
    pub governor_spent: u64,
    pub jepa_spent: u64,
    pub sae_spent: u64,
    pub ssm_spent: u64,
    pub lfm_spent: u64,
    pub tool_spent: u64,
    pub governor_tier_mean_q: u16,
    pub governor_tier_max: u8,
    pub policy_hash_prefix: [u8; 8],
    pub schema_version: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeBudgetViolationRecord {
    pub t: u64,
    pub stage: String,
    pub pool: String,
    pub reason: String,
    pub attempted_cost: u64,
    pub available: u64,
    pub schema_version: u16,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityIssuanceRecord {
    pub governor_score_before_q: u16,
    pub governor_score_after_q: u16,
    pub ebm_penalty_q: u16,
    pub nsr_penalty_q: u16,
    pub ebm_energy_used_q: u16,
    pub policy_bundle_hash: String,
    pub policy_graph_digest: String,
    pub t: u64,
    pub decision_id: u64,
    pub candidate_id: Option<u16>,
    pub requested_kinds: Vec<String>,
    pub granted_kinds: Vec<String>,
    pub denied_kinds: Vec<(String, String)>,
    pub tier: u8,
    pub effective_tier: u8,
    pub emergency_override: bool,
    pub governor_score_q: u16,
    pub governance_signals_digest: [u8; 32],
    pub throttle_state_digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
    pub schema_version: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyProvenanceRecord {
    pub t: u64,
    pub run_id: String,
    pub bundle_version: String,
    pub bundle_hash: String,
    pub policy_graph_digest: String,
    pub base_pack_digest: String,
    pub overlay_pack_digest: Option<String>,
    pub enabled_features: Vec<String>,
    pub schema_version: u16,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmConstraintProvenanceRecord {
    pub t: u64,
    pub policy_hash_prefix: [u8; 8],
    pub constraints_digest_prefix: [u8; 8],
    pub term_count: u16,
    pub schema_version: u16,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmergencyReasonCode {
    RunawayV,
    TrendDV,
    Saturation,
    NanInf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmergencyStateCode {
    Armed,
    Active,
    Off,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmergencyRecord {
    pub policy_bundle_hash: String,
    pub policy_graph_digest: String,
    pub t: u64,
    pub state: EmergencyStateCode,
    pub reason: EmergencyReasonCode,
    pub v_q: u16,
    pub dv_q: u16,
    pub state_norm_q: u16,
    pub deriv_norm_q: u16,
    pub lfm_digest: [u8; 32],
    pub backend_pack_digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
    pub schema_version: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThrottleRecord {
    pub t: u64,
    pub kind: String,
    pub tokens_remaining: u16,
    pub cooldown_ticks: u16,
    pub deny_count: u16,
    pub digest: [u8; 32],
    pub schema_version: u16,
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
pub struct EbmReasoningRecord {
    pub suppressed_by_emergency: bool,
    pub schema_version: u16,
    pub t: u64,
    pub run_id: u64,
    pub decision_id: u64,
    pub backend_pack_digest_prefix: [u8; 8],
    pub ebm_backend_id: u8,
    pub ebm_model_digest_prefix: [u8; 8],
    pub contract_version: u16,
    pub enablement_mode: u8,
    pub risk_q: u16,
    pub pressure_q: u16,
    pub surprise_q: u16,
    pub uncertainty_q: u16,
    pub aggregate_energy_q: u16,
    pub base_energy_q: u16,
    pub top_energies_q: Vec<u16>,
    pub top_candidate_ids: Vec<u16>,
    pub ebm_digest_prefix: [u8; 8],
    pub constraints_digest_prefix: [u8; 8],
    pub top_term_contributions: Vec<(u16, u16)>,
    pub search_enabled: bool,
    pub search_steps_used: u8,
    pub evidence_chain_digest_prefix: [u8; 8],
    pub status: u8,
    pub reason_code: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmEnvelopeViolationRecord {
    pub schema_version: u16,
    pub t: u64,
    pub decision_id: u64,
    pub violation_code: u8,
    pub details: String,
}
#[derive(Debug, Clone, PartialEq)]
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
    pub content_digest: [u8; 32],
    pub text: Option<String>,
    pub redacted: bool,
    pub payload_len: Option<u32>,
    pub payload_classification: Option<PayloadClassification>,
    pub redaction_policy_marker: Option<String>,
    pub evidence_chain_digest: [u8; 32],
    pub lfm_readout_digest: Option<[u8; 32]>,
    pub lfm_uncertainty: Option<f32>,
    pub lfm_stability: Option<f32>,
    pub max_tokens_eff: u32,
    pub output_override: Option<u8>,
    pub override_reasons: Vec<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadClassification {
    Safe,
    Private,
}

pub fn compute_content_digest(text: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"UCF:ESS:OUTPUT:CONTENT:v1");
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolPlanAuditRecord {
    pub plan_digest_prefix: [u8; 8],
    pub tool_id: String,
    pub tool_class_id: String,
    pub args_digest_prefix: [u8; 8],
    pub required_caps: Vec<String>,
    pub ebm_energy_q: Option<u16>,
    pub nsr_risk_q: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolIssueAuditRecord {
    pub plan_digest_prefix: [u8; 8],
    pub issued: bool,
    pub issued_caps: Vec<[u8; 8]>,
    pub deny_reasons: Vec<String>,
    pub policy_graph_digest_prefix: [u8; 8],
    pub security_chain_digest_prefix: [u8; 8],
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
    pub window_stats_digest: [u8; 32],
    pub backend_pack_digest: [u8; 32],
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
    pub window_stats_digest: [u8; 32],
    pub backend_pack_digest: [u8; 32],
    pub suppression_reason_code: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaRecommendationRecord {
    pub schema_version: u16,
    pub delta_id: [u8; 32],
    pub recommended_ops: [u8; 128],
    pub safety_clamps: [u8; 64],
    pub requires_human_apply: bool,
    pub evidence_chain_digest: [u8; 32],
    pub window_stats_digest: [u8; 32],
    pub backend_pack_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendPackRecord {
    pub schema_version: u16,
    pub t: u64,
    pub pack_name: String,
    pub pack_id: u32,
    pub fixtures_digest: [u8; 32],
    pub model_hashes_digest: [u8; 32],
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
    pub nsr_status: u8,
    pub nsr_confidence_q: u16,
    pub rules_digest_prefix: [u8; 8],
    pub policy_hint: u8,
    pub reasons: Vec<u16>,
    pub facts_digest: [u8; 32],
    pub assessment_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfmSummaryRecord {
    pub t: u64,
    pub decision_id: Option<u64>,
    pub evidence_chain_digest: [u8; 32],
    pub backend_pack_digest: [u8; 32],
    pub liquid_state_digest: [u8; 32],
    pub liquid_readout_digest: [u8; 32],
    pub uncertainty: f32,
    pub stability: f32,
    pub schema_version: u16,
    pub digest: [u8; 32],
}

impl LfmSummaryRecord {
    pub fn with_digest(mut self) -> Self {
        self.digest = self.compute_digest();
        self
    }

    pub fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.t.to_be_bytes());
        match self.decision_id {
            Some(id) => {
                hasher.update([1]);
                hasher.update(id.to_be_bytes());
            }
            None => hasher.update([0]),
        }
        hasher.update(self.evidence_chain_digest);
        hasher.update(self.backend_pack_digest);
        hasher.update(self.liquid_state_digest);
        hasher.update(self.liquid_readout_digest);
        hasher.update(quantize_unit(self.uncertainty, CANONICAL_UNIT_QUANT_MAX).to_be_bytes());
        hasher.update(quantize_unit(self.stability, CANONICAL_UNIT_QUANT_MAX).to_be_bytes());
        hasher.update(self.schema_version.to_be_bytes());
        hasher.finalize().into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfmWindowRecord {
    pub t0: u64,
    pub t1: u64,
    pub sample_count: u16,
    pub mean_uncertainty: f32,
    pub mean_stability: f32,
    pub rolling_digest: [u8; 32],
    pub schema_version: u16,
    pub digest: [u8; 32],
}

impl LfmWindowRecord {
    pub fn with_digest(mut self) -> Self {
        self.digest = self.compute_digest();
        self
    }

    pub fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.t0.to_be_bytes());
        hasher.update(self.t1.to_be_bytes());
        hasher.update(self.sample_count.to_be_bytes());
        hasher.update(quantize_unit(self.mean_uncertainty, CANONICAL_UNIT_QUANT_MAX).to_be_bytes());
        hasher.update(quantize_unit(self.mean_stability, CANONICAL_UNIT_QUANT_MAX).to_be_bytes());
        hasher.update(self.rolling_digest);
        hasher.update(self.schema_version.to_be_bytes());
        hasher.finalize().into()
    }
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_ebm_reasoning(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        ebm_record: EbmReasoningRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::EbmReasoning,
            payload: ExperiencePayload::Audit(AuditPayload::EbmReasoning(ebm_record)),
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_retrieval_decision(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        record: RetrievalDecisionRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::RetrievalDecision,
            payload: ExperiencePayload::Audit(AuditPayload::RetrievalDecision(record)),
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_ebm_envelope_violation(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        record: EbmEnvelopeViolationRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::EbmEnvelopeViolation,
            payload: ExperiencePayload::Audit(AuditPayload::EbmEnvelopeViolation(record)),
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_lfm_summary(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        lfm_summary_record: LfmSummaryRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::LfmSummary,
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
            backend_pack_record: None,
            lfm_summary_record: Some(lfm_summary_record),
            lfm_window_record: None,
            ebm_tag: None,
            audit_prev_digest: None,
            audit_digest: None,
        }
    }

    pub fn from_lfm_window(
        id: ExperienceId,
        time: SimTime,
        corr: CorrelationId,
        lfm_window_record: LfmWindowRecord,
    ) -> Self {
        Self {
            id,
            time,
            corr,
            kind: ExperienceKind::LfmWindow,
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
            backend_pack_record: None,
            lfm_summary_record: None,
            lfm_window_record: Some(lfm_window_record),
            ebm_tag: None,
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
            lfm_summary_record: None,
            lfm_window_record: None,
            ebm_tag: None,
            audit_prev_digest: Some(prev_digest),
            audit_digest: Some(digest),
        }
    }

    pub fn with_ebm_tag(mut self, ebm_tag: ExperienceEbmTagRecord) -> Self {
        self.ebm_tag = Some(ebm_tag.clamp_bounds());
        self
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
