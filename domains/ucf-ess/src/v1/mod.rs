pub mod errors;
pub mod governance;
pub mod record;
pub mod retrieval;
pub mod store;

pub use errors::EssError;
pub use governance::{apply_retention, DataClass, RetentionPolicyV1, RetentionStats};
pub use record::{
    compute_content_digest, AuditCheckpointRecord, AuditPayload, BackendPackRecord,
    CandidateSetRecord, CandidateSummaryRecord, CapabilityIssuanceRecord,
    ComputeBudgetViolationRecord, ComputeBudgetWindowRecord, DeltaEvaluationRecord,
    DeltaProposalRecord, DeltaRecommendationRecord, EbmConstraintProvenanceRecord,
    EbmEnvelopeViolationRecord, EbmReasoningRecord, EmergencyReasonCode, EmergencyRecord,
    EmergencyStateCode, ExperienceEbmTagRecord, ExperienceId, ExperienceKind, ExperiencePayload,
    ExperienceRecord, GpuParityRecord, GpuResourceViolationRecord, GpuUnavailableRecord,
    HormoneRecord, LfmSummaryRecord, LfmWindowRecord, LlmSummaryRecord, NeuroRecord, NsrRecord,
    OutputRecord, PayloadClassification, PolicyProvenanceRecord, RetrievalDecisionRecord,
    RetrievalReasonCode, RetrievalSelectionRecord, RetrievedExperienceRole, SaeSummaryRecord,
    SandboxCallRecord, SandboxReplyRecord, SsmSummaryRecord, ThrottleRecord, ToolAuthRecord,
    ToolExecutionRecord, ToolIssueAuditRecord, ToolPlanAuditRecord, ToolRequestRecord,
    WorldSummaryRecord,
};
pub use store::{ExperienceStore, IdAllocator, InMemoryEss};

pub use retrieval::{
    apply_ebm_bias, build_retrieval_decision_record, compute_query_digest_prefix,
    extract_retrieval_decision, find_ebm_energy, make_ebm_tag_from_reasoning, RetrievalCandidate,
    RetrievalContext, RetrievalPolicy, RetrievalResult,
};
