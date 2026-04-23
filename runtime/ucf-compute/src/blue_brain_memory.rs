use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const BLUE_BRAIN_MEMORY_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryCandidateClass {
    CommitEligible,
    Deferred,
    Rejected,
    Blocked,
    Insufficient,
    ReferenceOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryOrigin {
    Context,
    Evidence,
    Replay,
    Reference,
    ComputeResult,
    Selection,
    CommitFeedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryFreshness {
    Current,
    Stale,
    Partial,
    Caveated,
    Degraded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryCommitResultState {
    Committed,
    CommittedWithCaveat,
    Rejected,
    Blocked,
    Failed,
    NoOp,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryRetrievalState {
    RetrievedReferenceOnly,
    RetrievedWithCaveat,
    RetrievedStale,
    RetrievedInvalidated,
    Missing,
    Blocked,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryMaintenanceStatus {
    Current,
    Stale,
    Caveated,
    CaveatRefreshed,
    Invalidated,
    MaintenanceBlocked,
    RefreshUnavailable,
    NonCanonicalInternalOnlyPath,
}

impl Default for BlueBrainMemoryMaintenanceStatus {
    fn default() -> Self {
        Self::Current
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryCaveatRefreshState {
    Preserved,
    RefreshedFromReferenceOrEvidence,
    Strengthened,
    Weakened,
    RefreshUnavailable,
    RefreshBlocked,
}

impl Default for BlueBrainMemoryCaveatRefreshState {
    fn default() -> Self {
        Self::Preserved
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryMaintenanceResultState {
    Applied,
    NoOp,
    Blocked,
    Failed,
    Unavailable,
    Caveated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemoryDiagnosticClass {
    CommitDiagnostic,
    CommittedDiagnostic,
    CommittedWithCaveatDiagnostic,
    RejectedCommitDiagnostic,
    BlockedCommitDiagnostic,
    FailedCommitDiagnostic,
    NoOpCommitDiagnostic,
    RetrievalDiagnostic,
    RetrievedDiagnostic,
    MissingMemoryDiagnostic,
    StaleMemoryDiagnostic,
    CaveatedMemoryDiagnostic,
    CaveatRefreshedMemoryDiagnostic,
    InvalidatedMemoryDiagnostic,
    MaintenanceBlockedMemoryDiagnostic,
    RefreshUnavailableMemoryDiagnostic,
    MaintenanceDiagnostic,
    UnavailableMemoryDiagnostic,
    NonCanonicalInternalOnlyMemoryDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainMemoryDiagnosticLane {
    pub diagnostic_class: BlueBrainMemoryDiagnosticClass,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP: [BlueBrainMemoryDiagnosticLane; 19] = [
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::CommitDiagnostic,
        lane: "blue_brain_memory_commit_diagnostic",
        canonical_guard: "commit diagnostics must mirror store commit outcome without fabrication",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::CommittedDiagnostic,
        lane: "blue_brain_memory_committed_diagnostic",
        canonical_guard:
            "committed diagnostics require memory_record_id created in canonical store",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::CommittedWithCaveatDiagnostic,
        lane: "blue_brain_memory_committed_with_caveat_diagnostic",
        canonical_guard: "committed-with-caveat preserves persisted caveats",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::RejectedCommitDiagnostic,
        lane: "blue_brain_memory_commit_rejected_diagnostic",
        canonical_guard: "rejected commit diagnostics represent ineligible/rejected candidates",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::BlockedCommitDiagnostic,
        lane: "blue_brain_memory_commit_blocked_diagnostic",
        canonical_guard: "blocked commit diagnostics represent stale/internal/insufficient guards",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::FailedCommitDiagnostic,
        lane: "blue_brain_memory_commit_failed_diagnostic",
        canonical_guard: "failed commit diagnostics represent real store I/O/encode failures",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::NoOpCommitDiagnostic,
        lane: "blue_brain_memory_commit_noop_diagnostic",
        canonical_guard:
            "no-op commit diagnostics represent duplicate or equivalent persisted records",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::RetrievalDiagnostic,
        lane: "blue_brain_memory_retrieval_diagnostic",
        canonical_guard: "retrieval diagnostics must mirror canonical read/reference result",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::RetrievedDiagnostic,
        lane: "blue_brain_memory_retrieved_diagnostic",
        canonical_guard: "retrieved diagnostics require persisted memory reference returned",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::MissingMemoryDiagnostic,
        lane: "blue_brain_memory_missing_diagnostic",
        canonical_guard:
            "missing diagnostics indicate no persisted memory record for requested locator",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::StaleMemoryDiagnostic,
        lane: "blue_brain_memory_stale_diagnostic",
        canonical_guard: "stale diagnostics represent retrieved stale memory reference",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::CaveatedMemoryDiagnostic,
        lane: "blue_brain_memory_caveated_diagnostic",
        canonical_guard: "caveated diagnostics represent caveated commit/retrieval posture",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::CaveatRefreshedMemoryDiagnostic,
        lane: "blue_brain_memory_caveat_refreshed_diagnostic",
        canonical_guard: "caveat-refreshed diagnostics preserve refreshed caveat basis without auto-validation",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::InvalidatedMemoryDiagnostic,
        lane: "blue_brain_memory_invalidated_diagnostic",
        canonical_guard: "invalidated diagnostics prevent treating memory as strong candidate/proposal basis",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::MaintenanceBlockedMemoryDiagnostic,
        lane: "blue_brain_memory_maintenance_blocked_diagnostic",
        canonical_guard: "maintenance-blocked diagnostics preserve blocked maintenance guardrails",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::RefreshUnavailableMemoryDiagnostic,
        lane: "blue_brain_memory_refresh_unavailable_diagnostic",
        canonical_guard: "refresh-unavailable diagnostics keep caveat refresh unavailable explicit",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::MaintenanceDiagnostic,
        lane: "blue_brain_memory_maintenance_diagnostic",
        canonical_guard: "maintenance diagnostics represent canonical maintenance/invalidation/caveat refresh operations",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::UnavailableMemoryDiagnostic,
        lane: "blue_brain_memory_unavailable_diagnostic",
        canonical_guard:
            "unavailable diagnostics represent canonical store/retrieval path unavailable",
    },
    BlueBrainMemoryDiagnosticLane {
        diagnostic_class: BlueBrainMemoryDiagnosticClass::NonCanonicalInternalOnlyMemoryDiagnostic,
        lane: "blue_brain_memory_non_canonical_internal_only_diagnostic",
        canonical_guard:
            "internal/expert-only hooks are non-canonical unless explicitly down-mapped",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueBrainMemorySelectionDisposition {
    Selected,
    Supporting,
    Deferred,
    Ignored,
    Insufficient,
    Caveated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMemoryRuntimeFeedbackClass {
    MemoryCommitted,
    MemoryRetrieved,
    MemoryRetrievalMissingOrStaleOrCaveated,
    CommitOrRetrievalBlocked,
    CommitOrRetrievalFailedOrUnavailable,
    FeedbackObservedNoAutoComputeOrAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMemoryContextFeedbackClass {
    CommittedMemoryAttachedToCurrentContext,
    RetrievedMemoryAttachedToCurrentContext,
    MemoryCaveatCarriedIntoContext,
    StaleOrMissingMemoryLimitsContextUpdate,
    NoAutomaticMemoryCandidateCreation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMemorySelectionCandidateProposalFeedbackClass {
    RetrievedMemorySupportsCandidateBasis,
    StaleOrMissingMemoryWeakensCandidateBasis,
    CommittedMemoryMaySupportFutureProposalBasis,
    CaveatedMemoryYieldsCaveatedSelectionOrProposal,
    RetrievalDoesNotAutomaticallySelectProposeOrExecute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMemoryFeedbackBackbind {
    pub runtime_feedback: Vec<BlueBrainMemoryRuntimeFeedbackClass>,
    pub context_feedback: Vec<BlueBrainMemoryContextFeedbackClass>,
    pub selection_candidate_proposal_feedback:
        Vec<BlueBrainMemorySelectionCandidateProposalFeedbackClass>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMemoryCandidate {
    pub candidate_id: String,
    pub class: BlueBrainMemoryCandidateClass,
    pub origins: Vec<BlueBrainMemoryOrigin>,
    pub evidence_refs: Vec<String>,
    pub reference_refs: Vec<String>,
    pub context_basis_refs: Vec<String>,
    pub selection_basis_refs: Vec<String>,
    pub freshness: BlueBrainMemoryFreshness,
    pub caveats: Vec<String>,
    pub allow_caveated_commit: bool,
    pub allow_stale_context_commit: bool,
    pub has_internal_only_dependency: bool,
    pub commit_path_available: bool,
}

impl BlueBrainMemoryCandidate {
    pub fn canonicalized(mut self) -> Self {
        self.origins.sort_unstable_by_key(|origin| *origin as u8);
        self.origins.dedup();
        self.evidence_refs.sort_unstable();
        self.evidence_refs.dedup();
        self.reference_refs.sort_unstable();
        self.reference_refs.dedup();
        self.context_basis_refs.sort_unstable();
        self.context_basis_refs.dedup();
        self.selection_basis_refs.sort_unstable();
        self.selection_basis_refs.dedup();
        self.caveats.sort_unstable();
        self.caveats.dedup();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedBlueBrainMemoryRecord {
    pub schema_version: u16,
    pub memory_record_id: String,
    pub source_candidate_id: String,
    pub origins: Vec<BlueBrainMemoryOrigin>,
    pub evidence_refs: Vec<String>,
    pub reference_refs: Vec<String>,
    pub context_basis_refs: Vec<String>,
    pub selection_basis_refs: Vec<String>,
    pub freshness: BlueBrainMemoryFreshness,
    pub caveats: Vec<String>,
    #[serde(default)]
    pub maintenance_status: BlueBrainMemoryMaintenanceStatus,
    #[serde(default)]
    pub caveat_refresh_state: BlueBrainMemoryCaveatRefreshState,
    #[serde(default)]
    pub maintenance_note: Option<String>,
    #[serde(default)]
    pub maintenance_updated_at_unix_ms: Option<u64>,
    pub committed_at_unix_ms: u64,
    pub commit_result_state: BlueBrainMemoryCommitResultState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMemoryCommitReport {
    pub candidate_id: String,
    pub result_state: BlueBrainMemoryCommitResultState,
    pub memory_record_id: Option<String>,
    pub created_record: bool,
    pub diagnostic_class: BlueBrainMemoryDiagnosticClass,
    pub diagnostic: String,
    pub caveats: Vec<String>,
    pub feedback_backbind: BlueBrainMemoryFeedbackBackbind,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlueBrainMemoryReferenceMetadata {
    pub schema_version: u16,
    pub committed_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlueBrainMemoryReferenceRecord {
    pub memory_record_id: String,
    pub source_candidate_id: String,
    pub commit_result_state: BlueBrainMemoryCommitResultState,
    pub evidence_refs: Vec<String>,
    pub reference_refs: Vec<String>,
    pub context_basis_refs: Vec<String>,
    pub selection_basis_refs: Vec<String>,
    pub freshness: BlueBrainMemoryFreshness,
    pub caveats: Vec<String>,
    pub maintenance_status: BlueBrainMemoryMaintenanceStatus,
    pub caveat_refresh_state: BlueBrainMemoryCaveatRefreshState,
    pub maintenance_note: Option<String>,
    pub maintenance_updated_at_unix_ms: Option<u64>,
    pub metadata: BlueBrainMemoryReferenceMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMemoryMaintenanceLocator<'a> {
    MemoryRecordId(&'a str),
    SourceCandidateId(&'a str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlueBrainMemoryMaintenanceAction {
    MarkCurrent,
    MarkStale,
    Invalidate {
        reason: String,
    },
    MarkMaintenanceBlocked {
        reason: String,
    },
    MarkRefreshUnavailable {
        reason: String,
    },
    RefreshCaveats {
        caveats: Vec<String>,
        refresh_state: BlueBrainMemoryCaveatRefreshState,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMemoryMaintenanceRequest<'a> {
    pub locator: BlueBrainMemoryMaintenanceLocator<'a>,
    pub action: BlueBrainMemoryMaintenanceAction,
    pub canonical_maintenance_path_available: bool,
    pub allow_internal_only_locator: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMemoryMaintenanceReport {
    pub result_state: BlueBrainMemoryMaintenanceResultState,
    pub memory_record_id: Option<String>,
    pub maintenance_status: Option<BlueBrainMemoryMaintenanceStatus>,
    pub caveat_refresh_state: Option<BlueBrainMemoryCaveatRefreshState>,
    pub diagnostic_class: BlueBrainMemoryDiagnosticClass,
    pub diagnostic: String,
    pub caveats: Vec<String>,
    pub automatic_compute_triggered: bool,
    pub automatic_action_or_planning_triggered: bool,
    pub automatic_memory_commit_triggered: bool,
    pub feedback_backbind: BlueBrainMemoryFeedbackBackbind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMemoryReferenceLocator<'a> {
    MemoryRecordId(&'a str),
    SourceCandidateId(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainMemoryReadRequest<'a> {
    pub locator: BlueBrainMemoryReferenceLocator<'a>,
    pub canonical_retrieval_path_available: bool,
    pub allow_internal_only_locator: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMemoryReadResult {
    pub retrieval_state: BlueBrainMemoryRetrievalState,
    pub reference: Option<BlueBrainMemoryReferenceRecord>,
    pub diagnostic_class: BlueBrainMemoryDiagnosticClass,
    pub diagnostic: String,
    pub context_attached: bool,
    pub context_caveated: bool,
    pub context_stale: bool,
    pub context_insufficient_for_candidate_or_proposal: bool,
    pub automatic_compute_triggered: bool,
    pub automatic_action_or_planning_triggered: bool,
    pub automatic_memory_commit_triggered: bool,
    pub selection_disposition: BlueBrainMemorySelectionDisposition,
    pub feedback_backbind: BlueBrainMemoryFeedbackBackbind,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BlueBrainMemoryStoreError {
    #[error("blue-brain memory store io error during {operation} at {path}: {reason}")]
    Io {
        operation: &'static str,
        path: String,
        reason: String,
    },
    #[error("blue-brain memory store corrupted at line {line}: {reason}")]
    Corrupt { line: usize, reason: String },
    #[error("blue-brain memory encode failure: {reason}")]
    Encode { reason: String },
}

#[derive(Debug, Clone)]
pub struct BlueBrainMemoryStore {
    path: PathBuf,
    records: BTreeMap<String, PersistedBlueBrainMemoryRecord>,
    candidate_index: BTreeMap<String, String>,
}

impl BlueBrainMemoryStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, BlueBrainMemoryStoreError> {
        let path = path.into();
        let mut records = BTreeMap::new();
        let mut candidate_index = BTreeMap::new();
        if path.exists() {
            let file = fs::File::open(&path).map_err(|err| BlueBrainMemoryStoreError::Io {
                operation: "open",
                path: path.display().to_string(),
                reason: err.to_string(),
            })?;
            for (line_idx, line) in BufReader::new(file).lines().enumerate() {
                let line = line.map_err(|err| BlueBrainMemoryStoreError::Io {
                    operation: "read",
                    path: path.display().to_string(),
                    reason: err.to_string(),
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let parsed: PersistedBlueBrainMemoryRecord =
                    serde_json::from_str(&line).map_err(|err| {
                        BlueBrainMemoryStoreError::Corrupt {
                            line: line_idx + 1,
                            reason: err.to_string(),
                        }
                    })?;
                candidate_index.insert(
                    parsed.source_candidate_id.clone(),
                    parsed.memory_record_id.clone(),
                );
                records.insert(parsed.memory_record_id.clone(), parsed);
            }
        }
        Ok(Self {
            path,
            records,
            candidate_index,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, memory_record_id: &str) -> Option<&PersistedBlueBrainMemoryRecord> {
        self.records.get(memory_record_id)
    }

    pub fn get_by_candidate(&self, candidate_id: &str) -> Option<&PersistedBlueBrainMemoryRecord> {
        self.candidate_index
            .get(candidate_id)
            .and_then(|id| self.get(id))
    }

    pub fn read_reference(
        &self,
        request: BlueBrainMemoryReadRequest<'_>,
    ) -> BlueBrainMemoryReadResult {
        if !request.canonical_retrieval_path_available {
            return BlueBrainMemoryReadResult {
                retrieval_state: BlueBrainMemoryRetrievalState::Unavailable,
                reference: None,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::UnavailableMemoryDiagnostic,
                diagnostic: "canonical memory retrieval path unavailable".to_string(),
                context_attached: false,
                context_caveated: false,
                context_stale: false,
                context_insufficient_for_candidate_or_proposal: true,
                automatic_compute_triggered: false,
                automatic_action_or_planning_triggered: false,
                automatic_memory_commit_triggered: false,
                selection_disposition: BlueBrainMemorySelectionDisposition::Insufficient,
                feedback_backbind: BlueBrainMemoryFeedbackBackbind {
                    runtime_feedback: vec![
                        BlueBrainMemoryRuntimeFeedbackClass::CommitOrRetrievalFailedOrUnavailable,
                        BlueBrainMemoryRuntimeFeedbackClass::FeedbackObservedNoAutoComputeOrAction,
                    ],
                    context_feedback: vec![
                        BlueBrainMemoryContextFeedbackClass::StaleOrMissingMemoryLimitsContextUpdate,
                        BlueBrainMemoryContextFeedbackClass::NoAutomaticMemoryCandidateCreation,
                    ],
                    selection_candidate_proposal_feedback: vec![
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::StaleOrMissingMemoryWeakensCandidateBasis,
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievalDoesNotAutomaticallySelectProposeOrExecute,
                    ],
                },
            };
        }

        let locator = match request.locator {
            BlueBrainMemoryReferenceLocator::MemoryRecordId(id) => id,
            BlueBrainMemoryReferenceLocator::SourceCandidateId(id) => id,
        };
        if !request.allow_internal_only_locator && locator.starts_with("internal:") {
            return BlueBrainMemoryReadResult {
                retrieval_state: BlueBrainMemoryRetrievalState::Blocked,
                reference: None,
                diagnostic_class:
                    BlueBrainMemoryDiagnosticClass::NonCanonicalInternalOnlyMemoryDiagnostic,
                diagnostic: "retrieval blocked for internal/non-canonical locator".to_string(),
                context_attached: false,
                context_caveated: false,
                context_stale: false,
                context_insufficient_for_candidate_or_proposal: true,
                automatic_compute_triggered: false,
                automatic_action_or_planning_triggered: false,
                automatic_memory_commit_triggered: false,
                selection_disposition: BlueBrainMemorySelectionDisposition::Insufficient,
                feedback_backbind: BlueBrainMemoryFeedbackBackbind {
                    runtime_feedback: vec![
                        BlueBrainMemoryRuntimeFeedbackClass::CommitOrRetrievalBlocked,
                        BlueBrainMemoryRuntimeFeedbackClass::FeedbackObservedNoAutoComputeOrAction,
                    ],
                    context_feedback: vec![
                        BlueBrainMemoryContextFeedbackClass::StaleOrMissingMemoryLimitsContextUpdate,
                        BlueBrainMemoryContextFeedbackClass::NoAutomaticMemoryCandidateCreation,
                    ],
                    selection_candidate_proposal_feedback: vec![
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::StaleOrMissingMemoryWeakensCandidateBasis,
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievalDoesNotAutomaticallySelectProposeOrExecute,
                    ],
                },
            };
        }

        let record = match request.locator {
            BlueBrainMemoryReferenceLocator::MemoryRecordId(memory_record_id) => {
                self.get(memory_record_id)
            }
            BlueBrainMemoryReferenceLocator::SourceCandidateId(candidate_id) => {
                self.get_by_candidate(candidate_id)
            }
        };

        let Some(record) = record else {
            return BlueBrainMemoryReadResult {
                retrieval_state: BlueBrainMemoryRetrievalState::Missing,
                reference: None,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::MissingMemoryDiagnostic,
                diagnostic: "memory reference missing in canonical persisted memory store"
                    .to_string(),
                context_attached: false,
                context_caveated: false,
                context_stale: false,
                context_insufficient_for_candidate_or_proposal: true,
                automatic_compute_triggered: false,
                automatic_action_or_planning_triggered: false,
                automatic_memory_commit_triggered: false,
                selection_disposition: BlueBrainMemorySelectionDisposition::Insufficient,
                feedback_backbind: BlueBrainMemoryFeedbackBackbind {
                    runtime_feedback: vec![
                        BlueBrainMemoryRuntimeFeedbackClass::MemoryRetrievalMissingOrStaleOrCaveated,
                        BlueBrainMemoryRuntimeFeedbackClass::FeedbackObservedNoAutoComputeOrAction,
                    ],
                    context_feedback: vec![
                        BlueBrainMemoryContextFeedbackClass::StaleOrMissingMemoryLimitsContextUpdate,
                        BlueBrainMemoryContextFeedbackClass::NoAutomaticMemoryCandidateCreation,
                    ],
                    selection_candidate_proposal_feedback: vec![
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::StaleOrMissingMemoryWeakensCandidateBasis,
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievalDoesNotAutomaticallySelectProposeOrExecute,
                    ],
                },
            };
        };

        let retrieval_state = if matches!(
            record.maintenance_status,
            BlueBrainMemoryMaintenanceStatus::Invalidated
        ) {
            BlueBrainMemoryRetrievalState::RetrievedInvalidated
        } else if record.freshness == BlueBrainMemoryFreshness::Stale
            || matches!(
                record.maintenance_status,
                BlueBrainMemoryMaintenanceStatus::Stale
            )
        {
            BlueBrainMemoryRetrievalState::RetrievedStale
        } else if matches!(
            record.maintenance_status,
            BlueBrainMemoryMaintenanceStatus::MaintenanceBlocked
                | BlueBrainMemoryMaintenanceStatus::NonCanonicalInternalOnlyPath
        ) {
            BlueBrainMemoryRetrievalState::Blocked
        } else if !record.caveats.is_empty()
            || record.freshness != BlueBrainMemoryFreshness::Current
            || !matches!(
                record.maintenance_status,
                BlueBrainMemoryMaintenanceStatus::Current
            )
        {
            BlueBrainMemoryRetrievalState::RetrievedWithCaveat
        } else {
            BlueBrainMemoryRetrievalState::RetrievedReferenceOnly
        };

        let context_caveated =
            retrieval_state == BlueBrainMemoryRetrievalState::RetrievedWithCaveat;
        let context_stale = matches!(
            retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedStale
                | BlueBrainMemoryRetrievalState::RetrievedInvalidated
                | BlueBrainMemoryRetrievalState::Blocked
        );
        let context_insufficient = record.selection_basis_refs.is_empty();
        let diagnostic_class = match retrieval_state {
            BlueBrainMemoryRetrievalState::RetrievedReferenceOnly => {
                BlueBrainMemoryDiagnosticClass::RetrievedDiagnostic
            }
            BlueBrainMemoryRetrievalState::RetrievedWithCaveat => {
                if matches!(
                    record.maintenance_status,
                    BlueBrainMemoryMaintenanceStatus::CaveatRefreshed
                ) {
                    BlueBrainMemoryDiagnosticClass::CaveatRefreshedMemoryDiagnostic
                } else if matches!(
                    record.maintenance_status,
                    BlueBrainMemoryMaintenanceStatus::RefreshUnavailable
                ) {
                    BlueBrainMemoryDiagnosticClass::RefreshUnavailableMemoryDiagnostic
                } else {
                    BlueBrainMemoryDiagnosticClass::CaveatedMemoryDiagnostic
                }
            }
            BlueBrainMemoryRetrievalState::RetrievedStale => {
                BlueBrainMemoryDiagnosticClass::StaleMemoryDiagnostic
            }
            BlueBrainMemoryRetrievalState::RetrievedInvalidated => {
                BlueBrainMemoryDiagnosticClass::InvalidatedMemoryDiagnostic
            }
            BlueBrainMemoryRetrievalState::Missing => {
                BlueBrainMemoryDiagnosticClass::MissingMemoryDiagnostic
            }
            BlueBrainMemoryRetrievalState::Blocked => {
                if matches!(
                    record.maintenance_status,
                    BlueBrainMemoryMaintenanceStatus::NonCanonicalInternalOnlyPath
                ) {
                    BlueBrainMemoryDiagnosticClass::NonCanonicalInternalOnlyMemoryDiagnostic
                } else {
                    BlueBrainMemoryDiagnosticClass::MaintenanceBlockedMemoryDiagnostic
                }
            }
            BlueBrainMemoryRetrievalState::Unavailable => {
                BlueBrainMemoryDiagnosticClass::UnavailableMemoryDiagnostic
            }
        };
        let selection_disposition = if matches!(
            retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedInvalidated
                | BlueBrainMemoryRetrievalState::Blocked
        ) {
            BlueBrainMemorySelectionDisposition::Insufficient
        } else if context_stale {
            BlueBrainMemorySelectionDisposition::Deferred
        } else if context_insufficient {
            BlueBrainMemorySelectionDisposition::Insufficient
        } else if context_caveated {
            BlueBrainMemorySelectionDisposition::Caveated
        } else {
            BlueBrainMemorySelectionDisposition::Supporting
        };

        BlueBrainMemoryReadResult {
            retrieval_state,
            reference: Some(BlueBrainMemoryReferenceRecord {
                memory_record_id: record.memory_record_id.clone(),
                source_candidate_id: record.source_candidate_id.clone(),
                commit_result_state: record.commit_result_state,
                evidence_refs: record.evidence_refs.clone(),
                reference_refs: record.reference_refs.clone(),
                context_basis_refs: record.context_basis_refs.clone(),
                selection_basis_refs: record.selection_basis_refs.clone(),
                freshness: record.freshness,
                caveats: record.caveats.clone(),
                maintenance_status: record.maintenance_status,
                caveat_refresh_state: record.caveat_refresh_state,
                maintenance_note: record.maintenance_note.clone(),
                maintenance_updated_at_unix_ms: record.maintenance_updated_at_unix_ms,
                metadata: BlueBrainMemoryReferenceMetadata {
                    schema_version: record.schema_version,
                    committed_at_unix_ms: record.committed_at_unix_ms,
                },
            }),
            diagnostic_class,
            diagnostic: "memory reference observed and attached to current context".to_string(),
            context_attached: true,
            context_caveated,
            context_stale,
            context_insufficient_for_candidate_or_proposal: context_insufficient,
            automatic_compute_triggered: false,
            automatic_action_or_planning_triggered: false,
            automatic_memory_commit_triggered: false,
            selection_disposition,
            feedback_backbind: BlueBrainMemoryFeedbackBackbind {
                runtime_feedback: vec![
                    BlueBrainMemoryRuntimeFeedbackClass::MemoryRetrieved,
                    if context_caveated || context_stale {
                        BlueBrainMemoryRuntimeFeedbackClass::MemoryRetrievalMissingOrStaleOrCaveated
                    } else {
                        BlueBrainMemoryRuntimeFeedbackClass::FeedbackObservedNoAutoComputeOrAction
                    },
                    BlueBrainMemoryRuntimeFeedbackClass::FeedbackObservedNoAutoComputeOrAction,
                ],
                context_feedback: vec![
                    BlueBrainMemoryContextFeedbackClass::RetrievedMemoryAttachedToCurrentContext,
                    if context_caveated {
                        BlueBrainMemoryContextFeedbackClass::MemoryCaveatCarriedIntoContext
                    } else {
                        BlueBrainMemoryContextFeedbackClass::NoAutomaticMemoryCandidateCreation
                    },
                    if context_stale || context_insufficient {
                        BlueBrainMemoryContextFeedbackClass::StaleOrMissingMemoryLimitsContextUpdate
                    } else {
                        BlueBrainMemoryContextFeedbackClass::NoAutomaticMemoryCandidateCreation
                    },
                    BlueBrainMemoryContextFeedbackClass::NoAutomaticMemoryCandidateCreation,
                ],
                selection_candidate_proposal_feedback: vec![
                    if context_stale || context_insufficient {
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::StaleOrMissingMemoryWeakensCandidateBasis
                    } else {
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievedMemorySupportsCandidateBasis
                    },
                    if context_caveated {
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::CaveatedMemoryYieldsCaveatedSelectionOrProposal
                    } else {
                        BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievedMemorySupportsCandidateBasis
                    },
                    BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievalDoesNotAutomaticallySelectProposeOrExecute,
                ],
            },
        }
    }

    pub fn commit_candidate(
        &mut self,
        candidate: BlueBrainMemoryCandidate,
        committed_at_unix_ms: u64,
    ) -> BlueBrainMemoryCommitReport {
        let candidate = candidate.canonicalized();
        let mut caveats = candidate.caveats.clone();

        if !candidate.commit_path_available {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Unavailable,
                memory_record_id: None,
                created_record: false,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::UnavailableMemoryDiagnostic,
                diagnostic: "canonical memory store unavailable".to_string(),
                caveats: caveats.clone(),
                feedback_backbind: commit_feedback(
                    BlueBrainMemoryCommitResultState::Unavailable,
                    false,
                    false,
                ),
            };
        }

        if let Some(existing) = self.get_by_candidate(&candidate.candidate_id) {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::NoOp,
                memory_record_id: Some(existing.memory_record_id.clone()),
                created_record: false,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::NoOpCommitDiagnostic,
                diagnostic: "candidate already committed in canonical memory store".to_string(),
                caveats: caveats.clone(),
                feedback_backbind: commit_feedback(
                    BlueBrainMemoryCommitResultState::NoOp,
                    false,
                    !caveats.is_empty(),
                ),
            };
        }

        let eligibility_guard = match candidate.class {
            BlueBrainMemoryCandidateClass::CommitEligible => None,
            BlueBrainMemoryCandidateClass::Deferred => Some((
                BlueBrainMemoryCommitResultState::Blocked,
                "candidate deferred and not commit-eligible",
            )),
            BlueBrainMemoryCandidateClass::Rejected => Some((
                BlueBrainMemoryCommitResultState::Rejected,
                "candidate rejected and not commit-eligible",
            )),
            BlueBrainMemoryCandidateClass::Blocked => Some((
                BlueBrainMemoryCommitResultState::Blocked,
                "candidate blocked and not commit-eligible",
            )),
            BlueBrainMemoryCandidateClass::Insufficient => Some((
                BlueBrainMemoryCommitResultState::Blocked,
                "candidate insufficient and not commit-eligible",
            )),
            BlueBrainMemoryCandidateClass::ReferenceOnly => Some((
                BlueBrainMemoryCommitResultState::Rejected,
                "reference-only candidate cannot be committed as memory",
            )),
        };
        if let Some((state, diagnostic)) = eligibility_guard {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: state,
                memory_record_id: None,
                created_record: false,
                diagnostic_class: map_commit_diagnostic_class(state, false),
                diagnostic: diagnostic.to_string(),
                caveats: caveats.clone(),
                feedback_backbind: commit_feedback(state, false, !caveats.is_empty()),
            };
        }

        if candidate.has_internal_only_dependency {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Blocked,
                memory_record_id: None,
                created_record: false,
                diagnostic_class:
                    BlueBrainMemoryDiagnosticClass::NonCanonicalInternalOnlyMemoryDiagnostic,
                diagnostic: "candidate depends on internal/expert-only basis".to_string(),
                caveats: caveats.clone(),
                feedback_backbind: commit_feedback(
                    BlueBrainMemoryCommitResultState::Blocked,
                    false,
                    !caveats.is_empty(),
                ),
            };
        }

        let has_basis = !candidate.evidence_refs.is_empty()
            || !candidate.reference_refs.is_empty()
            || !candidate.context_basis_refs.is_empty();
        if !has_basis && !candidate.allow_caveated_commit {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Rejected,
                memory_record_id: None,
                created_record: false,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::RejectedCommitDiagnostic,
                diagnostic: "missing evidence/reference/context basis for commit".to_string(),
                caveats: caveats.clone(),
                feedback_backbind: commit_feedback(
                    BlueBrainMemoryCommitResultState::Rejected,
                    false,
                    !caveats.is_empty(),
                ),
            };
        }

        let stale_context = candidate.freshness == BlueBrainMemoryFreshness::Stale;
        if stale_context && !candidate.allow_stale_context_commit {
            return BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Blocked,
                memory_record_id: None,
                created_record: false,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::BlockedCommitDiagnostic,
                diagnostic: "stale context basis blocked commit".to_string(),
                caveats: caveats.clone(),
                feedback_backbind: commit_feedback(
                    BlueBrainMemoryCommitResultState::Blocked,
                    false,
                    !caveats.is_empty(),
                ),
            };
        }

        let result_state = if candidate.allow_caveated_commit
            || stale_context
            || !candidate.caveats.is_empty()
            || candidate.freshness != BlueBrainMemoryFreshness::Current
        {
            if caveats.is_empty() {
                caveats.push("commit accepted with caveat posture".to_string());
            }
            BlueBrainMemoryCommitResultState::CommittedWithCaveat
        } else {
            BlueBrainMemoryCommitResultState::Committed
        };

        let memory_record_id = build_memory_record_id(&candidate, committed_at_unix_ms);
        let persisted = PersistedBlueBrainMemoryRecord {
            schema_version: BLUE_BRAIN_MEMORY_SCHEMA_VERSION,
            memory_record_id: memory_record_id.clone(),
            source_candidate_id: candidate.candidate_id.clone(),
            origins: candidate.origins,
            evidence_refs: candidate.evidence_refs,
            reference_refs: candidate.reference_refs,
            context_basis_refs: candidate.context_basis_refs,
            selection_basis_refs: candidate.selection_basis_refs,
            freshness: candidate.freshness,
            caveats: caveats.clone(),
            maintenance_status: if stale_context {
                BlueBrainMemoryMaintenanceStatus::Stale
            } else if !caveats.is_empty() {
                BlueBrainMemoryMaintenanceStatus::Caveated
            } else {
                BlueBrainMemoryMaintenanceStatus::Current
            },
            caveat_refresh_state: BlueBrainMemoryCaveatRefreshState::Preserved,
            maintenance_note: None,
            maintenance_updated_at_unix_ms: None,
            committed_at_unix_ms,
            commit_result_state: result_state,
        };

        match self.upsert(persisted) {
            Ok(()) => BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state,
                memory_record_id: Some(memory_record_id),
                created_record: true,
                diagnostic_class: map_commit_diagnostic_class(result_state, true),
                diagnostic: "candidate committed into canonical persisted memory store".to_string(),
                caveats: caveats.clone(),
                feedback_backbind: commit_feedback(result_state, true, !caveats.is_empty()),
            },
            Err(err) => BlueBrainMemoryCommitReport {
                candidate_id: candidate.candidate_id,
                result_state: BlueBrainMemoryCommitResultState::Failed,
                memory_record_id: None,
                created_record: false,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::FailedCommitDiagnostic,
                diagnostic: format!("memory store write failed: {err}"),
                caveats: caveats.clone(),
                feedback_backbind: commit_feedback(
                    BlueBrainMemoryCommitResultState::Failed,
                    false,
                    !caveats.is_empty(),
                ),
            },
        }
    }

    pub fn apply_maintenance(
        &mut self,
        request: BlueBrainMemoryMaintenanceRequest<'_>,
        maintenance_updated_at_unix_ms: u64,
    ) -> BlueBrainMemoryMaintenanceReport {
        if !request.canonical_maintenance_path_available {
            return BlueBrainMemoryMaintenanceReport {
                result_state: BlueBrainMemoryMaintenanceResultState::Unavailable,
                memory_record_id: None,
                maintenance_status: None,
                caveat_refresh_state: None,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::UnavailableMemoryDiagnostic,
                diagnostic: "canonical memory maintenance path unavailable".to_string(),
                caveats: Vec::new(),
                automatic_compute_triggered: false,
                automatic_action_or_planning_triggered: false,
                automatic_memory_commit_triggered: false,
                feedback_backbind: maintenance_feedback(
                    BlueBrainMemoryMaintenanceResultState::Unavailable,
                    false,
                    false,
                ),
            };
        }

        let locator = match request.locator {
            BlueBrainMemoryMaintenanceLocator::MemoryRecordId(id) => id,
            BlueBrainMemoryMaintenanceLocator::SourceCandidateId(id) => id,
        };
        if !request.allow_internal_only_locator && locator.starts_with("internal:") {
            return BlueBrainMemoryMaintenanceReport {
                result_state: BlueBrainMemoryMaintenanceResultState::Blocked,
                memory_record_id: None,
                maintenance_status: Some(
                    BlueBrainMemoryMaintenanceStatus::NonCanonicalInternalOnlyPath,
                ),
                caveat_refresh_state: None,
                diagnostic_class:
                    BlueBrainMemoryDiagnosticClass::NonCanonicalInternalOnlyMemoryDiagnostic,
                diagnostic: "maintenance blocked for internal/non-canonical locator".to_string(),
                caveats: Vec::new(),
                automatic_compute_triggered: false,
                automatic_action_or_planning_triggered: false,
                automatic_memory_commit_triggered: false,
                feedback_backbind: maintenance_feedback(
                    BlueBrainMemoryMaintenanceResultState::Blocked,
                    false,
                    false,
                ),
            };
        }

        let record_id = match request.locator {
            BlueBrainMemoryMaintenanceLocator::MemoryRecordId(memory_record_id) => {
                memory_record_id.to_string()
            }
            BlueBrainMemoryMaintenanceLocator::SourceCandidateId(candidate_id) => {
                match self.candidate_index.get(candidate_id) {
                    Some(memory_record_id) => memory_record_id.clone(),
                    None => {
                        return BlueBrainMemoryMaintenanceReport {
                            result_state: BlueBrainMemoryMaintenanceResultState::NoOp,
                            memory_record_id: None,
                            maintenance_status: None,
                            caveat_refresh_state: None,
                            diagnostic_class: BlueBrainMemoryDiagnosticClass::MissingMemoryDiagnostic,
                            diagnostic:
                                "maintenance no-op: no persisted memory record for requested locator"
                                    .to_string(),
                            caveats: Vec::new(),
                            automatic_compute_triggered: false,
                            automatic_action_or_planning_triggered: false,
                            automatic_memory_commit_triggered: false,
                            feedback_backbind: maintenance_feedback(
                                BlueBrainMemoryMaintenanceResultState::NoOp,
                                false,
                                false,
                            ),
                        };
                    }
                }
            }
        };

        let Some(existing) = self.records.get(&record_id).cloned() else {
            return BlueBrainMemoryMaintenanceReport {
                result_state: BlueBrainMemoryMaintenanceResultState::NoOp,
                memory_record_id: None,
                maintenance_status: None,
                caveat_refresh_state: None,
                diagnostic_class: BlueBrainMemoryDiagnosticClass::MissingMemoryDiagnostic,
                diagnostic: "maintenance no-op: no persisted memory record for requested locator"
                    .to_string(),
                caveats: Vec::new(),
                automatic_compute_triggered: false,
                automatic_action_or_planning_triggered: false,
                automatic_memory_commit_triggered: false,
                feedback_backbind: maintenance_feedback(
                    BlueBrainMemoryMaintenanceResultState::NoOp,
                    false,
                    false,
                ),
            };
        };

        let mut next = existing.clone();
        next.maintenance_updated_at_unix_ms = Some(maintenance_updated_at_unix_ms);
        let (result_state, diagnostic_class, diagnostic, caveated_result) = match request.action {
            BlueBrainMemoryMaintenanceAction::MarkCurrent => {
                if next.maintenance_status == BlueBrainMemoryMaintenanceStatus::Current {
                    (
                        BlueBrainMemoryMaintenanceResultState::NoOp,
                        BlueBrainMemoryDiagnosticClass::MaintenanceDiagnostic,
                        "maintenance no-op: record already current".to_string(),
                        false,
                    )
                } else {
                    next.maintenance_status = BlueBrainMemoryMaintenanceStatus::Current;
                    (
                        BlueBrainMemoryMaintenanceResultState::Applied,
                        BlueBrainMemoryDiagnosticClass::MaintenanceDiagnostic,
                        "maintenance applied: record marked current".to_string(),
                        false,
                    )
                }
            }
            BlueBrainMemoryMaintenanceAction::MarkStale => {
                next.maintenance_status = BlueBrainMemoryMaintenanceStatus::Stale;
                (
                    BlueBrainMemoryMaintenanceResultState::Applied,
                    BlueBrainMemoryDiagnosticClass::StaleMemoryDiagnostic,
                    "maintenance applied: record marked stale".to_string(),
                    true,
                )
            }
            BlueBrainMemoryMaintenanceAction::Invalidate { reason } => {
                next.maintenance_status = BlueBrainMemoryMaintenanceStatus::Invalidated;
                next.maintenance_note = Some(reason);
                (
                    BlueBrainMemoryMaintenanceResultState::Applied,
                    BlueBrainMemoryDiagnosticClass::InvalidatedMemoryDiagnostic,
                    "maintenance applied: record invalidated".to_string(),
                    true,
                )
            }
            BlueBrainMemoryMaintenanceAction::MarkMaintenanceBlocked { reason } => {
                next.maintenance_status = BlueBrainMemoryMaintenanceStatus::MaintenanceBlocked;
                next.maintenance_note = Some(reason);
                (
                    BlueBrainMemoryMaintenanceResultState::Blocked,
                    BlueBrainMemoryDiagnosticClass::MaintenanceBlockedMemoryDiagnostic,
                    "maintenance blocked: maintenance path blocked for record".to_string(),
                    true,
                )
            }
            BlueBrainMemoryMaintenanceAction::MarkRefreshUnavailable { reason } => {
                next.maintenance_status = BlueBrainMemoryMaintenanceStatus::RefreshUnavailable;
                next.caveat_refresh_state = BlueBrainMemoryCaveatRefreshState::RefreshUnavailable;
                next.maintenance_note = Some(reason);
                (
                    BlueBrainMemoryMaintenanceResultState::Unavailable,
                    BlueBrainMemoryDiagnosticClass::RefreshUnavailableMemoryDiagnostic,
                    "maintenance unavailable: caveat refresh unavailable".to_string(),
                    true,
                )
            }
            BlueBrainMemoryMaintenanceAction::RefreshCaveats {
                mut caveats,
                refresh_state,
            } => {
                caveats.sort_unstable();
                caveats.dedup();
                if matches!(
                    refresh_state,
                    BlueBrainMemoryCaveatRefreshState::RefreshBlocked
                        | BlueBrainMemoryCaveatRefreshState::RefreshUnavailable
                ) {
                    next.maintenance_status = BlueBrainMemoryMaintenanceStatus::Caveated;
                } else {
                    next.maintenance_status = BlueBrainMemoryMaintenanceStatus::CaveatRefreshed;
                }
                next.caveat_refresh_state = refresh_state;
                next.caveats = caveats;
                (
                    BlueBrainMemoryMaintenanceResultState::Caveated,
                    BlueBrainMemoryDiagnosticClass::CaveatRefreshedMemoryDiagnostic,
                    "maintenance caveated: caveats refreshed/preserved from reference/evidence posture"
                        .to_string(),
                    true,
                )
            }
        };

        let caveats = next.caveats.clone();
        match self.upsert(next.clone()) {
            Ok(()) => BlueBrainMemoryMaintenanceReport {
                result_state,
                memory_record_id: Some(record_id),
                maintenance_status: Some(next.maintenance_status),
                caveat_refresh_state: Some(next.caveat_refresh_state),
                diagnostic_class,
                diagnostic,
                caveats,
                automatic_compute_triggered: false,
                automatic_action_or_planning_triggered: false,
                automatic_memory_commit_triggered: false,
                feedback_backbind: maintenance_feedback(result_state, true, caveated_result),
            },
            Err(err) => BlueBrainMemoryMaintenanceReport {
                result_state: BlueBrainMemoryMaintenanceResultState::Failed,
                memory_record_id: None,
                maintenance_status: Some(existing.maintenance_status),
                caveat_refresh_state: Some(existing.caveat_refresh_state),
                diagnostic_class: BlueBrainMemoryDiagnosticClass::FailedCommitDiagnostic,
                diagnostic: format!("memory maintenance write failed: {err}"),
                caveats: existing.caveats,
                automatic_compute_triggered: false,
                automatic_action_or_planning_triggered: false,
                automatic_memory_commit_triggered: false,
                feedback_backbind: maintenance_feedback(
                    BlueBrainMemoryMaintenanceResultState::Failed,
                    false,
                    caveated_result,
                ),
            },
        }
    }

    fn upsert(
        &mut self,
        persisted: PersistedBlueBrainMemoryRecord,
    ) -> Result<(), BlueBrainMemoryStoreError> {
        let encoded =
            serde_json::to_string(&persisted).map_err(|err| BlueBrainMemoryStoreError::Encode {
                reason: err.to_string(),
            })?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|err| BlueBrainMemoryStoreError::Io {
                operation: "mkdir",
                path: parent.display().to_string(),
                reason: err.to_string(),
            })?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|err| BlueBrainMemoryStoreError::Io {
                operation: "append-open",
                path: self.path.display().to_string(),
                reason: err.to_string(),
            })?;

        file.write_all(encoded.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.flush())
            .map_err(|err| BlueBrainMemoryStoreError::Io {
                operation: "append-write",
                path: self.path.display().to_string(),
                reason: err.to_string(),
            })?;

        self.candidate_index.insert(
            persisted.source_candidate_id.clone(),
            persisted.memory_record_id.clone(),
        );
        self.records
            .insert(persisted.memory_record_id.clone(), persisted);
        Ok(())
    }
}

fn build_memory_record_id(
    candidate: &BlueBrainMemoryCandidate,
    committed_at_unix_ms: u64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(candidate.candidate_id.as_bytes());
    hasher.update(b"|");
    hasher.update(committed_at_unix_ms.to_le_bytes());
    hasher.update(b"|");
    hasher.update(format!("{:?}", candidate.freshness).as_bytes());
    hasher.update(b"|");
    for item in &candidate.evidence_refs {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    for item in &candidate.reference_refs {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    for item in &candidate.context_basis_refs {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    for item in &candidate.selection_basis_refs {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    for item in &candidate.caveats {
        hasher.update(item.as_bytes());
        hasher.update(b";");
    }
    format!("bb_mem_{:x}", hasher.finalize())
}

fn map_commit_diagnostic_class(
    state: BlueBrainMemoryCommitResultState,
    created_record: bool,
) -> BlueBrainMemoryDiagnosticClass {
    match state {
        BlueBrainMemoryCommitResultState::Committed if created_record => {
            BlueBrainMemoryDiagnosticClass::CommittedDiagnostic
        }
        BlueBrainMemoryCommitResultState::CommittedWithCaveat if created_record => {
            BlueBrainMemoryDiagnosticClass::CommittedWithCaveatDiagnostic
        }
        BlueBrainMemoryCommitResultState::Committed => {
            BlueBrainMemoryDiagnosticClass::CommitDiagnostic
        }
        BlueBrainMemoryCommitResultState::CommittedWithCaveat => {
            BlueBrainMemoryDiagnosticClass::CaveatedMemoryDiagnostic
        }
        BlueBrainMemoryCommitResultState::Rejected => {
            BlueBrainMemoryDiagnosticClass::RejectedCommitDiagnostic
        }
        BlueBrainMemoryCommitResultState::Blocked => {
            BlueBrainMemoryDiagnosticClass::BlockedCommitDiagnostic
        }
        BlueBrainMemoryCommitResultState::Failed => {
            BlueBrainMemoryDiagnosticClass::FailedCommitDiagnostic
        }
        BlueBrainMemoryCommitResultState::NoOp => {
            BlueBrainMemoryDiagnosticClass::NoOpCommitDiagnostic
        }
        BlueBrainMemoryCommitResultState::Unavailable => {
            BlueBrainMemoryDiagnosticClass::UnavailableMemoryDiagnostic
        }
    }
}

fn commit_feedback(
    state: BlueBrainMemoryCommitResultState,
    created_record: bool,
    caveated: bool,
) -> BlueBrainMemoryFeedbackBackbind {
    let mut runtime_feedback =
        vec![BlueBrainMemoryRuntimeFeedbackClass::FeedbackObservedNoAutoComputeOrAction];
    let mut context_feedback =
        vec![BlueBrainMemoryContextFeedbackClass::NoAutomaticMemoryCandidateCreation];
    let mut selection_candidate_proposal_feedback = vec![
        BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievalDoesNotAutomaticallySelectProposeOrExecute,
    ];

    match state {
        BlueBrainMemoryCommitResultState::Committed
        | BlueBrainMemoryCommitResultState::CommittedWithCaveat
            if created_record =>
        {
            runtime_feedback.push(BlueBrainMemoryRuntimeFeedbackClass::MemoryCommitted);
            context_feedback
                .push(BlueBrainMemoryContextFeedbackClass::CommittedMemoryAttachedToCurrentContext);
            selection_candidate_proposal_feedback.push(
                BlueBrainMemorySelectionCandidateProposalFeedbackClass::CommittedMemoryMaySupportFutureProposalBasis,
            );
        }
        BlueBrainMemoryCommitResultState::Blocked => {
            runtime_feedback.push(BlueBrainMemoryRuntimeFeedbackClass::CommitOrRetrievalBlocked);
            context_feedback
                .push(BlueBrainMemoryContextFeedbackClass::StaleOrMissingMemoryLimitsContextUpdate);
            selection_candidate_proposal_feedback.push(
                BlueBrainMemorySelectionCandidateProposalFeedbackClass::StaleOrMissingMemoryWeakensCandidateBasis,
            );
        }
        BlueBrainMemoryCommitResultState::Rejected => {
            context_feedback
                .push(BlueBrainMemoryContextFeedbackClass::StaleOrMissingMemoryLimitsContextUpdate);
            selection_candidate_proposal_feedback.push(
                BlueBrainMemorySelectionCandidateProposalFeedbackClass::StaleOrMissingMemoryWeakensCandidateBasis,
            );
        }
        BlueBrainMemoryCommitResultState::Failed
        | BlueBrainMemoryCommitResultState::Unavailable => {
            runtime_feedback
                .push(BlueBrainMemoryRuntimeFeedbackClass::CommitOrRetrievalFailedOrUnavailable);
        }
        BlueBrainMemoryCommitResultState::NoOp => {}
        _ => {}
    }

    if caveated || state == BlueBrainMemoryCommitResultState::CommittedWithCaveat {
        runtime_feedback
            .push(BlueBrainMemoryRuntimeFeedbackClass::MemoryRetrievalMissingOrStaleOrCaveated);
        context_feedback.push(BlueBrainMemoryContextFeedbackClass::MemoryCaveatCarriedIntoContext);
        selection_candidate_proposal_feedback.push(
            BlueBrainMemorySelectionCandidateProposalFeedbackClass::CaveatedMemoryYieldsCaveatedSelectionOrProposal,
        );
    }

    BlueBrainMemoryFeedbackBackbind {
        runtime_feedback,
        context_feedback,
        selection_candidate_proposal_feedback,
    }
}

fn maintenance_feedback(
    state: BlueBrainMemoryMaintenanceResultState,
    maintenance_applied: bool,
    caveated: bool,
) -> BlueBrainMemoryFeedbackBackbind {
    let mut runtime_feedback =
        vec![BlueBrainMemoryRuntimeFeedbackClass::FeedbackObservedNoAutoComputeOrAction];
    let mut context_feedback =
        vec![BlueBrainMemoryContextFeedbackClass::NoAutomaticMemoryCandidateCreation];
    let mut selection_candidate_proposal_feedback = vec![
        BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievalDoesNotAutomaticallySelectProposeOrExecute,
    ];

    match state {
        BlueBrainMemoryMaintenanceResultState::Applied if maintenance_applied => {
            runtime_feedback.push(BlueBrainMemoryRuntimeFeedbackClass::MemoryRetrieved);
            context_feedback
                .push(BlueBrainMemoryContextFeedbackClass::RetrievedMemoryAttachedToCurrentContext);
        }
        BlueBrainMemoryMaintenanceResultState::Blocked => {
            runtime_feedback.push(BlueBrainMemoryRuntimeFeedbackClass::CommitOrRetrievalBlocked);
            context_feedback
                .push(BlueBrainMemoryContextFeedbackClass::StaleOrMissingMemoryLimitsContextUpdate);
            selection_candidate_proposal_feedback.push(
                BlueBrainMemorySelectionCandidateProposalFeedbackClass::StaleOrMissingMemoryWeakensCandidateBasis,
            );
        }
        BlueBrainMemoryMaintenanceResultState::Failed
        | BlueBrainMemoryMaintenanceResultState::Unavailable => {
            runtime_feedback
                .push(BlueBrainMemoryRuntimeFeedbackClass::CommitOrRetrievalFailedOrUnavailable);
        }
        _ => {}
    }

    if caveated || state == BlueBrainMemoryMaintenanceResultState::Caveated {
        runtime_feedback
            .push(BlueBrainMemoryRuntimeFeedbackClass::MemoryRetrievalMissingOrStaleOrCaveated);
        context_feedback.push(BlueBrainMemoryContextFeedbackClass::MemoryCaveatCarriedIntoContext);
        selection_candidate_proposal_feedback.push(
            BlueBrainMemorySelectionCandidateProposalFeedbackClass::CaveatedMemoryYieldsCaveatedSelectionOrProposal,
        );
    }

    BlueBrainMemoryFeedbackBackbind {
        runtime_feedback,
        context_feedback,
        selection_candidate_proposal_feedback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_PATH_SEQ: AtomicU64 = AtomicU64::new(0);

    fn temp_store_path(name: &str) -> PathBuf {
        let sequence = TEST_PATH_SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "ucf_blue_brain_memory_store_{}_{}_{}_{}.jsonl",
            name,
            std::process::id(),
            nanos,
            sequence
        ))
    }

    fn base_candidate(id: &str) -> BlueBrainMemoryCandidate {
        BlueBrainMemoryCandidate {
            candidate_id: id.to_string(),
            class: BlueBrainMemoryCandidateClass::CommitEligible,
            origins: vec![
                BlueBrainMemoryOrigin::Context,
                BlueBrainMemoryOrigin::Evidence,
            ],
            evidence_refs: vec!["evidence:bundle:42".to_string()],
            reference_refs: vec!["ref:status:7".to_string()],
            context_basis_refs: vec!["context:digest:abc".to_string()],
            selection_basis_refs: vec!["selection:attention:primary".to_string()],
            freshness: BlueBrainMemoryFreshness::Current,
            caveats: Vec::new(),
            allow_caveated_commit: false,
            allow_stale_context_commit: false,
            has_internal_only_dependency: false,
            commit_path_available: true,
        }
    }

    #[test]
    fn commit_eligible_candidate_is_persisted_and_readable() {
        let path = temp_store_path("commit_ok");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let report = store.commit_candidate(base_candidate("cand-1"), 100);
        assert_eq!(
            report.result_state,
            BlueBrainMemoryCommitResultState::Committed
        );
        let record_id = report.memory_record_id.expect("record id");
        assert!(report.created_record);

        let record = store.get(&record_id).expect("persisted record");
        assert_eq!(record.source_candidate_id, "cand-1");
        assert_eq!(
            record.commit_result_state,
            BlueBrainMemoryCommitResultState::Committed
        );
        assert_eq!(record.evidence_refs, vec!["evidence:bundle:42"]);
    }

    #[test]
    fn commit_with_caveat_is_persisted_with_caveats() {
        let path = temp_store_path("commit_caveat");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut candidate = base_candidate("cand-2");
        candidate.allow_caveated_commit = true;
        candidate.caveats = vec!["partial evidence basis".to_string()];

        let report = store.commit_candidate(candidate, 101);
        assert_eq!(
            report.result_state,
            BlueBrainMemoryCommitResultState::CommittedWithCaveat
        );
        let record = store
            .get(report.memory_record_id.as_deref().expect("record id"))
            .expect("record");
        assert_eq!(record.caveats, vec!["partial evidence basis"]);
    }

    #[test]
    fn deferred_rejected_insufficient_and_reference_only_are_not_committed() {
        let path = temp_store_path("guard_classes");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut deferred = base_candidate("cand-3");
        deferred.class = BlueBrainMemoryCandidateClass::Deferred;
        assert_eq!(
            store.commit_candidate(deferred, 102).result_state,
            BlueBrainMemoryCommitResultState::Blocked
        );

        let mut rejected = base_candidate("cand-4");
        rejected.class = BlueBrainMemoryCandidateClass::Rejected;
        assert_eq!(
            store.commit_candidate(rejected, 103).result_state,
            BlueBrainMemoryCommitResultState::Rejected
        );

        let mut insufficient = base_candidate("cand-5");
        insufficient.class = BlueBrainMemoryCandidateClass::Insufficient;
        assert_eq!(
            store.commit_candidate(insufficient, 104).result_state,
            BlueBrainMemoryCommitResultState::Blocked
        );

        let mut reference_only = base_candidate("cand-6");
        reference_only.class = BlueBrainMemoryCandidateClass::ReferenceOnly;
        assert_eq!(
            store.commit_candidate(reference_only, 105).result_state,
            BlueBrainMemoryCommitResultState::Rejected
        );

        assert_eq!(store.len(), 0);
    }

    #[test]
    fn stale_context_and_internal_only_dependency_are_blocked() {
        let path = temp_store_path("stale_internal_blocked");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut stale = base_candidate("cand-7");
        stale.freshness = BlueBrainMemoryFreshness::Stale;
        let stale_report = store.commit_candidate(stale, 106);
        assert_eq!(
            stale_report.result_state,
            BlueBrainMemoryCommitResultState::Blocked
        );

        let mut internal = base_candidate("cand-8");
        internal.has_internal_only_dependency = true;
        let internal_report = store.commit_candidate(internal, 107);
        assert_eq!(
            internal_report.result_state,
            BlueBrainMemoryCommitResultState::Blocked
        );
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn no_op_and_unavailable_states_are_reported_without_creating_records() {
        let path = temp_store_path("noop_unavailable");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let first = store.commit_candidate(base_candidate("cand-9"), 108);
        assert_eq!(
            first.result_state,
            BlueBrainMemoryCommitResultState::Committed
        );

        let second = store.commit_candidate(base_candidate("cand-9"), 109);
        assert_eq!(second.result_state, BlueBrainMemoryCommitResultState::NoOp);
        assert!(!second.created_record);

        let mut unavailable = base_candidate("cand-10");
        unavailable.commit_path_available = false;
        let unavailable = store.commit_candidate(unavailable, 110);
        assert_eq!(
            unavailable.result_state,
            BlueBrainMemoryCommitResultState::Unavailable
        );
        assert!(unavailable.memory_record_id.is_none());
    }

    #[test]
    fn record_id_is_stable_and_reopen_supports_read_by_id_or_candidate() {
        let path = temp_store_path("reopen_read");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");
        let candidate = base_candidate("cand-11");

        let expected_id = super::build_memory_record_id(&candidate, 111);
        let report = store.commit_candidate(candidate.clone(), 111);
        assert_eq!(
            report.memory_record_id.as_deref(),
            Some(expected_id.as_str())
        );

        drop(store);
        let reopened = BlueBrainMemoryStore::open(&path).expect("reopen memory store");
        let by_id = reopened.get(&expected_id).expect("by id");
        let by_candidate = reopened.get_by_candidate("cand-11").expect("by candidate");
        assert_eq!(by_id.memory_record_id, by_candidate.memory_record_id);
    }

    #[test]
    fn failed_result_is_reported_when_store_path_is_not_writable() {
        let path = temp_store_path("commit_failed");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let dir_path = std::env::temp_dir().join("ucf_blue_brain_memory_store_fail_dir");
        let _ = std::fs::remove_file(&dir_path);
        std::fs::create_dir_all(&dir_path).expect("create dir");
        store.path = dir_path.clone();

        let report = store.commit_candidate(base_candidate("cand-12"), 112);
        assert_eq!(
            report.result_state,
            BlueBrainMemoryCommitResultState::Failed
        );
        assert!(report.memory_record_id.is_none());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir_path);
    }

    #[test]
    fn read_reference_distinguishes_found_missing_blocked_and_unavailable() {
        let path = temp_store_path("read_states");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");
        let report = store.commit_candidate(base_candidate("cand-13"), 113);
        let record_id = report.memory_record_id.expect("record id");

        let found = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&record_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            found.retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedReferenceOnly
        );
        assert!(found.context_attached);
        assert!(!found.automatic_compute_triggered);
        assert!(!found.automatic_action_or_planning_triggered);
        assert!(!found.automatic_memory_commit_triggered);
        assert_eq!(
            found.selection_disposition,
            BlueBrainMemorySelectionDisposition::Supporting
        );

        let missing = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId("bb_mem_missing"),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            missing.retrieval_state,
            BlueBrainMemoryRetrievalState::Missing
        );

        let blocked = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::SourceCandidateId("internal:dev-hook"),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            blocked.retrieval_state,
            BlueBrainMemoryRetrievalState::Blocked
        );

        let unavailable = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&record_id),
            canonical_retrieval_path_available: false,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            unavailable.retrieval_state,
            BlueBrainMemoryRetrievalState::Unavailable
        );
    }

    #[test]
    fn read_reference_distinguishes_caveated_and_stale_states() {
        let path = temp_store_path("read_caveated_stale");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut caveated = base_candidate("cand-14");
        caveated.allow_caveated_commit = true;
        caveated.caveats = vec!["partial reference basis".to_string()];
        let caveated_id = store
            .commit_candidate(caveated, 114)
            .memory_record_id
            .expect("caveated id");

        let caveated_read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&caveated_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            caveated_read.retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedWithCaveat
        );
        assert!(caveated_read.context_caveated);
        assert_eq!(
            caveated_read.selection_disposition,
            BlueBrainMemorySelectionDisposition::Caveated
        );

        let mut stale = base_candidate("cand-15");
        stale.freshness = BlueBrainMemoryFreshness::Stale;
        stale.allow_stale_context_commit = true;
        let stale_id = store
            .commit_candidate(stale, 115)
            .memory_record_id
            .expect("stale id");

        let stale_read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&stale_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            stale_read.retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedStale
        );
        assert!(stale_read.context_stale);
        assert_eq!(
            stale_read.selection_disposition,
            BlueBrainMemorySelectionDisposition::Deferred
        );
    }

    #[test]
    fn read_reference_is_memory_only_and_not_history_snapshot_or_evidence_retrieval() {
        let path = temp_store_path("read_memory_only");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");
        let report = store.commit_candidate(base_candidate("cand-16"), 116);
        let record_id = report.memory_record_id.expect("record id");

        let read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&record_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        let reference = read.reference.expect("reference");
        assert_eq!(reference.memory_record_id, record_id);
        assert_eq!(reference.source_candidate_id, "cand-16");
        assert!(reference
            .evidence_refs
            .iter()
            .all(|item| !item.contains("history")
                && !item.contains("snapshot")
                && !item.contains("replay")));
    }

    #[test]
    fn diagnostics_map_covers_required_classes() {
        assert_eq!(CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP.len(), 19);
        assert!(CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP
            .iter()
            .any(|lane| {
                lane.diagnostic_class
                    == BlueBrainMemoryDiagnosticClass::CommittedWithCaveatDiagnostic
            }));
        assert!(CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP
            .iter()
            .any(|lane| {
                lane.diagnostic_class == BlueBrainMemoryDiagnosticClass::InvalidatedMemoryDiagnostic
            }));
        assert!(CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP
            .iter()
            .any(|lane| {
                lane.diagnostic_class
                    == BlueBrainMemoryDiagnosticClass::NonCanonicalInternalOnlyMemoryDiagnostic
            }));
    }

    #[test]
    fn commit_and_retrieval_feedback_has_no_auto_compute_action_or_commit_triggers() {
        let path = temp_store_path("feedback_no_auto_trigger");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut caveated = base_candidate("cand-17");
        caveated.allow_caveated_commit = true;
        caveated.caveats = vec!["partial basis".to_string()];
        let commit_report = store.commit_candidate(caveated, 117);
        assert_eq!(
            commit_report.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::CommittedWithCaveatDiagnostic
        );
        assert!(commit_report
            .feedback_backbind
            .selection_candidate_proposal_feedback
            .contains(
                &BlueBrainMemorySelectionCandidateProposalFeedbackClass::CommittedMemoryMaySupportFutureProposalBasis
            ));

        let read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::SourceCandidateId("cand-17"),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert!(read.context_attached);
        assert!(!read.automatic_compute_triggered);
        assert!(!read.automatic_action_or_planning_triggered);
        assert!(!read.automatic_memory_commit_triggered);
        assert!(read
            .feedback_backbind
            .selection_candidate_proposal_feedback
            .contains(
                &BlueBrainMemorySelectionCandidateProposalFeedbackClass::RetrievalDoesNotAutomaticallySelectProposeOrExecute
            ));
    }

    #[test]
    fn retrieval_diagnostics_distinguish_missing_stale_caveated_blocked_and_unavailable() {
        let path = temp_store_path("retrieval_diag_classes");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");

        let mut stale = base_candidate("cand-18");
        stale.freshness = BlueBrainMemoryFreshness::Stale;
        stale.allow_stale_context_commit = true;
        let stale_id = store
            .commit_candidate(stale, 118)
            .memory_record_id
            .expect("stale id");

        let stale_read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&stale_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            stale_read.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::StaleMemoryDiagnostic
        );

        let missing = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId("bb_mem_missing"),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            missing.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::MissingMemoryDiagnostic
        );

        let blocked = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::SourceCandidateId("internal:dev"),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            blocked.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::NonCanonicalInternalOnlyMemoryDiagnostic
        );

        let unavailable = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&stale_id),
            canonical_retrieval_path_available: false,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            unavailable.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::UnavailableMemoryDiagnostic
        );
    }

    #[test]
    fn maintenance_states_are_distinguishable_and_feed_retrieval_context_selection() {
        let path = temp_store_path("maintenance_states");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");
        let report = store.commit_candidate(base_candidate("cand-19"), 119);
        let record_id = report.memory_record_id.expect("record id");

        let stale = store.apply_maintenance(
            BlueBrainMemoryMaintenanceRequest {
                locator: BlueBrainMemoryMaintenanceLocator::MemoryRecordId(&record_id),
                action: BlueBrainMemoryMaintenanceAction::MarkStale,
                canonical_maintenance_path_available: true,
                allow_internal_only_locator: false,
            },
            120,
        );
        assert_eq!(
            stale.result_state,
            BlueBrainMemoryMaintenanceResultState::Applied
        );
        assert_eq!(
            stale.maintenance_status,
            Some(BlueBrainMemoryMaintenanceStatus::Stale)
        );

        let stale_read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&record_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            stale_read.retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedStale
        );
        assert_eq!(
            stale_read.selection_disposition,
            BlueBrainMemorySelectionDisposition::Deferred
        );

        let invalidated = store.apply_maintenance(
            BlueBrainMemoryMaintenanceRequest {
                locator: BlueBrainMemoryMaintenanceLocator::MemoryRecordId(&record_id),
                action: BlueBrainMemoryMaintenanceAction::Invalidate {
                    reason: "reference contradicted by fresh evidence".to_string(),
                },
                canonical_maintenance_path_available: true,
                allow_internal_only_locator: false,
            },
            121,
        );
        assert_eq!(
            invalidated.maintenance_status,
            Some(BlueBrainMemoryMaintenanceStatus::Invalidated)
        );

        let invalidated_read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&record_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            invalidated_read.retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedInvalidated
        );
        assert_eq!(
            invalidated_read.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::InvalidatedMemoryDiagnostic
        );
        assert_eq!(
            invalidated_read.selection_disposition,
            BlueBrainMemorySelectionDisposition::Insufficient
        );
    }

    #[test]
    fn caveat_refresh_preserves_or_updates_caveats_without_auto_triggers() {
        let path = temp_store_path("maintenance_caveat_refresh");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");
        let report = store.commit_candidate(base_candidate("cand-20"), 122);
        let record_id = report.memory_record_id.expect("record id");

        let refresh = store.apply_maintenance(
            BlueBrainMemoryMaintenanceRequest {
                locator: BlueBrainMemoryMaintenanceLocator::MemoryRecordId(&record_id),
                action: BlueBrainMemoryMaintenanceAction::RefreshCaveats {
                    caveats: vec![
                        "evidence freshness uncertain".to_string(),
                        "reference partially superseded".to_string(),
                    ],
                    refresh_state: BlueBrainMemoryCaveatRefreshState::Strengthened,
                },
                canonical_maintenance_path_available: true,
                allow_internal_only_locator: false,
            },
            123,
        );
        assert_eq!(
            refresh.result_state,
            BlueBrainMemoryMaintenanceResultState::Caveated
        );
        assert_eq!(
            refresh.caveat_refresh_state,
            Some(BlueBrainMemoryCaveatRefreshState::Strengthened)
        );
        assert!(!refresh.automatic_compute_triggered);
        assert!(!refresh.automatic_action_or_planning_triggered);
        assert!(!refresh.automatic_memory_commit_triggered);

        let read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&record_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            read.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::CaveatRefreshedMemoryDiagnostic
        );
        let reference = read.reference.expect("reference");
        assert_eq!(
            reference.maintenance_status,
            BlueBrainMemoryMaintenanceStatus::CaveatRefreshed
        );
        assert_eq!(
            reference.caveat_refresh_state,
            BlueBrainMemoryCaveatRefreshState::Strengthened
        );
    }

    #[test]
    fn maintenance_blocked_and_refresh_unavailable_states_are_explicit() {
        let path = temp_store_path("maintenance_blocked_unavailable");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");
        let report = store.commit_candidate(base_candidate("cand-21"), 124);
        let record_id = report.memory_record_id.expect("record id");

        let blocked = store.apply_maintenance(
            BlueBrainMemoryMaintenanceRequest {
                locator: BlueBrainMemoryMaintenanceLocator::MemoryRecordId(&record_id),
                action: BlueBrainMemoryMaintenanceAction::MarkMaintenanceBlocked {
                    reason: "maintenance guard triggered".to_string(),
                },
                canonical_maintenance_path_available: true,
                allow_internal_only_locator: false,
            },
            125,
        );
        assert_eq!(
            blocked.result_state,
            BlueBrainMemoryMaintenanceResultState::Blocked
        );

        let blocked_read = store.read_reference(BlueBrainMemoryReadRequest {
            locator: BlueBrainMemoryReferenceLocator::MemoryRecordId(&record_id),
            canonical_retrieval_path_available: true,
            allow_internal_only_locator: false,
        });
        assert_eq!(
            blocked_read.retrieval_state,
            BlueBrainMemoryRetrievalState::Blocked
        );
        assert_eq!(
            blocked_read.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::MaintenanceBlockedMemoryDiagnostic
        );

        let unavailable = store.apply_maintenance(
            BlueBrainMemoryMaintenanceRequest {
                locator: BlueBrainMemoryMaintenanceLocator::MemoryRecordId(&record_id),
                action: BlueBrainMemoryMaintenanceAction::MarkRefreshUnavailable {
                    reason: "reference refresh feed unavailable".to_string(),
                },
                canonical_maintenance_path_available: true,
                allow_internal_only_locator: false,
            },
            126,
        );
        assert_eq!(
            unavailable.result_state,
            BlueBrainMemoryMaintenanceResultState::Unavailable
        );
        assert_eq!(
            unavailable.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::RefreshUnavailableMemoryDiagnostic
        );
    }

    #[test]
    fn non_canonical_internal_only_maintenance_path_is_blocked_and_non_canonical() {
        let path = temp_store_path("maintenance_non_canonical");
        let _ = std::fs::remove_file(&path);
        let mut store = BlueBrainMemoryStore::open(&path).expect("open memory store");
        let _ = store.commit_candidate(base_candidate("cand-22"), 127);

        let blocked = store.apply_maintenance(
            BlueBrainMemoryMaintenanceRequest {
                locator: BlueBrainMemoryMaintenanceLocator::SourceCandidateId("internal:expert"),
                action: BlueBrainMemoryMaintenanceAction::MarkStale,
                canonical_maintenance_path_available: true,
                allow_internal_only_locator: false,
            },
            128,
        );
        assert_eq!(
            blocked.maintenance_status,
            Some(BlueBrainMemoryMaintenanceStatus::NonCanonicalInternalOnlyPath)
        );
        assert_eq!(
            blocked.diagnostic_class,
            BlueBrainMemoryDiagnosticClass::NonCanonicalInternalOnlyMemoryDiagnostic
        );
    }
}
