use crate::blue_brain_memory::{BlueBrainMemoryReadResult, BlueBrainMemoryRetrievalState};
use crate::blue_brain_minimal_execution::{
    BlueBrainExecutionOutcomeClass, BlueBrainMinimalExecutionReport, BlueBrainMinimalExecutionState,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainRetrievalConsolidationCandidateClass {
    MemoryRetrievalCandidate,
    ExecutionResultRetrievalCandidate,
    CombinedReferenceCandidate,
    RetrievalSupportingContextCandidate,
    ConsolidationCandidateOnly,
    InsufficientRetrievalBasis,
    NonCanonicalInternalOnlyRetrievalPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCombinedReferenceStatus {
    CombinedReferenceAvailable,
    CombinedReferenceCaveated,
    CombinedReferenceInsufficient,
    ConsolidationCandidateOnly,
    NoConsolidationPerformed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainCombinedRetrievalBasis {
    pub candidate_class: BlueBrainRetrievalConsolidationCandidateClass,
    pub combined_reference_status: BlueBrainCombinedReferenceStatus,
    pub memory_record_reference: Option<String>,
    pub execution_result_reference: Option<String>,
    pub candidate_reference: Option<String>,
    pub proposal_reference: Option<String>,
    pub context_reference: Option<String>,
    pub caveats: Vec<String>,
    pub freshness_or_staleness: Option<String>,
    pub maintenance_or_failure_state: Option<String>,
    pub reference_basis_observed: bool,
    pub reference_basis_attached_to_context: bool,
    pub reference_basis_supports_selection_or_proposal_only: bool,
    pub stale_invalidated_or_failed_references_weaken_basis: bool,
    pub automatic_compute_invoked: bool,
    pub automatic_action_executed: bool,
    pub automatic_memory_persisted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainCombinedRetrievalInput {
    pub memory_read: Option<BlueBrainMemoryReadResult>,
    pub execution_report: Option<BlueBrainMinimalExecutionReport>,
    pub candidate_reference: Option<String>,
    pub proposal_reference: Option<String>,
    pub context_reference: Option<String>,
}

pub fn blue_brain_build_combined_retrieval_basis(
    input: BlueBrainCombinedRetrievalInput,
) -> BlueBrainCombinedRetrievalBasis {
    let mut caveats = Vec::new();

    let memory_record_reference = input
        .memory_read
        .as_ref()
        .and_then(|read| read.reference.as_ref())
        .map(|reference| format!("bb8:memory_record:{}", reference.memory_record_id));
    let execution_result_reference = input.execution_report.as_ref().and_then(|report| {
        report
            .references
            .execution_result_reference
            .as_ref()
            .or(report.references.failure_result_reference.as_ref())
            .or(report.references.cancellation_result_reference.as_ref())
            .or(report.references.blocked_or_unavailable_reference.as_ref())
            .map(|reference| reference.path.clone())
    });

    let memory_non_canonical = input.memory_read.as_ref().is_some_and(|read| {
        matches!(
            read.retrieval_state,
            BlueBrainMemoryRetrievalState::Blocked | BlueBrainMemoryRetrievalState::Unavailable
        )
    });
    let execution_non_canonical = input.execution_report.as_ref().is_some_and(|report| {
        report.state == BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath
    });

    let has_memory_reference = memory_record_reference.is_some();
    let has_execution_reference = execution_result_reference.is_some();

    let stale_or_invalidated_memory = input.memory_read.as_ref().is_some_and(|read| {
        matches!(
            read.retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedStale
                | BlueBrainMemoryRetrievalState::RetrievedInvalidated
        )
    });
    let caveated_memory = input.memory_read.as_ref().is_some_and(|read| {
        matches!(
            read.retrieval_state,
            BlueBrainMemoryRetrievalState::RetrievedWithCaveat
        )
    });
    let failed_or_blocked_execution = input.execution_report.as_ref().is_some_and(|report| {
        matches!(
            report.outcome_class,
            BlueBrainExecutionOutcomeClass::ExecutionFailed
                | BlueBrainExecutionOutcomeClass::ExecutionBlocked
                | BlueBrainExecutionOutcomeClass::ExecutionUnavailable
                | BlueBrainExecutionOutcomeClass::ExecutionCancelled
        )
    });

    if stale_or_invalidated_memory {
        caveats.push("memory basis is stale or invalidated".to_string());
    }
    if caveated_memory {
        caveats.push("memory basis is caveated".to_string());
    }
    if failed_or_blocked_execution {
        caveats.push("execution reference is non-successful terminal state".to_string());
    }

    let stale_invalidated_or_failed_references_weaken_basis =
        stale_or_invalidated_memory || failed_or_blocked_execution;

    let candidate_class = if memory_non_canonical || execution_non_canonical {
        BlueBrainRetrievalConsolidationCandidateClass::NonCanonicalInternalOnlyRetrievalPath
    } else if has_memory_reference && has_execution_reference {
        BlueBrainRetrievalConsolidationCandidateClass::CombinedReferenceCandidate
    } else if has_memory_reference {
        BlueBrainRetrievalConsolidationCandidateClass::MemoryRetrievalCandidate
    } else if has_execution_reference {
        BlueBrainRetrievalConsolidationCandidateClass::ExecutionResultRetrievalCandidate
    } else if input.context_reference.is_some()
        || input.candidate_reference.is_some()
        || input.proposal_reference.is_some()
    {
        BlueBrainRetrievalConsolidationCandidateClass::RetrievalSupportingContextCandidate
    } else {
        BlueBrainRetrievalConsolidationCandidateClass::InsufficientRetrievalBasis
    };

    let combined_reference_status = match candidate_class {
        BlueBrainRetrievalConsolidationCandidateClass::CombinedReferenceCandidate => {
            if stale_invalidated_or_failed_references_weaken_basis || !caveats.is_empty() {
                BlueBrainCombinedReferenceStatus::CombinedReferenceCaveated
            } else {
                BlueBrainCombinedReferenceStatus::CombinedReferenceAvailable
            }
        }
        BlueBrainRetrievalConsolidationCandidateClass::MemoryRetrievalCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::ExecutionResultRetrievalCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::RetrievalSupportingContextCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::InsufficientRetrievalBasis
        | BlueBrainRetrievalConsolidationCandidateClass::NonCanonicalInternalOnlyRetrievalPath => {
            BlueBrainCombinedReferenceStatus::CombinedReferenceInsufficient
        }
        BlueBrainRetrievalConsolidationCandidateClass::ConsolidationCandidateOnly => {
            BlueBrainCombinedReferenceStatus::ConsolidationCandidateOnly
        }
    };

    let freshness_or_staleness =
        input
            .memory_read
            .as_ref()
            .map(|read| match read.retrieval_state {
                BlueBrainMemoryRetrievalState::RetrievedReferenceOnly => "current".to_string(),
                BlueBrainMemoryRetrievalState::RetrievedWithCaveat => "caveated".to_string(),
                BlueBrainMemoryRetrievalState::RetrievedStale => "stale".to_string(),
                BlueBrainMemoryRetrievalState::RetrievedInvalidated => "invalidated".to_string(),
                BlueBrainMemoryRetrievalState::Missing => "missing".to_string(),
                BlueBrainMemoryRetrievalState::Blocked => "blocked".to_string(),
                BlueBrainMemoryRetrievalState::Unavailable => "unavailable".to_string(),
            });

    let maintenance_or_failure_state = input
        .execution_report
        .as_ref()
        .map(|report| format!("{:?}:{:?}", report.state, report.outcome_class));

    let reference_basis_observed = has_memory_reference || has_execution_reference;
    let reference_basis_attached_to_context = input
        .memory_read
        .as_ref()
        .is_some_and(|read| read.context_attached)
        || input.context_reference.is_some();

    BlueBrainCombinedRetrievalBasis {
        candidate_class,
        combined_reference_status,
        memory_record_reference,
        execution_result_reference,
        candidate_reference: input.candidate_reference,
        proposal_reference: input.proposal_reference,
        context_reference: input.context_reference,
        caveats,
        freshness_or_staleness,
        maintenance_or_failure_state,
        reference_basis_observed,
        reference_basis_attached_to_context,
        reference_basis_supports_selection_or_proposal_only: true,
        stale_invalidated_or_failed_references_weaken_basis,
        automatic_compute_invoked: false,
        automatic_action_executed: false,
        automatic_memory_persisted: false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        blue_brain_build_combined_retrieval_basis, BlueBrainCombinedReferenceStatus,
        BlueBrainCombinedRetrievalInput, BlueBrainRetrievalConsolidationCandidateClass,
    };
    use crate::blue_brain_memory::{
        BlueBrainMemoryDiagnosticClass, BlueBrainMemoryReadResult,
        BlueBrainMemoryReferenceMetadata, BlueBrainMemoryReferenceRecord,
        BlueBrainMemoryRetrievalState, BlueBrainMemorySelectionDisposition,
    };
    use crate::blue_brain_minimal_execution::{
        execute_blue_brain_minimal_action, BlueBrainMinimalExecutionAction,
        BlueBrainMinimalExecutionRequest,
    };
    use crate::reference_map::{
        BlueBrainActionExecutionEligibilityClass, BlueBrainFutureActionHandoffClass,
        BlueBrainSafetyPrecheckClass,
    };

    fn sample_memory_read(
        retrieval_state: BlueBrainMemoryRetrievalState,
    ) -> BlueBrainMemoryReadResult {
        BlueBrainMemoryReadResult {
            retrieval_state,
            reference: Some(BlueBrainMemoryReferenceRecord {
                memory_record_id: "mem-1".to_string(),
                source_candidate_id: "cand-1".to_string(),
                commit_result_state:
                    crate::blue_brain_memory::BlueBrainMemoryCommitResultState::Committed,
                evidence_refs: vec!["ev-1".to_string()],
                reference_refs: vec!["ref-1".to_string()],
                context_basis_refs: vec!["ctx-1".to_string()],
                selection_basis_refs: vec!["sel-1".to_string()],
                freshness: crate::blue_brain_memory::BlueBrainMemoryFreshness::Current,
                caveats: Vec::new(),
                maintenance_status:
                    crate::blue_brain_memory::BlueBrainMemoryMaintenanceStatus::Current,
                caveat_refresh_state:
                    crate::blue_brain_memory::BlueBrainMemoryCaveatRefreshState::Preserved,
                maintenance_note: None,
                maintenance_updated_at_unix_ms: None,
                metadata: BlueBrainMemoryReferenceMetadata {
                    schema_version: 2,
                    committed_at_unix_ms: 42,
                },
            }),
            diagnostic_class: BlueBrainMemoryDiagnosticClass::RetrievedDiagnostic,
            diagnostic: "ok".to_string(),
            context_attached: true,
            context_caveated: false,
            context_stale: false,
            context_insufficient_for_candidate_or_proposal: false,
            automatic_compute_triggered: false,
            automatic_action_or_planning_triggered: false,
            automatic_memory_commit_triggered: false,
            selection_disposition: BlueBrainMemorySelectionDisposition::Supporting,
            feedback_backbind: crate::blue_brain_memory::BlueBrainMemoryFeedbackBackbind {
                runtime_feedback: Vec::new(),
                context_feedback: Vec::new(),
                selection_candidate_proposal_feedback: Vec::new(),
            },
        }
    }

    fn sample_execution_report(
    ) -> crate::blue_brain_minimal_execution::BlueBrainMinimalExecutionReport {
        execute_blue_brain_minimal_action(&BlueBrainMinimalExecutionRequest {
            handoff_id: "handoff-1".to_string(),
            handoff_class: BlueBrainFutureActionHandoffClass::FutureActionReady,
            eligibility_class: BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff,
            safety_precheck: BlueBrainSafetyPrecheckClass::Passed,
            action: BlueBrainMinimalExecutionAction::EmitCanonicalSignal,
            execution_requested: true,
            cancelled: false,
            internal_only_path: false,
            force_execution_failure: false,
            force_nonretryable_failure: false,
        })
    }

    #[test]
    fn combined_candidate_is_distinct_from_memory_or_execution_only() {
        let basis = blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
            memory_read: Some(sample_memory_read(
                BlueBrainMemoryRetrievalState::RetrievedReferenceOnly,
            )),
            execution_report: Some(sample_execution_report()),
            candidate_reference: Some("cand:1".to_string()),
            proposal_reference: Some("proposal:1".to_string()),
            context_reference: Some("ctx:1".to_string()),
        });
        assert_eq!(
            basis.candidate_class,
            BlueBrainRetrievalConsolidationCandidateClass::CombinedReferenceCandidate
        );
        assert_eq!(
            basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceAvailable
        );
        assert!(basis.reference_basis_supports_selection_or_proposal_only);
        assert!(!basis.automatic_compute_invoked);
        assert!(!basis.automatic_action_executed);
        assert!(!basis.automatic_memory_persisted);
    }

    #[test]
    fn stale_or_failed_basis_stays_caveated_and_weak() {
        let mut failed_report = sample_execution_report();
        failed_report.state =
            crate::blue_brain_minimal_execution::BlueBrainMinimalExecutionState::ExecutionFailed;
        failed_report.outcome_class =
            crate::blue_brain_minimal_execution::BlueBrainExecutionOutcomeClass::ExecutionFailed;

        let basis = blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
            memory_read: Some(sample_memory_read(
                BlueBrainMemoryRetrievalState::RetrievedStale,
            )),
            execution_report: Some(failed_report),
            candidate_reference: None,
            proposal_reference: None,
            context_reference: Some("ctx:2".to_string()),
        });

        assert_eq!(
            basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceCaveated
        );
        assert!(basis.stale_invalidated_or_failed_references_weaken_basis);
        assert!(!basis.caveats.is_empty());
    }

    #[test]
    fn context_only_input_is_not_promoted_to_combined_reference() {
        let basis = blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
            memory_read: None,
            execution_report: None,
            candidate_reference: None,
            proposal_reference: None,
            context_reference: Some("ctx-only".to_string()),
        });
        assert_eq!(
            basis.candidate_class,
            BlueBrainRetrievalConsolidationCandidateClass::RetrievalSupportingContextCandidate
        );
        assert_eq!(
            basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceInsufficient
        );
    }

    #[test]
    fn non_canonical_path_is_marked_non_canonical() {
        let mut report = sample_execution_report();
        report.state = crate::blue_brain_minimal_execution::BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath;

        let basis = blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
            memory_read: None,
            execution_report: Some(report),
            candidate_reference: None,
            proposal_reference: None,
            context_reference: None,
        });

        assert_eq!(
            basis.candidate_class,
            BlueBrainRetrievalConsolidationCandidateClass::NonCanonicalInternalOnlyRetrievalPath
        );
        assert_eq!(
            basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceInsufficient
        );
    }
}
