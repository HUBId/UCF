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
pub enum BlueBrainConsolidationCandidateState {
    ConsolidationCandidateOnly,
    CaveatedConsolidationCandidate,
    InsufficientConsolidationCandidate,
    BlockedConsolidationCandidate,
    NotAConsolidationCandidate,
    NonCanonicalInternalOnlyConsolidationPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainConsolidationCandidateLane {
    pub state: BlueBrainConsolidationCandidateState,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_CONSOLIDATION_CANDIDATE_MAP: [BlueBrainConsolidationCandidateLane;
    6] = [
    BlueBrainConsolidationCandidateLane {
        state: BlueBrainConsolidationCandidateState::ConsolidationCandidateOnly,
        lane: "blue_brain_consolidation_candidate_only",
        canonical_guard: "candidate-only marks bounded combined-reference eligibility and never performs merge, ranking, semantic retrieval, or reasoning",
    },
    BlueBrainConsolidationCandidateLane {
        state: BlueBrainConsolidationCandidateState::CaveatedConsolidationCandidate,
        lane: "blue_brain_consolidation_candidate_caveated",
        canonical_guard: "caveated candidate stays advisory-only and is never promoted to merged or ranked output",
    },
    BlueBrainConsolidationCandidateLane {
        state: BlueBrainConsolidationCandidateState::InsufficientConsolidationCandidate,
        lane: "blue_brain_consolidation_candidate_insufficient",
        canonical_guard: "insufficient candidate keeps weakened references explicit and never repairs basis through implicit consolidation",
    },
    BlueBrainConsolidationCandidateLane {
        state: BlueBrainConsolidationCandidateState::BlockedConsolidationCandidate,
        lane: "blue_brain_consolidation_candidate_blocked",
        canonical_guard: "blocked candidate remains blocked under stale/invalidated/failed/cancelled/blocked maintenance boundaries",
    },
    BlueBrainConsolidationCandidateLane {
        state: BlueBrainConsolidationCandidateState::NotAConsolidationCandidate,
        lane: "blue_brain_not_a_consolidation_candidate",
        canonical_guard: "single-source or context-only retrieval basis is not a consolidation candidate",
    },
    BlueBrainConsolidationCandidateLane {
        state: BlueBrainConsolidationCandidateState::NonCanonicalInternalOnlyConsolidationPath,
        lane: "blue_brain_non_canonical_internal_only_consolidation_path",
        canonical_guard: "internal-only paths remain non-canonical and cannot become canonical candidate outputs",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCombinedReferenceStatus {
    CombinedReferenceAvailable,
    CombinedReferenceCaveated,
    CombinedReferenceStale,
    CombinedReferenceInvalidated,
    CombinedReferenceFailed,
    CombinedReferenceCancelled,
    CombinedReferenceBlocked,
    CombinedReferenceInsufficient,
    ConsolidationCandidateOnly,
    NoConsolidationPerformed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCombinedRetrievalDiagnosticClass {
    CombinedReferenceAvailableDiagnostic,
    CombinedReferenceCaveatedDiagnostic,
    CombinedReferenceStaleDiagnostic,
    CombinedReferenceInvalidatedDiagnostic,
    CombinedReferenceFailedDiagnostic,
    CombinedReferenceCancelledDiagnostic,
    CombinedReferenceBlockedDiagnostic,
    CombinedReferenceInsufficientDiagnostic,
    NonCanonicalInternalOnlyCombinedReferenceDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainCombinedRetrievalDiagnosticLane {
    pub diagnostic_class: BlueBrainCombinedRetrievalDiagnosticClass,
    pub lane: &'static str,
    pub canonical_guard: &'static str,
}

pub const CANONICAL_BLUE_BRAIN_COMBINED_RETRIEVAL_DIAGNOSTICS_MAP:
    [BlueBrainCombinedRetrievalDiagnosticLane; 9] = [
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class:
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceAvailableDiagnostic,
        lane: "blue_brain_combined_reference_available_diagnostic",
        canonical_guard: "combined reference is available only when canonical memory and execution references are both present and uncaveated",
    },
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class:
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceCaveatedDiagnostic,
        lane: "blue_brain_combined_reference_caveated_diagnostic",
        canonical_guard: "caveated combined reference preserves advisory-only caveats without promotion to strong availability",
    },
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class: BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceStaleDiagnostic,
        lane: "blue_brain_combined_reference_stale_diagnostic",
        canonical_guard: "stale memory basis remains distinct from invalidated memory basis",
    },
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class:
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceInvalidatedDiagnostic,
        lane: "blue_brain_combined_reference_invalidated_diagnostic",
        canonical_guard: "invalidated memory basis prevents strong combined reference support",
    },
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class: BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceFailedDiagnostic,
        lane: "blue_brain_combined_reference_failed_diagnostic",
        canonical_guard: "failed execution reference is preserved as failed and not collapsed into cancelled or blocked",
    },
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class:
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceCancelledDiagnostic,
        lane: "blue_brain_combined_reference_cancelled_diagnostic",
        canonical_guard: "cancelled execution reference remains a distinct terminal outcome",
    },
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class:
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceBlockedDiagnostic,
        lane: "blue_brain_combined_reference_blocked_diagnostic",
        canonical_guard: "blocked or unavailable execution reference prevents strong combined reference availability",
    },
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class:
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceInsufficientDiagnostic,
        lane: "blue_brain_combined_reference_insufficient_diagnostic",
        canonical_guard: "insufficient diagnostics are advisory-only and do not trigger consolidation, ranking, semantic search, or execution",
    },
    BlueBrainCombinedRetrievalDiagnosticLane {
        diagnostic_class: BlueBrainCombinedRetrievalDiagnosticClass::
            NonCanonicalInternalOnlyCombinedReferenceDiagnostic,
        lane: "blue_brain_combined_reference_non_canonical_internal_only_diagnostic",
        canonical_guard: "internal-only paths remain non-canonical and are never normalized into canonical combined reference availability",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCombinedMemoryBasisState {
    Current,
    Caveated,
    Stale,
    Invalidated,
    Missing,
    Blocked,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainCombinedExecutionBasisState {
    Completed,
    Failed,
    Cancelled,
    Blocked,
    Unavailable,
    Unsupported,
    PlaceholderOnly,
    NonCanonicalInternalOnlyPath,
    NotObserved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainCombinedRetrievalBasis {
    pub candidate_class: BlueBrainRetrievalConsolidationCandidateClass,
    pub consolidation_candidate_state: BlueBrainConsolidationCandidateState,
    pub combined_reference_status: BlueBrainCombinedReferenceStatus,
    pub diagnostic_class: BlueBrainCombinedRetrievalDiagnosticClass,
    pub memory_record_reference: Option<String>,
    pub execution_result_reference: Option<String>,
    pub candidate_reference: Option<String>,
    pub proposal_reference: Option<String>,
    pub context_reference: Option<String>,
    pub memory_basis_state: Option<BlueBrainCombinedMemoryBasisState>,
    pub execution_basis_state: BlueBrainCombinedExecutionBasisState,
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
    pub merge_or_record_mutation_permitted: bool,
    pub ranking_permitted: bool,
    pub semantic_search_permitted: bool,
    pub reasoning_output_permitted: bool,
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

    let memory_basis_state = input
        .memory_read
        .as_ref()
        .map(|read| match read.retrieval_state {
            BlueBrainMemoryRetrievalState::RetrievedReferenceOnly => {
                BlueBrainCombinedMemoryBasisState::Current
            }
            BlueBrainMemoryRetrievalState::RetrievedWithCaveat => {
                BlueBrainCombinedMemoryBasisState::Caveated
            }
            BlueBrainMemoryRetrievalState::RetrievedStale => {
                BlueBrainCombinedMemoryBasisState::Stale
            }
            BlueBrainMemoryRetrievalState::RetrievedInvalidated => {
                BlueBrainCombinedMemoryBasisState::Invalidated
            }
            BlueBrainMemoryRetrievalState::Missing => BlueBrainCombinedMemoryBasisState::Missing,
            BlueBrainMemoryRetrievalState::Blocked => BlueBrainCombinedMemoryBasisState::Blocked,
            BlueBrainMemoryRetrievalState::Unavailable => {
                BlueBrainCombinedMemoryBasisState::Unavailable
            }
        });
    let execution_basis_state = input.execution_report.as_ref().map_or(
        BlueBrainCombinedExecutionBasisState::NotObserved,
        |report| match report.outcome_class {
            BlueBrainExecutionOutcomeClass::ExecutionCompleted => {
                BlueBrainCombinedExecutionBasisState::Completed
            }
            BlueBrainExecutionOutcomeClass::ExecutionFailed => {
                BlueBrainCombinedExecutionBasisState::Failed
            }
            BlueBrainExecutionOutcomeClass::ExecutionCancelled => {
                BlueBrainCombinedExecutionBasisState::Cancelled
            }
            BlueBrainExecutionOutcomeClass::ExecutionBlocked => {
                BlueBrainCombinedExecutionBasisState::Blocked
            }
            BlueBrainExecutionOutcomeClass::ExecutionUnavailable => {
                BlueBrainCombinedExecutionBasisState::Unavailable
            }
            BlueBrainExecutionOutcomeClass::ExecutionUnsupported => {
                BlueBrainCombinedExecutionBasisState::Unsupported
            }
            BlueBrainExecutionOutcomeClass::ExecutionPlaceholderOnly => {
                BlueBrainCombinedExecutionBasisState::PlaceholderOnly
            }
            BlueBrainExecutionOutcomeClass::NonCanonicalInternalOnlyPath => {
                BlueBrainCombinedExecutionBasisState::NonCanonicalInternalOnlyPath
            }
        },
    );

    match memory_basis_state {
        Some(BlueBrainCombinedMemoryBasisState::Stale) => {
            caveats.push("memory basis is stale".to_string())
        }
        Some(BlueBrainCombinedMemoryBasisState::Invalidated) => {
            caveats.push("memory basis is invalidated".to_string())
        }
        Some(BlueBrainCombinedMemoryBasisState::Caveated) => {
            caveats.push("memory basis is caveated".to_string())
        }
        Some(BlueBrainCombinedMemoryBasisState::Blocked) => {
            caveats.push("memory retrieval basis is blocked".to_string())
        }
        Some(BlueBrainCombinedMemoryBasisState::Unavailable) => {
            caveats.push("memory retrieval basis is unavailable".to_string())
        }
        Some(BlueBrainCombinedMemoryBasisState::Missing) => {
            caveats.push("memory retrieval basis is missing".to_string())
        }
        Some(BlueBrainCombinedMemoryBasisState::Current) | None => {}
    }
    match execution_basis_state {
        BlueBrainCombinedExecutionBasisState::Failed => {
            caveats.push("execution reference is failed".to_string())
        }
        BlueBrainCombinedExecutionBasisState::Cancelled => {
            caveats.push("execution reference is cancelled".to_string())
        }
        BlueBrainCombinedExecutionBasisState::Blocked => {
            caveats.push("execution reference is blocked".to_string())
        }
        BlueBrainCombinedExecutionBasisState::Unavailable => {
            caveats.push("execution reference is unavailable".to_string())
        }
        BlueBrainCombinedExecutionBasisState::Unsupported => {
            caveats.push("execution reference is unsupported".to_string())
        }
        BlueBrainCombinedExecutionBasisState::PlaceholderOnly => {
            caveats.push("execution basis is placeholder only".to_string())
        }
        BlueBrainCombinedExecutionBasisState::NonCanonicalInternalOnlyPath => {
            caveats.push("execution basis is non-canonical".to_string())
        }
        BlueBrainCombinedExecutionBasisState::Completed
        | BlueBrainCombinedExecutionBasisState::NotObserved => {}
    }

    let stale_invalidated_or_failed_references_weaken_basis = matches!(
        memory_basis_state,
        Some(
            BlueBrainCombinedMemoryBasisState::Stale
                | BlueBrainCombinedMemoryBasisState::Invalidated
                | BlueBrainCombinedMemoryBasisState::Blocked
                | BlueBrainCombinedMemoryBasisState::Unavailable
                | BlueBrainCombinedMemoryBasisState::Missing
        )
    ) || matches!(
        execution_basis_state,
        BlueBrainCombinedExecutionBasisState::Failed
            | BlueBrainCombinedExecutionBasisState::Cancelled
            | BlueBrainCombinedExecutionBasisState::Blocked
            | BlueBrainCombinedExecutionBasisState::Unavailable
            | BlueBrainCombinedExecutionBasisState::Unsupported
            | BlueBrainCombinedExecutionBasisState::PlaceholderOnly
    );

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
            match (memory_basis_state, execution_basis_state) {
                (Some(BlueBrainCombinedMemoryBasisState::Invalidated), _) => {
                    BlueBrainCombinedReferenceStatus::CombinedReferenceInvalidated
                }
                (Some(BlueBrainCombinedMemoryBasisState::Stale), _) => {
                    BlueBrainCombinedReferenceStatus::CombinedReferenceStale
                }
                (_, BlueBrainCombinedExecutionBasisState::Failed) => {
                    BlueBrainCombinedReferenceStatus::CombinedReferenceFailed
                }
                (_, BlueBrainCombinedExecutionBasisState::Cancelled) => {
                    BlueBrainCombinedReferenceStatus::CombinedReferenceCancelled
                }
                (
                    _,
                    BlueBrainCombinedExecutionBasisState::Blocked
                    | BlueBrainCombinedExecutionBasisState::Unavailable
                    | BlueBrainCombinedExecutionBasisState::Unsupported
                    | BlueBrainCombinedExecutionBasisState::PlaceholderOnly,
                ) => BlueBrainCombinedReferenceStatus::CombinedReferenceBlocked,
                (Some(BlueBrainCombinedMemoryBasisState::Caveated), _) => {
                    BlueBrainCombinedReferenceStatus::CombinedReferenceCaveated
                }
                (
                    Some(BlueBrainCombinedMemoryBasisState::Current),
                    BlueBrainCombinedExecutionBasisState::Completed,
                ) => BlueBrainCombinedReferenceStatus::CombinedReferenceAvailable,
                _ => BlueBrainCombinedReferenceStatus::CombinedReferenceCaveated,
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
    let consolidation_candidate_state = match candidate_class {
        BlueBrainRetrievalConsolidationCandidateClass::NonCanonicalInternalOnlyRetrievalPath => {
            BlueBrainConsolidationCandidateState::NonCanonicalInternalOnlyConsolidationPath
        }
        BlueBrainRetrievalConsolidationCandidateClass::CombinedReferenceCandidate => {
            match combined_reference_status {
                BlueBrainCombinedReferenceStatus::CombinedReferenceAvailable => {
                    BlueBrainConsolidationCandidateState::ConsolidationCandidateOnly
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceCaveated => {
                    BlueBrainConsolidationCandidateState::CaveatedConsolidationCandidate
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceStale
                | BlueBrainCombinedReferenceStatus::CombinedReferenceInvalidated
                | BlueBrainCombinedReferenceStatus::CombinedReferenceFailed
                | BlueBrainCombinedReferenceStatus::CombinedReferenceCancelled
                | BlueBrainCombinedReferenceStatus::CombinedReferenceInsufficient => {
                    BlueBrainConsolidationCandidateState::InsufficientConsolidationCandidate
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceBlocked => {
                    BlueBrainConsolidationCandidateState::BlockedConsolidationCandidate
                }
                BlueBrainCombinedReferenceStatus::ConsolidationCandidateOnly
                | BlueBrainCombinedReferenceStatus::NoConsolidationPerformed => {
                    BlueBrainConsolidationCandidateState::NotAConsolidationCandidate
                }
            }
        }
        BlueBrainRetrievalConsolidationCandidateClass::MemoryRetrievalCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::ExecutionResultRetrievalCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::RetrievalSupportingContextCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::InsufficientRetrievalBasis
        | BlueBrainRetrievalConsolidationCandidateClass::ConsolidationCandidateOnly => {
            BlueBrainConsolidationCandidateState::NotAConsolidationCandidate
        }
    };
    let diagnostic_class = match candidate_class {
        BlueBrainRetrievalConsolidationCandidateClass::NonCanonicalInternalOnlyRetrievalPath => {
            BlueBrainCombinedRetrievalDiagnosticClass::NonCanonicalInternalOnlyCombinedReferenceDiagnostic
        }
        BlueBrainRetrievalConsolidationCandidateClass::CombinedReferenceCandidate => {
            match combined_reference_status {
                BlueBrainCombinedReferenceStatus::CombinedReferenceAvailable => {
                    BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceAvailableDiagnostic
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceCaveated => {
                    BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceCaveatedDiagnostic
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceStale => {
                    BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceStaleDiagnostic
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceInvalidated => {
                    BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceInvalidatedDiagnostic
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceFailed => {
                    BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceFailedDiagnostic
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceCancelled => {
                    BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceCancelledDiagnostic
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceBlocked => {
                    BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceBlockedDiagnostic
                }
                BlueBrainCombinedReferenceStatus::CombinedReferenceInsufficient
                | BlueBrainCombinedReferenceStatus::ConsolidationCandidateOnly
                | BlueBrainCombinedReferenceStatus::NoConsolidationPerformed => {
                    BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceInsufficientDiagnostic
                }
            }
        }
        BlueBrainRetrievalConsolidationCandidateClass::MemoryRetrievalCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::ExecutionResultRetrievalCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::RetrievalSupportingContextCandidate
        | BlueBrainRetrievalConsolidationCandidateClass::InsufficientRetrievalBasis
        | BlueBrainRetrievalConsolidationCandidateClass::ConsolidationCandidateOnly => {
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceInsufficientDiagnostic
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
        consolidation_candidate_state,
        combined_reference_status,
        diagnostic_class,
        memory_record_reference,
        execution_result_reference,
        candidate_reference: input.candidate_reference,
        proposal_reference: input.proposal_reference,
        context_reference: input.context_reference,
        memory_basis_state,
        execution_basis_state,
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
        merge_or_record_mutation_permitted: false,
        ranking_permitted: false,
        semantic_search_permitted: false,
        reasoning_output_permitted: false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        blue_brain_build_combined_retrieval_basis, BlueBrainCombinedExecutionBasisState,
        BlueBrainCombinedMemoryBasisState, BlueBrainCombinedReferenceStatus,
        BlueBrainCombinedRetrievalDiagnosticClass, BlueBrainCombinedRetrievalInput,
        BlueBrainConsolidationCandidateState, BlueBrainRetrievalConsolidationCandidateClass,
        CANONICAL_BLUE_BRAIN_COMBINED_RETRIEVAL_DIAGNOSTICS_MAP,
        CANONICAL_BLUE_BRAIN_CONSOLIDATION_CANDIDATE_MAP,
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
        assert_eq!(
            basis.consolidation_candidate_state,
            BlueBrainConsolidationCandidateState::ConsolidationCandidateOnly
        );
        assert_eq!(
            basis.diagnostic_class,
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceAvailableDiagnostic
        );
        assert_eq!(
            basis.memory_basis_state,
            Some(BlueBrainCombinedMemoryBasisState::Current)
        );
        assert_eq!(
            basis.execution_basis_state,
            BlueBrainCombinedExecutionBasisState::Completed
        );
        assert!(basis.reference_basis_supports_selection_or_proposal_only);
        assert!(!basis.automatic_compute_invoked);
        assert!(!basis.automatic_action_executed);
        assert!(!basis.automatic_memory_persisted);
        assert!(!basis.merge_or_record_mutation_permitted);
        assert!(!basis.ranking_permitted);
        assert!(!basis.semantic_search_permitted);
        assert!(!basis.reasoning_output_permitted);
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
            BlueBrainCombinedReferenceStatus::CombinedReferenceStale
        );
        assert_eq!(
            basis.consolidation_candidate_state,
            BlueBrainConsolidationCandidateState::InsufficientConsolidationCandidate
        );
        assert_eq!(
            basis.diagnostic_class,
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceStaleDiagnostic
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
        assert_eq!(
            basis.consolidation_candidate_state,
            BlueBrainConsolidationCandidateState::NotAConsolidationCandidate
        );
        assert_eq!(
            basis.diagnostic_class,
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceInsufficientDiagnostic
        );
    }

    #[test]
    fn memory_only_or_execution_only_reference_stays_not_a_consolidation_candidate() {
        let memory_only =
            blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
                memory_read: Some(sample_memory_read(
                    BlueBrainMemoryRetrievalState::RetrievedReferenceOnly,
                )),
                execution_report: None,
                candidate_reference: Some("cand-memory-only".to_string()),
                proposal_reference: None,
                context_reference: Some("ctx-memory-only".to_string()),
            });
        assert_eq!(
            memory_only.candidate_class,
            BlueBrainRetrievalConsolidationCandidateClass::MemoryRetrievalCandidate
        );
        assert_eq!(
            memory_only.consolidation_candidate_state,
            BlueBrainConsolidationCandidateState::NotAConsolidationCandidate
        );
        assert_eq!(
            memory_only.diagnostic_class,
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceInsufficientDiagnostic
        );

        let execution_only =
            blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
                memory_read: None,
                execution_report: Some(sample_execution_report()),
                candidate_reference: None,
                proposal_reference: Some("proposal-exec-only".to_string()),
                context_reference: Some("ctx-exec-only".to_string()),
            });
        assert_eq!(
            execution_only.candidate_class,
            BlueBrainRetrievalConsolidationCandidateClass::ExecutionResultRetrievalCandidate
        );
        assert_eq!(
            execution_only.consolidation_candidate_state,
            BlueBrainConsolidationCandidateState::NotAConsolidationCandidate
        );
        assert_eq!(
            execution_only.diagnostic_class,
            BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceInsufficientDiagnostic
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
        assert_eq!(
            basis.diagnostic_class,
            BlueBrainCombinedRetrievalDiagnosticClass::NonCanonicalInternalOnlyCombinedReferenceDiagnostic
        );
        assert_eq!(
            basis.consolidation_candidate_state,
            BlueBrainConsolidationCandidateState::NonCanonicalInternalOnlyConsolidationPath
        );
    }

    #[test]
    fn combined_reference_diagnostics_map_covers_required_classes() {
        assert_eq!(
            CANONICAL_BLUE_BRAIN_COMBINED_RETRIEVAL_DIAGNOSTICS_MAP.len(),
            9
        );
        assert!(CANONICAL_BLUE_BRAIN_COMBINED_RETRIEVAL_DIAGNOSTICS_MAP
            .iter()
            .any(|lane| {
                lane.diagnostic_class
                == BlueBrainCombinedRetrievalDiagnosticClass::CombinedReferenceCancelledDiagnostic
            }));
    }

    #[test]
    fn consolidation_candidate_map_covers_required_states() {
        assert_eq!(CANONICAL_BLUE_BRAIN_CONSOLIDATION_CANDIDATE_MAP.len(), 6);
        assert!(CANONICAL_BLUE_BRAIN_CONSOLIDATION_CANDIDATE_MAP
            .iter()
            .any(|lane| lane.state
                == BlueBrainConsolidationCandidateState::BlockedConsolidationCandidate));
    }

    #[test]
    fn stale_invalidated_failed_cancelled_and_blocked_stay_distinct() {
        let mut failed_report = sample_execution_report();
        failed_report.outcome_class =
            crate::blue_brain_minimal_execution::BlueBrainExecutionOutcomeClass::ExecutionFailed;
        let failed_basis =
            blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
                memory_read: Some(sample_memory_read(
                    BlueBrainMemoryRetrievalState::RetrievedReferenceOnly,
                )),
                execution_report: Some(failed_report),
                candidate_reference: None,
                proposal_reference: None,
                context_reference: None,
            });
        assert_eq!(
            failed_basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceFailed
        );

        let mut cancelled_report = sample_execution_report();
        cancelled_report.outcome_class =
            crate::blue_brain_minimal_execution::BlueBrainExecutionOutcomeClass::ExecutionCancelled;
        let cancelled_basis =
            blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
                memory_read: Some(sample_memory_read(
                    BlueBrainMemoryRetrievalState::RetrievedReferenceOnly,
                )),
                execution_report: Some(cancelled_report),
                candidate_reference: None,
                proposal_reference: None,
                context_reference: None,
            });
        assert_eq!(
            cancelled_basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceCancelled
        );

        let mut blocked_report = sample_execution_report();
        blocked_report.outcome_class =
            crate::blue_brain_minimal_execution::BlueBrainExecutionOutcomeClass::ExecutionBlocked;
        let blocked_basis =
            blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
                memory_read: Some(sample_memory_read(
                    BlueBrainMemoryRetrievalState::RetrievedReferenceOnly,
                )),
                execution_report: Some(blocked_report),
                candidate_reference: None,
                proposal_reference: None,
                context_reference: None,
            });
        assert_eq!(
            blocked_basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceBlocked
        );

        let invalidated_basis =
            blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
                memory_read: Some(sample_memory_read(
                    BlueBrainMemoryRetrievalState::RetrievedInvalidated,
                )),
                execution_report: Some(sample_execution_report()),
                candidate_reference: None,
                proposal_reference: None,
                context_reference: None,
            });
        assert_eq!(
            invalidated_basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceInvalidated
        );

        let stale_basis =
            blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
                memory_read: Some(sample_memory_read(
                    BlueBrainMemoryRetrievalState::RetrievedStale,
                )),
                execution_report: Some(sample_execution_report()),
                candidate_reference: None,
                proposal_reference: None,
                context_reference: None,
            });
        assert_eq!(
            stale_basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceStale
        );
    }

    #[test]
    fn combined_basis_remains_advisory_only_and_never_merge_or_ranking_capable() {
        let basis = blue_brain_build_combined_retrieval_basis(BlueBrainCombinedRetrievalInput {
            memory_read: Some(sample_memory_read(
                BlueBrainMemoryRetrievalState::RetrievedWithCaveat,
            )),
            execution_report: Some(sample_execution_report()),
            candidate_reference: Some("cand-advisory".to_string()),
            proposal_reference: Some("proposal-advisory".to_string()),
            context_reference: Some("ctx-advisory".to_string()),
        });

        assert_eq!(
            basis.combined_reference_status,
            BlueBrainCombinedReferenceStatus::CombinedReferenceCaveated
        );
        assert!(!basis.automatic_compute_invoked);
        assert!(!basis.automatic_action_executed);
        assert!(!basis.automatic_memory_persisted);
        assert!(!basis.merge_or_record_mutation_permitted);
        assert!(!basis.ranking_permitted);
        assert!(!basis.semantic_search_permitted);
        assert!(!basis.reasoning_output_permitted);
    }
}
