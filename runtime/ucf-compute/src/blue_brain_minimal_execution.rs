use crate::reference_map::{
    BlueBrainActionExecutionEligibilityClass, BlueBrainFutureActionHandoffClass,
    BlueBrainSafetyPrecheckClass,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMinimalExecutionAction {
    EmitCanonicalSignal,
    UnsupportedAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMinimalExecutionState {
    ExecutionEligibleButNotExecuted,
    ExecutionRequested,
    ExecutionStarted,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionUnsupported,
    ExecutionBlocked,
    ExecutionCancelled,
    ExecutionUnavailable,
    NonCanonicalInternalOnlyPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionOutcomeClass {
    ExecutionCompleted,
    ExecutionBlocked,
    ExecutionUnavailable,
    ExecutionFailed,
    ExecutionCancelled,
    ExecutionUnsupported,
    ExecutionPlaceholderOnly,
    NonCanonicalInternalOnlyPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionRetryDisposition {
    RetryableFailure,
    NonRetryableFailure,
    RetryNotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionFailurePathClass {
    CanonicalFailurePath,
    NonCanonicalInternalOnlyFailurePath,
    NotAFailurePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMinimalExecutionResultBoundary {
    PlaceholderOnly,
    ExecutionRequested,
    ActualExecutionResult,
    FailedExecutionResult,
    CancelledExecutionResult,
    BlockedNoResult,
    UnsupportedNoResult,
    UnavailableExecutionPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionReferenceClass {
    ExecutionRequestReference,
    ExecutionResultReference,
    FailureResultReference,
    CancellationResultReference,
    BlockedOrUnavailableReference,
    PlaceholderReference,
    EligibilityReference,
    NonCanonicalInternalOnlyReferencePath,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionReference {
    pub class: BlueBrainExecutionReferenceClass,
    pub path: String,
    pub terminal: bool,
    pub canonical: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionReferenceMap {
    pub execution_request_reference: BlueBrainExecutionReference,
    pub execution_result_reference: Option<BlueBrainExecutionReference>,
    pub failure_result_reference: Option<BlueBrainExecutionReference>,
    pub cancellation_result_reference: Option<BlueBrainExecutionReference>,
    pub blocked_or_unavailable_reference: Option<BlueBrainExecutionReference>,
    pub placeholder_reference: Option<BlueBrainExecutionReference>,
    pub eligibility_reference: BlueBrainExecutionReference,
    pub non_canonical_internal_only_reference_path: Option<BlueBrainExecutionReference>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMinimalExecutionTraceCore {
    pub canonical_action: BlueBrainMinimalExecutionAction,
    pub handoff_id: String,
    pub eligibility_class: BlueBrainActionExecutionEligibilityClass,
    pub safety_precheck: BlueBrainSafetyPrecheckClass,
    pub state: BlueBrainMinimalExecutionState,
    pub outcome_class: BlueBrainExecutionOutcomeClass,
    pub retry_disposition: BlueBrainExecutionRetryDisposition,
    pub failure_path_class: BlueBrainExecutionFailurePathClass,
    pub result_boundary: BlueBrainMinimalExecutionResultBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionResultIntegrityClass {
    ResultRecordedCanonical,
    ResultFailedCanonical,
    ResultCancelledCanonical,
    ResultBlockedCanonical,
    ResultUnavailableCanonical,
    ResultCaveatedCanonical,
    IntegrityMismatch,
    NonCanonicalInternalOnlyResultPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionTransitionClass {
    PreExecutionBoundary,
    EnteredExecution,
    TerminalCompleted,
    TerminalFailed,
    TerminalCancelled,
    TerminalBlocked,
    TerminalUnavailable,
    TerminalUnsupported,
    TerminalNonCanonical,
    InvalidTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionProductionHardeningPathClass {
    HardenedCanonicalExecutionPath,
    HardenedFailurePath,
    HardenedBlockedOrUnavailablePath,
    HardenedCancellationPath,
    HardenedReferenceOrResultPath,
    GuardSensitivePath,
    NonCanonicalInternalOnlyExecutionPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlueBrainExecutionEdgeCaseClass {
    BlockedBeforeStartEdgeCase,
    CancelledAfterStartEdgeCase,
    FailureAfterStartEdgeCase,
    PartialExecutionPath,
    IncompleteResultPath,
    ConflictingTerminalStateAttempt,
    DuplicateTerminalizationAttempt,
    NonCanonicalInternalOnlyEdgePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMinimalCapabilityScopeClass {
    AllowedCanonicalAction,
    AllowedCanonicalToolCall,
    BlockedAction,
    UnsupportedAction,
    UnavailableAction,
    NonCanonicalInternalOnlyActionPath,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMinimalExecutionRequest {
    pub handoff_id: String,
    pub handoff_class: BlueBrainFutureActionHandoffClass,
    pub eligibility_class: BlueBrainActionExecutionEligibilityClass,
    pub safety_precheck: BlueBrainSafetyPrecheckClass,
    pub action: BlueBrainMinimalExecutionAction,
    pub execution_requested: bool,
    pub cancelled: bool,
    pub internal_only_path: bool,
    pub force_execution_failure: bool,
    pub force_nonretryable_failure: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMinimalExecutionReport {
    pub trace_core: BlueBrainMinimalExecutionTraceCore,
    pub references: BlueBrainExecutionReferenceMap,
    pub state: BlueBrainMinimalExecutionState,
    pub outcome_class: BlueBrainExecutionOutcomeClass,
    pub retry_disposition: BlueBrainExecutionRetryDisposition,
    pub failure_path_class: BlueBrainExecutionFailurePathClass,
    pub result_boundary: BlueBrainMinimalExecutionResultBoundary,
    pub execution_requested: bool,
    pub execution_started: bool,
    pub execution_completed: bool,
    pub execution_failed: bool,
    pub executed_action_result: Option<String>,
    pub error_code: Option<&'static str>,
    pub notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionFeedbackClass {
    ExecutionCompletedFeedback,
    ExecutionFailedFeedback,
    ExecutionUnsupportedFeedback,
    ExecutionCancelledFeedback,
    ExecutionBlockedFeedback,
    ExecutionUnavailableFeedback,
    ExecutionCaveatedFeedback,
    EligibilityPlaceholderOnlyFeedback,
    NonCanonicalInternalOnlyExecutionFeedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainExecutionFailureReasonClass {
    ExecutionPathError,
    UnsupportedAction,
    CancelledBeforeCompletion,
    NotRequestedPlaceholderOnly,
    SafetyOrBoundaryBlocked,
    ExecutionPathUnavailable,
    CaveatedCompletion,
    NonCanonicalInternalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainProposalExecutionFeedbackClass {
    ProposalNotConsumedByExecution,
    ProposalConsumedByExecution,
    ProposalExecutionCompleted,
    ProposalExecutionFailed,
    ProposalExecutionUnsupported,
    ProposalExecutionCancelled,
    ProposalExecutionBlocked,
    ProposalExecutionUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionRuntimeFeedback {
    pub class: BlueBrainExecutionFeedbackClass,
    pub reason: Option<BlueBrainExecutionFailureReasonClass>,
    pub sees_actual_execution_result: bool,
    pub sees_placeholder_only: bool,
    pub retry_disposition: BlueBrainExecutionRetryDisposition,
    pub canonical_result_reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionSelectionFeedback {
    pub proposal_feedback_class: BlueBrainProposalExecutionFeedbackClass,
    pub proposal_consumed_by_execution: bool,
    pub automatic_next_proposal_generation: bool,
    pub canonical_request_reference: String,
    pub canonical_eligibility_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionMemoryFeedback {
    pub may_attach_context_reference: bool,
    pub may_attach_diagnostic_reference: bool,
    pub automatic_memory_commit_performed: bool,
    pub canonical_memory_basis_reference: Option<String>,
}

fn minimal_action_token(action: BlueBrainMinimalExecutionAction) -> &'static str {
    match action {
        BlueBrainMinimalExecutionAction::EmitCanonicalSignal => "emit_canonical_signal",
        BlueBrainMinimalExecutionAction::UnsupportedAction => "unsupported_action",
    }
}

fn make_reference(
    class: BlueBrainExecutionReferenceClass,
    path: String,
    terminal: bool,
    canonical: bool,
) -> BlueBrainExecutionReference {
    BlueBrainExecutionReference {
        class,
        path,
        terminal,
        canonical,
    }
}

fn build_reference_map(
    request: &BlueBrainMinimalExecutionRequest,
    report: &BlueBrainMinimalExecutionReport,
) -> BlueBrainExecutionReferenceMap {
    let action = minimal_action_token(request.action);
    let base = format!(
        "bb14:minimal_execution:{}:{}",
        request.handoff_id.trim(),
        action
    );
    let execution_request_reference = make_reference(
        BlueBrainExecutionReferenceClass::ExecutionRequestReference,
        format!("{base}:request"),
        false,
        !request.internal_only_path,
    );
    let eligibility_reference = make_reference(
        BlueBrainExecutionReferenceClass::EligibilityReference,
        format!(
            "{base}:eligibility:{:?}:precheck:{:?}",
            request.eligibility_class, request.safety_precheck
        ),
        false,
        !request.internal_only_path,
    );
    let execution_result_reference =
        (report.state == BlueBrainMinimalExecutionState::ExecutionCompleted).then(|| {
            make_reference(
                BlueBrainExecutionReferenceClass::ExecutionResultReference,
                format!("{base}:result:completed"),
                true,
                true,
            )
        });
    let failure_result_reference =
        (report.state == BlueBrainMinimalExecutionState::ExecutionFailed).then(|| {
            make_reference(
                BlueBrainExecutionReferenceClass::FailureResultReference,
                format!("{base}:result:failed"),
                true,
                true,
            )
        });
    let cancellation_result_reference =
        (report.state == BlueBrainMinimalExecutionState::ExecutionCancelled).then(|| {
            make_reference(
                BlueBrainExecutionReferenceClass::CancellationResultReference,
                format!("{base}:result:cancelled"),
                true,
                true,
            )
        });
    let blocked_or_unavailable_reference = matches!(
        report.state,
        BlueBrainMinimalExecutionState::ExecutionBlocked
            | BlueBrainMinimalExecutionState::ExecutionUnavailable
            | BlueBrainMinimalExecutionState::ExecutionUnsupported
    )
    .then(|| {
        make_reference(
            BlueBrainExecutionReferenceClass::BlockedOrUnavailableReference,
            format!("{base}:result:{:?}", report.state),
            true,
            true,
        )
    });
    let placeholder_reference = (report.state
        == BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted)
        .then(|| {
            make_reference(
                BlueBrainExecutionReferenceClass::PlaceholderReference,
                format!("{base}:placeholder:eligible_not_executed"),
                false,
                true,
            )
        });
    let non_canonical_internal_only_reference_path =
        (report.state == BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath).then(|| {
            make_reference(
                BlueBrainExecutionReferenceClass::NonCanonicalInternalOnlyReferencePath,
                format!("{base}:non_canonical_internal_only"),
                true,
                false,
            )
        });

    BlueBrainExecutionReferenceMap {
        execution_request_reference,
        execution_result_reference,
        failure_result_reference,
        cancellation_result_reference,
        blocked_or_unavailable_reference,
        placeholder_reference,
        eligibility_reference,
        non_canonical_internal_only_reference_path,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionFeedbackBackbind {
    pub runtime: BlueBrainExecutionRuntimeFeedback,
    pub selection: BlueBrainExecutionSelectionFeedback,
    pub memory: BlueBrainExecutionMemoryFeedback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlueBrainExecutionResultIntegrity {
    pub class: BlueBrainExecutionResultIntegrityClass,
    pub transition: BlueBrainExecutionTransitionClass,
    pub is_terminal: bool,
    pub canonical: bool,
}

pub fn blue_brain_execution_production_hardening_path(
    report: &BlueBrainMinimalExecutionReport,
) -> BlueBrainExecutionProductionHardeningPathClass {
    use BlueBrainExecutionProductionHardeningPathClass as HardeningPath;

    if matches!(
        report.state,
        BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath
    ) {
        return HardeningPath::NonCanonicalInternalOnlyExecutionPath;
    }

    if matches!(
        report.state,
        BlueBrainMinimalExecutionState::ExecutionBlocked
            | BlueBrainMinimalExecutionState::ExecutionUnavailable
            | BlueBrainMinimalExecutionState::ExecutionUnsupported
    ) {
        return HardeningPath::HardenedBlockedOrUnavailablePath;
    }

    if report.state == BlueBrainMinimalExecutionState::ExecutionCancelled {
        return HardeningPath::HardenedCancellationPath;
    }

    if report.state == BlueBrainMinimalExecutionState::ExecutionFailed {
        return HardeningPath::HardenedFailurePath;
    }

    if report.references.execution_result_reference.is_some()
        || report.references.failure_result_reference.is_some()
        || report.references.cancellation_result_reference.is_some()
    {
        return HardeningPath::HardenedReferenceOrResultPath;
    }

    if matches!(
        report.state,
        BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted
            | BlueBrainMinimalExecutionState::ExecutionRequested
            | BlueBrainMinimalExecutionState::ExecutionStarted
    ) {
        return HardeningPath::GuardSensitivePath;
    }

    HardeningPath::HardenedCanonicalExecutionPath
}

pub fn blue_brain_execution_result_integrity(
    report: &BlueBrainMinimalExecutionReport,
) -> BlueBrainExecutionResultIntegrity {
    use BlueBrainExecutionResultIntegrityClass as Integrity;
    use BlueBrainExecutionTransitionClass as Transition;

    let transition = match report.state {
        BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted => {
            Transition::PreExecutionBoundary
        }
        BlueBrainMinimalExecutionState::ExecutionRequested
        | BlueBrainMinimalExecutionState::ExecutionStarted => Transition::EnteredExecution,
        BlueBrainMinimalExecutionState::ExecutionCompleted => Transition::TerminalCompleted,
        BlueBrainMinimalExecutionState::ExecutionFailed => Transition::TerminalFailed,
        BlueBrainMinimalExecutionState::ExecutionCancelled => Transition::TerminalCancelled,
        BlueBrainMinimalExecutionState::ExecutionBlocked => Transition::TerminalBlocked,
        BlueBrainMinimalExecutionState::ExecutionUnavailable => Transition::TerminalUnavailable,
        BlueBrainMinimalExecutionState::ExecutionUnsupported => Transition::TerminalUnsupported,
        BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath => {
            Transition::TerminalNonCanonical
        }
    };

    let class = match report.state {
        BlueBrainMinimalExecutionState::ExecutionCompleted => {
            if report
                .notes
                .iter()
                .any(|n| n.contains("caveat") || *n == "execution-caveated")
            {
                Integrity::ResultCaveatedCanonical
            } else {
                Integrity::ResultRecordedCanonical
            }
        }
        BlueBrainMinimalExecutionState::ExecutionFailed => Integrity::ResultFailedCanonical,
        BlueBrainMinimalExecutionState::ExecutionCancelled => Integrity::ResultCancelledCanonical,
        BlueBrainMinimalExecutionState::ExecutionBlocked
        | BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted
        | BlueBrainMinimalExecutionState::ExecutionRequested
        | BlueBrainMinimalExecutionState::ExecutionStarted => Integrity::ResultBlockedCanonical,
        BlueBrainMinimalExecutionState::ExecutionUnavailable
        | BlueBrainMinimalExecutionState::ExecutionUnsupported => {
            Integrity::ResultUnavailableCanonical
        }
        BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath => {
            Integrity::NonCanonicalInternalOnlyResultPath
        }
    };

    let boundary_matches = matches!(
        (report.state, report.result_boundary),
        (
            BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted,
            BlueBrainMinimalExecutionResultBoundary::PlaceholderOnly,
        ) | (
            BlueBrainMinimalExecutionState::ExecutionRequested,
            BlueBrainMinimalExecutionResultBoundary::ExecutionRequested,
        ) | (
            BlueBrainMinimalExecutionState::ExecutionStarted,
            BlueBrainMinimalExecutionResultBoundary::ExecutionRequested,
        ) | (
            BlueBrainMinimalExecutionState::ExecutionCompleted,
            BlueBrainMinimalExecutionResultBoundary::ActualExecutionResult,
        ) | (
            BlueBrainMinimalExecutionState::ExecutionFailed,
            BlueBrainMinimalExecutionResultBoundary::FailedExecutionResult,
        ) | (
            BlueBrainMinimalExecutionState::ExecutionCancelled,
            BlueBrainMinimalExecutionResultBoundary::CancelledExecutionResult,
        ) | (
            BlueBrainMinimalExecutionState::ExecutionBlocked,
            BlueBrainMinimalExecutionResultBoundary::BlockedNoResult,
        ) | (
            BlueBrainMinimalExecutionState::ExecutionUnavailable,
            BlueBrainMinimalExecutionResultBoundary::UnavailableExecutionPath,
        ) | (
            BlueBrainMinimalExecutionState::ExecutionUnsupported,
            BlueBrainMinimalExecutionResultBoundary::UnsupportedNoResult,
        ) | (
            BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath,
            BlueBrainMinimalExecutionResultBoundary::UnavailableExecutionPath,
        )
    );

    let lifecycle_matches = match report.state {
        BlueBrainMinimalExecutionState::ExecutionCompleted => {
            report.execution_started && report.execution_completed && !report.execution_failed
        }
        BlueBrainMinimalExecutionState::ExecutionFailed => {
            report.execution_started && !report.execution_completed && report.execution_failed
        }
        BlueBrainMinimalExecutionState::ExecutionCancelled => {
            !report.execution_started && !report.execution_completed && !report.execution_failed
        }
        BlueBrainMinimalExecutionState::ExecutionBlocked
        | BlueBrainMinimalExecutionState::ExecutionUnavailable
        | BlueBrainMinimalExecutionState::ExecutionUnsupported
        | BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted
        | BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath => {
            !report.execution_started && !report.execution_completed && !report.execution_failed
        }
        BlueBrainMinimalExecutionState::ExecutionRequested => {
            report.execution_requested
                && !report.execution_started
                && !report.execution_completed
                && !report.execution_failed
        }
        BlueBrainMinimalExecutionState::ExecutionStarted => {
            report.execution_requested
                && report.execution_started
                && !report.execution_completed
                && !report.execution_failed
        }
    };

    let result_matches = match report.state {
        BlueBrainMinimalExecutionState::ExecutionCompleted => {
            report.executed_action_result.is_some()
        }
        _ => report.executed_action_result.is_none(),
    };

    let edge_cases = blue_brain_execution_edge_case_map(report);
    let has_disallowed_edge_case = edge_cases.iter().any(|edge| {
        matches!(
            edge,
            BlueBrainExecutionEdgeCaseClass::CancelledAfterStartEdgeCase
                | BlueBrainExecutionEdgeCaseClass::PartialExecutionPath
                | BlueBrainExecutionEdgeCaseClass::IncompleteResultPath
                | BlueBrainExecutionEdgeCaseClass::ConflictingTerminalStateAttempt
                | BlueBrainExecutionEdgeCaseClass::DuplicateTerminalizationAttempt
        )
    });

    if !boundary_matches || !lifecycle_matches || !result_matches || has_disallowed_edge_case {
        return BlueBrainExecutionResultIntegrity {
            class: Integrity::IntegrityMismatch,
            transition: Transition::InvalidTransition,
            is_terminal: true,
            canonical: false,
        };
    }

    BlueBrainExecutionResultIntegrity {
        class,
        transition,
        is_terminal: !matches!(
            transition,
            Transition::PreExecutionBoundary | Transition::EnteredExecution
        ),
        canonical: !matches!(class, Integrity::NonCanonicalInternalOnlyResultPath),
    }
}

pub fn blue_brain_execution_edge_case_map(
    report: &BlueBrainMinimalExecutionReport,
) -> Vec<BlueBrainExecutionEdgeCaseClass> {
    use BlueBrainExecutionEdgeCaseClass as EdgeCase;
    use BlueBrainMinimalExecutionState as State;

    let mut edge_cases = Vec::new();
    let terminal_reference_count =
        usize::from(report.references.execution_result_reference.is_some())
            + usize::from(report.references.failure_result_reference.is_some())
            + usize::from(report.references.cancellation_result_reference.is_some())
            + usize::from(report.references.blocked_or_unavailable_reference.is_some())
            + usize::from(
                report
                    .references
                    .non_canonical_internal_only_reference_path
                    .is_some(),
            );

    let is_terminal = matches!(
        report.state,
        State::ExecutionCompleted
            | State::ExecutionFailed
            | State::ExecutionCancelled
            | State::ExecutionBlocked
            | State::ExecutionUnavailable
            | State::ExecutionUnsupported
            | State::NonCanonicalInternalOnlyPath
    );

    let has_conflicting_terminal_reference = match report.state {
        State::ExecutionCompleted => {
            report.references.failure_result_reference.is_some()
                || report.references.cancellation_result_reference.is_some()
                || report.references.blocked_or_unavailable_reference.is_some()
                || report
                    .references
                    .non_canonical_internal_only_reference_path
                    .is_some()
        }
        State::ExecutionFailed => {
            report.references.execution_result_reference.is_some()
                || report.references.cancellation_result_reference.is_some()
                || report.references.blocked_or_unavailable_reference.is_some()
        }
        State::ExecutionCancelled => {
            report.references.execution_result_reference.is_some()
                || report.references.failure_result_reference.is_some()
                || report.references.blocked_or_unavailable_reference.is_some()
        }
        State::ExecutionBlocked | State::ExecutionUnavailable | State::ExecutionUnsupported => {
            report.references.execution_result_reference.is_some()
                || report.references.failure_result_reference.is_some()
                || report.references.cancellation_result_reference.is_some()
        }
        State::NonCanonicalInternalOnlyPath => {
            report.references.execution_result_reference.is_some()
                || report.references.failure_result_reference.is_some()
                || report.references.cancellation_result_reference.is_some()
                || report.references.blocked_or_unavailable_reference.is_some()
        }
        State::ExecutionEligibleButNotExecuted
        | State::ExecutionRequested
        | State::ExecutionStarted => terminal_reference_count > 0,
    };

    if has_conflicting_terminal_reference {
        edge_cases.push(EdgeCase::ConflictingTerminalStateAttempt);
    }

    if terminal_reference_count > 1 {
        edge_cases.push(EdgeCase::DuplicateTerminalizationAttempt);
    }

    if report.state == State::ExecutionBlocked && !report.execution_started {
        edge_cases.push(EdgeCase::BlockedBeforeStartEdgeCase);
    }
    if report.state == State::ExecutionCancelled && report.execution_started {
        edge_cases.push(EdgeCase::CancelledAfterStartEdgeCase);
    }
    if report.state == State::ExecutionFailed && report.execution_started {
        edge_cases.push(EdgeCase::FailureAfterStartEdgeCase);
    }

    let partial_execution_path = match report.state {
        State::ExecutionRequested => report.execution_started || report.execution_completed,
        State::ExecutionStarted => !report.execution_requested || report.execution_completed,
        State::ExecutionCompleted => !report.execution_started || !report.execution_completed,
        State::ExecutionFailed => !report.execution_started,
        State::ExecutionCancelled => {
            report.execution_started || report.execution_completed || report.execution_failed
        }
        State::ExecutionBlocked
        | State::ExecutionUnavailable
        | State::ExecutionUnsupported
        | State::ExecutionEligibleButNotExecuted
        | State::NonCanonicalInternalOnlyPath => {
            report.execution_started || report.execution_completed || report.execution_failed
        }
    };
    if partial_execution_path {
        edge_cases.push(EdgeCase::PartialExecutionPath);
    }

    let incomplete_result_path = match report.state {
        State::ExecutionCompleted => {
            report.executed_action_result.is_none()
                || report.references.execution_result_reference.is_none()
        }
        State::ExecutionFailed => report.references.failure_result_reference.is_none(),
        State::ExecutionCancelled => report.references.cancellation_result_reference.is_none(),
        State::ExecutionBlocked | State::ExecutionUnavailable | State::ExecutionUnsupported => {
            report.references.blocked_or_unavailable_reference.is_none()
        }
        State::ExecutionEligibleButNotExecuted => report.references.placeholder_reference.is_none(),
        State::ExecutionRequested | State::ExecutionStarted => {
            report.executed_action_result.is_some()
                || report.references.execution_result_reference.is_some()
                || report.references.failure_result_reference.is_some()
                || report.references.cancellation_result_reference.is_some()
        }
        State::NonCanonicalInternalOnlyPath => report
            .references
            .non_canonical_internal_only_reference_path
            .is_none(),
    };
    if incomplete_result_path {
        edge_cases.push(EdgeCase::IncompleteResultPath);
    }

    if matches!(report.state, State::NonCanonicalInternalOnlyPath)
        || report
            .references
            .non_canonical_internal_only_reference_path
            .is_some()
    {
        edge_cases.push(EdgeCase::NonCanonicalInternalOnlyEdgePath);
    }

    if is_terminal && report.state != State::ExecutionCancelled && !report.execution_requested {
        edge_cases.push(EdgeCase::PartialExecutionPath);
    }

    edge_cases.sort_unstable();
    edge_cases.dedup();
    edge_cases
}

pub fn blue_brain_execution_feedback_backbind(
    report: &BlueBrainMinimalExecutionReport,
) -> BlueBrainExecutionFeedbackBackbind {
    let (runtime_class, reason, proposal_feedback_class, proposal_consumed_by_execution) =
        match report.state {
            BlueBrainMinimalExecutionState::ExecutionCompleted => {
                let class = if report
                    .notes
                    .iter()
                    .any(|n| n.contains("caveat") || *n == "execution-caveated")
                {
                    BlueBrainExecutionFeedbackClass::ExecutionCaveatedFeedback
                } else {
                    BlueBrainExecutionFeedbackClass::ExecutionCompletedFeedback
                };
                let reason = if class == BlueBrainExecutionFeedbackClass::ExecutionCaveatedFeedback
                {
                    Some(BlueBrainExecutionFailureReasonClass::CaveatedCompletion)
                } else {
                    None
                };
                (
                    class,
                    reason,
                    BlueBrainProposalExecutionFeedbackClass::ProposalExecutionCompleted,
                    true,
                )
            }
            BlueBrainMinimalExecutionState::ExecutionFailed => (
                BlueBrainExecutionFeedbackClass::ExecutionFailedFeedback,
                Some(BlueBrainExecutionFailureReasonClass::ExecutionPathError),
                BlueBrainProposalExecutionFeedbackClass::ProposalExecutionFailed,
                true,
            ),
            BlueBrainMinimalExecutionState::ExecutionUnsupported => (
                BlueBrainExecutionFeedbackClass::ExecutionUnsupportedFeedback,
                Some(BlueBrainExecutionFailureReasonClass::UnsupportedAction),
                BlueBrainProposalExecutionFeedbackClass::ProposalExecutionUnsupported,
                false,
            ),
            BlueBrainMinimalExecutionState::ExecutionCancelled => (
                BlueBrainExecutionFeedbackClass::ExecutionCancelledFeedback,
                Some(BlueBrainExecutionFailureReasonClass::CancelledBeforeCompletion),
                BlueBrainProposalExecutionFeedbackClass::ProposalExecutionCancelled,
                false,
            ),
            BlueBrainMinimalExecutionState::ExecutionBlocked => (
                BlueBrainExecutionFeedbackClass::ExecutionBlockedFeedback,
                Some(BlueBrainExecutionFailureReasonClass::SafetyOrBoundaryBlocked),
                BlueBrainProposalExecutionFeedbackClass::ProposalExecutionBlocked,
                false,
            ),
            BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted => (
                BlueBrainExecutionFeedbackClass::EligibilityPlaceholderOnlyFeedback,
                Some(BlueBrainExecutionFailureReasonClass::NotRequestedPlaceholderOnly),
                BlueBrainProposalExecutionFeedbackClass::ProposalNotConsumedByExecution,
                false,
            ),
            BlueBrainMinimalExecutionState::ExecutionUnavailable => (
                BlueBrainExecutionFeedbackClass::ExecutionUnavailableFeedback,
                Some(BlueBrainExecutionFailureReasonClass::ExecutionPathUnavailable),
                BlueBrainProposalExecutionFeedbackClass::ProposalExecutionUnavailable,
                false,
            ),
            BlueBrainMinimalExecutionState::ExecutionRequested
            | BlueBrainMinimalExecutionState::ExecutionStarted => (
                BlueBrainExecutionFeedbackClass::ExecutionBlockedFeedback,
                Some(BlueBrainExecutionFailureReasonClass::SafetyOrBoundaryBlocked),
                BlueBrainProposalExecutionFeedbackClass::ProposalConsumedByExecution,
                false,
            ),
            BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath => (
                BlueBrainExecutionFeedbackClass::NonCanonicalInternalOnlyExecutionFeedback,
                Some(BlueBrainExecutionFailureReasonClass::NonCanonicalInternalOnly),
                BlueBrainProposalExecutionFeedbackClass::ProposalNotConsumedByExecution,
                false,
            ),
        };

    let sees_actual_execution_result =
        report.result_boundary == BlueBrainMinimalExecutionResultBoundary::ActualExecutionResult;
    let sees_placeholder_only =
        report.result_boundary == BlueBrainMinimalExecutionResultBoundary::PlaceholderOnly;

    BlueBrainExecutionFeedbackBackbind {
        runtime: BlueBrainExecutionRuntimeFeedback {
            class: runtime_class,
            reason,
            sees_actual_execution_result,
            sees_placeholder_only,
            retry_disposition: report.retry_disposition,
            canonical_result_reference: report
                .references
                .execution_result_reference
                .as_ref()
                .map(|reference| reference.path.clone()),
        },
        selection: BlueBrainExecutionSelectionFeedback {
            proposal_feedback_class,
            proposal_consumed_by_execution,
            automatic_next_proposal_generation: false,
            canonical_request_reference: report.references.execution_request_reference.path.clone(),
            canonical_eligibility_reference: report.references.eligibility_reference.path.clone(),
        },
        memory: BlueBrainExecutionMemoryFeedback {
            may_attach_context_reference: true,
            may_attach_diagnostic_reference: true,
            automatic_memory_commit_performed: false,
            canonical_memory_basis_reference: report
                .references
                .execution_result_reference
                .as_ref()
                .or(report.references.failure_result_reference.as_ref())
                .or(report.references.cancellation_result_reference.as_ref())
                .or(report.references.blocked_or_unavailable_reference.as_ref())
                .map(|reference| reference.path.clone()),
        },
    }
}

pub fn blue_brain_minimal_capability_scope(
    request: &BlueBrainMinimalExecutionRequest,
) -> BlueBrainMinimalCapabilityScopeClass {
    if request.internal_only_path {
        return BlueBrainMinimalCapabilityScopeClass::NonCanonicalInternalOnlyActionPath;
    }

    if request.action == BlueBrainMinimalExecutionAction::UnsupportedAction {
        return BlueBrainMinimalCapabilityScopeClass::UnsupportedAction;
    }

    if request.safety_precheck == BlueBrainSafetyPrecheckClass::Unavailable {
        return BlueBrainMinimalCapabilityScopeClass::UnavailableAction;
    }

    if request.handoff_class != BlueBrainFutureActionHandoffClass::FutureActionReady
        || request.eligibility_class
            != BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff
        || matches!(
            request.safety_precheck,
            BlueBrainSafetyPrecheckClass::Failed
                | BlueBrainSafetyPrecheckClass::Blocked
                | BlueBrainSafetyPrecheckClass::Insufficient
                | BlueBrainSafetyPrecheckClass::NotApplicable
        )
    {
        return BlueBrainMinimalCapabilityScopeClass::BlockedAction;
    }

    BlueBrainMinimalCapabilityScopeClass::AllowedCanonicalAction
}

pub fn execute_blue_brain_minimal_action(
    request: &BlueBrainMinimalExecutionRequest,
) -> BlueBrainMinimalExecutionReport {
    match blue_brain_minimal_capability_scope(request) {
        BlueBrainMinimalCapabilityScopeClass::NonCanonicalInternalOnlyActionPath => {
            let mut report = BlueBrainMinimalExecutionReport {
                trace_core: BlueBrainMinimalExecutionTraceCore {
                    canonical_action: request.action,
                    handoff_id: request.handoff_id.clone(),
                    eligibility_class: request.eligibility_class,
                    safety_precheck: request.safety_precheck,
                    state: BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath,
                    outcome_class: BlueBrainExecutionOutcomeClass::NonCanonicalInternalOnlyPath,
                    retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                    failure_path_class:
                        BlueBrainExecutionFailurePathClass::NonCanonicalInternalOnlyFailurePath,
                    result_boundary:
                        BlueBrainMinimalExecutionResultBoundary::UnavailableExecutionPath,
                },
                references: empty_reference_map(request),
                state: BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath,
                outcome_class: BlueBrainExecutionOutcomeClass::NonCanonicalInternalOnlyPath,
                retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                failure_path_class:
                    BlueBrainExecutionFailurePathClass::NonCanonicalInternalOnlyFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::UnavailableExecutionPath,
                execution_requested: request.execution_requested,
                execution_started: false,
                execution_completed: false,
                execution_failed: false,
                executed_action_result: None,
                error_code: Some("non_canonical_internal_only_execution_path"),
                notes: vec![
                    "canonical=false",
                    "non-canonical/internal-only execution path is never executable",
                ],
            };
            report.references = build_reference_map(request, &report);
            return report;
        }
        BlueBrainMinimalCapabilityScopeClass::UnsupportedAction => {
            let mut report = BlueBrainMinimalExecutionReport {
                trace_core: BlueBrainMinimalExecutionTraceCore {
                    canonical_action: request.action,
                    handoff_id: request.handoff_id.clone(),
                    eligibility_class: request.eligibility_class,
                    safety_precheck: request.safety_precheck,
                    state: BlueBrainMinimalExecutionState::ExecutionUnsupported,
                    outcome_class: BlueBrainExecutionOutcomeClass::ExecutionUnsupported,
                    retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                    failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                    result_boundary: BlueBrainMinimalExecutionResultBoundary::UnsupportedNoResult,
                },
                references: empty_reference_map(request),
                state: BlueBrainMinimalExecutionState::ExecutionUnsupported,
                outcome_class: BlueBrainExecutionOutcomeClass::ExecutionUnsupported,
                retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::UnsupportedNoResult,
                execution_requested: request.execution_requested,
                execution_started: false,
                execution_completed: false,
                execution_failed: false,
                executed_action_result: None,
                error_code: Some("execution_action_unsupported"),
                notes: vec![
                    "requested action is outside canonical minimal execution surface",
                    "unsupported action is not promoted to blocked/unavailable",
                    "no_action_executed",
                    "no_tool_result",
                ],
            };
            report.references = build_reference_map(request, &report);
            return report;
        }
        BlueBrainMinimalCapabilityScopeClass::UnavailableAction => {
            let mut report = BlueBrainMinimalExecutionReport {
                trace_core: BlueBrainMinimalExecutionTraceCore {
                    canonical_action: request.action,
                    handoff_id: request.handoff_id.clone(),
                    eligibility_class: request.eligibility_class,
                    safety_precheck: request.safety_precheck,
                    state: BlueBrainMinimalExecutionState::ExecutionUnavailable,
                    outcome_class: BlueBrainExecutionOutcomeClass::ExecutionUnavailable,
                    retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                    failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                    result_boundary:
                        BlueBrainMinimalExecutionResultBoundary::UnavailableExecutionPath,
                },
                references: empty_reference_map(request),
                state: BlueBrainMinimalExecutionState::ExecutionUnavailable,
                outcome_class: BlueBrainExecutionOutcomeClass::ExecutionUnavailable,
                retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::UnavailableExecutionPath,
                execution_requested: request.execution_requested,
                execution_started: false,
                execution_completed: false,
                execution_failed: false,
                executed_action_result: None,
                error_code: Some("safety_precheck_unavailable"),
                notes: vec![
                    "execution subsystem unavailable through safety path",
                    "no_action_executed",
                    "no_tool_result",
                ],
            };
            report.references = build_reference_map(request, &report);
            return report;
        }
        BlueBrainMinimalCapabilityScopeClass::BlockedAction
        | BlueBrainMinimalCapabilityScopeClass::AllowedCanonicalToolCall
        | BlueBrainMinimalCapabilityScopeClass::AllowedCanonicalAction => {}
    }

    if request.cancelled {
        let mut report = BlueBrainMinimalExecutionReport {
            trace_core: BlueBrainMinimalExecutionTraceCore {
                canonical_action: request.action,
                handoff_id: request.handoff_id.clone(),
                eligibility_class: request.eligibility_class,
                safety_precheck: request.safety_precheck,
                state: BlueBrainMinimalExecutionState::ExecutionCancelled,
                outcome_class: BlueBrainExecutionOutcomeClass::ExecutionCancelled,
                retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::CancelledExecutionResult,
            },
            references: empty_reference_map(request),
            state: BlueBrainMinimalExecutionState::ExecutionCancelled,
            outcome_class: BlueBrainExecutionOutcomeClass::ExecutionCancelled,
            retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
            failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
            result_boundary: BlueBrainMinimalExecutionResultBoundary::CancelledExecutionResult,
            execution_requested: request.execution_requested,
            execution_started: false,
            execution_completed: false,
            execution_failed: false,
            executed_action_result: None,
            error_code: None,
            notes: vec![
                "execution-cancelled before start",
                "no_action_executed",
                "no_tool_result",
            ],
        };
        report.references = build_reference_map(request, &report);
        return report;
    }

    if request.handoff_class != BlueBrainFutureActionHandoffClass::FutureActionReady
        || request.eligibility_class
            != BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff
    {
        let mut report = BlueBrainMinimalExecutionReport {
            trace_core: BlueBrainMinimalExecutionTraceCore {
                canonical_action: request.action,
                handoff_id: request.handoff_id.clone(),
                eligibility_class: request.eligibility_class,
                safety_precheck: request.safety_precheck,
                state: BlueBrainMinimalExecutionState::ExecutionBlocked,
                outcome_class: BlueBrainExecutionOutcomeClass::ExecutionBlocked,
                retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::BlockedNoResult,
            },
            references: empty_reference_map(request),
            state: BlueBrainMinimalExecutionState::ExecutionBlocked,
            outcome_class: BlueBrainExecutionOutcomeClass::ExecutionBlocked,
            retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
            failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
            result_boundary: BlueBrainMinimalExecutionResultBoundary::BlockedNoResult,
            execution_requested: request.execution_requested,
            execution_started: false,
            execution_completed: false,
            execution_failed: false,
            executed_action_result: None,
            error_code: Some("execution_not_canonically_eligible"),
            notes: vec![
                "only execution-eligible handoffs can enter minimal execution path",
                "no_action_executed",
                "no_tool_result",
            ],
        };
        report.references = build_reference_map(request, &report);
        return report;
    }

    match request.safety_precheck {
        BlueBrainSafetyPrecheckClass::Failed | BlueBrainSafetyPrecheckClass::Blocked => {
            let mut report = BlueBrainMinimalExecutionReport {
                trace_core: BlueBrainMinimalExecutionTraceCore {
                    canonical_action: request.action,
                    handoff_id: request.handoff_id.clone(),
                    eligibility_class: request.eligibility_class,
                    safety_precheck: request.safety_precheck,
                    state: BlueBrainMinimalExecutionState::ExecutionBlocked,
                    outcome_class: BlueBrainExecutionOutcomeClass::ExecutionBlocked,
                    retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                    failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                    result_boundary: BlueBrainMinimalExecutionResultBoundary::BlockedNoResult,
                },
                references: empty_reference_map(request),
                state: BlueBrainMinimalExecutionState::ExecutionBlocked,
                outcome_class: BlueBrainExecutionOutcomeClass::ExecutionBlocked,
                retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::BlockedNoResult,
                execution_requested: request.execution_requested,
                execution_started: false,
                execution_completed: false,
                execution_failed: false,
                executed_action_result: None,
                error_code: Some("safety_precheck_blocked_or_failed"),
                notes: vec![
                    "safety precheck blocked/failed",
                    "no_action_executed",
                    "no_tool_result",
                ],
            };
            report.references = build_reference_map(request, &report);
            return report;
        }
        BlueBrainSafetyPrecheckClass::Unavailable => unreachable!(
            "unavailable precheck is classified by blue_brain_minimal_capability_scope"
        ),
        BlueBrainSafetyPrecheckClass::Insufficient
        | BlueBrainSafetyPrecheckClass::NotApplicable => {
            let mut report = BlueBrainMinimalExecutionReport {
                trace_core: BlueBrainMinimalExecutionTraceCore {
                    canonical_action: request.action,
                    handoff_id: request.handoff_id.clone(),
                    eligibility_class: request.eligibility_class,
                    safety_precheck: request.safety_precheck,
                    state: BlueBrainMinimalExecutionState::ExecutionBlocked,
                    outcome_class: BlueBrainExecutionOutcomeClass::ExecutionBlocked,
                    retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                    failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                    result_boundary: BlueBrainMinimalExecutionResultBoundary::BlockedNoResult,
                },
                references: empty_reference_map(request),
                state: BlueBrainMinimalExecutionState::ExecutionBlocked,
                outcome_class: BlueBrainExecutionOutcomeClass::ExecutionBlocked,
                retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::BlockedNoResult,
                execution_requested: request.execution_requested,
                execution_started: false,
                execution_completed: false,
                execution_failed: false,
                executed_action_result: None,
                error_code: Some("safety_precheck_not_executable"),
                notes: vec![
                    "safety precheck insufficient/not-applicable for execution",
                    "no_action_executed",
                    "no_tool_result",
                ],
            };
            report.references = build_reference_map(request, &report);
            return report;
        }
        BlueBrainSafetyPrecheckClass::Passed | BlueBrainSafetyPrecheckClass::Caveated => {}
    }

    if !request.execution_requested {
        let mut report = BlueBrainMinimalExecutionReport {
            trace_core: BlueBrainMinimalExecutionTraceCore {
                canonical_action: request.action,
                handoff_id: request.handoff_id.clone(),
                eligibility_class: request.eligibility_class,
                safety_precheck: request.safety_precheck,
                state: BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted,
                outcome_class: BlueBrainExecutionOutcomeClass::ExecutionPlaceholderOnly,
                retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
                failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::PlaceholderOnly,
            },
            references: empty_reference_map(request),
            state: BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted,
            outcome_class: BlueBrainExecutionOutcomeClass::ExecutionPlaceholderOnly,
            retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
            failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
            result_boundary: BlueBrainMinimalExecutionResultBoundary::PlaceholderOnly,
            execution_requested: false,
            execution_started: false,
            execution_completed: false,
            execution_failed: false,
            executed_action_result: None,
            error_code: None,
            notes: vec![
                "execution-eligible but not executed",
                "placeholder-only result boundary",
            ],
        };
        report.references = build_reference_map(request, &report);
        return report;
    }

    if request.force_execution_failure {
        let retry_disposition = if request.force_nonretryable_failure {
            BlueBrainExecutionRetryDisposition::NonRetryableFailure
        } else {
            BlueBrainExecutionRetryDisposition::RetryableFailure
        };
        let mut report = BlueBrainMinimalExecutionReport {
            trace_core: BlueBrainMinimalExecutionTraceCore {
                canonical_action: request.action,
                handoff_id: request.handoff_id.clone(),
                eligibility_class: request.eligibility_class,
                safety_precheck: request.safety_precheck,
                state: BlueBrainMinimalExecutionState::ExecutionFailed,
                outcome_class: BlueBrainExecutionOutcomeClass::ExecutionFailed,
                retry_disposition,
                failure_path_class: BlueBrainExecutionFailurePathClass::CanonicalFailurePath,
                result_boundary: BlueBrainMinimalExecutionResultBoundary::FailedExecutionResult,
            },
            references: empty_reference_map(request),
            state: BlueBrainMinimalExecutionState::ExecutionFailed,
            outcome_class: BlueBrainExecutionOutcomeClass::ExecutionFailed,
            retry_disposition,
            failure_path_class: BlueBrainExecutionFailurePathClass::CanonicalFailurePath,
            result_boundary: BlueBrainMinimalExecutionResultBoundary::FailedExecutionResult,
            execution_requested: true,
            execution_started: true,
            execution_completed: false,
            execution_failed: true,
            executed_action_result: None,
            error_code: Some("minimal_execution_failed"),
            notes: vec![
                "execution-started",
                "execution-failed",
                if request.force_nonretryable_failure {
                    "failure-nonretryable"
                } else {
                    "failure-retryable"
                },
                "no_memory_commit",
            ],
        };
        report.references = build_reference_map(request, &report);
        return report;
    }

    let action_token = match request.action {
        BlueBrainMinimalExecutionAction::EmitCanonicalSignal => "emit_canonical_signal",
        BlueBrainMinimalExecutionAction::UnsupportedAction => {
            unreachable!("unsupported action is classified by blue_brain_minimal_capability_scope")
        }
    };
    let result = format!("executed:{action_token}:{}", request.handoff_id.trim());

    let mut report = BlueBrainMinimalExecutionReport {
        trace_core: BlueBrainMinimalExecutionTraceCore {
            canonical_action: request.action,
            handoff_id: request.handoff_id.clone(),
            eligibility_class: request.eligibility_class,
            safety_precheck: request.safety_precheck,
            state: BlueBrainMinimalExecutionState::ExecutionCompleted,
            outcome_class: BlueBrainExecutionOutcomeClass::ExecutionCompleted,
            retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
            failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
            result_boundary: BlueBrainMinimalExecutionResultBoundary::ActualExecutionResult,
        },
        references: empty_reference_map(request),
        state: BlueBrainMinimalExecutionState::ExecutionCompleted,
        outcome_class: BlueBrainExecutionOutcomeClass::ExecutionCompleted,
        retry_disposition: BlueBrainExecutionRetryDisposition::RetryNotApplicable,
        failure_path_class: BlueBrainExecutionFailurePathClass::NotAFailurePath,
        result_boundary: BlueBrainMinimalExecutionResultBoundary::ActualExecutionResult,
        execution_requested: true,
        execution_started: true,
        execution_completed: true,
        execution_failed: false,
        executed_action_result: Some(result),
        error_code: None,
        notes: vec![
            "execution-started",
            "execution-completed",
            "actual_result_separate_from_placeholder",
            "no_memory_commit",
            "no_compute_core_mutation",
        ],
    };
    report.references = build_reference_map(request, &report);
    report
}

fn empty_reference_map(
    request: &BlueBrainMinimalExecutionRequest,
) -> BlueBrainExecutionReferenceMap {
    let action = minimal_action_token(request.action);
    let request_ref = make_reference(
        BlueBrainExecutionReferenceClass::ExecutionRequestReference,
        format!(
            "bb14:minimal_execution:{}:{}:request:pending",
            request.handoff_id.trim(),
            action
        ),
        false,
        !request.internal_only_path,
    );
    let eligibility = make_reference(
        BlueBrainExecutionReferenceClass::EligibilityReference,
        format!(
            "bb14:minimal_execution:{}:{}:eligibility:pending",
            request.handoff_id.trim(),
            action
        ),
        false,
        !request.internal_only_path,
    );
    BlueBrainExecutionReferenceMap {
        execution_request_reference: request_ref,
        execution_result_reference: None,
        failure_result_reference: None,
        cancellation_result_reference: None,
        blocked_or_unavailable_reference: None,
        placeholder_reference: None,
        eligibility_reference: eligibility,
        non_canonical_internal_only_reference_path: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        blue_brain_execution_edge_case_map, blue_brain_execution_feedback_backbind,
        blue_brain_execution_production_hardening_path, blue_brain_execution_result_integrity,
        blue_brain_minimal_capability_scope, execute_blue_brain_minimal_action,
        BlueBrainExecutionEdgeCaseClass, BlueBrainExecutionFailurePathClass,
        BlueBrainExecutionFeedbackClass, BlueBrainExecutionProductionHardeningPathClass,
        BlueBrainExecutionReferenceClass, BlueBrainExecutionResultIntegrityClass,
        BlueBrainExecutionRetryDisposition, BlueBrainExecutionTransitionClass,
        BlueBrainMinimalCapabilityScopeClass, BlueBrainMinimalExecutionAction,
        BlueBrainMinimalExecutionRequest, BlueBrainMinimalExecutionResultBoundary,
        BlueBrainMinimalExecutionState,
    };
    use crate::reference_map::{
        BlueBrainActionExecutionEligibilityClass, BlueBrainFutureActionHandoffClass,
        BlueBrainSafetyPrecheckClass,
    };

    fn base_request() -> BlueBrainMinimalExecutionRequest {
        BlueBrainMinimalExecutionRequest {
            handoff_id: "handoff-42".to_string(),
            handoff_class: BlueBrainFutureActionHandoffClass::FutureActionReady,
            eligibility_class: BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff,
            safety_precheck: BlueBrainSafetyPrecheckClass::Passed,
            action: BlueBrainMinimalExecutionAction::EmitCanonicalSignal,
            execution_requested: false,
            cancelled: false,
            internal_only_path: false,
            force_execution_failure: false,
            force_nonretryable_failure: false,
        }
    }

    #[test]
    fn eligible_without_request_stays_placeholder_only() {
        let report = execute_blue_brain_minimal_action(&base_request());
        assert_eq!(
            report.state,
            BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted
        );
        assert_eq!(
            report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::PlaceholderOnly
        );
        assert!(!report.execution_started);
        assert!(report.executed_action_result.is_none());
    }

    #[test]
    fn canonical_request_with_passed_precheck_executes() {
        let mut request = base_request();
        request.execution_requested = true;
        let report = execute_blue_brain_minimal_action(&request);
        assert_eq!(
            report.state,
            BlueBrainMinimalExecutionState::ExecutionCompleted
        );
        assert_eq!(
            report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::ActualExecutionResult
        );
        assert!(report.execution_started);
        assert!(report.execution_completed);
        assert_eq!(
            report.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
        assert_eq!(
            report.executed_action_result.as_deref(),
            Some("executed:emit_canonical_signal:handoff-42")
        );
    }

    #[test]
    fn failed_or_blocked_precheck_prevents_execution() {
        let mut failed = base_request();
        failed.execution_requested = true;
        failed.safety_precheck = BlueBrainSafetyPrecheckClass::Failed;
        let failed_report = execute_blue_brain_minimal_action(&failed);
        assert_eq!(
            failed_report.state,
            BlueBrainMinimalExecutionState::ExecutionBlocked
        );
        assert_eq!(
            failed_report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::BlockedNoResult
        );
        assert!(failed_report.executed_action_result.is_none());

        let mut unavailable = base_request();
        unavailable.execution_requested = true;
        unavailable.safety_precheck = BlueBrainSafetyPrecheckClass::Unavailable;
        let unavailable_report = execute_blue_brain_minimal_action(&unavailable);
        assert_eq!(
            unavailable_report.state,
            BlueBrainMinimalExecutionState::ExecutionUnavailable
        );
        assert_eq!(
            unavailable_report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::UnavailableExecutionPath
        );
        assert!(unavailable_report.executed_action_result.is_none());
    }

    #[test]
    fn unsupported_vs_blocked_vs_unavailable_vs_non_canonical_stay_distinct() {
        let mut unsupported = base_request();
        unsupported.execution_requested = true;
        unsupported.action = BlueBrainMinimalExecutionAction::UnsupportedAction;
        assert_eq!(
            blue_brain_minimal_capability_scope(&unsupported),
            BlueBrainMinimalCapabilityScopeClass::UnsupportedAction
        );
        let unsupported_report = execute_blue_brain_minimal_action(&unsupported);
        assert_eq!(
            unsupported_report.state,
            BlueBrainMinimalExecutionState::ExecutionUnsupported
        );
        assert_eq!(
            unsupported_report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::UnsupportedNoResult
        );

        let mut blocked = base_request();
        blocked.execution_requested = true;
        blocked.handoff_class = BlueBrainFutureActionHandoffClass::HandoffBlocked;
        assert_eq!(
            blue_brain_minimal_capability_scope(&blocked),
            BlueBrainMinimalCapabilityScopeClass::BlockedAction
        );

        let mut unavailable = base_request();
        unavailable.execution_requested = true;
        unavailable.safety_precheck = BlueBrainSafetyPrecheckClass::Unavailable;
        assert_eq!(
            blue_brain_minimal_capability_scope(&unavailable),
            BlueBrainMinimalCapabilityScopeClass::UnavailableAction
        );

        let mut non_canonical = base_request();
        non_canonical.execution_requested = true;
        non_canonical.internal_only_path = true;
        assert_eq!(
            blue_brain_minimal_capability_scope(&non_canonical),
            BlueBrainMinimalCapabilityScopeClass::NonCanonicalInternalOnlyActionPath
        );
    }

    #[test]
    fn execution_failure_and_cancelled_are_distinct() {
        let mut failed = base_request();
        failed.execution_requested = true;
        failed.force_execution_failure = true;
        let failed_report = execute_blue_brain_minimal_action(&failed);
        assert_eq!(
            failed_report.state,
            BlueBrainMinimalExecutionState::ExecutionFailed
        );
        assert_eq!(
            failed_report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::FailedExecutionResult
        );
        assert!(failed_report.execution_started);
        assert_eq!(
            failed_report.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryableFailure
        );

        let mut cancelled = base_request();
        cancelled.execution_requested = true;
        cancelled.cancelled = true;
        let cancelled_report = execute_blue_brain_minimal_action(&cancelled);
        assert_eq!(
            cancelled_report.state,
            BlueBrainMinimalExecutionState::ExecutionCancelled
        );
        assert_eq!(
            cancelled_report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::CancelledExecutionResult
        );
        assert!(!cancelled_report.execution_started);
        assert_eq!(
            cancelled_report.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
    }

    #[test]
    fn non_canonical_internal_only_path_stays_unavailable() {
        let mut request = base_request();
        request.execution_requested = true;
        request.internal_only_path = true;
        let report = execute_blue_brain_minimal_action(&request);
        assert_eq!(
            report.state,
            BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath
        );
        assert_eq!(
            report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::UnavailableExecutionPath
        );
    }

    #[test]
    fn backbind_distinguishes_completed_failed_cancelled_blocked_and_unavailable() {
        let mut completed = base_request();
        completed.execution_requested = true;
        let completed_report = execute_blue_brain_minimal_action(&completed);
        let completed_feedback = blue_brain_execution_feedback_backbind(&completed_report);
        assert_eq!(
            completed_feedback.runtime.class,
            BlueBrainExecutionFeedbackClass::ExecutionCompletedFeedback
        );
        assert!(completed_feedback.selection.proposal_consumed_by_execution);

        let mut failed = base_request();
        failed.execution_requested = true;
        failed.force_execution_failure = true;
        let failed_feedback =
            blue_brain_execution_feedback_backbind(&execute_blue_brain_minimal_action(&failed));
        assert_eq!(
            failed_feedback.runtime.class,
            BlueBrainExecutionFeedbackClass::ExecutionFailedFeedback
        );

        let mut unsupported = base_request();
        unsupported.execution_requested = true;
        unsupported.action = BlueBrainMinimalExecutionAction::UnsupportedAction;
        let unsupported_feedback = blue_brain_execution_feedback_backbind(
            &execute_blue_brain_minimal_action(&unsupported),
        );
        assert_eq!(
            unsupported_feedback.runtime.class,
            BlueBrainExecutionFeedbackClass::ExecutionUnsupportedFeedback
        );

        let mut cancelled = base_request();
        cancelled.execution_requested = true;
        cancelled.cancelled = true;
        let cancelled_feedback =
            blue_brain_execution_feedback_backbind(&execute_blue_brain_minimal_action(&cancelled));
        assert_eq!(
            cancelled_feedback.runtime.class,
            BlueBrainExecutionFeedbackClass::ExecutionCancelledFeedback
        );

        let blocked_feedback = blue_brain_execution_feedback_backbind(
            &execute_blue_brain_minimal_action(&base_request()),
        );
        assert_eq!(
            blocked_feedback.runtime.class,
            BlueBrainExecutionFeedbackClass::EligibilityPlaceholderOnlyFeedback
        );

        let mut unavailable = base_request();
        unavailable.execution_requested = true;
        unavailable.safety_precheck = BlueBrainSafetyPrecheckClass::Unavailable;
        let unavailable_feedback = blue_brain_execution_feedback_backbind(
            &execute_blue_brain_minimal_action(&unavailable),
        );
        assert_eq!(
            unavailable_feedback.runtime.class,
            BlueBrainExecutionFeedbackClass::ExecutionUnavailableFeedback
        );
    }

    #[test]
    fn backbind_preserves_no_direct_followup_execution_and_no_memory_commit() {
        let mut request = base_request();
        request.execution_requested = true;
        let report = execute_blue_brain_minimal_action(&request);
        let feedback = blue_brain_execution_feedback_backbind(&report);
        assert!(!feedback.selection.automatic_next_proposal_generation);
        assert!(!feedback.memory.automatic_memory_commit_performed);
        assert!(feedback
            .selection
            .canonical_request_reference
            .contains(":request"));
        assert!(feedback
            .selection
            .canonical_eligibility_reference
            .contains(":eligibility:"));
    }

    #[test]
    fn placeholder_eligibility_is_not_execution_or_failed_result() {
        let report = execute_blue_brain_minimal_action(&base_request());
        assert_eq!(
            report.state,
            BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted
        );
        assert_eq!(
            report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::PlaceholderOnly
        );
        assert!(!report.execution_started);
        assert!(!report.execution_failed);
        assert!(report.executed_action_result.is_none());

        let feedback = blue_brain_execution_feedback_backbind(&report);
        assert!(feedback.runtime.sees_placeholder_only);
        assert!(!feedback.runtime.sees_actual_execution_result);
    }

    #[test]
    fn cancelled_is_not_failed_and_not_consumed_by_execution() {
        let mut request = base_request();
        request.execution_requested = true;
        request.cancelled = true;
        let report = execute_blue_brain_minimal_action(&request);
        assert_eq!(
            report.state,
            BlueBrainMinimalExecutionState::ExecutionCancelled
        );
        assert_eq!(
            report.result_boundary,
            BlueBrainMinimalExecutionResultBoundary::CancelledExecutionResult
        );
        assert!(!report.execution_failed);
        assert!(!report.execution_started);

        let feedback = blue_brain_execution_feedback_backbind(&report);
        assert_eq!(
            feedback.runtime.class,
            BlueBrainExecutionFeedbackClass::ExecutionCancelledFeedback
        );
        assert!(!feedback.selection.proposal_consumed_by_execution);
        assert_eq!(
            feedback.runtime.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
    }

    #[test]
    fn retry_boundary_distinguishes_retryable_and_nonretryable_failures() {
        let mut retryable = base_request();
        retryable.execution_requested = true;
        retryable.force_execution_failure = true;
        let retryable_report = execute_blue_brain_minimal_action(&retryable);
        assert_eq!(
            retryable_report.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryableFailure
        );
        assert_eq!(
            retryable_report.failure_path_class,
            BlueBrainExecutionFailurePathClass::CanonicalFailurePath
        );

        let mut nonretryable = retryable;
        nonretryable.force_nonretryable_failure = true;
        let nonretryable_report = execute_blue_brain_minimal_action(&nonretryable);
        assert_eq!(
            nonretryable_report.retry_disposition,
            BlueBrainExecutionRetryDisposition::NonRetryableFailure
        );
        assert!(nonretryable_report.notes.contains(&"failure-nonretryable"));
    }

    #[test]
    fn blocked_unavailable_and_cancelled_are_retry_not_applicable() {
        let mut blocked_request = base_request();
        blocked_request.handoff_class = BlueBrainFutureActionHandoffClass::HandoffBlocked;
        let blocked = execute_blue_brain_minimal_action(&blocked_request);
        assert_eq!(
            blocked.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
        assert_eq!(
            blocked.failure_path_class,
            BlueBrainExecutionFailurePathClass::NotAFailurePath
        );

        let mut unavailable_request = base_request();
        unavailable_request.safety_precheck = BlueBrainSafetyPrecheckClass::Unavailable;
        let unavailable = execute_blue_brain_minimal_action(&unavailable_request);
        assert_eq!(
            unavailable.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
        assert_eq!(
            unavailable.failure_path_class,
            BlueBrainExecutionFailurePathClass::NotAFailurePath
        );

        let mut cancelled_request = base_request();
        cancelled_request.execution_requested = true;
        cancelled_request.cancelled = true;
        let cancelled = execute_blue_brain_minimal_action(&cancelled_request);
        assert_eq!(
            cancelled.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
        assert_eq!(
            cancelled.failure_path_class,
            BlueBrainExecutionFailurePathClass::NotAFailurePath
        );
    }

    #[test]
    fn placeholder_unsupported_and_non_canonical_are_nonretryable_nonfailure_paths() {
        let placeholder = execute_blue_brain_minimal_action(&base_request());
        assert_eq!(
            placeholder.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
        assert_eq!(
            placeholder.failure_path_class,
            BlueBrainExecutionFailurePathClass::NotAFailurePath
        );

        let mut unsupported_request = base_request();
        unsupported_request.execution_requested = true;
        unsupported_request.action = BlueBrainMinimalExecutionAction::UnsupportedAction;
        let unsupported = execute_blue_brain_minimal_action(&unsupported_request);
        assert_eq!(
            unsupported.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
        assert_eq!(
            unsupported.failure_path_class,
            BlueBrainExecutionFailurePathClass::NotAFailurePath
        );

        let mut non_canonical_request = base_request();
        non_canonical_request.execution_requested = true;
        non_canonical_request.internal_only_path = true;
        let non_canonical = execute_blue_brain_minimal_action(&non_canonical_request);
        assert_eq!(
            non_canonical.retry_disposition,
            BlueBrainExecutionRetryDisposition::RetryNotApplicable
        );
        assert_eq!(
            non_canonical.failure_path_class,
            BlueBrainExecutionFailurePathClass::NonCanonicalInternalOnlyFailurePath
        );
    }

    #[test]
    fn integrity_map_distinguishes_terminal_classes_and_detects_mismatch() {
        let placeholder = execute_blue_brain_minimal_action(&base_request());
        let placeholder_integrity = blue_brain_execution_result_integrity(&placeholder);
        assert_eq!(
            placeholder_integrity.class,
            BlueBrainExecutionResultIntegrityClass::ResultBlockedCanonical
        );
        assert_eq!(
            placeholder_integrity.transition,
            BlueBrainExecutionTransitionClass::PreExecutionBoundary
        );

        let mut completed = base_request();
        completed.execution_requested = true;
        let completed_integrity =
            blue_brain_execution_result_integrity(&execute_blue_brain_minimal_action(&completed));
        assert_eq!(
            completed_integrity.class,
            BlueBrainExecutionResultIntegrityClass::ResultRecordedCanonical
        );
        assert_eq!(
            completed_integrity.transition,
            BlueBrainExecutionTransitionClass::TerminalCompleted
        );

        let mut cancelled = base_request();
        cancelled.execution_requested = true;
        cancelled.cancelled = true;
        let cancelled_integrity =
            blue_brain_execution_result_integrity(&execute_blue_brain_minimal_action(&cancelled));
        assert_eq!(
            cancelled_integrity.class,
            BlueBrainExecutionResultIntegrityClass::ResultCancelledCanonical
        );

        let mut unavailable = base_request();
        unavailable.execution_requested = true;
        unavailable.safety_precheck = BlueBrainSafetyPrecheckClass::Unavailable;
        let unavailable_integrity =
            blue_brain_execution_result_integrity(&execute_blue_brain_minimal_action(&unavailable));
        assert_eq!(
            unavailable_integrity.class,
            BlueBrainExecutionResultIntegrityClass::ResultUnavailableCanonical
        );

        let mut mismatch = execute_blue_brain_minimal_action(&completed);
        mismatch.execution_completed = false;
        let mismatch_integrity = blue_brain_execution_result_integrity(&mismatch);
        assert_eq!(
            mismatch_integrity.class,
            BlueBrainExecutionResultIntegrityClass::IntegrityMismatch
        );
        assert_eq!(
            mismatch_integrity.transition,
            BlueBrainExecutionTransitionClass::InvalidTransition
        );
        assert!(!mismatch_integrity.canonical);
    }

    #[test]
    fn canonical_reference_map_keeps_result_types_strictly_separated() {
        let placeholder = execute_blue_brain_minimal_action(&base_request());
        assert!(placeholder.references.execution_result_reference.is_none());
        assert!(placeholder.references.failure_result_reference.is_none());
        assert!(placeholder
            .references
            .cancellation_result_reference
            .is_none());
        assert!(placeholder
            .references
            .placeholder_reference
            .as_ref()
            .is_some_and(|reference| {
                reference.class == BlueBrainExecutionReferenceClass::PlaceholderReference
            }));

        let mut completed_request = base_request();
        completed_request.execution_requested = true;
        let completed = execute_blue_brain_minimal_action(&completed_request);
        assert!(completed
            .references
            .execution_result_reference
            .as_ref()
            .is_some_and(|reference| {
                reference.class == BlueBrainExecutionReferenceClass::ExecutionResultReference
                    && reference.terminal
            }));
        assert!(completed.references.placeholder_reference.is_none());
        assert!(completed.references.failure_result_reference.is_none());

        let mut failed_request = completed_request.clone();
        failed_request.force_execution_failure = true;
        let failed = execute_blue_brain_minimal_action(&failed_request);
        assert!(failed.references.execution_result_reference.is_none());
        assert!(failed
            .references
            .failure_result_reference
            .as_ref()
            .is_some_and(|reference| {
                reference.class == BlueBrainExecutionReferenceClass::FailureResultReference
            }));

        let mut cancelled_request = completed_request.clone();
        cancelled_request.cancelled = true;
        let cancelled = execute_blue_brain_minimal_action(&cancelled_request);
        assert!(cancelled
            .references
            .cancellation_result_reference
            .as_ref()
            .is_some_and(|reference| {
                reference.class == BlueBrainExecutionReferenceClass::CancellationResultReference
            }));
        assert!(cancelled.references.failure_result_reference.is_none());
    }

    #[test]
    fn blocked_unavailable_and_non_canonical_references_do_not_claim_completed_result() {
        let mut blocked_request = base_request();
        blocked_request.handoff_class = BlueBrainFutureActionHandoffClass::HandoffBlocked;
        let blocked = execute_blue_brain_minimal_action(&blocked_request);
        assert!(blocked.references.execution_result_reference.is_none());
        assert!(blocked
            .references
            .blocked_or_unavailable_reference
            .as_ref()
            .is_some_and(|reference| {
                reference.class == BlueBrainExecutionReferenceClass::BlockedOrUnavailableReference
            }));

        let mut unavailable_request = base_request();
        unavailable_request.safety_precheck = BlueBrainSafetyPrecheckClass::Unavailable;
        let unavailable = execute_blue_brain_minimal_action(&unavailable_request);
        assert!(unavailable.references.execution_result_reference.is_none());
        assert!(unavailable.references.failure_result_reference.is_none());
        assert!(unavailable
            .references
            .cancellation_result_reference
            .is_none());

        let mut non_canonical_request = base_request();
        non_canonical_request.internal_only_path = true;
        let non_canonical = execute_blue_brain_minimal_action(&non_canonical_request);
        assert!(non_canonical
            .references
            .non_canonical_internal_only_reference_path
            .as_ref()
            .is_some_and(|reference| {
                reference.class
                    == BlueBrainExecutionReferenceClass::NonCanonicalInternalOnlyReferencePath
                    && !reference.canonical
            }));
    }

    #[test]
    fn production_hardening_path_map_stays_narrow_and_distinct() {
        let placeholder = execute_blue_brain_minimal_action(&base_request());
        assert_eq!(
            blue_brain_execution_production_hardening_path(&placeholder),
            BlueBrainExecutionProductionHardeningPathClass::GuardSensitivePath
        );

        let mut completed_request = base_request();
        completed_request.execution_requested = true;
        let completed = execute_blue_brain_minimal_action(&completed_request);
        assert_eq!(
            blue_brain_execution_production_hardening_path(&completed),
            BlueBrainExecutionProductionHardeningPathClass::HardenedReferenceOrResultPath
        );

        let mut failed_request = completed_request.clone();
        failed_request.force_execution_failure = true;
        let failed = execute_blue_brain_minimal_action(&failed_request);
        assert_eq!(
            blue_brain_execution_production_hardening_path(&failed),
            BlueBrainExecutionProductionHardeningPathClass::HardenedFailurePath
        );

        let mut blocked_request = base_request();
        blocked_request.handoff_class = BlueBrainFutureActionHandoffClass::HandoffBlocked;
        let blocked = execute_blue_brain_minimal_action(&blocked_request);
        assert_eq!(
            blue_brain_execution_production_hardening_path(&blocked),
            BlueBrainExecutionProductionHardeningPathClass::HardenedBlockedOrUnavailablePath
        );

        let mut cancelled_request = completed_request.clone();
        cancelled_request.cancelled = true;
        let cancelled = execute_blue_brain_minimal_action(&cancelled_request);
        assert_eq!(
            blue_brain_execution_production_hardening_path(&cancelled),
            BlueBrainExecutionProductionHardeningPathClass::HardenedCancellationPath
        );
    }

    #[test]
    fn integrity_rejects_requested_or_started_state_drift() {
        let mut requested = execute_blue_brain_minimal_action(&base_request());
        requested.state = BlueBrainMinimalExecutionState::ExecutionRequested;
        requested.result_boundary = BlueBrainMinimalExecutionResultBoundary::ExecutionRequested;
        requested.execution_requested = true;
        requested.execution_started = false;
        requested.execution_completed = false;
        requested.execution_failed = false;
        assert_ne!(
            blue_brain_execution_result_integrity(&requested).class,
            BlueBrainExecutionResultIntegrityClass::IntegrityMismatch
        );

        requested.execution_started = true;
        assert_eq!(
            blue_brain_execution_result_integrity(&requested).class,
            BlueBrainExecutionResultIntegrityClass::IntegrityMismatch
        );

        let mut started = execute_blue_brain_minimal_action(&base_request());
        started.state = BlueBrainMinimalExecutionState::ExecutionStarted;
        started.result_boundary = BlueBrainMinimalExecutionResultBoundary::ExecutionRequested;
        started.execution_requested = true;
        started.execution_started = true;
        started.execution_completed = false;
        started.execution_failed = false;
        assert_ne!(
            blue_brain_execution_result_integrity(&started).class,
            BlueBrainExecutionResultIntegrityClass::IntegrityMismatch
        );

        started.execution_requested = false;
        assert_eq!(
            blue_brain_execution_result_integrity(&started).class,
            BlueBrainExecutionResultIntegrityClass::IntegrityMismatch
        );
    }

    #[test]
    fn edge_case_map_captures_duplicate_and_conflicting_terminalization() {
        let mut request = base_request();
        request.execution_requested = true;
        let mut report = execute_blue_brain_minimal_action(&request);
        report.references.failure_result_reference =
            report.references.execution_result_reference.clone();
        let edge_cases = blue_brain_execution_edge_case_map(&report);
        assert!(
            edge_cases.contains(&BlueBrainExecutionEdgeCaseClass::ConflictingTerminalStateAttempt)
        );
        assert!(
            edge_cases.contains(&BlueBrainExecutionEdgeCaseClass::DuplicateTerminalizationAttempt)
        );
    }

    #[test]
    fn integrity_rejects_cancelled_after_start_and_partial_paths() {
        let mut request = base_request();
        request.execution_requested = true;
        request.cancelled = true;
        let mut report = execute_blue_brain_minimal_action(&request);
        report.execution_started = true;
        let edge_cases = blue_brain_execution_edge_case_map(&report);
        assert!(edge_cases.contains(&BlueBrainExecutionEdgeCaseClass::CancelledAfterStartEdgeCase));
        assert!(edge_cases.contains(&BlueBrainExecutionEdgeCaseClass::PartialExecutionPath));
        assert_eq!(
            blue_brain_execution_result_integrity(&report).class,
            BlueBrainExecutionResultIntegrityClass::IntegrityMismatch
        );
    }

    #[test]
    fn edge_case_map_marks_blocked_before_start_and_failure_after_start_without_merging() {
        let mut blocked_request = base_request();
        blocked_request.handoff_class = BlueBrainFutureActionHandoffClass::HandoffBlocked;
        let blocked = execute_blue_brain_minimal_action(&blocked_request);
        let blocked_edges = blue_brain_execution_edge_case_map(&blocked);
        assert!(
            blocked_edges.contains(&BlueBrainExecutionEdgeCaseClass::BlockedBeforeStartEdgeCase)
        );
        assert!(
            !blocked_edges.contains(&BlueBrainExecutionEdgeCaseClass::FailureAfterStartEdgeCase)
        );

        let mut failed_request = base_request();
        failed_request.execution_requested = true;
        failed_request.force_execution_failure = true;
        let failed = execute_blue_brain_minimal_action(&failed_request);
        let failed_edges = blue_brain_execution_edge_case_map(&failed);
        assert!(failed_edges.contains(&BlueBrainExecutionEdgeCaseClass::FailureAfterStartEdgeCase));
        assert!(
            !failed_edges.contains(&BlueBrainExecutionEdgeCaseClass::BlockedBeforeStartEdgeCase)
        );
    }

    #[test]
    fn edge_case_map_flags_incomplete_result_path_without_second_result_language() {
        let mut request = base_request();
        request.execution_requested = true;
        let mut report = execute_blue_brain_minimal_action(&request);
        report.references.execution_result_reference = None;
        let edge_cases = blue_brain_execution_edge_case_map(&report);
        assert!(edge_cases.contains(&BlueBrainExecutionEdgeCaseClass::IncompleteResultPath));
        assert_eq!(
            blue_brain_execution_result_integrity(&report).class,
            BlueBrainExecutionResultIntegrityClass::IntegrityMismatch
        );
    }
}
