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
pub enum BlueBrainMinimalExecutionResultBoundary {
    PlaceholderOnly,
    ExecutionRequested,
    ActualExecutionResult,
    FailedExecutionResult,
    UnsupportedNoResult,
    BlockedNoResult,
    UnavailableExecutionPath,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainMinimalExecutionReport {
    pub state: BlueBrainMinimalExecutionState,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionSelectionFeedback {
    pub proposal_feedback_class: BlueBrainProposalExecutionFeedbackClass,
    pub proposal_consumed_by_execution: bool,
    pub automatic_next_proposal_generation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionMemoryFeedback {
    pub may_attach_context_reference: bool,
    pub may_attach_diagnostic_reference: bool,
    pub automatic_memory_commit_performed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueBrainExecutionFeedbackBackbind {
    pub runtime: BlueBrainExecutionRuntimeFeedback,
    pub selection: BlueBrainExecutionSelectionFeedback,
    pub memory: BlueBrainExecutionMemoryFeedback,
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
                BlueBrainExecutionFeedbackClass::ExecutionBlockedFeedback,
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
        },
        selection: BlueBrainExecutionSelectionFeedback {
            proposal_feedback_class,
            proposal_consumed_by_execution,
            automatic_next_proposal_generation: false,
        },
        memory: BlueBrainExecutionMemoryFeedback {
            may_attach_context_reference: true,
            may_attach_diagnostic_reference: true,
            automatic_memory_commit_performed: false,
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
            return BlueBrainMinimalExecutionReport {
                state: BlueBrainMinimalExecutionState::NonCanonicalInternalOnlyPath,
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
        }
        BlueBrainMinimalCapabilityScopeClass::UnsupportedAction => {
            return BlueBrainMinimalExecutionReport {
                state: BlueBrainMinimalExecutionState::ExecutionUnsupported,
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
        }
        BlueBrainMinimalCapabilityScopeClass::UnavailableAction => {
            return BlueBrainMinimalExecutionReport {
                state: BlueBrainMinimalExecutionState::ExecutionUnavailable,
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
        }
        BlueBrainMinimalCapabilityScopeClass::BlockedAction
        | BlueBrainMinimalCapabilityScopeClass::AllowedCanonicalToolCall
        | BlueBrainMinimalCapabilityScopeClass::AllowedCanonicalAction => {}
    }

    if request.cancelled {
        return BlueBrainMinimalExecutionReport {
            state: BlueBrainMinimalExecutionState::ExecutionCancelled,
            result_boundary: BlueBrainMinimalExecutionResultBoundary::ExecutionRequested,
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
    }

    if request.handoff_class != BlueBrainFutureActionHandoffClass::FutureActionReady
        || request.eligibility_class
            != BlueBrainActionExecutionEligibilityClass::ExecutionEligibleHandoff
    {
        return BlueBrainMinimalExecutionReport {
            state: BlueBrainMinimalExecutionState::ExecutionBlocked,
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
    }

    match request.safety_precheck {
        BlueBrainSafetyPrecheckClass::Failed | BlueBrainSafetyPrecheckClass::Blocked => {
            return BlueBrainMinimalExecutionReport {
                state: BlueBrainMinimalExecutionState::ExecutionBlocked,
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
        }
        BlueBrainSafetyPrecheckClass::Unavailable => unreachable!(
            "unavailable precheck is classified by blue_brain_minimal_capability_scope"
        ),
        BlueBrainSafetyPrecheckClass::Insufficient
        | BlueBrainSafetyPrecheckClass::NotApplicable => {
            return BlueBrainMinimalExecutionReport {
                state: BlueBrainMinimalExecutionState::ExecutionBlocked,
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
        }
        BlueBrainSafetyPrecheckClass::Passed | BlueBrainSafetyPrecheckClass::Caveated => {}
    }

    if !request.execution_requested {
        return BlueBrainMinimalExecutionReport {
            state: BlueBrainMinimalExecutionState::ExecutionEligibleButNotExecuted,
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
    }

    if request.force_execution_failure {
        return BlueBrainMinimalExecutionReport {
            state: BlueBrainMinimalExecutionState::ExecutionFailed,
            result_boundary: BlueBrainMinimalExecutionResultBoundary::FailedExecutionResult,
            execution_requested: true,
            execution_started: true,
            execution_completed: false,
            execution_failed: true,
            executed_action_result: None,
            error_code: Some("minimal_execution_failed"),
            notes: vec!["execution-started", "execution-failed", "no_memory_commit"],
        };
    }

    let action_token = match request.action {
        BlueBrainMinimalExecutionAction::EmitCanonicalSignal => "emit_canonical_signal",
        BlueBrainMinimalExecutionAction::UnsupportedAction => {
            unreachable!("unsupported action is classified by blue_brain_minimal_capability_scope")
        }
    };
    let result = format!("executed:{action_token}:{}", request.handoff_id.trim());

    BlueBrainMinimalExecutionReport {
        state: BlueBrainMinimalExecutionState::ExecutionCompleted,
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
    }
}

#[cfg(test)]
mod tests {
    use super::{
        blue_brain_execution_feedback_backbind, blue_brain_minimal_capability_scope,
        execute_blue_brain_minimal_action, BlueBrainExecutionFeedbackClass,
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
            BlueBrainMinimalExecutionResultBoundary::ExecutionRequested
        );
        assert!(!cancelled_report.execution_started);
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
            BlueBrainExecutionFeedbackClass::ExecutionBlockedFeedback
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
            BlueBrainMinimalExecutionResultBoundary::ExecutionRequested
        );
        assert!(!report.execution_failed);
        assert!(!report.execution_started);

        let feedback = blue_brain_execution_feedback_backbind(&report);
        assert_eq!(
            feedback.runtime.class,
            BlueBrainExecutionFeedbackClass::ExecutionCancelledFeedback
        );
        assert!(!feedback.selection.proposal_consumed_by_execution);
    }
}
