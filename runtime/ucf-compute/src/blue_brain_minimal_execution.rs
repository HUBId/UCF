use crate::reference_map::{
    BlueBrainActionExecutionEligibilityClass, BlueBrainFutureActionHandoffClass,
    BlueBrainSafetyPrecheckClass,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMinimalExecutionAction {
    EmitCanonicalSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueBrainMinimalExecutionState {
    ExecutionEligibleButNotExecuted,
    ExecutionRequested,
    ExecutionStarted,
    ExecutionCompleted,
    ExecutionFailed,
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
    BlockedNoResult,
    UnavailableExecutionPath,
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

pub fn execute_blue_brain_minimal_action(
    request: &BlueBrainMinimalExecutionRequest,
) -> BlueBrainMinimalExecutionReport {
    if request.internal_only_path {
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

    if request.cancelled {
        return BlueBrainMinimalExecutionReport {
            state: BlueBrainMinimalExecutionState::ExecutionCancelled,
            result_boundary: BlueBrainMinimalExecutionResultBoundary::BlockedNoResult,
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
        BlueBrainSafetyPrecheckClass::Unavailable => {
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
        execute_blue_brain_minimal_action, BlueBrainMinimalExecutionAction,
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
            BlueBrainMinimalExecutionResultBoundary::BlockedNoResult
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
}
