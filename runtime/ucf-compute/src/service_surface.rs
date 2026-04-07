use crate::compute_service::{
    InMemoryComputeService, JobCompletionClass, JobExecutionPath, JobId, JobLifecycleEvent,
    JobLifecycleState, JobRecord, JobSubmissionMeta,
};
use crate::pipeline::{
    CanonicalFailureKind, CanonicalPipelineFailure, CanonicalPipelineRequest,
    CanonicalPipelineState, CanonicalWorkSummary,
};
use crate::{ModelSlot, ModelSlotProvenance, SlotRuntimeStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeExecutionMode {
    EnqueueOnly,
    ExecuteInline,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputeSubmitRequest {
    pub pipeline_request: CanonicalPipelineRequest,
    pub submitted_by: Option<String>,
    pub submitted_at_unix_ms: Option<u64>,
    pub execution_mode: ComputeExecutionMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeRequestValidationCode {
    SubmittedByEmpty,
    SubmittedByTooLong,
    SubmittedByControlChar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeInvalidRequest {
    pub code: ComputeRequestValidationCode,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeJobHandle {
    pub job_id: JobId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputeJobStatus {
    pub handle: ComputeJobHandle,
    pub lifecycle_state: JobLifecycleState,
    pub execution_path: JobExecutionPath,
    pub completion_class: Option<JobCompletionClass>,
    pub admission_failure: Option<CanonicalPipelineFailure>,
    pub execution_failure: Option<CanonicalPipelineFailure>,
    pub failure_kind: Option<CanonicalFailureKind>,
    pub pipeline_state: Option<CanonicalPipelineState>,
    pub work_summary: Option<CanonicalWorkSummary>,
    pub model_slots: Vec<ModelSlotProvenance>,
    pub submitted_at_unix_ms: u64,
    pub finished_at_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComputeSubmitOutcome {
    Invalid(ComputeInvalidRequest),
    Rejected {
        status: ComputeJobStatus,
    },
    Accepted {
        status: ComputeJobStatus,
        completion: Option<ComputeJobStatus>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeOpsState {
    HealthyReady,
    Degraded,
    PartiallyUnavailable,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeSignalState {
    Known,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeOperation {
    Snapshot,
    DrainScheduler { max_jobs: usize },
    RefreshRuntime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeOperationCode {
    Applied,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeOperationOutcome {
    pub operation: RuntimeOperation,
    pub code: RuntimeOperationCode,
    pub detail: String,
    pub completed_jobs: Vec<JobId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeQueueSnapshot {
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub max_concurrent_jobs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeJobSummary {
    pub submitted_total: usize,
    pub completed_total: usize,
    pub failed_total: usize,
    pub rejected_total: usize,
    pub timed_out_total: usize,
    pub degraded_total: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSlotSnapshot {
    pub slot: ModelSlot,
    pub status: SlotRuntimeStatus,
    pub required_for_pack: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeOpsSnapshot {
    pub state: RuntimeOpsState,
    pub state_signal: RuntimeSignalState,
    pub execution_path: JobExecutionPath,
    pub queue: ComputeQueueSnapshot,
    pub jobs: ComputeJobSummary,
    pub available_slots: Vec<RuntimeSlotSnapshot>,
    pub active_job: Option<ComputeJobHandle>,
    pub candidate_job: Option<ComputeJobHandle>,
    pub compare_job: Option<ComputeJobHandle>,
    pub shadow_job: Option<ComputeJobHandle>,
    pub has_missing_required_slot: bool,
}

pub struct CanonicalComputeEntryPoint {
    service: InMemoryComputeService,
}

impl CanonicalComputeEntryPoint {
    pub fn new(service: InMemoryComputeService) -> Self {
        Self { service }
    }

    pub fn submit(
        &mut self,
        request: ComputeSubmitRequest,
    ) -> Result<ComputeSubmitOutcome, crate::ComputeError> {
        if let Some(invalid) = validate_request(&request) {
            return Ok(ComputeSubmitOutcome::Invalid(invalid));
        }

        let submitted_at_unix_ms = request.submitted_at_unix_ms.unwrap_or_else(now_unix_ms);
        let submitted = self.service.submit(
            request.pipeline_request,
            JobSubmissionMeta {
                submitted_at_unix_ms,
                submitted_by: request.submitted_by,
            },
        );
        let submit_status = status_from_record(submitted);
        if submit_status.lifecycle_state == JobLifecycleState::Rejected {
            return Ok(ComputeSubmitOutcome::Rejected {
                status: submit_status,
            });
        }

        match request.execution_mode {
            ComputeExecutionMode::EnqueueOnly => Ok(ComputeSubmitOutcome::Accepted {
                status: submit_status,
                completion: None,
            }),
            ComputeExecutionMode::ExecuteInline => {
                let completed = self.service.run_next()?;
                let completion = completed.map(status_from_record);
                Ok(ComputeSubmitOutcome::Accepted {
                    status: submit_status,
                    completion,
                })
            }
        }
    }

    pub fn status(&self, handle: ComputeJobHandle) -> Option<ComputeJobStatus> {
        self.service.job(handle.job_id).map(status_from_record)
    }

    pub fn lifecycle(&self, handle: ComputeJobHandle) -> Vec<JobLifecycleEvent> {
        self.service
            .lifecycle_events()
            .iter()
            .filter(|event| event.job_id == handle.job_id)
            .cloned()
            .collect()
    }

    pub fn service(&self) -> &InMemoryComputeService {
        &self.service
    }

    pub fn service_mut(&mut self) -> &mut InMemoryComputeService {
        &mut self.service
    }

    pub fn operations_snapshot(&self) -> RuntimeOpsSnapshot {
        let scheduler = self.service.scheduler_snapshot();
        let mut submitted_total = 0usize;
        let mut completed_total = 0usize;
        let mut failed_total = 0usize;
        let mut rejected_total = 0usize;
        let mut timed_out_total = 0usize;
        let mut degraded_total = 0usize;
        let mut slots = Vec::new();
        let mut last_job = None;

        for record in self.service.jobs() {
            submitted_total = submitted_total.saturating_add(1);
            last_job = Some(record.job.id);
            match record.state {
                JobLifecycleState::Completed => {
                    completed_total = completed_total.saturating_add(1);
                    if record.accounting.pipeline_state == Some(CanonicalPipelineState::Degraded) {
                        degraded_total = degraded_total.saturating_add(1);
                    }
                }
                JobLifecycleState::Failed => failed_total = failed_total.saturating_add(1),
                JobLifecycleState::Rejected => rejected_total = rejected_total.saturating_add(1),
                JobLifecycleState::TimedOut => timed_out_total = timed_out_total.saturating_add(1),
                _ => {}
            }
            if !record.accounting.model_slots.is_empty() {
                slots = record
                    .accounting
                    .model_slots
                    .iter()
                    .map(|slot| RuntimeSlotSnapshot {
                        slot: slot.slot,
                        status: slot.status,
                        required_for_pack: slot.required_for_pack,
                    })
                    .collect();
            }
        }
        slots.sort_by_key(|slot| slot.slot);
        let has_missing_required_slot = slots.iter().any(|slot| {
            slot.required_for_pack
                && !matches!(
                    slot.status,
                    SlotRuntimeStatus::Used | SlotRuntimeStatus::Disabled
                )
        });
        let total_terminal_failures = failed_total
            .saturating_add(rejected_total)
            .saturating_add(timed_out_total);
        let has_failure_ratio_degraded =
            submitted_total > 0 && total_terminal_failures.saturating_mul(2) >= submitted_total;
        let state_signal = if submitted_total == 0 {
            RuntimeSignalState::Unknown
        } else {
            RuntimeSignalState::Known
        };
        let no_successful_runtime_path = submitted_total > 0
            && completed_total == 0
            && total_terminal_failures == submitted_total;
        let state =
            if (has_missing_required_slot && completed_total == 0) || no_successful_runtime_path {
                RuntimeOpsState::Unavailable
            } else if has_missing_required_slot || scheduler.queued_jobs > 0 {
                RuntimeOpsState::PartiallyUnavailable
            } else if has_failure_ratio_degraded || degraded_total > 0 {
                RuntimeOpsState::Degraded
            } else {
                RuntimeOpsState::HealthyReady
            };
        RuntimeOpsSnapshot {
            state,
            state_signal,
            execution_path: scheduler.execution_path,
            queue: ComputeQueueSnapshot {
                queued_jobs: scheduler.queued_jobs,
                running_jobs: scheduler.running_jobs,
                max_concurrent_jobs: scheduler.max_concurrent_jobs,
            },
            jobs: ComputeJobSummary {
                submitted_total,
                completed_total,
                failed_total,
                rejected_total,
                timed_out_total,
                degraded_total,
            },
            available_slots: slots,
            active_job: last_job.map(|job_id| ComputeJobHandle { job_id }),
            candidate_job: None,
            compare_job: None,
            shadow_job: None,
            has_missing_required_slot,
        }
    }

    pub fn run_operation(
        &mut self,
        operation: RuntimeOperation,
    ) -> Result<RuntimeOperationOutcome, crate::ComputeError> {
        match operation {
            RuntimeOperation::Snapshot => Ok(RuntimeOperationOutcome {
                operation,
                code: RuntimeOperationCode::Applied,
                detail: "runtime snapshot captured".to_string(),
                completed_jobs: Vec::new(),
            }),
            RuntimeOperation::DrainScheduler { max_jobs } => {
                let completed_jobs = self.service.run_scheduler_cycle(max_jobs.max(1))?;
                Ok(RuntimeOperationOutcome {
                    operation,
                    code: RuntimeOperationCode::Applied,
                    detail: format!("scheduler drained {} jobs", completed_jobs.len()),
                    completed_jobs,
                })
            }
            RuntimeOperation::RefreshRuntime => Ok(RuntimeOperationOutcome {
                operation,
                code: RuntimeOperationCode::Unsupported,
                detail: "refresh_runtime unsupported for in-memory compute service".to_string(),
                completed_jobs: Vec::new(),
            }),
        }
    }
}

fn validate_request(request: &ComputeSubmitRequest) -> Option<ComputeInvalidRequest> {
    if let Some(submitted_by) = request.submitted_by.as_ref() {
        if submitted_by.trim().is_empty() {
            return Some(ComputeInvalidRequest {
                code: ComputeRequestValidationCode::SubmittedByEmpty,
                detail: "submitted_by must not be empty".to_string(),
            });
        }
        if submitted_by.len() > 128 {
            return Some(ComputeInvalidRequest {
                code: ComputeRequestValidationCode::SubmittedByTooLong,
                detail: "submitted_by must be <= 128 chars".to_string(),
            });
        }
        if submitted_by.chars().any(char::is_control) {
            return Some(ComputeInvalidRequest {
                code: ComputeRequestValidationCode::SubmittedByControlChar,
                detail: "submitted_by contains control characters".to_string(),
            });
        }
    }
    None
}

fn status_from_record(record: &JobRecord) -> ComputeJobStatus {
    ComputeJobStatus {
        handle: ComputeJobHandle {
            job_id: record.job.id,
        },
        lifecycle_state: record.state,
        execution_path: record.execution_path,
        completion_class: if matches!(
            record.state,
            JobLifecycleState::Completed
                | JobLifecycleState::Failed
                | JobLifecycleState::Rejected
                | JobLifecycleState::TimedOut
        ) {
            Some(record.accounting.completion_class)
        } else {
            None
        },
        admission_failure: record.rejection.clone(),
        execution_failure: record.execution_failure.clone(),
        failure_kind: record.accounting.failure_kind,
        pipeline_state: record.accounting.pipeline_state,
        work_summary: record.accounting.work_summary,
        model_slots: record.accounting.model_slots.clone(),
        submitted_at_unix_ms: record.accounting.submitted_at_unix_ms,
        finished_at_unix_ms: record.accounting.finished_at_unix_ms,
    }
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::{
        CanonicalComputeEntryPoint, ComputeExecutionMode, ComputeRequestValidationCode,
        ComputeSubmitOutcome, ComputeSubmitRequest, RuntimeOperation, RuntimeOperationCode,
        RuntimeOpsState, RuntimeSignalState,
    };
    use crate::pipeline::{CanonicalFailureKind, CanonicalPipelineRequest};
    use crate::{InMemoryComputeService, JobLifecycleState};

    fn service() -> CanonicalComputeEntryPoint {
        CanonicalComputeEntryPoint::new(InMemoryComputeService::new(
            crate::pipeline::ComputePipelineBackend::stub(),
        ))
    }

    fn valid_request() -> CanonicalPipelineRequest {
        CanonicalPipelineRequest {
            input: crate::ComputeInput {
                frame_id: crate::FrameId(11),
                t: 42,
                context_digest: [1; 32],
            },
            budget: crate::ComputeBudget::default(),
        }
    }

    #[test]
    fn invalid_submit_is_rejected_before_service_submit() {
        let mut entry = service();
        let outcome = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some(" ".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("submit should not fail");
        match outcome {
            ComputeSubmitOutcome::Invalid(invalid) => {
                assert_eq!(invalid.code, ComputeRequestValidationCode::SubmittedByEmpty);
            }
            other => panic!("expected invalid, got {other:?}"),
        }
    }

    #[test]
    fn admission_rejection_is_structured() {
        let mut entry = service();
        let mut request = valid_request();
        request.input.t = 0;
        let outcome = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: request,
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("submit should not fail");
        match outcome {
            ComputeSubmitOutcome::Rejected { status } => {
                assert_eq!(status.lifecycle_state, JobLifecycleState::Rejected);
                assert_eq!(
                    status.failure_kind,
                    Some(CanonicalFailureKind::InvalidInput)
                );
                assert!(status.admission_failure.is_some());
            }
            other => panic!("expected rejected, got {other:?}"),
        }
    }

    #[test]
    fn execute_inline_reports_completion_and_status_surface() {
        let mut entry = service();
        let outcome = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("submit should not fail");
        let handle = match outcome {
            ComputeSubmitOutcome::Accepted { status, completion } => {
                assert_eq!(status.lifecycle_state, JobLifecycleState::Queued);
                let completion = completion.expect("inline should return completion");
                assert!(matches!(
                    completion.lifecycle_state,
                    JobLifecycleState::Completed
                        | JobLifecycleState::Failed
                        | JobLifecycleState::TimedOut
                ));
                completion.handle
            }
            other => panic!("expected accepted, got {other:?}"),
        };
        let status = entry.status(handle).expect("status must exist");
        assert!(matches!(
            status.lifecycle_state,
            JobLifecycleState::Completed | JobLifecycleState::Failed | JobLifecycleState::TimedOut
        ));
        assert!(!entry.lifecycle(handle).is_empty());
    }

    #[test]
    fn operations_snapshot_marks_unknown_without_job_signal() {
        let entry = service();
        let snapshot = entry.operations_snapshot();
        assert_eq!(snapshot.state, RuntimeOpsState::HealthyReady);
        assert_eq!(snapshot.state_signal, RuntimeSignalState::Unknown);
        assert_eq!(snapshot.jobs.submitted_total, 0);
    }

    #[test]
    fn operations_snapshot_distinguishes_unavailable_and_partially_unavailable() {
        let mut entry = service();
        let mut rejected = valid_request();
        rejected.input.t = 0;
        entry
            .submit(ComputeSubmitRequest {
                pipeline_request: rejected,
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::EnqueueOnly,
            })
            .expect("submit should not fail");
        let unavailable = entry.operations_snapshot();
        assert_eq!(unavailable.state_signal, RuntimeSignalState::Known);
        assert_eq!(unavailable.state, RuntimeOpsState::Unavailable);

        entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(101),
                execution_mode: ComputeExecutionMode::EnqueueOnly,
            })
            .expect("submit should not fail");
        let partially_unavailable = entry.operations_snapshot();
        assert_eq!(
            partially_unavailable.state,
            RuntimeOpsState::PartiallyUnavailable
        );
    }

    #[test]
    fn operations_actions_are_applied_or_structured_unsupported() {
        let mut entry = service();
        entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::EnqueueOnly,
            })
            .expect("submit should not fail");

        let drained = entry
            .run_operation(RuntimeOperation::DrainScheduler { max_jobs: 1 })
            .expect("drain operation");
        assert_eq!(drained.code, RuntimeOperationCode::Applied);
        assert_eq!(drained.completed_jobs.len(), 1);

        let unsupported = entry
            .run_operation(RuntimeOperation::RefreshRuntime)
            .expect("refresh operation");
        assert_eq!(unsupported.code, RuntimeOperationCode::Unsupported);
    }
}
