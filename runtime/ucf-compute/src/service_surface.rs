use crate::compute_service::{
    InMemoryComputeService, JobCompletionClass, JobExecutionPath, JobId, JobLifecycleEvent,
    JobLifecycleState, JobRecord, JobSubmissionMeta,
};
use crate::pipeline::{
    CanonicalFailureKind, CanonicalPipelineFailure, CanonicalPipelineRequest,
    CanonicalPipelineState, CanonicalWorkSummary,
};
use crate::ModelSlotProvenance;

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
        ComputeSubmitOutcome, ComputeSubmitRequest,
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
}
