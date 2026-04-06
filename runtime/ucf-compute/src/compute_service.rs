use std::collections::{BTreeMap, VecDeque};

use crate::pipeline::{
    CanonicalAdmissionDecision, CanonicalFailureKind, CanonicalPipelineFailure,
    CanonicalPipelineRequest, CanonicalPipelineResult, CanonicalPipelineState,
    ComputePipelineBackend,
};
use crate::ComputeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobSubmissionMeta {
    pub submitted_at_unix_ms: u64,
    pub submitted_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputeJob {
    pub id: JobId,
    pub request: CanonicalPipelineRequest,
    pub meta: JobSubmissionMeta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobLifecycleState {
    Submitted,
    Admitted,
    Rejected,
    Queued,
    Running,
    Completed,
    Failed,
    TimedOut,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobLifecycleEvent {
    pub job_id: JobId,
    pub state: JobLifecycleState,
    pub failure_kind: Option<CanonicalFailureKind>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JobRecord {
    pub job: ComputeJob,
    pub state: JobLifecycleState,
    pub admission: CanonicalAdmissionDecision,
    pub rejection: Option<CanonicalPipelineFailure>,
    pub execution_failure: Option<CanonicalPipelineFailure>,
    pub result: Option<CanonicalPipelineResult>,
}

pub struct InMemoryComputeService {
    backend: ComputePipelineBackend,
    next_job_id: u64,
    jobs: BTreeMap<JobId, JobRecord>,
    queue: VecDeque<JobId>,
    lifecycle: Vec<JobLifecycleEvent>,
}

impl InMemoryComputeService {
    pub fn new(backend: ComputePipelineBackend) -> Self {
        Self {
            backend,
            next_job_id: 1,
            jobs: BTreeMap::new(),
            queue: VecDeque::new(),
            lifecycle: Vec::new(),
        }
    }

    pub fn submit(
        &mut self,
        request: CanonicalPipelineRequest,
        meta: JobSubmissionMeta,
    ) -> &JobRecord {
        let job_id = JobId(self.next_job_id);
        self.next_job_id = self.next_job_id.saturating_add(1);
        let job = ComputeJob {
            id: job_id,
            request: request.clone(),
            meta,
        };
        let admission = self.backend.technical_admission(&request);
        let mut record = JobRecord {
            job,
            state: JobLifecycleState::Submitted,
            admission: admission.clone(),
            rejection: None,
            execution_failure: None,
            result: None,
        };
        self.record_event(job_id, JobLifecycleState::Submitted, None, None);
        match admission.failure {
            Some(failure) => {
                record.state = JobLifecycleState::Rejected;
                record.rejection = Some(failure.clone());
                self.record_event(
                    job_id,
                    JobLifecycleState::Rejected,
                    Some(failure.kind),
                    Some(failure.detail),
                );
            }
            None => {
                record.state = JobLifecycleState::Admitted;
                self.record_event(job_id, JobLifecycleState::Admitted, None, None);
                record.state = JobLifecycleState::Queued;
                self.queue.push_back(job_id);
                self.record_event(job_id, JobLifecycleState::Queued, None, None);
            }
        }
        self.jobs.insert(job_id, record);
        self.jobs.get(&job_id).expect("inserted record must exist")
    }

    pub fn run_next(&mut self) -> Result<Option<&JobRecord>, ComputeError> {
        let Some(job_id) = self.queue.pop_front() else {
            return Ok(None);
        };
        let request = match self.jobs.get_mut(&job_id) {
            Some(record) => {
                record.state = JobLifecycleState::Running;
                record.job.request.clone()
            }
            None => return Ok(None),
        };
        self.record_event(job_id, JobLifecycleState::Running, None, None);

        let result = self.backend.compute_canonical(request)?;
        let failure = result.failure.clone();
        let state = match (&result.state, &failure) {
            (
                _,
                Some(CanonicalPipelineFailure {
                    kind: CanonicalFailureKind::Timeout,
                    ..
                }),
            ) => JobLifecycleState::TimedOut,
            (_, Some(_)) => JobLifecycleState::Failed,
            (CanonicalPipelineState::Unavailable, None) => JobLifecycleState::Failed,
            _ => JobLifecycleState::Completed,
        };
        let Some(record) = self.jobs.get_mut(&job_id) else {
            return Ok(None);
        };
        record.result = Some(result);
        record.state = state;
        if let Some(failure) = failure {
            record.execution_failure = Some(failure.clone());
            self.record_event(
                job_id,
                state,
                Some(failure.kind),
                Some(failure.detail.clone()),
            );
        } else {
            self.record_event(job_id, state, None, None);
        }
        Ok(self.jobs.get(&job_id))
    }

    pub fn job(&self, job_id: JobId) -> Option<&JobRecord> {
        self.jobs.get(&job_id)
    }

    pub fn lifecycle_events(&self) -> &[JobLifecycleEvent] {
        &self.lifecycle
    }

    fn record_event(
        &mut self,
        job_id: JobId,
        state: JobLifecycleState,
        failure_kind: Option<CanonicalFailureKind>,
        detail: Option<String>,
    ) {
        self.lifecycle.push(JobLifecycleEvent {
            job_id,
            state,
            failure_kind,
            detail,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::backend_pack::{
        ArtifactFailureCode, BackendComponentId, BackendPack, BackendPackId, BackendPackMeta,
        ModelSlotProvenance, SlotRuntimeStatus,
    };
    use crate::capabilities::{
        LlmInference, LlmRequest, LlmResponse, SaeExtractor, WorldModelPredictor,
    };
    use crate::feature_extractor::ToySaeExtractor;
    use crate::lfm::{LfmKernel, ToyLfmKernel};
    use crate::pipeline::{
        CanonicalFailureKind, CanonicalPipelineRequest, ComputePipelineBackend, FusionConfig,
        LimitsConfig,
    };
    use crate::ssm::{SsmKernel, ToySsmKernel};
    use crate::world_model::MockJepaPredictor;
    use crate::{ComputeBudget, ComputeError, ComputeInput, FrameId, ModelSlot};

    use super::{InMemoryComputeService, JobLifecycleState, JobSubmissionMeta};

    struct NullLlm;
    impl LlmInference for NullLlm {
        fn name(&self) -> &'static str {
            "null_llm"
        }

        fn infer(
            &self,
            _req: &LlmRequest,
            _budget: ComputeBudget,
        ) -> Result<LlmResponse, ComputeError> {
            Ok(LlmResponse::new(
                crate::capabilities::LlmStatus::Refused,
                String::new(),
                0,
                crate::capabilities::FinishReason::PolicyRefusal,
            ))
        }
    }

    struct TestPack {
        meta: BackendPackMeta,
        slots: Vec<ModelSlotProvenance>,
        llm: Arc<dyn LlmInference + Send + Sync>,
        world: Mutex<Box<dyn WorldModelPredictor + Send + Sync>>,
        sae: Arc<dyn SaeExtractor + Send + Sync>,
        ssm: Mutex<Box<dyn SsmKernel + Send + Sync>>,
        lfm: Mutex<Box<dyn LfmKernel + Send + Sync>>,
    }

    impl BackendPack for TestPack {
        fn meta(&self) -> &BackendPackMeta {
            &self.meta
        }

        fn model_slot_provenance(&self) -> &[ModelSlotProvenance] {
            &self.slots
        }

        fn llm(&self) -> &dyn LlmInference {
            self.llm.as_ref()
        }

        fn world(&self) -> &Mutex<Box<dyn WorldModelPredictor + Send + Sync>> {
            &self.world
        }

        fn sae(&self) -> &dyn SaeExtractor {
            self.sae.as_ref()
        }

        fn ssm(&self) -> &Mutex<Box<dyn SsmKernel + Send + Sync>> {
            &self.ssm
        }

        fn lfm(&self) -> &Mutex<Box<dyn LfmKernel + Send + Sync>> {
            &self.lfm
        }
    }

    fn pack_with(
        world_backend: BackendComponentId,
        slots: Vec<ModelSlotProvenance>,
    ) -> Arc<dyn BackendPack> {
        let mut meta = BackendPackMeta {
            schema_version: 1,
            pack_name: "test_pack",
            pack_id: BackendPackId(999),
            llm_backend: BackendComponentId::ToyV1,
            world_backend,
            sae_backend: BackendComponentId::ToyV1,
            ssm_backend: BackendComponentId::ToyV1,
            lfm_backend: BackendComponentId::ToyV1,
            fixtures_digest: [0; 32],
            model_hashes_digest: [0; 32],
            code_version: crate::CodeVersionTag::current(),
            digest: [0; 32],
        };
        meta.digest = meta.canonical_digest();
        Arc::new(TestPack {
            meta,
            slots,
            llm: Arc::new(NullLlm),
            world: Mutex::new(Box::new(MockJepaPredictor::default())),
            sae: Arc::new(ToySaeExtractor::default()),
            ssm: Mutex::new(Box::new(ToySsmKernel::default())),
            lfm: Mutex::new(Box::new(ToyLfmKernel::default())),
        })
    }

    fn service_with_pack(pack: Arc<dyn BackendPack>) -> InMemoryComputeService {
        let backend =
            ComputePipelineBackend::new(pack, FusionConfig::default(), LimitsConfig::default());
        InMemoryComputeService::new(backend)
    }

    fn valid_request() -> CanonicalPipelineRequest {
        CanonicalPipelineRequest {
            input: ComputeInput {
                frame_id: FrameId(7),
                t: 11,
                context_digest: [9; 32],
            },
            budget: ComputeBudget::default(),
        }
    }

    #[test]
    fn valid_job_is_submitted_admitted_and_queued() {
        let mut service = service_with_pack(pack_with(BackendComponentId::ToyV1, Vec::new()));
        let record = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 1,
                submitted_by: Some("test".to_string()),
            },
        );
        assert_eq!(record.state, JobLifecycleState::Queued);
        assert!(record.rejection.is_none());
        assert_eq!(
            service
                .lifecycle_events()
                .iter()
                .map(|event| event.state)
                .collect::<Vec<_>>(),
            vec![
                JobLifecycleState::Submitted,
                JobLifecycleState::Admitted,
                JobLifecycleState::Queued
            ]
        );
    }

    #[test]
    fn invalid_request_is_rejected_with_structured_failure() {
        let mut service = service_with_pack(pack_with(BackendComponentId::ToyV1, Vec::new()));
        let mut request = valid_request();
        request.input.t = 0;
        let record = service.submit(
            request,
            JobSubmissionMeta {
                submitted_at_unix_ms: 2,
                submitted_by: None,
            },
        );
        assert_eq!(record.state, JobLifecycleState::Rejected);
        let rejection = record.rejection.as_ref().expect("rejection reason");
        assert_eq!(rejection.kind, CanonicalFailureKind::InvalidInput);
    }

    #[test]
    fn incompatible_budget_is_rejected_in_admission() {
        let mut service = service_with_pack(pack_with(BackendComponentId::ToyV1, Vec::new()));
        let mut request = valid_request();
        request.budget.global_work_units = 0;
        let record = service.submit(
            request,
            JobSubmissionMeta {
                submitted_at_unix_ms: 6,
                submitted_by: None,
            },
        );
        assert_eq!(record.state, JobLifecycleState::Rejected);
        assert_eq!(
            record.rejection.as_ref().expect("budget rejection").kind,
            CanonicalFailureKind::BudgetExceeded
        );
    }

    #[test]
    fn artifact_or_backend_issue_rejects_during_admission() {
        let mut artifact_service = service_with_pack(pack_with(
            BackendComponentId::ToyV1,
            vec![ModelSlotProvenance {
                slot: ModelSlot::WorldJepa,
                stage: "world",
                required_for_pack: true,
                status: SlotRuntimeStatus::Unavailable,
                code: Some(ArtifactFailureCode::ArtifactUnavailable),
                detail: Some("missing".to_string()),
                resolved_path: None,
                hash_prefix: None,
                contract_version: None,
                format: None,
            }],
        ));
        let artifact_record = artifact_service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 3,
                submitted_by: None,
            },
        );
        assert_eq!(artifact_record.state, JobLifecycleState::Rejected);
        assert_eq!(
            artifact_record
                .rejection
                .as_ref()
                .expect("artifact rejection")
                .kind,
            CanonicalFailureKind::ArtifactUnavailable
        );

        let mut backend_service =
            service_with_pack(pack_with(BackendComponentId::Disabled, Vec::new()));
        let backend_record = backend_service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 4,
                submitted_by: None,
            },
        );
        assert_eq!(backend_record.state, JobLifecycleState::Rejected);
        assert_eq!(
            backend_record
                .rejection
                .as_ref()
                .expect("backend rejection")
                .kind,
            CanonicalFailureKind::BackendDisabled
        );
    }

    #[test]
    fn admitted_job_runs_on_canonical_pipeline_path() {
        let mut service = service_with_pack(pack_with(BackendComponentId::ToyV1, Vec::new()));
        let job_id = service
            .submit(
                valid_request(),
                JobSubmissionMeta {
                    submitted_at_unix_ms: 5,
                    submitted_by: Some("runner".to_string()),
                },
            )
            .job
            .id;
        let executed = service
            .run_next()
            .expect("run should execute")
            .expect("queued job should exist");
        assert_eq!(executed.job.id, job_id);
        assert!(matches!(
            executed.state,
            JobLifecycleState::Completed | JobLifecycleState::Failed | JobLifecycleState::TimedOut
        ));
        assert!(executed.result.is_some());
        let result = executed.result.as_ref().expect("canonical result");
        assert_eq!(result.request, executed.job.request.input);
    }
}
