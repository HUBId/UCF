use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::backend_pack::{
    BackendPackConfig, BackendPackFactory, BackendPackKind, ModelSlotProvenance,
};
use crate::pipeline::{
    BackendExecutionLane, CanonicalAdmissionDecision, CanonicalFailureKind,
    CanonicalPipelineFailure, CanonicalPipelineRequest, CanonicalPipelineResult,
    CanonicalPipelineState, CanonicalStageId, CanonicalWorkSummary, ComputePipelineBackend,
    FusionConfig, LimitsConfig,
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
    pub observed_at_unix_ms: u64,
    pub execution_path: JobExecutionPath,
    pub completion_class: Option<JobCompletionClass>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobCompletionClass {
    RejectedBeforeExecution,
    Completed,
    DegradedCompleted,
    FailedDuringExecution,
    TimedOut,
    WorkerIpcFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobAccountingSummary {
    pub job_id: JobId,
    pub status: JobLifecycleState,
    pub completion_class: JobCompletionClass,
    pub submitted_at_unix_ms: u64,
    pub started_at_unix_ms: Option<u64>,
    pub finished_at_unix_ms: Option<u64>,
    pub queue_wait_ms: Option<u64>,
    pub execution_duration_micros: Option<u64>,
    pub total_duration_ms: Option<u64>,
    pub failure_kind: Option<CanonicalFailureKind>,
    pub work_summary: Option<CanonicalWorkSummary>,
    pub pipeline_state: Option<CanonicalPipelineState>,
    pub stage_order: Option<[CanonicalStageId; 4]>,
    pub executed_stages: Vec<CanonicalStageId>,
    pub model_slots: Vec<ModelSlotProvenance>,
    pub execution_path: JobExecutionPath,
    pub execution_lane: BackendExecutionLane,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JobRecord {
    pub job: ComputeJob,
    pub state: JobLifecycleState,
    pub admission: CanonicalAdmissionDecision,
    pub rejection: Option<CanonicalPipelineFailure>,
    pub execution_failure: Option<CanonicalPipelineFailure>,
    pub result: Option<CanonicalPipelineResult>,
    pub execution_path: JobExecutionPath,
    pub accounting: JobAccountingSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobExecutionPath {
    LocalCanonical,
    WorkerIpc,
}

impl JobExecutionPath {
    fn as_detail(self) -> &'static str {
        match self {
            Self::LocalCanonical => "execution_path=local_canonical",
            Self::WorkerIpc => "execution_path=worker_ipc",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub max_concurrent_jobs: usize,
    pub execution_path: JobExecutionPath,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_jobs: 1,
            execution_path: JobExecutionPath::LocalCanonical,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerSnapshot {
    pub max_concurrent_jobs: usize,
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub execution_path: JobExecutionPath,
}

pub struct InMemoryComputeService {
    backend: ComputePipelineBackend,
    scheduler: SchedulerConfig,
    next_job_id: u64,
    jobs: BTreeMap<JobId, JobRecord>,
    queue: VecDeque<JobId>,
    running: BTreeSet<JobId>,
    lifecycle: Vec<JobLifecycleEvent>,
}

impl InMemoryComputeService {
    pub fn new(backend: ComputePipelineBackend) -> Self {
        Self::with_scheduler(backend, SchedulerConfig::default())
    }

    pub fn with_scheduler(backend: ComputePipelineBackend, mut scheduler: SchedulerConfig) -> Self {
        scheduler.max_concurrent_jobs = scheduler.max_concurrent_jobs.max(1);
        Self {
            backend,
            scheduler,
            next_job_id: 1,
            jobs: BTreeMap::new(),
            queue: VecDeque::new(),
            running: BTreeSet::new(),
            lifecycle: Vec::new(),
        }
    }

    pub fn new_worker(seed: u64, max_concurrent_jobs: usize) -> Result<Self, ComputeError> {
        let pack = BackendPackFactory::build(BackendPackConfig {
            pack: BackendPackKind::WorkerV1,
            seed,
        })?;
        let backend =
            ComputePipelineBackend::new(pack, FusionConfig::default(), LimitsConfig::default());
        Ok(Self::with_scheduler(
            backend,
            SchedulerConfig {
                max_concurrent_jobs,
                execution_path: JobExecutionPath::WorkerIpc,
            },
        ))
    }

    pub fn submit(
        &mut self,
        request: CanonicalPipelineRequest,
        meta: JobSubmissionMeta,
    ) -> &JobRecord {
        let job_id = JobId(self.next_job_id);
        self.next_job_id = self.next_job_id.saturating_add(1);
        let submitted_at_unix_ms = meta.submitted_at_unix_ms;
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
            execution_path: self.scheduler.execution_path,
            accounting: JobAccountingSummary {
                job_id,
                status: JobLifecycleState::Submitted,
                completion_class: JobCompletionClass::RejectedBeforeExecution,
                submitted_at_unix_ms,
                started_at_unix_ms: None,
                finished_at_unix_ms: None,
                queue_wait_ms: None,
                execution_duration_micros: None,
                total_duration_ms: None,
                failure_kind: None,
                work_summary: None,
                pipeline_state: None,
                stage_order: None,
                executed_stages: Vec::new(),
                model_slots: Vec::new(),
                execution_path: self.scheduler.execution_path,
                execution_lane: self.backend.execution_lane(),
            },
        };
        self.record_event(JobLifecycleEvent {
            job_id,
            state: JobLifecycleState::Submitted,
            failure_kind: None,
            detail: None,
            observed_at_unix_ms: submitted_at_unix_ms,
            execution_path: self.scheduler.execution_path,
            completion_class: None,
        });
        match admission.failure {
            Some(failure) => {
                record.state = JobLifecycleState::Rejected;
                record.rejection = Some(failure.clone());
                record.accounting.status = JobLifecycleState::Rejected;
                record.accounting.completion_class = JobCompletionClass::RejectedBeforeExecution;
                record.accounting.failure_kind = Some(failure.kind);
                self.record_event(JobLifecycleEvent {
                    job_id,
                    state: JobLifecycleState::Rejected,
                    failure_kind: Some(failure.kind),
                    detail: Some(failure.detail),
                    observed_at_unix_ms: now_unix_ms(),
                    execution_path: self.scheduler.execution_path,
                    completion_class: Some(JobCompletionClass::RejectedBeforeExecution),
                });
            }
            None => {
                record.state = JobLifecycleState::Admitted;
                record.accounting.status = JobLifecycleState::Admitted;
                self.record_event(JobLifecycleEvent {
                    job_id,
                    state: JobLifecycleState::Admitted,
                    failure_kind: None,
                    detail: None,
                    observed_at_unix_ms: now_unix_ms(),
                    execution_path: self.scheduler.execution_path,
                    completion_class: None,
                });
                record.state = JobLifecycleState::Queued;
                record.accounting.status = JobLifecycleState::Queued;
                self.queue.push_back(job_id);
                self.record_event(JobLifecycleEvent {
                    job_id,
                    state: JobLifecycleState::Queued,
                    failure_kind: None,
                    detail: None,
                    observed_at_unix_ms: now_unix_ms(),
                    execution_path: self.scheduler.execution_path,
                    completion_class: None,
                });
            }
        }
        self.jobs.insert(job_id, record);
        self.jobs.get(&job_id).expect("inserted record must exist")
    }

    pub fn run_next(&mut self) -> Result<Option<&JobRecord>, ComputeError> {
        self.run_scheduler_cycle(1)
            .map(|mut done| done.pop().and_then(|job_id| self.jobs.get(&job_id)))
    }

    pub fn run_scheduler_cycle(&mut self, max_jobs: usize) -> Result<Vec<JobId>, ComputeError> {
        let mut completed = Vec::new();
        let dispatch_cap = max_jobs.max(1);
        while completed.len() < dispatch_cap
            && self.running.len() < self.scheduler.max_concurrent_jobs
        {
            let Some(job_id) = self.queue.pop_front() else {
                break;
            };
            self.execute_job(job_id)?;
            completed.push(job_id);
        }
        Ok(completed)
    }

    fn execute_job(&mut self, job_id: JobId) -> Result<(), ComputeError> {
        let started_at_unix_ms = now_unix_ms();
        let execution_started = Instant::now();
        let request = match self.jobs.get_mut(&job_id) {
            Some(record) => {
                record.state = JobLifecycleState::Running;
                record.accounting.status = JobLifecycleState::Running;
                record.accounting.started_at_unix_ms = Some(started_at_unix_ms);
                record.accounting.queue_wait_ms =
                    Some(started_at_unix_ms.saturating_sub(record.accounting.submitted_at_unix_ms));
                record.job.request.clone()
            }
            None => return Ok(()),
        };
        self.running.insert(job_id);
        self.record_event(JobLifecycleEvent {
            job_id,
            state: JobLifecycleState::Running,
            failure_kind: None,
            detail: Some(self.scheduler.execution_path.as_detail().to_string()),
            observed_at_unix_ms: started_at_unix_ms,
            execution_path: self.scheduler.execution_path,
            completion_class: None,
        });

        let run_outcome = self.backend.compute_canonical(request);
        let (state, result, execution_failure) = match run_outcome {
            Ok(result) => {
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
                (state, Some(result), failure)
            }
            Err(err) => {
                let failure = canonical_execution_failure(err);
                let state = if failure.kind == CanonicalFailureKind::Timeout {
                    JobLifecycleState::TimedOut
                } else {
                    JobLifecycleState::Failed
                };
                (state, None, Some(failure))
            }
        };

        let Some(record) = self.jobs.get_mut(&job_id) else {
            self.running.remove(&job_id);
            return Ok(());
        };
        let finished_at_unix_ms = now_unix_ms();
        let execution_duration_micros = execution_started.elapsed().as_micros() as u64;
        let completion_class = completion_class_for(
            state,
            &execution_failure,
            self.scheduler.execution_path,
            result.as_ref(),
        );
        record.result = result;
        record.state = state;
        record.execution_failure = execution_failure.clone();
        record.accounting.status = state;
        record.accounting.completion_class = completion_class;
        record.accounting.finished_at_unix_ms = Some(finished_at_unix_ms);
        record.accounting.execution_duration_micros = Some(execution_duration_micros);
        record.accounting.total_duration_ms =
            Some(finished_at_unix_ms.saturating_sub(record.accounting.submitted_at_unix_ms));
        record.accounting.failure_kind = execution_failure.as_ref().map(|f| f.kind);
        if let Some(canonical_result) = record.result.as_ref() {
            record.accounting.work_summary = Some(canonical_result.diagnostics.work);
            record.accounting.pipeline_state = Some(canonical_result.state);
            record.accounting.stage_order = Some(canonical_result.stage_order);
            record.accounting.executed_stages = canonical_result.executed_stages.clone();
            record.accounting.model_slots = canonical_result.model_slots.clone();
        }
        self.running.remove(&job_id);
        if let Some(failure) = execution_failure {
            self.record_event(JobLifecycleEvent {
                job_id,
                state,
                failure_kind: Some(failure.kind),
                detail: Some(format!(
                    "{}; {}",
                    self.scheduler.execution_path.as_detail(),
                    failure.detail
                )),
                observed_at_unix_ms: finished_at_unix_ms,
                execution_path: self.scheduler.execution_path,
                completion_class: Some(completion_class),
            });
        } else {
            self.record_event(JobLifecycleEvent {
                job_id,
                state,
                failure_kind: None,
                detail: Some(self.scheduler.execution_path.as_detail().to_string()),
                observed_at_unix_ms: finished_at_unix_ms,
                execution_path: self.scheduler.execution_path,
                completion_class: Some(completion_class),
            });
        }
        Ok(())
    }

    pub fn scheduler_snapshot(&self) -> SchedulerSnapshot {
        SchedulerSnapshot {
            max_concurrent_jobs: self.scheduler.max_concurrent_jobs,
            queued_jobs: self.queue.len(),
            running_jobs: self.running.len(),
            execution_path: self.scheduler.execution_path,
        }
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn execution_path(&self) -> JobExecutionPath {
        self.scheduler.execution_path
    }

    pub fn technical_admission(
        &self,
        request: &CanonicalPipelineRequest,
    ) -> CanonicalAdmissionDecision {
        self.backend.technical_admission(request)
    }

    pub fn execution_lane(&self) -> BackendExecutionLane {
        self.backend.execution_lane()
    }

    pub fn job(&self, job_id: JobId) -> Option<&JobRecord> {
        self.jobs.get(&job_id)
    }

    pub fn lifecycle_events(&self) -> &[JobLifecycleEvent] {
        &self.lifecycle
    }

    pub fn jobs(&self) -> impl Iterator<Item = &JobRecord> {
        self.jobs.values()
    }

    fn record_event(&mut self, event: JobLifecycleEvent) {
        self.lifecycle.push(event);
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

fn completion_class_for(
    state: JobLifecycleState,
    execution_failure: &Option<CanonicalPipelineFailure>,
    execution_path: JobExecutionPath,
    result: Option<&CanonicalPipelineResult>,
) -> JobCompletionClass {
    match state {
        JobLifecycleState::Rejected => JobCompletionClass::RejectedBeforeExecution,
        JobLifecycleState::TimedOut => JobCompletionClass::TimedOut,
        JobLifecycleState::Completed => {
            if result
                .map(|r| r.state == CanonicalPipelineState::Degraded)
                .unwrap_or(false)
            {
                JobCompletionClass::DegradedCompleted
            } else {
                JobCompletionClass::Completed
            }
        }
        JobLifecycleState::Failed => {
            if execution_path == JobExecutionPath::WorkerIpc
                && execution_failure
                    .as_ref()
                    .map(|failure| failure.kind == CanonicalFailureKind::ExecutionError)
                    .unwrap_or(false)
            {
                JobCompletionClass::WorkerIpcFailure
            } else {
                JobCompletionClass::FailedDuringExecution
            }
        }
        _ => JobCompletionClass::FailedDuringExecution,
    }
}

fn canonical_execution_failure(err: ComputeError) -> CanonicalPipelineFailure {
    let kind = match err {
        ComputeError::BudgetExceeded { .. } => CanonicalFailureKind::Timeout,
        _ => CanonicalFailureKind::ExecutionError,
    };
    CanonicalPipelineFailure {
        kind,
        stage: None,
        detail: err.to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionUnitId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionUnitKind {
    Local,
    Worker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerAvailability {
    Available,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerDispatchOutcome {
    Unavailable,
    DispatchFailure,
    TransportFailure,
    ExecutionFailure,
    Timeout,
    Completed,
    RedispatchedLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRuntimeStatus {
    Known,
    Ready,
    Busy,
    Saturated,
    Unavailable,
    Unhealthy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlacement {
    pub unit_id: ExecutionUnitId,
    pub unit_kind: ExecutionUnitKind,
    pub device_class: ExecutionDeviceClass,
    pub execution_path: JobExecutionPath,
    pub lane: BackendExecutionLane,
    pub suitability: PlacementSuitability,
    pub device_suitability: DeviceSuitability,
    pub device_preference: Option<ExecutionDeviceClass>,
    pub device_preference_met: bool,
    pub device_fallback_from: Option<ExecutionDeviceClass>,
    pub degraded_fallback: bool,
    pub reason: String,
    pub considered: Vec<PlacementCandidateAssessment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementSuitability {
    Suitable,
    Incompatible,
    Disabled,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementFailureKind {
    NoSuitableBackend,
    NoSuitableDevice,
    BackendIncompatible,
    BackendDeviceIncompatible,
    DeviceUnavailable,
    BackendUnavailable,
    WorkerPlacementFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionDeviceClass {
    Cpu,
    Worker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSuitability {
    Suitable,
    Unsuitable,
    Disabled,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementCandidateAssessment {
    pub unit_id: ExecutionUnitId,
    pub unit_kind: ExecutionUnitKind,
    pub device_class: ExecutionDeviceClass,
    pub lane: BackendExecutionLane,
    pub backend_suitability: PlacementSuitability,
    pub device_suitability: DeviceSuitability,
    pub suitability: PlacementSuitability,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiWorkerJobRecord {
    pub id: JobId,
    pub state: JobLifecycleState,
    pub execution_failure: Option<CanonicalPipelineFailure>,
    pub result: Option<CanonicalPipelineResult>,
    pub placement: ExecutionPlacement,
    pub worker_dispatch_outcome: Option<WorkerDispatchOutcome>,
    pub placement_failure: Option<PlacementFailureKind>,
    pub provenance: WorkerExecutionProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerExecutionProvenance {
    pub selected_unit: ExecutionUnitId,
    pub completed_unit: ExecutionUnitId,
    pub was_remote: bool,
    pub redispatched_to_local: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionUnitSnapshot {
    pub id: ExecutionUnitId,
    pub kind: ExecutionUnitKind,
    pub availability: WorkerAvailability,
    pub status: WorkerRuntimeStatus,
    pub max_parallel_jobs: usize,
    pub active_jobs: usize,
    pub consecutive_failures: u32,
    pub last_job_id: Option<JobId>,
    pub last_dispatch_outcome: Option<WorkerDispatchOutcome>,
    pub last_error: Option<String>,
    pub last_used_at_unix_ms: Option<u64>,
}

struct ExecutionUnit {
    id: ExecutionUnitId,
    kind: ExecutionUnitKind,
    availability: WorkerAvailability,
    max_parallel_jobs: usize,
    active_jobs: usize,
    consecutive_failures: u32,
    last_job_id: Option<JobId>,
    last_dispatch_outcome: Option<WorkerDispatchOutcome>,
    last_error: Option<String>,
    last_used_at_unix_ms: Option<u64>,
    service: InMemoryComputeService,
}

#[derive(Debug, Clone)]
struct QueuedJob {
    id: JobId,
    request: CanonicalPipelineRequest,
    meta: JobSubmissionMeta,
    requested_unit: Option<ExecutionUnitId>,
}

struct UnitSelection {
    idx: usize,
    placement: ExecutionPlacement,
}

pub struct MultiWorkerComputeService {
    next_job_id: u64,
    queue: VecDeque<QueuedJob>,
    units: Vec<ExecutionUnit>,
    records: BTreeMap<JobId, MultiWorkerJobRecord>,
    round_robin_cursor: usize,
}

impl MultiWorkerComputeService {
    pub fn new(local_backend: ComputePipelineBackend, max_parallel_jobs: usize) -> Self {
        let local = ExecutionUnit {
            id: ExecutionUnitId("local".to_string()),
            kind: ExecutionUnitKind::Local,
            availability: WorkerAvailability::Available,
            max_parallel_jobs: max_parallel_jobs.max(1),
            active_jobs: 0,
            consecutive_failures: 0,
            last_job_id: None,
            last_dispatch_outcome: None,
            last_error: None,
            last_used_at_unix_ms: None,
            service: InMemoryComputeService::with_scheduler(
                local_backend,
                SchedulerConfig {
                    max_concurrent_jobs: max_parallel_jobs.max(1),
                    execution_path: JobExecutionPath::LocalCanonical,
                },
            ),
        };
        Self {
            next_job_id: 1,
            queue: VecDeque::new(),
            units: vec![local],
            records: BTreeMap::new(),
            round_robin_cursor: 0,
        }
    }

    pub fn register_worker(
        &mut self,
        worker_id: impl Into<String>,
        seed: u64,
        max_parallel_jobs: usize,
    ) -> Result<ExecutionUnitId, ComputeError> {
        let id = ExecutionUnitId(worker_id.into());
        let worker = InMemoryComputeService::new_worker(seed, max_parallel_jobs.max(1))?;
        self.register_worker_service(id.clone(), worker, max_parallel_jobs.max(1));
        Ok(id)
    }

    pub fn register_worker_backend(
        &mut self,
        worker_id: impl Into<String>,
        backend: ComputePipelineBackend,
        max_parallel_jobs: usize,
    ) -> ExecutionUnitId {
        let id = ExecutionUnitId(worker_id.into());
        let worker = InMemoryComputeService::with_scheduler(
            backend,
            SchedulerConfig {
                max_concurrent_jobs: max_parallel_jobs.max(1),
                execution_path: JobExecutionPath::WorkerIpc,
            },
        );
        self.register_worker_service(id.clone(), worker, max_parallel_jobs.max(1));
        id
    }

    fn register_worker_service(
        &mut self,
        id: ExecutionUnitId,
        worker: InMemoryComputeService,
        max_parallel_jobs: usize,
    ) {
        self.units.push(ExecutionUnit {
            id: id.clone(),
            kind: ExecutionUnitKind::Worker,
            availability: WorkerAvailability::Available,
            max_parallel_jobs: max_parallel_jobs.max(1),
            active_jobs: 0,
            consecutive_failures: 0,
            last_job_id: None,
            last_dispatch_outcome: None,
            last_error: None,
            last_used_at_unix_ms: None,
            service: worker,
        });
    }

    pub fn set_worker_availability(
        &mut self,
        worker_id: &ExecutionUnitId,
        availability: WorkerAvailability,
    ) {
        for unit in &mut self.units {
            if &unit.id == worker_id {
                unit.availability = availability;
            }
        }
    }

    pub fn submit(
        &mut self,
        request: CanonicalPipelineRequest,
        meta: JobSubmissionMeta,
        requested_unit: Option<ExecutionUnitId>,
    ) -> JobId {
        let id = JobId(self.next_job_id);
        self.next_job_id = self.next_job_id.saturating_add(1);
        self.queue.push_back(QueuedJob {
            id,
            request,
            meta,
            requested_unit,
        });
        id
    }

    pub fn run_scheduler_cycle(&mut self, max_jobs: usize) -> Vec<JobId> {
        let mut done = Vec::new();
        while done.len() < max_jobs.max(1) {
            let Some(job) = self.queue.pop_front() else {
                break;
            };
            let selection = match self.select_unit(&job.request, job.requested_unit.clone()) {
                Some(selection) => selection,
                None => {
                    let (placement, placement_failure, worker_dispatch_outcome) =
                        self.rejected_placement(&job);
                    if let Some(requested_unit) = job.requested_unit.clone() {
                        let record = MultiWorkerJobRecord {
                            id: job.id,
                            state: JobLifecycleState::Failed,
                            execution_failure: Some(CanonicalPipelineFailure {
                                kind: CanonicalFailureKind::ExecutionError,
                                stage: None,
                                detail: format!("worker placement failed: {}", requested_unit.0),
                            }),
                            result: None,
                            placement,
                            worker_dispatch_outcome,
                            placement_failure: Some(placement_failure),
                            provenance: WorkerExecutionProvenance {
                                selected_unit: requested_unit.clone(),
                                completed_unit: requested_unit,
                                was_remote: true,
                                redispatched_to_local: false,
                            },
                        };
                        done.push(record.id);
                        self.records.insert(record.id, record);
                        continue;
                    }
                    let record = MultiWorkerJobRecord {
                        id: job.id,
                        state: JobLifecycleState::Failed,
                        execution_failure: Some(CanonicalPipelineFailure {
                            kind: CanonicalFailureKind::ExecutionError,
                            stage: None,
                            detail: "no suitable backend".to_string(),
                        }),
                        result: None,
                        placement,
                        worker_dispatch_outcome,
                        placement_failure: Some(placement_failure),
                        provenance: WorkerExecutionProvenance {
                            selected_unit: ExecutionUnitId("none".to_string()),
                            completed_unit: ExecutionUnitId("none".to_string()),
                            was_remote: false,
                            redispatched_to_local: false,
                        },
                    };
                    done.push(record.id);
                    self.records.insert(record.id, record);
                    continue;
                }
            };
            let record = self.execute(job, selection);
            done.push(record.id);
            self.records.insert(record.id, record);
        }
        done
    }

    pub fn job(&self, id: JobId) -> Option<&MultiWorkerJobRecord> {
        self.records.get(&id)
    }

    pub fn execution_units(&self) -> Vec<ExecutionUnitSnapshot> {
        self.units
            .iter()
            .map(|unit| ExecutionUnitSnapshot {
                id: unit.id.clone(),
                kind: unit.kind,
                availability: unit.availability,
                status: unit.runtime_status(),
                max_parallel_jobs: unit.max_parallel_jobs,
                active_jobs: unit.active_jobs,
                consecutive_failures: unit.consecutive_failures,
                last_job_id: unit.last_job_id,
                last_dispatch_outcome: unit.last_dispatch_outcome,
                last_error: unit.last_error.clone(),
                last_used_at_unix_ms: unit.last_used_at_unix_ms,
            })
            .collect()
    }

    fn rejected_placement(
        &self,
        job: &QueuedJob,
    ) -> (
        ExecutionPlacement,
        PlacementFailureKind,
        Option<WorkerDispatchOutcome>,
    ) {
        let mut considered = self.assess_candidates(&job.request);
        if let Some(requested) = job.requested_unit.clone() {
            let candidate = considered
                .iter()
                .find(|candidate| candidate.unit_id == requested)
                .cloned();
            let suitability = candidate
                .as_ref()
                .map(|candidate| candidate.suitability)
                .unwrap_or(PlacementSuitability::Unavailable);
            let device_suitability = candidate
                .as_ref()
                .map(|candidate| candidate.device_suitability)
                .unwrap_or(DeviceSuitability::Unavailable);
            let kind = if candidate.is_none() {
                PlacementFailureKind::WorkerPlacementFailed
            } else {
                placement_failure_kind(suitability, device_suitability)
            };
            let dispatch =
                if suitability == PlacementSuitability::Unavailable && candidate.is_some() {
                    Some(WorkerDispatchOutcome::Unavailable)
                } else {
                    Some(WorkerDispatchOutcome::DispatchFailure)
                };
            return (
                ExecutionPlacement {
                    unit_id: requested,
                    unit_kind: ExecutionUnitKind::Worker,
                    device_class: ExecutionDeviceClass::Worker,
                    execution_path: JobExecutionPath::WorkerIpc,
                    lane: BackendExecutionLane::Worker,
                    suitability,
                    device_suitability,
                    device_preference: Some(ExecutionDeviceClass::Worker),
                    device_preference_met: candidate
                        .as_ref()
                        .map(|c| {
                            c.device_class == ExecutionDeviceClass::Worker
                                && c.device_suitability == DeviceSuitability::Suitable
                        })
                        .unwrap_or(false),
                    device_fallback_from: None,
                    degraded_fallback: false,
                    reason: "requested unit not placeable".to_string(),
                    considered,
                },
                kind,
                dispatch,
            );
        }
        let best = if considered
            .iter()
            .any(|candidate| candidate.suitability == PlacementSuitability::Incompatible)
        {
            PlacementFailureKind::BackendIncompatible
        } else if considered
            .iter()
            .any(|candidate| candidate.device_suitability == DeviceSuitability::Unsuitable)
        {
            PlacementFailureKind::BackendDeviceIncompatible
        } else if considered
            .iter()
            .any(|candidate| candidate.device_suitability == DeviceSuitability::Unavailable)
        {
            PlacementFailureKind::DeviceUnavailable
        } else if considered
            .iter()
            .any(|candidate| candidate.suitability == PlacementSuitability::Disabled)
        {
            PlacementFailureKind::BackendUnavailable
        } else if considered.iter().any(|candidate| {
            candidate.suitability == PlacementSuitability::Suitable
                && candidate.device_suitability == DeviceSuitability::Disabled
        }) {
            PlacementFailureKind::NoSuitableDevice
        } else {
            PlacementFailureKind::NoSuitableBackend
        };
        considered.sort_by(|a, b| a.unit_id.cmp(&b.unit_id));
        (
            ExecutionPlacement {
                unit_id: ExecutionUnitId("none".to_string()),
                unit_kind: ExecutionUnitKind::Local,
                device_class: ExecutionDeviceClass::Cpu,
                execution_path: JobExecutionPath::LocalCanonical,
                lane: BackendExecutionLane::Toy,
                suitability: PlacementSuitability::Unavailable,
                device_suitability: DeviceSuitability::Unavailable,
                device_preference: None,
                device_preference_met: true,
                device_fallback_from: None,
                degraded_fallback: false,
                reason: "no suitable execution unit".to_string(),
                considered,
            },
            best,
            None,
        )
    }

    fn assess_candidates(
        &self,
        request: &CanonicalPipelineRequest,
    ) -> Vec<PlacementCandidateAssessment> {
        self.units
            .iter()
            .map(|unit| {
                let lane = unit.service.execution_lane();
                let device_class = unit_device_class(unit.kind);
                let status = unit.runtime_status();
                let device_suitability = device_suitability_for(unit.kind, lane, status);
                if !unit.can_accept_dispatch() {
                    return PlacementCandidateAssessment {
                        unit_id: unit.id.clone(),
                        unit_kind: unit.kind,
                        device_class,
                        lane,
                        backend_suitability: PlacementSuitability::Unavailable,
                        device_suitability,
                        suitability: PlacementSuitability::Unavailable,
                        detail: format!("unit not dispatchable ({status:?})"),
                    };
                }
                let admission = unit.service.technical_admission(request);
                let backend_suitability = admission
                    .failure
                    .as_ref()
                    .map(|failure| suitability_for_failure(failure.kind))
                    .unwrap_or(PlacementSuitability::Suitable);
                let suitability = combine_suitability(backend_suitability, device_suitability);
                if let Some(failure) = admission.failure {
                    PlacementCandidateAssessment {
                        unit_id: unit.id.clone(),
                        unit_kind: unit.kind,
                        device_class,
                        lane,
                        backend_suitability,
                        device_suitability,
                        suitability,
                        detail: failure.detail,
                    }
                } else {
                    PlacementCandidateAssessment {
                        unit_id: unit.id.clone(),
                        unit_kind: unit.kind,
                        device_class,
                        lane,
                        backend_suitability,
                        device_suitability,
                        suitability,
                        detail: "admitted".to_string(),
                    }
                }
            })
            .collect()
    }

    fn select_unit(
        &mut self,
        request: &CanonicalPipelineRequest,
        requested: Option<ExecutionUnitId>,
    ) -> Option<UnitSelection> {
        let assessments = self.assess_candidates(request);
        if let Some(requested) = requested {
            let idx = self.units.iter().position(|unit| unit.id == requested)?;
            let selected = assessments
                .iter()
                .find(|candidate| candidate.unit_id == requested)?;
            if selected.suitability != PlacementSuitability::Suitable {
                return None;
            }
            return Some(UnitSelection {
                idx,
                placement: ExecutionPlacement {
                    unit_id: requested,
                    unit_kind: selected.unit_kind,
                    device_class: selected.device_class,
                    execution_path: match selected.unit_kind {
                        ExecutionUnitKind::Local => JobExecutionPath::LocalCanonical,
                        ExecutionUnitKind::Worker => JobExecutionPath::WorkerIpc,
                    },
                    lane: selected.lane,
                    suitability: selected.suitability,
                    device_suitability: selected.device_suitability,
                    device_preference: Some(unit_device_class(selected.unit_kind)),
                    device_preference_met: true,
                    device_fallback_from: None,
                    degraded_fallback: false,
                    reason: "requested execution unit selected".to_string(),
                    considered: assessments,
                },
            });
        }
        let mut suitable = assessments
            .iter()
            .filter(|candidate| candidate.suitability == PlacementSuitability::Suitable)
            .cloned()
            .collect::<Vec<_>>();
        if suitable.is_empty() {
            return None;
        }
        suitable.sort_by_key(|candidate| {
            let lane_rank = match candidate.lane {
                BackendExecutionLane::Burn => 0usize,
                BackendExecutionLane::Candle => 1,
                BackendExecutionLane::Worker => 2,
                BackendExecutionLane::Toy | BackendExecutionLane::Mixed => 3,
            };
            let base = self
                .units
                .iter()
                .position(|unit| unit.id == candidate.unit_id)
                .unwrap_or(usize::MAX);
            (
                lane_rank,
                (base + self.round_robin_cursor) % self.units.len().max(1),
            )
        });
        let selected = suitable.first()?.clone();
        let idx = self
            .units
            .iter()
            .position(|unit| unit.id == selected.unit_id)?;
        self.round_robin_cursor = (idx + 1) % self.units.len().max(1);
        Some(UnitSelection {
            idx,
            placement: ExecutionPlacement {
                unit_id: selected.unit_id,
                unit_kind: selected.unit_kind,
                device_class: selected.device_class,
                execution_path: match selected.unit_kind {
                    ExecutionUnitKind::Local => JobExecutionPath::LocalCanonical,
                    ExecutionUnitKind::Worker => JobExecutionPath::WorkerIpc,
                },
                lane: selected.lane,
                suitability: selected.suitability,
                device_suitability: selected.device_suitability,
                device_preference: None,
                device_preference_met: true,
                device_fallback_from: None,
                degraded_fallback: selected.lane == BackendExecutionLane::Candle,
                reason: if selected.lane == BackendExecutionLane::Burn {
                    "selected burn-capable unit".to_string()
                } else if selected.lane == BackendExecutionLane::Candle {
                    "burn unavailable; selected candle fallback".to_string()
                } else {
                    "selected available non-burn unit".to_string()
                },
                considered: assessments,
            },
        })
    }

    fn execute(&mut self, job: QueuedJob, selection: UnitSelection) -> MultiWorkerJobRecord {
        let placement = selection.placement;
        let unit_idx = selection.idx;
        if !self.units[unit_idx].can_accept_dispatch() {
            let unit = &mut self.units[unit_idx];
            let failure = CanonicalPipelineFailure {
                kind: CanonicalFailureKind::ExecutionError,
                stage: None,
                detail: format!("worker unavailable: {}", unit.id.0),
            };
            unit.note_failure(
                job.id,
                WorkerDispatchOutcome::Unavailable,
                failure.detail.clone(),
            );
            return MultiWorkerJobRecord {
                id: job.id,
                state: JobLifecycleState::Failed,
                execution_failure: Some(failure),
                result: None,
                placement,
                worker_dispatch_outcome: Some(WorkerDispatchOutcome::Unavailable),
                placement_failure: Some(PlacementFailureKind::BackendUnavailable),
                provenance: WorkerExecutionProvenance {
                    selected_unit: unit.id.clone(),
                    completed_unit: unit.id.clone(),
                    was_remote: unit.kind == ExecutionUnitKind::Worker,
                    redispatched_to_local: false,
                },
            };
        }

        let (submitted_job_id, run_result) = {
            let unit = &mut self.units[unit_idx];
            unit.active_jobs = unit.active_jobs.saturating_add(1);
            unit.last_job_id = Some(job.id);
            unit.last_used_at_unix_ms = Some(now_unix_ms());
            let submitted = unit.service.submit(job.request.clone(), job.meta.clone());
            let submitted_job_id = submitted.job.id;
            let run_result = match unit.service.run_next() {
                Ok(Some(record)) => Ok(Some((
                    record.job.id,
                    record.state,
                    record.execution_failure.clone(),
                    record.result.clone(),
                ))),
                Ok(None) => Ok(None),
                Err(err) => Err(err),
            };
            unit.active_jobs = unit.active_jobs.saturating_sub(1);
            (submitted_job_id, run_result)
        };

        match run_result {
            Ok(Some((record_job_id, state, execution_failure, result)))
                if submitted_job_id == record_job_id =>
            {
                let unit = &mut self.units[unit_idx];
                let selected_unit = unit.id.clone();
                let was_remote = unit.kind == ExecutionUnitKind::Worker;
                let dispatch_outcome = if placement.unit_kind != ExecutionUnitKind::Worker {
                    None
                } else if state == JobLifecycleState::TimedOut {
                    Some(WorkerDispatchOutcome::Timeout)
                } else if result.is_some() {
                    Some(WorkerDispatchOutcome::Completed)
                } else if state == JobLifecycleState::Failed {
                    Some(WorkerDispatchOutcome::ExecutionFailure)
                } else {
                    Some(WorkerDispatchOutcome::DispatchFailure)
                };
                if let Some(outcome) = dispatch_outcome {
                    if state == JobLifecycleState::Completed && result.is_some() {
                        unit.note_success(job.id, outcome);
                    } else {
                        let detail = execution_failure
                            .as_ref()
                            .map(|f| f.detail.clone())
                            .unwrap_or_else(|| "worker execution failure".to_string());
                        unit.note_failure(job.id, outcome, detail);
                    }
                }
                let fallback_candidate = if placement.unit_kind == ExecutionUnitKind::Worker
                    && state == JobLifecycleState::Failed
                    && job.requested_unit.is_none()
                {
                    Some(selected_unit.clone())
                } else {
                    None
                };
                if let Some(failed_worker) = fallback_candidate {
                    if let Some(redispatched) =
                        self.try_local_fallback(&job, &placement, failed_worker)
                    {
                        return redispatched;
                    }
                }
                MultiWorkerJobRecord {
                    id: job.id,
                    state,
                    execution_failure,
                    result,
                    placement,
                    worker_dispatch_outcome: dispatch_outcome,
                    placement_failure: None,
                    provenance: WorkerExecutionProvenance {
                        selected_unit: selected_unit.clone(),
                        completed_unit: selected_unit,
                        was_remote,
                        redispatched_to_local: false,
                    },
                }
            }
            Ok(_) => {
                let failed_unit = {
                    let unit = &mut self.units[unit_idx];
                    unit.note_failure(
                        job.id,
                        WorkerDispatchOutcome::TransportFailure,
                        format!("worker transport failure: {}", unit.id.0),
                    );
                    unit.id.clone()
                };
                if self.units[unit_idx].kind == ExecutionUnitKind::Worker
                    && job.requested_unit.is_none()
                {
                    if let Some(redispatched) =
                        self.try_local_fallback(&job, &placement, failed_unit.clone())
                    {
                        return redispatched;
                    }
                }
                MultiWorkerJobRecord {
                    id: job.id,
                    state: JobLifecycleState::Failed,
                    execution_failure: Some(CanonicalPipelineFailure {
                        kind: CanonicalFailureKind::ExecutionError,
                        stage: None,
                        detail: format!("worker transport failure: {}", failed_unit.0),
                    }),
                    result: None,
                    placement,
                    worker_dispatch_outcome: Some(WorkerDispatchOutcome::TransportFailure),
                    placement_failure: Some(PlacementFailureKind::WorkerPlacementFailed),
                    provenance: WorkerExecutionProvenance {
                        selected_unit: failed_unit.clone(),
                        completed_unit: failed_unit,
                        was_remote: true,
                        redispatched_to_local: false,
                    },
                }
            }
            Err(err) => {
                let failure = canonical_execution_failure(err);
                let state = if failure.kind == CanonicalFailureKind::Timeout {
                    JobLifecycleState::TimedOut
                } else {
                    JobLifecycleState::Failed
                };
                let dispatch_outcome = if state == JobLifecycleState::TimedOut {
                    WorkerDispatchOutcome::Timeout
                } else {
                    WorkerDispatchOutcome::ExecutionFailure
                };
                let failed_unit = {
                    let unit = &mut self.units[unit_idx];
                    unit.note_failure(job.id, dispatch_outcome, failure.detail.clone());
                    unit.id.clone()
                };
                if self.units[unit_idx].kind == ExecutionUnitKind::Worker
                    && dispatch_outcome != WorkerDispatchOutcome::Timeout
                    && job.requested_unit.is_none()
                {
                    if let Some(redispatched) =
                        self.try_local_fallback(&job, &placement, failed_unit.clone())
                    {
                        return redispatched;
                    }
                }
                MultiWorkerJobRecord {
                    id: job.id,
                    state,
                    execution_failure: Some(failure),
                    result: None,
                    placement,
                    worker_dispatch_outcome: Some(dispatch_outcome),
                    placement_failure: Some(PlacementFailureKind::WorkerPlacementFailed),
                    provenance: WorkerExecutionProvenance {
                        selected_unit: failed_unit.clone(),
                        completed_unit: failed_unit,
                        was_remote: self.units[unit_idx].kind == ExecutionUnitKind::Worker,
                        redispatched_to_local: false,
                    },
                }
            }
        }
    }

    fn try_local_fallback(
        &mut self,
        job: &QueuedJob,
        original_placement: &ExecutionPlacement,
        failed_worker: ExecutionUnitId,
    ) -> Option<MultiWorkerJobRecord> {
        let local_idx = self
            .units
            .iter()
            .position(|unit| unit.kind == ExecutionUnitKind::Local && unit.can_accept_dispatch())?;
        let local = &mut self.units[local_idx];
        local.active_jobs = local.active_jobs.saturating_add(1);
        local.last_job_id = Some(job.id);
        local.last_used_at_unix_ms = Some(now_unix_ms());
        let submitted_job_id = {
            let submitted = local.service.submit(job.request.clone(), job.meta.clone());
            submitted.job.id
        };
        let run_result = match local.service.run_next() {
            Ok(Some(record)) => Ok(Some((
                record.job.id,
                record.state,
                record.execution_failure.clone(),
                record.result.clone(),
            ))),
            Ok(None) => Ok(None),
            Err(err) => Err(err),
        };
        local.active_jobs = local.active_jobs.saturating_sub(1);
        match run_result {
            Ok(Some((record_job_id, state, execution_failure, result)))
                if submitted_job_id == record_job_id =>
            {
                local.note_success(job.id, WorkerDispatchOutcome::RedispatchedLocal);
                Some(MultiWorkerJobRecord {
                    id: job.id,
                    state,
                    execution_failure,
                    result,
                    placement: ExecutionPlacement {
                        unit_id: local.id.clone(),
                        unit_kind: ExecutionUnitKind::Local,
                        device_class: ExecutionDeviceClass::Cpu,
                        execution_path: JobExecutionPath::LocalCanonical,
                        lane: local.service.execution_lane(),
                        suitability: PlacementSuitability::Suitable,
                        device_suitability: DeviceSuitability::Suitable,
                        device_preference: Some(ExecutionDeviceClass::Worker),
                        device_preference_met: false,
                        device_fallback_from: Some(ExecutionDeviceClass::Worker),
                        degraded_fallback: true,
                        reason: format!("worker {} failed; redispatched to local", failed_worker.0),
                        considered: original_placement.considered.clone(),
                    },
                    worker_dispatch_outcome: Some(WorkerDispatchOutcome::RedispatchedLocal),
                    placement_failure: None,
                    provenance: WorkerExecutionProvenance {
                        selected_unit: failed_worker,
                        completed_unit: local.id.clone(),
                        was_remote: true,
                        redispatched_to_local: true,
                    },
                })
            }
            _ => None,
        }
    }
}

impl ExecutionUnit {
    fn runtime_status(&self) -> WorkerRuntimeStatus {
        if self.availability != WorkerAvailability::Available {
            return WorkerRuntimeStatus::Unavailable;
        }
        if self.consecutive_failures >= 3 {
            return WorkerRuntimeStatus::Unhealthy;
        }
        if self.active_jobs >= self.max_parallel_jobs {
            return WorkerRuntimeStatus::Saturated;
        }
        if self.active_jobs > 0 {
            WorkerRuntimeStatus::Busy
        } else if self.last_job_id.is_some() {
            WorkerRuntimeStatus::Ready
        } else {
            WorkerRuntimeStatus::Known
        }
    }

    fn can_accept_dispatch(&self) -> bool {
        matches!(
            self.runtime_status(),
            WorkerRuntimeStatus::Known | WorkerRuntimeStatus::Ready | WorkerRuntimeStatus::Busy
        )
    }

    fn note_success(&mut self, job_id: JobId, outcome: WorkerDispatchOutcome) {
        self.last_job_id = Some(job_id);
        self.last_dispatch_outcome = Some(outcome);
        self.last_error = None;
        self.consecutive_failures = 0;
        self.last_used_at_unix_ms = Some(now_unix_ms());
    }

    fn note_failure(&mut self, job_id: JobId, outcome: WorkerDispatchOutcome, detail: String) {
        self.last_job_id = Some(job_id);
        self.last_dispatch_outcome = Some(outcome);
        self.last_error = Some(detail);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        self.last_used_at_unix_ms = Some(now_unix_ms());
    }
}

fn suitability_for_failure(kind: CanonicalFailureKind) -> PlacementSuitability {
    match kind {
        CanonicalFailureKind::BackendDisabled => PlacementSuitability::Disabled,
        CanonicalFailureKind::StageContractMismatch
        | CanonicalFailureKind::ContractMismatch
        | CanonicalFailureKind::ArtifactIncompatible => PlacementSuitability::Incompatible,
        CanonicalFailureKind::ArtifactUnavailable
        | CanonicalFailureKind::ArtifactVerificationFailed
        | CanonicalFailureKind::StageUnavailable
        | CanonicalFailureKind::NsrUnavailable
        | CanonicalFailureKind::NsrBackendUnavailable => PlacementSuitability::Unavailable,
        _ => PlacementSuitability::Unavailable,
    }
}

fn unit_device_class(kind: ExecutionUnitKind) -> ExecutionDeviceClass {
    match kind {
        ExecutionUnitKind::Local => ExecutionDeviceClass::Cpu,
        ExecutionUnitKind::Worker => ExecutionDeviceClass::Worker,
    }
}

fn device_suitability_for(
    unit_kind: ExecutionUnitKind,
    lane: BackendExecutionLane,
    status: WorkerRuntimeStatus,
) -> DeviceSuitability {
    if matches!(
        status,
        WorkerRuntimeStatus::Unavailable
            | WorkerRuntimeStatus::Unhealthy
            | WorkerRuntimeStatus::Saturated
    ) {
        return DeviceSuitability::Unavailable;
    }
    match (unit_kind, lane) {
        (ExecutionUnitKind::Local, BackendExecutionLane::Worker) => DeviceSuitability::Unsuitable,
        _ => DeviceSuitability::Suitable,
    }
}

fn combine_suitability(
    backend_suitability: PlacementSuitability,
    device_suitability: DeviceSuitability,
) -> PlacementSuitability {
    match device_suitability {
        DeviceSuitability::Suitable => backend_suitability,
        DeviceSuitability::Unsuitable => {
            if backend_suitability == PlacementSuitability::Suitable {
                PlacementSuitability::Incompatible
            } else {
                backend_suitability
            }
        }
        DeviceSuitability::Disabled => PlacementSuitability::Disabled,
        DeviceSuitability::Unavailable => PlacementSuitability::Unavailable,
    }
}

fn placement_failure_kind(
    suitability: PlacementSuitability,
    device_suitability: DeviceSuitability,
) -> PlacementFailureKind {
    match (suitability, device_suitability) {
        (PlacementSuitability::Suitable, DeviceSuitability::Suitable) => {
            PlacementFailureKind::WorkerPlacementFailed
        }
        (_, DeviceSuitability::Unsuitable) => PlacementFailureKind::BackendDeviceIncompatible,
        (_, DeviceSuitability::Unavailable) => PlacementFailureKind::DeviceUnavailable,
        (_, DeviceSuitability::Disabled) => PlacementFailureKind::NoSuitableDevice,
        (PlacementSuitability::Incompatible, _) => PlacementFailureKind::BackendIncompatible,
        (PlacementSuitability::Disabled | PlacementSuitability::Unavailable, _) => {
            PlacementFailureKind::BackendUnavailable
        }
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

    use super::{
        DeviceSuitability, ExecutionDeviceClass, ExecutionUnitId, ExecutionUnitKind,
        InMemoryComputeService, JobCompletionClass, JobExecutionPath, JobLifecycleState,
        JobSubmissionMeta, MultiWorkerComputeService, PlacementFailureKind, PlacementSuitability,
        SchedulerConfig, WorkerAvailability, WorkerDispatchOutcome, WorkerRuntimeStatus,
    };

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
        pack_with_name("test_pack", world_backend, slots)
    }

    fn pack_with_name(
        pack_name: &'static str,
        world_backend: BackendComponentId,
        slots: Vec<ModelSlotProvenance>,
    ) -> Arc<dyn BackendPack> {
        let mut meta = BackendPackMeta {
            schema_version: 1,
            pack_name,
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

    fn service_with_pack_and_scheduler(
        pack: Arc<dyn BackendPack>,
        scheduler: SchedulerConfig,
    ) -> InMemoryComputeService {
        let backend =
            ComputePipelineBackend::new(pack, FusionConfig::default(), LimitsConfig::default());
        InMemoryComputeService::with_scheduler(backend, scheduler)
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
                gate: Default::default(),
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

    #[test]
    fn scheduler_cycle_respects_bounded_dispatch_capacity() {
        let mut service = service_with_pack_and_scheduler(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            SchedulerConfig {
                max_concurrent_jobs: 2,
                execution_path: JobExecutionPath::LocalCanonical,
            },
        );
        for t in 10..13 {
            let mut request = valid_request();
            request.input.t = t;
            service.submit(
                request,
                JobSubmissionMeta {
                    submitted_at_unix_ms: t,
                    submitted_by: Some("scheduler".to_string()),
                },
            );
        }
        let completed = service
            .run_scheduler_cycle(2)
            .expect("scheduler cycle should run");
        assert_eq!(completed.len(), 2);
        assert_eq!(service.queue_len(), 1);
        let snapshot = service.scheduler_snapshot();
        assert_eq!(snapshot.max_concurrent_jobs, 2);
        assert_eq!(snapshot.execution_path, JobExecutionPath::LocalCanonical);
    }

    #[test]
    fn budget_exceeded_execution_is_classified_as_timed_out() {
        let mut service = service_with_pack(pack_with(BackendComponentId::ToyV1, Vec::new()));
        let mut request = valid_request();
        request.budget.global_work_units = 1;
        request.budget.world_units = 1;
        let job_id = service
            .submit(
                request,
                JobSubmissionMeta {
                    submitted_at_unix_ms: 7,
                    submitted_by: Some("timeout".to_string()),
                },
            )
            .job
            .id;
        let record = service
            .run_next()
            .expect("scheduler run should succeed")
            .expect("job should exist");
        assert_eq!(record.job.id, job_id);
        assert_eq!(record.state, JobLifecycleState::TimedOut);
        assert_eq!(
            record
                .execution_failure
                .as_ref()
                .expect("failure should be set")
                .kind,
            CanonicalFailureKind::Timeout
        );
    }

    #[test]
    fn worker_launch_failure_is_reported_as_structured_execution_error() {
        let previous = std::env::var("UCF_WORKER_BIN").ok();
        std::env::set_var("UCF_WORKER_BIN", "definitely-missing-ucf-worker-binary");
        let mut service =
            InMemoryComputeService::new_worker(42, 1).expect("worker backend should construct");
        let job_id = service
            .submit(
                valid_request(),
                JobSubmissionMeta {
                    submitted_at_unix_ms: 8,
                    submitted_by: Some("worker".to_string()),
                },
            )
            .job
            .id;
        let record = service
            .run_next()
            .expect("run_next should map errors into lifecycle record")
            .expect("job record should be available");
        assert_eq!(record.job.id, job_id);
        assert_eq!(record.state, JobLifecycleState::Failed);
        let failure = record
            .execution_failure
            .as_ref()
            .expect("worker spawn failure should be mapped");
        assert_eq!(failure.kind, CanonicalFailureKind::ExecutionError);
        assert!(failure.detail.contains("spawn worker failed"));
        assert_eq!(record.execution_path, JobExecutionPath::WorkerIpc);
        assert!(service
            .lifecycle_events()
            .iter()
            .filter(|event| event.job_id == job_id)
            .any(|event| event
                .detail
                .as_deref()
                .unwrap_or_default()
                .contains("execution_path=worker_ipc")));

        match previous {
            Some(value) => std::env::set_var("UCF_WORKER_BIN", value),
            None => std::env::remove_var("UCF_WORKER_BIN"),
        }
    }

    #[test]
    fn smoke_lifecycle_accounting_and_provenance_are_populated() {
        let mut service = service_with_pack(pack_with(BackendComponentId::ToyV1, Vec::new()));
        let job_id = service
            .submit(
                valid_request(),
                JobSubmissionMeta {
                    submitted_at_unix_ms: 100,
                    submitted_by: Some("smoke".to_string()),
                },
            )
            .job
            .id;
        let record = service
            .run_next()
            .expect("run_next should succeed")
            .expect("job must exist");
        assert_eq!(record.job.id, job_id);
        assert!(record.accounting.started_at_unix_ms.is_some());
        assert!(record.accounting.finished_at_unix_ms.is_some());
        assert!(record.accounting.execution_duration_micros.is_some());
        assert!(record.accounting.total_duration_ms.is_some());
        assert!(record.accounting.work_summary.is_some());
        assert!(record.accounting.pipeline_state.is_some());
        assert!(record.accounting.stage_order.is_some());
        assert!(!record.accounting.executed_stages.is_empty());
        assert_eq!(
            record.accounting.execution_path,
            JobExecutionPath::LocalCanonical
        );
        assert_eq!(record.accounting.job_id, job_id);
    }

    #[test]
    fn rejection_sets_rejected_completion_class() {
        let mut service = service_with_pack(pack_with(BackendComponentId::ToyV1, Vec::new()));
        let mut request = valid_request();
        request.input.t = 0;
        let record = service.submit(
            request,
            JobSubmissionMeta {
                submitted_at_unix_ms: 9,
                submitted_by: None,
            },
        );
        assert_eq!(record.state, JobLifecycleState::Rejected);
        assert_eq!(
            record.accounting.completion_class,
            JobCompletionClass::RejectedBeforeExecution
        );
        assert_eq!(
            record.accounting.failure_kind,
            Some(CanonicalFailureKind::InvalidInput)
        );
        let rejected_event = service
            .lifecycle_events()
            .iter()
            .find(|event| event.state == JobLifecycleState::Rejected)
            .expect("missing rejection event");
        assert_eq!(
            rejected_event.completion_class,
            Some(JobCompletionClass::RejectedBeforeExecution)
        );
    }

    #[test]
    fn integration_onboarding_reference_backend_via_service_keeps_pipeline_surface() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();
        let backend = match crate::build_onboarding_reference_backend(13) {
            Ok(backend) => backend,
            Err(ComputeError::BackendDisabled) | Err(ComputeError::InvalidInput { .. }) => return,
            Err(other) => panic!("unexpected onboarding backend init error: {other:?}"),
        };
        let mut service = InMemoryComputeService::new(backend);
        let record = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 12,
                submitted_by: Some("integration".to_string()),
            },
        );
        assert_eq!(record.state, JobLifecycleState::Queued);
        let completed = service
            .run_next()
            .expect("run_next should execute")
            .expect("queued job must run");
        let result = completed
            .result
            .as_ref()
            .expect("onboarding backend should return canonical result");
        assert_eq!(result.stage_order, crate::CANONICAL_STAGE_SEQUENCE);
        assert_eq!(
            completed.accounting.stage_order,
            Some(crate::CANONICAL_STAGE_SEQUENCE)
        );
        assert_eq!(
            completed.accounting.executed_stages,
            result.executed_stages.clone()
        );
        assert_eq!(completed.accounting.pipeline_state, Some(result.state));
    }

    #[test]
    fn multi_worker_service_runs_job_locally() {
        let backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(backend, 1);
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 13,
                submitted_by: Some("local".to_string()),
            },
            None,
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job_id).expect("job result must exist");
        assert_eq!(record.placement.unit_kind, ExecutionUnitKind::Local);
        assert_eq!(record.placement.device_class, ExecutionDeviceClass::Cpu);
        assert_eq!(
            record.placement.execution_path,
            JobExecutionPath::LocalCanonical
        );
        assert_eq!(record.placement.suitability, PlacementSuitability::Suitable);
        assert_eq!(
            record.placement.device_suitability,
            DeviceSuitability::Suitable
        );
        assert!(!record.placement.considered.is_empty());
        assert!(record.result.is_some());
    }

    #[test]
    fn multi_worker_service_runs_job_on_secondary_worker() {
        let backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(backend, 1);
        let worker_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let worker_id = service.register_worker_backend("secondary-a", worker_backend, 1);
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 14,
                submitted_by: Some("remote".to_string()),
            },
            Some(worker_id.clone()),
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job_id).expect("job result must exist");
        assert_eq!(record.placement.unit_id, worker_id);
        assert_eq!(record.placement.unit_kind, ExecutionUnitKind::Worker);
        assert_eq!(record.placement.device_class, ExecutionDeviceClass::Worker);
        assert_eq!(record.placement.suitability, PlacementSuitability::Suitable);
        assert_eq!(
            record.placement.device_suitability,
            DeviceSuitability::Suitable
        );
        assert_eq!(
            record.placement.device_preference,
            Some(ExecutionDeviceClass::Worker)
        );
        assert!(record.placement.device_preference_met);
        assert_eq!(
            record.worker_dispatch_outcome,
            Some(WorkerDispatchOutcome::Completed)
        );
        let result = record
            .result
            .as_ref()
            .expect("canonical result should exist");
        assert_eq!(result.stage_order, crate::CANONICAL_STAGE_SEQUENCE);
    }

    #[test]
    fn multi_worker_unavailable_and_dispatch_failure_are_structured() {
        let backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(backend, 1);
        let worker_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let worker_id = service.register_worker_backend("secondary-b", worker_backend, 1);
        service.set_worker_availability(&worker_id, WorkerAvailability::Unavailable);
        let unavailable_job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 15,
                submitted_by: Some("remote".to_string()),
            },
            Some(worker_id.clone()),
        );
        let dispatch_failure_job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 16,
                submitted_by: Some("remote".to_string()),
            },
            Some(ExecutionUnitId("missing-worker".to_string())),
        );
        service.run_scheduler_cycle(2);
        let unavailable = service
            .job(unavailable_job)
            .expect("unavailable record should exist");
        assert_eq!(
            unavailable.worker_dispatch_outcome,
            Some(WorkerDispatchOutcome::Unavailable)
        );
        assert_eq!(
            unavailable.placement_failure,
            Some(PlacementFailureKind::DeviceUnavailable)
        );
        assert!(unavailable
            .execution_failure
            .as_ref()
            .expect("failure expected")
            .detail
            .contains("worker placement failed"));

        let dispatch_failure = service
            .job(dispatch_failure_job)
            .expect("dispatch failure record should exist");
        assert_eq!(
            dispatch_failure.worker_dispatch_outcome,
            Some(WorkerDispatchOutcome::DispatchFailure)
        );
        assert_eq!(
            dispatch_failure.placement_failure,
            Some(PlacementFailureKind::WorkerPlacementFailed)
        );
        assert!(dispatch_failure
            .execution_failure
            .as_ref()
            .expect("failure expected")
            .detail
            .contains("worker placement failed"));
    }

    #[test]
    fn multi_worker_prefers_burn_then_candle_fallback_with_provenance() {
        let local_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::CandleJepaV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(local_backend, 1);
        let burn_worker = ComputePipelineBackend::new(
            pack_with(BackendComponentId::BurnJepaV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        service.register_worker_backend("burn-worker", burn_worker, 1);

        let burn_job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 17,
                submitted_by: Some("placement".to_string()),
            },
            None,
        );
        service.run_scheduler_cycle(1);
        let burn_record = service.job(burn_job).expect("burn result");
        if burn_record.worker_dispatch_outcome == Some(WorkerDispatchOutcome::RedispatchedLocal) {
            assert!(burn_record.placement.degraded_fallback);
            assert!(burn_record.placement.reason.contains("redispatched"));
        } else {
            assert!(!burn_record.placement.degraded_fallback);
            assert!(burn_record.placement.reason.contains("burn"));
        }

        for unit in service.execution_units() {
            if unit.id.0 == "burn-worker" {
                service.set_worker_availability(&unit.id, WorkerAvailability::Unavailable);
            }
        }
        let candle_job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 18,
                submitted_by: Some("placement".to_string()),
            },
            None,
        );
        service.run_scheduler_cycle(1);
        let candle_record = service.job(candle_job).expect("candle result");
        assert!(candle_record.placement.degraded_fallback);
        assert!(candle_record.placement.reason.contains("fallback"));
    }

    #[test]
    fn requested_incompatible_worker_is_rejected_with_structured_failure() {
        let local_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(local_backend, 1);
        let disabled_world = vec![ModelSlotProvenance {
            slot: ModelSlot::WorldJepa,
            stage: "world",
            required_for_pack: true,
            status: SlotRuntimeStatus::Incompatible,
            code: Some(ArtifactFailureCode::ArtifactIncompatible),
            detail: Some("slot incompatible for test".to_string()),
            resolved_path: None,
            hash_prefix: None,
            contract_version: Some("v1".to_string()),
            format: None,
            gate: Default::default(),
        }];
        let incompatible_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, disabled_world),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let worker_id =
            service.register_worker_backend("incompatible-worker", incompatible_backend, 1);
        let job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 19,
                submitted_by: Some("placement".to_string()),
            },
            Some(worker_id),
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job).expect("record");
        assert_eq!(
            record.placement_failure,
            Some(PlacementFailureKind::BackendIncompatible)
        );
        assert_eq!(record.result, None);
    }

    #[test]
    fn requested_worker_tracks_device_provenance_when_unavailable() {
        let local_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(local_backend, 1);
        let worker_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let worker_id = service.register_worker_backend("secondary-c", worker_backend, 1);
        service.set_worker_availability(&worker_id, WorkerAvailability::Unavailable);
        let job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 20,
                submitted_by: Some("device".to_string()),
            },
            Some(worker_id),
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job).expect("record");
        assert_eq!(record.placement.device_class, ExecutionDeviceClass::Worker);
        assert_eq!(
            record.placement.device_suitability,
            DeviceSuitability::Unavailable
        );
        assert_eq!(
            record.placement.device_preference,
            Some(ExecutionDeviceClass::Worker)
        );
        assert!(!record.placement.device_preference_met);
    }

    #[test]
    fn backend_device_incompatibility_is_reported_for_local_worker_lane() {
        let local_backend = ComputePipelineBackend::new(
            pack_with_name("worker_v1", BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(local_backend, 1);
        let job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 21,
                submitted_by: Some("device".to_string()),
            },
            None,
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job).expect("record");
        assert_eq!(
            record.placement_failure,
            Some(PlacementFailureKind::BackendIncompatible)
        );
        assert!(record
            .placement
            .considered
            .iter()
            .any(|candidate| candidate.device_suitability == DeviceSuitability::Unsuitable));
    }

    #[test]
    fn remote_execution_failure_can_redispatch_to_local_with_provenance() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var("UCF_WORKER_BIN").ok();
        std::env::set_var("UCF_WORKER_BIN", "definitely-missing-ucf-worker-binary");
        let local_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(local_backend, 1);
        service
            .register_worker("remote-failing", 99, 1)
            .expect("register worker");
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 22,
                submitted_by: Some("fallback".to_string()),
            },
            None,
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job_id).expect("record");
        assert_eq!(
            record.worker_dispatch_outcome,
            Some(WorkerDispatchOutcome::RedispatchedLocal)
        );
        assert!(record.provenance.redispatched_to_local);
        assert!(record.provenance.was_remote);
        assert_eq!(record.provenance.completed_unit.0, "local");
        assert!(record.result.is_some());
        match previous {
            Some(value) => std::env::set_var("UCF_WORKER_BIN", value),
            None => std::env::remove_var("UCF_WORKER_BIN"),
        }
    }

    #[test]
    fn worker_snapshot_reports_unhealthy_after_repeated_failures() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var("UCF_WORKER_BIN").ok();
        std::env::set_var("UCF_WORKER_BIN", "definitely-missing-ucf-worker-binary");
        let local_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(local_backend, 1);
        let worker_id = service
            .register_worker("remote-unhealthy", 100, 1)
            .expect("register worker");
        for t in 30..33 {
            let job_id = service.submit(
                valid_request(),
                JobSubmissionMeta {
                    submitted_at_unix_ms: t,
                    submitted_by: Some("health".to_string()),
                },
                Some(worker_id.clone()),
            );
            service.run_scheduler_cycle(1);
            let record = service.job(job_id).expect("record");
            assert_eq!(record.state, JobLifecycleState::Failed);
        }
        let snapshot = service
            .execution_units()
            .into_iter()
            .find(|unit| unit.id == worker_id)
            .expect("worker snapshot");
        assert_eq!(snapshot.status, WorkerRuntimeStatus::Unhealthy);
        assert!(snapshot.last_error.is_some());
        match previous {
            Some(value) => std::env::set_var("UCF_WORKER_BIN", value),
            None => std::env::remove_var("UCF_WORKER_BIN"),
        }
    }
}
