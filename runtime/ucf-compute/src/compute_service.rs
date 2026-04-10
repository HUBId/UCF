use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::backend_pack::{
    BackendPackConfig, BackendPackFactory, BackendPackKind, ModelSlotProvenance,
};
use crate::pipeline::{
    BackendExecutionLane, CanonicalAdmissionDecision, CanonicalFailureKind,
    CanonicalHotspotSummary, CanonicalPipelineFailure, CanonicalPipelineRequest,
    CanonicalPipelineResult, CanonicalPipelineState, CanonicalStageId, CanonicalStageProfile,
    CanonicalWorkSummary, ComputePipelineBackend, FusionConfig, LimitsConfig,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceClass {
    Light,
    Standard,
    Heavy,
}

impl ResourceClass {
    fn classify(request: &CanonicalPipelineRequest) -> Self {
        let budget = request.budget.global_work_units;
        if budget <= 32_768 {
            Self::Light
        } else if budget <= 98_304 {
            Self::Standard
        } else {
            Self::Heavy
        }
    }

    fn capacity_weight(self) -> usize {
        match self {
            Self::Light => 1,
            Self::Standard => 2,
            Self::Heavy => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkCostProvenance {
    EstimatedFromBudget,
    RuntimeMeasured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkCostTension {
    Nominal,
    ExpensiveButSuccessful,
    ExpensiveAndDegraded,
    RetriedWithAdditionalCost,
    LowCostButBlocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidatedWorkCostSummary {
    pub provenance: WorkCostProvenance,
    pub resource_class: ResourceClass,
    pub estimated_total_work_units: u64,
    pub runtime_consumed_work_units: Option<u64>,
    pub runtime_remaining_work_units: Option<u64>,
    pub dominant_stage: Option<CanonicalStageId>,
    pub dominant_stage_share_bps: Option<u16>,
    pub degraded_stage_count: u8,
    pub retry_attempts: u8,
    pub redispatched_to_local: bool,
    pub queue_deferred_by_capacity: bool,
    pub pressure: CapacityPressure,
    pub queue_disposition: CapacityQueueDisposition,
    pub tension: WorkCostTension,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityPressure {
    Healthy,
    Constrained,
    Saturated,
    Backpressured,
    TemporarilyUnschedulable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityQueueDisposition {
    None,
    QueuedDueToCapacity,
    DeferredDueToCapacity,
    DegradedPlacementDueToPressure,
    RejectedDueToCapacity,
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
    pub work_cost_summary: Option<ConsolidatedWorkCostSummary>,
    pub stage_profiles: Vec<CanonicalStageProfile>,
    pub hotspot_summary: Option<CanonicalHotspotSummary>,
    pub pipeline_state: Option<CanonicalPipelineState>,
    pub stage_order: Option<[CanonicalStageId; 4]>,
    pub executed_stages: Vec<CanonicalStageId>,
    pub model_slots: Vec<ModelSlotProvenance>,
    pub execution_path: JobExecutionPath,
    pub execution_lane: BackendExecutionLane,
    pub resource_class: ResourceClass,
    pub capacity_queue_disposition: CapacityQueueDisposition,
    pub capacity_pressure: CapacityPressure,
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

impl SchedulerConfig {
    fn capacity_limit_units(self) -> usize {
        self.max_concurrent_jobs.max(1).saturating_mul(2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerSnapshot {
    pub max_concurrent_jobs: usize,
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub execution_path: JobExecutionPath,
    pub capacity_limit_units: usize,
    pub used_capacity_units: usize,
    pub free_capacity_units: usize,
    pub pressure: CapacityPressure,
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
        let resource_class = ResourceClass::classify(&request);
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
                work_cost_summary: Some(estimated_work_cost_summary(
                    &request,
                    resource_class,
                    CapacityPressure::Healthy,
                    CapacityQueueDisposition::None,
                )),
                stage_profiles: Vec::new(),
                hotspot_summary: None,
                pipeline_state: None,
                stage_order: None,
                executed_stages: Vec::new(),
                model_slots: Vec::new(),
                execution_path: self.scheduler.execution_path,
                execution_lane: self.backend.execution_lane(),
                resource_class,
                capacity_queue_disposition: CapacityQueueDisposition::None,
                capacity_pressure: CapacityPressure::Healthy,
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
                record.accounting.capacity_queue_disposition =
                    CapacityQueueDisposition::QueuedDueToCapacity;
                record.accounting.capacity_pressure = if self.queue.is_empty() {
                    CapacityPressure::Healthy
                } else {
                    CapacityPressure::Saturated
                };
                record.accounting.work_cost_summary = Some(estimated_work_cost_summary(
                    &record.job.request,
                    record.accounting.resource_class,
                    record.accounting.capacity_pressure,
                    record.accounting.capacity_queue_disposition,
                ));
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
                record.accounting.capacity_queue_disposition = CapacityQueueDisposition::None;
                record.accounting.work_cost_summary = Some(estimated_work_cost_summary(
                    &record.job.request,
                    record.accounting.resource_class,
                    record.accounting.capacity_pressure,
                    record.accounting.capacity_queue_disposition,
                ));
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

        let used_capacity_units = self
            .running
            .iter()
            .filter_map(|running_job| self.jobs.get(running_job))
            .map(|job| job.accounting.resource_class.capacity_weight())
            .sum::<usize>();
        let capacity_limit_units = self.scheduler.capacity_limit_units();
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
        record.accounting.capacity_pressure = capacity_pressure_for(
            used_capacity_units,
            capacity_limit_units,
            !self.queue.is_empty(),
        );
        if let Some(canonical_result) = record.result.as_ref() {
            record.accounting.work_summary = Some(canonical_result.diagnostics.work);
            record.accounting.stage_profiles = canonical_result.diagnostics.stage_profiles.clone();
            record.accounting.hotspot_summary = Some(canonical_result.diagnostics.hotspots);
            record.accounting.pipeline_state = Some(canonical_result.state);
            record.accounting.stage_order = Some(canonical_result.stage_order);
            record.accounting.executed_stages = canonical_result.executed_stages.clone();
            record.accounting.model_slots = canonical_result.model_slots.clone();
        }
        record.accounting.work_cost_summary = Some(runtime_work_cost_summary(
            &record.job.request,
            record.accounting.resource_class,
            record.accounting.work_summary,
            record.accounting.hotspot_summary,
            record.accounting.completion_class,
            record.accounting.capacity_pressure,
            record.accounting.capacity_queue_disposition,
            1,
            false,
        ));
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
        let used_capacity_units = self
            .running
            .iter()
            .filter_map(|job_id| self.jobs.get(job_id))
            .map(|record| record.accounting.resource_class.capacity_weight())
            .sum::<usize>();
        let capacity_limit_units = self.scheduler.capacity_limit_units();
        SchedulerSnapshot {
            max_concurrent_jobs: self.scheduler.max_concurrent_jobs,
            queued_jobs: self.queue.len(),
            running_jobs: self.running.len(),
            execution_path: self.scheduler.execution_path,
            capacity_limit_units,
            used_capacity_units,
            free_capacity_units: capacity_limit_units.saturating_sub(used_capacity_units),
            pressure: capacity_pressure_for(
                used_capacity_units,
                capacity_limit_units,
                !self.queue.is_empty(),
            ),
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

fn classify_worker_failure(failure: &CanonicalPipelineFailure) -> WorkerFailureKind {
    let detail = failure.detail.as_str();
    if detail.contains("worker_dispatch_failed_before_execution") {
        WorkerFailureKind::DispatchFailedBeforeExecution
    } else if detail.contains("worker_transport_failure") || detail.contains("transport failure") {
        WorkerFailureKind::TransportFailure
    } else if detail.contains("worker_unavailable_or_stale")
        || detail.contains("worker unavailable")
    {
        WorkerFailureKind::WorkerUnavailableOrStale
    } else if detail.contains("worker_execution_crashed")
        || failure.kind == CanonicalFailureKind::Timeout
    {
        WorkerFailureKind::WorkerExecutionCrashed
    } else if detail.contains("worker_structured_execution_failure") {
        WorkerFailureKind::StructuredExecutionFailure
    } else {
        WorkerFailureKind::TerminalComputeExecutionFailure
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
pub enum WorkerClass {
    LocalPrimary,
    RemoteSecondary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRegistryRole {
    Primary,
    Secondary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerAvailability {
    Available,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerDispatchOutcome {
    Unavailable,
    Deferred,
    DispatchFailure,
    TransportFailure,
    ExecutionFailure,
    Timeout,
    Completed,
    RedispatchedLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerFailureKind {
    DispatchFailedBeforeExecution,
    TransportFailure,
    WorkerUnavailableOrStale,
    WorkerExecutionCrashed,
    StructuredExecutionFailure,
    TerminalComputeExecutionFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRecoveryKind {
    RetrySameWorker,
    RedispatchAlternateWorker,
    LocalFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InFlightCoordinationState {
    Queued,
    Dispatching,
    Running,
    AwaitingWorkerOutcome,
    RetryPending,
    RedispatchPending,
    Uncertain,
    Stale,
    Completed,
    Failed,
    TimedOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinationFreshness {
    Current,
    Stale,
    Uncertain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinationIssueKind {
    StaleWorkerOwnership,
    MissingWorkerOutcome,
    OrphanedInFlightJob,
    RecoveredCoordinationState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoverySignal {
    SafeToRedispatch,
    UnsafeUncertainPriorAttempt,
    AwaitWorkerOutcome,
    RecoveryDecisionRequired,
    Terminal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRetrySummary {
    pub attempts: u8,
    pub retries_exhausted: bool,
    pub uncertain_prior_attempt_outcome: bool,
    pub recovered_by: Option<WorkerRecoveryKind>,
    pub last_failure_kind: Option<WorkerFailureKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRuntimeStatus {
    Known,
    Ready,
    Busy,
    Constrained,
    Saturated,
    Backpressured,
    Degraded,
    Unavailable,
    Stale,
    Unknown,
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
    pub resource_class: ResourceClass,
    pub capacity_pressure: CapacityPressure,
    pub distributed: DistributedPlacementSummary,
    pub reason: String,
    pub considered: Vec<PlacementCandidateAssessment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedPlacementState {
    AdmissibleAndPlaceable,
    AdmissiblePlaceableOnSubset,
    AdmissibleButCurrentlyUnschedulable,
    AdmissibleDegradedOnly,
    BlockedIncompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedPlacementLocality {
    None,
    LocalOnly,
    RemoteOnly,
    LocalAndRemote,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedPlacementSummary {
    pub state: DistributedPlacementState,
    pub locality: DistributedPlacementLocality,
    pub admissible_units: Vec<ExecutionUnitId>,
    pub placeable_units: Vec<ExecutionUnitId>,
    pub degraded_fallback_possible: bool,
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
    CurrentlyUnschedulable,
    CapacityRejected,
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
    pub worker_class: WorkerClass,
    pub registry_role: WorkerRegistryRole,
    pub device_class: ExecutionDeviceClass,
    pub lane: BackendExecutionLane,
    pub runtime_status: WorkerRuntimeStatus,
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
    pub work_cost_summary: ConsolidatedWorkCostSummary,
    pub placement: ExecutionPlacement,
    pub worker_dispatch_outcome: Option<WorkerDispatchOutcome>,
    pub placement_failure: Option<PlacementFailureKind>,
    pub capacity_disposition: CapacityQueueDisposition,
    pub provenance: WorkerExecutionProvenance,
    pub retry_summary: WorkerRetrySummary,
    pub coordination: JobCoordinationSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerExecutionProvenance {
    pub selected_unit: ExecutionUnitId,
    pub completed_unit: ExecutionUnitId,
    pub was_remote: bool,
    pub redispatched_to_local: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobCoordinationSnapshot {
    pub state: InFlightCoordinationState,
    pub last_in_flight_state: Option<InFlightCoordinationState>,
    pub owner: Option<ExecutionUnitId>,
    pub owner_kind: Option<ExecutionUnitKind>,
    pub owner_last_contact_at_unix_ms: Option<u64>,
    pub freshness: CoordinationFreshness,
    pub issue: Option<CoordinationIssueKind>,
    pub recovery_signal: RecoverySignal,
}

struct TerminalCoordinationInput {
    state: JobLifecycleState,
    last_in_flight_state: Option<InFlightCoordinationState>,
    owner: ExecutionUnitId,
    owner_kind: ExecutionUnitKind,
    owner_last_contact_at_unix_ms: Option<u64>,
    issue: Option<CoordinationIssueKind>,
    recovered: bool,
    uncertain: bool,
    awaiting_outcome: bool,
}

impl WorkerRetrySummary {
    fn no_retry() -> Self {
        Self {
            attempts: 1,
            retries_exhausted: false,
            uncertain_prior_attempt_outcome: false,
            recovered_by: None,
            last_failure_kind: None,
        }
    }
}

impl JobCoordinationSnapshot {
    fn queued() -> Self {
        Self {
            state: InFlightCoordinationState::Queued,
            last_in_flight_state: Some(InFlightCoordinationState::Queued),
            owner: None,
            owner_kind: None,
            owner_last_contact_at_unix_ms: None,
            freshness: CoordinationFreshness::Current,
            issue: None,
            recovery_signal: RecoverySignal::SafeToRedispatch,
        }
    }

    fn stale_without_dispatch(owner: ExecutionUnitId, owner_kind: ExecutionUnitKind) -> Self {
        Self {
            state: InFlightCoordinationState::Stale,
            last_in_flight_state: Some(InFlightCoordinationState::Dispatching),
            owner: Some(owner),
            owner_kind: Some(owner_kind),
            owner_last_contact_at_unix_ms: None,
            freshness: CoordinationFreshness::Stale,
            issue: Some(CoordinationIssueKind::StaleWorkerOwnership),
            recovery_signal: RecoverySignal::RecoveryDecisionRequired,
        }
    }

    fn from_terminal(input: TerminalCoordinationInput) -> Self {
        let terminal = match input.state {
            JobLifecycleState::Completed => InFlightCoordinationState::Completed,
            JobLifecycleState::TimedOut => InFlightCoordinationState::TimedOut,
            _ => InFlightCoordinationState::Failed,
        };
        let freshness = if input.uncertain {
            CoordinationFreshness::Uncertain
        } else if input.issue == Some(CoordinationIssueKind::StaleWorkerOwnership) {
            CoordinationFreshness::Stale
        } else {
            CoordinationFreshness::Current
        };
        let recovery_signal =
            if input.recovered || matches!(input.state, JobLifecycleState::Completed) {
                RecoverySignal::Terminal
            } else if input.awaiting_outcome {
                RecoverySignal::AwaitWorkerOutcome
            } else if input.uncertain {
                RecoverySignal::UnsafeUncertainPriorAttempt
            } else if input.issue.is_some() {
                RecoverySignal::RecoveryDecisionRequired
            } else {
                RecoverySignal::Terminal
            };
        Self {
            state: terminal,
            last_in_flight_state: input.last_in_flight_state,
            owner: Some(input.owner),
            owner_kind: Some(input.owner_kind),
            owner_last_contact_at_unix_ms: input.owner_last_contact_at_unix_ms,
            freshness,
            issue: if input.recovered {
                Some(CoordinationIssueKind::RecoveredCoordinationState)
            } else {
                input.issue
            },
            recovery_signal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionUnitSnapshot {
    pub id: ExecutionUnitId,
    pub kind: ExecutionUnitKind,
    pub worker_class: WorkerClass,
    pub registry_role: WorkerRegistryRole,
    pub availability: WorkerAvailability,
    pub status: WorkerRuntimeStatus,
    pub max_parallel_jobs: usize,
    pub active_jobs: usize,
    pub used_capacity_units: usize,
    pub free_capacity_units: usize,
    pub capacity_pressure: CapacityPressure,
    pub consecutive_failures: u32,
    pub last_job_id: Option<JobId>,
    pub last_dispatch_outcome: Option<WorkerDispatchOutcome>,
    pub last_error: Option<String>,
    pub last_used_at_unix_ms: Option<u64>,
    pub last_health_contact_at_unix_ms: Option<u64>,
    pub quarantine_until_unix_ms: Option<u64>,
    pub placement_eligible: bool,
    pub degradation_state: DistributedDegradationState,
    pub recovered_at_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedPressureSnapshot {
    pub service_pressure: CapacityPressure,
    pub queued_jobs: usize,
    pub queued_light_jobs: usize,
    pub queued_standard_jobs: usize,
    pub queued_heavy_jobs: usize,
    pub saturated_units: Vec<ExecutionUnitId>,
    pub constrained_units: Vec<ExecutionUnitId>,
    pub backpressured_units: Vec<ExecutionUnitId>,
    pub temporarily_unschedulable_units: Vec<ExecutionUnitId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedDegradationState {
    Healthy,
    PartiallyDegraded,
    ConstrainedButServiceable,
    RecoveryInProgress,
    UnrecoverableUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedRecoverySnapshot {
    pub state: DistributedDegradationState,
    pub total_units: usize,
    pub healthy_units: usize,
    pub constrained_serviceable_units: usize,
    pub degraded_units: usize,
    pub recovering_units: usize,
    pub unavailable_units: usize,
    pub placement_eligible_units: Vec<ExecutionUnitId>,
    pub excluded_units: Vec<ExecutionUnitId>,
    pub recovered_units: Vec<ExecutionUnitId>,
    pub queued_jobs: usize,
    pub uncertain_jobs: usize,
    pub recovery_required_jobs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InFlightJobSnapshot {
    pub job_id: JobId,
    pub state: InFlightCoordinationState,
    pub owner: Option<ExecutionUnitId>,
    pub owner_kind: Option<ExecutionUnitKind>,
    pub owner_last_contact_at_unix_ms: Option<u64>,
    pub freshness: CoordinationFreshness,
    pub issue: Option<CoordinationIssueKind>,
    pub recovery_signal: RecoverySignal,
}

struct ExecutionUnit {
    id: ExecutionUnitId,
    kind: ExecutionUnitKind,
    worker_class: WorkerClass,
    registry_role: WorkerRegistryRole,
    availability: WorkerAvailability,
    max_parallel_jobs: usize,
    active_jobs: usize,
    used_capacity_units: usize,
    consecutive_failures: u32,
    last_job_id: Option<JobId>,
    last_dispatch_outcome: Option<WorkerDispatchOutcome>,
    last_error: Option<String>,
    last_used_at_unix_ms: Option<u64>,
    last_health_contact_at_unix_ms: Option<u64>,
    quarantine_until_unix_ms: Option<u64>,
    recovered_at_unix_ms: Option<u64>,
    service: InMemoryComputeService,
}

#[derive(Debug, Clone)]
struct QueuedJob {
    id: JobId,
    request: CanonicalPipelineRequest,
    meta: JobSubmissionMeta,
    resource_class: ResourceClass,
    requested_unit: Option<ExecutionUnitId>,
    placement_attempts: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulingDecision {
    RunNow,
    QueueRequired,
    NotPlaceable(PlacementFailureKind),
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
    max_placement_attempts: u8,
}

impl MultiWorkerComputeService {
    pub fn new(local_backend: ComputePipelineBackend, max_parallel_jobs: usize) -> Self {
        let local = ExecutionUnit {
            id: ExecutionUnitId("local".to_string()),
            kind: ExecutionUnitKind::Local,
            worker_class: WorkerClass::LocalPrimary,
            registry_role: WorkerRegistryRole::Primary,
            availability: WorkerAvailability::Available,
            max_parallel_jobs: max_parallel_jobs.max(1),
            active_jobs: 0,
            used_capacity_units: 0,
            consecutive_failures: 0,
            last_job_id: None,
            last_dispatch_outcome: None,
            last_error: None,
            last_used_at_unix_ms: None,
            last_health_contact_at_unix_ms: Some(now_unix_ms()),
            quarantine_until_unix_ms: None,
            recovered_at_unix_ms: None,
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
            max_placement_attempts: 3,
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
            worker_class: WorkerClass::RemoteSecondary,
            registry_role: WorkerRegistryRole::Secondary,
            availability: WorkerAvailability::Available,
            max_parallel_jobs: max_parallel_jobs.max(1),
            active_jobs: 0,
            used_capacity_units: 0,
            consecutive_failures: 0,
            last_job_id: None,
            last_dispatch_outcome: None,
            last_error: None,
            last_used_at_unix_ms: None,
            last_health_contact_at_unix_ms: Some(now_unix_ms()),
            quarantine_until_unix_ms: None,
            recovered_at_unix_ms: None,
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
                let previous = unit.runtime_status();
                unit.availability = availability;
                unit.last_health_contact_at_unix_ms = Some(now_unix_ms());
                if availability == WorkerAvailability::Available {
                    unit.quarantine_until_unix_ms = None;
                    if matches!(
                        previous,
                        WorkerRuntimeStatus::Unavailable
                            | WorkerRuntimeStatus::Stale
                            | WorkerRuntimeStatus::Unhealthy
                            | WorkerRuntimeStatus::Degraded
                    ) {
                        unit.recovered_at_unix_ms = unit.last_health_contact_at_unix_ms;
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn set_worker_last_health_contact_for_test(
        &mut self,
        worker_id: &ExecutionUnitId,
        last_health_contact_at_unix_ms: Option<u64>,
    ) {
        for unit in &mut self.units {
            if &unit.id == worker_id {
                unit.last_health_contact_at_unix_ms = last_health_contact_at_unix_ms;
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
        let resource_class = ResourceClass::classify(&request);
        self.queue.push_back(QueuedJob {
            id,
            request,
            meta,
            resource_class,
            requested_unit,
            placement_attempts: 0,
        });
        id
    }

    pub fn run_scheduler_cycle(&mut self, max_jobs: usize) -> Vec<JobId> {
        let mut done = Vec::new();
        while done.len() < max_jobs.max(1) {
            let Some(job) = self.queue.pop_front() else {
                break;
            };
            let selection = match self.select_unit(
                &job.request,
                job.resource_class,
                job.requested_unit.clone(),
            ) {
                Some(selection) => selection,
                None => {
                    let scheduling = self.scheduling_decision(&job);
                    if scheduling == SchedulingDecision::QueueRequired
                        && job.requested_unit.is_none()
                        && job.placement_attempts < self.max_placement_attempts
                    {
                        let mut deferred = job.clone();
                        deferred.placement_attempts = deferred.placement_attempts.saturating_add(1);
                        self.queue.push_back(deferred);
                        let (placement, _, worker_dispatch_outcome) = self.rejected_placement(&job);
                        let record = MultiWorkerJobRecord {
                            id: job.id,
                            state: JobLifecycleState::Queued,
                            execution_failure: None,
                            result: None,
                            work_cost_summary: estimated_work_cost_summary(
                                &job.request,
                                job.resource_class,
                                CapacityPressure::TemporarilyUnschedulable,
                                CapacityQueueDisposition::DeferredDueToCapacity,
                            ),
                            placement,
                            worker_dispatch_outcome: worker_dispatch_outcome
                                .or(Some(WorkerDispatchOutcome::Deferred)),
                            placement_failure: Some(PlacementFailureKind::CurrentlyUnschedulable),
                            capacity_disposition: CapacityQueueDisposition::DeferredDueToCapacity,
                            provenance: WorkerExecutionProvenance {
                                selected_unit: ExecutionUnitId("deferred".to_string()),
                                completed_unit: ExecutionUnitId("deferred".to_string()),
                                was_remote: false,
                                redispatched_to_local: false,
                            },
                            retry_summary: WorkerRetrySummary::no_retry(),
                            coordination: JobCoordinationSnapshot::queued(),
                        };
                        done.push(record.id);
                        self.records.insert(record.id, record);
                        continue;
                    }
                    let (placement, placement_failure, worker_dispatch_outcome) =
                        self.rejected_placement(&job);
                    if let Some(requested_unit) = job.requested_unit.clone() {
                        let placement_unit_id = placement.unit_id.clone();
                        let placement_unit_kind = placement.unit_kind;
                        let record = MultiWorkerJobRecord {
                            id: job.id,
                            state: JobLifecycleState::Failed,
                            execution_failure: Some(CanonicalPipelineFailure {
                                kind: CanonicalFailureKind::ExecutionError,
                                stage: None,
                                detail: format!("worker placement failed: {}", requested_unit.0),
                            }),
                            result: None,
                            work_cost_summary: estimated_work_cost_summary(
                                &job.request,
                                job.resource_class,
                                CapacityPressure::Backpressured,
                                CapacityQueueDisposition::RejectedDueToCapacity,
                            ),
                            placement,
                            worker_dispatch_outcome,
                            placement_failure: Some(placement_failure),
                            capacity_disposition: CapacityQueueDisposition::RejectedDueToCapacity,
                            provenance: WorkerExecutionProvenance {
                                selected_unit: requested_unit.clone(),
                                completed_unit: requested_unit,
                                was_remote: true,
                                redispatched_to_local: false,
                            },
                            retry_summary: WorkerRetrySummary::no_retry(),
                            coordination: JobCoordinationSnapshot::stale_without_dispatch(
                                placement_unit_id,
                                placement_unit_kind,
                            ),
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
                            detail: match scheduling {
                                SchedulingDecision::NotPlaceable(
                                    PlacementFailureKind::BackendIncompatible,
                                )
                                | SchedulingDecision::NotPlaceable(
                                    PlacementFailureKind::BackendDeviceIncompatible,
                                ) => "no suitable backend/device for request".to_string(),
                                SchedulingDecision::NotPlaceable(
                                    PlacementFailureKind::CurrentlyUnschedulable,
                                )
                                | SchedulingDecision::QueueRequired => {
                                    "job remained unschedulable under current capacity".to_string()
                                }
                                SchedulingDecision::NotPlaceable(
                                    PlacementFailureKind::CapacityRejected,
                                ) => "job rejected due to resource-class capacity pressure"
                                    .to_string(),
                                _ => "no suitable backend".to_string(),
                            },
                        }),
                        result: None,
                        work_cost_summary: estimated_work_cost_summary(
                            &job.request,
                            job.resource_class,
                            CapacityPressure::Backpressured,
                            CapacityQueueDisposition::RejectedDueToCapacity,
                        ),
                        placement,
                        worker_dispatch_outcome,
                        placement_failure: Some(match scheduling {
                            SchedulingDecision::NotPlaceable(kind) => kind,
                            _ => placement_failure,
                        }),
                        capacity_disposition: CapacityQueueDisposition::RejectedDueToCapacity,
                        provenance: WorkerExecutionProvenance {
                            selected_unit: ExecutionUnitId("none".to_string()),
                            completed_unit: ExecutionUnitId("none".to_string()),
                            was_remote: false,
                            redispatched_to_local: false,
                        },
                        retry_summary: WorkerRetrySummary::no_retry(),
                        coordination: JobCoordinationSnapshot::queued(),
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

    pub fn in_flight_jobs(&self) -> Vec<InFlightJobSnapshot> {
        let mut snapshots = self
            .records
            .values()
            .filter(|record| {
                matches!(
                    record.coordination.state,
                    InFlightCoordinationState::Queued
                        | InFlightCoordinationState::Dispatching
                        | InFlightCoordinationState::Running
                        | InFlightCoordinationState::AwaitingWorkerOutcome
                        | InFlightCoordinationState::RetryPending
                        | InFlightCoordinationState::RedispatchPending
                        | InFlightCoordinationState::Uncertain
                        | InFlightCoordinationState::Stale
                )
            })
            .map(|record| InFlightJobSnapshot {
                job_id: record.id,
                state: record.coordination.state,
                owner: record.coordination.owner.clone(),
                owner_kind: record.coordination.owner_kind,
                owner_last_contact_at_unix_ms: record.coordination.owner_last_contact_at_unix_ms,
                freshness: record.coordination.freshness,
                issue: record.coordination.issue,
                recovery_signal: record.coordination.recovery_signal,
            })
            .collect::<Vec<_>>();
        snapshots.extend(self.queue.iter().map(|queued| InFlightJobSnapshot {
            job_id: queued.id,
            state: InFlightCoordinationState::Queued,
            owner: None,
            owner_kind: None,
            owner_last_contact_at_unix_ms: None,
            freshness: CoordinationFreshness::Current,
            issue: None,
            recovery_signal: RecoverySignal::SafeToRedispatch,
        }));
        snapshots.sort_by_key(|snapshot| snapshot.job_id);
        snapshots
    }

    pub fn execution_units(&self) -> Vec<ExecutionUnitSnapshot> {
        self.units
            .iter()
            .map(|unit| ExecutionUnitSnapshot {
                id: unit.id.clone(),
                kind: unit.kind,
                worker_class: unit.worker_class,
                registry_role: unit.registry_role,
                availability: unit.availability,
                status: unit.runtime_status(),
                max_parallel_jobs: unit.max_parallel_jobs,
                active_jobs: unit.active_jobs,
                used_capacity_units: unit.used_capacity_units,
                free_capacity_units: unit
                    .capacity_limit_units()
                    .saturating_sub(unit.used_capacity_units),
                capacity_pressure: unit.capacity_pressure(),
                consecutive_failures: unit.consecutive_failures,
                last_job_id: unit.last_job_id,
                last_dispatch_outcome: unit.last_dispatch_outcome,
                last_error: unit.last_error.clone(),
                last_used_at_unix_ms: unit.last_used_at_unix_ms,
                last_health_contact_at_unix_ms: unit.last_health_contact_at_unix_ms,
                quarantine_until_unix_ms: unit.quarantine_until_unix_ms,
                placement_eligible: unit.can_accept_dispatch(),
                degradation_state: unit.degradation_state(),
                recovered_at_unix_ms: unit.recovered_at_unix_ms,
            })
            .collect()
    }

    pub fn pressure_snapshot(&self) -> DistributedPressureSnapshot {
        let mut saturated_units = Vec::new();
        let mut constrained_units = Vec::new();
        let mut backpressured_units = Vec::new();
        let mut temporarily_unschedulable_units = Vec::new();
        for unit in &self.units {
            match unit.capacity_pressure() {
                CapacityPressure::Healthy => {}
                CapacityPressure::Constrained => constrained_units.push(unit.id.clone()),
                CapacityPressure::Saturated => saturated_units.push(unit.id.clone()),
                CapacityPressure::Backpressured => backpressured_units.push(unit.id.clone()),
                CapacityPressure::TemporarilyUnschedulable => {
                    temporarily_unschedulable_units.push(unit.id.clone())
                }
            }
        }
        let mut queued_light_jobs = 0usize;
        let mut queued_standard_jobs = 0usize;
        let mut queued_heavy_jobs = 0usize;
        for queued in &self.queue {
            match queued.resource_class {
                ResourceClass::Light => queued_light_jobs += 1,
                ResourceClass::Standard => queued_standard_jobs += 1,
                ResourceClass::Heavy => queued_heavy_jobs += 1,
            }
        }
        DistributedPressureSnapshot {
            service_pressure: if !self.queue.is_empty() && backpressured_units.is_empty() {
                CapacityPressure::TemporarilyUnschedulable
            } else {
                capacity_pressure_for(
                    self.units.iter().map(|u| u.used_capacity_units).sum(),
                    self.units
                        .iter()
                        .map(ExecutionUnit::capacity_limit_units)
                        .sum(),
                    !self.queue.is_empty(),
                )
            },
            queued_jobs: self.queue.len(),
            queued_light_jobs,
            queued_standard_jobs,
            queued_heavy_jobs,
            saturated_units,
            constrained_units,
            backpressured_units,
            temporarily_unschedulable_units,
        }
    }

    pub fn distributed_recovery_snapshot(&self) -> DistributedRecoverySnapshot {
        let mut healthy_units = 0usize;
        let mut constrained_serviceable_units = 0usize;
        let mut degraded_units = 0usize;
        let mut recovering_units = 0usize;
        let mut unavailable_units = 0usize;
        let mut placement_eligible_units = Vec::new();
        let mut excluded_units = Vec::new();
        let mut recovered_units = Vec::new();
        for unit in &self.units {
            let degradation = unit.degradation_state();
            match degradation {
                DistributedDegradationState::Healthy => healthy_units += 1,
                DistributedDegradationState::ConstrainedButServiceable => {
                    constrained_serviceable_units += 1
                }
                DistributedDegradationState::PartiallyDegraded => degraded_units += 1,
                DistributedDegradationState::RecoveryInProgress => recovering_units += 1,
                DistributedDegradationState::UnrecoverableUnavailable => unavailable_units += 1,
            }
            if unit.can_accept_dispatch() {
                placement_eligible_units.push(unit.id.clone());
            } else {
                excluded_units.push(unit.id.clone());
            }
            if unit.recovered_at_unix_ms.is_some() {
                recovered_units.push(unit.id.clone());
            }
        }
        placement_eligible_units.sort();
        excluded_units.sort();
        recovered_units.sort();

        let uncertain_jobs = self
            .records
            .values()
            .filter(|record| {
                record.coordination.freshness == CoordinationFreshness::Uncertain
                    || matches!(
                        record.coordination.issue,
                        Some(CoordinationIssueKind::OrphanedInFlightJob)
                            | Some(CoordinationIssueKind::MissingWorkerOutcome)
                            | Some(CoordinationIssueKind::StaleWorkerOwnership)
                    )
            })
            .count();
        let recovery_required_jobs = self
            .records
            .values()
            .filter(|record| {
                record.coordination.recovery_signal == RecoverySignal::RecoveryDecisionRequired
                    || record.coordination.recovery_signal == RecoverySignal::AwaitWorkerOutcome
            })
            .count();
        let state = if unavailable_units == self.units.len() {
            DistributedDegradationState::UnrecoverableUnavailable
        } else if recovering_units > 0 {
            DistributedDegradationState::RecoveryInProgress
        } else if degraded_units > 0 || unavailable_units > 0 {
            DistributedDegradationState::PartiallyDegraded
        } else if constrained_serviceable_units > 0 || !self.queue.is_empty() {
            DistributedDegradationState::ConstrainedButServiceable
        } else {
            DistributedDegradationState::Healthy
        };

        DistributedRecoverySnapshot {
            state,
            total_units: self.units.len(),
            healthy_units,
            constrained_serviceable_units,
            degraded_units,
            recovering_units,
            unavailable_units,
            placement_eligible_units,
            excluded_units,
            recovered_units,
            queued_jobs: self.queue.len(),
            uncertain_jobs,
            recovery_required_jobs,
        }
    }

    fn rejected_placement(
        &self,
        job: &QueuedJob,
    ) -> (
        ExecutionPlacement,
        PlacementFailureKind,
        Option<WorkerDispatchOutcome>,
    ) {
        let mut considered = self.assess_candidates(&job.request, job.resource_class);
        let distributed = distributed_summary(&considered);
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
                    resource_class: job.resource_class,
                    capacity_pressure: CapacityPressure::Backpressured,
                    distributed,
                    reason: "requested unit not placeable".to_string(),
                    considered,
                },
                kind,
                dispatch,
            );
        }
        let best = if considered
            .iter()
            .any(|candidate| candidate.detail.contains("insufficient capacity units"))
        {
            PlacementFailureKind::CapacityRejected
        } else if considered
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
                resource_class: job.resource_class,
                capacity_pressure: CapacityPressure::Backpressured,
                distributed,
                reason: "no suitable execution unit".to_string(),
                considered,
            },
            best,
            None,
        )
    }

    fn scheduling_decision(&self, job: &QueuedJob) -> SchedulingDecision {
        let considered = self.assess_candidates(&job.request, job.resource_class);
        let distributed = distributed_summary(&considered);
        if self.selectable_candidate_exists(
            &job.request,
            job.resource_class,
            job.requested_unit.as_ref(),
        ) {
            return SchedulingDecision::RunNow;
        }
        if job.requested_unit.is_some() {
            return SchedulingDecision::NotPlaceable(PlacementFailureKind::WorkerPlacementFailed);
        }
        if distributed.state == DistributedPlacementState::AdmissibleButCurrentlyUnschedulable {
            return SchedulingDecision::QueueRequired;
        }
        if considered
            .iter()
            .any(|candidate| candidate.suitability == PlacementSuitability::Incompatible)
        {
            return SchedulingDecision::NotPlaceable(PlacementFailureKind::BackendIncompatible);
        }
        if considered
            .iter()
            .any(|candidate| candidate.device_suitability == DeviceSuitability::Unsuitable)
        {
            return SchedulingDecision::NotPlaceable(
                PlacementFailureKind::BackendDeviceIncompatible,
            );
        }
        if considered
            .iter()
            .any(|candidate| candidate.detail.contains("insufficient capacity units"))
        {
            return SchedulingDecision::NotPlaceable(PlacementFailureKind::CapacityRejected);
        }
        SchedulingDecision::NotPlaceable(PlacementFailureKind::NoSuitableBackend)
    }

    fn selectable_candidate_exists(
        &self,
        request: &CanonicalPipelineRequest,
        resource_class: ResourceClass,
        requested: Option<&ExecutionUnitId>,
    ) -> bool {
        let considered = self.assess_candidates(request, resource_class);
        match requested {
            Some(requested_unit) => considered.iter().any(|candidate| {
                &candidate.unit_id == requested_unit
                    && candidate.suitability == PlacementSuitability::Suitable
            }),
            None => considered
                .iter()
                .any(|candidate| candidate.suitability == PlacementSuitability::Suitable),
        }
    }

    fn assess_candidates(
        &self,
        request: &CanonicalPipelineRequest,
        resource_class: ResourceClass,
    ) -> Vec<PlacementCandidateAssessment> {
        self.units
            .iter()
            .map(|unit| {
                let lane = unit.service.execution_lane();
                let device_class = unit_device_class(unit.kind);
                let status = unit.runtime_status();
                let device_suitability = device_suitability_for(unit.kind, lane, status);
                let admission = unit.service.technical_admission(request);
                let backend_suitability = admission
                    .failure
                    .as_ref()
                    .map(|failure| suitability_for_failure(failure.kind))
                    .unwrap_or(PlacementSuitability::Suitable);
                if !unit.can_accept_dispatch() {
                    return PlacementCandidateAssessment {
                        unit_id: unit.id.clone(),
                        unit_kind: unit.kind,
                        worker_class: unit.worker_class,
                        registry_role: unit.registry_role,
                        device_class,
                        lane,
                        runtime_status: status,
                        backend_suitability,
                        device_suitability,
                        suitability: combine_suitability(
                            PlacementSuitability::Unavailable,
                            device_suitability,
                        ),
                        detail: format!("unit not dispatchable ({status:?})"),
                    };
                }
                let required_units = resource_class.capacity_weight();
                let free_units = unit
                    .capacity_limit_units()
                    .saturating_sub(unit.used_capacity_units);
                if required_units > free_units {
                    return PlacementCandidateAssessment {
                        unit_id: unit.id.clone(),
                        unit_kind: unit.kind,
                        worker_class: unit.worker_class,
                        registry_role: unit.registry_role,
                        device_class,
                        lane,
                        runtime_status: status,
                        backend_suitability,
                        device_suitability,
                        suitability: combine_suitability(
                            PlacementSuitability::Unavailable,
                            device_suitability,
                        ),
                        detail: format!(
                            "insufficient capacity units required={required_units} free={free_units}"
                        ),
                    };
                }
                let suitability = combine_suitability(backend_suitability, device_suitability);
                if let Some(failure) = admission.failure {
                    PlacementCandidateAssessment {
                        unit_id: unit.id.clone(),
                        unit_kind: unit.kind,
                        worker_class: unit.worker_class,
                        registry_role: unit.registry_role,
                        device_class,
                        lane,
                        runtime_status: status,
                        backend_suitability,
                        device_suitability,
                        suitability,
                        detail: failure.detail,
                    }
                } else {
                    PlacementCandidateAssessment {
                        unit_id: unit.id.clone(),
                        unit_kind: unit.kind,
                        worker_class: unit.worker_class,
                        registry_role: unit.registry_role,
                        device_class,
                        lane,
                        runtime_status: status,
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
        resource_class: ResourceClass,
        requested: Option<ExecutionUnitId>,
    ) -> Option<UnitSelection> {
        let assessments = self.assess_candidates(request, resource_class);
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
                    resource_class: ResourceClass::classify(request),
                    capacity_pressure: self.units[idx].capacity_pressure(),
                    distributed: distributed_summary(&assessments),
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
                resource_class: ResourceClass::classify(request),
                capacity_pressure: self.units[idx].capacity_pressure(),
                distributed: distributed_summary(&assessments),
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
        let required_units = job.resource_class.capacity_weight();
        if !self.units[unit_idx].can_accept_dispatch() {
            let unit = &mut self.units[unit_idx];
            let status = unit.runtime_status();
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
                work_cost_summary: estimated_work_cost_summary(
                    &job.request,
                    job.resource_class,
                    placement.capacity_pressure,
                    CapacityQueueDisposition::RejectedDueToCapacity,
                ),
                placement,
                worker_dispatch_outcome: Some(WorkerDispatchOutcome::Unavailable),
                placement_failure: Some(PlacementFailureKind::BackendUnavailable),
                capacity_disposition: CapacityQueueDisposition::RejectedDueToCapacity,
                provenance: WorkerExecutionProvenance {
                    selected_unit: unit.id.clone(),
                    completed_unit: unit.id.clone(),
                    was_remote: unit.kind == ExecutionUnitKind::Worker,
                    redispatched_to_local: false,
                },
                retry_summary: WorkerRetrySummary::no_retry(),
                coordination: if status == WorkerRuntimeStatus::Stale {
                    JobCoordinationSnapshot::stale_without_dispatch(unit.id.clone(), unit.kind)
                } else {
                    JobCoordinationSnapshot {
                        state: InFlightCoordinationState::Failed,
                        last_in_flight_state: Some(InFlightCoordinationState::Dispatching),
                        owner: Some(unit.id.clone()),
                        owner_kind: Some(unit.kind),
                        owner_last_contact_at_unix_ms: unit.last_health_contact_at_unix_ms,
                        freshness: CoordinationFreshness::Current,
                        issue: None,
                        recovery_signal: RecoverySignal::RecoveryDecisionRequired,
                    }
                },
            };
        }

        let (submitted_job_id, run_result) = {
            let unit = &mut self.units[unit_idx];
            unit.active_jobs = unit.active_jobs.saturating_add(1);
            unit.used_capacity_units = unit.used_capacity_units.saturating_add(required_units);
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
            unit.used_capacity_units = unit.used_capacity_units.saturating_sub(required_units);
            (submitted_job_id, run_result)
        };

        match run_result {
            Ok(Some((record_job_id, state, execution_failure, result)))
                if submitted_job_id == record_job_id =>
            {
                let (selected_unit, selected_kind, selected_last_contact, was_remote) = {
                    let unit = &self.units[unit_idx];
                    (
                        unit.id.clone(),
                        unit.kind,
                        unit.last_health_contact_at_unix_ms,
                        unit.kind == ExecutionUnitKind::Worker,
                    )
                };
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
                    let unit = &mut self.units[unit_idx];
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
                let work_cost_summary = runtime_work_cost_summary(
                    &job.request,
                    job.resource_class,
                    result.as_ref().map(|r| r.diagnostics.work),
                    result.as_ref().map(|r| r.diagnostics.hotspots),
                    completion_class_for(
                        state,
                        &execution_failure,
                        placement.execution_path,
                        result.as_ref(),
                    ),
                    placement.capacity_pressure,
                    CapacityQueueDisposition::None,
                    1,
                    false,
                );
                MultiWorkerJobRecord {
                    id: job.id,
                    state,
                    execution_failure,
                    result,
                    work_cost_summary,
                    placement,
                    worker_dispatch_outcome: dispatch_outcome,
                    placement_failure: None,
                    capacity_disposition: CapacityQueueDisposition::None,
                    provenance: WorkerExecutionProvenance {
                        selected_unit: selected_unit.clone(),
                        completed_unit: selected_unit.clone(),
                        was_remote,
                        redispatched_to_local: false,
                    },
                    retry_summary: WorkerRetrySummary::no_retry(),
                    coordination: JobCoordinationSnapshot::from_terminal(
                        TerminalCoordinationInput {
                            state,
                            last_in_flight_state: Some(InFlightCoordinationState::Running),
                            owner: selected_unit.clone(),
                            owner_kind: selected_kind,
                            owner_last_contact_at_unix_ms: selected_last_contact,
                            issue: None,
                            recovered: false,
                            uncertain: false,
                            awaiting_outcome: false,
                        },
                    ),
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
                    work_cost_summary: estimated_work_cost_summary(
                        &job.request,
                        job.resource_class,
                        placement.capacity_pressure,
                        CapacityQueueDisposition::RejectedDueToCapacity,
                    ),
                    placement,
                    worker_dispatch_outcome: Some(WorkerDispatchOutcome::TransportFailure),
                    placement_failure: Some(PlacementFailureKind::WorkerPlacementFailed),
                    capacity_disposition: CapacityQueueDisposition::RejectedDueToCapacity,
                    provenance: WorkerExecutionProvenance {
                        selected_unit: failed_unit.clone(),
                        completed_unit: failed_unit.clone(),
                        was_remote: true,
                        redispatched_to_local: false,
                    },
                    retry_summary: WorkerRetrySummary {
                        attempts: 1,
                        retries_exhausted: false,
                        uncertain_prior_attempt_outcome: true,
                        recovered_by: None,
                        last_failure_kind: Some(WorkerFailureKind::TransportFailure),
                    },
                    coordination: JobCoordinationSnapshot::from_terminal(
                        TerminalCoordinationInput {
                            state: JobLifecycleState::Failed,
                            last_in_flight_state: Some(
                                InFlightCoordinationState::AwaitingWorkerOutcome,
                            ),
                            owner: failed_unit,
                            owner_kind: ExecutionUnitKind::Worker,
                            owner_last_contact_at_unix_ms: None,
                            issue: Some(CoordinationIssueKind::MissingWorkerOutcome),
                            recovered: false,
                            uncertain: true,
                            awaiting_outcome: true,
                        },
                    ),
                }
            }
            Err(err) => {
                let failure = canonical_execution_failure(err);
                let state = if failure.kind == CanonicalFailureKind::Timeout {
                    JobLifecycleState::TimedOut
                } else {
                    JobLifecycleState::Failed
                };
                let failure_kind = classify_worker_failure(&failure);
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
                    && failure_kind != WorkerFailureKind::TerminalComputeExecutionFailure
                    && failure_kind != WorkerFailureKind::StructuredExecutionFailure
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
                    work_cost_summary: estimated_work_cost_summary(
                        &job.request,
                        job.resource_class,
                        placement.capacity_pressure,
                        CapacityQueueDisposition::RejectedDueToCapacity,
                    ),
                    placement,
                    worker_dispatch_outcome: Some(dispatch_outcome),
                    placement_failure: Some(PlacementFailureKind::WorkerPlacementFailed),
                    capacity_disposition: CapacityQueueDisposition::RejectedDueToCapacity,
                    provenance: WorkerExecutionProvenance {
                        selected_unit: failed_unit.clone(),
                        completed_unit: failed_unit,
                        was_remote: self.units[unit_idx].kind == ExecutionUnitKind::Worker,
                        redispatched_to_local: false,
                    },
                    retry_summary: WorkerRetrySummary {
                        attempts: 1,
                        retries_exhausted: false,
                        uncertain_prior_attempt_outcome: false,
                        recovered_by: None,
                        last_failure_kind: Some(failure_kind),
                    },
                    coordination: JobCoordinationSnapshot::from_terminal(
                        TerminalCoordinationInput {
                            state,
                            last_in_flight_state: Some(InFlightCoordinationState::Running),
                            owner: self.units[unit_idx].id.clone(),
                            owner_kind: self.units[unit_idx].kind,
                            owner_last_contact_at_unix_ms: self.units[unit_idx]
                                .last_health_contact_at_unix_ms,
                            issue: if matches!(
                                failure_kind,
                                WorkerFailureKind::DispatchFailedBeforeExecution
                                    | WorkerFailureKind::WorkerUnavailableOrStale
                            ) {
                                Some(CoordinationIssueKind::OrphanedInFlightJob)
                            } else {
                                None
                            },
                            recovered: false,
                            uncertain: matches!(
                                failure_kind,
                                WorkerFailureKind::WorkerUnavailableOrStale
                                    | WorkerFailureKind::DispatchFailedBeforeExecution
                            ),
                            awaiting_outcome: state == JobLifecycleState::TimedOut,
                        },
                    ),
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
        let required_units = job.resource_class.capacity_weight();
        local.active_jobs = local.active_jobs.saturating_add(1);
        local.used_capacity_units = local.used_capacity_units.saturating_add(required_units);
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
        local.used_capacity_units = local.used_capacity_units.saturating_sub(required_units);
        match run_result {
            Ok(Some((record_job_id, state, execution_failure, result)))
                if submitted_job_id == record_job_id =>
            {
                local.note_success(job.id, WorkerDispatchOutcome::RedispatchedLocal);
                let mut distributed = original_placement.distributed.clone();
                if !distributed.placeable_units.contains(&local.id) {
                    distributed.placeable_units.push(local.id.clone());
                    distributed.placeable_units.sort();
                }
                distributed.state = DistributedPlacementState::AdmissibleDegradedOnly;
                distributed.locality = DistributedPlacementLocality::LocalAndRemote;
                let work_cost_summary = runtime_work_cost_summary(
                    &job.request,
                    job.resource_class,
                    result.as_ref().map(|r| r.diagnostics.work),
                    result.as_ref().map(|r| r.diagnostics.hotspots),
                    completion_class_for(
                        state,
                        &execution_failure,
                        JobExecutionPath::LocalCanonical,
                        result.as_ref(),
                    ),
                    local.capacity_pressure(),
                    CapacityQueueDisposition::DegradedPlacementDueToPressure,
                    2,
                    true,
                );
                Some(MultiWorkerJobRecord {
                    id: job.id,
                    state,
                    execution_failure,
                    result,
                    work_cost_summary,
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
                        resource_class: job.resource_class,
                        capacity_pressure: local.capacity_pressure(),
                        distributed,
                        reason: format!("worker {} failed; redispatched to local", failed_worker.0),
                        considered: original_placement.considered.clone(),
                    },
                    worker_dispatch_outcome: Some(WorkerDispatchOutcome::RedispatchedLocal),
                    placement_failure: None,
                    capacity_disposition: CapacityQueueDisposition::DegradedPlacementDueToPressure,
                    provenance: WorkerExecutionProvenance {
                        selected_unit: failed_worker,
                        completed_unit: local.id.clone(),
                        was_remote: true,
                        redispatched_to_local: true,
                    },
                    retry_summary: WorkerRetrySummary {
                        attempts: 2,
                        retries_exhausted: false,
                        uncertain_prior_attempt_outcome: false,
                        recovered_by: Some(WorkerRecoveryKind::LocalFallback),
                        last_failure_kind: Some(WorkerFailureKind::WorkerExecutionCrashed),
                    },
                    coordination: JobCoordinationSnapshot::from_terminal(
                        TerminalCoordinationInput {
                            state,
                            last_in_flight_state: Some(
                                InFlightCoordinationState::RedispatchPending,
                            ),
                            owner: local.id.clone(),
                            owner_kind: local.kind,
                            owner_last_contact_at_unix_ms: local.last_health_contact_at_unix_ms,
                            issue: None,
                            recovered: true,
                            uncertain: false,
                            awaiting_outcome: false,
                        },
                    ),
                })
            }
            _ => None,
        }
    }
}

impl ExecutionUnit {
    const STALE_AFTER_MS: u64 = 30_000;
    const COOLDOWN_BASE_MS: u64 = 2_000;
    const COOLDOWN_MAX_MS: u64 = 30_000;
    const RECOVERY_WINDOW_MS: u64 = 15_000;

    fn capacity_limit_units(&self) -> usize {
        self.max_parallel_jobs.max(1).saturating_mul(2)
    }

    fn runtime_status(&self) -> WorkerRuntimeStatus {
        let now = now_unix_ms();
        let last_contact = self.last_health_contact_at_unix_ms;
        if last_contact.is_none() {
            return WorkerRuntimeStatus::Unknown;
        }
        if now.saturating_sub(last_contact.unwrap_or(now)) > Self::STALE_AFTER_MS {
            return WorkerRuntimeStatus::Stale;
        }
        if self.availability != WorkerAvailability::Available {
            return WorkerRuntimeStatus::Unavailable;
        }
        if self
            .quarantine_until_unix_ms
            .is_some_and(|until| now < until)
        {
            return WorkerRuntimeStatus::Degraded;
        }
        if self.consecutive_failures >= 3 {
            return WorkerRuntimeStatus::Unhealthy;
        }
        let limit = self.capacity_limit_units().max(1);
        if self.used_capacity_units >= limit {
            return WorkerRuntimeStatus::Backpressured;
        }
        if self.active_jobs >= self.max_parallel_jobs
            || self.used_capacity_units.saturating_mul(10) >= limit.saturating_mul(9)
        {
            return WorkerRuntimeStatus::Saturated;
        }
        if self.used_capacity_units.saturating_mul(10) >= limit.saturating_mul(7) {
            return WorkerRuntimeStatus::Constrained;
        }
        if self.active_jobs > 0 {
            WorkerRuntimeStatus::Busy
        } else if self.last_job_id.is_some() {
            WorkerRuntimeStatus::Ready
        } else {
            WorkerRuntimeStatus::Known
        }
    }

    fn capacity_pressure(&self) -> CapacityPressure {
        capacity_pressure_for(
            self.used_capacity_units,
            self.capacity_limit_units(),
            self.active_jobs > 0,
        )
    }

    fn can_accept_dispatch(&self) -> bool {
        matches!(
            self.runtime_status(),
            WorkerRuntimeStatus::Known
                | WorkerRuntimeStatus::Ready
                | WorkerRuntimeStatus::Busy
                | WorkerRuntimeStatus::Constrained
        )
    }

    fn degradation_state(&self) -> DistributedDegradationState {
        let now = now_unix_ms();
        if self
            .recovered_at_unix_ms
            .is_some_and(|at| now.saturating_sub(at) <= Self::RECOVERY_WINDOW_MS)
        {
            return DistributedDegradationState::RecoveryInProgress;
        }
        match self.runtime_status() {
            WorkerRuntimeStatus::Known | WorkerRuntimeStatus::Ready | WorkerRuntimeStatus::Busy => {
                DistributedDegradationState::Healthy
            }
            WorkerRuntimeStatus::Constrained => {
                DistributedDegradationState::ConstrainedButServiceable
            }
            WorkerRuntimeStatus::Saturated
            | WorkerRuntimeStatus::Backpressured
            | WorkerRuntimeStatus::Degraded
            | WorkerRuntimeStatus::Unhealthy => DistributedDegradationState::PartiallyDegraded,
            WorkerRuntimeStatus::Unavailable
            | WorkerRuntimeStatus::Stale
            | WorkerRuntimeStatus::Unknown => DistributedDegradationState::UnrecoverableUnavailable,
        }
    }

    fn note_success(&mut self, job_id: JobId, outcome: WorkerDispatchOutcome) {
        let previous = self.runtime_status();
        self.last_job_id = Some(job_id);
        self.last_dispatch_outcome = Some(outcome);
        self.last_error = None;
        self.consecutive_failures = 0;
        self.last_used_at_unix_ms = Some(now_unix_ms());
        self.last_health_contact_at_unix_ms = self.last_used_at_unix_ms;
        self.quarantine_until_unix_ms = None;
        if matches!(
            previous,
            WorkerRuntimeStatus::Degraded
                | WorkerRuntimeStatus::Unhealthy
                | WorkerRuntimeStatus::Saturated
                | WorkerRuntimeStatus::Backpressured
                | WorkerRuntimeStatus::Unavailable
                | WorkerRuntimeStatus::Stale
        ) {
            self.recovered_at_unix_ms = self.last_used_at_unix_ms;
        }
    }

    fn note_failure(&mut self, job_id: JobId, outcome: WorkerDispatchOutcome, detail: String) {
        self.last_job_id = Some(job_id);
        self.last_dispatch_outcome = Some(outcome);
        self.last_error = Some(detail);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        let now = now_unix_ms();
        self.last_used_at_unix_ms = Some(now);
        self.last_health_contact_at_unix_ms = Some(now);
        let exp = self.consecutive_failures.saturating_sub(1).min(4);
        let cooldown_ms =
            (Self::COOLDOWN_BASE_MS.saturating_mul(1_u64 << exp)).min(Self::COOLDOWN_MAX_MS);
        self.quarantine_until_unix_ms = Some(now.saturating_add(cooldown_ms));
        self.recovered_at_unix_ms = None;
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
            | WorkerRuntimeStatus::Backpressured
            | WorkerRuntimeStatus::Degraded
            | WorkerRuntimeStatus::Stale
            | WorkerRuntimeStatus::Unknown
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

fn capacity_pressure_for(
    used_units: usize,
    limit_units: usize,
    has_backlog: bool,
) -> CapacityPressure {
    let limit_units = limit_units.max(1);
    if has_backlog && used_units < limit_units.saturating_mul(3) / 5 {
        CapacityPressure::TemporarilyUnschedulable
    } else if used_units >= limit_units
        || (has_backlog && used_units.saturating_mul(10) >= limit_units.saturating_mul(9))
    {
        CapacityPressure::Backpressured
    } else if has_backlog || used_units.saturating_mul(10) >= limit_units.saturating_mul(8) {
        CapacityPressure::Saturated
    } else if used_units.saturating_mul(10) >= limit_units.saturating_mul(6) {
        CapacityPressure::Constrained
    } else {
        CapacityPressure::Healthy
    }
}

fn estimated_work_cost_summary(
    request: &CanonicalPipelineRequest,
    resource_class: ResourceClass,
    pressure: CapacityPressure,
    queue_disposition: CapacityQueueDisposition,
) -> ConsolidatedWorkCostSummary {
    let estimated_total_work_units = request.budget.global_work_units;
    ConsolidatedWorkCostSummary {
        provenance: WorkCostProvenance::EstimatedFromBudget,
        resource_class,
        estimated_total_work_units,
        runtime_consumed_work_units: None,
        runtime_remaining_work_units: None,
        dominant_stage: None,
        dominant_stage_share_bps: None,
        degraded_stage_count: 0,
        retry_attempts: 1,
        redispatched_to_local: false,
        queue_deferred_by_capacity: matches!(
            queue_disposition,
            CapacityQueueDisposition::QueuedDueToCapacity
                | CapacityQueueDisposition::DeferredDueToCapacity
        ),
        pressure,
        queue_disposition,
        tension: if matches!(
            queue_disposition,
            CapacityQueueDisposition::RejectedDueToCapacity
        ) {
            WorkCostTension::LowCostButBlocked
        } else {
            WorkCostTension::Nominal
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_work_cost_summary(
    request: &CanonicalPipelineRequest,
    resource_class: ResourceClass,
    work_summary: Option<CanonicalWorkSummary>,
    hotspot_summary: Option<CanonicalHotspotSummary>,
    completion_class: JobCompletionClass,
    pressure: CapacityPressure,
    queue_disposition: CapacityQueueDisposition,
    retry_attempts: u8,
    redispatched_to_local: bool,
) -> ConsolidatedWorkCostSummary {
    let estimated_total_work_units = request.budget.global_work_units;
    let (runtime_consumed_work_units, runtime_remaining_work_units) = work_summary
        .map(|work| {
            (
                Some(
                    work.global_budget_units
                        .saturating_sub(work.global_remaining_units),
                ),
                Some(work.global_remaining_units),
            )
        })
        .unwrap_or((None, None));
    let degraded_stage_count = hotspot_summary
        .map(|hotspot| hotspot.degraded_stage_count)
        .unwrap_or(0);
    let tension = if retry_attempts > 1 {
        WorkCostTension::RetriedWithAdditionalCost
    } else if degraded_stage_count > 0 {
        WorkCostTension::ExpensiveAndDegraded
    } else if matches!(completion_class, JobCompletionClass::Completed)
        && runtime_consumed_work_units
            .map(|consumed| {
                consumed.saturating_mul(10) >= estimated_total_work_units.saturating_mul(8)
            })
            .unwrap_or(false)
    {
        WorkCostTension::ExpensiveButSuccessful
    } else {
        WorkCostTension::Nominal
    };
    ConsolidatedWorkCostSummary {
        provenance: WorkCostProvenance::RuntimeMeasured,
        resource_class,
        estimated_total_work_units,
        runtime_consumed_work_units,
        runtime_remaining_work_units,
        dominant_stage: hotspot_summary.and_then(|hotspot| hotspot.dominant_stage),
        dominant_stage_share_bps: hotspot_summary
            .and_then(|hotspot| hotspot.dominant_stage_share_bps),
        degraded_stage_count,
        retry_attempts,
        redispatched_to_local,
        queue_deferred_by_capacity: matches!(
            queue_disposition,
            CapacityQueueDisposition::QueuedDueToCapacity
                | CapacityQueueDisposition::DeferredDueToCapacity
        ),
        pressure,
        queue_disposition,
        tension,
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

fn distributed_summary(
    assessments: &[PlacementCandidateAssessment],
) -> DistributedPlacementSummary {
    let mut admissible_units = assessments
        .iter()
        .filter(|candidate| {
            candidate.backend_suitability == PlacementSuitability::Suitable
                && candidate.device_suitability != DeviceSuitability::Unsuitable
        })
        .map(|candidate| candidate.unit_id.clone())
        .collect::<Vec<_>>();
    admissible_units.sort();

    let mut placeable_units = assessments
        .iter()
        .filter(|candidate| candidate.suitability == PlacementSuitability::Suitable)
        .map(|candidate| candidate.unit_id.clone())
        .collect::<Vec<_>>();
    placeable_units.sort();

    let locality = locality_for_assessments(assessments, &admissible_units);
    let placeable_is_degraded_only = !placeable_units.is_empty()
        && assessments
            .iter()
            .filter(|candidate| candidate.suitability == PlacementSuitability::Suitable)
            .all(|candidate| candidate.lane == BackendExecutionLane::Candle);
    let state = if !placeable_units.is_empty() {
        if placeable_is_degraded_only {
            DistributedPlacementState::AdmissibleDegradedOnly
        } else if placeable_units.len() < assessments.len() {
            DistributedPlacementState::AdmissiblePlaceableOnSubset
        } else {
            DistributedPlacementState::AdmissibleAndPlaceable
        }
    } else if !admissible_units.is_empty() {
        DistributedPlacementState::AdmissibleButCurrentlyUnschedulable
    } else {
        DistributedPlacementState::BlockedIncompatible
    };

    let has_burn = assessments.iter().any(|candidate| {
        candidate.suitability == PlacementSuitability::Suitable
            && candidate.lane == BackendExecutionLane::Burn
    });
    let has_candle = assessments.iter().any(|candidate| {
        candidate.suitability == PlacementSuitability::Suitable
            && candidate.lane == BackendExecutionLane::Candle
    });

    DistributedPlacementSummary {
        state,
        locality,
        admissible_units,
        placeable_units,
        degraded_fallback_possible: has_candle && !has_burn,
    }
}

fn locality_for_assessments(
    assessments: &[PlacementCandidateAssessment],
    admissible_units: &[ExecutionUnitId],
) -> DistributedPlacementLocality {
    let mut has_local = false;
    let mut has_remote = false;
    for candidate in assessments {
        if !admissible_units.contains(&candidate.unit_id) {
            continue;
        }
        match candidate.unit_kind {
            ExecutionUnitKind::Local => has_local = true,
            ExecutionUnitKind::Worker => has_remote = true,
        }
    }
    match (has_local, has_remote) {
        (true, true) => DistributedPlacementLocality::LocalAndRemote,
        (true, false) => DistributedPlacementLocality::LocalOnly,
        (false, true) => DistributedPlacementLocality::RemoteOnly,
        (false, false) => DistributedPlacementLocality::None,
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
        CapacityPressure, CapacityQueueDisposition, CoordinationFreshness, CoordinationIssueKind,
        DeviceSuitability, DistributedDegradationState, DistributedPlacementLocality,
        DistributedPlacementState, ExecutionDeviceClass, ExecutionUnitId, ExecutionUnitKind,
        InFlightCoordinationState, InMemoryComputeService, JobCompletionClass,
        JobCoordinationSnapshot, JobExecutionPath, JobLifecycleState, JobSubmissionMeta,
        MultiWorkerComputeService, PlacementFailureKind, PlacementSuitability, RecoverySignal,
        ResourceClass, SchedulerConfig, TerminalCoordinationInput, WorkCostProvenance,
        WorkCostTension, WorkerAvailability, WorkerClass, WorkerDispatchOutcome,
        WorkerRecoveryKind, WorkerRegistryRole, WorkerRuntimeStatus,
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
        assert_eq!(record.accounting.resource_class, ResourceClass::Light);
        assert_eq!(
            record.accounting.capacity_queue_disposition,
            CapacityQueueDisposition::QueuedDueToCapacity
        );
        let work_cost = record
            .accounting
            .work_cost_summary
            .as_ref()
            .expect("work/cost summary");
        assert_eq!(
            work_cost.provenance,
            WorkCostProvenance::EstimatedFromBudget
        );
        assert_eq!(work_cost.tension, WorkCostTension::Nominal);
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
                rollout: None,
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
        let work_cost = record
            .accounting
            .work_cost_summary
            .as_ref()
            .expect("work/cost summary");
        assert_eq!(work_cost.provenance, WorkCostProvenance::RuntimeMeasured);
        assert!(!record.accounting.stage_profiles.is_empty());
        assert!(record.accounting.hotspot_summary.is_some());
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
        assert_eq!(
            completed.accounting.stage_profiles,
            result.diagnostics.stage_profiles
        );
        assert_eq!(
            completed.accounting.hotspot_summary,
            Some(result.diagnostics.hotspots)
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
    fn multi_worker_transient_unavailability_is_deferred_then_runs() {
        let backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(backend, 1);
        for unit in service.execution_units() {
            service.set_worker_availability(&unit.id, WorkerAvailability::Unavailable);
        }
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 17,
                submitted_by: Some("scheduler".to_string()),
            },
            None,
        );
        service.run_scheduler_cycle(1);
        let deferred = service.job(job_id).expect("deferred record");
        assert_eq!(deferred.state, JobLifecycleState::Queued);
        assert_eq!(
            deferred.placement_failure,
            Some(PlacementFailureKind::CurrentlyUnschedulable)
        );
        assert_eq!(
            deferred.worker_dispatch_outcome,
            Some(WorkerDispatchOutcome::Deferred)
        );
        assert_eq!(
            deferred.capacity_disposition,
            CapacityQueueDisposition::DeferredDueToCapacity
        );
        assert_eq!(
            deferred.placement.distributed.state,
            DistributedPlacementState::AdmissibleButCurrentlyUnschedulable
        );
        assert_eq!(
            deferred.placement.distributed.locality,
            DistributedPlacementLocality::LocalOnly
        );
        assert_eq!(deferred.placement.distributed.admissible_units.len(), 1);
        assert!(deferred.placement.distributed.placeable_units.is_empty());

        for unit in service.execution_units() {
            service.set_worker_availability(&unit.id, WorkerAvailability::Available);
        }
        service.run_scheduler_cycle(1);
        let ran = service.job(job_id).expect("executed record");
        assert_ne!(ran.state, JobLifecycleState::Queued);
        assert_ne!(
            ran.placement_failure,
            Some(PlacementFailureKind::CurrentlyUnschedulable)
        );
    }

    #[test]
    fn multi_worker_unschedulable_exhausts_retries_and_fails() {
        let backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(backend, 1);
        for unit in service.execution_units() {
            service.set_worker_availability(&unit.id, WorkerAvailability::Unavailable);
        }
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 18,
                submitted_by: Some("scheduler".to_string()),
            },
            None,
        );
        service.run_scheduler_cycle(1);
        service.run_scheduler_cycle(1);
        service.run_scheduler_cycle(1);
        service.run_scheduler_cycle(1);
        let failed = service.job(job_id).expect("terminal record");
        assert_eq!(failed.state, JobLifecycleState::Failed);
        assert_eq!(
            failed.placement_failure,
            Some(PlacementFailureKind::DeviceUnavailable)
        );
        assert_eq!(
            failed.capacity_disposition,
            CapacityQueueDisposition::RejectedDueToCapacity
        );
        assert!(failed
            .execution_failure
            .as_ref()
            .expect("failure expected")
            .detail
            .contains("unschedulable"));
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
        assert_eq!(
            candle_record.placement.distributed.state,
            DistributedPlacementState::AdmissibleDegradedOnly
        );
        assert!(
            candle_record
                .placement
                .distributed
                .degraded_fallback_possible
        );
    }

    #[test]
    fn distributed_placement_reports_local_only_subset_when_remote_is_incompatible() {
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
            detail: Some("slot incompatible for distributed semantics test".to_string()),
            resolved_path: None,
            hash_prefix: None,
            contract_version: Some("v1".to_string()),
            format: None,
            gate: Default::default(),
            rollout: None,
        }];
        let incompatible_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, disabled_world),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        service.register_worker_backend("remote-incompatible", incompatible_backend, 1);
        let job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 18,
                submitted_by: Some("distributed".to_string()),
            },
            None,
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job).expect("record");
        assert_eq!(
            record.placement.distributed.state,
            DistributedPlacementState::AdmissiblePlaceableOnSubset
        );
        assert_eq!(
            record.placement.distributed.locality,
            DistributedPlacementLocality::LocalOnly
        );
        assert_eq!(
            record.placement.distributed.admissible_units,
            vec![ExecutionUnitId("local".to_string())]
        );
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
            rollout: None,
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
    fn queued_jobs_are_visible_in_in_flight_snapshot() {
        let backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(backend, 1);
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 21,
                submitted_by: Some("queue".to_string()),
            },
            None,
        );
        let in_flight = service.in_flight_jobs();
        let queued = in_flight
            .iter()
            .find(|snapshot| snapshot.job_id == job_id)
            .expect("queued snapshot");
        assert_eq!(queued.state, InFlightCoordinationState::Queued);
        assert_eq!(queued.recovery_signal, RecoverySignal::SafeToRedispatch);
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
        assert_eq!(record.retry_summary.attempts, 2);
        assert_eq!(
            record.retry_summary.recovered_by,
            Some(WorkerRecoveryKind::LocalFallback)
        );
        assert_eq!(
            record.coordination.issue,
            Some(CoordinationIssueKind::RecoveredCoordinationState)
        );
        assert_eq!(
            record.coordination.recovery_signal,
            RecoverySignal::Terminal
        );
        assert!(record.result.is_some());
        match previous {
            Some(value) => std::env::set_var("UCF_WORKER_BIN", value),
            None => std::env::remove_var("UCF_WORKER_BIN"),
        }
    }

    #[test]
    fn requested_worker_failure_stays_terminal_without_auto_fallback() {
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
            .register_worker("remote-terminal", 200, 1)
            .expect("register worker");
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 23,
                submitted_by: Some("terminal".to_string()),
            },
            Some(worker_id.clone()),
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job_id).expect("record");
        assert_eq!(record.state, JobLifecycleState::Failed);
        assert_eq!(record.provenance.selected_unit, worker_id);
        assert!(!record.provenance.redispatched_to_local);
        assert_eq!(record.retry_summary.attempts, 1);
        assert!(record.retry_summary.recovered_by.is_none());
        assert_ne!(
            record.retry_summary.recovered_by,
            Some(WorkerRecoveryKind::LocalFallback)
        );
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
        assert!(matches!(
            snapshot.status,
            WorkerRuntimeStatus::Degraded | WorkerRuntimeStatus::Unhealthy
        ));
        assert!(snapshot.last_error.is_some());
        match previous {
            Some(value) => std::env::set_var("UCF_WORKER_BIN", value),
            None => std::env::remove_var("UCF_WORKER_BIN"),
        }
    }

    #[test]
    fn worker_registry_snapshot_reports_class_role_and_health_timestamps() {
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
        let worker_id = service.register_worker_backend("registry-worker", worker_backend, 1);
        let snapshots = service.execution_units();
        let local = snapshots
            .iter()
            .find(|unit| unit.id.0 == "local")
            .expect("local snapshot");
        assert_eq!(local.worker_class, WorkerClass::LocalPrimary);
        assert_eq!(local.registry_role, WorkerRegistryRole::Primary);
        assert!(local.last_health_contact_at_unix_ms.is_some());
        let worker = snapshots
            .iter()
            .find(|unit| unit.id == worker_id)
            .expect("worker snapshot");
        assert_eq!(worker.worker_class, WorkerClass::RemoteSecondary);
        assert_eq!(worker.registry_role, WorkerRegistryRole::Secondary);
        assert!(worker.last_health_contact_at_unix_ms.is_some());
    }

    #[test]
    fn pressure_snapshot_reports_backpressure_and_unschedulable_sets() {
        let backend = ComputePipelineBackend::new(
            pack_with_name("worker_v1", BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(backend, 1);
        let worker_backend = ComputePipelineBackend::new(
            pack_with_name("worker_v1", BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let worker_id = service.register_worker_backend("pressure-worker", worker_backend, 1);
        service.set_worker_availability(&worker_id, WorkerAvailability::Unavailable);
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 9,
                submitted_by: Some("pressure".to_string()),
            },
            None,
        );
        let _ = service.run_scheduler_cycle(1);
        let record = service.job(job_id).expect("record");
        assert_eq!(
            record.capacity_disposition,
            CapacityQueueDisposition::DeferredDueToCapacity
        );
        assert_eq!(
            record.work_cost_summary.provenance,
            WorkCostProvenance::EstimatedFromBudget
        );
        assert!(record.work_cost_summary.queue_deferred_by_capacity);
        let pressure = service.pressure_snapshot();
        assert_eq!(
            pressure.service_pressure,
            CapacityPressure::TemporarilyUnschedulable
        );
        assert_eq!(pressure.queued_jobs, 1);
        assert_eq!(pressure.queued_light_jobs, 1);
        assert_eq!(pressure.queued_standard_jobs, 0);
        assert_eq!(pressure.queued_heavy_jobs, 0);
    }

    #[test]
    fn distributed_recovery_snapshot_marks_partial_degradation_and_excludes_worker() {
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
        let worker_id = service.register_worker_backend("degraded-worker", worker_backend, 1);
        service.set_worker_availability(&worker_id, WorkerAvailability::Unavailable);

        let snapshot = service.distributed_recovery_snapshot();
        assert_eq!(
            snapshot.state,
            DistributedDegradationState::PartiallyDegraded
        );
        assert!(snapshot.excluded_units.contains(&worker_id));
        assert!(!snapshot.placement_eligible_units.contains(&worker_id));
        assert_eq!(snapshot.unavailable_units, 1);
    }

    #[test]
    fn distributed_recovery_snapshot_recovers_to_recovery_in_progress() {
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
        let worker_id = service.register_worker_backend("recover-worker", worker_backend, 1);
        service.set_worker_availability(&worker_id, WorkerAvailability::Unavailable);
        let degraded = service.distributed_recovery_snapshot();
        assert_eq!(
            degraded.state,
            DistributedDegradationState::PartiallyDegraded
        );

        service.set_worker_availability(&worker_id, WorkerAvailability::Available);
        let recovered = service.distributed_recovery_snapshot();
        assert_eq!(
            recovered.state,
            DistributedDegradationState::RecoveryInProgress
        );
        assert!(recovered.recovered_units.contains(&worker_id));
        assert!(recovered.placement_eligible_units.contains(&worker_id));
    }

    #[test]
    fn distributed_recovery_snapshot_tracks_uncertain_and_recovery_required_jobs() {
        let coordination = JobCoordinationSnapshot::from_terminal(TerminalCoordinationInput {
            state: JobLifecycleState::Failed,
            last_in_flight_state: Some(InFlightCoordinationState::AwaitingWorkerOutcome),
            owner: ExecutionUnitId("remote-orphan".to_string()),
            owner_kind: ExecutionUnitKind::Worker,
            owner_last_contact_at_unix_ms: None,
            issue: Some(CoordinationIssueKind::OrphanedInFlightJob),
            recovered: false,
            uncertain: true,
            awaiting_outcome: true,
        });
        assert_eq!(
            coordination.recovery_signal,
            RecoverySignal::AwaitWorkerOutcome
        );

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
        let worker_id = service.register_worker_backend("orphan-worker", worker_backend, 1);
        service.set_worker_last_health_contact_for_test(&worker_id, Some(1));
        let job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 41,
                submitted_by: Some("orphaned".to_string()),
            },
            Some(worker_id),
        );
        service.run_scheduler_cycle(1);
        let snapshot = service.distributed_recovery_snapshot();
        assert_eq!(snapshot.uncertain_jobs, 1);
        assert_eq!(snapshot.recovery_required_jobs, 1);
        let record = service.job(job).expect("record");
        assert_eq!(record.coordination.freshness, CoordinationFreshness::Stale);
        assert_eq!(
            record.coordination.issue,
            Some(CoordinationIssueKind::StaleWorkerOwnership)
        );
    }

    #[test]
    fn local_redispatch_is_marked_as_degraded_due_to_pressure() {
        let previous = std::env::var("UCF_WORKER_BIN").ok();
        std::env::set_var("UCF_WORKER_BIN", "definitely-missing-ucf-worker-binary");

        let local_backend = ComputePipelineBackend::new(
            pack_with(BackendComponentId::ToyV1, Vec::new()),
            FusionConfig::default(),
            LimitsConfig::default(),
        );
        let mut service = MultiWorkerComputeService::new(local_backend, 1);
        service
            .register_worker("remote-failing-for-pressure", 77, 1)
            .expect("register worker");
        let job_id = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 10,
                submitted_by: Some("pressure-fallback".to_string()),
            },
            None,
        );
        let _ = service.run_scheduler_cycle(1);
        let record = service.job(job_id).expect("redispatched record");
        assert_eq!(
            record.worker_dispatch_outcome,
            Some(WorkerDispatchOutcome::RedispatchedLocal)
        );
        assert_eq!(
            record.capacity_disposition,
            CapacityQueueDisposition::DegradedPlacementDueToPressure
        );

        match previous {
            Some(value) => std::env::set_var("UCF_WORKER_BIN", value),
            None => std::env::remove_var("UCF_WORKER_BIN"),
        }
    }

    #[test]
    fn stale_worker_status_is_classified_and_rejected_for_requested_dispatch() {
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
        let worker_id = service.register_worker_backend("stale-worker", worker_backend, 1);
        service.set_worker_last_health_contact_for_test(&worker_id, Some(1));
        let job = service.submit(
            valid_request(),
            JobSubmissionMeta {
                submitted_at_unix_ms: 40,
                submitted_by: Some("stale".to_string()),
            },
            Some(worker_id.clone()),
        );
        service.run_scheduler_cycle(1);
        let record = service.job(job).expect("record");
        assert_eq!(
            record.worker_dispatch_outcome,
            Some(WorkerDispatchOutcome::Unavailable)
        );
        assert_eq!(
            record.placement_failure,
            Some(PlacementFailureKind::DeviceUnavailable)
        );
        assert!(record
            .placement
            .considered
            .iter()
            .any(|c| c.unit_id == worker_id
                && c.runtime_status == WorkerRuntimeStatus::Stale
                && c.detail.contains("not dispatchable")));
        assert_eq!(record.coordination.state, InFlightCoordinationState::Stale);
        assert_eq!(
            record.coordination.issue,
            Some(CoordinationIssueKind::StaleWorkerOwnership)
        );
        assert_eq!(
            record.coordination.recovery_signal,
            RecoverySignal::RecoveryDecisionRequired
        );
    }

    #[test]
    fn uncertain_orphaned_coordination_requires_recovery_signal() {
        let coordination = JobCoordinationSnapshot::from_terminal(TerminalCoordinationInput {
            state: JobLifecycleState::Failed,
            last_in_flight_state: Some(InFlightCoordinationState::AwaitingWorkerOutcome),
            owner: ExecutionUnitId("remote-uncertain".to_string()),
            owner_kind: ExecutionUnitKind::Worker,
            owner_last_contact_at_unix_ms: None,
            issue: Some(CoordinationIssueKind::OrphanedInFlightJob),
            recovered: false,
            uncertain: true,
            awaiting_outcome: true,
        });
        assert_eq!(coordination.freshness, CoordinationFreshness::Uncertain);
        assert_eq!(
            coordination.issue,
            Some(CoordinationIssueKind::OrphanedInFlightJob)
        );
        assert_eq!(
            coordination.recovery_signal,
            RecoverySignal::AwaitWorkerOutcome
        );
    }
}
