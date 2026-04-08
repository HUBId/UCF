use crate::compute_service::{
    InMemoryComputeService, JobCompletionClass, JobExecutionPath, JobId, JobLifecycleEvent,
    JobLifecycleState, JobRecord, JobSubmissionMeta,
};
use crate::job_history::{
    JobHistoryStore, JobHistoryStoreError, PersistedCanonicalRequest, PersistedJobRecord,
};
use crate::pipeline::{
    CanonicalFailureKind, CanonicalPipelineFailure, CanonicalPipelineRequest,
    CanonicalPipelineState, CanonicalWorkSummary,
};
use crate::{ModelSlot, ModelSlotProvenance, SlotRuntimeStatus};
use std::collections::BTreeMap;

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
    pub recovery_disposition: Option<RecoveryDisposition>,
    pub recovery_source_job_id: Option<JobId>,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
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
    pub latest_baseline_comparison: Option<BaselineComparisonSummary>,
    pub recovery: Option<ComputeRecoverySnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryDisposition {
    CompletedBeforeRestart,
    PersistedNotYetResumed,
    RunningStateUncertainAfterRestart,
    Resumable,
    ResumeUnsupported,
    RerunRequired,
    LostDueToRestart,
    RecoveryCompletedSuccessfully,
    RestartRecoveryFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredJobStatus {
    pub source_job_id: JobId,
    pub source_lifecycle_state: String,
    pub disposition: RecoveryDisposition,
    pub resumed_as_job_id: Option<JobId>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeRecoverySnapshot {
    pub recovered_jobs: usize,
    pub resumed_jobs: usize,
    pub rerun_required_jobs: usize,
    pub uncertain_jobs: usize,
    pub failed_jobs: usize,
    pub records: Vec<RecoveredJobStatus>,
}

pub struct CanonicalComputeEntryPoint {
    service: InMemoryComputeService,
    history_store: Option<JobHistoryStore>,
    last_history_error: Option<JobHistoryStoreError>,
    latest_baseline_comparison: Option<BaselineComparisonSummary>,
    recovery_by_job: BTreeMap<JobId, RecoveredJobStatus>,
    recovery_snapshot: Option<ComputeRecoverySnapshot>,
}

impl CanonicalComputeEntryPoint {
    pub fn new(service: InMemoryComputeService) -> Self {
        Self {
            service,
            history_store: None,
            last_history_error: None,
            latest_baseline_comparison: None,
            recovery_by_job: BTreeMap::new(),
            recovery_snapshot: None,
        }
    }

    pub fn with_history_store(
        service: InMemoryComputeService,
        history_store: JobHistoryStore,
    ) -> Self {
        let mut entry = Self {
            service,
            history_store: Some(history_store),
            last_history_error: None,
            latest_baseline_comparison: None,
            recovery_by_job: BTreeMap::new(),
            recovery_snapshot: None,
        };
        entry.rehydrate_from_history();
        entry
    }

    pub fn with_history_path(
        service: InMemoryComputeService,
        path: impl Into<std::path::PathBuf>,
    ) -> Result<Self, JobHistoryStoreError> {
        let history_store = JobHistoryStore::open(path)?;
        Ok(Self::with_history_store(service, history_store))
    }

    pub fn submit(
        &mut self,
        request: ComputeSubmitRequest,
    ) -> Result<ComputeSubmitOutcome, crate::ComputeError> {
        if let Some(invalid) = validate_request(&request) {
            return Ok(ComputeSubmitOutcome::Invalid(invalid));
        }

        let submitted_at_unix_ms = request.submitted_at_unix_ms.unwrap_or_else(now_unix_ms);
        let (submit_status, submitted_job_id) = {
            let submitted = self.service.submit(
                request.pipeline_request,
                JobSubmissionMeta {
                    submitted_at_unix_ms,
                    submitted_by: request.submitted_by,
                },
            );
            (status_from_record(submitted, None), submitted.job.id)
        };
        self.persist_job(submitted_job_id);
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
                let (completion, completed_job_id) = {
                    let completion = completed.map(|record| status_from_record(record, None));
                    let completed_job_id = completed.map(|record| record.job.id);
                    (completion, completed_job_id)
                };
                if let Some(job_id) = completed_job_id {
                    self.persist_job(job_id);
                }
                Ok(ComputeSubmitOutcome::Accepted {
                    status: submit_status,
                    completion,
                })
            }
        }
    }

    pub fn status(&self, handle: ComputeJobHandle) -> Option<ComputeJobStatus> {
        self.service
            .job(handle.job_id)
            .map(|record| self.status_from_record(record))
    }

    pub fn recovery_status(&self) -> Option<&ComputeRecoverySnapshot> {
        self.recovery_snapshot.as_ref()
    }

    pub fn history_status(&self) -> ComputeHistoryStoreStatus {
        ComputeHistoryStoreStatus {
            configured: self.history_store.is_some(),
            available: self.last_history_error.is_none(),
            persisted_jobs: self.history_store.as_ref().map_or(0, JobHistoryStore::len),
            path: self
                .history_store
                .as_ref()
                .map(|store| store.path().display().to_string()),
            last_error: self.last_history_error.clone(),
        }
    }

    pub fn history_lookup(
        &self,
        handle: ComputeJobHandle,
    ) -> Result<ComputeJobHistoryLookup, ComputeHistoryLookupError> {
        if let Some(record) = self.service.job(handle.job_id) {
            let recovery = self.recovery_by_job.get(&handle.job_id);
            return Ok(ComputeJobHistoryLookup::Found(Box::new(
                PersistedJobRecord::from_job_record(record).with_recovery(
                    recovery.map(|r| r.source_job_id.0),
                    recovery.map(|r| recovery_disposition_name(r.disposition).to_string()),
                    recovery.map(|r| r.detail.clone()),
                ),
            )));
        }
        let store = self
            .history_store
            .as_ref()
            .ok_or(ComputeHistoryLookupError::StoreUnavailable)?;
        let Some(found) = store.get(handle.job_id) else {
            return Ok(ComputeJobHistoryLookup::NotFound);
        };
        Ok(ComputeJobHistoryLookup::Found(Box::new(found.clone())))
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

    pub fn replay(
        &mut self,
        handle: ComputeJobHandle,
    ) -> Result<ComputeReplayOutcome, crate::ComputeError> {
        let source = match self.replay_source(handle.job_id) {
            Some(source) => source,
            None => {
                return Ok(ComputeReplayOutcome::NotReplayable {
                    source_job_id: handle.job_id,
                    code: ReplayFailureCode::RecordMissing,
                    detail: "replay record missing".to_string(),
                })
            }
        };
        let Some(request) = source.request else {
            return Ok(ComputeReplayOutcome::NotReplayable {
                source_job_id: source.job_id,
                code: ReplayFailureCode::ConfigurationIncomplete,
                detail: "replay configuration incomplete (canonical request unavailable)"
                    .to_string(),
            });
        };
        let admission = self.service.technical_admission(&request);
        if let Some(failure) = admission.failure {
            let (code, detail) = match failure.kind {
                CanonicalFailureKind::ArtifactUnavailable
                | CanonicalFailureKind::ArtifactVerificationFailed
                | CanonicalFailureKind::ArtifactIncompatible => (
                    ReplayFailureCode::RequiredArtifactUnavailable,
                    format!(
                        "required artifact/slot no longer available: {}",
                        failure.detail
                    ),
                ),
                CanonicalFailureKind::BackendDisabled
                | CanonicalFailureKind::StageUnavailable
                | CanonicalFailureKind::NsrBackendUnavailable => (
                    ReplayFailureCode::BackendOrDeviceUnavailable,
                    format!(
                        "backend/worker/device no longer suitable: {}",
                        failure.detail
                    ),
                ),
                _ => (
                    ReplayFailureCode::ConfigurationIncomplete,
                    format!("replay configuration incomplete: {}", failure.detail),
                ),
            };
            return Ok(ComputeReplayOutcome::NotReplayable {
                source_job_id: source.job_id,
                code,
                detail,
            });
        }

        let submitted = self.service.submit(
            request,
            JobSubmissionMeta {
                submitted_at_unix_ms: now_unix_ms(),
                submitted_by: Some(format!("replay_of_job_{}", source.job_id.0)),
            },
        );
        let replay_id = submitted.job.id;
        self.persist_job(replay_id);
        let _ = self.service.run_next()?;
        self.persist_job(replay_id);
        let replayed = self
            .service
            .job(replay_id)
            .expect("replay job should be available");

        let replay_slots = replayed
            .accounting
            .model_slots
            .iter()
            .map(|slot| {
                format!(
                    "{:?}:{:?}:{}",
                    slot.slot, slot.status, slot.required_for_pack
                )
            })
            .collect::<Vec<_>>();
        let diff = ReplayConfigurationDiff {
            execution_path_match: source.execution_path == format!("{:?}", replayed.execution_path),
            execution_lane_match: source.execution_lane
                == Some(format!("{:?}", replayed.accounting.execution_lane)),
            backend_route_match: source.backend_route == replayed.result.as_ref().map(|r| r.route),
            model_slots_match: source.model_slots == replay_slots,
        };
        let replay_succeeded = replayed.state == JobLifecycleState::Completed;
        let completion_class_match = source.completion_class
            == Some(completion_class_name(replayed.accounting.completion_class).to_string());
        let failure_kind_match = source.failure_kind
            == replayed
                .accounting
                .failure_kind
                .map(|kind| canonical_failure_kind_name(kind).to_string());
        let determinism_class = if diff.execution_path_match
            && diff.execution_lane_match
            && diff.backend_route_match
            && diff.model_slots_match
        {
            ReplayDeterminismClass::SameEffectiveConfiguration
        } else if replay_succeeded {
            ReplayDeterminismClass::ReplayableNotStrictlyDeterministic
        } else {
            ReplayDeterminismClass::NotReplayableUnderCurrentRuntimeState
        };
        let replay_failure = if !replay_succeeded {
            Some(ReplayFailureCode::ReplayExecutionFailed)
        } else if determinism_class == ReplayDeterminismClass::ReplayableNotStrictlyDeterministic {
            Some(ReplayFailureCode::ReplayCompletedWithChangedConfiguration)
        } else {
            None
        };
        Ok(ComputeReplayOutcome::Completed(ComputeReplayReport {
            source_job_id: source.job_id,
            replay_job_id: replay_id,
            determinism_class,
            configuration_diff: diff,
            replay_succeeded,
            completion_class_match,
            failure_kind_match,
            replay_failure,
        }))
    }

    pub fn compare_against_baseline(
        &mut self,
        candidate: ComputeJobHandle,
        baseline: BaselineReference,
    ) -> BaselineComparisonResult {
        let Some(candidate_record) = self.replay_source(candidate.job_id) else {
            return BaselineComparisonResult::NotComparable {
                candidate_job_id: candidate.job_id,
                baseline_job_id: None,
                code: BaselineComparisonFailureCode::CandidateIncompatible,
                detail: "candidate record missing".to_string(),
            };
        };
        if candidate_record.completion_class.is_none() {
            return BaselineComparisonResult::NotComparable {
                candidate_job_id: candidate.job_id,
                baseline_job_id: None,
                code: BaselineComparisonFailureCode::CandidateIncompatible,
                detail: "candidate must be terminal before baseline comparison".to_string(),
            };
        }
        let baseline_record = match baseline {
            BaselineReference::Job(handle) => self.replay_source(handle.job_id),
            BaselineReference::LatestByRequestIdentity => {
                self.latest_baseline_for_candidate(candidate.job_id)
            }
        };
        let Some(baseline_record) = baseline_record else {
            return BaselineComparisonResult::NotComparable {
                candidate_job_id: candidate.job_id,
                baseline_job_id: None,
                code: BaselineComparisonFailureCode::NoBaselineAvailable,
                detail: "no baseline available for candidate context".to_string(),
            };
        };
        if baseline_record.job_id == candidate.job_id {
            return BaselineComparisonResult::NotComparable {
                candidate_job_id: candidate.job_id,
                baseline_job_id: Some(baseline_record.job_id),
                code: BaselineComparisonFailureCode::BaselineIncompatible,
                detail: "baseline must reference a different completed job".to_string(),
            };
        }
        if baseline_record.completion_class.is_none() {
            return BaselineComparisonResult::NotComparable {
                candidate_job_id: candidate.job_id,
                baseline_job_id: Some(baseline_record.job_id),
                code: BaselineComparisonFailureCode::BaselineIncompatible,
                detail: "baseline must be terminal before comparison".to_string(),
            };
        }

        let config_equal = candidate_record.execution_path == baseline_record.execution_path
            && candidate_record.execution_lane == baseline_record.execution_lane
            && candidate_record.backend_route == baseline_record.backend_route
            && candidate_record.model_slots == baseline_record.model_slots
            && candidate_record.request_identity == baseline_record.request_identity
            && candidate_record.request_budget == baseline_record.request_budget;
        if !config_equal {
            return BaselineComparisonResult::NotComparable {
                candidate_job_id: candidate.job_id,
                baseline_job_id: Some(baseline_record.job_id),
                code: BaselineComparisonFailureCode::NotMeaningfulUnderRuntimeChange,
                detail:
                    "candidate and baseline are not comparable under changed runtime configuration"
                        .to_string(),
            };
        }

        let completion_class_changed =
            candidate_record.completion_class != baseline_record.completion_class;
        let failure_kind_changed = candidate_record.failure_kind != baseline_record.failure_kind;
        let degraded_changed = candidate_record.pipeline_state != baseline_record.pipeline_state;
        let work_equal = candidate_record.work_summary == baseline_record.work_summary;

        let outcome = match (
            completion_rank(candidate_record.completion_class.as_deref()),
            completion_rank(baseline_record.completion_class.as_deref()),
        ) {
            (Some(c), Some(b)) if c > b => BaselineComparisonOutcome::Improved,
            (Some(c), Some(b)) if c < b => BaselineComparisonOutcome::Regressed,
            (Some(_), Some(_)) if degraded_changed || failure_kind_changed => {
                if candidate_record.pipeline_state.as_deref() == Some("degraded")
                    || candidate_record.failure_kind.is_some()
                {
                    BaselineComparisonOutcome::Regressed
                } else {
                    BaselineComparisonOutcome::Improved
                }
            }
            _ => BaselineComparisonOutcome::Equivalent,
        };

        let summary = BaselineComparisonSummary {
            candidate_job_id: candidate.job_id,
            baseline_job_id: baseline_record.job_id,
            outcome,
            completion_class_changed,
            failure_kind_changed,
            degraded_changed,
            config_equal,
            work_equal,
            candidate_remaining_global_units: candidate_record
                .work_summary
                .as_ref()
                .map(|summary| summary.global_remaining_units),
            baseline_remaining_global_units: baseline_record
                .work_summary
                .as_ref()
                .map(|summary| summary.global_remaining_units),
        };
        self.latest_baseline_comparison = Some(summary.clone());
        BaselineComparisonResult::Compared(summary)
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
            latest_baseline_comparison: self.latest_baseline_comparison.clone(),
            recovery: self.recovery_snapshot.clone(),
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
                for job_id in &completed_jobs {
                    self.persist_job(*job_id);
                }
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

    fn rehydrate_from_history(&mut self) {
        let Some(store) = self.history_store.as_ref() else {
            self.recovery_snapshot = None;
            return;
        };
        let persisted = store.records().cloned().collect::<Vec<_>>();
        let mut records = Vec::new();
        let mut resumed_jobs = 0_usize;
        let mut rerun_required_jobs = 0_usize;
        let mut uncertain_jobs = 0_usize;
        let mut failed_jobs = 0_usize;

        for record in persisted {
            let source_job_id = JobId(record.job_id);
            let lifecycle = record.lifecycle_state.clone();
            let (disposition, detail, should_resume) =
                classify_recovery_disposition(&record, source_job_id);
            let mut recovered = RecoveredJobStatus {
                source_job_id,
                source_lifecycle_state: lifecycle,
                disposition,
                resumed_as_job_id: None,
                detail,
            };
            if should_resume {
                if let Some(request) = record.canonical_request.as_ref().and_then(rebuild_request) {
                    let submitted = self.service.submit(
                        request,
                        JobSubmissionMeta {
                            submitted_at_unix_ms: now_unix_ms(),
                            submitted_by: Some(format!(
                                "recovery_resume_of_job_{}",
                                source_job_id.0
                            )),
                        },
                    );
                    let resumed_id = submitted.job.id;
                    self.recovery_by_job.insert(
                        resumed_id,
                        RecoveredJobStatus {
                            source_job_id,
                            source_lifecycle_state: "rehydrated".to_string(),
                            disposition: RecoveryDisposition::RecoveryCompletedSuccessfully,
                            resumed_as_job_id: Some(resumed_id),
                            detail: "resumed as queued job from persisted pre-execution state"
                                .to_string(),
                        },
                    );
                    self.persist_job(resumed_id);
                    recovered.disposition = RecoveryDisposition::RecoveryCompletedSuccessfully;
                    recovered.resumed_as_job_id = Some(resumed_id);
                    recovered.detail = format!("resumed as queued job {}", resumed_id.0);
                    resumed_jobs = resumed_jobs.saturating_add(1);
                } else {
                    recovered.disposition = RecoveryDisposition::RestartRecoveryFailed;
                    recovered.detail =
                        "restart recovery failed: canonical request unavailable".to_string();
                    failed_jobs = failed_jobs.saturating_add(1);
                }
            } else {
                match recovered.disposition {
                    RecoveryDisposition::RerunRequired => {
                        rerun_required_jobs = rerun_required_jobs.saturating_add(1)
                    }
                    RecoveryDisposition::ResumeUnsupported => {
                        rerun_required_jobs = rerun_required_jobs.saturating_add(1)
                    }
                    RecoveryDisposition::RunningStateUncertainAfterRestart
                    | RecoveryDisposition::LostDueToRestart => {
                        uncertain_jobs = uncertain_jobs.saturating_add(1)
                    }
                    RecoveryDisposition::RestartRecoveryFailed => {
                        failed_jobs = failed_jobs.saturating_add(1)
                    }
                    _ => {}
                }
            }
            self.recovery_by_job
                .insert(source_job_id, recovered.clone());
            records.push(recovered);
        }
        self.recovery_snapshot = Some(ComputeRecoverySnapshot {
            recovered_jobs: records.len(),
            resumed_jobs,
            rerun_required_jobs,
            uncertain_jobs,
            failed_jobs,
            records,
        });
    }

    fn replay_source(&self, job_id: JobId) -> Option<ReplaySourceRecord> {
        if let Some(record) = self.service.job(job_id) {
            return Some(ReplaySourceRecord::from_record(record));
        }
        let persisted = self.history_store.as_ref()?.get(job_id)?;
        Some(ReplaySourceRecord::from_persisted(persisted))
    }

    fn persist_job(&mut self, job_id: JobId) {
        let Some(store) = self.history_store.as_mut() else {
            return;
        };
        let Some(record) = self.service.job(job_id) else {
            return;
        };
        let recovery = self.recovery_by_job.get(&job_id);
        let persisted = PersistedJobRecord::from_job_record(record).with_recovery(
            recovery.map(|r| r.source_job_id.0),
            recovery.map(|r| recovery_disposition_name(r.disposition).to_string()),
            recovery.map(|r| r.detail.clone()),
        );
        if let Err(err) = store.upsert(persisted) {
            self.last_history_error = Some(err);
        } else {
            self.last_history_error = None;
        }
    }

    fn status_from_record(&self, record: &JobRecord) -> ComputeJobStatus {
        let recovery = self.recovery_by_job.get(&record.job.id);
        status_from_record(record, recovery)
    }

    fn latest_baseline_for_candidate(&self, candidate_job_id: JobId) -> Option<ReplaySourceRecord> {
        let candidate = self.replay_source(candidate_job_id)?;
        self.replay_sources_desc().into_iter().find(|record| {
            record.job_id != candidate_job_id
                && record.request_identity == candidate.request_identity
                && record.request_budget == candidate.request_budget
                && record.completion_class.is_some()
                && record.execution_lane == candidate.execution_lane
                && record.backend_route == candidate.backend_route
        })
    }

    fn replay_sources_desc(&self) -> Vec<ReplaySourceRecord> {
        let mut records = self
            .service
            .jobs()
            .map(ReplaySourceRecord::from_record)
            .collect::<Vec<_>>();
        if let Some(store) = self.history_store.as_ref() {
            records.extend(store.records().map(ReplaySourceRecord::from_persisted));
        }
        records.sort_by_key(|record| std::cmp::Reverse(record.job_id.0));
        records
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeHistoryStoreStatus {
    pub configured: bool,
    pub available: bool,
    pub persisted_jobs: usize,
    pub path: Option<String>,
    pub last_error: Option<JobHistoryStoreError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeJobHistoryLookup {
    Found(Box<PersistedJobRecord>),
    NotFound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeHistoryLookupError {
    StoreUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDeterminismClass {
    SameEffectiveConfiguration,
    ReplayableNotStrictlyDeterministic,
    NotReplayableUnderCurrentRuntimeState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayFailureCode {
    RecordMissing,
    ConfigurationIncomplete,
    RequiredArtifactUnavailable,
    BackendOrDeviceUnavailable,
    ReplayExecutionFailed,
    ReplayCompletedWithChangedConfiguration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayConfigurationDiff {
    pub execution_path_match: bool,
    pub execution_lane_match: bool,
    pub backend_route_match: bool,
    pub model_slots_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeReplayReport {
    pub source_job_id: JobId,
    pub replay_job_id: JobId,
    pub determinism_class: ReplayDeterminismClass,
    pub configuration_diff: ReplayConfigurationDiff,
    pub replay_succeeded: bool,
    pub completion_class_match: bool,
    pub failure_kind_match: bool,
    pub replay_failure: Option<ReplayFailureCode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeReplayOutcome {
    Completed(ComputeReplayReport),
    NotReplayable {
        source_job_id: JobId,
        code: ReplayFailureCode,
        detail: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineReference {
    Job(ComputeJobHandle),
    LatestByRequestIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineComparisonOutcome {
    Improved,
    Equivalent,
    Regressed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineComparisonFailureCode {
    NoBaselineAvailable,
    BaselineIncompatible,
    CandidateIncompatible,
    ComparisonExecutionFailed,
    NotMeaningfulUnderRuntimeChange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineComparisonSummary {
    pub candidate_job_id: JobId,
    pub baseline_job_id: JobId,
    pub outcome: BaselineComparisonOutcome,
    pub completion_class_changed: bool,
    pub failure_kind_changed: bool,
    pub degraded_changed: bool,
    pub config_equal: bool,
    pub work_equal: bool,
    pub candidate_remaining_global_units: Option<u64>,
    pub baseline_remaining_global_units: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaselineComparisonResult {
    Compared(BaselineComparisonSummary),
    NotComparable {
        candidate_job_id: JobId,
        baseline_job_id: Option<JobId>,
        code: BaselineComparisonFailureCode,
        detail: String,
    },
}

type BudgetComparisonFingerprint = (u64, u64, u64, u64, u64, u64, u64, u32, String);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReplaySourceRecord {
    job_id: JobId,
    request: Option<CanonicalPipelineRequest>,
    request_identity: Option<(u64, u64, String)>,
    request_budget: Option<BudgetComparisonFingerprint>,
    execution_path: String,
    execution_lane: Option<String>,
    backend_route: Option<crate::pipeline::CanonicalBackendRoute>,
    model_slots: Vec<String>,
    completion_class: Option<String>,
    failure_kind: Option<String>,
    pipeline_state: Option<String>,
    work_summary: Option<CanonicalWorkSummary>,
}

impl ReplaySourceRecord {
    fn from_record(record: &JobRecord) -> Self {
        let request = record.job.request.clone();
        Self {
            job_id: record.job.id,
            request_identity: Some((
                request.input.frame_id.0,
                request.input.t,
                hex::encode(request.input.context_digest),
            )),
            request_budget: Some((
                request.budget.max_micros,
                request.budget.hard_timeout_micros,
                request.budget.global_work_units,
                request.budget.world_units,
                request.budget.sae_units,
                request.budget.ssm_units,
                request.budget.lfm_units,
                request.budget.profile_id,
                format!("{:?}", request.budget.degrade_policy),
            )),
            request: Some(request),
            execution_path: format!("{:?}", record.execution_path),
            execution_lane: Some(format!("{:?}", record.accounting.execution_lane)),
            backend_route: record.result.as_ref().map(|result| result.route),
            model_slots: record
                .accounting
                .model_slots
                .iter()
                .map(|slot| {
                    format!(
                        "{:?}:{:?}:{}",
                        slot.slot, slot.status, slot.required_for_pack
                    )
                })
                .collect(),
            completion_class: Some(
                completion_class_name(record.accounting.completion_class).to_string(),
            ),
            failure_kind: record
                .accounting
                .failure_kind
                .map(|kind| canonical_failure_kind_name(kind).to_string()),
            pipeline_state: record
                .accounting
                .pipeline_state
                .map(pipeline_state_name)
                .map(str::to_string),
            work_summary: record.accounting.work_summary,
        }
    }

    fn from_persisted(persisted: &PersistedJobRecord) -> Self {
        Self {
            job_id: JobId(persisted.job_id),
            request: persisted
                .canonical_request
                .as_ref()
                .and_then(canonical_request_from_persisted),
            request_identity: Some((
                persisted.request.frame_id,
                persisted.request.t,
                persisted.request.context_digest_hex.clone(),
            )),
            request_budget: persisted.canonical_request.as_ref().map(|request| {
                (
                    request.budget.max_micros,
                    request.budget.hard_timeout_micros,
                    request.budget.global_work_units,
                    request.budget.world_units,
                    request.budget.sae_units,
                    request.budget.ssm_units,
                    request.budget.lfm_units,
                    request.budget.profile_id,
                    request.budget.degrade_policy.clone(),
                )
            }),
            execution_path: persisted.execution_path.clone(),
            execution_lane: persisted.execution_lane.clone(),
            backend_route: persisted.backend_route.as_ref().map(|route| {
                crate::pipeline::CanonicalBackendRoute {
                    pack_id: route.pack_id,
                    world_backend: route.world_backend,
                    sae_backend: route.sae_backend,
                    ssm_backend: route.ssm_backend,
                    lfm_backend: route.lfm_backend,
                }
            }),
            model_slots: persisted
                .model_slots
                .iter()
                .map(|slot| format!("{}:{}:{}", slot.slot, slot.status, slot.required_for_pack))
                .collect(),
            completion_class: persisted.completion_class.clone(),
            failure_kind: persisted.failure_kind.clone(),
            pipeline_state: persisted.pipeline_state.clone(),
            work_summary: persisted
                .work_summary
                .as_ref()
                .map(|summary| CanonicalWorkSummary {
                    global_budget_units: summary.global_budget_units,
                    global_remaining_units: summary.global_remaining_units,
                    world_remaining_units: summary.world_remaining_units,
                    sae_remaining_units: summary.sae_remaining_units,
                    ssm_remaining_units: summary.ssm_remaining_units,
                    lfm_remaining_units: summary.lfm_remaining_units,
                    budget_exceeded_stage: None,
                }),
        }
    }
}

fn canonical_request_from_persisted(
    request: &PersistedCanonicalRequest,
) -> Option<CanonicalPipelineRequest> {
    let context_digest = hex::decode(&request.context_digest_hex).ok()?;
    if context_digest.len() != 32 {
        return None;
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&context_digest);
    let degrade_policy = match request.budget.degrade_policy.as_str() {
        "DegradeStages" => crate::DegradePolicy::DegradeStages,
        "FailFast" => crate::DegradePolicy::FailFast,
        _ => return None,
    };
    Some(CanonicalPipelineRequest {
        input: crate::ComputeInput {
            frame_id: crate::FrameId(request.frame_id),
            t: request.t,
            context_digest: digest,
        },
        budget: crate::ComputeBudget {
            max_micros: request.budget.max_micros,
            hard_timeout_micros: request.budget.hard_timeout_micros,
            seed: request.budget.seed,
            profile_id: request.budget.profile_id,
            global_work_units: request.budget.global_work_units,
            world_units: request.budget.world_units,
            sae_units: request.budget.sae_units,
            ssm_units: request.budget.ssm_units,
            lfm_units: request.budget.lfm_units,
            degrade_policy,
            governor_tier: request.budget.governor_tier,
        },
    })
}

fn canonical_failure_kind_name(kind: CanonicalFailureKind) -> &'static str {
    match kind {
        CanonicalFailureKind::InvalidInput => "invalid_input",
        CanonicalFailureKind::BackendDisabled => "backend_disabled",
        CanonicalFailureKind::ContractMismatch => "contract_mismatch",
        CanonicalFailureKind::StageContractMismatch => "stage_contract_mismatch",
        CanonicalFailureKind::ArtifactUnavailable => "artifact_unavailable",
        CanonicalFailureKind::ArtifactVerificationFailed => "artifact_verification_failed",
        CanonicalFailureKind::ArtifactIncompatible => "artifact_incompatible",
        CanonicalFailureKind::StageUnavailable => "stage_unavailable",
        CanonicalFailureKind::DegradedFallback => "degraded_fallback",
        CanonicalFailureKind::ValidationDegraded => "validation_degraded",
        CanonicalFailureKind::BudgetExceeded => "budget_exceeded",
        CanonicalFailureKind::Timeout => "timeout",
        CanonicalFailureKind::ExecutionError => "execution_error",
        CanonicalFailureKind::NsrDisabled => "nsr_disabled",
        CanonicalFailureKind::NsrUnavailable => "nsr_unavailable",
        CanonicalFailureKind::NsrArtifactVerificationFailed => "nsr_artifact_verification_failed",
        CanonicalFailureKind::NsrContractMismatch => "nsr_contract_mismatch",
        CanonicalFailureKind::NsrBackendUnavailable => "nsr_backend_unavailable",
        CanonicalFailureKind::NsrExecutionError => "nsr_execution_error",
    }
}

fn completion_class_name(class: JobCompletionClass) -> &'static str {
    match class {
        JobCompletionClass::RejectedBeforeExecution => "rejected_before_execution",
        JobCompletionClass::Completed => "completed",
        JobCompletionClass::DegradedCompleted => "degraded_completed",
        JobCompletionClass::FailedDuringExecution => "failed_during_execution",
        JobCompletionClass::TimedOut => "timed_out",
        JobCompletionClass::WorkerIpcFailure => "worker_ipc_failure",
    }
}

fn pipeline_state_name(state: CanonicalPipelineState) -> &'static str {
    match state {
        CanonicalPipelineState::Ok => "ok",
        CanonicalPipelineState::Degraded => "degraded",
        CanonicalPipelineState::Unavailable => "unavailable",
    }
}

fn completion_rank(class: Option<&str>) -> Option<u8> {
    match class {
        Some("completed") => Some(4),
        Some("degraded_completed") => Some(3),
        Some("timed_out") => Some(2),
        Some("failed_during_execution") | Some("worker_ipc_failure") => Some(1),
        Some("rejected_before_execution") => Some(0),
        _ => None,
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

fn status_from_record(
    record: &JobRecord,
    recovery: Option<&RecoveredJobStatus>,
) -> ComputeJobStatus {
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
        recovery_disposition: recovery.map(|r| r.disposition),
        recovery_source_job_id: recovery.map(|r| r.source_job_id),
    }
}

fn recovery_disposition_name(disposition: RecoveryDisposition) -> &'static str {
    match disposition {
        RecoveryDisposition::CompletedBeforeRestart => "completed_before_restart",
        RecoveryDisposition::PersistedNotYetResumed => "persisted_not_yet_resumed",
        RecoveryDisposition::RunningStateUncertainAfterRestart => {
            "running_state_uncertain_after_restart"
        }
        RecoveryDisposition::Resumable => "resumable",
        RecoveryDisposition::ResumeUnsupported => "resume_unsupported",
        RecoveryDisposition::RerunRequired => "rerun_required",
        RecoveryDisposition::LostDueToRestart => "lost_due_to_restart",
        RecoveryDisposition::RecoveryCompletedSuccessfully => "recovery_completed_successfully",
        RecoveryDisposition::RestartRecoveryFailed => "restart_recovery_failed",
    }
}

fn classify_recovery_disposition(
    record: &PersistedJobRecord,
    source_job_id: JobId,
) -> (RecoveryDisposition, String, bool) {
    let has_request = record.canonical_request.is_some();
    match record.lifecycle_state.as_str() {
        "completed" | "failed" | "timed_out" | "rejected" => (
            RecoveryDisposition::CompletedBeforeRestart,
            "job already terminal before restart".to_string(),
            false,
        ),
        "running" => {
            if has_request {
                (
                    RecoveryDisposition::RunningStateUncertainAfterRestart,
                    format!(
                        "job {id} was running at restart; worker status uncertain, rerun required",
                        id = source_job_id.0
                    ),
                    false,
                )
            } else {
                (
                    RecoveryDisposition::LostDueToRestart,
                    format!(
                        "job {id} was running and is not resumable without canonical request",
                        id = source_job_id.0
                    ),
                    false,
                )
            }
        }
        "submitted" | "admitted" | "queued" => {
            if has_request {
                (
                    RecoveryDisposition::Resumable,
                    "persisted pre-execution state can be resumed".to_string(),
                    true,
                )
            } else {
                (
                    RecoveryDisposition::ResumeUnsupported,
                    "resume unsupported: canonical request missing, rerun required".to_string(),
                    false,
                )
            }
        }
        _ => (
            RecoveryDisposition::RerunRequired,
            "unknown persisted lifecycle state after restart; rerun required".to_string(),
            false,
        ),
    }
}

fn rebuild_request(request: &PersistedCanonicalRequest) -> Option<CanonicalPipelineRequest> {
    let context_digest = hex::decode(&request.context_digest_hex).ok()?;
    if context_digest.len() != 32 {
        return None;
    }
    let mut digest = [0_u8; 32];
    digest.copy_from_slice(&context_digest[..32]);
    Some(CanonicalPipelineRequest {
        input: crate::ComputeInput {
            frame_id: crate::FrameId(request.frame_id),
            t: request.t,
            context_digest: digest,
        },
        budget: crate::ComputeBudget {
            max_micros: request.budget.max_micros,
            hard_timeout_micros: request.budget.hard_timeout_micros,
            seed: request.budget.seed,
            profile_id: request.budget.profile_id,
            global_work_units: request.budget.global_work_units,
            world_units: request.budget.world_units,
            sae_units: request.budget.sae_units,
            ssm_units: request.budget.ssm_units,
            lfm_units: request.budget.lfm_units,
            degrade_policy: parse_degrade_policy(&request.budget.degrade_policy)?,
            governor_tier: request.budget.governor_tier,
        },
    })
}

fn parse_degrade_policy(value: &str) -> Option<crate::DegradePolicy> {
    match value {
        "DegradeStages" => Some(crate::DegradePolicy::DegradeStages),
        "FailFast" => Some(crate::DegradePolicy::FailFast),
        _ => None,
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
        BaselineComparisonFailureCode, BaselineComparisonResult, BaselineReference,
        CanonicalComputeEntryPoint, ComputeExecutionMode, ComputeHistoryLookupError,
        ComputeJobHandle, ComputeJobHistoryLookup, ComputeReplayOutcome,
        ComputeRequestValidationCode, ComputeSubmitOutcome, ComputeSubmitRequest,
        RecoveryDisposition, ReplayDeterminismClass, ReplayFailureCode, RuntimeOperation,
        RuntimeOperationCode, RuntimeOpsState, RuntimeSignalState,
    };
    use crate::pipeline::{CanonicalFailureKind, CanonicalPipelineRequest};
    use crate::{InMemoryComputeService, JobHistoryStore, JobId, JobLifecycleState};

    fn service() -> CanonicalComputeEntryPoint {
        CanonicalComputeEntryPoint::new(InMemoryComputeService::new(
            crate::pipeline::ComputePipelineBackend::stub(),
        ))
    }

    fn service_with_history(path: &std::path::Path) -> CanonicalComputeEntryPoint {
        let store = JobHistoryStore::open(path).expect("history store should open");
        CanonicalComputeEntryPoint::with_history_store(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            store,
        )
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

    #[test]
    fn completed_jobs_are_persisted_in_history_lookup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        let mut entry = service_with_history(&history_path);
        let outcome = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("submit should not fail");
        let handle = match outcome {
            ComputeSubmitOutcome::Accepted { completion, .. } => {
                completion.expect("completion should exist").handle
            }
            other => panic!("expected accepted, got {other:?}"),
        };
        let lookup = entry.history_lookup(handle).expect("lookup should succeed");
        match lookup {
            ComputeJobHistoryLookup::Found(record) => {
                assert_eq!(record.job_id, handle.job_id.0);
                assert!(record.completion_class.is_some());
                assert!(record.finished_at_unix_ms.is_some());
            }
            ComputeJobHistoryLookup::NotFound => panic!("history record expected"),
        }
    }

    #[test]
    fn rejected_and_failed_jobs_keep_failure_summary_in_history() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        let mut entry = service_with_history(&history_path);
        let mut rejected = valid_request();
        rejected.input.t = 0;
        let outcome = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: rejected,
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::EnqueueOnly,
            })
            .expect("submit should not fail");
        let handle = match outcome {
            ComputeSubmitOutcome::Rejected { status } => status.handle,
            other => panic!("expected rejected, got {other:?}"),
        };
        let lookup = entry.history_lookup(handle).expect("lookup should succeed");
        match lookup {
            ComputeJobHistoryLookup::Found(record) => {
                assert_eq!(record.lifecycle_state, "rejected");
                assert_eq!(record.failure_kind.as_deref(), Some("invalid_input"));
            }
            ComputeJobHistoryLookup::NotFound => panic!("history record expected"),
        }
    }

    #[test]
    fn history_lookup_reports_store_unavailable_and_not_found_distinctly() {
        let entry = service();
        let error = entry
            .history_lookup(ComputeJobHandle { job_id: JobId(999) })
            .expect_err("missing store should error");
        assert_eq!(error, ComputeHistoryLookupError::StoreUnavailable);

        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        let entry = service_with_history(&history_path);
        let lookup = entry
            .history_lookup(ComputeJobHandle { job_id: JobId(999) })
            .expect("configured store should not error");
        assert_eq!(lookup, ComputeJobHistoryLookup::NotFound);
    }

    #[test]
    fn persistence_failure_is_exposed_without_hiding_job_execution() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        let mut entry = service_with_history(&history_path);
        std::fs::remove_file(&history_path).ok();
        std::fs::create_dir(&history_path).expect("create directory to force append-open failure");

        let outcome = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("execution should still complete");
        assert!(matches!(outcome, ComputeSubmitOutcome::Accepted { .. }));
        let history = entry.history_status();
        assert!(history.configured);
        assert!(!history.available);
        assert!(history.last_error.is_some());
    }

    #[test]
    fn replay_runs_through_canonical_path_and_links_source_job() {
        let mut entry = service();
        let outcome = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("submit should not fail");
        let source = match outcome {
            ComputeSubmitOutcome::Accepted { completion, .. } => {
                completion.expect("completion should exist").handle
            }
            other => panic!("expected accepted, got {other:?}"),
        };
        let replay = entry.replay(source).expect("replay call");
        match replay {
            ComputeReplayOutcome::Completed(report) => {
                assert_eq!(report.source_job_id, source.job_id);
                assert_ne!(report.replay_job_id, source.job_id);
                assert!(matches!(
                    report.determinism_class,
                    ReplayDeterminismClass::SameEffectiveConfiguration
                        | ReplayDeterminismClass::ReplayableNotStrictlyDeterministic
                        | ReplayDeterminismClass::NotReplayableUnderCurrentRuntimeState
                ));
            }
            other => panic!("expected completed replay, got {other:?}"),
        }
    }

    #[test]
    fn replay_reports_missing_record() {
        let mut entry = service();
        let replay = entry
            .replay(ComputeJobHandle { job_id: JobId(42) })
            .expect("replay");
        match replay {
            ComputeReplayOutcome::NotReplayable { code, .. } => {
                assert_eq!(code, ReplayFailureCode::RecordMissing);
            }
            other => panic!("expected non-replayable, got {other:?}"),
        }
    }

    #[test]
    fn replay_from_legacy_history_without_request_is_configuration_incomplete() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            r#"{"schema_version":1,"job_id":7,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"lifecycle_state":"completed","completion_class":"completed","execution_path":"LocalCanonical","submitted_at_unix_ms":1,"started_at_unix_ms":2,"finished_at_unix_ms":3,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","work_summary":null,"model_slots":[]}"#,
        )
        .expect("history fixture");
        let mut entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            &history_path,
        )
        .expect("entry with history");
        let replay = entry
            .replay(ComputeJobHandle { job_id: JobId(7) })
            .expect("replay");
        match replay {
            ComputeReplayOutcome::NotReplayable { code, .. } => {
                assert_eq!(code, ReplayFailureCode::ConfigurationIncomplete);
            }
            other => panic!("expected non-replayable, got {other:?}"),
        }
    }

    #[test]
    fn recovery_rehydrates_queued_jobs_and_marks_running_jobs_uncertain() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            concat!(
                r#"{"schema_version":3,"job_id":10,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"queued","completion_class":null,"execution_path":"LocalCanonical","submitted_at_unix_ms":1,"started_at_unix_ms":null,"finished_at_unix_ms":null,"queue_wait_ms":null,"execution_duration_micros":null,"total_duration_ms":null,"failure_kind":null,"pipeline_state":null,"work_summary":null,"model_slots":[]}"#,
                "\n",
                r#"{"schema_version":3,"job_id":11,"submitted_by":"svc","request":{"frame_id":2,"t":10,"context_digest_hex":"0202020202020202020202020202020202020202020202020202020202020202"},"canonical_request":{"frame_id":2,"t":10,"context_digest_hex":"0202020202020202020202020202020202020202020202020202020202020202","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":10,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"running","completion_class":null,"execution_path":"WorkerIpc","submitted_at_unix_ms":2,"started_at_unix_ms":3,"finished_at_unix_ms":null,"queue_wait_ms":1,"execution_duration_micros":null,"total_duration_ms":null,"failure_kind":null,"pipeline_state":null,"work_summary":null,"model_slots":[]}"#
            ),
        )
        .expect("history fixture");
        let entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            &history_path,
        )
        .expect("entry with history");

        let recovery = entry.recovery_status().expect("recovery status");
        assert_eq!(recovery.recovered_jobs, 2);
        assert_eq!(recovery.resumed_jobs, 1);
        assert_eq!(recovery.uncertain_jobs, 1);
        assert!(recovery
            .records
            .iter()
            .any(|record| record.source_job_id == JobId(10) && record.resumed_as_job_id.is_some()));
        assert!(recovery.records.iter().any(|record| {
            record.source_job_id == JobId(11)
                && record.disposition == RecoveryDisposition::RunningStateUncertainAfterRestart
        }));
    }

    #[test]
    fn recovery_status_is_carried_into_job_status_and_history() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            r#"{"schema_version":3,"job_id":20,"submitted_by":"svc","request":{"frame_id":3,"t":12,"context_digest_hex":"0303030303030303030303030303030303030303030303030303030303030303"},"canonical_request":{"frame_id":3,"t":12,"context_digest_hex":"0303030303030303030303030303030303030303030303030303030303030303","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":12,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"queued","completion_class":null,"execution_path":"LocalCanonical","submitted_at_unix_ms":1,"started_at_unix_ms":null,"finished_at_unix_ms":null,"queue_wait_ms":null,"execution_duration_micros":null,"total_duration_ms":null,"failure_kind":null,"pipeline_state":null,"work_summary":null,"model_slots":[]}"#,
        )
        .expect("history fixture");
        let entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            &history_path,
        )
        .expect("entry with history");
        let resumed = entry
            .recovery_status()
            .expect("recovery")
            .records
            .iter()
            .find_map(|record| record.resumed_as_job_id)
            .expect("resumed job id");
        let status = entry
            .status(ComputeJobHandle { job_id: resumed })
            .expect("status");
        assert_eq!(
            status.recovery_disposition,
            Some(RecoveryDisposition::RecoveryCompletedSuccessfully)
        );
        let history = entry
            .history_lookup(ComputeJobHandle { job_id: resumed })
            .expect("lookup");
        match history {
            ComputeJobHistoryLookup::Found(record) => {
                assert_eq!(
                    record.recovery_status.as_deref(),
                    Some("recovery_completed_successfully")
                );
                assert_eq!(record.recovery_source_job_id, Some(20));
            }
            ComputeJobHistoryLookup::NotFound => panic!("expected recovered record"),
        }
    }

    #[test]
    fn candidate_can_be_compared_against_explicit_baseline() {
        let mut entry = service();
        let baseline = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("baseline submit");
        let baseline_handle = match baseline {
            ComputeSubmitOutcome::Accepted { completion, .. } => {
                completion.expect("completion").handle
            }
            other => panic!("expected accepted baseline, got {other:?}"),
        };
        let candidate = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(101),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("candidate submit");
        let candidate_handle = match candidate {
            ComputeSubmitOutcome::Accepted { completion, .. } => {
                completion.expect("completion").handle
            }
            other => panic!("expected accepted candidate, got {other:?}"),
        };
        let compare = entry
            .compare_against_baseline(candidate_handle, BaselineReference::Job(baseline_handle));
        match compare {
            BaselineComparisonResult::Compared(summary) => {
                assert_eq!(summary.candidate_job_id, candidate_handle.job_id);
                assert_eq!(summary.baseline_job_id, baseline_handle.job_id);
                assert!(summary.config_equal);
            }
            other => panic!("expected compared result, got {other:?}"),
        }
    }

    #[test]
    fn missing_or_incompatible_baseline_is_structured() {
        let mut entry = service();
        let candidate = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("candidate submit");
        let candidate_handle = match candidate {
            ComputeSubmitOutcome::Accepted { completion, .. } => {
                completion.expect("completion").handle
            }
            other => panic!("expected accepted candidate, got {other:?}"),
        };

        let missing = entry.compare_against_baseline(
            candidate_handle,
            BaselineReference::Job(ComputeJobHandle {
                job_id: JobId(9999),
            }),
        );
        match missing {
            BaselineComparisonResult::NotComparable { code, .. } => {
                assert_eq!(code, BaselineComparisonFailureCode::NoBaselineAvailable);
            }
            other => panic!("expected not-comparable result, got {other:?}"),
        }

        let mut changed = valid_request();
        changed.budget.profile_id = 2;
        let baseline = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: changed,
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(101),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("changed baseline submit");
        let baseline_handle = match baseline {
            ComputeSubmitOutcome::Accepted { completion, .. } => {
                completion.expect("completion").handle
            }
            other => panic!("expected accepted baseline, got {other:?}"),
        };
        let changed_compare = entry
            .compare_against_baseline(candidate_handle, BaselineReference::Job(baseline_handle));
        match changed_compare {
            BaselineComparisonResult::NotComparable { code, .. } => {
                assert_eq!(
                    code,
                    BaselineComparisonFailureCode::NotMeaningfulUnderRuntimeChange
                );
            }
            other => panic!("expected not-comparable changed-config result, got {other:?}"),
        }
    }

    #[test]
    fn baseline_comparison_is_visible_in_runtime_snapshot() {
        let mut entry = service();
        let baseline = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(100),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("baseline submit");
        let baseline_handle = match baseline {
            ComputeSubmitOutcome::Accepted { completion, .. } => {
                completion.expect("completion").handle
            }
            other => panic!("expected accepted baseline, got {other:?}"),
        };
        let candidate = entry
            .submit(ComputeSubmitRequest {
                pipeline_request: valid_request(),
                submitted_by: Some("svc-test".to_string()),
                submitted_at_unix_ms: Some(101),
                execution_mode: ComputeExecutionMode::ExecuteInline,
            })
            .expect("candidate submit");
        let candidate_handle = match candidate {
            ComputeSubmitOutcome::Accepted { completion, .. } => {
                completion.expect("completion").handle
            }
            other => panic!("expected accepted candidate, got {other:?}"),
        };

        let _ = entry
            .compare_against_baseline(candidate_handle, BaselineReference::Job(baseline_handle));
        let snapshot = entry.operations_snapshot();
        assert_eq!(
            snapshot
                .latest_baseline_comparison
                .as_ref()
                .map(|summary| summary.candidate_job_id),
            Some(candidate_handle.job_id)
        );
    }
}
