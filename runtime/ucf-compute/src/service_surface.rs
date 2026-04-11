//! Canonical service/reference surface over the bounded real-compute core.
//!
//! Terminology and boundary conventions used by load-bearing paths:
//! - `request`: external submission envelope (`ComputeSubmitRequest`).
//! - `job`: admitted runtime unit with lifecycle/accounting (`ComputeJobStatus`).
//! - `run`: execution attempt of a job through canonical pipeline execution.
//! - `replay`: history-backed rerun/re-evaluation path (`ComputeReplay*` types).
//!
//! Rollout/runtime context conventions:
//! - `active`: productive path.
//! - `candidate|compare|shadow`: non-primary rollout/diagnostic side paths.
//! - `degraded|unavailable|failed`: canonical state/failure distinctions from
//!   pipeline/service contracts (not interchangeable synonyms).
//!
use crate::compute_service::{
    InMemoryComputeService, JobCompletionClass, JobExecutionPath, JobId, JobLifecycleEvent,
    JobLifecycleState, JobRecord, JobSubmissionMeta, ResourceClass,
};
use crate::job_history::{
    JobHistoryStore, JobHistoryStoreError, PersistedCanonicalRequest, PersistedJobRecord,
    PersistedSnapshotReadiness,
};
use crate::pipeline::{
    classify_failure_kind, CanonicalFailureKind, CanonicalFaultDomain, CanonicalHotspotSummary,
    CanonicalIsolationDisposition, CanonicalPipelineFailure, CanonicalPipelineRequest,
    CanonicalPipelineState, CanonicalStageCostAttribution, CanonicalStageId, CanonicalWorkSummary,
};
use crate::{
    DeploymentProfile, ModelSlot, ModelSlotProvenance, RuntimeDiagnosticFlags, RuntimeMode,
    RuntimeProfile, SlotRuntimeStatus,
};
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
    pub fault_domain: Option<CanonicalFaultDomain>,
    pub fault_isolation: Option<CanonicalIsolationDisposition>,
    pub fault_systemic: Option<bool>,
    pub pipeline_state: Option<CanonicalPipelineState>,
    pub work_summary: Option<CanonicalWorkSummary>,
    pub stage_cost_attribution: Vec<CanonicalStageCostAttribution>,
    pub hotspot_summary: Option<CanonicalHotspotSummary>,
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
    pub warmup_state: RuntimeWarmupState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeWarmupState {
    Cold,
    Preparing,
    Ready,
    Stale,
    Blocked,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeOpsSnapshot {
    pub state: RuntimeOpsState,
    pub runtime_mode: RuntimeMode,
    pub deployment_profile: DeploymentProfile,
    pub diagnostic_flags: RuntimeDiagnosticFlags,
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
    pub repeated_hotspot_stage: Option<CanonicalStageId>,
    pub repeated_hotspot_runs: usize,
    pub optimization_view: RuntimeOptimizationOpsView,
    pub replay_snapshot_coverage: ReplaySnapshotCoverage,
    pub recovery: Option<ComputeRecoverySnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplaySnapshotCoverage {
    pub replay_ready: usize,
    pub partial: usize,
    pub insufficient: usize,
    pub stale_or_incomplete: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeOptimizationOpsView {
    pub current_state: String,
    pub main_bottleneck: String,
    pub queue_pressure: bool,
    pub capacity_pressure: bool,
    pub cold_or_warmup_pressure: bool,
    pub stage_hotspot_pressure: bool,
    pub mixed_bottlenecks: bool,
    pub historical_feedback_alignment: String,
    pub caveats: Vec<String>,
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
        let preflight = self.replay_preflight(handle);
        if matches!(
            preflight.replayability,
            ReplayabilityClass::BlockedForReplay | ReplayabilityClass::InsufficientForReplay
        ) {
            let (code, detail) = replay_preflight_failure(&preflight);
            return Ok(ComputeReplayOutcome::NotReplayable {
                source_job_id: preflight.source_job_id,
                code,
                detail,
            });
        }
        let source = match self.replay_source(preflight.source_job_id) {
            Some(source) => source,
            None => {
                return Ok(ComputeReplayOutcome::NotReplayable {
                    source_job_id: preflight.source_job_id,
                    code: ReplayFailureCode::RecordMissing,
                    detail: "replay record missing".to_string(),
                })
            }
        };
        let source_execution_mode = source.execution_mode();
        let Some(request) = source.request.clone() else {
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
            resource_class_match: source.resource_class == Some(replayed.accounting.resource_class),
            capacity_pressure_match: source.capacity_pressure
                == Some(format!("{:?}", replayed.accounting.capacity_pressure)),
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
            && diff.resource_class_match
            && diff.capacity_pressure_match
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
        let replay_execution_mode = if replayed.execution_path == JobExecutionPath::WorkerIpc {
            ReplayExecutionMode::RemoteWorkerIpc
        } else {
            ReplayExecutionMode::Local
        };
        let replay_context = ReplayExecutionContextDescriptor {
            execution_mode: replay_execution_mode,
            execution_path: format!("{:?}", replayed.execution_path),
            execution_lane: Some(format!("{:?}", replayed.accounting.execution_lane)),
            resource_class: Some(format!("{:?}", replayed.accounting.resource_class)),
            capacity_pressure: Some(format!("{:?}", replayed.accounting.capacity_pressure)),
            has_backend_route: replayed.result.is_some(),
            remote_context_completeness: if replay_execution_mode
                == ReplayExecutionMode::RemoteWorkerIpc
                && replayed.result.is_some()
            {
                "complete".to_string()
            } else if replay_execution_mode == ReplayExecutionMode::RemoteWorkerIpc {
                "partial".to_string()
            } else {
                "not_applicable".to_string()
            },
        };
        let source_context = source.context_descriptor();
        let context_bridge = build_context_bridge_summary(&source_context, &replay_context);
        let rollout_context = classify_rollout_context_comparability(
            source.rollout_context_hint.as_deref(),
            derive_live_rollout_context_hint(replayed).as_deref(),
            false,
            source.snapshot_readiness == PersistedSnapshotReadiness::Partial,
            !replay_succeeded,
            false,
        );
        let context_consistency_class = classify_context_consistency(
            &context_bridge,
            replay_succeeded,
            source.snapshot_readiness == PersistedSnapshotReadiness::Partial,
        );
        let remote_context_reproducibility = match (
            source_execution_mode,
            replay_execution_mode,
            diff.execution_lane_match
                && diff.backend_route_match
                && diff.model_slots_match
                && diff.resource_class_match
                && diff.capacity_pressure_match,
        ) {
            (ReplayExecutionMode::Local, ReplayExecutionMode::Local, _) => {
                ReplayRemoteContextReproducibility::NotApplicableLocal
            }
            (ReplayExecutionMode::RemoteWorkerIpc, ReplayExecutionMode::RemoteWorkerIpc, true) => {
                ReplayRemoteContextReproducibility::Exact
            }
            (ReplayExecutionMode::RemoteWorkerIpc, _, false) => {
                ReplayRemoteContextReproducibility::Partial
            }
            _ => ReplayRemoteContextReproducibility::Missing,
        };
        Ok(ComputeReplayOutcome::Completed(ComputeReplayReport {
            source_job_id: source.job_id,
            replay_job_id: replay_id,
            determinism_class,
            source_execution_mode,
            replay_execution_mode,
            remote_context_reproducibility,
            context_consistency_class,
            context_bridge,
            rollout_context,
            configuration_diff: diff,
            replay_succeeded,
            completion_class_match,
            failure_kind_match,
            replay_failure,
        }))
    }

    pub fn replay_preflight(&self, handle: ComputeJobHandle) -> ComputeReplayPreflight {
        let ops = self.operations_snapshot();
        let current_mode = current_replay_mode(ops.execution_path);
        let current_context = current_context_descriptor(&ops);
        let Some(source) = self.replay_source(handle.job_id) else {
            let source_context = ReplayExecutionContextDescriptor {
                execution_mode: current_mode,
                execution_path: "unknown".to_string(),
                execution_lane: None,
                resource_class: None,
                capacity_pressure: None,
                has_backend_route: false,
                remote_context_completeness: "unavailable".to_string(),
            };
            let context_bridge = build_context_bridge_summary(&source_context, &current_context);
            return ComputeReplayPreflight {
                source_job_id: handle.job_id,
                replayability: ReplayabilityClass::BlockedForReplay,
                source_execution_mode: current_mode,
                current_execution_mode: current_mode,
                snapshot_readiness: None,
                locality: ReplayPreflightLocality::ChangedContextOnly,
                context_consistency_class: ReplayContextConsistencyClass::NotMeaningfullyComparable,
                context_bridge,
                rollout_context: RolloutReplayComparisonContext {
                    source: RolloutReplayContextClass::Unavailable,
                    replay: RolloutReplayContextClass::Unavailable,
                    source_hint: None,
                    replay_hint: None,
                    comparability: RolloutReplayComparability::BlockedInsufficientRolloutContext,
                },
                fidelity_equivalent_possible: false,
                issues: vec![ReplayPreflightIssue {
                    code: ReplayPreflightIssueCode::RecordMissing,
                    detail: "replay record missing".to_string(),
                }],
            };
        };

        let mut issues = Vec::new();
        let source_mode = source.execution_mode();
        if matches!(
            source.snapshot_readiness,
            PersistedSnapshotReadiness::Insufficient
                | PersistedSnapshotReadiness::StaleOrIncomplete
        ) {
            issues.push(ReplayPreflightIssue {
                code: ReplayPreflightIssueCode::SnapshotIncomplete,
                detail: "replay snapshot is insufficient or stale for load-bearing replay"
                    .to_string(),
            });
        }
        if source.request.is_none() {
            issues.push(ReplayPreflightIssue {
                code: ReplayPreflightIssueCode::CanonicalRequestMissing,
                detail: "canonical request unavailable for replay".to_string(),
            });
        }
        if source_mode == ReplayExecutionMode::RemoteWorkerIpc && !source.has_remote_context() {
            issues.push(ReplayPreflightIssue {
                code: ReplayPreflightIssueCode::MissingRemoteExecutionContext,
                detail: "source remote execution context is incomplete".to_string(),
            });
            issues.push(ReplayPreflightIssue {
                code: ReplayPreflightIssueCode::OriginalContextUnavailable,
                detail: "original remote execution context unavailable for context bridge"
                    .to_string(),
            });
        }
        if source_mode != current_mode {
            issues.push(ReplayPreflightIssue {
                code: ReplayPreflightIssueCode::LocalRemoteConstraintMismatch,
                detail: format!(
                    "current runtime execution mode changed from {:?} to {:?}",
                    source_mode, current_mode
                ),
            });
            issues.push(ReplayPreflightIssue {
                code: ReplayPreflightIssueCode::AlternativeContextWithCaveats,
                detail:
                    "replay requires local-vs-remote context bridge; diagnostics will be caveated"
                        .to_string(),
            });
        }
        if let (Some(source_hint), Some(current_hint)) = (
            source.rollout_context_hint.as_deref(),
            current_rollout_context_hint(self.operations_snapshot().available_slots.as_slice()),
        ) {
            if source_hint != current_hint {
                issues.push(ReplayPreflightIssue {
                    code: ReplayPreflightIssueCode::RolloutContextChangedTooMuch,
                    detail: format!(
                        "rollout context changed from `{source_hint}` to `{current_hint}`"
                    ),
                });
            }
        }

        if let Some(request) = source.request.clone() {
            let admission = self.service.technical_admission(&request);
            if let Some(failure) = admission.failure {
                let issue = match failure.kind {
                    CanonicalFailureKind::ArtifactUnavailable
                    | CanonicalFailureKind::ArtifactVerificationFailed
                    | CanonicalFailureKind::ArtifactIncompatible => ReplayPreflightIssue {
                        code: ReplayPreflightIssueCode::MissingArtifactOrSlot,
                        detail: format!(
                            "required artifact/slot no longer available: {}",
                            failure.detail
                        ),
                    },
                    CanonicalFailureKind::BackendDisabled
                    | CanonicalFailureKind::StageUnavailable
                    | CanonicalFailureKind::NsrBackendUnavailable => ReplayPreflightIssue {
                        code: ReplayPreflightIssueCode::ChangedBackendDeviceWorkerContext,
                        detail: format!(
                            "backend/worker/device no longer suitable: {}",
                            failure.detail
                        ),
                    },
                    _ => ReplayPreflightIssue {
                        code: ReplayPreflightIssueCode::SnapshotIncomplete,
                        detail: format!("replay configuration incomplete: {}", failure.detail),
                    },
                };
                issues.push(issue);
            }
        }

        let blocked = issues.iter().any(|issue| {
            matches!(
                issue.code,
                ReplayPreflightIssueCode::RecordMissing
                    | ReplayPreflightIssueCode::MissingArtifactOrSlot
                    | ReplayPreflightIssueCode::ChangedBackendDeviceWorkerContext
                    | ReplayPreflightIssueCode::MissingRemoteExecutionContext
                    | ReplayPreflightIssueCode::OriginalContextUnavailable
                    | ReplayPreflightIssueCode::ContextBridgeTooLossy
            )
        });
        let insufficient = issues.iter().any(|issue| {
            matches!(
                issue.code,
                ReplayPreflightIssueCode::SnapshotIncomplete
                    | ReplayPreflightIssueCode::CanonicalRequestMissing
            )
        });
        let changed_context = issues.iter().any(|issue| {
            matches!(
                issue.code,
                ReplayPreflightIssueCode::RolloutContextChangedTooMuch
                    | ReplayPreflightIssueCode::LocalRemoteConstraintMismatch
                    | ReplayPreflightIssueCode::AlternativeContextWithCaveats
            )
        });
        let has_caveat = source.snapshot_readiness == PersistedSnapshotReadiness::Partial;
        if source_mode != current_mode && has_caveat {
            issues.push(ReplayPreflightIssue {
                code: ReplayPreflightIssueCode::ContextBridgeTooLossy,
                detail: "context bridge is too lossy because source snapshot already carries partial fidelity".to_string(),
            });
        }
        if has_caveat {
            issues.push(ReplayPreflightIssue {
                code: ReplayPreflightIssueCode::ReplayNotFidelityEquivalent,
                detail: "snapshot readiness is partial; replay may complete without fidelity equivalence".to_string(),
            });
        }

        let replayability = if blocked {
            ReplayabilityClass::BlockedForReplay
        } else if insufficient {
            ReplayabilityClass::InsufficientForReplay
        } else if changed_context {
            ReplayabilityClass::ReplayableOnlyUnderChangedContext
        } else if has_caveat {
            ReplayabilityClass::ReplayableWithCaveats
        } else {
            ReplayabilityClass::ReplayReady
        };
        let locality = match (source_mode, current_mode) {
            (ReplayExecutionMode::Local, ReplayExecutionMode::Local) => {
                ReplayPreflightLocality::LocalOnly
            }
            (ReplayExecutionMode::RemoteWorkerIpc, ReplayExecutionMode::RemoteWorkerIpc) => {
                ReplayPreflightLocality::RemoteOnly
            }
            (ReplayExecutionMode::Local, ReplayExecutionMode::RemoteWorkerIpc)
            | (ReplayExecutionMode::RemoteWorkerIpc, ReplayExecutionMode::Local) => {
                ReplayPreflightLocality::ChangedContextOnly
            }
        };
        let fidelity_equivalent_possible =
            !changed_context && !has_caveat && !blocked && !insufficient;
        let source_context = source.context_descriptor();
        let context_bridge = build_context_bridge_summary(&source_context, &current_context);
        let context_consistency_class =
            classify_preflight_context_consistency(replayability, changed_context, has_caveat);
        let rollout_context = classify_rollout_context_comparability(
            source.rollout_context_hint.as_deref(),
            current_rollout_context_hint(self.operations_snapshot().available_slots.as_slice()),
            changed_context,
            has_caveat,
            blocked,
            insufficient,
        );

        ComputeReplayPreflight {
            source_job_id: source.job_id,
            replayability,
            source_execution_mode: source_mode,
            current_execution_mode: current_mode,
            snapshot_readiness: Some(source.snapshot_readiness),
            locality,
            context_consistency_class,
            context_bridge,
            rollout_context,
            fidelity_equivalent_possible,
            issues,
        }
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
            && candidate_record.resource_class == baseline_record.resource_class
            && candidate_record.capacity_pressure == baseline_record.capacity_pressure
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
            rollout_context: classify_rollout_context_comparability(
                baseline_record.rollout_context_hint.as_deref(),
                candidate_record.rollout_context_hint.as_deref(),
                false,
                candidate_record.snapshot_readiness == PersistedSnapshotReadiness::Partial
                    || baseline_record.snapshot_readiness == PersistedSnapshotReadiness::Partial,
                false,
                false,
            ),
        };
        if summary.rollout_context.comparability
            == RolloutReplayComparability::NotMeaningfullyComparableAcrossRolloutBoundary
        {
            return BaselineComparisonResult::NotComparable {
                candidate_job_id: candidate.job_id,
                baseline_job_id: Some(baseline_record.job_id),
                code: BaselineComparisonFailureCode::NotMeaningfulUnderRuntimeChange,
                detail: "candidate and baseline crossed an incompatible rollout boundary"
                    .to_string(),
            };
        }
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
        let mut dominant_stage_counts: BTreeMap<CanonicalStageId, usize> = BTreeMap::new();
        let mut slots = Vec::new();
        let mut last_job = None;
        let mut replay_ready = 0usize;
        let mut partial = 0usize;
        let mut insufficient = 0usize;
        let mut stale_or_incomplete = 0usize;

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
            if let Some(hotspot) = record.accounting.hotspot_summary {
                if let Some(stage) = hotspot.dominant_stage {
                    *dominant_stage_counts.entry(stage).or_insert(0) += 1;
                }
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
                        warmup_state: parse_warmup_state(slot.detail.as_deref()),
                    })
                    .collect();
            }
            match derive_live_snapshot_readiness(record) {
                PersistedSnapshotReadiness::ReplayReady => {
                    replay_ready = replay_ready.saturating_add(1)
                }
                PersistedSnapshotReadiness::Partial => partial = partial.saturating_add(1),
                PersistedSnapshotReadiness::Insufficient => {
                    insufficient = insufficient.saturating_add(1)
                }
                PersistedSnapshotReadiness::StaleOrIncomplete => {
                    stale_or_incomplete = stale_or_incomplete.saturating_add(1)
                }
            }
        }
        if let Some(store) = self.history_store.as_ref() {
            for persisted in store.records() {
                if self.service.job(JobId(persisted.job_id)).is_some() {
                    continue;
                }
                match persisted
                    .execution_snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.readiness)
                    .unwrap_or(PersistedSnapshotReadiness::Insufficient)
                {
                    PersistedSnapshotReadiness::ReplayReady => {
                        replay_ready = replay_ready.saturating_add(1)
                    }
                    PersistedSnapshotReadiness::Partial => partial = partial.saturating_add(1),
                    PersistedSnapshotReadiness::Insufficient => {
                        insufficient = insufficient.saturating_add(1)
                    }
                    PersistedSnapshotReadiness::StaleOrIncomplete => {
                        stale_or_incomplete = stale_or_incomplete.saturating_add(1)
                    }
                }
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
        let runtime_profile = RuntimeProfile::from_runtime_env().unwrap_or_else(|_| {
            RuntimeProfile::fallback_for_execution_path(scheduler.execution_path)
        });
        let repeated_hotspot = dominant_stage_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .filter(|(_, count)| *count >= 2);
        let queue_pressure = scheduler.queued_jobs > 0;
        let capacity_pressure = queue_pressure
            || matches!(
                state,
                RuntimeOpsState::PartiallyUnavailable | RuntimeOpsState::Unavailable
            );
        let cold_or_warmup_pressure = slots.iter().any(|slot| {
            matches!(
                slot.warmup_state,
                RuntimeWarmupState::Cold
                    | RuntimeWarmupState::Preparing
                    | RuntimeWarmupState::Blocked
            )
        });
        let stage_hotspot_pressure = repeated_hotspot.is_some();
        let mixed_bottlenecks = [
            queue_pressure,
            capacity_pressure,
            cold_or_warmup_pressure,
            stage_hotspot_pressure,
        ]
        .into_iter()
        .filter(|flag| *flag)
        .count()
            > 1;
        let mut caveats = Vec::new();
        if queue_pressure {
            caveats.push("queue_backlog_active".to_string());
        }
        if cold_or_warmup_pressure {
            caveats.push("cold_or_warmup_slots_present".to_string());
        }
        if stage_hotspot_pressure {
            caveats.push("repeated_stage_hotspot_pattern".to_string());
        }
        if mixed_bottlenecks {
            caveats.push("mixed_bottleneck_picture".to_string());
        }
        let (current_state, main_bottleneck) = if submitted_total == 0 {
            (
                "inconclusive".to_string(),
                "insufficient_signal".to_string(),
            )
        } else if mixed_bottlenecks {
            (
                "mixed_optimization_picture".to_string(),
                "mixed".to_string(),
            )
        } else if queue_pressure || capacity_pressure {
            (
                "constrained_by_capacity".to_string(),
                "queue_or_capacity".to_string(),
            )
        } else if cold_or_warmup_pressure {
            (
                "constrained_by_cold_or_warmup".to_string(),
                "warmup_readiness".to_string(),
            )
        } else if stage_hotspot_pressure {
            (
                "constrained_by_dominant_stage_hotspot".to_string(),
                "stage_hotspot".to_string(),
            )
        } else if state == RuntimeOpsState::Degraded {
            (
                "degraded_but_serviceable".to_string(),
                "degraded_path".to_string(),
            )
        } else {
            ("healthy_and_efficient".to_string(), "none".to_string())
        };
        let historical_feedback_alignment = self
            .latest_baseline_comparison
            .as_ref()
            .map(|comparison| format!("{:?}", comparison.outcome).to_ascii_lowercase())
            .unwrap_or_else(|| "insufficient_history_signal".to_string());

        RuntimeOpsSnapshot {
            state,
            runtime_mode: runtime_profile.mode,
            deployment_profile: runtime_profile.deployment,
            diagnostic_flags: runtime_profile.diagnostics,
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
            repeated_hotspot_stage: repeated_hotspot.map(|(stage, _)| stage),
            repeated_hotspot_runs: repeated_hotspot.map(|(_, count)| count).unwrap_or(0),
            optimization_view: RuntimeOptimizationOpsView {
                current_state,
                main_bottleneck,
                queue_pressure,
                capacity_pressure,
                cold_or_warmup_pressure,
                stage_hotspot_pressure,
                mixed_bottlenecks,
                historical_feedback_alignment,
                caveats,
            },
            replay_snapshot_coverage: ReplaySnapshotCoverage {
                replay_ready,
                partial,
                insufficient,
                stale_or_incomplete,
            },
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

fn parse_warmup_state(detail: Option<&str>) -> RuntimeWarmupState {
    let Some(detail) = detail else {
        return RuntimeWarmupState::Unknown;
    };
    if detail.contains("warmup=") && detail.contains("Active:warm:") {
        RuntimeWarmupState::Ready
    } else if detail.contains("warmup=")
        && (detail.contains("Active:prepared:")
            || detail.contains("Candidate:prepared:")
            || detail.contains("Compare:prepared:")
            || detail.contains("Shadow:prepared:"))
    {
        RuntimeWarmupState::Preparing
    } else if detail.contains("warmup=")
        && (detail.contains("Active:blocked:") || detail.contains("Blocked:blocked:"))
    {
        RuntimeWarmupState::Blocked
    } else if detail.contains("warmup=")
        && (detail.contains("Active:stale:")
            || detail.contains("Candidate:stale:")
            || detail.contains("Compare:stale:")
            || detail.contains("Shadow:stale:"))
    {
        RuntimeWarmupState::Stale
    } else if detail.contains("warmup=") && detail.contains("Active:cold:") {
        RuntimeWarmupState::Cold
    } else {
        RuntimeWarmupState::Unknown
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
    ChangedRuntimeContextIncompatible,
    MissingRemoteExecutionContext,
    ReplayExecutionFailed,
    ReplayCompletedWithChangedConfiguration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayExecutionMode {
    Local,
    RemoteWorkerIpc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayRemoteContextReproducibility {
    NotApplicableLocal,
    Exact,
    Partial,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayContextConsistencyClass {
    SameEffectiveExecutionContext,
    ChangedComparableExecutionContext,
    ChangedContextWithFidelityCaveat,
    NotMeaningfullyComparable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutReplayContextClass {
    ActiveOrWarm,
    GuardedOrCandidate,
    FallbackOrRollback,
    MixedOrUnknown,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutReplayComparability {
    ComparableAcrossRolloutBoundary,
    ComparableWithRolloutCaveat,
    NotMeaningfullyComparableAcrossRolloutBoundary,
    BlockedInsufficientRolloutContext,
    BlockedChangedExecutionContextBeyondUsefulComparison,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RolloutReplayComparisonContext {
    pub source: RolloutReplayContextClass,
    pub replay: RolloutReplayContextClass,
    pub source_hint: Option<String>,
    pub replay_hint: Option<String>,
    pub comparability: RolloutReplayComparability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayContextTransition {
    LocalToLocal,
    LocalToRemote,
    RemoteToLocal,
    RemoteToRemoteSame,
    RemoteToRemoteChanged,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayExecutionContextDescriptor {
    pub execution_mode: ReplayExecutionMode,
    pub execution_path: String,
    pub execution_lane: Option<String>,
    pub resource_class: Option<String>,
    pub capacity_pressure: Option<String>,
    pub has_backend_route: bool,
    pub remote_context_completeness: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayContextBridgeSummary {
    pub transition: ReplayContextTransition,
    pub source: ReplayExecutionContextDescriptor,
    pub replay: ReplayExecutionContextDescriptor,
    pub major_mismatches: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayabilityClass {
    ReplayReady,
    ReplayableWithCaveats,
    ReplayableOnlyUnderChangedContext,
    InsufficientForReplay,
    BlockedForReplay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayPreflightIssueCode {
    RecordMissing,
    SnapshotIncomplete,
    CanonicalRequestMissing,
    MissingArtifactOrSlot,
    ChangedBackendDeviceWorkerContext,
    MissingRemoteExecutionContext,
    OriginalContextUnavailable,
    AlternativeContextWithCaveats,
    ContextBridgeTooLossy,
    RolloutContextChangedTooMuch,
    LocalRemoteConstraintMismatch,
    ReplayNotFidelityEquivalent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayPreflightIssue {
    pub code: ReplayPreflightIssueCode,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayPreflightLocality {
    LocalOnly,
    RemoteOnly,
    Either,
    ChangedContextOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeReplayPreflight {
    pub source_job_id: JobId,
    pub replayability: ReplayabilityClass,
    pub source_execution_mode: ReplayExecutionMode,
    pub current_execution_mode: ReplayExecutionMode,
    pub snapshot_readiness: Option<PersistedSnapshotReadiness>,
    pub locality: ReplayPreflightLocality,
    pub context_consistency_class: ReplayContextConsistencyClass,
    pub context_bridge: ReplayContextBridgeSummary,
    pub rollout_context: RolloutReplayComparisonContext,
    pub fidelity_equivalent_possible: bool,
    pub issues: Vec<ReplayPreflightIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayConfigurationDiff {
    pub execution_path_match: bool,
    pub execution_lane_match: bool,
    pub backend_route_match: bool,
    pub model_slots_match: bool,
    pub resource_class_match: bool,
    pub capacity_pressure_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeReplayReport {
    pub source_job_id: JobId,
    pub replay_job_id: JobId,
    pub determinism_class: ReplayDeterminismClass,
    pub source_execution_mode: ReplayExecutionMode,
    pub replay_execution_mode: ReplayExecutionMode,
    pub remote_context_reproducibility: ReplayRemoteContextReproducibility,
    pub context_consistency_class: ReplayContextConsistencyClass,
    pub context_bridge: ReplayContextBridgeSummary,
    pub rollout_context: RolloutReplayComparisonContext,
    pub configuration_diff: ReplayConfigurationDiff,
    pub replay_succeeded: bool,
    pub completion_class_match: bool,
    pub failure_kind_match: bool,
    pub replay_failure: Option<ReplayFailureCode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
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
    pub rollout_context: RolloutReplayComparisonContext,
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
    resource_class: Option<ResourceClass>,
    capacity_pressure: Option<String>,
    backend_route: Option<crate::pipeline::CanonicalBackendRoute>,
    model_slots: Vec<String>,
    has_backend_route: bool,
    completion_class: Option<String>,
    failure_kind: Option<String>,
    pipeline_state: Option<String>,
    work_summary: Option<CanonicalWorkSummary>,
    snapshot_readiness: PersistedSnapshotReadiness,
    rollout_context_hint: Option<String>,
    remote_context_completeness: Option<String>,
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
            resource_class: Some(record.accounting.resource_class),
            capacity_pressure: Some(format!("{:?}", record.accounting.capacity_pressure)),
            backend_route: record.result.as_ref().map(|result| result.route),
            has_backend_route: record.result.is_some(),
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
            snapshot_readiness: derive_live_snapshot_readiness(record),
            rollout_context_hint: derive_live_rollout_context_hint(record),
            remote_context_completeness: Some(
                if matches!(record.execution_path, JobExecutionPath::WorkerIpc)
                    && record.result.is_some()
                {
                    "complete".to_string()
                } else if matches!(record.execution_path, JobExecutionPath::WorkerIpc) {
                    "partial".to_string()
                } else {
                    "not_applicable".to_string()
                },
            ),
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
            resource_class: persisted.resource_class.as_ref().and_then(|class| {
                match class.as_str() {
                    "light" => Some(ResourceClass::Light),
                    "standard" => Some(ResourceClass::Standard),
                    "heavy" => Some(ResourceClass::Heavy),
                    _ => None,
                }
            }),
            capacity_pressure: persisted.capacity_pressure.clone(),
            backend_route: persisted.backend_route.as_ref().map(|route| {
                crate::pipeline::CanonicalBackendRoute {
                    pack_id: route.pack_id,
                    world_backend: route.world_backend,
                    sae_backend: route.sae_backend,
                    ssm_backend: route.ssm_backend,
                    lfm_backend: route.lfm_backend,
                }
            }),
            has_backend_route: persisted.backend_route.is_some(),
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
            snapshot_readiness: persisted
                .execution_snapshot
                .as_ref()
                .map(|snapshot| snapshot.readiness)
                .unwrap_or_else(|| infer_persisted_snapshot_readiness(persisted)),
            rollout_context_hint: persisted
                .execution_snapshot
                .as_ref()
                .map(|snapshot| snapshot.rollout.rollout_context_hint.clone()),
            remote_context_completeness: persisted
                .remote_execution_context
                .as_ref()
                .map(|ctx| ctx.context_completeness.clone()),
        }
    }

    fn execution_mode(&self) -> ReplayExecutionMode {
        if self.execution_path == "WorkerIpc" {
            ReplayExecutionMode::RemoteWorkerIpc
        } else {
            ReplayExecutionMode::Local
        }
    }

    fn has_remote_context(&self) -> bool {
        self.execution_mode() == ReplayExecutionMode::RemoteWorkerIpc
            && self.execution_lane.is_some()
            && self.backend_route.is_some()
    }

    fn context_descriptor(&self) -> ReplayExecutionContextDescriptor {
        ReplayExecutionContextDescriptor {
            execution_mode: self.execution_mode(),
            execution_path: self.execution_path.clone(),
            execution_lane: self.execution_lane.clone(),
            resource_class: self.resource_class.map(|class| format!("{class:?}")),
            capacity_pressure: self.capacity_pressure.clone(),
            has_backend_route: self.has_backend_route,
            remote_context_completeness: self
                .remote_context_completeness
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        }
    }
}

fn derive_live_snapshot_readiness(record: &JobRecord) -> PersistedSnapshotReadiness {
    let was_remote = matches!(record.execution_path, JobExecutionPath::WorkerIpc);
    if record.result.is_none()
        && !matches!(
            record.state,
            JobLifecycleState::Completed
                | JobLifecycleState::Failed
                | JobLifecycleState::Rejected
                | JobLifecycleState::TimedOut
        )
    {
        return PersistedSnapshotReadiness::Insufficient;
    }
    if was_remote && record.result.is_none() {
        return PersistedSnapshotReadiness::StaleOrIncomplete;
    }
    if record.accounting.model_slots.is_empty() || record.accounting.work_cost_summary.is_none() {
        return PersistedSnapshotReadiness::Partial;
    }
    PersistedSnapshotReadiness::ReplayReady
}

fn derive_live_rollout_context_hint(record: &JobRecord) -> Option<String> {
    let states = record
        .accounting
        .model_slots
        .iter()
        .map(|slot| parse_warmup_state(slot.detail.as_deref()))
        .collect::<Vec<_>>();
    classify_rollout_context_hint(states.as_slice()).map(str::to_string)
}

fn infer_persisted_snapshot_readiness(record: &PersistedJobRecord) -> PersistedSnapshotReadiness {
    if record.canonical_request.is_none() {
        return PersistedSnapshotReadiness::Insufficient;
    }
    if record.execution_path == "WorkerIpc" {
        let complete_remote_context = record
            .remote_execution_context
            .as_ref()
            .is_some_and(|ctx| ctx.context_completeness == "complete");
        if complete_remote_context {
            PersistedSnapshotReadiness::ReplayReady
        } else {
            PersistedSnapshotReadiness::Partial
        }
    } else if record.backend_route.is_some() {
        PersistedSnapshotReadiness::ReplayReady
    } else {
        PersistedSnapshotReadiness::Partial
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
    let failure_classification = record.accounting.failure_kind.map(classify_failure_kind);
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
        fault_domain: failure_classification.map(|classification| classification.domain),
        fault_isolation: failure_classification.map(|classification| classification.isolation),
        fault_systemic: failure_classification.map(|classification| classification.systemic),
        pipeline_state: record.accounting.pipeline_state,
        work_summary: record.accounting.work_summary,
        stage_cost_attribution: record.accounting.stage_cost_attribution.clone(),
        hotspot_summary: record.accounting.hotspot_summary,
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

fn current_replay_mode(path: JobExecutionPath) -> ReplayExecutionMode {
    if path == JobExecutionPath::WorkerIpc {
        ReplayExecutionMode::RemoteWorkerIpc
    } else {
        ReplayExecutionMode::Local
    }
}

fn current_rollout_context_hint(slots: &[RuntimeSlotSnapshot]) -> Option<&'static str> {
    let states = slots
        .iter()
        .map(|slot| slot.warmup_state)
        .collect::<Vec<_>>();
    classify_rollout_context_hint(states.as_slice())
}

fn classify_rollout_context_hint(states: &[RuntimeWarmupState]) -> Option<&'static str> {
    if states.is_empty() {
        return None;
    }
    if states
        .iter()
        .any(|state| matches!(state, RuntimeWarmupState::Blocked))
    {
        return Some("blocked_or_stale");
    }
    if states
        .iter()
        .any(|state| matches!(state, RuntimeWarmupState::Stale))
    {
        return Some("blocked_or_stale");
    }
    if states
        .iter()
        .any(|state| matches!(state, RuntimeWarmupState::Preparing))
    {
        return Some("active_plus_candidate");
    }
    if states
        .iter()
        .all(|state| matches!(state, RuntimeWarmupState::Ready))
    {
        return Some("active_or_warm");
    }
    Some("mixed_or_unknown")
}

fn rollout_context_class_from_hint(hint: Option<&str>) -> RolloutReplayContextClass {
    match hint {
        Some("active_or_warm") => RolloutReplayContextClass::ActiveOrWarm,
        Some("active_plus_candidate") => RolloutReplayContextClass::GuardedOrCandidate,
        Some("blocked_or_stale") => RolloutReplayContextClass::FallbackOrRollback,
        Some("mixed_or_unknown") => RolloutReplayContextClass::MixedOrUnknown,
        Some(_) => RolloutReplayContextClass::MixedOrUnknown,
        None => RolloutReplayContextClass::Unavailable,
    }
}

fn classify_rollout_context_comparability(
    source_hint: Option<&str>,
    replay_hint: Option<&str>,
    changed_context: bool,
    has_caveat: bool,
    blocked: bool,
    insufficient: bool,
) -> RolloutReplayComparisonContext {
    let source = rollout_context_class_from_hint(source_hint);
    let replay = rollout_context_class_from_hint(replay_hint);
    let comparability = if blocked && changed_context {
        RolloutReplayComparability::BlockedChangedExecutionContextBeyondUsefulComparison
    } else if insufficient
        || matches!(source, RolloutReplayContextClass::Unavailable)
        || matches!(replay, RolloutReplayContextClass::Unavailable)
    {
        RolloutReplayComparability::BlockedInsufficientRolloutContext
    } else if matches!(source, RolloutReplayContextClass::FallbackOrRollback)
        != matches!(replay, RolloutReplayContextClass::FallbackOrRollback)
    {
        RolloutReplayComparability::NotMeaningfullyComparableAcrossRolloutBoundary
    } else if source != replay || has_caveat || changed_context {
        RolloutReplayComparability::ComparableWithRolloutCaveat
    } else {
        RolloutReplayComparability::ComparableAcrossRolloutBoundary
    };
    RolloutReplayComparisonContext {
        source,
        replay,
        source_hint: source_hint.map(str::to_string),
        replay_hint: replay_hint.map(str::to_string),
        comparability,
    }
}

fn current_context_descriptor(snapshot: &RuntimeOpsSnapshot) -> ReplayExecutionContextDescriptor {
    ReplayExecutionContextDescriptor {
        execution_mode: current_replay_mode(snapshot.execution_path),
        execution_path: format!("{:?}", snapshot.execution_path),
        execution_lane: None,
        resource_class: None,
        capacity_pressure: None,
        has_backend_route: false,
        remote_context_completeness: match snapshot.execution_path {
            JobExecutionPath::WorkerIpc => "partial".to_string(),
            JobExecutionPath::LocalCanonical => "not_applicable".to_string(),
        },
    }
}

fn build_context_bridge_summary(
    source: &ReplayExecutionContextDescriptor,
    replay: &ReplayExecutionContextDescriptor,
) -> ReplayContextBridgeSummary {
    let transition = classify_context_transition(source, replay);
    let mut major_mismatches = Vec::new();
    if source.execution_mode != replay.execution_mode {
        major_mismatches.push(format!(
            "execution_mode:{:?}->{:?}",
            source.execution_mode, replay.execution_mode
        ));
    }
    if source.execution_lane != replay.execution_lane {
        major_mismatches.push(format!(
            "execution_lane:{:?}->{:?}",
            source.execution_lane, replay.execution_lane
        ));
    }
    if source.resource_class != replay.resource_class {
        major_mismatches.push(format!(
            "resource_class:{:?}->{:?}",
            source.resource_class, replay.resource_class
        ));
    }
    if source.capacity_pressure != replay.capacity_pressure {
        major_mismatches.push(format!(
            "capacity_pressure:{:?}->{:?}",
            source.capacity_pressure, replay.capacity_pressure
        ));
    }
    if source.has_backend_route != replay.has_backend_route {
        major_mismatches.push(format!(
            "backend_route:{}->{}",
            source.has_backend_route, replay.has_backend_route
        ));
    }
    if source.remote_context_completeness != replay.remote_context_completeness {
        major_mismatches.push(format!(
            "remote_context:{}->{}",
            source.remote_context_completeness, replay.remote_context_completeness
        ));
    }
    ReplayContextBridgeSummary {
        transition,
        source: source.clone(),
        replay: replay.clone(),
        major_mismatches,
    }
}

fn classify_context_transition(
    source: &ReplayExecutionContextDescriptor,
    replay: &ReplayExecutionContextDescriptor,
) -> ReplayContextTransition {
    match (source.execution_mode, replay.execution_mode) {
        (ReplayExecutionMode::Local, ReplayExecutionMode::Local) => {
            ReplayContextTransition::LocalToLocal
        }
        (ReplayExecutionMode::Local, ReplayExecutionMode::RemoteWorkerIpc) => {
            ReplayContextTransition::LocalToRemote
        }
        (ReplayExecutionMode::RemoteWorkerIpc, ReplayExecutionMode::Local) => {
            ReplayContextTransition::RemoteToLocal
        }
        (ReplayExecutionMode::RemoteWorkerIpc, ReplayExecutionMode::RemoteWorkerIpc) => {
            let same_remote = source.execution_lane == replay.execution_lane
                && source.resource_class == replay.resource_class
                && source.capacity_pressure == replay.capacity_pressure
                && source.has_backend_route == replay.has_backend_route;
            if same_remote {
                ReplayContextTransition::RemoteToRemoteSame
            } else {
                ReplayContextTransition::RemoteToRemoteChanged
            }
        }
    }
}

fn classify_preflight_context_consistency(
    replayability: ReplayabilityClass,
    changed_context: bool,
    has_caveat: bool,
) -> ReplayContextConsistencyClass {
    match replayability {
        ReplayabilityClass::BlockedForReplay | ReplayabilityClass::InsufficientForReplay => {
            ReplayContextConsistencyClass::NotMeaningfullyComparable
        }
        _ if changed_context && has_caveat => {
            ReplayContextConsistencyClass::ChangedContextWithFidelityCaveat
        }
        _ if changed_context => ReplayContextConsistencyClass::ChangedComparableExecutionContext,
        _ if has_caveat => ReplayContextConsistencyClass::ChangedContextWithFidelityCaveat,
        _ => ReplayContextConsistencyClass::SameEffectiveExecutionContext,
    }
}

fn classify_context_consistency(
    bridge: &ReplayContextBridgeSummary,
    replay_succeeded: bool,
    has_snapshot_caveat: bool,
) -> ReplayContextConsistencyClass {
    if !replay_succeeded {
        return ReplayContextConsistencyClass::NotMeaningfullyComparable;
    }
    if bridge.major_mismatches.is_empty() {
        return ReplayContextConsistencyClass::SameEffectiveExecutionContext;
    }
    if has_snapshot_caveat || bridge.major_mismatches.len() > 2 {
        return ReplayContextConsistencyClass::ChangedContextWithFidelityCaveat;
    }
    ReplayContextConsistencyClass::ChangedComparableExecutionContext
}

fn replay_preflight_failure(preflight: &ComputeReplayPreflight) -> (ReplayFailureCode, String) {
    for issue in &preflight.issues {
        let code = match issue.code {
            ReplayPreflightIssueCode::RecordMissing => Some(ReplayFailureCode::RecordMissing),
            ReplayPreflightIssueCode::MissingArtifactOrSlot => {
                Some(ReplayFailureCode::RequiredArtifactUnavailable)
            }
            ReplayPreflightIssueCode::ChangedBackendDeviceWorkerContext => {
                Some(ReplayFailureCode::BackendOrDeviceUnavailable)
            }
            ReplayPreflightIssueCode::MissingRemoteExecutionContext => {
                Some(ReplayFailureCode::MissingRemoteExecutionContext)
            }
            ReplayPreflightIssueCode::OriginalContextUnavailable => {
                Some(ReplayFailureCode::MissingRemoteExecutionContext)
            }
            ReplayPreflightIssueCode::ContextBridgeTooLossy => {
                Some(ReplayFailureCode::ChangedRuntimeContextIncompatible)
            }
            ReplayPreflightIssueCode::RolloutContextChangedTooMuch
            | ReplayPreflightIssueCode::LocalRemoteConstraintMismatch
            | ReplayPreflightIssueCode::AlternativeContextWithCaveats => {
                Some(ReplayFailureCode::ChangedRuntimeContextIncompatible)
            }
            ReplayPreflightIssueCode::SnapshotIncomplete
            | ReplayPreflightIssueCode::CanonicalRequestMissing => {
                Some(ReplayFailureCode::ConfigurationIncomplete)
            }
            ReplayPreflightIssueCode::ReplayNotFidelityEquivalent => None,
        };
        if let Some(code) = code {
            return (code, issue.detail.clone());
        }
    }
    (
        ReplayFailureCode::ConfigurationIncomplete,
        "replay preflight failed with unresolved prerequisites".to_string(),
    )
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::{
        parse_warmup_state, BaselineComparisonFailureCode, BaselineComparisonResult,
        BaselineReference, CanonicalComputeEntryPoint, ComputeExecutionMode,
        ComputeHistoryLookupError, ComputeJobHandle, ComputeJobHistoryLookup, ComputeReplayOutcome,
        ComputeRequestValidationCode, ComputeSubmitOutcome, ComputeSubmitRequest,
        RecoveryDisposition, ReplayContextConsistencyClass, ReplayContextTransition,
        ReplayDeterminismClass, ReplayExecutionMode, ReplayFailureCode, ReplayPreflightIssueCode,
        ReplayRemoteContextReproducibility, ReplayabilityClass, RolloutReplayComparability,
        RuntimeOperation, RuntimeOperationCode, RuntimeOpsState, RuntimeSignalState,
        RuntimeWarmupState,
    };
    use crate::pipeline::{CanonicalFailureKind, CanonicalPipelineRequest};
    use crate::{
        compute_service::SchedulerConfig, InMemoryComputeService, JobExecutionPath,
        JobHistoryStore, JobId, JobLifecycleState,
    };

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
        assert!(!status.stage_cost_attribution.is_empty());
        assert!(!entry.lifecycle(handle).is_empty());
    }

    #[test]
    fn operations_snapshot_marks_unknown_without_job_signal() {
        let entry = service();
        let snapshot = entry.operations_snapshot();
        assert_eq!(snapshot.state, RuntimeOpsState::HealthyReady);
        assert_eq!(snapshot.state_signal, RuntimeSignalState::Unknown);
        assert_eq!(snapshot.jobs.submitted_total, 0);
        assert_eq!(snapshot.optimization_view.current_state, "inconclusive");
        assert_eq!(
            snapshot.optimization_view.main_bottleneck,
            "insufficient_signal"
        );
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
    fn operations_snapshot_surfaces_runtime_profile_flags() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = crate::test_env::clear_model_env_overrides();

        std::env::set_var("UCF_RUNTIME_MODE", "diagnostic");
        std::env::set_var("UCF_REAL_ENABLEMENT_MODE", "compare");

        let entry = service();
        let snapshot = entry.operations_snapshot();

        assert_eq!(snapshot.runtime_mode, crate::RuntimeMode::Diagnostic);
        assert_eq!(
            snapshot.deployment_profile,
            crate::DeploymentProfile::LocalOnly
        );
        assert!(snapshot.diagnostic_flags.compare_enabled);
        assert!(snapshot.diagnostic_flags.shadow_enabled);
    }

    #[test]
    fn operations_snapshot_keeps_hotspot_signals_consistent() {
        let mut entry = service();
        for t in [9_u64, 10_u64] {
            let mut request = valid_request();
            request.input.t = t;
            let outcome = entry
                .submit(ComputeSubmitRequest {
                    pipeline_request: request,
                    submitted_by: Some("svc-test".to_string()),
                    submitted_at_unix_ms: Some(200 + t),
                    execution_mode: ComputeExecutionMode::ExecuteInline,
                })
                .expect("submit should not fail");
            assert!(matches!(outcome, ComputeSubmitOutcome::Accepted { .. }));
        }
        let snapshot = entry.operations_snapshot();
        assert_eq!(
            snapshot.repeated_hotspot_stage.is_some(),
            snapshot.optimization_view.stage_hotspot_pressure
        );
        if snapshot.repeated_hotspot_stage.is_some() {
            assert!(snapshot.repeated_hotspot_runs >= 2);
        } else {
            assert_eq!(snapshot.repeated_hotspot_runs, 0);
        }
        assert!(!snapshot.optimization_view.current_state.is_empty());
    }

    #[test]
    fn parse_warmup_state_distinguishes_cold_preparing_ready_blocked() {
        assert_eq!(
            parse_warmup_state(Some("rollout=x;warmup=Active:warm:artifact verified")),
            RuntimeWarmupState::Ready
        );
        assert_eq!(
            parse_warmup_state(Some("rollout=x;warmup=Candidate:prepared:verified")),
            RuntimeWarmupState::Preparing
        );
        assert_eq!(
            parse_warmup_state(Some("rollout=x;warmup=Active:blocked:warmup failed")),
            RuntimeWarmupState::Blocked
        );
        assert_eq!(
            parse_warmup_state(Some("rollout=x;warmup=Candidate:stale:prefetch expired")),
            RuntimeWarmupState::Stale
        );
        assert_eq!(
            parse_warmup_state(Some("rollout=x;warmup=Active:cold:not configured")),
            RuntimeWarmupState::Cold
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
                assert_eq!(report.source_execution_mode, ReplayExecutionMode::Local);
                assert_eq!(report.replay_execution_mode, ReplayExecutionMode::Local);
                assert_eq!(
                    report.remote_context_reproducibility,
                    ReplayRemoteContextReproducibility::NotApplicableLocal
                );
                assert!(matches!(
                    report.context_consistency_class,
                    ReplayContextConsistencyClass::SameEffectiveExecutionContext
                        | ReplayContextConsistencyClass::ChangedComparableExecutionContext
                        | ReplayContextConsistencyClass::ChangedContextWithFidelityCaveat
                        | ReplayContextConsistencyClass::NotMeaningfullyComparable
                ));
                assert_eq!(
                    report.context_bridge.transition,
                    ReplayContextTransition::LocalToLocal
                );
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
    fn replay_preflight_classifies_ready_run() {
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
        let preflight = entry.replay_preflight(source);
        assert_eq!(preflight.replayability, ReplayabilityClass::ReplayReady);
        assert_eq!(
            preflight.context_consistency_class,
            ReplayContextConsistencyClass::SameEffectiveExecutionContext
        );
        assert_eq!(
            preflight.context_bridge.transition,
            ReplayContextTransition::LocalToLocal
        );
        assert!(preflight.fidelity_equivalent_possible);
        assert!(preflight.issues.is_empty());
    }

    #[test]
    fn replay_preflight_classifies_local_to_remote_as_changed_context_with_caveat() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            r#"{"schema_version":8,"job_id":61,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"completed","completion_class":"completed","execution_path":"LocalCanonical","submitted_at_unix_ms":1,"started_at_unix_ms":2,"finished_at_unix_ms":3,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","execution_lane":"standard","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","backend_route":{"pack_id":1,"world_backend":1,"sae_backend":1,"ssm_backend":1,"lfm_backend":1},"work_summary":null,"stage_profiles":[],"model_slots":[]}"#,
        )
        .expect("history fixture");
        let entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::with_scheduler(
                crate::pipeline::ComputePipelineBackend::stub(),
                SchedulerConfig {
                    max_concurrent_jobs: 1,
                    execution_path: JobExecutionPath::WorkerIpc,
                },
            ),
            &history_path,
        )
        .expect("entry with history");
        let preflight = entry.replay_preflight(ComputeJobHandle { job_id: JobId(61) });
        assert_eq!(
            preflight.replayability,
            ReplayabilityClass::ReplayableOnlyUnderChangedContext
        );
        assert_eq!(
            preflight.context_bridge.transition,
            ReplayContextTransition::LocalToRemote
        );
        assert_eq!(
            preflight.context_consistency_class,
            ReplayContextConsistencyClass::ChangedComparableExecutionContext
        );
    }

    #[test]
    fn replay_preflight_classifies_remote_to_local_with_bridge_signals() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            r#"{"schema_version":8,"job_id":62,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"completed","completion_class":"completed","execution_path":"WorkerIpc","submitted_at_unix_ms":1,"started_at_unix_ms":2,"finished_at_unix_ms":3,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","execution_lane":"worker","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","backend_route":{"pack_id":1,"world_backend":1,"sae_backend":1,"ssm_backend":1,"lfm_backend":1},"work_summary":null,"stage_profiles":[],"model_slots":[],"remote_execution_context":{"was_remote":true,"execution_path":"WorkerIpc","execution_lane":"worker","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","context_completeness":"complete"}}"#,
        )
        .expect("history fixture");
        let entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            &history_path,
        )
        .expect("entry with history");
        let preflight = entry.replay_preflight(ComputeJobHandle { job_id: JobId(62) });
        assert_eq!(
            preflight.context_bridge.transition,
            ReplayContextTransition::RemoteToLocal
        );
        assert!(preflight.issues.iter().any(|issue| {
            issue.code == ReplayPreflightIssueCode::AlternativeContextWithCaveats
        }));
    }

    #[test]
    fn replay_preflight_distinguishes_remote_to_other_remote_context() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            r#"{"schema_version":8,"job_id":63,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"completed","completion_class":"completed","execution_path":"WorkerIpc","submitted_at_unix_ms":1,"started_at_unix_ms":2,"finished_at_unix_ms":3,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","execution_lane":"worker","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","backend_route":{"pack_id":1,"world_backend":1,"sae_backend":1,"ssm_backend":1,"lfm_backend":1},"work_summary":null,"stage_profiles":[],"model_slots":[],"remote_execution_context":{"was_remote":true,"execution_path":"WorkerIpc","execution_lane":"worker","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","context_completeness":"complete"}}"#,
        )
        .expect("history fixture");
        let entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::with_scheduler(
                crate::pipeline::ComputePipelineBackend::stub(),
                SchedulerConfig {
                    max_concurrent_jobs: 1,
                    execution_path: JobExecutionPath::WorkerIpc,
                },
            ),
            &history_path,
        )
        .expect("entry with history");
        let preflight = entry.replay_preflight(ComputeJobHandle { job_id: JobId(63) });
        assert_eq!(
            preflight.context_bridge.transition,
            ReplayContextTransition::RemoteToRemoteChanged
        );
        assert!(!preflight.context_bridge.major_mismatches.is_empty());
    }

    #[test]
    fn replay_reports_missing_record() {
        let mut entry = service();
        let preflight = entry.replay_preflight(ComputeJobHandle { job_id: JobId(42) });
        assert_eq!(
            preflight.replayability,
            ReplayabilityClass::BlockedForReplay
        );
        assert!(preflight
            .issues
            .iter()
            .any(|issue| issue.code == ReplayPreflightIssueCode::RecordMissing));
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
        let preflight = entry.replay_preflight(ComputeJobHandle { job_id: JobId(7) });
        assert_eq!(
            preflight.replayability,
            ReplayabilityClass::InsufficientForReplay
        );
        assert!(preflight.issues.iter().any(|issue| {
            matches!(
                issue.code,
                ReplayPreflightIssueCode::SnapshotIncomplete
                    | ReplayPreflightIssueCode::CanonicalRequestMissing
            )
        }));
        let replay = entry
            .replay(ComputeJobHandle { job_id: JobId(7) })
            .expect("replay");
        match replay {
            ComputeReplayOutcome::NotReplayable { code, .. } => {
                assert_eq!(code, ReplayFailureCode::ConfigurationIncomplete);
            }
            other => panic!("expected non-replayable, got {other:?}"),
        }
        let coverage = entry.operations_snapshot().replay_snapshot_coverage;
        assert_eq!(coverage.insufficient, 1);
        assert_eq!(coverage.replay_ready, 0);
    }

    #[test]
    fn replay_from_remote_history_without_remote_context_is_blocked() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            r#"{"schema_version":8,"job_id":33,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"completed","completion_class":"completed","execution_path":"WorkerIpc","submitted_at_unix_ms":1,"started_at_unix_ms":2,"finished_at_unix_ms":3,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","backend_route":null,"execution_lane":null,"resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","work_summary":null,"stage_profiles":[],"model_slots":[]}"#,
        )
        .expect("history fixture");
        let mut entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            &history_path,
        )
        .expect("entry with history");
        let preflight = entry.replay_preflight(ComputeJobHandle { job_id: JobId(33) });
        assert_eq!(
            preflight.replayability,
            ReplayabilityClass::BlockedForReplay
        );
        assert!(preflight.issues.iter().any(|issue| {
            issue.code == ReplayPreflightIssueCode::MissingRemoteExecutionContext
        }));
        let replay = entry
            .replay(ComputeJobHandle { job_id: JobId(33) })
            .expect("replay");
        match replay {
            ComputeReplayOutcome::NotReplayable { code, .. } => {
                assert_eq!(code, ReplayFailureCode::MissingRemoteExecutionContext);
            }
            other => panic!("expected non-replayable, got {other:?}"),
        }
        let coverage = entry.operations_snapshot().replay_snapshot_coverage;
        assert_eq!(coverage.replay_ready, 0);
        assert_eq!(
            coverage.partial + coverage.insufficient + coverage.stale_or_incomplete,
            1
        );
    }

    #[test]
    fn replay_preflight_marks_partial_snapshot_as_caveat() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            r#"{"schema_version":8,"job_id":51,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"completed","completion_class":"completed","execution_path":"LocalCanonical","submitted_at_unix_ms":1,"started_at_unix_ms":2,"finished_at_unix_ms":3,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","execution_lane":"standard","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","backend_route":{"pack_id":1,"world_backend":1,"sae_backend":1,"ssm_backend":1,"lfm_backend":1},"work_summary":null,"stage_profiles":[],"model_slots":[],"execution_snapshot":{"request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request_available":true,"backend_route_available":true,"model_slot_count":0,"path":{"requested_execution_path":"LocalCanonical","executed_execution_path":"LocalCanonical","execution_lane":"standard","resource_class":"standard","was_remote":false,"redispatched_to_local":false,"retry_attempts":0},"rollout":{"active_or_warm_slots":1,"candidate_or_guarded_slots":0,"stale_or_blocked_slots":0,"rollout_context_hint":"active_or_warm"},"result":{"lifecycle_state":"completed","completion_class":"completed","pipeline_state":"ok","failure_kind":null},"readiness":"partial"}}"#,
        )
        .expect("history fixture");
        let entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            &history_path,
        )
        .expect("entry with history");
        let preflight = entry.replay_preflight(ComputeJobHandle { job_id: JobId(51) });
        assert_eq!(
            preflight.replayability,
            ReplayabilityClass::ReplayableWithCaveats
        );
        assert!(preflight
            .issues
            .iter()
            .any(|issue| issue.code == ReplayPreflightIssueCode::ReplayNotFidelityEquivalent));
        assert_eq!(
            preflight.rollout_context.comparability,
            RolloutReplayComparability::BlockedInsufficientRolloutContext
        );
    }

    #[test]
    fn replay_preflight_marks_missing_rollout_context_as_blocked_for_comparison() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            r#"{"schema_version":8,"job_id":52,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"completed","completion_class":"completed","execution_path":"LocalCanonical","submitted_at_unix_ms":1,"started_at_unix_ms":2,"finished_at_unix_ms":3,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","execution_lane":"standard","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","backend_route":{"pack_id":1,"world_backend":1,"sae_backend":1,"ssm_backend":1,"lfm_backend":1},"work_summary":null,"stage_profiles":[],"model_slots":[]}"#,
        )
        .expect("history fixture");
        let entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            &history_path,
        )
        .expect("entry with history");
        let preflight = entry.replay_preflight(ComputeJobHandle { job_id: JobId(52) });
        assert_eq!(
            preflight.rollout_context.comparability,
            RolloutReplayComparability::BlockedInsufficientRolloutContext
        );
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
    fn baseline_comparison_marks_rollout_boundary_as_not_meaningful() {
        let dir = tempfile::tempdir().expect("tempdir");
        let history_path = dir.path().join("job_history.jsonl");
        std::fs::write(
            &history_path,
            concat!(
                r#"{"schema_version":8,"job_id":70,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"completed","completion_class":"completed","execution_path":"LocalCanonical","submitted_at_unix_ms":1,"started_at_unix_ms":2,"finished_at_unix_ms":3,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","execution_lane":"standard","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","backend_route":{"pack_id":1,"world_backend":1,"sae_backend":1,"ssm_backend":1,"lfm_backend":1},"work_summary":null,"stage_profiles":[],"model_slots":[],"execution_snapshot":{"request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request_available":true,"backend_route_available":true,"model_slot_count":0,"path":{"requested_execution_path":"LocalCanonical","executed_execution_path":"LocalCanonical","execution_lane":"standard","resource_class":"standard","was_remote":false,"redispatched_to_local":false,"retry_attempts":0},"rollout":{"active_or_warm_slots":1,"candidate_or_guarded_slots":0,"stale_or_blocked_slots":0,"rollout_context_hint":"active_or_warm"},"result":{"lifecycle_state":"completed","completion_class":"completed","pipeline_state":"ok","failure_kind":null},"readiness":"replay_ready"}}"#,
                "\n",
                r#"{"schema_version":8,"job_id":71,"submitted_by":"svc","request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101","budget":{"max_micros":5000,"hard_timeout_micros":5000,"seed":9,"profile_id":0,"global_work_units":65536,"world_units":16384,"sae_units":16384,"ssm_units":16384,"lfm_units":16384,"degrade_policy":"DegradeStages","governor_tier":1}},"lifecycle_state":"completed","completion_class":"completed","execution_path":"LocalCanonical","submitted_at_unix_ms":4,"started_at_unix_ms":5,"finished_at_unix_ms":6,"queue_wait_ms":1,"execution_duration_micros":5,"total_duration_ms":2,"failure_kind":null,"pipeline_state":"ok","execution_lane":"standard","resource_class":"standard","capacity_pressure":"nominal","capacity_queue_disposition":"none","backend_route":{"pack_id":1,"world_backend":1,"sae_backend":1,"ssm_backend":1,"lfm_backend":1},"work_summary":null,"stage_profiles":[],"model_slots":[],"execution_snapshot":{"request":{"frame_id":1,"t":9,"context_digest_hex":"0101010101010101010101010101010101010101010101010101010101010101"},"canonical_request_available":true,"backend_route_available":true,"model_slot_count":0,"path":{"requested_execution_path":"LocalCanonical","executed_execution_path":"LocalCanonical","execution_lane":"standard","resource_class":"standard","was_remote":false,"redispatched_to_local":false,"retry_attempts":0},"rollout":{"active_or_warm_slots":0,"candidate_or_guarded_slots":0,"stale_or_blocked_slots":1,"rollout_context_hint":"blocked_or_stale"},"result":{"lifecycle_state":"completed","completion_class":"completed","pipeline_state":"ok","failure_kind":null},"readiness":"replay_ready"}}"#
            ),
        )
        .expect("history fixture");
        let mut entry = CanonicalComputeEntryPoint::with_history_path(
            InMemoryComputeService::new(crate::pipeline::ComputePipelineBackend::stub()),
            &history_path,
        )
        .expect("entry with history");
        let compare = entry.compare_against_baseline(
            ComputeJobHandle { job_id: JobId(71) },
            BaselineReference::Job(ComputeJobHandle { job_id: JobId(70) }),
        );
        match compare {
            BaselineComparisonResult::NotComparable { code, .. } => {
                assert_eq!(
                    code,
                    BaselineComparisonFailureCode::NotMeaningfulUnderRuntimeChange
                );
            }
            other => panic!("expected rollout-boundary not comparable, got {other:?}"),
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
