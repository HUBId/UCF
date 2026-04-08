use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::compute_service::{
    CapacityPressure, CapacityQueueDisposition, JobCompletionClass, JobId, JobLifecycleState,
    JobRecord, ResourceClass,
};
use crate::pipeline::{CanonicalFailureKind, CanonicalPipelineState};

const JOB_HISTORY_SCHEMA_VERSION: u16 = 6;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedJobRequestIdentity {
    pub frame_id: u64,
    pub t: u64,
    pub context_digest_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedComputeBudget {
    pub max_micros: u64,
    pub hard_timeout_micros: u64,
    pub seed: u64,
    pub profile_id: u32,
    pub global_work_units: u64,
    pub world_units: u64,
    pub sae_units: u64,
    pub ssm_units: u64,
    pub lfm_units: u64,
    pub degrade_policy: String,
    pub governor_tier: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedCanonicalRequest {
    pub frame_id: u64,
    pub t: u64,
    pub context_digest_hex: String,
    pub budget: PersistedComputeBudget,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedWorkSummary {
    pub global_budget_units: u64,
    pub global_remaining_units: u64,
    pub world_remaining_units: u64,
    pub sae_remaining_units: u64,
    pub ssm_remaining_units: u64,
    pub lfm_remaining_units: u64,
    pub budget_exceeded_stage: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedStageProfileSummary {
    pub stage: String,
    pub state: String,
    pub duration_micros: Option<u64>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedHotspotSummary {
    pub slowest_stage: Option<String>,
    pub dominant_stage: Option<String>,
    pub dominant_stage_share_bps: Option<u16>,
    pub degraded_stage_count: u8,
    pub skipped_stage_count: u8,
    pub unavailable_stage_count: u8,
    pub failed_stage_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedModelSlotSummary {
    pub slot: String,
    pub status: String,
    pub required_for_pack: bool,
    #[serde(default)]
    pub warmup_state: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedBackendRouteSummary {
    pub pack_id: u32,
    pub world_backend: u8,
    pub sae_backend: u8,
    pub ssm_backend: u8,
    pub lfm_backend: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedJobRecord {
    pub schema_version: u16,
    pub job_id: u64,
    pub submitted_by: Option<String>,
    pub request: PersistedJobRequestIdentity,
    #[serde(default)]
    pub canonical_request: Option<PersistedCanonicalRequest>,
    pub lifecycle_state: String,
    pub completion_class: Option<String>,
    pub execution_path: String,
    pub submitted_at_unix_ms: u64,
    pub started_at_unix_ms: Option<u64>,
    pub finished_at_unix_ms: Option<u64>,
    pub queue_wait_ms: Option<u64>,
    pub execution_duration_micros: Option<u64>,
    pub total_duration_ms: Option<u64>,
    pub failure_kind: Option<String>,
    pub pipeline_state: Option<String>,
    #[serde(default)]
    pub execution_lane: Option<String>,
    #[serde(default)]
    pub resource_class: Option<String>,
    #[serde(default)]
    pub capacity_queue_disposition: Option<String>,
    #[serde(default)]
    pub capacity_pressure: Option<String>,
    #[serde(default)]
    pub backend_route: Option<PersistedBackendRouteSummary>,
    pub work_summary: Option<PersistedWorkSummary>,
    #[serde(default)]
    pub stage_profiles: Vec<PersistedStageProfileSummary>,
    #[serde(default)]
    pub hotspot_summary: Option<PersistedHotspotSummary>,
    pub model_slots: Vec<PersistedModelSlotSummary>,
    #[serde(default)]
    pub recovery_source_job_id: Option<u64>,
    #[serde(default)]
    pub recovery_status: Option<String>,
    #[serde(default)]
    pub recovery_note: Option<String>,
}

impl PersistedJobRecord {
    pub fn from_job_record(record: &JobRecord) -> Self {
        let request = &record.job.request.input;
        let accounting = &record.accounting;
        Self {
            schema_version: JOB_HISTORY_SCHEMA_VERSION,
            job_id: record.job.id.0,
            submitted_by: record.job.meta.submitted_by.clone(),
            request: PersistedJobRequestIdentity {
                frame_id: request.frame_id.0,
                t: request.t,
                context_digest_hex: hex::encode(request.context_digest),
            },
            canonical_request: Some(PersistedCanonicalRequest {
                frame_id: request.frame_id.0,
                t: request.t,
                context_digest_hex: hex::encode(request.context_digest),
                budget: PersistedComputeBudget {
                    max_micros: record.job.request.budget.max_micros,
                    hard_timeout_micros: record.job.request.budget.hard_timeout_micros,
                    seed: record.job.request.budget.seed,
                    profile_id: record.job.request.budget.profile_id,
                    global_work_units: record.job.request.budget.global_work_units,
                    world_units: record.job.request.budget.world_units,
                    sae_units: record.job.request.budget.sae_units,
                    ssm_units: record.job.request.budget.ssm_units,
                    lfm_units: record.job.request.budget.lfm_units,
                    degrade_policy: format!("{:?}", record.job.request.budget.degrade_policy),
                    governor_tier: record.job.request.budget.governor_tier,
                },
            }),
            lifecycle_state: lifecycle_state_name(record.state).to_string(),
            completion_class: is_terminal(record.state)
                .then(|| completion_class_name(record.accounting.completion_class).to_string()),
            execution_path: format!("{:?}", record.execution_path),
            submitted_at_unix_ms: accounting.submitted_at_unix_ms,
            started_at_unix_ms: accounting.started_at_unix_ms,
            finished_at_unix_ms: accounting.finished_at_unix_ms,
            queue_wait_ms: accounting.queue_wait_ms,
            execution_duration_micros: accounting.execution_duration_micros,
            total_duration_ms: accounting.total_duration_ms,
            failure_kind: accounting
                .failure_kind
                .map(|kind| failure_kind_name(kind).to_string()),
            pipeline_state: accounting
                .pipeline_state
                .map(|state| pipeline_state_name(state).to_string()),
            execution_lane: Some(execution_lane_name(accounting.execution_lane).to_string()),
            resource_class: Some(resource_class_name(accounting.resource_class).to_string()),
            capacity_queue_disposition: Some(
                capacity_queue_disposition_name(accounting.capacity_queue_disposition).to_string(),
            ),
            capacity_pressure: Some(
                capacity_pressure_name(accounting.capacity_pressure).to_string(),
            ),
            backend_route: record
                .result
                .as_ref()
                .map(|result| PersistedBackendRouteSummary {
                    pack_id: result.route.pack_id,
                    world_backend: result.route.world_backend,
                    sae_backend: result.route.sae_backend,
                    ssm_backend: result.route.ssm_backend,
                    lfm_backend: result.route.lfm_backend,
                }),
            work_summary: accounting.work_summary.map(|summary| PersistedWorkSummary {
                global_budget_units: summary.global_budget_units,
                global_remaining_units: summary.global_remaining_units,
                world_remaining_units: summary.world_remaining_units,
                sae_remaining_units: summary.sae_remaining_units,
                ssm_remaining_units: summary.ssm_remaining_units,
                lfm_remaining_units: summary.lfm_remaining_units,
                budget_exceeded_stage: summary.budget_exceeded_stage.map(str::to_string),
            }),
            stage_profiles: accounting
                .stage_profiles
                .iter()
                .map(|profile| PersistedStageProfileSummary {
                    stage: format!("{:?}", profile.stage).to_ascii_lowercase(),
                    state: format!("{:?}", profile.state).to_ascii_lowercase(),
                    duration_micros: profile.duration_micros,
                    detail: profile.detail.clone(),
                })
                .collect(),
            hotspot_summary: accounting
                .hotspot_summary
                .map(|hotspot| PersistedHotspotSummary {
                    slowest_stage: hotspot
                        .slowest_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
                    dominant_stage: hotspot
                        .dominant_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
                    dominant_stage_share_bps: hotspot.dominant_stage_share_bps,
                    degraded_stage_count: hotspot.degraded_stage_count,
                    skipped_stage_count: hotspot.skipped_stage_count,
                    unavailable_stage_count: hotspot.unavailable_stage_count,
                    failed_stage_count: hotspot.failed_stage_count,
                }),
            model_slots: accounting
                .model_slots
                .iter()
                .map(|slot| PersistedModelSlotSummary {
                    slot: format!("{:?}", slot.slot),
                    status: format!("{:?}", slot.status),
                    required_for_pack: slot.required_for_pack,
                    warmup_state: extract_warmup_state(slot.detail.as_deref()),
                })
                .collect(),
            recovery_source_job_id: None,
            recovery_status: None,
            recovery_note: None,
        }
    }

    pub fn with_recovery(
        mut self,
        source_job_id: Option<u64>,
        recovery_status: Option<String>,
        recovery_note: Option<String>,
    ) -> Self {
        self.recovery_source_job_id = source_job_id;
        self.recovery_status = recovery_status;
        self.recovery_note = recovery_note;
        self
    }
}

fn extract_warmup_state(detail: Option<&str>) -> Option<String> {
    let detail = detail?;
    if detail.contains("Active:warm:") {
        Some("warm".to_string())
    } else if detail.contains("Active:prepared:")
        || detail.contains("Candidate:prepared:")
        || detail.contains("Compare:prepared:")
        || detail.contains("Shadow:prepared:")
    {
        Some("prepared".to_string())
    } else if detail.contains("Active:blocked:") || detail.contains("Blocked:blocked:") {
        Some("blocked".to_string())
    } else if detail.contains("Active:cold:") {
        Some("cold".to_string())
    } else {
        None
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum JobHistoryStoreError {
    #[error("job history store io error during {operation} at {path}: {reason}")]
    Io {
        operation: &'static str,
        path: String,
        reason: String,
    },
    #[error("job history store corrupted at line {line}: {reason}")]
    Corrupt { line: usize, reason: String },
    #[error("job history encode failure: {reason}")]
    Encode { reason: String },
}

#[derive(Debug, Clone)]
pub struct JobHistoryStore {
    path: PathBuf,
    records: BTreeMap<JobId, PersistedJobRecord>,
}

impl JobHistoryStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, JobHistoryStoreError> {
        let path = path.into();
        let mut records = BTreeMap::new();
        if path.exists() {
            let file = fs::File::open(&path).map_err(|err| JobHistoryStoreError::Io {
                operation: "open",
                path: path.display().to_string(),
                reason: err.to_string(),
            })?;
            for (line_idx, line) in BufReader::new(file).lines().enumerate() {
                let line = line.map_err(|err| JobHistoryStoreError::Io {
                    operation: "read",
                    path: path.display().to_string(),
                    reason: err.to_string(),
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let parsed: PersistedJobRecord =
                    serde_json::from_str(&line).map_err(|err| JobHistoryStoreError::Corrupt {
                        line: line_idx + 1,
                        reason: err.to_string(),
                    })?;
                records.insert(JobId(parsed.job_id), parsed);
            }
        }
        Ok(Self { path, records })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, id: JobId) -> Option<&PersistedJobRecord> {
        self.records.get(&id)
    }

    pub fn records(&self) -> impl Iterator<Item = &PersistedJobRecord> {
        self.records.values()
    }

    pub fn upsert_from_job_record(
        &mut self,
        record: &JobRecord,
    ) -> Result<(), JobHistoryStoreError> {
        let persisted = PersistedJobRecord::from_job_record(record);
        self.upsert(persisted)
    }

    pub fn upsert(&mut self, persisted: PersistedJobRecord) -> Result<(), JobHistoryStoreError> {
        let encoded =
            serde_json::to_string(&persisted).map_err(|err| JobHistoryStoreError::Encode {
                reason: err.to_string(),
            })?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|err| JobHistoryStoreError::Io {
                operation: "mkdir",
                path: parent.display().to_string(),
                reason: err.to_string(),
            })?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|err| JobHistoryStoreError::Io {
                operation: "append-open",
                path: self.path.display().to_string(),
                reason: err.to_string(),
            })?;
        file.write_all(encoded.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.flush())
            .map_err(|err| JobHistoryStoreError::Io {
                operation: "append-write",
                path: self.path.display().to_string(),
                reason: err.to_string(),
            })?;
        self.records.insert(JobId(persisted.job_id), persisted);
        Ok(())
    }
}

fn is_terminal(state: JobLifecycleState) -> bool {
    matches!(
        state,
        JobLifecycleState::Completed
            | JobLifecycleState::Failed
            | JobLifecycleState::TimedOut
            | JobLifecycleState::Rejected
    )
}

fn lifecycle_state_name(state: JobLifecycleState) -> &'static str {
    match state {
        JobLifecycleState::Submitted => "submitted",
        JobLifecycleState::Admitted => "admitted",
        JobLifecycleState::Rejected => "rejected",
        JobLifecycleState::Queued => "queued",
        JobLifecycleState::Running => "running",
        JobLifecycleState::Completed => "completed",
        JobLifecycleState::Failed => "failed",
        JobLifecycleState::TimedOut => "timed_out",
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

fn failure_kind_name(kind: CanonicalFailureKind) -> &'static str {
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

fn pipeline_state_name(state: CanonicalPipelineState) -> &'static str {
    match state {
        CanonicalPipelineState::Ok => "ok",
        CanonicalPipelineState::Degraded => "degraded",
        CanonicalPipelineState::Unavailable => "unavailable",
    }
}

fn execution_lane_name(lane: crate::pipeline::BackendExecutionLane) -> &'static str {
    match lane {
        crate::pipeline::BackendExecutionLane::Toy => "toy",
        crate::pipeline::BackendExecutionLane::Candle => "candle",
        crate::pipeline::BackendExecutionLane::Burn => "burn",
        crate::pipeline::BackendExecutionLane::Mixed => "mixed",
        crate::pipeline::BackendExecutionLane::Worker => "worker",
    }
}

fn resource_class_name(class: ResourceClass) -> &'static str {
    match class {
        ResourceClass::Light => "light",
        ResourceClass::Standard => "standard",
        ResourceClass::Heavy => "heavy",
    }
}

fn capacity_queue_disposition_name(disposition: CapacityQueueDisposition) -> &'static str {
    match disposition {
        CapacityQueueDisposition::None => "none",
        CapacityQueueDisposition::QueuedDueToCapacity => "queued_due_to_capacity",
        CapacityQueueDisposition::DeferredDueToCapacity => "deferred_due_to_capacity",
        CapacityQueueDisposition::RejectedDueToCapacity => "rejected_due_to_capacity",
    }
}

fn capacity_pressure_name(pressure: CapacityPressure) -> &'static str {
    match pressure {
        CapacityPressure::Nominal => "nominal",
        CapacityPressure::Saturated => "saturated",
        CapacityPressure::Overloaded => "overloaded",
    }
}

#[cfg(test)]
mod tests {
    use super::JobHistoryStore;
    use crate::compute_service::JobSubmissionMeta;
    use crate::pipeline::{CanonicalPipelineRequest, ComputePipelineBackend};
    use crate::{ComputeBudget, ComputeInput, FrameId, InMemoryComputeService};

    #[test]
    fn history_store_roundtrips_terminal_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
        let mut service = InMemoryComputeService::new(ComputePipelineBackend::stub());
        let record = service.submit(
            CanonicalPipelineRequest {
                input: ComputeInput {
                    frame_id: FrameId(1),
                    t: 7,
                    context_digest: [9; 32],
                },
                budget: ComputeBudget::default(),
            },
            JobSubmissionMeta {
                submitted_at_unix_ms: 123,
                submitted_by: Some("test".to_string()),
            },
        );
        let job_id = record.job.id;
        service.run_next().expect("run");
        let finished = service.job(job_id).expect("job should exist");

        let mut store = JobHistoryStore::open(&path).expect("open empty");
        store
            .upsert_from_job_record(finished)
            .expect("persist should work");

        let reopened = JobHistoryStore::open(&path).expect("reopen");
        let loaded = reopened.get(job_id).expect("record exists");
        assert_eq!(loaded.job_id, job_id.0);
        assert!(loaded.completion_class.is_some());
        assert!(loaded.finished_at_unix_ms.is_some());
        assert_eq!(loaded.resource_class.as_deref(), Some("light"));
        assert_eq!(loaded.capacity_queue_disposition.as_deref(), Some("none"));
        assert!(loaded.capacity_pressure.is_some());
        assert!(!loaded.stage_profiles.is_empty());
        assert!(loaded.hotspot_summary.is_some());
    }
}
