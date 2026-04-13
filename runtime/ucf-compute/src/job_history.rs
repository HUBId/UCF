use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::compute_service::{
    CapacityPressure, CapacityQueueDisposition, JobCompletionClass, JobId, JobLifecycleState,
    JobRecord, ResourceClass, WorkCostProvenance, WorkCostTension,
};
use crate::pipeline::{
    classify_failure_kind, CanonicalFailureKind, CanonicalFaultDomain,
    CanonicalIsolationDisposition, CanonicalPipelineState,
};

const JOB_HISTORY_SCHEMA_VERSION: u16 = 14;

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
pub struct PersistedWorkCostSummary {
    pub provenance: String,
    pub resource_class: String,
    pub estimated_total_work_units: u64,
    pub runtime_consumed_work_units: Option<u64>,
    pub runtime_remaining_work_units: Option<u64>,
    pub dominant_stage: Option<String>,
    pub dominant_stage_share_bps: Option<u16>,
    #[serde(default)]
    pub dominant_work_stage: Option<String>,
    #[serde(default)]
    pub dominant_work_stage_share_bps: Option<u16>,
    #[serde(default)]
    pub degraded_stage: Option<String>,
    #[serde(default)]
    pub fallback_stage: Option<String>,
    pub degraded_stage_count: u8,
    pub retry_attempts: u8,
    pub redispatched_to_local: bool,
    pub queue_deferred_by_capacity: bool,
    pub pressure: String,
    pub queue_disposition: String,
    pub tension: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedStageProfileSummary {
    pub stage: String,
    pub state: String,
    pub duration_micros: Option<u64>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedStageCostAttribution {
    pub stage: String,
    pub state: String,
    pub timing_micros: Option<u64>,
    pub timing_share_bps: Option<u16>,
    pub work_consumed_units: u64,
    pub work_share_bps: Option<u16>,
    pub pattern: String,
    pub dominant_timing: bool,
    pub dominant_work: bool,
    pub timing_provenance: String,
    pub work_provenance: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedHotspotSummary {
    pub slowest_stage: Option<String>,
    pub dominant_stage: Option<String>,
    pub dominant_stage_share_bps: Option<u16>,
    #[serde(default)]
    pub dominant_work_stage: Option<String>,
    #[serde(default)]
    pub dominant_work_stage_share_bps: Option<u16>,
    #[serde(default)]
    pub degraded_stage: Option<String>,
    #[serde(default)]
    pub fallback_stage: Option<String>,
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
pub struct PersistedRemoteExecutionContext {
    pub was_remote: bool,
    pub execution_path: String,
    #[serde(default)]
    pub execution_lane: Option<String>,
    #[serde(default)]
    pub resource_class: Option<String>,
    #[serde(default)]
    pub capacity_pressure: Option<String>,
    #[serde(default)]
    pub capacity_queue_disposition: Option<String>,
    pub context_completeness: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistedSnapshotReadiness {
    ReplayReady,
    Partial,
    Insufficient,
    StaleOrIncomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedExecutionResultSummary {
    pub lifecycle_state: String,
    pub completion_class: Option<String>,
    pub pipeline_state: Option<String>,
    pub failure_kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedExecutionPathSummary {
    pub requested_execution_path: Option<String>,
    pub executed_execution_path: String,
    pub execution_lane: Option<String>,
    pub resource_class: Option<String>,
    pub was_remote: bool,
    pub redispatched_to_local: bool,
    pub retry_attempts: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedRolloutContextSummary {
    pub active_or_warm_slots: usize,
    pub candidate_or_guarded_slots: usize,
    pub stale_or_blocked_slots: usize,
    pub rollout_context_hint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedExecutionSnapshot {
    pub request: PersistedJobRequestIdentity,
    pub canonical_request_available: bool,
    pub backend_route_available: bool,
    pub model_slot_count: usize,
    pub path: PersistedExecutionPathSummary,
    pub rollout: PersistedRolloutContextSummary,
    pub result: PersistedExecutionResultSummary,
    pub readiness: PersistedSnapshotReadiness,
    #[serde(default)]
    pub deterministic_subset_class: Option<String>,
    #[serde(default)]
    pub deterministic_subset_reasons: Vec<String>,
    #[serde(default)]
    pub stage_capability_contracts: Vec<PersistedStageCapabilityContractSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedStageCapabilityContractSummary {
    pub stage: String,
    pub support: String,
    pub constraints: Vec<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PersistedOptimizationView {
    pub state: String,
    pub bottleneck: String,
    pub queue_pressure: bool,
    pub capacity_pressure: bool,
    pub cold_or_warmup_pressure: bool,
    pub stage_hotspot_pressure: bool,
    pub caveats: Vec<String>,
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
    #[serde(default)]
    pub fault_domain: Option<String>,
    #[serde(default)]
    pub fault_isolation: Option<String>,
    #[serde(default)]
    pub fault_systemic: Option<bool>,
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
    pub work_cost_summary: Option<PersistedWorkCostSummary>,
    #[serde(default)]
    pub stage_profiles: Vec<PersistedStageProfileSummary>,
    #[serde(default)]
    pub stage_cost_attribution: Vec<PersistedStageCostAttribution>,
    #[serde(default)]
    pub hotspot_summary: Option<PersistedHotspotSummary>,
    pub model_slots: Vec<PersistedModelSlotSummary>,
    #[serde(default)]
    pub remote_execution_context: Option<PersistedRemoteExecutionContext>,
    #[serde(default)]
    pub optimization_view: Option<PersistedOptimizationView>,
    #[serde(default)]
    pub execution_snapshot: Option<PersistedExecutionSnapshot>,
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
            fault_domain: accounting
                .failure_kind
                .map(classify_failure_kind)
                .map(|classification| fault_domain_name(classification.domain).to_string()),
            fault_isolation: accounting
                .failure_kind
                .map(classify_failure_kind)
                .map(|classification| fault_isolation_name(classification.isolation).to_string()),
            fault_systemic: accounting
                .failure_kind
                .map(classify_failure_kind)
                .map(|classification| classification.systemic),
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
            work_cost_summary: accounting.work_cost_summary.as_ref().map(|summary| {
                PersistedWorkCostSummary {
                    provenance: work_cost_provenance_name(summary.provenance).to_string(),
                    resource_class: resource_class_name(summary.resource_class).to_string(),
                    estimated_total_work_units: summary.estimated_total_work_units,
                    runtime_consumed_work_units: summary.runtime_consumed_work_units,
                    runtime_remaining_work_units: summary.runtime_remaining_work_units,
                    dominant_stage: summary
                        .dominant_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
                    dominant_stage_share_bps: summary.dominant_stage_share_bps,
                    dominant_work_stage: summary
                        .dominant_work_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
                    dominant_work_stage_share_bps: summary.dominant_work_stage_share_bps,
                    degraded_stage: summary
                        .degraded_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
                    fallback_stage: summary
                        .fallback_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
                    degraded_stage_count: summary.degraded_stage_count,
                    retry_attempts: summary.retry_attempts,
                    redispatched_to_local: summary.redispatched_to_local,
                    queue_deferred_by_capacity: summary.queue_deferred_by_capacity,
                    pressure: capacity_pressure_name(summary.pressure).to_string(),
                    queue_disposition: capacity_queue_disposition_name(summary.queue_disposition)
                        .to_string(),
                    tension: work_cost_tension_name(summary.tension).to_string(),
                }
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
            stage_cost_attribution: accounting
                .stage_cost_attribution
                .iter()
                .map(|entry| PersistedStageCostAttribution {
                    stage: format!("{:?}", entry.stage).to_ascii_lowercase(),
                    state: format!("{:?}", entry.state).to_ascii_lowercase(),
                    timing_micros: entry.timing_micros,
                    timing_share_bps: entry.timing_share_bps,
                    work_consumed_units: entry.work_consumed_units,
                    work_share_bps: entry.work_share_bps,
                    pattern: format!("{:?}", entry.pattern).to_ascii_lowercase(),
                    dominant_timing: entry.dominant_timing,
                    dominant_work: entry.dominant_work,
                    timing_provenance: format!("{:?}", entry.timing_provenance)
                        .to_ascii_lowercase(),
                    work_provenance: format!("{:?}", entry.work_provenance).to_ascii_lowercase(),
                    detail: entry.detail.clone(),
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
                    dominant_work_stage: hotspot
                        .dominant_work_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
                    dominant_work_stage_share_bps: hotspot.dominant_work_stage_share_bps,
                    degraded_stage: hotspot
                        .degraded_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
                    fallback_stage: hotspot
                        .fallback_stage
                        .map(|stage| format!("{:?}", stage).to_ascii_lowercase()),
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
            remote_execution_context: Some(PersistedRemoteExecutionContext {
                was_remote: matches!(
                    record.execution_path,
                    crate::compute_service::JobExecutionPath::WorkerIpc
                ),
                execution_path: format!("{:?}", record.execution_path),
                execution_lane: Some(execution_lane_name(accounting.execution_lane).to_string()),
                resource_class: Some(resource_class_name(accounting.resource_class).to_string()),
                capacity_pressure: Some(
                    capacity_pressure_name(accounting.capacity_pressure).to_string(),
                ),
                capacity_queue_disposition: Some(
                    capacity_queue_disposition_name(accounting.capacity_queue_disposition)
                        .to_string(),
                ),
                context_completeness: if matches!(
                    record.execution_path,
                    crate::compute_service::JobExecutionPath::WorkerIpc
                ) && record.result.is_some()
                {
                    "complete".to_string()
                } else if matches!(
                    record.execution_path,
                    crate::compute_service::JobExecutionPath::WorkerIpc
                ) {
                    "partial".to_string()
                } else {
                    "not_applicable".to_string()
                },
            }),
            optimization_view: Some(build_persisted_optimization_view(accounting)),
            execution_snapshot: Some(build_execution_snapshot(record)),
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
    } else if detail.contains("Active:stale:")
        || detail.contains("Candidate:stale:")
        || detail.contains("Compare:stale:")
        || detail.contains("Shadow:stale:")
    {
        Some("stale".to_string())
    } else if detail.contains("Active:cold:") {
        Some("cold".to_string())
    } else {
        None
    }
}

fn build_persisted_optimization_view(
    accounting: &crate::compute_service::JobAccountingSummary,
) -> PersistedOptimizationView {
    let queue_pressure = matches!(
        accounting.capacity_queue_disposition,
        CapacityQueueDisposition::QueuedDueToCapacity
            | CapacityQueueDisposition::DeferredDueToCapacity
    ) || accounting.queue_wait_ms.is_some_and(|wait| wait > 0);
    let capacity_pressure = matches!(
        accounting.capacity_pressure,
        CapacityPressure::Constrained
            | CapacityPressure::Saturated
            | CapacityPressure::Backpressured
            | CapacityPressure::TemporarilyUnschedulable
    );
    let cold_or_warmup_pressure = accounting
        .model_slots
        .iter()
        .filter_map(|slot| extract_warmup_state(slot.detail.as_deref()))
        .any(|state| state == "cold" || state == "blocked" || state == "stale");
    let stage_hotspot_pressure = accounting.hotspot_summary.is_some_and(|hotspot| {
        hotspot.degraded_stage_count > 0
            || hotspot
                .dominant_stage_share_bps
                .is_some_and(|bps| bps >= 6_500)
    });
    let mut caveats = Vec::new();
    if queue_pressure {
        caveats.push("queue_pressure".to_string());
    }
    if capacity_pressure {
        caveats.push("capacity_pressure".to_string());
    }
    if cold_or_warmup_pressure {
        caveats.push("cold_or_warmup_pressure".to_string());
    }
    if stage_hotspot_pressure {
        caveats.push("stage_hotspot_or_degraded_path".to_string());
    }
    let signals = [
        queue_pressure,
        capacity_pressure,
        cold_or_warmup_pressure,
        stage_hotspot_pressure,
    ]
    .into_iter()
    .filter(|s| *s)
    .count();
    let (state, bottleneck) = if signals == 0 {
        ("healthy_and_efficient", "none")
    } else if signals > 1 {
        ("mixed_optimization_picture", "mixed")
    } else if queue_pressure || capacity_pressure {
        ("constrained_by_capacity", "capacity_or_queue")
    } else if cold_or_warmup_pressure {
        ("constrained_by_cold_or_warmup", "warmup_readiness")
    } else if stage_hotspot_pressure {
        ("constrained_by_dominant_stage_hotspot", "stage_hotspot")
    } else {
        ("inconclusive", "none")
    };

    PersistedOptimizationView {
        state: state.to_string(),
        bottleneck: bottleneck.to_string(),
        queue_pressure,
        capacity_pressure,
        cold_or_warmup_pressure,
        stage_hotspot_pressure,
        caveats,
    }
}

fn build_execution_snapshot(record: &JobRecord) -> PersistedExecutionSnapshot {
    let request = &record.job.request.input;
    let work_cost = record.accounting.work_cost_summary.as_ref();
    let execution_path = format!("{:?}", record.execution_path);
    let execution_lane = Some(execution_lane_name(record.accounting.execution_lane).to_string());
    let resource_class = Some(resource_class_name(record.accounting.resource_class).to_string());
    let was_remote = matches!(
        record.execution_path,
        crate::compute_service::JobExecutionPath::WorkerIpc
    );
    let redispatched_to_local = work_cost.is_some_and(|summary| summary.redispatched_to_local);
    let retry_attempts = work_cost.map_or(0, |summary| summary.retry_attempts);
    let (active_or_warm_slots, candidate_or_guarded_slots, stale_or_blocked_slots) =
        classify_rollout_slots(&record.accounting.model_slots);
    let readiness = derive_snapshot_readiness(record, was_remote);
    let (deterministic_subset_class, deterministic_subset_reasons) =
        classify_deterministic_subset_snapshot(
            readiness,
            was_remote,
            redispatched_to_local,
            retry_attempts,
            record.accounting.work_cost_summary.as_ref(),
            candidate_or_guarded_slots,
            stale_or_blocked_slots,
        );
    PersistedExecutionSnapshot {
        request: PersistedJobRequestIdentity {
            frame_id: request.frame_id.0,
            t: request.t,
            context_digest_hex: hex::encode(request.context_digest),
        },
        canonical_request_available: true,
        backend_route_available: record.result.is_some(),
        model_slot_count: record.accounting.model_slots.len(),
        path: PersistedExecutionPathSummary {
            requested_execution_path: Some(execution_path.clone()),
            executed_execution_path: execution_path,
            execution_lane,
            resource_class,
            was_remote,
            redispatched_to_local,
            retry_attempts,
        },
        rollout: PersistedRolloutContextSummary {
            active_or_warm_slots,
            candidate_or_guarded_slots,
            stale_or_blocked_slots,
            rollout_context_hint: rollout_context_hint(
                active_or_warm_slots,
                candidate_or_guarded_slots,
                stale_or_blocked_slots,
            ),
        },
        result: PersistedExecutionResultSummary {
            lifecycle_state: lifecycle_state_name(record.state).to_string(),
            completion_class: is_terminal(record.state)
                .then(|| completion_class_name(record.accounting.completion_class).to_string()),
            pipeline_state: record
                .accounting
                .pipeline_state
                .map(|state| pipeline_state_name(state).to_string()),
            failure_kind: record
                .accounting
                .failure_kind
                .map(|kind| failure_kind_name(kind).to_string()),
        },
        readiness,
        deterministic_subset_class: Some(deterministic_subset_class),
        deterministic_subset_reasons,
        stage_capability_contracts: record
            .result
            .as_ref()
            .map(|result| {
                result
                    .stage_capability_contracts
                    .iter()
                    .map(|entry| PersistedStageCapabilityContractSummary {
                        stage: format!("{:?}", entry.stage).to_ascii_lowercase(),
                        support: format!("{:?}", entry.contract.support).to_ascii_lowercase(),
                        constraints: entry
                            .contract
                            .constraints
                            .iter()
                            .map(|constraint| {
                                format!("{constraint:?}")
                                    .trim_start_matches("Capability")
                                    .to_ascii_lowercase()
                            })
                            .collect(),
                        detail: entry.contract.detail.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn classify_deterministic_subset_snapshot(
    readiness: PersistedSnapshotReadiness,
    was_remote: bool,
    redispatched_to_local: bool,
    retry_attempts: u8,
    work_cost: Option<&crate::compute_service::ConsolidatedWorkCostSummary>,
    candidate_or_guarded_slots: usize,
    stale_or_blocked_slots: usize,
) -> (String, Vec<String>) {
    let mut reasons = Vec::new();
    if matches!(
        readiness,
        PersistedSnapshotReadiness::Insufficient | PersistedSnapshotReadiness::StaleOrIncomplete
    ) {
        reasons.push("incomplete_snapshot".to_string());
        return ("excluded_from_deterministic_subset".to_string(), reasons);
    }
    if readiness == PersistedSnapshotReadiness::Partial {
        reasons.push("insufficient_snapshot_signal".to_string());
        return ("deterministic_subset_uncertain".to_string(), reasons);
    }
    if was_remote {
        reasons.push("remote_worker_context".to_string());
    }
    if candidate_or_guarded_slots > 0 || stale_or_blocked_slots > 0 {
        reasons.push("rollout_boundary_relevant".to_string());
    }
    if redispatched_to_local || retry_attempts > 0 {
        reasons.push("retry_or_redispatch_path".to_string());
    }
    if work_cost.is_some_and(|summary| {
        summary.degraded_stage_count > 0
            || summary.fallback_stage.is_some()
            || summary.queue_deferred_by_capacity
    }) {
        reasons.push("degraded_or_fallback_context".to_string());
    }
    if reasons.is_empty() {
        ("deterministic_subset_candidate".to_string(), Vec::new())
    } else {
        (
            "replayable_but_not_deterministic_subset".to_string(),
            reasons,
        )
    }
}

fn derive_snapshot_readiness(record: &JobRecord, was_remote: bool) -> PersistedSnapshotReadiness {
    if record.result.is_none() && !is_terminal(record.state) {
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

fn classify_rollout_slots(
    slots: &[crate::backend_pack::ModelSlotProvenance],
) -> (usize, usize, usize) {
    let mut active_or_warm_slots = 0usize;
    let mut candidate_or_guarded_slots = 0usize;
    let mut stale_or_blocked_slots = 0usize;
    for slot in slots {
        let detail = slot.detail.as_deref().unwrap_or_default();
        if detail.contains("warm:")
            || detail.contains("Active:prepared:")
            || detail.contains("Compare:prepared:")
        {
            active_or_warm_slots = active_or_warm_slots.saturating_add(1);
        }
        if detail.contains("Candidate:")
            || detail.contains("Compare:")
            || detail.contains("Shadow:")
            || detail.contains("Guarded")
        {
            candidate_or_guarded_slots = candidate_or_guarded_slots.saturating_add(1);
        }
        if detail.contains("stale:") || detail.contains("blocked:") {
            stale_or_blocked_slots = stale_or_blocked_slots.saturating_add(1);
        }
    }
    (
        active_or_warm_slots,
        candidate_or_guarded_slots,
        stale_or_blocked_slots,
    )
}

fn rollout_context_hint(
    active_or_warm: usize,
    candidate_or_guarded: usize,
    stale: usize,
) -> String {
    if stale > 0 {
        "fallback_or_stale_path".to_string()
    } else if candidate_or_guarded > 0 {
        "guarded_or_candidate_path".to_string()
    } else if active_or_warm > 0 {
        "active_path".to_string()
    } else {
        "unknown".to_string()
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

fn fault_domain_name(domain: CanonicalFaultDomain) -> &'static str {
    match domain {
        CanonicalFaultDomain::ArtifactModel => "artifact_model",
        CanonicalFaultDomain::Stage => "stage",
        CanonicalFaultDomain::Backend => "backend",
        CanonicalFaultDomain::WorkerTransport => "worker_transport",
        CanonicalFaultDomain::PlacementCapacity => "placement_capacity",
        CanonicalFaultDomain::RuntimeService => "runtime_service",
    }
}

fn fault_isolation_name(isolation: CanonicalIsolationDisposition) -> &'static str {
    match isolation {
        CanonicalIsolationDisposition::LocallyIsolated => "locally_isolated",
        CanonicalIsolationDisposition::DegradedButServiceable => "degraded_but_serviceable",
        CanonicalIsolationDisposition::HardEscalationJobFailure => "hard_escalation_job_failure",
        CanonicalIsolationDisposition::ServiceRuntimeImpact => "service_runtime_impact",
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
        CapacityQueueDisposition::DegradedPlacementDueToPressure => {
            "degraded_placement_due_to_pressure"
        }
        CapacityQueueDisposition::RejectedDueToCapacity => "rejected_due_to_capacity",
    }
}

fn capacity_pressure_name(pressure: CapacityPressure) -> &'static str {
    match pressure {
        CapacityPressure::Healthy => "healthy",
        CapacityPressure::Constrained => "constrained",
        CapacityPressure::Saturated => "saturated",
        CapacityPressure::Backpressured => "backpressured",
        CapacityPressure::TemporarilyUnschedulable => "temporarily_unschedulable",
    }
}

fn work_cost_provenance_name(provenance: WorkCostProvenance) -> &'static str {
    match provenance {
        WorkCostProvenance::EstimatedFromBudget => "estimated_from_budget",
        WorkCostProvenance::RuntimeMeasured => "runtime_measured",
    }
}

fn work_cost_tension_name(tension: WorkCostTension) -> &'static str {
    match tension {
        WorkCostTension::Nominal => "nominal",
        WorkCostTension::ExpensiveButSuccessful => "expensive_but_successful",
        WorkCostTension::ExpensiveAndDegraded => "expensive_and_degraded",
        WorkCostTension::RetriedWithAdditionalCost => "retried_with_additional_cost",
        WorkCostTension::LowCostButBlocked => "low_cost_but_blocked",
    }
}

#[cfg(test)]
mod tests {
    use super::{JobHistoryStore, PersistedSnapshotReadiness};
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
        assert!(!loaded.stage_cost_attribution.is_empty());
        assert!(loaded.hotspot_summary.is_some());
        assert!(loaded.optimization_view.is_some());
        let snapshot = loaded
            .execution_snapshot
            .as_ref()
            .expect("execution snapshot should exist");
        assert_eq!(snapshot.readiness, PersistedSnapshotReadiness::ReplayReady);
        assert!(snapshot.canonical_request_available);
        assert!(matches!(
            snapshot.deterministic_subset_class.as_deref(),
            Some("deterministic_subset_candidate")
                | Some("replayable_but_not_deterministic_subset")
        ));
    }

    #[test]
    fn history_persists_fault_domain_for_failed_job() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
        let mut service = InMemoryComputeService::new(ComputePipelineBackend::stub());
        let record = service.submit(
            CanonicalPipelineRequest {
                input: ComputeInput {
                    frame_id: FrameId(2),
                    t: 0,
                    context_digest: [3; 32],
                },
                budget: ComputeBudget::default(),
            },
            JobSubmissionMeta {
                submitted_at_unix_ms: 124,
                submitted_by: Some("test".to_string()),
            },
        );
        let mut store = JobHistoryStore::open(&path).expect("open empty");
        store.upsert_from_job_record(record).expect("persist");
        let loaded = store.get(record.job.id).expect("record exists");
        assert_eq!(loaded.failure_kind.as_deref(), Some("invalid_input"));
        assert_eq!(loaded.fault_domain.as_deref(), Some("runtime_service"));
        assert_eq!(
            loaded.fault_isolation.as_deref(),
            Some("service_runtime_impact")
        );
        assert_eq!(loaded.fault_systemic, Some(true));
        assert!(loaded.optimization_view.is_some());
        assert!(loaded.execution_snapshot.is_some());
    }
}
