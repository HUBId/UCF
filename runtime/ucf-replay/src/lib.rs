#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{collections::HashMap, fmt::Write};

use hex::FromHex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use ucf_compute::ComputeSignalsSummary as RecomputedComputeSummary;
use ucf_compute::{
    build_backend, compute_input_from_control, stable_budget_profile_id, ComputeBackendConfig,
    ComputeBackendKind, ComputeBudget,
};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{
    AuditPayload, CapabilityIssuanceRecord, ExperienceKind, ExperiencePayload, ExperienceRecord,
    LfmSummaryRecord,
};
use ucf_frames::v1::{
    ChannelCode, ComputeSignalsSummary, ControlFrame, CorrelationId, DecisionFrame, Intent,
    IntentId, IntentKind,
};

const REPORT_CAP: usize = 1000;
const REPLAY_DIVERGENCE_CAP: usize = 64;
static UCF_COMPUTE_CHAIN_MISMATCH_TOTAL: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayStrictness {
    VerifyOnly,
    RecomputeStages,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayPlan {
    pub t0: u64,
    pub t1: u64,
    pub expected_backend_pack_digest: Option<[u8; 32]>,
    pub strictness: ReplayStrictness,
    pub stop_on_first_divergence: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayOverallStatus {
    Ok,
    DriftFound,
    MissingData,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayComponent {
    BackendPack,
    World,
    Sae,
    Ssm,
    Lfm,
    Risk,
    Nsr,
    Coherence,
    Governor,
    Issuance,
    Output,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Divergence {
    pub t: u64,
    pub component: ReplayComponent,
    pub expected_digest: String,
    pub observed_digest: String,
    pub hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCounters {
    pub missing_records: u64,
    pub mismatched_digests: u64,
    pub degraded_steps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayReport {
    pub range: (u64, u64),
    pub overall_status: ReplayOverallStatus,
    pub first_divergence: Option<Divergence>,
    pub counters: ReplayCounters,
    pub details: Vec<Divergence>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    ComputeOnly,
    DecisionScoring,
    FullNoAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySpec {
    pub from_tick: u64,
    pub to_tick: u64,
    pub backend_override: Option<ComputeBackendKind>,
    pub seed_override: Option<u64>,
    pub budget_override: Option<u32>,
    pub mode: ReplayMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPolicy {
    pub eps: f32,
    pub digest_allowlist: Vec<String>,
}

impl DiffPolicy {
    pub fn for_backend(backend: &str) -> Self {
        let eps = if backend == "stub" { 1e-6 } else { 1e-5 };
        Self {
            eps,
            digest_allowlist: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    pub total_items: usize,
    pub matched: usize,
    pub drifted: usize,
    pub unreplayable: usize,
    pub items: Vec<ReplayItem>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayItem {
    pub decision_id: u64,
    pub correlation_id: u64,
    pub persisted: PersistedSummary,
    pub recomputed: Option<RecomputedSummary>,
    pub diff: DiffSummary,
    pub status: ReplayStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSummary {
    pub backend: String,
    pub risk: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub pressure: f32,
    pub risk_quality: Option<u8>,
    pub spikes_digest_hex: String,
    pub context_digest_hex: Option<String>,
    pub chain_digest_hex: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecomputedSummary {
    pub backend: String,
    pub risk: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub pressure: f32,
    pub risk_quality: Option<u8>,
    pub spikes_digest_hex: String,
    pub context_digest_hex: Option<String>,
    pub chain_digest_hex: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub risk_abs: Option<f32>,
    pub confidence_abs: Option<f32>,
    pub surprise_abs: Option<f32>,
    pub pressure_abs: Option<f32>,
    pub pass: bool,
    pub reasons: Vec<DriftReason>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayStatus {
    Match,
    Drift,
    Unreplayable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DriftReason {
    DigestMismatch {
        field: String,
        expected_prefix: String,
        got_prefix: String,
    },
    FloatMismatch {
        field: String,
        expected: f32,
        got: f32,
        abs_diff: f32,
    },
    MissingPersistedField {
        field: String,
    },
    BackendUnavailable {
        backend_profile: String,
    },
    DecisionScoringUnavailable,
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn replay_records(records: &[ExperienceRecord], spec: &ReplaySpec) -> ReplayResult {
    let mut items = Vec::new();
    let mut matched = 0usize;
    let mut drifted = 0usize;
    let mut unreplayable = 0usize;

    for rec in records {
        if rec.kind != ExperienceKind::DecisionOut {
            continue;
        }
        if rec.time.tick.get() < spec.from_tick || rec.time.tick.get() > spec.to_tick {
            continue;
        }

        let decision = match &rec.payload {
            ExperiencePayload::Decision(d) => d,
            _ => continue,
        };

        let persisted = match decision.compute_summary {
            Some(summary) => summary,
            None => {
                let item = ReplayItem {
                    decision_id: rec.id.0,
                    correlation_id: rec.corr.0,
                    persisted: empty_persisted("missing"),
                    recomputed: None,
                    diff: DiffSummary {
                        risk_abs: None,
                        confidence_abs: None,
                        surprise_abs: None,
                        pressure_abs: None,
                        pass: false,
                        reasons: vec![DriftReason::MissingPersistedField {
                            field: "compute_summary".to_string(),
                        }],
                    },
                    status: ReplayStatus::Unreplayable,
                };
                unreplayable += 1;
                items.push(item);
                continue;
            }
        };

        let control = records
            .iter()
            .rev()
            .find(|candidate| {
                candidate.kind == ExperienceKind::ControlIn
                    && candidate.corr == rec.corr
                    && candidate.time.tick.get() <= rec.time.tick.get()
            })
            .and_then(|candidate| match &candidate.payload {
                ExperiencePayload::Control(ctrl) => Some(ctrl.clone()),
                _ => None,
            });

        let Some(control) = control else {
            let item = ReplayItem {
                decision_id: rec.id.0,
                correlation_id: rec.corr.0,
                persisted: to_persisted(&persisted),
                recomputed: None,
                diff: DiffSummary {
                    risk_abs: None,
                    confidence_abs: None,
                    surprise_abs: None,
                    pressure_abs: None,
                    pass: false,
                    reasons: vec![DriftReason::MissingPersistedField {
                        field: "control_frame".to_string(),
                    }],
                },
                status: ReplayStatus::Unreplayable,
            };
            unreplayable += 1;
            items.push(item);
            continue;
        };

        let backend_kind = spec
            .backend_override
            .or_else(|| {
                ComputeBackendKind::parse(persisted.backend_profile.unwrap_or(persisted.backend))
            })
            .unwrap_or(ComputeBackendKind::Stub);
        let seed = spec
            .seed_override
            .or(persisted.seed)
            .unwrap_or(ComputeBudget::default().seed);
        let _budget_profile = spec
            .budget_override
            .or(persisted.budget_profile_id)
            .unwrap_or(stable_budget_profile_id(
                ComputeBudget::default().max_micros,
                ComputeBudget::default().hard_timeout_micros,
            ));

        let cfg = ComputeBackendConfig {
            kind: backend_kind,
            seed,
            ..ComputeBackendConfig::default()
        };
        let backend = match build_backend(&cfg) {
            Ok(backend) => backend,
            Err(_) => {
                let item = ReplayItem {
                    decision_id: rec.id.0,
                    correlation_id: rec.corr.0,
                    persisted: to_persisted(&persisted),
                    recomputed: None,
                    diff: DiffSummary {
                        risk_abs: None,
                        confidence_abs: None,
                        surprise_abs: None,
                        pressure_abs: None,
                        pass: false,
                        reasons: vec![DriftReason::BackendUnavailable {
                            backend_profile: format!("{:?}", backend_kind),
                        }],
                    },
                    status: ReplayStatus::Unreplayable,
                };
                unreplayable += 1;
                items.push(item);
                continue;
            }
        };

        let recomputed = match backend.compute(
            &compute_input_from_control(&control),
            ComputeBudget {
                seed,
                ..ComputeBudget::default()
            },
        ) {
            Ok(signals) => signals.summary(backend.name()),
            Err(_) => {
                unreplayable += 1;
                items.push(ReplayItem {
                    decision_id: rec.id.0,
                    correlation_id: rec.corr.0,
                    persisted: to_persisted(&persisted),
                    recomputed: None,
                    diff: DiffSummary {
                        risk_abs: None,
                        confidence_abs: None,
                        surprise_abs: None,
                        pressure_abs: None,
                        pass: false,
                        reasons: vec![DriftReason::BackendUnavailable {
                            backend_profile: backend.name().to_string(),
                        }],
                    },
                    status: ReplayStatus::Unreplayable,
                });
                continue;
            }
        };

        let policy = DiffPolicy::for_backend(recomputed.backend);
        let mut reasons = compare_summaries(&persisted, &recomputed, &policy);
        if matches!(
            spec.mode,
            ReplayMode::DecisionScoring | ReplayMode::FullNoAction
        ) {
            reasons.push(DriftReason::DecisionScoringUnavailable);
        }

        let diff = DiffSummary {
            risk_abs: Some((persisted.risk - recomputed.risk).abs()),
            confidence_abs: Some((persisted.confidence - recomputed.confidence).abs()),
            surprise_abs: Some((persisted.surprise - recomputed.surprise).abs()),
            pressure_abs: Some((persisted.pressure - recomputed.pressure).abs()),
            pass: reasons.is_empty(),
            reasons,
        };

        let status = if diff.pass {
            matched += 1;
            ReplayStatus::Match
        } else {
            drifted += 1;
            ReplayStatus::Drift
        };

        items.push(ReplayItem {
            decision_id: rec.id.0,
            correlation_id: rec.corr.0,
            persisted: to_persisted(&persisted),
            recomputed: Some(to_recomputed(&recomputed)),
            diff,
            status,
        });
    }

    let total_items = items.len();
    let truncated = items.len() > REPORT_CAP;
    items.truncate(REPORT_CAP);

    ReplayResult {
        total_items,
        matched,
        drifted,
        unreplayable,
        items,
        truncated,
    }
}

fn compare_summaries(
    persisted: &ComputeSignalsSummary,
    recomputed: &RecomputedComputeSummary,
    policy: &DiffPolicy,
) -> Vec<DriftReason> {
    if persisted.compute_chain_digest == Some(recomputed.compute_chain_digest) {
        return Vec::new();
    }

    let mut reasons = Vec::new();

    if let Some(expected) = persisted.compute_chain_digest {
        if expected != recomputed.compute_chain_digest {
            UCF_COMPUTE_CHAIN_MISMATCH_TOTAL.fetch_add(1, Ordering::Relaxed);
            reasons.push(DriftReason::DigestMismatch {
                field: "compute_chain_digest".to_string(),
                expected_prefix: opt_digest_prefix(Some(expected)),
                got_prefix: opt_digest_prefix(Some(recomputed.compute_chain_digest)),
            });
        }
    }

    compare_float(
        "risk",
        persisted.risk,
        recomputed.risk,
        policy.eps,
        &mut reasons,
    );
    compare_float(
        "confidence",
        persisted.confidence,
        recomputed.confidence,
        policy.eps,
        &mut reasons,
    );
    compare_float(
        "surprise",
        persisted.surprise,
        recomputed.surprise,
        policy.eps,
        &mut reasons,
    );
    compare_float(
        "pressure",
        persisted.pressure,
        recomputed.pressure,
        policy.eps,
        &mut reasons,
    );

    if persisted.spikes_digest != recomputed.spikes_digest {
        reasons.push(DriftReason::DigestMismatch {
            field: "spikes_digest".to_string(),
            expected_prefix: hex::encode(&persisted.spikes_digest[..6]),
            got_prefix: hex::encode(&recomputed.spikes_digest[..6]),
        });
    }
    if persisted.evidence_context_digest != Some(recomputed.evidence_context_digest) {
        reasons.push(DriftReason::DigestMismatch {
            field: "evidence_context_digest".to_string(),
            expected_prefix: opt_digest_prefix(persisted.evidence_context_digest),
            got_prefix: opt_digest_prefix(Some(recomputed.evidence_context_digest)),
        });
    }

    reasons
}

fn compare_float(field: &str, expected: f32, got: f32, eps: f32, reasons: &mut Vec<DriftReason>) {
    let abs_diff = (expected - got).abs();
    if abs_diff > eps {
        reasons.push(DriftReason::FloatMismatch {
            field: field.to_string(),
            expected,
            got,
            abs_diff,
        });
    }
}

fn to_persisted(summary: &ComputeSignalsSummary) -> PersistedSummary {
    PersistedSummary {
        backend: summary.backend.to_string(),
        risk: summary.risk,
        confidence: summary.confidence,
        surprise: summary.surprise,
        pressure: summary.pressure,
        risk_quality: summary.risk_quality,
        spikes_digest_hex: hex::encode(summary.spikes_digest),
        context_digest_hex: summary.evidence_context_digest.map(hex::encode),
        chain_digest_hex: summary.compute_chain_digest.map(hex::encode),
    }
}

fn to_recomputed(summary: &RecomputedComputeSummary) -> RecomputedSummary {
    RecomputedSummary {
        backend: summary.backend.to_string(),
        risk: summary.risk,
        confidence: summary.confidence,
        surprise: summary.surprise,
        pressure: summary.pressure,
        risk_quality: Some(summary.risk_quality),
        spikes_digest_hex: hex::encode(summary.spikes_digest),
        context_digest_hex: Some(hex::encode(summary.evidence_context_digest)),
        chain_digest_hex: Some(hex::encode(summary.compute_chain_digest)),
    }
}

fn empty_persisted(backend: &str) -> PersistedSummary {
    PersistedSummary {
        backend: backend.to_string(),
        risk: 0.0,
        confidence: 0.0,
        surprise: 0.0,
        pressure: 0.0,
        risk_quality: None,
        spikes_digest_hex: String::new(),
        context_digest_hex: None,
        chain_digest_hex: None,
    }
}

fn opt_digest_prefix(value: Option<[u8; 32]>) -> String {
    value
        .map(|digest| hex::encode(&digest[..6]))
        .unwrap_or_else(|| "none".to_string())
}

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub decisions: Vec<FixtureDecision>,
}

#[derive(Debug, Deserialize)]
pub struct FixtureDecision {
    pub decision_id: u64,
    pub corr: u64,
    pub tick: u64,
    pub window: u64,
    pub text: String,
    pub backend: String,
    pub risk: f32,
    pub confidence: f32,
    pub surprise: f32,
    pub pressure: f32,
    pub spike_count: u16,
    pub spikes_digest_hex: String,
    pub evidence_context_digest_hex: String,
    pub budget_profile_id: u32,
    pub seed: u64,
    pub risk_quality: u8,
}

pub fn load_fixture_records(path: &Path) -> Result<Vec<ExperienceRecord>, ReplayError> {
    let data = fs::read_to_string(path)?;
    let fixture: Fixture = serde_json::from_str(&data)?;
    let mut out = Vec::new();

    for entry in fixture.decisions {
        let time = SimTime {
            tick: Tick::new(entry.tick),
            window: WindowId::new(entry.window),
        };
        let ctrl = ControlFrame::new_text(
            time,
            CorrelationId(entry.corr),
            ChannelCode::ExternalOutput,
            Intent::new(IntentId(entry.corr), IntentKind::Speak, "fixture"),
            entry.text,
        );
        out.push(ExperienceRecord::from_control(
            ucf_ess::v1::ExperienceId(entry.decision_id * 10),
            ctrl,
        ));

        let spikes_digest = <[u8; 32]>::from_hex(entry.spikes_digest_hex).unwrap_or([0; 32]);
        let context_digest =
            <[u8; 32]>::from_hex(entry.evidence_context_digest_hex).unwrap_or([0; 32]);
        let backend_name = entry.backend.clone();
        let summary = ComputeSignalsSummary {
            backend: leak_str(backend_name.clone()),
            surprise: entry.surprise,
            pressure: entry.pressure,
            risk: entry.risk,
            confidence: entry.confidence,
            spike_count: entry.spike_count,
            spikes_digest,
            sparsity: None,
            energy: None,
            ssm_readout: None,
            ssm_digest: None,
            world_digest: None,
            risk_quality: Some(entry.risk_quality),
            evidence_context_digest: Some(context_digest),
            evidence_world_digest: None,
            evidence_spikes_digest: None,
            evidence_ssm_digest: None,
            evidence_lfm_digest: None,
            backend_profile: Some(leak_str(backend_name)),
            backend_pack_id: None,
            fixtures_digest: None,
            llm_backend: None,
            world_backend: None,
            sae_backend: None,
            ssm_backend: None,
            lfm_backend: None,
            lfm_uncertainty: None,
            lfm_stability: None,
            lfm_digest: None,
            budget_profile_id: Some(entry.budget_profile_id),
            seed: Some(entry.seed),
            risk_contract_version: Some(1),
            compute_schema_version: Some(1),
            compute_chain_digest: None,
            compute_code_version: None,
            budget_exceeded_stage: None,
            lfm_quality: None,
            coherence: None,
            instability: None,
            phi_proxy: None,
            coherence_digest: None,
        };

        let decision = DecisionFrame::allow(time, CorrelationId(entry.corr), "fixture")
            .with_compute_summary(summary);
        out.push(ExperienceRecord::from_decision(
            ucf_ess::v1::ExperienceId(entry.decision_id),
            decision,
        ));
    }

    Ok(out)
}

fn leak_str(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

pub fn write_report(path: &Path, result: &ReplayResult) -> Result<(), ReplayError> {
    let body = serde_json::to_string_pretty(result)?;
    fs::write(path, body)?;
    Ok(())
}

pub fn ucf_compute_chain_mismatch_total() -> u64 {
    UCF_COMPUTE_CHAIN_MISMATCH_TOTAL.load(Ordering::Relaxed)
}

pub fn replay_audit(records: &[ExperienceRecord], plan: &ReplayPlan) -> ReplayReport {
    let mut report = ReplayReport {
        range: (plan.t0, plan.t1),
        overall_status: ReplayOverallStatus::Ok,
        first_divergence: None,
        counters: ReplayCounters {
            missing_records: 0,
            mismatched_digests: 0,
            degraded_steps: 0,
        },
        details: Vec::new(),
    };

    let in_range: Vec<&ExperienceRecord> = records
        .iter()
        .filter(|r| {
            let t = r.time.tick.get();
            t >= plan.t0 && t <= plan.t1
        })
        .collect();

    if in_range.is_empty() {
        report.overall_status = ReplayOverallStatus::MissingData;
        report.counters.missing_records += 1;
        return report;
    }

    let mut decision_chain_by_corr = HashMap::new();
    let mut decision_by_id = HashMap::new();

    for record in &in_range {
        if let ExperiencePayload::Decision(decision) = &record.payload {
            decision_by_id.insert(record.id.0, (**decision).clone());
            if let Some(summary) = decision.compute_summary {
                if let Some(chain) = summary.compute_chain_digest {
                    decision_chain_by_corr.insert(record.corr.0, chain);
                } else {
                    push_divergence(
                        &mut report,
                        Divergence {
                            t: record.time.tick.get(),
                            component: ReplayComponent::Risk,
                            expected_digest: "present".to_string(),
                            observed_digest: "missing".to_string(),
                            hint: "compute_chain_digest missing in DecisionFrame.compute_summary"
                                .to_string(),
                        },
                    );
                }
                verify_summary_links(
                    &mut report,
                    record.time.tick.get(),
                    summary,
                    plan.strictness,
                );
            } else {
                report.counters.missing_records += 1;
            }
        }
    }

    verify_backend_pack(records, plan, &mut report);

    for record in &in_range {
        let t = record.time.tick.get();
        match (&record.kind, &record.payload) {
            (ExperienceKind::Nsr, _) => {
                if let Some(nsr) = &record.nsr_record {
                    verify_chain_ref(
                        &mut report,
                        t,
                        ReplayComponent::Nsr,
                        nsr.evidence_chain_digest,
                        decision_chain_by_corr.get(&record.corr.0).copied(),
                    );
                }
            }
            (ExperienceKind::LfmSummary, _) => {
                if let Some(summary) = record.lfm_summary_record {
                    verify_lfm_summary_digest(&mut report, t, summary);
                    verify_chain_ref(
                        &mut report,
                        t,
                        ReplayComponent::Lfm,
                        summary.evidence_chain_digest,
                        decision_chain_by_corr.get(&record.corr.0).copied(),
                    );
                }
            }
            (
                ExperienceKind::CapabilityIssuance,
                ExperiencePayload::Audit(AuditPayload::CapabilityIssuance(issuance)),
            ) => {
                verify_chain_ref(
                    &mut report,
                    t,
                    ReplayComponent::Issuance,
                    issuance.evidence_chain_digest,
                    decision_chain_by_corr.get(&record.corr.0).copied(),
                );
                verify_issuance_decision(
                    &mut report,
                    t,
                    issuance,
                    &decision_by_id,
                    records,
                    plan.strictness,
                );
            }
            (ExperienceKind::Output, ExperiencePayload::Audit(AuditPayload::Output(out))) => {
                verify_chain_ref(
                    &mut report,
                    t,
                    ReplayComponent::Output,
                    out.evidence_chain_digest,
                    decision_chain_by_corr.get(&record.corr.0).copied(),
                );
                if out.llm_request_digest == [0; 32] || out.llm_response_digest == [0; 32] {
                    push_divergence(
                        &mut report,
                        Divergence {
                            t,
                            component: ReplayComponent::Output,
                            expected_digest: "non_zero".to_string(),
                            observed_digest: "zero".to_string(),
                            hint: "llm request/response digest missing or zero".to_string(),
                        },
                    );
                }
            }
            _ => {}
        }

        if plan.stop_on_first_divergence && report.first_divergence.is_some() {
            finalize_status(&mut report);
            return report;
        }
    }

    if matches!(plan.strictness, ReplayStrictness::RecomputeStages) {
        recompute_decision_chain(&mut report, &in_range);
    }

    finalize_status(&mut report);
    report
}

fn verify_summary_links(
    report: &mut ReplayReport,
    t: u64,
    summary: ComputeSignalsSummary,
    strictness: ReplayStrictness,
) {
    let checks = [
        (
            summary.world_digest,
            summary.evidence_world_digest,
            ReplayComponent::World,
        ),
        (
            Some(summary.spikes_digest),
            summary.evidence_spikes_digest,
            ReplayComponent::Sae,
        ),
        (
            summary.ssm_digest,
            summary.evidence_ssm_digest,
            ReplayComponent::Ssm,
        ),
        (
            summary.lfm_digest,
            summary.evidence_lfm_digest,
            ReplayComponent::Lfm,
        ),
    ];
    for (raw, evidence, component) in checks {
        if raw.is_some() && evidence.is_none() {
            push_divergence(
                report,
                Divergence {
                    t,
                    component,
                    expected_digest: "present".to_string(),
                    observed_digest: "missing".to_string(),
                    hint: "evidence-chain link missing".to_string(),
                },
            );
        }
    }

    if matches!(strictness, ReplayStrictness::RecomputeStages)
        && summary.coherence.is_some()
        && summary.coherence_digest.is_none()
    {
        push_divergence(
            report,
            Divergence {
                t,
                component: ReplayComponent::Coherence,
                expected_digest: "present".to_string(),
                observed_digest: "missing".to_string(),
                hint: "coherence active but coherence_digest missing".to_string(),
            },
        );
    }
}

fn verify_backend_pack(records: &[ExperienceRecord], plan: &ReplayPlan, report: &mut ReplayReport) {
    let mut stable: Option<[u8; 32]> = None;
    for record in records.iter().filter(|r| {
        r.kind == ExperienceKind::BackendPack
            && r.time.tick.get() >= plan.t0
            && r.time.tick.get() <= plan.t1
    }) {
        let Some(pack) = &record.backend_pack_record else {
            report.counters.missing_records += 1;
            continue;
        };
        if let Some(expected) = plan.expected_backend_pack_digest {
            if pack.meta_digest != expected {
                push_divergence(
                    report,
                    Divergence {
                        t: record.time.tick.get(),
                        component: ReplayComponent::BackendPack,
                        expected_digest: digest_prefix(expected),
                        observed_digest: digest_prefix(pack.meta_digest),
                        hint: "expected_backend_pack_digest mismatch".to_string(),
                    },
                );
            }
        }
        if let Some(first) = stable {
            if first != pack.meta_digest {
                push_divergence(
                    report,
                    Divergence {
                        t: record.time.tick.get(),
                        component: ReplayComponent::BackendPack,
                        expected_digest: digest_prefix(first),
                        observed_digest: digest_prefix(pack.meta_digest),
                        hint: "backend pack drift inside replay range".to_string(),
                    },
                );
            }
        } else {
            stable = Some(pack.meta_digest);
        }
    }
}

fn verify_chain_ref(
    report: &mut ReplayReport,
    t: u64,
    component: ReplayComponent,
    observed: [u8; 32],
    expected: Option<[u8; 32]>,
) {
    let Some(expected) = expected else {
        report.counters.missing_records += 1;
        push_divergence(
            report,
            Divergence {
                t,
                component,
                expected_digest: "decision_chain".to_string(),
                observed_digest: digest_prefix(observed),
                hint: "associated DecisionFrame/compute_chain_digest missing".to_string(),
            },
        );
        return;
    };
    if observed != expected {
        push_divergence(
            report,
            Divergence {
                t,
                component,
                expected_digest: digest_prefix(expected),
                observed_digest: digest_prefix(observed),
                hint: "evidence_chain_digest mismatch".to_string(),
            },
        );
    }
}

fn verify_lfm_summary_digest(report: &mut ReplayReport, t: u64, summary: LfmSummaryRecord) {
    let expected = summary.compute_digest();
    if summary.digest != expected {
        push_divergence(
            report,
            Divergence {
                t,
                component: ReplayComponent::Lfm,
                expected_digest: digest_prefix(expected),
                observed_digest: digest_prefix(summary.digest),
                hint: "LfmSummaryRecord.digest invalid".to_string(),
            },
        );
    }
}

fn verify_issuance_decision(
    report: &mut ReplayReport,
    t: u64,
    issuance: &CapabilityIssuanceRecord,
    decision_by_id: &HashMap<u64, DecisionFrame>,
    records: &[ExperienceRecord],
    strictness: ReplayStrictness,
) {
    let decision = decision_by_id.get(&issuance.decision_id);
    let nsr_risk = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::Nsr && r.time.tick.get() <= t)
        .find_map(|r| r.nsr_record.as_ref())
        .map(|n| f32::from(n.nsr_risk_q) / 65535.0);
    let hormone_stress = records
        .iter()
        .filter(|r| r.kind == ExperienceKind::Hormone && r.time.tick.get() <= t)
        .find_map(|r| r.hormone_record)
        .map(|h| f32::from(h.stress_index_q) / 65535.0);

    let expected_signals_digest = governance_signals_digest(decision, t, nsr_risk, hormone_stress);
    if issuance.governance_signals_digest != expected_signals_digest {
        push_divergence(
            report,
            Divergence {
                t,
                component: ReplayComponent::Governor,
                expected_digest: digest_prefix(expected_signals_digest),
                observed_digest: digest_prefix(issuance.governance_signals_digest),
                hint: "governance_signals_digest mismatch".to_string(),
            },
        );
    }

    if matches!(strictness, ReplayStrictness::RecomputeStages) {
        let Some(summary) = decision.and_then(|d| d.compute_summary) else {
            report.counters.missing_records += 1;
            return;
        };
        let score = governor_score(
            nsr_risk.unwrap_or(summary.risk),
            summary.coherence,
            summary.instability,
            summary.lfm_uncertainty,
            hormone_stress,
        );
        let tier = issuance_tier(score);
        if issuance.tier != tier {
            push_divergence(
                report,
                Divergence {
                    t,
                    component: ReplayComponent::Issuance,
                    expected_digest: format!("tier:{tier}"),
                    observed_digest: format!("tier:{}", issuance.tier),
                    hint: "tier does not match recomputed governor score".to_string(),
                },
            );
        }

        let q = quantize_unit_u16(score);
        if issuance.governor_score_q != q {
            push_divergence(
                report,
                Divergence {
                    t,
                    component: ReplayComponent::Governor,
                    expected_digest: format!("score_q:{q}"),
                    observed_digest: format!("score_q:{}", issuance.governor_score_q),
                    hint: "governor_score_q mismatch".to_string(),
                },
            );
        }
    }
}

fn recompute_decision_chain(report: &mut ReplayReport, in_range: &[&ExperienceRecord]) {
    for rec in in_range {
        if rec.kind != ExperienceKind::DecisionOut {
            continue;
        }
        let decision = match &rec.payload {
            ExperiencePayload::Decision(d) => d,
            _ => continue,
        };
        let Some(persisted) = decision.compute_summary else {
            continue;
        };
        let control = in_range
            .iter()
            .rev()
            .find(|candidate| {
                candidate.kind == ExperienceKind::ControlIn
                    && candidate.corr == rec.corr
                    && candidate.time.tick.get() <= rec.time.tick.get()
            })
            .and_then(|candidate| match &candidate.payload {
                ExperiencePayload::Control(ctrl) => Some(ctrl.clone()),
                _ => None,
            });
        let Some(control) = control else {
            continue;
        };
        let backend_kind =
            ComputeBackendKind::parse(persisted.backend_profile.unwrap_or(persisted.backend))
                .unwrap_or(ComputeBackendKind::Stub);
        let seed = persisted.seed.unwrap_or(ComputeBudget::default().seed);
        let cfg = ComputeBackendConfig {
            kind: backend_kind,
            seed,
            ..ComputeBackendConfig::default()
        };
        let Ok(backend) = build_backend(&cfg) else {
            continue;
        };
        let Ok(recomputed) = backend.compute(
            &compute_input_from_control(&control),
            ComputeBudget {
                seed,
                ..ComputeBudget::default()
            },
        ) else {
            continue;
        };
        let recomputed = recomputed.summary(backend.name());
        if let Some(persisted_chain) = persisted.compute_chain_digest {
            if persisted_chain != recomputed.compute_chain_digest {
                push_divergence(
                    report,
                    Divergence {
                        t: rec.time.tick.get(),
                        component: ReplayComponent::Risk,
                        expected_digest: digest_prefix(recomputed.compute_chain_digest),
                        observed_digest: digest_prefix(persisted_chain),
                        hint: "recomputed compute_chain_digest mismatch".to_string(),
                    },
                );
            }
        }
    }
}

fn governance_signals_digest(
    decision: Option<&DecisionFrame>,
    t: u64,
    nsr_risk: Option<f32>,
    hormone_stress: Option<f32>,
) -> [u8; 32] {
    let summary = decision.and_then(|d| d.compute_summary);
    let risk = summary.map(|s| s.risk).unwrap_or(1.0).clamp(0.0, 1.0);
    let confidence = summary.map(|s| s.confidence).unwrap_or(0.0).clamp(0.0, 1.0);
    let coherence = summary.and_then(|s| s.coherence).map(|v| v.clamp(0.0, 1.0));
    let instability = summary
        .and_then(|s| s.instability)
        .map(|v| v.clamp(0.0, 1.0));
    let pressure = summary.map(|s| s.pressure).unwrap_or(1.0).clamp(0.0, 1.0);
    let surprise = summary.map(|s| s.surprise).unwrap_or(1.0).clamp(0.0, 1.0);
    let lfm_uncertainty = summary
        .and_then(|s| s.lfm_uncertainty)
        .map(|v| v.clamp(0.0, 1.0));
    let lfm_stability = summary
        .and_then(|s| s.lfm_stability)
        .map(|v| v.clamp(0.0, 1.0));

    let mut hasher = Sha256::new();
    hasher.update(t.to_le_bytes());
    hasher.update(risk.to_bits().to_le_bytes());
    hasher.update(confidence.to_bits().to_le_bytes());
    put_opt_f32(&mut hasher, nsr_risk.map(|v| v.clamp(0.0, 1.0)));
    put_opt_f32(&mut hasher, coherence);
    put_opt_f32(&mut hasher, instability);
    hasher.update(pressure.to_bits().to_le_bytes());
    hasher.update(surprise.to_bits().to_le_bytes());
    put_opt_f32(&mut hasher, lfm_uncertainty);
    put_opt_f32(&mut hasher, lfm_stability);
    put_opt_f32(&mut hasher, hormone_stress.map(|v| v.clamp(0.0, 1.0)));
    hasher.finalize().into()
}

fn put_opt_f32(hasher: &mut Sha256, value: Option<f32>) {
    if let Some(v) = value {
        hasher.update([1]);
        hasher.update(v.to_bits().to_le_bytes());
    } else {
        hasher.update([0]);
    }
}

fn governor_score(
    risk: f32,
    coherence: Option<f32>,
    instability: Option<f32>,
    lfm_uncertainty: Option<f32>,
    hormone_stress: Option<f32>,
) -> f32 {
    (0.35 * risk
        + 0.20 * (1.0 - coherence.unwrap_or(1.0))
        + 0.20 * instability.unwrap_or(0.0)
        + 0.15 * lfm_uncertainty.unwrap_or(0.0)
        + 0.10 * hormone_stress.unwrap_or(0.0))
    .clamp(0.0, 1.0)
}

fn issuance_tier(score: f32) -> u8 {
    if score < 0.25 {
        0
    } else if score < 0.5 {
        1
    } else if score < 0.75 {
        2
    } else {
        3
    }
}

fn quantize_unit_u16(value: f32) -> u16 {
    ((value.clamp(0.0, 1.0) * 65535.0).round()) as u16
}

fn push_divergence(report: &mut ReplayReport, divergence: Divergence) {
    if report.first_divergence.is_none() {
        report.first_divergence = Some(divergence.clone());
    }
    if report.details.len() < REPLAY_DIVERGENCE_CAP {
        report.details.push(divergence);
    }
    report.counters.mismatched_digests = report.counters.mismatched_digests.saturating_add(1);
}

fn finalize_status(report: &mut ReplayReport) {
    if report.counters.missing_records > 0 {
        report.overall_status = ReplayOverallStatus::MissingData;
    }
    if report.first_divergence.is_some() {
        report.overall_status = ReplayOverallStatus::DriftFound;
    }
}

fn digest_prefix(digest: [u8; 32]) -> String {
    let mut out = String::new();
    for byte in &digest[..6] {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}
