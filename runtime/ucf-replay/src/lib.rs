#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use hex::FromHex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use ucf_compute::ComputeSignalsSummary as RecomputedComputeSummary;
use ucf_compute::{
    build_backend, compute_input_from_control, stable_budget_profile_id, ComputeBackendConfig,
    ComputeBackendKind, ComputeBudget,
};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{ExperienceKind, ExperiencePayload, ExperienceRecord};
use ucf_frames::v1::{
    ChannelCode, ComputeSignalsSummary, ControlFrame, CorrelationId, DecisionFrame, Intent,
    IntentId, IntentKind,
};

const REPORT_CAP: usize = 1000;
static UCF_COMPUTE_CHAIN_MISMATCH_TOTAL: AtomicU64 = AtomicU64::new(0);

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
            backend_profile: Some(leak_str(backend_name)),
            budget_profile_id: Some(entry.budget_profile_id),
            seed: Some(entry.seed),
            risk_contract_version: Some(1),
            compute_schema_version: Some(1),
            compute_chain_digest: None,
            compute_code_version: None,
            budget_exceeded_stage: None,
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
