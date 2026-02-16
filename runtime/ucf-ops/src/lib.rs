#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use ucf_compute::{
    build_backend, compute_input_from_control, stable_budget_profile_id, ComputeBackendConfig,
    ComputeBackendKind,
};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{ExperienceKind, ExperiencePayload};
use ucf_frames::v1::{
    ChannelCode, ControlFrame, ControlPayload, CorrelationId, Intent, IntentId, IntentKind,
};
use ucf_policy::adapter::MockAdapter;
use ucf_replay::{
    load_fixture_records, replay_audit as run_replay_audit, replay_records, write_report,
    ReplayMode, ReplayPlan, ReplaySpec, ReplayStrictness,
};
use ucf_runtime::RuntimeOrchestrator;

#[derive(Debug, Error)]
pub enum OpsError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("runtime error: {0}")]
    Runtime(#[from] ucf_runtime::errors::RuntimeError),
    #[error("compute error: {0}")]
    Compute(#[from] ucf_compute::ComputeError),
    #[error("bugreport invalid: {0}")]
    Invalid(String),
    #[error("replay error: {0}")]
    Replay(#[from] ucf_replay::ReplayError),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpsConfig {
    pub compute_backend: ComputeBackendKind,
    pub compute_seed: u64,
    pub compute_budget_profile: String,
    pub isolation_runtime: String,
    pub capabilities_default: String,
}

impl Default for OpsConfig {
    fn default() -> Self {
        Self {
            compute_backend: ComputeBackendKind::Stub,
            compute_seed: 0xDEC0DED,
            compute_budget_profile: "default".to_string(),
            isolation_runtime: "inproc".to_string(),
            capabilities_default: "deny".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BringupResult {
    pub workdir: PathBuf,
    pub ess_fixture_path: PathBuf,
    pub log_path: PathBuf,
    pub decision_count: usize,
    pub ess_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagCheck {
    pub name: String,
    pub pass: bool,
    pub detail: String,
    pub remediation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagReport {
    pub checks: Vec<DiagCheck>,
}

impl DiagReport {
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|c| c.pass)
    }
}

#[derive(Debug, Clone)]
pub struct ExportArgs {
    pub last: Option<usize>,
    pub include_sandbox: bool,
    pub include_audit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpsFixture {
    decisions: Vec<OpsFixtureDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpsFixtureDecision {
    decision_id: u64,
    corr: u64,
    tick: u64,
    window: u64,
    text: String,
    backend: String,
    risk: f32,
    confidence: f32,
    surprise: f32,
    pressure: f32,
    spike_count: u16,
    spikes_digest_hex: String,
    evidence_context_digest_hex: String,
    budget_profile_id: u32,
    seed: u64,
    risk_quality: u8,
}

pub fn bringup(workdir: &Path, demo: bool, ticks: u64) -> Result<BringupResult, OpsError> {
    ensure_layout(workdir)?;
    let cfg = load_or_init_config(workdir)?;

    std::env::set_var("UCF_COMPUTE_BACKEND", cfg.compute_backend.as_env_str());
    std::env::set_var("UCF_COMPUTE_SEED", cfg.compute_seed.to_string());
    std::env::set_var("UCF_COMPUTE_BUDGET_PROFILE", &cfg.compute_budget_profile);

    let mut orchestrator = RuntimeOrchestrator::try_new_from_env()?;
    let mut adapter = MockAdapter::default();
    let mut fixture_decisions = Vec::new();

    let max_ticks = if demo { ticks } else { ticks.max(10) };
    for step in 0..max_ticks {
        let time = SimTime {
            tick: Tick::new(step + 1),
            window: WindowId::new(0),
        };
        let corr = CorrelationId(step + 1);
        let ctrl = ControlFrame::new_text(
            time,
            corr,
            ChannelCode::ExternalOutput,
            Intent::new(IntentId(corr.0), IntentKind::Speak, "ops-demo"),
            format!("demo_text_{step}"),
        );

        let decision = orchestrator.ingest_and_process(&mut adapter, ctrl.clone())?;
        if let Some(summary) = decision.compute_summary {
            fixture_decisions.push(OpsFixtureDecision {
                decision_id: step + 1,
                corr: corr.0,
                tick: time.tick.get(),
                window: time.window.get(),
                text: extract_text(&ctrl),
                backend: summary.backend.to_string(),
                risk: summary.risk,
                confidence: summary.confidence,
                surprise: summary.surprise,
                pressure: summary.pressure,
                spike_count: summary.spike_count,
                spikes_digest_hex: hex::encode(summary.spikes_digest),
                evidence_context_digest_hex: summary
                    .evidence_context_digest
                    .map(hex::encode)
                    .unwrap_or_else(|| hex::encode([0u8; 32])),
                budget_profile_id: summary
                    .budget_profile_id
                    .unwrap_or(stable_budget_profile_id(1_000, 5_000)),
                seed: summary.seed.unwrap_or(cfg.compute_seed),
                risk_quality: summary.risk_quality.unwrap_or(2),
            });
        }
    }

    let fixture = OpsFixture {
        decisions: fixture_decisions,
    };

    let ess_fixture_path = workdir.join("ess").join("ess_fixture.json");
    let fixture_text = serde_json::to_string_pretty(&fixture)?;
    fs::write(&ess_fixture_path, fixture_text.as_bytes())?;

    let ess_digest = sha256_hex(fixture_text.as_bytes());
    let log_path = workdir.join("logs").join("bringup.log");
    let log_line = format!(
        "status=ok mode={} ticks={} ess={} digest={}\n",
        if demo { "demo" } else { "continuous" },
        max_ticks,
        ess_fixture_path.display(),
        ess_digest
    );
    fs::write(&log_path, log_line)?;

    Ok(BringupResult {
        workdir: workdir.to_path_buf(),
        ess_fixture_path,
        log_path,
        decision_count: fixture.decisions.len(),
        ess_digest,
    })
}

pub fn diagnostics(workdir: &Path) -> Result<DiagReport, OpsError> {
    let mut checks = Vec::new();
    let cfg = load_or_init_config(workdir)?;

    checks.push(DiagCheck {
        name: "workspace_build_tag".to_string(),
        pass: !build_tag()?.git_commit.is_empty(),
        detail: format!("commit={}", build_tag()?.git_commit),
        remediation: "run inside a git worktree.".to_string(),
    });

    checks.push(DiagCheck {
        name: "config_resolved".to_string(),
        pass: cfg.capabilities_default == "deny",
        detail: format!(
            "backend={:?} seed={} budget={} isolation={} caps_default={}",
            cfg.compute_backend,
            cfg.compute_seed,
            cfg.compute_budget_profile,
            cfg.isolation_runtime,
            cfg.capabilities_default
        ),
        remediation: "set capabilities_default to deny for safe operation.".to_string(),
    });

    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let ess_records = load_fixture_records(&fixture_path)?;
    checks.push(DiagCheck {
        name: "ess_health".to_string(),
        pass: !ess_records.is_empty(),
        detail: format!("records={}", ess_records.len()),
        remediation: "run `ucf-ops bringup --demo` to seed ESS fixture.".to_string(),
    });

    let audit_ok = ess_records
        .iter()
        .filter(|r| r.kind == ExperienceKind::AuditCheckpoint)
        .all(|r| r.audit_digest.is_some());
    checks.push(DiagCheck {
        name: "audit_chain".to_string(),
        pass: audit_ok,
        detail: "audit checkpoints parsed (or none present)".to_string(),
        remediation: "run workload including tool gate operations to emit checkpoints.".to_string(),
    });

    let compute_ok = run_compute_probe(&cfg)?;
    checks.push(compute_ok);

    checks.push(DiagCheck {
        name: "sandbox_runtime".to_string(),
        pass: cfg.isolation_runtime == "inproc",
        detail: format!("runtime={}", cfg.isolation_runtime),
        remediation: "set isolation_runtime to inproc for offline diagnostics.".to_string(),
    });

    let log_exists = workdir.join("logs").join("bringup.log").exists();
    checks.push(DiagCheck {
        name: "metrics_tracing".to_string(),
        pass: log_exists,
        detail: "bringup.log present".to_string(),
        remediation: "run `ucf-ops bringup --demo` to initialize logging.".to_string(),
    });

    Ok(DiagReport { checks })
}

pub fn export_bugreport(workdir: &Path, args: &ExportArgs) -> Result<PathBuf, OpsError> {
    ensure_layout(workdir)?;
    let fixture_path = workdir.join("ess").join("ess_fixture.json");
    let fixture: OpsFixture = serde_json::from_str(&fs::read_to_string(&fixture_path)?)?;

    let selected = if let Some(last) = args.last {
        let len = fixture.decisions.len();
        fixture.decisions[len.saturating_sub(last)..].to_vec()
    } else {
        fixture.decisions.clone()
    };

    let timestamp = selected.last().map(|d| d.tick).unwrap_or(0);
    let out_dir = workdir
        .join("reports")
        .join(format!("bugreport_{timestamp:010}"));
    fs::create_dir_all(&out_dir)?;

    let config = load_or_init_config(workdir)?;
    write_json(out_dir.join("config_resolved.json"), &config)?;
    write_json(out_dir.join("build_tag.json"), &build_tag()?)?;
    write_json(
        out_dir.join("ess_slice.json"),
        &OpsFixture {
            decisions: selected.clone(),
        },
    )?;

    let mut indices = BTreeMap::<String, serde_json::Value>::new();
    indices.insert("count".to_string(), serde_json::json!(selected.len()));
    indices.insert(
        "range".to_string(),
        serde_json::json!({
            "from_tick": selected.first().map(|d| d.tick),
            "to_tick": selected.last().map(|d| d.tick),
            "include_sandbox": args.include_sandbox,
            "include_audit": args.include_audit,
        }),
    );
    write_json(out_dir.join("indices.json"), &indices)?;

    fs::write(
        out_dir.join("README.txt"),
        "Replay with:\nucf-ops replay-bugreport <path> --mode compute\n",
    )?;

    let checksums = build_checksums(&out_dir)?;
    write_json(out_dir.join("checksums.json"), &checksums)?;

    Ok(out_dir)
}

pub fn verify_bugreport(path: &Path) -> Result<(), OpsError> {
    let checksum_path = path.join("checksums.json");
    let checksums: ChecksumManifest = serde_json::from_str(&fs::read_to_string(checksum_path)?)?;

    for (file, expected) in &checksums.files {
        let data = fs::read(path.join(file))?;
        let got = sha256_hex(&data);
        if &got != expected {
            return Err(OpsError::Invalid(format!("checksum mismatch for {file}")));
        }
    }

    let fixture: OpsFixture =
        serde_json::from_str(&fs::read_to_string(path.join("ess_slice.json"))?)?;

    if fixture.decisions.len() > 10_000 {
        return Err(OpsError::Invalid("ess slice too large".to_string()));
    }

    Ok(())
}

pub fn replay_audit(
    workdir: &Path,
    from_tick: u64,
    to_tick: u64,
    strictness: ReplayStrictness,
    stop_on_first_divergence: bool,
    report_path: &Path,
) -> Result<(), OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let plan = ReplayPlan {
        t0: from_tick,
        t1: to_tick,
        expected_backend_pack_digest: None,
        strictness,
        stop_on_first_divergence,
    };
    let report = run_replay_audit(&records, &plan);
    let body = serde_json::to_string_pretty(&report)?;
    fs::write(report_path, body)?;
    Ok(())
}

pub fn replay_bugreport(path: &Path, mode: ReplayMode) -> Result<PathBuf, OpsError> {
    verify_bugreport(path)?;
    let records = load_fixture_records(&path.join("ess_slice.json"))?;
    let spec = ReplaySpec {
        from_tick: 0,
        to_tick: u64::MAX,
        backend_override: None,
        seed_override: None,
        budget_override: None,
        mode,
    };
    let result = replay_records(&records, &spec);
    let report_path = path.join("replay_report.json");
    write_report(&report_path, &result)?;
    Ok(report_path)
}

pub fn metrics_snapshot(workdir: &Path) -> Result<serde_json::Value, OpsError> {
    let records = load_fixture_records(&workdir.join("ess").join("ess_fixture.json"))?;
    let mut risk_buckets = [0u64; 3];
    for record in &records {
        if let ExperiencePayload::Decision(decision) = &record.payload {
            if let Some(summary) = decision.compute_summary {
                let idx = if summary.risk < 0.33 {
                    0
                } else if summary.risk < 0.66 {
                    1
                } else {
                    2
                };
                risk_buckets[idx] += 1;
            }
        }
    }

    Ok(serde_json::json!({
        "compute": {
            "risk_distribution": risk_buckets,
            "budget_exceeded_total": 0
        },
        "sandbox": {
            "denied_total": 0,
            "rate_limited_total": 0
        },
        "audit": {
            "checkpoint_total": records.iter().filter(|r| r.kind == ExperienceKind::AuditCheckpoint).count()
        },
        "ess": {
            "records": records.len()
        }
    }))
}

pub fn load_or_init_config(workdir: &Path) -> Result<OpsConfig, OpsError> {
    let path = workdir.join("config_resolved.json");
    if !path.exists() {
        let cfg = OpsConfig::default();
        write_json(&path, &cfg)?;
        return Ok(cfg);
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn run_compute_probe(cfg: &OpsConfig) -> Result<DiagCheck, OpsError> {
    let backend_cfg = ComputeBackendConfig {
        kind: cfg.compute_backend,
        seed: cfg.compute_seed,
        ..ComputeBackendConfig::default()
    };
    let budget = backend_cfg.to_budget();
    let backend = build_backend(&backend_cfg)?;
    let ctrl = ControlFrame::new_text(
        SimTime {
            tick: Tick::new(1),
            window: WindowId::new(0),
        },
        CorrelationId(777),
        ChannelCode::ExternalOutput,
        Intent::new(IntentId(777), IntentKind::Speak, "diag"),
        "compute_probe",
    );
    let input = compute_input_from_control(&ctrl);
    let out = backend.compute(&input, budget)?;
    let pass = (0.0..=1.0).contains(&out.risk) && (0.0..=1.0).contains(&out.confidence);

    Ok(DiagCheck {
        name: "compute_probe".to_string(),
        pass,
        detail: format!("risk={:.3} confidence={:.3}", out.risk, out.confidence),
        remediation: "ensure compute backend feature flags and seed are set.".to_string(),
    })
}

fn ensure_layout(workdir: &Path) -> Result<(), OpsError> {
    for dir in ["ess", "logs", "reports", "fixtures"] {
        fs::create_dir_all(workdir.join(dir))?;
    }
    Ok(())
}

fn extract_text(ctrl: &ControlFrame) -> String {
    match &ctrl.payload {
        ControlPayload::Text(text) => text.to_string(),
        _ => "demo".to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildTag {
    git_commit: String,
    package_version: String,
}

fn build_tag() -> Result<BuildTag, OpsError> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()?;
    let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(BuildTag {
        git_commit: commit,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChecksumManifest {
    files: BTreeMap<String, String>,
    bundle_digest: String,
}

fn build_checksums(dir: &Path) -> Result<ChecksumManifest, OpsError> {
    let mut files = BTreeMap::new();
    for name in [
        "build_tag.json",
        "config_resolved.json",
        "ess_slice.json",
        "indices.json",
        "README.txt",
    ] {
        let data = fs::read(dir.join(name))?;
        files.insert(name.to_string(), sha256_hex(&data));
    }

    let mut bundle_hasher = Sha256::new();
    for (name, digest) in &files {
        bundle_hasher.update(name.as_bytes());
        bundle_hasher.update(digest.as_bytes());
    }

    Ok(ChecksumManifest {
        files,
        bundle_digest: hex::encode(bundle_hasher.finalize()),
    })
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<(), OpsError> {
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn export_and_verify_roundtrip() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 20).expect("bringup");
        let report_dir = export_bugreport(
            dir.path(),
            &ExportArgs {
                last: Some(5),
                include_sandbox: false,
                include_audit: false,
            },
        )
        .expect("export");

        verify_bugreport(&report_dir).expect("verify");
        assert!(report_dir.join("checksums.json").exists());
    }

    #[test]
    fn verify_catches_tampering() {
        let dir = tempdir().expect("tempdir");
        bringup(dir.path(), true, 8).expect("bringup");
        let report_dir = export_bugreport(
            dir.path(),
            &ExportArgs {
                last: Some(4),
                include_sandbox: false,
                include_audit: false,
            },
        )
        .expect("export");

        fs::write(report_dir.join("README.txt"), "tampered").expect("tamper");
        let err = verify_bugreport(&report_dir).expect_err("must fail");
        assert!(format!("{err}").contains("checksum mismatch"));
    }
}
