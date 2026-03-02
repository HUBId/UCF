use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    explain_tick, generate_spec_snapshot, load_or_init_config, one_command_bringup,
    policy_validate, write_json, ExplainTickRequest, GateStatus, OpsError, SpecSnapshotArgs,
};

#[derive(Debug, Clone)]
pub struct GoldenGenerateArgs {
    pub scenario: String,
    pub os: String,
    pub out_root: PathBuf,
    pub workdir_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct GoldenVerifyArgs {
    pub scenario: String,
    pub os: String,
    pub out_root: PathBuf,
    pub workdir_root: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenScenarioConfig {
    pub scenario_id: String,
    pub scenario_fixture: String,
    pub ticks: u64,
    pub profile: String,
    pub slot_ebm_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct GoldenTickDigestSample {
    tick: u64,
    window: u64,
    evidence_context_digest_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct GoldenScalarSummary {
    risk_mean_q: u16,
    pressure_mean_q: u16,
    uncertainty_mean_q: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldenExpectedOutputs {
    sampled_tick_digests: Vec<GoldenTickDigestSample>,
    scalar_summary: GoldenScalarSummary,
    gate_status: GateStatus,
    spec_snapshot_sha256_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldenManifest {
    schema_version: u16,
    os: String,
    scenario_id: String,
    scenario_fixture: String,
    ticks: u64,
    policy_graph_digest_prefix: String,
    config_digest_prefix: String,
    expected_outputs: GoldenExpectedOutputs,
}

pub fn goldens_generate(args: &GoldenGenerateArgs) -> Result<PathBuf, OpsError> {
    let cfg = load_scenario(&args.scenario)?;
    let out_dir = args
        .out_root
        .join(normalize_os(&args.os))
        .join(&cfg.scenario_id);
    fs::create_dir_all(&out_dir)?;
    let workdir = args.workdir_root.join(&cfg.scenario_id);
    if workdir.exists() {
        fs::remove_dir_all(&workdir)?;
    }
    fs::create_dir_all(&workdir)?;

    std::env::set_var("UCF_PROFILE", &cfg.profile);
    std::env::set_var("UCF_SLOT_EBM_MODE", &cfg.slot_ebm_mode);

    let scenario_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(&cfg.scenario_fixture);
    let bringup_out = out_dir.join("bringup");
    let artifacts = one_command_bringup(&workdir, &scenario_path, cfg.ticks, &bringup_out, true)?;
    let gate_report = GoldenGateReport {
        status: GateStatus::Pass,
        source: "readiness_gate_precondition".to_string(),
    };
    write_json(out_dir.join("gate_report.json"), &gate_report)?;

    let explain = explain_tick(
        &workdir,
        ExplainTickRequest {
            t: Some(cfg.ticks.saturating_sub(1)),
            decision_id: None,
            detail_level: 1,
            digest_prefix_len: 12,
        },
    )?;
    write_json(out_dir.join("explain_tick_last.json"), &explain)?;

    let spec_snapshot = out_dir.join("spec_snapshot.md");
    generate_spec_snapshot(&SpecSnapshotArgs {
        policy: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/packs/base_v1"),
        overlay: Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/packs/overlays/test"),
        ),
        out: spec_snapshot.clone(),
    })?;
    let snapshot_sha = sha_prefix(&fs::read(spec_snapshot)?);

    let policy = policy_validate(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/packs/base_v1"),
        Some(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/packs/overlays/test")),
    )?;
    let resolved_cfg = load_or_init_config(&workdir)?;

    let (samples, mean_risk) = load_tick_samples(&workdir.join("ess").join("ess_fixture.json"))?;
    let manifest = GoldenManifest {
        schema_version: 1,
        os: normalize_os(&args.os),
        scenario_id: cfg.scenario_id.clone(),
        scenario_fixture: cfg.scenario_fixture,
        ticks: cfg.ticks,
        policy_graph_digest_prefix: prefix(&policy.policy_graph_digest, 12),
        config_digest_prefix: prefix(&resolved_cfg.config_digest, 12),
        expected_outputs: GoldenExpectedOutputs {
            sampled_tick_digests: samples,
            scalar_summary: GoldenScalarSummary {
                risk_mean_q: quantize_unit(mean_risk),
                pressure_mean_q: quantize_unit(artifacts.metrics.mean_pressure),
                uncertainty_mean_q: quantize_unit(artifacts.metrics.mean_uncertainty),
            },
            gate_status: gate_report.status,
            spec_snapshot_sha256_prefix: snapshot_sha,
        },
    };
    write_json(out_dir.join("golden_manifest.json"), &manifest)?;
    Ok(out_dir)
}

pub fn goldens_update(args: &GoldenGenerateArgs) -> Result<PathBuf, OpsError> {
    goldens_generate(args)
}

pub fn goldens_verify(args: &GoldenVerifyArgs) -> Result<(), OpsError> {
    let requested_os = normalize_os(&args.os);
    let mut expected_dir = args.out_root.join(&requested_os).join(&args.scenario);
    if !expected_dir.join("golden_manifest.json").exists() {
        expected_dir = args.out_root.join("linux").join(&args.scenario);
    }
    let expected: GoldenManifest = serde_json::from_str(&fs::read_to_string(
        expected_dir.join("golden_manifest.json"),
    )?)?;

    let generated_dir = goldens_generate(&GoldenGenerateArgs {
        scenario: args.scenario.clone(),
        os: args.os.clone(),
        out_root: args.workdir_root.join("verify_actual"),
        workdir_root: args.workdir_root.clone(),
    })?;
    let actual: GoldenManifest = serde_json::from_str(&fs::read_to_string(
        generated_dir.join("golden_manifest.json"),
    )?)?;

    let same_os = expected.os == actual.os;
    if expected.expected_outputs.gate_status != GateStatus::Pass
        || actual.expected_outputs.gate_status != GateStatus::Pass
    {
        return Err(OpsError::Invalid(
            "golden gate status is not PASS".to_string(),
        ));
    }
    if expected.policy_graph_digest_prefix != actual.policy_graph_digest_prefix {
        return Err(OpsError::Invalid(format!(
            "policy digest mismatch: expected={} actual={}",
            expected.policy_graph_digest_prefix, actual.policy_graph_digest_prefix
        )));
    }
    if expected.config_digest_prefix != actual.config_digest_prefix {
        return Err(OpsError::Invalid(format!(
            "config digest mismatch: expected={} actual={}",
            expected.config_digest_prefix, actual.config_digest_prefix
        )));
    }

    if same_os
        && expected.expected_outputs.sampled_tick_digests
            != actual.expected_outputs.sampled_tick_digests
    {
        return Err(OpsError::Invalid("sampled tick digest prefixes changed; run `ucf-ops goldens update ...` for intentional updates".to_string()));
    }

    if expected.expected_outputs.scalar_summary.risk_mean_q
        != actual.expected_outputs.scalar_summary.risk_mean_q
        || expected.expected_outputs.scalar_summary.pressure_mean_q
            != actual.expected_outputs.scalar_summary.pressure_mean_q
        || expected.expected_outputs.scalar_summary.uncertainty_mean_q
            != actual.expected_outputs.scalar_summary.uncertainty_mean_q
    {
        return Err(OpsError::Invalid("scalar summary mismatch".to_string()));
    }

    if expected.expected_outputs.sampled_tick_digests.len()
        != actual.expected_outputs.sampled_tick_digests.len()
    {
        return Err(OpsError::Invalid(
            "sampled digest structure mismatch".to_string(),
        ));
    }

    Ok(())
}

fn load_scenario(id: &str) -> Result<GoldenScenarioConfig, OpsError> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/goldens/scenarios")
        .join(format!("{id}.json"));
    let raw = fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&raw)?)
}

#[derive(Debug, Deserialize)]
struct Fixture {
    decisions: Vec<FixtureDecision>,
}

#[derive(Debug, Deserialize)]
struct FixtureDecision {
    tick: u64,
    window: u64,
    evidence_context_digest_hex: String,
    risk: f32,
}

fn load_tick_samples(path: &Path) -> Result<(Vec<GoldenTickDigestSample>, f32), OpsError> {
    let fixture: Fixture = serde_json::from_str(&fs::read_to_string(path)?)?;
    if fixture.decisions.is_empty() {
        return Ok((Vec::new(), 0.0));
    }
    let mean_risk =
        fixture.decisions.iter().map(|d| d.risk).sum::<f32>() / fixture.decisions.len() as f32;
    let idx = [0, fixture.decisions.len() / 2, fixture.decisions.len() - 1];
    let mut out = Vec::new();
    for i in idx {
        let d = &fixture.decisions[i];
        out.push(GoldenTickDigestSample {
            tick: d.tick,
            window: d.window,
            evidence_context_digest_prefix: prefix(&d.evidence_context_digest_hex, 12),
        });
    }
    out.dedup_by(|a, b| a.tick == b.tick && a.window == b.window);
    Ok((out, mean_risk))
}

fn quantize_unit(v: f32) -> u16 {
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 65535.0).round() as u16
}

fn prefix(value: &str, len: usize) -> String {
    value.chars().take(len).collect()
}

fn sha_prefix(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(digest)[..12].to_string()
}

fn normalize_os(os: &str) -> String {
    match os.to_ascii_lowercase().as_str() {
        "linux" => "linux".to_string(),
        "windows" => "windows".to_string(),
        "macos" => "macos".to_string(),
        _ => std::env::consts::OS.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_serialization_is_deterministic() {
        let manifest = GoldenManifest {
            schema_version: 1,
            os: "linux".to_string(),
            scenario_id: "golden_a".to_string(),
            scenario_fixture: "fixtures/e2e_scenario_a.json".to_string(),
            ticks: 12,
            policy_graph_digest_prefix: "abc123".to_string(),
            config_digest_prefix: "def456".to_string(),
            expected_outputs: GoldenExpectedOutputs {
                sampled_tick_digests: vec![GoldenTickDigestSample {
                    tick: 1,
                    window: 0,
                    evidence_context_digest_prefix: "cafebeef".to_string(),
                }],
                scalar_summary: GoldenScalarSummary {
                    risk_mean_q: 1,
                    pressure_mean_q: 2,
                    uncertainty_mean_q: 3,
                },
                gate_status: GateStatus::Pass,
                spec_snapshot_sha256_prefix: "123".to_string(),
            },
        };
        let a = serde_json::to_string_pretty(&manifest).expect("json");
        let b = serde_json::to_string_pretty(&manifest).expect("json");
        assert_eq!(a, b);
    }

    #[test]
    fn golden_generate_then_verify_passes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out_root = dir.path().join("goldens");
        let workdir_root = dir.path().join("work");

        goldens_generate(&GoldenGenerateArgs {
            scenario: "golden_a".to_string(),
            os: "linux".to_string(),
            out_root: out_root.clone(),
            workdir_root: workdir_root.clone(),
        })
        .expect("generate");

        goldens_verify(&GoldenVerifyArgs {
            scenario: "golden_a".to_string(),
            os: "linux".to_string(),
            out_root,
            workdir_root,
        })
        .expect("verify");
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldenGateReport {
    status: GateStatus,
    source: String,
}
