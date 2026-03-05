use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{sha256_hex, write_json, OpsError};

const SOAK_SCHEMA_VERSION: u16 = 1;
const MAX_SERIES_POINTS: usize = 256;

#[derive(Debug, Clone)]
pub struct SoakRunArgs {
    pub duration_secs: u64,
    pub scenario: String,
    pub out: PathBuf,
    pub health_poll_secs: u64,
    pub memory_sample_secs: u64,
    pub inject: Vec<InjectTrigger>,
    pub postmortem: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SoakStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InjectKind {
    Timeout,
    Drift,
    GatewayAuthFails,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InjectTrigger {
    pub kind: InjectKind,
    pub target: Option<String>,
    pub at_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakSamplePoint {
    pub t_sec: u64,
    pub drift_alarms: u64,
    pub fallbacks: u64,
    pub gateway_abuse: u64,
    pub emergency_active_ticks: u64,
    pub last_tick_age_ms: u64,
    pub health_status: String,
    pub rss_mb: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakSentinelReport {
    pub slope_mb_per_hour: f64,
    pub sustained_growth_windows: u32,
    pub threshold_mb_per_hour: f64,
    pub sustained_window_threshold: u32,
    pub leak_suspected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakReport {
    pub schema_version: u16,
    pub run_id: String,
    pub scenario: String,
    pub scenario_fixture: String,
    pub duration_secs: u64,
    pub monitoring: BTreeMap<String, u64>,
    pub counters: BTreeMap<String, u64>,
    pub status: SoakStatus,
    pub failure_reasons: Vec<String>,
    pub leak_sentinel: LeakSentinelReport,
    pub sampled_points: Vec<SoakSamplePoint>,
    pub postmortem_bundle: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostmortemManifest {
    schema_version: u16,
    run_id: String,
    generated_at_unix: u64,
    entries: Vec<PostmortemEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostmortemEntry {
    file: String,
    sha256: String,
}

pub fn parse_duration_secs(input: &str) -> Result<u64, OpsError> {
    let trimmed = input.trim();
    if let Some(hours) = trimmed.strip_suffix('h') {
        let h = hours
            .parse::<u64>()
            .map_err(|e| OpsError::Invalid(format!("invalid duration hours: {e}")))?;
        return Ok(h.saturating_mul(3600));
    }
    if let Some(mins) = trimmed.strip_suffix('m') {
        let m = mins
            .parse::<u64>()
            .map_err(|e| OpsError::Invalid(format!("invalid duration minutes: {e}")))?;
        return Ok(m.saturating_mul(60));
    }
    if let Some(secs) = trimmed.strip_suffix('s') {
        let s = secs
            .parse::<u64>()
            .map_err(|e| OpsError::Invalid(format!("invalid duration seconds: {e}")))?;
        return Ok(s);
    }
    trimmed
        .parse::<u64>()
        .map_err(|e| OpsError::Invalid(format!("invalid duration seconds: {e}")))
}

pub fn parse_inject(value: &str) -> Result<InjectTrigger, OpsError> {
    let (head, time_part) = value
        .split_once("@t=")
        .ok_or_else(|| OpsError::Invalid(format!("invalid inject format: {value}")))?;
    let at_sec = time_part
        .parse::<u64>()
        .map_err(|e| OpsError::Invalid(format!("invalid inject time: {e}")))?;
    let (kind, target) = if let Some((k, t)) = head.split_once(':') {
        (k, Some(t.to_string()))
    } else {
        (head, None)
    };
    let kind = match kind {
        "timeout" => InjectKind::Timeout,
        "drift" => InjectKind::Drift,
        "gateway_auth_fails" => InjectKind::GatewayAuthFails,
        other => {
            return Err(OpsError::Invalid(format!(
                "unsupported inject kind: {other}"
            )))
        }
    };
    Ok(InjectTrigger {
        kind,
        target,
        at_sec,
    })
}

pub fn soak_run(workdir: &Path, args: &SoakRunArgs) -> Result<SoakReport, OpsError> {
    std::env::set_var("UCF_OFFLINE", "1");
    let run_id = format!("soak-{}", now_unix_secs());
    fs::create_dir_all(&args.out)?;
    let scenario_fixture = resolve_scenario_fixture(&args.scenario)?;

    let health_poll = args.health_poll_secs.max(1);
    let memory_every = args.memory_sample_secs.max(1);
    let mut counters: BTreeMap<String, u64> = BTreeMap::from([
        ("drift_alarms".to_string(), 0),
        ("fallbacks".to_string(), 0),
        ("gateway_abuse".to_string(), 0),
        ("emergency_active_ticks".to_string(), 0),
        ("timeouts".to_string(), 0),
    ]);
    let mut points = Vec::new();
    let mut rss_series = Vec::new();
    let mut reasons = Vec::new();

    for t in 0..=args.duration_secs {
        for trigger in args.inject.iter().filter(|x| x.at_sec == t) {
            match trigger.kind {
                InjectKind::Timeout => {
                    *counters.entry("timeouts".to_string()).or_default() += 1;
                    *counters.entry("fallbacks".to_string()).or_default() += 1;
                    *counters
                        .entry("emergency_active_ticks".to_string())
                        .or_default() += 1;
                    reasons.push(format!("injected timeout at t={t}"));
                }
                InjectKind::Drift => {
                    *counters.entry("drift_alarms".to_string()).or_default() += 1;
                    reasons.push(format!("injected drift at t={t}"));
                }
                InjectKind::GatewayAuthFails => {
                    *counters.entry("gateway_abuse".to_string()).or_default() += 1;
                    reasons.push(format!("injected gateway_auth_fails at t={t}"));
                }
            }
        }

        if t % health_poll == 0 {
            let emergency = counters.get("emergency_active_ticks").copied().unwrap_or(0) > 0;
            let drift = counters.get("drift_alarms").copied().unwrap_or(0);
            let abuse = counters.get("gateway_abuse").copied().unwrap_or(0);
            let health = if emergency {
                "FAIL"
            } else if drift > 0 || abuse > 0 {
                "DEGRADED"
            } else {
                "OK"
            };
            let mut rss = None;
            if t % memory_every == 0 {
                rss = read_rss_mb();
                if let Some(v) = rss {
                    rss_series.push((t, v));
                }
            }
            points.push(SoakSamplePoint {
                t_sec: t,
                drift_alarms: counters.get("drift_alarms").copied().unwrap_or(0),
                fallbacks: counters.get("fallbacks").copied().unwrap_or(0),
                gateway_abuse: counters.get("gateway_abuse").copied().unwrap_or(0),
                emergency_active_ticks: counters
                    .get("emergency_active_ticks")
                    .copied()
                    .unwrap_or(0),
                last_tick_age_ms: 0,
                health_status: health.to_string(),
                rss_mb: rss,
            });
        }
    }

    let leak = detect_leak(&rss_series, 24.0, 6);
    if leak.leak_suspected {
        reasons.push("rss leak sentinel triggered".to_string());
    }
    let mut status = if reasons.is_empty() {
        SoakStatus::Pass
    } else {
        SoakStatus::Fail
    };
    if args.inject.is_empty() && leak.leak_suspected {
        status = SoakStatus::Fail;
    }

    let mut report = SoakReport {
        schema_version: SOAK_SCHEMA_VERSION,
        run_id: run_id.clone(),
        scenario: args.scenario.clone(),
        scenario_fixture,
        duration_secs: args.duration_secs,
        monitoring: BTreeMap::from([
            ("health_poll_secs".to_string(), health_poll),
            ("memory_sample_secs".to_string(), memory_every),
            ("max_points".to_string(), MAX_SERIES_POINTS as u64),
        ]),
        counters,
        status,
        failure_reasons: reasons,
        leak_sentinel: leak,
        sampled_points: downsample_points(points, MAX_SERIES_POINTS),
        postmortem_bundle: None,
    };

    write_json(
        args.out.join("soak_timeseries.json"),
        &report.sampled_points,
    )?;

    if matches!(report.status, SoakStatus::Fail) || args.postmortem {
        let bundle = create_postmortem_bundle(workdir, &args.out, &report)?;
        report.postmortem_bundle = Some(bundle.display().to_string());
    }

    write_json(args.out.join("soak_report.json"), &report)?;
    Ok(report)
}

fn resolve_scenario_fixture(scenario: &str) -> Result<String, OpsError> {
    let direct = PathBuf::from(scenario);
    if direct.exists() {
        return Ok(direct.display().to_string());
    }

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let scenario_candidates = [
        PathBuf::from("fixtures/goldens/scenarios").join(format!("{scenario}.json")),
        workspace_root
            .join("fixtures/goldens/scenarios")
            .join(format!("{scenario}.json")),
    ];

    let scenario_path = scenario_candidates
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| OpsError::Invalid(format!("unknown scenario: {scenario}")))?;

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&scenario_path)?)?;
    let fixture = value
        .get("scenario_fixture")
        .and_then(|v| v.as_str())
        .unwrap_or("fixtures/e2e_scenario_a.json");

    let fixture_path = PathBuf::from(fixture);
    if fixture_path.exists() {
        return Ok(fixture_path.display().to_string());
    }
    let workspace_fixture = workspace_root.join(fixture);
    if workspace_fixture.exists() {
        return Ok(workspace_fixture.display().to_string());
    }

    Ok(fixture.to_string())
}

fn create_postmortem_bundle(
    _workdir: &Path,
    out_dir: &Path,
    report: &SoakReport,
) -> Result<PathBuf, OpsError> {
    let stamp = now_unix_secs();
    let bundle_path = out_dir.join(format!("postmortem_{stamp}.zip"));
    let diag = serde_json::json!({"redaction_safe": true, "run_id": report.run_id, "failure_reasons": report.failure_reasons});
    let repro = serde_json::json!({"redaction_safe": true, "scenario": report.scenario, "scenario_fixture": report.scenario_fixture});
    let alerts = serde_json::json!({"drift_alarms": report.counters.get("drift_alarms").copied().unwrap_or(0), "gateway_abuse": report.counters.get("gateway_abuse").copied().unwrap_or(0)});
    let drift = serde_json::json!({"drift_alarms": report.counters.get("drift_alarms").copied().unwrap_or(0)});
    let health = report.sampled_points.last().cloned();

    let mut files = BTreeMap::new();
    files.insert(
        "diagnostics_bundle.json".to_string(),
        serde_json::to_vec_pretty(&diag)?,
    );
    files.insert(
        "repro_pack.json".to_string(),
        serde_json::to_vec_pretty(&repro)?,
    );
    files.insert(
        "alerts_report.json".to_string(),
        serde_json::to_vec_pretty(&alerts)?,
    );
    files.insert(
        "drift_report.json".to_string(),
        serde_json::to_vec_pretty(&drift)?,
    );
    files.insert(
        "health_snapshot.json".to_string(),
        serde_json::to_vec_pretty(&health)?,
    );

    let mut manifest = PostmortemManifest {
        schema_version: 1,
        run_id: report.run_id.clone(),
        generated_at_unix: stamp,
        entries: Vec::new(),
    };
    for (name, data) in &files {
        manifest.entries.push(PostmortemEntry {
            file: name.clone(),
            sha256: sha256_hex(data),
        });
    }
    files.insert(
        "manifest.json".to_string(),
        serde_json::to_vec_pretty(&manifest).map_err(OpsError::from)?,
    );

    let file = fs::File::create(&bundle_path)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, data) in files {
        zip.start_file(name, opts)
            .map_err(|e| OpsError::Invalid(format!("zip start failed: {e}")))?;
        zip.write_all(&data)
            .map_err(|e| OpsError::Invalid(format!("zip write failed: {e}")))?;
    }
    zip.finish()
        .map_err(|e| OpsError::Invalid(format!("zip finalize failed: {e}")))?;
    Ok(bundle_path)
}

fn downsample_points(mut points: Vec<SoakSamplePoint>, max_points: usize) -> Vec<SoakSamplePoint> {
    if points.len() <= max_points {
        return points;
    }
    let step = (points.len() as f64 / max_points as f64).ceil() as usize;
    points = points
        .into_iter()
        .enumerate()
        .filter_map(|(idx, p)| if idx % step == 0 { Some(p) } else { None })
        .collect::<Vec<_>>();
    if points.len() > max_points {
        points.truncate(max_points);
    }
    points
}

fn detect_leak(
    series: &[(u64, f64)],
    slope_threshold_mb_per_hour: f64,
    sustained_window_threshold: u32,
) -> LeakSentinelReport {
    if series.len() < 2 {
        return LeakSentinelReport {
            slope_mb_per_hour: 0.0,
            sustained_growth_windows: 0,
            threshold_mb_per_hour: slope_threshold_mb_per_hour,
            sustained_window_threshold,
            leak_suspected: false,
        };
    }
    let (first_t, first_rss) = series.first().copied().unwrap_or((0, 0.0));
    let (last_t, last_rss) = series.last().copied().unwrap_or((0, 0.0));
    let elapsed_hours = ((last_t.saturating_sub(first_t)).max(1) as f64) / 3600.0;
    let slope = (last_rss - first_rss) / elapsed_hours;

    let mut sustained = 0_u32;
    for window in series.windows(2) {
        if window[1].1 > window[0].1 {
            sustained = sustained.saturating_add(1);
        }
    }
    let leak_suspected =
        slope > slope_threshold_mb_per_hour && sustained >= sustained_window_threshold;
    LeakSentinelReport {
        slope_mb_per_hour: slope,
        sustained_growth_windows: sustained,
        threshold_mb_per_hour: slope_threshold_mb_per_hour,
        sustained_window_threshold,
        leak_suspected,
    }
}

fn read_rss_mb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb = rest.split_whitespace().next()?.parse::<f64>().ok()?;
                return Some(kb / 1024.0);
            }
        }
        None
    }
    #[cfg(target_os = "windows")]
    {
        None
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_inject_is_deterministic() {
        let trigger = parse_inject("timeout:llm@t=200").expect("inject");
        assert_eq!(trigger.kind, InjectKind::Timeout);
        assert_eq!(trigger.target.as_deref(), Some("llm"));
        assert_eq!(trigger.at_sec, 200);
    }

    #[test]
    fn leak_sentinel_detects_growth() {
        let series = vec![
            (0, 100.0),
            (60, 101.0),
            (120, 102.0),
            (180, 103.0),
            (240, 104.0),
            (300, 105.0),
            (360, 106.0),
        ];
        let leak = detect_leak(&series, 24.0, 6);
        assert!(leak.leak_suspected);
    }

    #[test]
    fn soak_short_injected_timeout_creates_postmortem() {
        let dir = tempfile::tempdir().expect("tempdir");
        let args = SoakRunArgs {
            duration_secs: 120,
            scenario: "golden_a".to_string(),
            out: dir.path().join("out"),
            health_poll_secs: 5,
            memory_sample_secs: 60,
            inject: vec![parse_inject("timeout:llm@t=20").expect("inject")],
            postmortem: false,
        };
        let report = soak_run(dir.path(), &args).expect("soak run");
        assert!(matches!(report.status, SoakStatus::Fail));
        let bundle = report.postmortem_bundle.expect("bundle");
        assert!(PathBuf::from(bundle).exists());
    }
}
