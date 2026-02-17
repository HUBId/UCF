#![forbid(unsafe_code)]

use std::path::PathBuf;

use ucf_ops::{
    bringup, diagnostics, explain_tick, export_bugreport, metrics_snapshot, metrics_summary,
    metrics_trend, models_probe, models_verify, one_command_bringup, readiness_gate, replay_audit,
    replay_bugreport, security_verify_chain, verify_bugreport, ExplainTickRequest, ExportArgs,
    GateStatus,
};
use ucf_replay::{ReplayMode, ReplayStrictness};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    if cmd == "--version" || cmd == "version" {
        println!("ucf-ops {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let workdir = arg_value(&args, "--workdir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".ucf"));

    match cmd {
        "bringup" => {
            let ticks = parse_u64(&args, "--ticks", 32);
            if let Some(manifest) = arg_value(&args, "--manifest") {
                std::env::set_var("UCF_MODEL_MANIFEST", manifest);
            }
            if let Some(scenario) = arg_value(&args, "--scenario") {
                let out_dir = arg_value(&args, "--out")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("./out"));
                let replay_verify = !has_flag(&args, "--no-replay");
                let artifacts = one_command_bringup(
                    &workdir,
                    &PathBuf::from(scenario),
                    ticks,
                    &out_dir,
                    replay_verify,
                )?;
                println!("pid={} status=ok", std::process::id());
                println!("workdir={}", workdir.display());
                println!("profile={}", artifacts.run_metadata.profile);
                println!("run_id={}", artifacts.run_metadata.run_id);
                println!("out={}", out_dir.display());
                if let Some(report) = artifacts.replay_report {
                    println!("replay_report={report}");
                }
            } else {
                let demo = has_flag(&args, "--demo");
                let out = bringup(&workdir, demo, ticks)?;
                println!("pid={} status=ok", std::process::id());
                println!("workdir={}", out.workdir.display());
                println!("ess={}", out.ess_fixture_path.display());
                println!("logs={}", out.log_path.display());
                println!("decisions={} digest={}", out.decision_count, out.ess_digest);
            }
        }
        "diag" => {
            let report = diagnostics(&workdir)?;
            if has_flag(&args, "--json") {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                for check in &report.checks {
                    let status = if check.pass { "PASS" } else { "FAIL" };
                    println!("[{status}] {} :: {}", check.name, check.detail);
                    if !check.pass {
                        println!("  remedy: {}", check.remediation);
                    }
                }
                println!("overall={}", if report.ok() { "PASS" } else { "FAIL" });
            }
            if !report.ok() {
                std::process::exit(2);
            }
        }
        "export-bugreport" => {
            let last = arg_value(&args, "--last").and_then(|v| v.parse::<usize>().ok());
            let path = export_bugreport(
                &workdir,
                &ExportArgs {
                    last,
                    include_sandbox: has_flag(&args, "--include-sandbox"),
                    include_audit: has_flag(&args, "--include-audit"),
                },
            )?;
            println!("bugreport={}", path.display());
        }
        "verify-bugreport" => {
            let Some(path) = args.get(2) else {
                return Err("usage: ucf-ops verify-bugreport <path>".into());
            };
            verify_bugreport(&PathBuf::from(path))?;
            println!("verify=ok path={path}");
        }
        "replay-bugreport" => {
            let Some(path) = args.get(2) else {
                return Err(
                    "usage: ucf-ops replay-bugreport <path> [--mode compute|score|full]".into(),
                );
            };
            let mode = match arg_value(&args, "--mode").as_deref() {
                Some("score") => ReplayMode::DecisionScoring,
                Some("full") => ReplayMode::FullNoAction,
                _ => ReplayMode::ComputeOnly,
            };
            let report_path = replay_bugreport(&PathBuf::from(path), mode)?;
            println!("replay_report={}", report_path.display());
        }
        "replay" => {
            let from = parse_u64(&args, "--from", 0);
            let to = parse_u64(&args, "--to", u64::MAX);
            let strictness = match arg_value(&args, "--strict").as_deref() {
                Some("recompute") => ReplayStrictness::RecomputeStages,
                _ => ReplayStrictness::VerifyOnly,
            };
            let report = arg_value(&args, "--report")
                .map(PathBuf::from)
                .unwrap_or_else(|| workdir.join("replay_report.json"));
            let stop_on_first_divergence = !has_flag(&args, "--continue");
            replay_audit(
                &workdir,
                from,
                to,
                strictness,
                stop_on_first_divergence,
                &report,
            )?;
            println!("replay_report={}", report.display());
        }
        "metrics-snapshot" => {
            let snapshot = metrics_snapshot(&workdir)?;
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        }
        "explain-tick" => {
            let req = ExplainTickRequest {
                t: arg_value(&args, "--t").and_then(|v| v.parse::<u64>().ok()),
                decision_id: arg_value(&args, "--decision-id").and_then(|v| v.parse::<u64>().ok()),
                detail_level: arg_value(&args, "--detail-level")
                    .and_then(|v| v.parse::<u8>().ok())
                    .unwrap_or(1),
                digest_prefix_len: arg_value(&args, "--digest-prefix-len")
                    .and_then(|v| v.parse::<u8>().ok())
                    .unwrap_or(8),
            };
            let report = explain_tick(&workdir, req)?;
            if has_flag(&args, "--json") {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "tick={} decision_id={}",
                    report.header.t, report.header.decision_id
                );
                println!(
                    "chain={} backend_pack={}",
                    report
                        .header
                        .evidence_chain_digest_prefix
                        .as_deref()
                        .unwrap_or("none"),
                    report
                        .header
                        .backend_pack_digest_prefix
                        .as_deref()
                        .unwrap_or("none")
                );
                println!(
                    "risk={:?} confidence={:?} surprise={:?} pressure={:?}",
                    report.compute.risk.risk,
                    report.compute.risk.confidence,
                    report.compute.world.surprise,
                    report.compute.ssm.pressure
                );
                println!(
                    "tier={:?} emergency_active={} issuance={}",
                    report.governance.tier,
                    report.governance.emergency_active,
                    report.governance.issuance.len()
                );
                println!(
                    "candidates={:?} selected={:?} output_class={:?}",
                    report.decision.candidate_count,
                    report.decision.selected_candidate_id,
                    report.output.output_class
                );
                for warning in report.warnings {
                    println!("warning={warning}");
                }
            }
        }
        "metrics" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "summary" => {
                    let last = arg_value(&args, "--last")
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(64);
                    let report = metrics_summary(&workdir, last)?;
                    if has_flag(&args, "--json") {
                        println!("{}", serde_json::to_string_pretty(&report)?);
                    } else {
                        println!(
                            "ticks={} mean_surprise={:.4} max_surprise={:.4}",
                            report.ticks_observed, report.mean_surprise, report.max_surprise
                        );
                        println!(
                            "mean_pressure={:.4} max_pressure={:.4} mean_uncertainty={:.4}",
                            report.mean_pressure, report.max_pressure, report.mean_uncertainty
                        );
                        println!(
                            "tier2_3_percent={:.2} emergency_triggers={} deny_rate={:.4}",
                            report.governor_tier_2_3_percent,
                            report.emergency_triggers,
                            report.tool_issuance_deny_rate
                        );
                    }
                }
                "trend" => {
                    let from = parse_u64(&args, "--from", 0);
                    let to = parse_u64(&args, "--to", u64::MAX);
                    let trend = metrics_trend(&workdir, from, to)?;
                    if has_flag(&args, "--json") {
                        println!("{}", serde_json::to_string_pretty(&trend)?);
                    } else {
                        for p in trend {
                            println!(
                                "t={} surprise={:?} pressure={:?} uncertainty={:?} risk={:?}",
                                p.t, p.surprise, p.pressure, p.uncertainty, p.risk
                            );
                        }
                    }
                }
                _ => return Err("usage: ucf-ops metrics <summary|trend> ...".into()),
            }
        }
        "models" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "verify" => {
                    let manifest = arg_value(&args, "--manifest")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("models/manifest.toml"));
                    let report = models_verify(&manifest)?;
                    println!("{}", serde_json::to_string_pretty(&report)?);
                    let all_ok = report
                        .slots
                        .iter()
                        .all(|s| s.status == "verified" || s.status == "disabled");
                    if !all_ok {
                        std::process::exit(2);
                    }
                }
                "probe" => {
                    let manifest = arg_value(&args, "--manifest")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("models/manifest.toml"));
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/probe_report.json"));
                    let report = models_probe(&workdir, &manifest, &out)?;
                    println!("{}", serde_json::to_string_pretty(&report)?);
                    println!("out={}", out.display());
                    if !report.summary.pass {
                        std::process::exit(2);
                    }
                }
                _ => {
                    return Err(
                        "usage: ucf-ops models <verify|probe> [--manifest <path>] [--out <path>]"
                            .into(),
                    )
                }
            }
        }
        "security" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "verify-chain" => {
                    let from = parse_u64(&args, "--from", 0);
                    let to = parse_u64(&args, "--to", u64::MAX);
                    security_verify_chain(&workdir, from, to)?;
                    println!("security_chain=ok from={from} to={to}");
                }
                _ => {
                    return Err(
                        "usage: ucf-ops security verify-chain [--from <t0>] [--to <t1>]".into(),
                    )
                }
            }
        }
        "readiness-gate" => {
            let profile = arg_value(&args, "--profile").unwrap_or_else(|| "test".to_string());
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/gate_report.json"));
            let report = readiness_gate(&workdir, &profile, &out)?;
            println!("status={:?}", report.status);
            println!("out={}", out.display());
            if report.status != GateStatus::Pass {
                std::process::exit(2);
            }
        }
        _ => {
            eprintln!(
                "usage: ucf-ops <bringup|diag|export-bugreport|verify-bugreport|replay-bugreport|replay|metrics-snapshot|explain-tick|metrics|models|security|readiness-gate|version> [--workdir <path>]"
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(value) = iter.next() {
        if value == name {
            return iter.next().cloned();
        }
    }
    None
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_u64(args: &[String], name: &str, default: u64) -> u64 {
    arg_value(args, name)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}
