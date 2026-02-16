#![forbid(unsafe_code)]

use std::path::PathBuf;

use ucf_ops::{
    bringup, diagnostics, export_bugreport, metrics_snapshot, replay_audit, replay_bugreport,
    verify_bugreport, ExportArgs,
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
    let workdir = arg_value(&args, "--workdir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".ucf"));

    match cmd {
        "bringup" => {
            let demo = has_flag(&args, "--demo");
            let ticks = parse_u64(&args, "--ticks", 100);
            let out = bringup(&workdir, demo, ticks)?;
            println!("pid={} status=ok", std::process::id());
            println!("workdir={}", out.workdir.display());
            println!("ess={}", out.ess_fixture_path.display());
            println!("logs={}", out.log_path.display());
            println!("decisions={} digest={}", out.decision_count, out.ess_digest);
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
        _ => {
            eprintln!(
                "usage: ucf-ops <bringup|diag|export-bugreport|verify-bugreport|replay-bugreport|replay|metrics-snapshot> [--workdir <path>]"
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
