#![forbid(unsafe_code)]

use std::path::PathBuf;

use ucf_compute::ComputeBackendKind;
use ucf_replay::{load_fixture_records, replay_records, write_report, ReplayMode, ReplaySpec};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] != "replay" {
        eprintln!(
            "usage: ucf-replay replay --fixture <path> --from <tick> --to <tick> [--mode compute|score|full] [--backend stub|candle|burn] [--seed <u64>] [--report <path>]"
        );
        std::process::exit(1);
    }

    let fixture = arg_value(&args, "--fixture")
        .unwrap_or_else(|| "runtime/ucf-replay/fixtures/golden_replay_fixture.json".to_string());
    let from_tick = parse_u64(&args, "--from", 0);
    let to_tick = parse_u64(&args, "--to", u64::MAX);
    let mode = match arg_value(&args, "--mode").as_deref() {
        Some("score") => ReplayMode::DecisionScoring,
        Some("full") => ReplayMode::FullNoAction,
        _ => ReplayMode::ComputeOnly,
    };
    let backend_override =
        arg_value(&args, "--backend").and_then(|v| ComputeBackendKind::parse(&v));
    let seed_override = arg_value(&args, "--seed").and_then(|v| v.parse::<u64>().ok());
    let report_path = arg_value(&args, "--report")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("replay_report.json"));

    let spec = ReplaySpec {
        from_tick,
        to_tick,
        backend_override,
        seed_override,
        budget_override: None,
        mode,
    };

    let records = load_fixture_records(PathBuf::from(fixture).as_path()).expect("load fixture");
    let result = replay_records(&records, &spec);

    println!(
        "replay summary: total={} match={} drift={} unreplayable={} truncated={}",
        result.total_items, result.matched, result.drifted, result.unreplayable, result.truncated
    );
    for item in result.items.iter().filter(|i| !i.diff.pass).take(10) {
        println!(
            "drift decision_id={} corr={} reasons={}",
            item.decision_id,
            item.correlation_id,
            serde_json::to_string(&item.diff.reasons).expect("reasons")
        );
    }

    write_report(&report_path, &result).expect("write report");
    println!("report={}", report_path.display());
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

fn parse_u64(args: &[String], name: &str, default: u64) -> u64 {
    arg_value(args, name)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}
