#![forbid(unsafe_code)]

use std::path::PathBuf;

use ucf_ops::{
    adversarial_run, bench_run, bringup, causal_slice, determinism_scan, diagnostics,
    diagnostics_collect, ebm_export_dataset, ess_compact, ess_snapshot, event_id_for_decision,
    explain_tick, explain_why, export_bugreport, load_signoff_checklist, metrics_snapshot,
    metrics_summary, metrics_trend, models_probe, models_verify, one_command_bringup, out_manifest,
    policy_diff, policy_explain, policy_validate, readiness_gate, release_rc1_gate,
    release_signoff_validate, replay_audit, replay_bugreport, run_status, runs_list, runs_search,
    runs_show, save_counterfactual_result, security_verify_chain, simulate_counterfactual,
    verify_bugreport, write_slice, AdversarialRunArgs, BenchArgs, CounterfactualRequest,
    ExplainTickRequest, ExportArgs, GateStatus,
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
        "diagnostics" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "collect" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err(
                            "usage: ucf-ops diagnostics collect --run <id> --out <path>".into()
                        );
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/diag_{run_id}.zip")));
                    let report = diagnostics_collect(&workdir, &run_id, &out)?;
                    println!("run_id={}", report.run_id);
                    println!("out={}", report.out);
                    println!("entries={}", report.entries.len());
                }
                _ => {
                    return Err("usage: ucf-ops diagnostics collect --run <id> --out <path>".into())
                }
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
        "ess" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "snapshot" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| workdir.join("ess/snapshot.snap"));
                    let manifest = ess_snapshot(&workdir, &out)?;
                    println!(
                        "snapshot={} digest={}",
                        out.display(),
                        manifest.snapshot_digest
                    );
                    println!("manifest_digest={}", manifest.manifest_digest);
                }
                "compact" => {
                    let policy = arg_value(&args, "--policy")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("policies/bundle_v1/retention_v1.json"));
                    let manifest = ess_compact(&workdir, &policy)?;
                    println!("compaction_manifest={}", serde_json::to_string(&manifest)?);
                }
                _ => return Err("usage: ucf-ops ess <snapshot|compact> ...".into()),
            }
        }
        "ebm" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "export-dataset" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err(
                            "usage: ucf-ops ebm export-dataset --run <id> --from <t0> --to <t1> --out <path> [--policy <path>]".into(),
                        );
                    };
                    let from = parse_u64(&args, "--from", 0);
                    let to = parse_u64(&args, "--to", u64::MAX);
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/ebm_dataset_v1.jsonl"));
                    let policy = arg_value(&args, "--policy")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("policies/bundle_v1/retention_v1.json"));
                    let count = ebm_export_dataset(&workdir, &run_id, from, to, &out, &policy)?;
                    println!("dataset={} samples={count}", out.display());
                }
                _ => {
                    return Err(
                        "usage: ucf-ops ebm export-dataset --run <id> --from <t0> --to <t1> --out <path> [--policy <path>]"
                            .into(),
                    )
                }
            }
        }

        "determinism" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "scan" => {
                    let report = determinism_scan(&PathBuf::from("."))?;
                    if report.violations.is_empty() {
                        println!("determinism_scan=ok violations=0");
                    } else {
                        for v in &report.violations {
                            println!("violation={}::{} pattern={}", v.path, v.line, v.pattern);
                        }
                        return Err("determinism scan failed".into());
                    }
                }
                _ => return Err("usage: ucf-ops determinism scan".into()),
            }
        }

        "policy" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "validate" => {
                    let pack = arg_value(&args, "--pack")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("policies/packs/base_v1"));
                    let overlay = arg_value(&args, "--overlay").map(PathBuf::from);
                    let report = policy_validate(&pack, overlay.as_deref())?;
                    println!("policy_graph_digest={}", report.policy_graph_digest);
                    println!("base_pack_digest={}", report.base_pack);
                    if let Some(ov) = report.overlay_pack {
                        println!("overlay_pack_digest={ov}");
                    }
                }
                "diff" => {
                    let a_pack = arg_value(&args, "--a-pack")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("policies/packs/base_v1"));
                    let b_pack = arg_value(&args, "--b-pack")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("policies/packs/base_v1"));
                    let a_overlay = arg_value(&args, "--a-overlay").map(PathBuf::from);
                    let b_overlay = arg_value(&args, "--b-overlay").map(PathBuf::from);
                    let report =
                        policy_diff(&a_pack, a_overlay.as_deref(), &b_pack, b_overlay.as_deref())?;
                    println!("digest_a={}", report.digest_a);
                    println!("digest_b={}", report.digest_b);
                    for d in report.thresholds {
                        println!("threshold_diff={d}");
                    }
                    for d in report.budgets {
                        println!("budget_diff={d}");
                    }
                    for d in report.allowlists {
                        println!("allowlist_diff={d}");
                    }
                }
                "explain" => {
                    let Some(prefix) = arg_value(&args, "--digest") else {
                        return Err("usage: ucf-ops policy explain --digest <prefix>".into());
                    };
                    if let Some(report) = policy_explain(&workdir, &prefix)? {
                        println!("run_id={}", report.run_id);
                        println!("policy_graph_digest={}", report.policy_graph_digest);
                        println!("base_pack_digest={}", report.base_pack_digest);
                        if let Some(ov) = report.overlay_pack_digest {
                            println!("overlay_pack_digest={ov}");
                        }
                    } else {
                        println!("not_found");
                    }
                }
                _ => return Err("usage: ucf-ops policy <validate|diff|explain> ...".into()),
            }
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
        "causal" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "slice" => {
                    let run_id = arg_value(&args, "--run").unwrap_or_else(|| "unknown".to_string());
                    let radius = arg_value(&args, "--radius")
                        .and_then(|v| v.parse::<u8>().ok())
                        .unwrap_or(2);
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/causal_slice.json"));
                    let event_id = if let Some(event) = arg_value(&args, "--event") {
                        event
                    } else if let Some(decision_id) = arg_value(&args, "--decision").and_then(|v| v.parse::<u64>().ok()) {
                        event_id_for_decision(&workdir, &run_id, decision_id)?
                            .ok_or_else(|| "decision event not found".to_string())?
                    } else {
                        return Err("usage: ucf-ops causal slice --run <id> --event <event_id> [--radius <n>] [--out <path>]".into());
                    };

                    let slice = causal_slice(&workdir, &run_id, &event_id, radius)?;
                    write_slice(&slice, &out)?;
                    println!("center_event_id={}", slice.center_event_id);
                    println!("nodes={} edges={}", slice.nodes.len(), slice.edges.len());
                    println!("out={}", out.display());
                }
                _ => return Err("usage: ucf-ops causal slice --run <id> --event <event_id> [--radius <n>] [--out <path>]".into()),
            }
        }
        "explain" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "why" => {
                    let Some(decision_id) =
                        arg_value(&args, "--decision").and_then(|v| v.parse::<u64>().ok())
                    else {
                        return Err("usage: ucf-ops explain why --decision <id>".into());
                    };
                    let report = explain_why(&workdir, decision_id)?;
                    if has_flag(&args, "--json") {
                        println!("{}", serde_json::to_string_pretty(&report)?);
                    } else {
                        println!(
                            "decision_id={} center_event_id={}",
                            report.decision_id, report.center_event_id
                        );
                        println!("incoming_causes={}", report.incoming_causes.len());
                        for edge in &report.incoming_causes {
                            println!(
                                "  cause {:?}: {} -> {} evidence={}",
                                edge.edge_type,
                                edge.src_event_id,
                                edge.dst_event_id,
                                edge.evidence_digest_prefix
                            );
                        }
                        println!("outgoing_effects={}", report.outgoing_effects.len());
                        for edge in &report.outgoing_effects {
                            println!(
                                "  effect {:?}: {} -> {} evidence={}",
                                edge.edge_type,
                                edge.src_event_id,
                                edge.dst_event_id,
                                edge.evidence_digest_prefix
                            );
                        }
                    }
                }
                _ => return Err("usage: ucf-ops explain why --decision <id>".into()),
            }
        }
        "counterfactual" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "simulate" => {
                    let Some(base_decision_id) = arg_value(&args, "--decision").and_then(|v| v.parse::<u64>().ok()) else {
                        return Err("usage: ucf-ops counterfactual simulate --decision <id> --candidate <id> [--out <path>]".into());
                    };
                    let Some(alternative_candidate_id) = arg_value(&args, "--candidate").and_then(|v| v.parse::<u16>().ok()) else {
                        return Err("usage: ucf-ops counterfactual simulate --decision <id> --candidate <id> [--out <path>]".into());
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/counterfactual_result.json"));
                    let result = simulate_counterfactual(
                        &workdir,
                        CounterfactualRequest {
                            base_decision_id,
                            alternative_candidate_id,
                        },
                    )?;
                    save_counterfactual_result(&result, &out)?;
                    println!("would_choose_candidate={}", result.would_choose_candidate);
                    println!("would_issue_tool={}", result.would_issue_tool);
                    println!("risk_delta_q={} energy_delta_q={}", result.risk_delta_q, result.energy_delta_q);
                    println!("out={}", out.display());
                }
                _ => return Err("usage: ucf-ops counterfactual simulate --decision <id> --candidate <id> [--out <path>]".into()),
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
        "adversarial-run" => {
            let suite = arg_value(&args, "--suite").unwrap_or_else(|| "v1".to_string());
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/adversarial_report.json"));
            let report = adversarial_run(&AdversarialRunArgs {
                workdir: workdir.clone(),
                suite,
                out: out.clone(),
            })?;
            println!(
                "suite={} pass={} cases={}",
                report.suite_version,
                report.pass,
                report.cases.len()
            );
            println!("out={}", out.display());
            if !report.pass {
                std::process::exit(2);
            }
        }
        "out" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "manifest" => {
                    let Some(dir) = arg_value(&args, "--dir") else {
                        return Err("usage: ucf-ops out manifest --dir <path>".into());
                    };
                    let manifest = out_manifest(&PathBuf::from(dir))?;
                    println!("{}", serde_json::to_string_pretty(&manifest)?);
                }
                _ => return Err("usage: ucf-ops out manifest --dir <path>".into()),
            }
        }
        "release" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "signoff" => {
                    let validate = has_flag(&args, "--validate");
                    if !validate {
                        return Err("usage: ucf-ops release signoff --validate --out <dir> --emit <path> [--checklist <path>]".into());
                    }
                    let Some(out) = arg_value(&args, "--out") else {
                        return Err("usage: ucf-ops release signoff --validate --out <dir> --emit <path> [--checklist <path>]".into());
                    };
                    let emit = arg_value(&args, "--emit")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("release/v0_signoff_result.json"));
                    let checklist = arg_value(&args, "--checklist")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("release/v0_signoff_checklist.toml"));
                    let _ = load_signoff_checklist(&checklist)?;
                    let report = release_signoff_validate(&PathBuf::from(out), &checklist, &emit)?;
                    println!("pass={}", report.pass);
                    println!("emit={}", emit.display());
                    if !report.pass {
                        std::process::exit(2);
                    }
                }
                "rc1-gate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/rc1_gate.json"));
                    let include_load_smoke = has_flag(&args, "--load-smoke");
                    let report = release_rc1_gate(&workdir, &out, include_load_smoke)?;
                    println!("status={:?}", report.status);
                    println!("out={}", out.display());
                    if report.status != GateStatus::Pass {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops release <signoff|rc1-gate> ...".into()),
            }
        }
        "bench" => {
            let scenario = arg_value(&args, "--scenario")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("fixtures/e2e_scenario_a.json"));
            let ticks = parse_u64(&args, "--ticks", 256);
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/bench_report.json"));
            let rss_sample_every = parse_u64(&args, "--rss-sample-every", 16);
            let rss_cap_mb = arg_value(&args, "--rss-cap-mb").and_then(|v| v.parse::<u64>().ok());

            let report = bench_run(&BenchArgs {
                scenario,
                ticks,
                out: out.clone(),
                rss_sample_every,
                rss_cap_mb,
            })?;
            println!("bench_run_id={}", report.run_id);
            println!("scenario_id={}", report.scenario_id);
            println!("ticks={}", report.ticks);
            println!(
                "throughput_ticks_per_sec={:.3}",
                report.throughput_ticks_per_sec
            );
            println!("out={}", out.display());
            if report.memory.cap_exceeded {
                std::process::exit(2);
            }
        }

        "runs" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "list" => {
                    let last = arg_value(&args, "--last")
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(50);
                    let entries = runs_list(&workdir, last)?;
                    for e in entries {
                        println!(
                            "run_id={} started_at={} parent={} reason={} policy={} pack={} model={} profile={} status={} last_tick={}",
                            e.run_id,
                            e.started_at_tick,
                            e.parent_run_id.as_deref().unwrap_or("none"),
                            e.resume_reason
                                .as_ref()
                                .map(|r| format!("{:?}", r))
                                .unwrap_or_else(|| "none".to_string()),
                            e.policy_bundle_hash_prefix,
                            e.pack_digest_prefix,
                            e.model_hashes_digest_prefix,
                            e.profile,
                            e.status,
                            e.last_tick.map(|t| t.to_string()).unwrap_or_else(|| "none".to_string())
                        );
                    }
                }
                "show" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err("usage: ucf-ops runs show --run <id>".into());
                    };
                    let run = runs_show(&workdir, &run_id)?
                        .ok_or_else(|| format!("run not found: {run_id}"))?;
                    println!("{}", serde_json::to_string_pretty(&run)?);
                }
                "search" => {
                    let pack = arg_value(&args, "--pack");
                    let policy = arg_value(&args, "--policy");
                    let model = arg_value(&args, "--model");
                    let entries = runs_search(
                        &workdir,
                        pack.as_deref(),
                        policy.as_deref(),
                        model.as_deref(),
                    )?;
                    for e in entries {
                        println!(
                            "run_id={} started_at={} parent={} reason={} policy={} pack={} model={} profile={} status={} last_tick={}",
                            e.run_id,
                            e.started_at_tick,
                            e.parent_run_id.as_deref().unwrap_or("none"),
                            e.resume_reason
                                .as_ref()
                                .map(|r| format!("{:?}", r))
                                .unwrap_or_else(|| "none".to_string()),
                            e.policy_bundle_hash_prefix,
                            e.pack_digest_prefix,
                            e.model_hashes_digest_prefix,
                            e.profile,
                            e.status,
                            e.last_tick.map(|t| t.to_string()).unwrap_or_else(|| "none".to_string())
                        );
                    }
                }
                _ => return Err("usage: ucf-ops runs <list|show|search> ...".into()),
            }
        }
        "status" => {
            let Some(run_id) = arg_value(&args, "--run") else {
                return Err("usage: ucf-ops status --run <id>".into());
            };
            let status = run_status(&workdir, &run_id)?;
            println!("run_id={}", status.run_id);
            println!("active_slots={}", status.active_slots.join(","));
            println!(
                "governor_tier={} governor_score={:.4} emergency_active={}",
                status.governor_tier, status.governor_score, status.emergency_active
            );
            for p in status.last_ticks {
                println!(
                    "tick={} pressure={:?} surprise={:?} uncertainty={:?} risk={:?}",
                    p.t, p.pressure, p.surprise, p.uncertainty, p.risk
                );
            }
            for (kind, reason) in status.issuance_denies {
                println!("deny kind={} reason={}", kind, reason);
            }
        }
        _ => {
            eprintln!(
                "usage: ucf-ops <bringup|diag|diagnostics|export-bugreport|verify-bugreport|replay-bugreport|replay|metrics-snapshot|explain-tick|metrics|models|security|readiness-gate|adversarial-run|out|release|bench|runs|status|ess|ebm|version> [--workdir <path>]"
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
