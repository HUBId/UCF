#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use ucf_ops::{
    adversarial_run, airgap_export_models, airgap_export_policies, airgap_export_repro,
    airgap_export_run_cert, airgap_import, alerts_report, attest_bundle, attest_keys_generate,
    attest_run, attest_verify, audit_scan, bench_run, bringup, causal_slice, determinism_scan,
    diagnostics, diagnostics_collect, drift_report, ebm_export_dataset, ess_compact, ess_snapshot,
    event_id_for_decision, explain_tick, explain_why, export_bugreport,
    export_policy_key_registry_v1, exports_bundle_spine_check, exports_normalize_check,
    exports_roundtrip_check, gateway_threat_test, goldens_generate, goldens_update, goldens_verify,
    goldens_verify_detailed, governance_entry_check, governance_surfaces_check, hardware_scan,
    interop_consistency_matrix, load_applied_supported_set_context_v1, load_signoff_checklist,
    logs_prove, logs_verify_proof, metrics_snapshot, metrics_summary, metrics_trend,
    migrate_config_v1, models_active_check, models_active_evidence, models_active_review_snapshot,
    models_applied_scope_check, models_backend_resolution, models_consistency_check,
    models_eligibility, models_evidence_snapshot, models_list, models_probe, models_probe_slot,
    models_promote, models_recommend_rollback, models_rollback, models_shadow_ready, models_stage,
    models_supported_scope_execute, models_supported_scope_reevaluate, models_supported_set_apply,
    models_supported_set_review, models_verify, models_verify_lifecycle, net_deps_audit,
    nightly_summarize, one_command_bringup, operator_export_chain_check, operator_report,
    operator_report_text, operator_review_packet, operator_review_packet_text,
    operator_roundtrip_chain_check, operator_signoff, operator_signoff_text,
    operator_workflow_chain, operator_workflow_chain_text, out_manifest, parse_duration_secs,
    parse_inject, parse_slot, path_scan, policy_diff, policy_explain, policy_validate,
    portability_check, portability_report, preflight, readiness_gate, readiness_spine_check,
    release_build_rc, release_rc1_gate, release_signoff_validate, remediation_consistency_check,
    remediation_interop_check, remediation_spine_check, replay_audit, replay_bugreport, repro_pack,
    repro_verify, review_truth_check, run_status, runs_list, runs_search, runs_show,
    save_counterfactual_result, scope_authority_check, second_slot_parity_report,
    security_verify_chain, simulate_counterfactual, soak_run, strict_check, strict_explain,
    troubleshoot, v0_gate, v1_smoke, v2_gate, v3_gate, v4_gate, v5_gate, v6_gate, v7_gate,
    verify_bugreport, world_parity_report, world_shadow_report, write_slice, AdversarialRunArgs,
    AirgapArtifactType, AirgapImportArgs, AirgapImportMode, BenchArgs, BugKitBuildArgs,
    ChangeImpactArgs, ConfigV1, CounterfactualRequest, DevLoopArgs, DocsLintArgs, DocsLintMode,
    DocsLintStatus, ExplainTickRequest, ExportArgs, GateStatus, GoldenGenerateArgs,
    GoldenVerifyArgs, GoldenVerifyReport, GovernanceEntryCheckStatusV1, NightlySummarizeArgs,
    OperatorReportArgs, OperatorReviewPacketArgs, OperatorSignoffArgs, OperatorWorkflowArgs,
    ReleaseBuildRcArgs, SoakRunArgs, SpecSnapshotArgs, StrictEvidenceContextV1,
    V2GateOverallStatus, V3GateOverallStatus, V4GateOverallStatus, V5GateOverallStatus,
    V6GateOverallStatus, V7GateOverallStatus,
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
    if has_flag(&args, "--strict") {
        std::env::set_var("UCF_STRICT_MODE", "1");
    }
    if let Some(bundle_root) = arg_value(&args, "--bundle") {
        std::env::set_current_dir(bundle_root)?;
    }
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
        "health" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "check" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/health.json"));
                    let endpoint = arg_value(&args, "--endpoint")
                        .unwrap_or_else(|| "unix://.ucf/data/ipc/gateway.sock".to_string());
                    let auth = arg_value(&args, "--auth").unwrap_or_default();
                    let output = std::process::Command::new("cargo")
                        .args([
                            "run",
                            "-p",
                            "ucf-client",
                            "--",
                            "--endpoint",
                            endpoint.as_str(),
                            "--auth",
                            auth.as_str(),
                            "health",
                        ])
                        .output()?;
                    if !output.status.success() {
                        return Err(format!(
                            "gateway health query failed: {}",
                            String::from_utf8_lossy(&output.stderr)
                        )
                        .into());
                    }
                    let mut report: serde_json::Value = serde_json::from_slice(&output.stdout)
                        .map_err(|e| format!("invalid health payload: {e}"))?;
                    if let Some(slots) = report
                        .get("active_slots_summary")
                        .and_then(|v| v.as_str())
                        .map(|s| s.chars().take(128).collect::<String>())
                    {
                        report["active_slots_summary"] = serde_json::Value::String(slots);
                    }
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
                    println!("out={}", out.display());
                    let code = match report.get("status").and_then(|v| v.as_i64()) {
                        Some(1) => 0,
                        Some(2) => 2,
                        Some(3) => 3,
                        _ => 3,
                    };
                    println!("status_code={code}");
                    if code != 0 {
                        std::process::exit(code);
                    }
                }
                _ => {
                    return Err(
                        "usage: ucf-ops health check [--out <path>] [--endpoint <local>] [--auth <token>]"
                            .into(),
                    )
                }
            }
        }
        "diagnostics" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "collect" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err(
                            "usage: ucf-ops diagnostics collect --run <id> --out <path> [--include_backtrace]".into()
                        );
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/diag_{run_id}.zip")));
                    let report = diagnostics_collect(
                        &workdir,
                        &run_id,
                        &out,
                        has_flag(&args, "--include_backtrace"),
                    )?;
                    println!("run_id={}", report.run_id);
                    println!("out={}", report.out);
                    println!("entries={}", report.entries.len());
                }
                _ => {
                    return Err("usage: ucf-ops diagnostics collect --run <id> --out <path> [--include_backtrace]".into())
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

        "audit" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "scan" => {
                    let report = audit_scan(&PathBuf::from("."))?;
                    if report.violations.is_empty() {
                        println!("audit_scan=ok violations=0");
                    } else {
                        for v in &report.violations {
                            println!("violation={}::{} pattern={}", v.path, v.line, v.pattern);
                        }
                        return Err("audit scan failed".into());
                    }
                }
                "hardware-scan" => {
                    let report = hardware_scan(&PathBuf::from("."))?;
                    if report.violations.is_empty() {
                        println!("hardware_scan=ok violations=0");
                    } else {
                        println!("hardware_scan=fail violations={}", report.violations.len());
                        for violation in report.violations {
                            println!(
                                "{}:{}:{}",
                                violation.path, violation.line, violation.pattern
                            );
                        }
                        return Err("hardware scan failed".into());
                    }
                }
                "path-scan" => {
                    let report = path_scan(&PathBuf::from("."))?;
                    if report.violations.is_empty() {
                        println!("path_scan=ok violations=0");
                    } else {
                        println!("path_scan=fail violations={}", report.violations.len());
                        for violation in report.violations {
                            println!(
                                "{}:{}:{}",
                                violation.path, violation.line, violation.pattern
                            );
                        }
                        return Err("path scan failed".into());
                    }
                }
                "net-deps" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/net_deps.json"));
                    let allowlist = arg_value(&args, "--allowlist")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("docs/network_allowlist.toml"));
                    let report = net_deps_audit(&PathBuf::from("."), &allowlist)?;
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
                    println!("out={}", out.display());
                    if report.violations.is_empty() {
                        println!("net_deps=ok violations=0");
                    } else {
                        println!("net_deps=fail violations={}", report.violations.len());
                        for violation in report.violations {
                            println!(
                                "root={} forbidden={} path={}",
                                violation.root_crate,
                                violation.forbidden_crate,
                                violation.path.join(" -> ")
                            );
                        }
                        return Err("network dependency audit failed".into());
                    }
                }
                _ => {
                    return Err("usage: ucf-ops audit scan|hardware-scan|path-scan|net-deps".into())
                }
            }
        }

        "portability" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "check" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/portability.json"));
                    let report = portability_check(&out)?;
                    println!("schema_version={}", report.schema_version);
                    println!("os={} arch={}", report.os, report.arch);
                    println!(
                        "deterministic_within_os={}",
                        if report.deterministic_within_os {
                            "PASS"
                        } else {
                            "FAIL"
                        }
                    );
                    for (name, prefix) in report.digest_prefixes {
                        println!("digest_prefix_{}={}", name, prefix);
                    }
                    println!(
                        "fixed_point_summary=count:{} risk_q:{} pressure_q:{} surprise_q:{} uncertainty_q:{}",
                        report.fixed_point_summary.sample_count,
                        report.fixed_point_summary.mean_risk_q,
                        report.fixed_point_summary.mean_pressure_q,
                        report.fixed_point_summary.mean_surprise_q,
                        report.fixed_point_summary.mean_uncertainty_q
                    );
                    println!("out={}", out.display());
                    if !report.deterministic_within_os {
                        for remedy in report.remediation {
                            println!("remedy: {}", remedy);
                        }
                        std::process::exit(2);
                    }
                }
                "report" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/portability_report.json"));
                    let report = portability_report(&workdir, &out)?;
                    let has_fail = [
                        &report.docs_lint,
                        &report.path_scan,
                        &report.hardware_scan,
                        &report.artifact_schema_snapshot_check,
                        &report.active_review_snapshot_smoke,
                        &report.repro_pack_smoke,
                        &report.bugkit_smoke,
                        &report.remediation_consistency_smoke,
                        &report.v0_gate,
                        &report.v1_gate,
                        &report.v2_gate,
                        &report.eligibility_smoke,
                        &report.strict_check_smoke,
                        &report.operator_report_smoke,
                    ]
                    .iter()
                    .any(|check| matches!(check.status, ucf_ops::PortabilityGateStatus::Fail));
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if has_fail {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops portability <check|report> [--out <path>]".into()),
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
                "keys" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("docs/policy_key_registry.md"));
                    export_policy_key_registry_v1(&out)?;
                    println!("policy_keys_registry={}", out.display());
                }
                _ => return Err("usage: ucf-ops policy <validate|diff|explain|keys> ...".into()),
            }
        }

        "config" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "migrate" => {
                    let Some(input) = arg_value(&args, "--in") else {
                        return Err("usage: ucf-ops config migrate --in <old.toml> --out <new.toml> --diff <path>".into());
                    };
                    let Some(output) = arg_value(&args, "--out") else {
                        return Err("usage: ucf-ops config migrate --in <old.toml> --out <new.toml> --diff <path>".into());
                    };
                    let diff = arg_value(&args, "--diff")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/config_diff.txt"));
                    let output_path = PathBuf::from(&output);
                    let report = migrate_config_v1(&PathBuf::from(input), &output_path, &diff)?;
                    println!("migrated_out={output}");
                    println!("diff={}", diff.display());
                    println!("old_digest={}", report.old_digest);
                    println!("new_digest={}", report.new_digest);
                    for warning in report.warnings {
                        println!("warning={warning}");
                    }
                }
                "validate" => {
                    let Some(input) = arg_value(&args, "--in") else {
                        return Err("usage: ucf-ops config validate --in <config_v1.toml>".into());
                    };
                    let raw = std::fs::read_to_string(&input)?;
                    let _ = ConfigV1::from_toml_str(&raw)?;
                    println!("config_v1_valid={input}");
                }
                _ => return Err("usage: ucf-ops config <migrate|validate> ...".into()),
            }
        }

        "change-impact" => {
            let base = arg_value(&args, "--base").unwrap_or_else(|| "HEAD~1".to_string());
            let head = arg_value(&args, "--head").unwrap_or_else(|| "HEAD".to_string());
            let out_md = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/change_impact_plan.md"));
            let out_json = arg_value(&args, "--json-out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/change_impact_plan.json"));
            let rules_path = arg_value(&args, "--rules")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("docs/change_impact_rules.toml"));
            let plan = ucf_ops::change_impact(&ChangeImpactArgs {
                base,
                head,
                rules_path,
                out_md: out_md.clone(),
                out_json: out_json.clone(),
            })?;
            println!("changed_files={}", plan.total_changed_files);
            println!("required_gates={}", plan.required_gates.join(","));
            println!("out_md={}", out_md.display());
            println!("out_json={}", out_json.display());
        }

        "spec" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "snapshot" => {
                    let policy = arg_value(&args, "--policy")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("policies/packs/base_v1"));
                    let overlay = arg_value(&args, "--overlay").map(PathBuf::from);
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("docs/spec_snapshot.md"));
                    ucf_ops::generate_spec_snapshot(&SpecSnapshotArgs {
                        policy,
                        overlay,
                        out: out.clone(),
                    })?;
                    println!("spec_snapshot={}", out.display());
                }
                "artifact-schemas" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("docs/artifact_schema_snapshots"));
                    let covered = ucf_ops::generate_artifact_schema_snapshots(&ucf_ops::ArtifactSchemaArgs {
                        repo_root: PathBuf::from("."),
                        out_dir: out.clone(),
                    })?;
                    println!("artifact_schema_snapshots={}", out.display());
                    println!("covered_artifacts={}", covered.join(","));
                }
                "artifact-schemas-check" => {
                    let snapshot_dir = PathBuf::from("docs/artifact_schema_snapshots");
                    let report = ucf_ops::check_artifact_schema_snapshots(&ucf_ops::ArtifactSchemaArgs {
                        repo_root: PathBuf::from("."),
                        out_dir: snapshot_dir,
                    })?;
                    println!("overall={}", if report.ok { "PASS" } else { "FAIL" });
                    for diff in &report.diffs {
                        println!("[{:?}] {} :: {}", diff.drift_kind, diff.artifact, diff.summary);
                    }
                    if !report.ok {
                        println!("remediation={}", report.remediation);
                    }
                    if let Some(out) = arg_value(&args, "--out").map(PathBuf::from) {
                        if let Some(parent) = out.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
                        println!("report={}", out.display());
                    }
                    if !report.ok {
                        std::process::exit(2);
                    }
                }
                _ => {
                    return Err(
                        "usage: ucf-ops spec snapshot [--policy <dir>] [--overlay <dir>] [--out <path>] | spec artifact-schemas [--out <dir>] | spec artifact-schemas-check [--out <path>]"
                            .into(),
                    )
                }
            }
        }
        "docs" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "lint" => {
                    let mode = if has_flag(&args, "--warn") {
                        DocsLintMode::Warn
                    } else {
                        DocsLintMode::Strict
                    };
                    let report = ucf_ops::docs_lint(&DocsLintArgs {
                        repo_root: PathBuf::from("."),
                        policy_pack: arg_value(&args, "--policy")
                            .map(PathBuf::from)
                            .unwrap_or_else(|| PathBuf::from("policies/packs/base_v1")),
                        overlay_pack: if has_flag(&args, "--no-overlay") {
                            None
                        } else {
                            arg_value(&args, "--overlay")
                                .map(PathBuf::from)
                                .or(Some(PathBuf::from("policies/packs/overlays/test")))
                        },
                        spec_snapshot: arg_value(&args, "--spec")
                            .map(PathBuf::from)
                            .unwrap_or_else(|| PathBuf::from("docs/spec_snapshot.md")),
                        prompt_index: arg_value(&args, "--prompt-index")
                            .map(PathBuf::from)
                            .unwrap_or_else(|| PathBuf::from("docs/prompt_series_index.md")),
                        module_map: arg_value(&args, "--module-map")
                            .map(PathBuf::from)
                            .unwrap_or_else(|| PathBuf::from("docs/module_map.md")),
                        deploy_doc: PathBuf::from("docs/deploy_portable.md"),
                        artifact_schema_snapshot_dir: PathBuf::from("docs/artifact_schema_snapshots"),
                        mode,
                    })?;
                    for check in &report.checks {
                        let status = match check.status {
                            DocsLintStatus::Pass => "PASS",
                            DocsLintStatus::Warn => "WARN",
                            DocsLintStatus::Fail => "FAIL",
                        };
                        println!("[{status}] {} :: {}", check.name, check.detail);
                        if let Some(remediation) = &check.remediation {
                            if check.status != DocsLintStatus::Pass {
                                println!("  remedy: {remediation}");
                            }
                        }
                    }
                    println!("overall={}", if report.ok { "PASS" } else { "FAIL" });
                    if let Some(out) = arg_value(&args, "--out").map(PathBuf::from) {
                        if let Some(parent) = out.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
                        println!("report={}", out.display());
                    }
                    if !report.ok {
                        std::process::exit(2);
                    }
                }
                "remediation-codes" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("docs/remediation_codes_v1.md"));
                    ucf_ops::generate_remediation_codes_doc(&out)?;
                    println!("remediation_codes_doc={}", out.display());
                }
                _ => {
                    return Err("usage: ucf-ops docs lint [--strict|--warn] [--out <path>] [--policy <dir>] [--overlay <dir>|--no-overlay] [--spec <path>] [--prompt-index <path>] [--module-map <path>] | docs remediation-codes [--out <path>]".into())
                }
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
                        .unwrap_or_else(|| PathBuf::from("models/MANIFEST.toml"));
                    let out = arg_value(&args, "--out").map(PathBuf::from);
                    let use_legacy = manifest
                        .file_name()
                        .and_then(|v| v.to_str())
                        .map(|v| v.eq_ignore_ascii_case("manifest.toml") && v != "MANIFEST.toml")
                        .unwrap_or(false);
                    if use_legacy {
                        let report = models_verify(&manifest)?;
                        if let Some(path) = out {
                            if let Some(parent) = path.parent() {
                                std::fs::create_dir_all(parent)?;
                            }
                            std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
                        }
                        println!("{}", serde_json::to_string_pretty(&report)?);
                        let all_ok = report
                            .slots
                            .iter()
                            .all(|s| s.status == "verified" || s.status == "disabled");
                        if !all_ok {
                            std::process::exit(2);
                        }
                    } else {
                        let report = models_verify_lifecycle(&manifest)?;
                        if let Some(path) = out {
                            if let Some(parent) = path.parent() {
                                std::fs::create_dir_all(parent)?;
                            }
                            std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
                        }
                        println!("{}", serde_json::to_string_pretty(&report)?);
                        let all_ok = report.manifest_present
                            && report.digest_match
                            && report.promoted_hashes_exist
                            && report.files_verified;
                        if !all_ok {
                            std::process::exit(2);
                        }
                    }
                }
                "stage" => {
                    let slot = parse_slot(&arg_value(&args, "--slot").ok_or("missing --slot")?)?;
                    let path = PathBuf::from(arg_value(&args, "--path").ok_or("missing --path")?);
                    let out = arg_value(&args, "--out").map(PathBuf::from);
                    let result = models_stage(slot, &path)?;
                    if let Some(path) = out {
                        if let Some(parent) = path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::write(&path, serde_json::to_vec_pretty(&result)?)?;
                    }
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
                "probe" => {
                    if let Some(slot_value) = arg_value(&args, "--slot") {
                        let slot = parse_slot(&slot_value)?;
                        let hash = arg_value(&args, "--hash");
                        let out = arg_value(&args, "--out")
                            .map(PathBuf::from)
                            .unwrap_or_else(|| PathBuf::from(format!("./out/probe_{}.json", slot.as_str())));
                        let report = models_probe_slot(slot, hash.as_deref(), &out)?;
                        println!("{}", serde_json::to_string_pretty(&report)?);
                        println!("out={}", out.display());
                        if !report.pass() {
                            std::process::exit(2);
                        }
                    } else {
                        let manifest = arg_value(&args, "--manifest")
                            .map(PathBuf::from)
                            .unwrap_or_else(|| PathBuf::from("models/MANIFEST.toml"));
                        let out = arg_value(&args, "--out")
                            .map(PathBuf::from)
                            .unwrap_or_else(|| PathBuf::from("./out/probe_report.json"));
                        let report = models_probe(&workdir, &manifest, &out)?;
                        println!("{}", serde_json::to_string_pretty(&report)?);
                        println!("out={}", out.display());
                        if !report.summary.pass {
                            let all_disabled = report
                                .results
                                .iter()
                                .all(|r| matches!(r.status, ucf_ops::ProbeStatus::Disabled));
                            if !all_disabled {
                                std::process::exit(2);
                            }
                        }
                    }
                }
                "promote" => {
                    let slot = parse_slot(&arg_value(&args, "--slot").ok_or("missing --slot")?)?;
                    let hash = arg_value(&args, "--hash").ok_or("missing --hash")?;
                    let keep = arg_value(&args, "--history-keep").and_then(|v| v.parse::<usize>().ok());
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/models_promote.json"));
                    let report = models_promote(slot, &hash, keep)?;
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
                    println!("{}", serde_json::to_string_pretty(&report)?);
                }
                "rollback" => {
                    let slot = parse_slot(&arg_value(&args, "--slot").ok_or("missing --slot")?)?;
                    let to = arg_value(&args, "--to");
                    let steps = arg_value(&args, "--steps").and_then(|v| v.parse::<usize>().ok());
                    if to.is_none() && steps.is_none() {
                        return Err("usage: ucf-ops models rollback --slot <slot> (--to <hash> | --steps <n>) [--history-keep <n>] [--out <path>]".into());
                    }
                    let keep = arg_value(&args, "--history-keep").and_then(|v| v.parse::<usize>().ok());
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/models_rollback.json"));
                    let report = models_rollback(slot, to.as_deref(), steps, keep)?;
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
                    println!("{}", serde_json::to_string_pretty(&report)?);
                }
                "list" => {
                    let slot = parse_slot(&arg_value(&args, "--slot").ok_or("missing --slot")?)?;
                    let report = models_list(slot)?;
                    println!("{}", serde_json::to_string_pretty(&report)?);
                }
                "active-check" => {
                    let slot = parse_slot(&arg_value(&args, "--slot").ok_or("missing --slot")?)?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/active_check_{}.json", slot.as_str())));
                    let report = models_active_check(slot, &workdir, &out)?;
                    println!("{}", serde_json::to_string_pretty(&report)?);
                    println!("out={}", out.display());
                    if !matches!(report.status, ucf_ops::ActiveCheckStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                "recommend-rollback" => {
                    let slot = parse_slot(&arg_value(&args, "--slot").ok_or("missing --slot")?)?;
                    let report = models_recommend_rollback(slot, &workdir)?;
                    println!("{}", serde_json::to_string_pretty(&report)?);
                }
                "shadow-ready" => {
                    let slot = arg_value(&args, "--slot").map(|v| parse_slot(&v)).transpose()?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/shadow_ready_report.json"));
                    let report = models_shadow_ready(&workdir, slot, &out)?;
                    for slot_report in &report.slots {
                        let target_prefix = slot_report.target_hash.chars().take(16).collect::<String>();
                        println!(
                            "slot={} target={} shadow_ready={}{}",
                            slot_report.slot_id,
                            target_prefix,
                            if slot_report.shadow_ready { "yes" } else { "no" },
                            slot_report
                                .denial_reason_code
                                .as_ref()
                                .map(|v| format!(" reason={v}"))
                                .unwrap_or_default()
                        );
                    }
                    println!("overall={:?}", report.overall_status);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, ucf_ops::AggregatedStatusV1::Pass) {
                        std::process::exit(2);
                    }
                }
                "active-evidence" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/active_evidence_report.json"));
                    let report = models_active_evidence(&workdir, &out)?;
                    for slot in &report.slots {
                        println!(
                            "slot={} target={} eligible={}{}",
                            slot.slot_id,
                            slot.target_hash.chars().take(16).collect::<String>(),
                            if slot.active_eligible { "yes" } else { "no" },
                            slot.denial_reason_code
                                .as_ref()
                                .map(|v| format!(" reason={v}"))
                                .unwrap_or_default()
                        );
                    }
                    println!(
                        "overall_all_supported_slots_active_eligible={}",
                        if report.all_supported_slots_active_eligible {
                            "yes"
                        } else {
                            "no"
                        }
                    );
                    println!("out={}", out.display());
                    println!("remediation:");
                    println!("- cargo run -p ucf-ops -- models probe --slot <slot> --out ./out/probe_<slot>.json");
                    println!("- cargo run -p ucf-ops -- models shadow-ready --out ./out/shadow_ready_report.json");
                    println!("- cargo run -p ucf-ops -- drift report --run <run_id> --windows 20 --out ./out/drift_report.json");
                    println!("- keep slot in shadow until active evidence eligibility is PASS");
                    if !report.all_supported_slots_active_eligible {
                        std::process::exit(2);
                    }
                }
                "parity" => {
                    let slot = parse_slot(&arg_value(&args, "--slot").ok_or("missing --slot")?)?;
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err("usage: ucf-ops models parity --slot <sae|ssm> --run <id> --out <path>".into());
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/{}_parity_report.json", slot.as_str())));
                    let report = second_slot_parity_report(&workdir, &run_id, &out, Some(slot))?;
                    println!("slot={}", report.slot_id);
                    println!("parity_windows={}", report.parity_records.len());
                    println!("compared_backends={}", report.compared_backends.join(","));
                    println!("candle_status={:?}", report.candle_status);
                    println!("burn_support_state={:?}", report.burn_support_state);
                    println!("burn_parity_status={:?}", report.burn_parity_status);
                    println!("burn_parity_present={}", if report.burn_parity_present { "yes" } else { "no" });
                    println!("parity_ready_hint={}", if report.parity_ready_hint { "yes" } else { "no" });
                    println!("shadow_ready_hint={}", if report.shadow_ready_hint { "yes" } else { "no" });
                    println!("out={}", out.display());
                }
                "eligibility" => {
                    let slot = arg_value(&args, "--slot").map(|v| parse_slot(&v)).transpose()?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/models_eligibility_report.json"));
                    let report = models_eligibility(&workdir, slot, &out)?;
                    for slot in &report.slots {
                        println!(
                            "slot={} target={} probe_ready={} shadow_ready={} active_eligible={}{}",
                            slot.slot_id,
                            slot.target_hash_prefix,
                            if slot.probe_ready { "yes" } else { "no" },
                            if slot.shadow_ready { "yes" } else { "no" },
                            if slot.active_eligible { "yes" } else { "no" },
                            slot
                                .denial_reason_active
                                .as_ref()
                                .or(slot.denial_reason_shadow.as_ref())
                                .or(slot.denial_reason_probe.as_ref())
                                .map(|v| format!(" reason={v}"))
                                .unwrap_or_default()
                        );
                    }
                    println!("overall={:?}", report.overall_status);
                    println!("out={}", out.display());
                }
                "evidence-snapshot" => {
                    let slot = arg_value(&args, "--slot").map(|v| parse_slot(&v)).transpose()?;
                    let run_id = arg_value(&args, "--run");
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/backend_evidence_snapshot.json"));
                    let report = models_evidence_snapshot(&workdir, slot, run_id.as_deref())?;
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
                    for slot in &report.slots {
                        println!(
                            "slot={} target={} probe_ready={} shadow_ready={} active_eligible={} backend_support=stub:{:?},candle:{:?},burn:{:?}{}",
                            slot.slot_id,
                            slot.target_hash_prefix,
                            if slot.readiness.probe_ready { "yes" } else { "no" },
                            if slot.readiness.shadow_ready { "yes" } else { "no" },
                            if slot.readiness.active_eligible { "yes" } else { "no" },
                            slot.backend_support.stub,
                            slot.backend_support.candle,
                            slot.backend_support.burn,
                            slot
                                .denials
                                .active
                                .as_ref()
                                .or(slot.denials.shadow.as_ref())
                                .or(slot.denials.probe.as_ref())
                                .map(|v| format!(" reason={:?}", v))
                                .unwrap_or_default(),
                        );
                    }
                    println!("snapshot_digest={}", report.snapshot_digest);
                    println!("out={}", out.display());
                }
                "active-review-snapshot" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/active_review_snapshot.json"));
                    let report = models_active_review_snapshot(&workdir, &out)?;
                    for slot in &report.slots {
                        let blocking = if slot.strict_blocking {
                            "strict"
                        } else if slot.drift_blocking {
                            "drift"
                        } else if slot.alert_blocking {
                            "alert"
                        } else if !slot.active_eligible {
                            "eligibility"
                        } else {
                            "none"
                        };
                        println!(
                            "slot={} target={} active_eligible={} blocking={} remediation={}",
                            slot.slot_id,
                            slot.target_hash_prefix,
                            if slot.active_eligible { "yes" } else { "no" },
                            blocking,
                            slot.remediation_codes
                                .first()
                                .cloned()
                                .unwrap_or_else(|| "none".to_string())
                        );
                    }
                    println!("overall={:?}", report.overall_review_status);
                    println!(
                        "signoff_alignment={} code={}",
                        if report.signoff_alignment.aligned { "yes" } else { "no" },
                        report.signoff_alignment.status_code
                    );
                    println!("snapshot_digest={}", report.snapshot_digest);
                    println!("out={}", out.display());
                }
                "supported-set-review" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/supported_set_review.json"));
                    let report = models_supported_set_review(&workdir, &out)?;
                    println!(
                        "current_supported_slots={}",
                        report.policy.current_supported_slots.join(",")
                    );
                    println!(
                        "decision={:?}",
                        report.policy.decision
                    );
                    if let Some(slot) = report.policy.chosen_candidate_slot.as_ref() {
                        println!("chosen_candidate_slot={slot}");
                    }
                    println!("rationale_codes={}", report.policy.rationale_codes.join(","));
                    println!("out={}", out.display());
                }
                "supported-scope-reevaluate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/supported_scope_reeval.json"));
                    let report = models_supported_scope_reevaluate(&workdir, &out)?;
                    let applied = load_applied_supported_set_context_v1(&workdir)?;
                    println!("current_applied_set={}", applied.slots.join(","));
                    println!("reevaluation_decision={:?}", report.reevaluation_decision);
                    if let Some(slot) = report.chosen_candidate_slot.as_ref() {
                        println!("chosen_candidate_slot={slot}");
                    }
                    println!("primary_reasons={}", report.rationale_codes.join(","));
                    println!("out={}", out.display());
                }
                "supported-scope-execute" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/supported_scope_execute_v3.json"));
                    let report = models_supported_scope_execute(&workdir, &out)?;
                    let applied = load_applied_supported_set_context_v1(&workdir)?;
                    println!("previous_applied_set={}", applied.slots.join(","));
                    println!("decision={:?}", report.execution_decision);
                    if let Some(slot) = report.chosen_candidate_slot.as_ref() {
                        println!("chosen_candidate_slot={slot}");
                    }
                    println!("resulting_set_digest_prefix={}", report.resulting_supported_set_digest_prefix);
                    println!("primary_rationale={}", report.rationale_codes.first().cloned().unwrap_or_else(|| "NONE".to_string()));
                    println!("out={}", out.display());
                }
                "supported-set-apply" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/supported_set_apply.json"));
                    let report = models_supported_set_apply(&workdir, &out)?;
                    println!("previous_set={}", report.previous_slots.join(","));
                    println!("new_set={}", report.resulting_slots.join(","));
                    println!("decision={:?}", report.decision);
                    println!(
                        "primary_rationale_or_denial={}",
                        report
                            .denial_code
                            .as_ref()
                            .map(|c| format!("{:?}", c))
                            .or_else(|| report.rationale_codes.first().cloned())
                            .unwrap_or_else(|| "NONE".to_string())
                    );
                    println!("out={}", out.display());
                }
                "backend-resolution" => {
                    let slot = parse_slot(&arg_value(&args, "--slot").ok_or("missing --slot")?)?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/backend_resolution_{}.json", slot.as_str())));
                    let run_id = arg_value(&args, "--run");
                    let resolution = models_backend_resolution(&workdir, slot, run_id.as_deref())?;
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&out, serde_json::to_vec_pretty(&resolution)?)?;
                    let evidence = models_evidence_snapshot(&workdir, Some(slot), run_id.as_deref())?;
                    let candle_state = evidence
                        .slots
                        .iter()
                        .find(|s| s.slot_id == slot.as_str())
                        .map(|s| format!("{:?}", s.backend_support.candle))
                        .unwrap_or_else(|| "Unknown".to_string());
                    println!("slot={}", resolution.slot_id);
                    println!("candle_support_state={}", candle_state);
                    println!("burn_resolution={:?}", resolution.resolution);
                    println!("burn_support_state={:?}", resolution.support_state);
                    println!("rationale_codes={}", resolution.rationale_codes.join(","));
                    println!("shadow_compare_available={}", if matches!(resolution.resolution, ucf_ops::BurnResolutionStatusV1::BurnSupportedForShadowCompare) { "yes" } else { "no" });
                    println!("out={}", out.display());
                }
                "consistency-check" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/models_consistency_check.json"));
                    let report = models_consistency_check(&workdir, &out)?;
                    println!("status={}", report.status);
                    println!("slot_set_digest={}", report.slot_set_digest);
                    println!("mismatch_categories={}", report.mismatch_categories.join(","));
                    println!("out={}", out.display());
                    if report.status != "PASS" {
                        std::process::exit(2);
                    }
                }
                "applied-scope-check" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/applied_scope_check.json"));
                    let report = models_applied_scope_check(&workdir, &out)?;
                    println!("status={}", report.status);
                    println!("applied_scope_digest={}", report.applied_scope_digest);
                    println!(
                        "mismatch_categories={}",
                        report.mismatch_categories.join(",")
                    );
                    println!("out={}", out.display());
                    if report.status != "PASS" {
                        std::process::exit(2);
                    }
                }
                _ => {
                    return Err(
                        "usage: ucf-ops models <verify|probe|stage|promote|rollback|list|active-check|active-evidence|eligibility|evidence-snapshot|active-review-snapshot|supported-set-review|supported-scope-reevaluate|supported-scope-execute|supported-set-apply|backend-resolution|consistency-check|applied-scope-check|recommend-rollback|shadow-ready> ..."
                            .into(),
                    )
                }
            }
        }
        "drift" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "report" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err(
                            "usage: ucf-ops drift report --run <id> --windows <n> --out <path>"
                                .into(),
                        );
                    };
                    let windows = parse_usize(&args, "--windows", 20);
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/drift_report.json"));
                    let report = drift_report(&workdir, &run_id, windows, &out)?;
                    println!("status={:?}", report.status);
                    println!("alarms={}", report.alarms.len());
                    println!("out={}", out.display());
                    if report.status != GateStatus::Pass {
                        std::process::exit(2);
                    }
                }
                _ => {
                    return Err(
                        "usage: ucf-ops drift report --run <id> --windows <n> --out <path>".into(),
                    )
                }
            }
        }
        "alerts" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "report" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err("usage: ucf-ops alerts report --run <id> --out <path>".into());
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/alerts_report.json"));
                    let report = alerts_report(&workdir, &run_id, &out)?;
                    println!("active_alerts={}", report.active_alerts.len());
                    println!("last_triggers={}", report.last_triggers.len());
                    println!("out={}", out.display());
                }
                _ => return Err("usage: ucf-ops alerts report --run <id> --out <path>".into()),
            }
        }
        "world" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "shadow-report" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err("usage: ucf-ops world shadow-report --run <id> --windows <n> --out <path>".into());
                    };
                    let windows = parse_usize(&args, "--windows", 10);
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/world_shadow_report.json"));
                    let report = world_shadow_report(&workdir, &run_id, windows, &out)?;
                    println!("status={:?}", report.status);
                    println!("windows={}", report.window_count);
                    println!("out={}", out.display());
                    if report.status != GateStatus::Pass {
                        std::process::exit(2);
                    }
                }
                "parity-report" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err("usage: ucf-ops world parity-report --run <id> --out <path>".into());
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/world_parity_report.json"));
                    let report = world_parity_report(&workdir, &run_id, &out)?;
                    println!("parity_windows={}", report.parity_records.len());
                    println!("eligibility_entries={}", report.eligibility.len());
                    println!("out={}", out.display());
                }
                _ => {
                    return Err(
                        "usage: ucf-ops world <shadow-report|parity-report> --run <id> [--windows <n>] --out <path>"
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

        "logs" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "prove" => {
                    let digest = arg_value(&args, "--record-digest")
                        .ok_or("usage: ucf-ops logs prove --record-digest <hex> [--out <path>] [--segment-size <n>]")?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/proof.json"));
                    let segment_size = parse_usize(&args, "--segment-size", 1024).max(1);
                    let proof = logs_prove(&workdir, &digest, &out, segment_size)?;
                    println!(
                        "segment={} leaf_index={} out={}",
                        proof.segment_id.segment_index,
                        proof.leaf_index,
                        out.display()
                    );
                }
                "verify-proof" => {
                    let proof = arg_value(&args, "--proof")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/proof.json"));
                    logs_verify_proof(&proof)?;
                    println!("proof=ok path={}", proof.display());
                }
                _ => return Err("usage: ucf-ops logs <prove|verify-proof> ...".into()),
            }
        }
        "attest" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "keys" => {
                    let force = has_flag(&args, "--force");
                    attest_keys_generate(&workdir, force)?;
                    println!("attestation_keys=ok");
                }
                "run" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err("usage: ucf-ops attest run --run <id> --out <path>".into());
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/run_cert_{run_id}.json")));
                    let cert = attest_run(&workdir, &run_id, &out)?;
                    println!("run_id={}", cert.run_id);
                    println!("certificate_digest={}", cert.certificate_digest);
                    println!("out={}", out.display());
                }
                "verify" => {
                    let cert = arg_value(&args, "--cert")
                        .map(PathBuf::from)
                        .ok_or("usage: ucf-ops attest verify --cert <path> --ess <path>")?;
                    let ess = arg_value(&args, "--ess")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| workdir.join("ess").join("ess_fixture.json"));
                    let report = attest_verify(&workdir, &cert, &ess)?;
                    println!("pass={}", report.pass);
                    for reason in report.reasons {
                        println!("reason={reason}");
                    }
                    if !report.pass {
                        std::process::exit(2);
                    }
                }
                "bundle" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err("usage: ucf-ops attest bundle --run <id> --out <path>".into());
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/bundle_{run_id}.zip")));
                    let report = attest_bundle(&workdir, &run_id, &out)?;
                    println!("run_id={}", report.run_id);
                    println!("entries={}", report.entries.len());
                    println!("out={}", report.out);
                }
                _ => return Err("usage: ucf-ops attest <keys|run|verify|bundle> ...".into()),
            }
        }
        "repro" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "pack" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err(
                            "usage: ucf-ops repro pack --run <id> --out <path> [--range last]"
                                .into(),
                        );
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/repro_{run_id}.zip")));
                    let report = repro_pack(&workdir, &run_id, &out)?;
                    println!("run_id={}", report.run_id);
                    println!("pack_id={}", report.pack_id);
                    println!("entries={}", report.entry_count);
                    println!("out={}", report.out);
                }
                "verify" => {
                    let pack = arg_value(&args, "--pack")
                        .map(PathBuf::from)
                        .ok_or("usage: ucf-ops repro verify --pack <zip> --out <path>")?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/repro_verify.json"));
                    let report = repro_verify(&pack, &out)?;
                    println!("status={}", if report.pass { "PASS" } else { "FAIL" });
                    println!("out={}", out.display());
                    if !report.pass {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops repro <pack|verify> ...".into()),
            }
        }
        "airgap" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "export" => {
                    let kind = args.get(3).map(String::as_str).unwrap_or("help");
                    match kind {
                        "policies" => {
                            let pack = arg_value(&args, "--pack").map(PathBuf::from).ok_or("usage: ucf-ops airgap export policies --pack <path> [--overlay <path>] --out <zip>")?;
                            let overlay = arg_value(&args, "--overlay").map(PathBuf::from);
                            let out = arg_value(&args, "--out").map(PathBuf::from).ok_or("usage: ucf-ops airgap export policies --pack <path> [--overlay <path>] --out <zip>")?;
                            let report =
                                airgap_export_policies(&workdir, &pack, overlay.as_deref(), &out)?;
                            println!("digest={}", report.overall_digest);
                            println!("out={}", report.out);
                        }
                        "models" => {
                            let slot = arg_value(&args, "--slot").ok_or("usage: ucf-ops airgap export models --slot <slot> --hash <hash> --out <zip>")?;
                            let hash = arg_value(&args, "--hash").ok_or("usage: ucf-ops airgap export models --slot <slot> --hash <hash> --out <zip>")?;
                            let out = arg_value(&args, "--out").map(PathBuf::from).ok_or("usage: ucf-ops airgap export models --slot <slot> --hash <hash> --out <zip>")?;
                            let report = airgap_export_models(&workdir, &slot, &hash, &out)?;
                            println!("digest={}", report.overall_digest);
                            println!("out={}", report.out);
                        }
                        "run-cert" => {
                            let run_id = arg_value(&args, "--run").ok_or(
                                "usage: ucf-ops airgap export run-cert --run <id> --out <zip>",
                            )?;
                            let out = arg_value(&args, "--out").map(PathBuf::from).ok_or(
                                "usage: ucf-ops airgap export run-cert --run <id> --out <zip>",
                            )?;
                            let report = airgap_export_run_cert(&workdir, &run_id, &out)?;
                            println!("digest={}", report.overall_digest);
                            println!("out={}", report.out);
                        }
                        "repro" => {
                            let run_id = arg_value(&args, "--run").ok_or(
                                "usage: ucf-ops airgap export repro --run <id> --out <zip>",
                            )?;
                            let out = arg_value(&args, "--out").map(PathBuf::from).ok_or(
                                "usage: ucf-ops airgap export repro --run <id> --out <zip>",
                            )?;
                            let report = airgap_export_repro(&workdir, &run_id, &out)?;
                            println!("digest={}", report.overall_digest);
                            println!("out={}", report.out);
                        }
                        _ => {
                            return Err(
                                "usage: ucf-ops airgap export <policies|models|run-cert|repro> ..."
                                    .into(),
                            )
                        }
                    }
                }
                "import" => {
                    let kind = args.get(3).map(String::as_str).unwrap_or("help");
                    let artifact_type = match kind {
                        "policies" => AirgapArtifactType::Policies,
                        "models" => AirgapArtifactType::Models,
                        "run-cert" => AirgapArtifactType::RunCert,
                        "repro" => AirgapArtifactType::Repro,
                        _ => return Err("usage: ucf-ops airgap import <policies|models|run-cert|repro> --in <zip> --out <json> [--mode staging|promoted]".into()),
                    };
                    let input = arg_value(&args, "--in")
                        .map(PathBuf::from)
                        .ok_or("usage: ucf-ops airgap import <type> --in <zip> --out <json>")?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/airgap_import_report.json"));
                    let mode = match arg_value(&args, "--mode")
                        .unwrap_or_else(|| "staging".to_string())
                        .as_str()
                    {
                        "staging" => AirgapImportMode::Staging,
                        "promoted" => AirgapImportMode::Promoted,
                        _ => return Err("--mode must be staging|promoted".into()),
                    };
                    let pack = arg_value(&args, "--pack")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("policies/packs/base_v1"));
                    let overlay = arg_value(&args, "--overlay")
                        .map(PathBuf::from)
                        .or_else(|| Some(PathBuf::from("policies/packs/overlays/test")));
                    let strict_signer = !has_flag(&args, "--allow-untrusted");
                    let report = airgap_import(
                        &workdir,
                        &AirgapImportArgs {
                            artifact_type,
                            input,
                            out: out.clone(),
                            mode,
                            policy_pack: pack,
                            policy_overlay: overlay,
                            strict_signer,
                        },
                    )?;
                    println!("pass={}", report.pass);
                    println!("out={}", out.display());
                    if !report.pass {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops airgap <export|import> ...".into()),
            }
        }
        "bugkit" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "build" => {
                    let Some(run_id) = arg_value(&args, "--run") else {
                        return Err("usage: ucf-ops bugkit build --run <id> --out <path> [--include_payload] [--include_weights] [--max-bytes <n>]".into());
                    };
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/bugkit_{run_id}.zip")));
                    let max_bytes = parse_u64(&args, "--max-bytes", 50 * 1024 * 1024);
                    let report = ucf_ops::bugkit_build(
                        &workdir,
                        &run_id,
                        &out,
                        &BugKitBuildArgs {
                            include_payload: has_flag(&args, "--include_payload"),
                            include_weights: has_flag(&args, "--include_weights"),
                            max_bytes,
                        },
                    )?;
                    println!("run_id={}", report.run_id);
                    println!("out={}", report.out);
                    println!("total_bytes={}", report.total_bytes);
                    println!("files={}", report.file_count);
                    for warning in report.warnings {
                        println!("warning={warning}");
                    }
                }
                _ => {
                    return Err("usage: ucf-ops bugkit build --run <id> --out <path> [--include_payload] [--include_weights] [--max-bytes <n>]".into())
                }
            }
        }
        "exports" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "normalize-check" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/export_normalize_check.json"));
                    let report = exports_normalize_check(&workdir, &out)?;
                    println!("pass={}", report.pass);
                    println!("mismatch_count={}", report.mismatch_count);
                    println!("out={}", out.display());
                    if !report.pass {
                        std::process::exit(2);
                    }
                }
                "roundtrip-check" => {
                    let input = arg_value(&args, "--in").map(PathBuf::from).ok_or(
                        "usage: ucf-ops exports roundtrip-check --in <bundle> --out <path>",
                    )?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/export_roundtrip_check.json"));
                    let report = exports_roundtrip_check(&input, &out)?;
                    println!("bundle_kind={:?}", report.bundle_kind);
                    println!(
                        "context_match={}",
                        matches!(
                            report.context_match_status,
                            ucf_ops::BundleRoundTripMatchStatusV1::Match
                        )
                    );
                    println!(
                        "scope_match={}",
                        matches!(
                            report.scope_match_status,
                            ucf_ops::BundleRoundTripMatchStatusV1::Match
                        )
                    );
                    println!(
                        "policy_match={}",
                        matches!(
                            report.policy_match_status,
                            ucf_ops::BundleRoundTripMatchStatusV1::Match
                        )
                    );
                    println!(
                        "manifest_match={}",
                        matches!(
                            report.manifest_match_status,
                            ucf_ops::BundleRoundTripMatchStatusV1::Match
                        )
                    );
                    if let Some(code) = report.mismatch_codes.first() {
                        println!("main_mismatch_code={code}");
                    }
                    println!("out={}", out.display());
                    if matches!(
                        report.overall_status,
                        ucf_ops::BundleRoundTripOverallStatusV1::Fail
                    ) {
                        std::process::exit(2);
                    }
                }
                "bundle-spine-check" => {
                    let input = arg_value(&args, "--in").map(PathBuf::from).ok_or(
                        "usage: ucf-ops exports bundle-spine-check --in <bundle> --out <path>",
                    )?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/bundle_spine_check.json"));
                    let report = exports_bundle_spine_check(&input, &out)?;
                    println!("bundle_kind={:?}", report.bundle_kind);
                    println!(
                        "applied_scope_digest={}",
                        report.spine.applied_supported_set_digest_prefix
                    );
                    println!(
                        "governance_coherent={}",
                        !report
                            .mismatch_codes
                            .iter()
                            .any(|c| c == "BUNDLE_SPINE_GOVERNANCE_MISMATCH")
                    );
                    println!(
                        "readiness_coherent={}",
                        !report
                            .mismatch_codes
                            .iter()
                            .any(|c| c == "BUNDLE_SPINE_READINESS_MISMATCH")
                    );
                    if let Some(code) = report.mismatch_codes.first() {
                        println!("main_mismatch_code={code}");
                    }
                    println!("out={}", out.display());
                    if !report.pass {
                        std::process::exit(2);
                    }
                }
                _ => {
                    return Err(
                        "usage: ucf-ops exports <normalize-check|roundtrip-check|bundle-spine-check> ...".into(),
                    )
                }
            }
        }
        "preflight" => {
            let bundle = arg_value(&args, "--bundle")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/preflight.json"));
            let report = preflight(&bundle, &out)?;
            println!("bundle={}", report.bundle);
            println!("overall={:?}", report.overall);
            println!("out={}", out.display());
            if report.exit_code != 0 {
                std::process::exit(report.exit_code);
            }
        }
        "operator" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "report" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/operator_report.json"));
                    let report = operator_report(
                        &workdir,
                        &OperatorReportArgs {
                            run_id: arg_value(&args, "--run"),
                            latest: has_flag(&args, "--latest"),
                        },
                        &out,
                    )?;
                    println!("out={}", out.display());
                    println!("overall_status={:?}", report.overall_status);
                    println!("{}", operator_report_text(&report));
                    if has_flag(&args, "--text") {
                        println!("text_mode=true");
                    }
                }
                "signoff" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/operator_signoff.json"));
                    let profile = std::env::var("UCF_PROFILE").unwrap_or_else(|_| "test".to_string());
                    let decision = operator_signoff(
                        &workdir,
                        &OperatorSignoffArgs {
                            run_id: arg_value(&args, "--run"),
                            latest: has_flag(&args, "--latest"),
                            profile,
                        },
                        &out,
                    )?;
                    println!("out={}", out.display());
                    println!("decision={:?}", decision.decision);
                    println!("gate_v3_digest={}", decision.gate_report_digests.v3);
                    if !decision.reasons.is_empty() {
                        println!("reasons={}", decision.reasons.join(","));
                    }
                    println!("{}", operator_signoff_text(&decision));
                    if has_flag(&args, "--text") {
                        println!("text_mode=true");
                    }
                }
                "review-packet" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/operator_review_packet.json"));
                    let packet = operator_review_packet(
                        &workdir,
                        &OperatorReviewPacketArgs {
                            run_id: arg_value(&args, "--run"),
                            latest: has_flag(&args, "--latest"),
                        },
                        &out,
                    )?;
                    println!("out={}", out.display());
                    println!("review_stage={:?}", packet.review_stage);
                    if !packet.blocking_codes.is_empty() {
                        println!("blocking_codes={}", packet.blocking_codes.join(","));
                    }
                    if !packet.remediation_codes.is_empty() {
                        println!("remediation_codes={}", packet.remediation_codes.join(","));
                    }
                    println!("{}", operator_review_packet_text(&packet));
                    if has_flag(&args, "--text") {
                        println!("text_mode=true");
                    }
                }
                "review-truth-check" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/review_truth_check.json"));
                    let report = review_truth_check(&workdir, &out)?;
                    println!("out={}", out.display());
                    println!("status={:?}", report.status);
                    if !report.mismatch_categories.is_empty() {
                        println!("mismatch_categories={}", report.mismatch_categories.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>().join(","));
                    }
                    if !report.remediation_codes.is_empty() {
                        println!("remediation_codes={}", report.remediation_codes.join(","));
                    }
                    if !matches!(report.status, ucf_ops::ReviewTruthCheckStatusV1::Pass) {
                        std::process::exit(2);
                    }
                }
                "workflow" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/operator_workflow_chain.json"));
                    let chain = operator_workflow_chain(
                        &workdir,
                        &OperatorWorkflowArgs {
                            run_id: arg_value(&args, "--run"),
                            latest: has_flag(&args, "--latest"),
                        },
                        &out,
                    )?;
                    println!("out={}", out.display());
                    println!("workflow_stage={:?}", chain.workflow_stage);
                    if !chain.blocking_codes.is_empty() {
                        println!("blocking_codes={}", chain.blocking_codes.join(","));
                    }
                    println!("repro_ready={}", chain.export_targets.repro_ready);
                    println!("bugkit_ready={}", chain.export_targets.bugkit_ready);
                    if has_flag(&args, "--text") {
                        println!("{}", operator_workflow_chain_text(&chain));
                        println!("text_mode=true");
                    }
                    if !matches!(
                        chain.workflow_stage,
                        ucf_ops::OperatorWorkflowStageV2::WorkflowExportReady
                            | ucf_ops::OperatorWorkflowStageV2::WorkflowReviewReady
                    ) {
                        std::process::exit(2);
                    }
                }

                "export-chain-check" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/operator_export_chain_check.json"));
                    let chain = operator_export_chain_check(&workdir, &out)?;
                    println!("out={}", out.display());
                    println!("authority_chain_status={:?}", chain.authority_chain_status);
                    if !chain.blocking_codes.is_empty() {
                        println!("blocking_codes={}", chain.blocking_codes.join(","));
                    }
                    if !matches!(
                        chain.authority_chain_status,
                        ucf_ops::OperatorExportAuthorityChainStatusV1::Pass
                    ) {
                        std::process::exit(2);
                    }
                }
                "roundtrip-chain-check" => {
                    let bundle = arg_value(&args, "--bundle")
                        .map(PathBuf::from)
                        .ok_or_else(|| {
                            "usage: ucf-ops operator roundtrip-chain-check --bundle <path> --out ./out/operator_roundtrip_chain_check.json".to_string()
                        })?;
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/operator_roundtrip_chain_check.json"));
                    let chain = operator_roundtrip_chain_check(&workdir, &bundle, &out)?;
                    println!("out={}", out.display());
                    println!("roundtrip_status={:?}", chain.roundtrip_status);
                    if !chain.blocking_codes.is_empty() {
                        println!("blocking_codes={}", chain.blocking_codes.join(","));
                        println!("primary_mismatch_code={}", chain.blocking_codes[0]);
                    }
                    if !matches!(
                        chain.roundtrip_status,
                        ucf_ops::CanonicalRoundTripChainStatusV1::Pass
                    ) {
                        std::process::exit(2);
                    }
                }
                _ => {
                    return Err(
                        "usage: ucf-ops operator <report|signoff|review-packet|review-truth-check|workflow|export-chain-check|roundtrip-chain-check> [--run <id>] [--latest] [--text] [--bundle <path>] [--out <path>]".into(),
                    )
                }
            }
        }
        "readiness-spine-check" => {
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/readiness_spine_check.json"));
            let report = readiness_spine_check(&workdir, &out)?;
            println!("out={}", out.display());
            println!("status={:?}", report.status);
            if !report.mismatch_categories.is_empty() {
                println!(
                    "mismatch_categories={}",
                    report
                        .mismatch_categories
                        .iter()
                        .map(|v| format!("{:?}", v))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
            if !matches!(report.status, ucf_ops::ReadinessSpineCheckStatusV1::Pass) {
                std::process::exit(2);
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
        "v1" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "smoke" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v1_smoke_report.json"));
                    let report = v1_smoke(&workdir, &out, has_flag(&args, "--shadow"))?;
                    println!("schema_version={}", report.schema_version);
                    for check in &report.checks {
                        println!(
                            "check={} status={:?} detail={}",
                            check.name, check.status, check.detail
                        );
                    }
                    println!("out={}", out.display());
                    if report.checks.iter().any(|c| c.status == GateStatus::Fail) {
                        std::process::exit(2);
                    }
                }
                "gate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v1_gate_report.json"));
                    let report = ucf_ops::v1_gate(&workdir, &out)?;
                    println!("overall={:?}", report.overall_status);
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, ucf_ops::V1GateOverallStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops v1 <smoke|gate> [--shadow] [--out <path>]".into()),
            }
        }
        "v2" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "gate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v2_gate_report.json"));
                    let report = v2_gate(&workdir, &out)?;
                    println!("overall={:?}", report.overall_status);
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, V2GateOverallStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops v2 gate [--out <path>]".into()),
            }
        }
        "v3" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "gate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v3_gate_report.json"));
                    let report = v3_gate(&workdir, &out)?;
                    println!("overall={:?}", report.overall_status);
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, V3GateOverallStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops v3 gate [--out <path>]".into()),
            }
        }
        "v4" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "gate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v4_gate_report.json"));
                    let report = v4_gate(&workdir, &out)?;
                    println!("overall={:?}", report.overall_status);
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, V4GateOverallStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops v4 gate [--out <path>]".into()),
            }
        }
        "governance-entry-check" => {
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/governance_entry_check.json"));
            let report = governance_entry_check(&workdir, &out)?;
            println!("status={:?}", report.status);
            println!("authority_digest_prefix={}", report.authority_digest_prefix);
            println!("out={}", out.display());
            if !matches!(report.status, GovernanceEntryCheckStatusV1::Pass) {
                std::process::exit(2);
            }
        }

        "governance-surfaces-check" => {
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/governance_surfaces_check.json"));
            let report = governance_surfaces_check(&workdir, &out)?;
            println!("status={}", report.status);
            println!("summary_code={}", report.summary_code);
            if let Some(surfaces) = report.governance_primary_surfaces.as_ref() {
                println!(
                    "governance_surfaces_digest={}",
                    surfaces.governance_surfaces_digest
                );
            }
            println!("out={}", out.display());
            if report.status != "PASS" {
                std::process::exit(2);
            }
        }

        "v5" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "gate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v5_gate_report.json"));
                    let report = v5_gate(&workdir, &out)?;
                    println!("overall={:?}", report.overall_status);
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, V5GateOverallStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops v5 gate [--out <path>]".into()),
            }
        }
        "v6" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "gate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v6_gate_report.json"));
                    let report = v6_gate(&workdir, &out)?;
                    println!("overall={:?}", report.overall_status);
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, V6GateOverallStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops v6 gate [--out <path>]".into()),
            }
        }
        "v7" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "gate" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v7_gate_report.json"));
                    let report = v7_gate(&workdir, &out)?;
                    println!("overall={:?}", report.overall_status);
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, V7GateOverallStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops v7 gate [--out <path>]".into()),
            }
        }
        "remediation-consistency-check" => {
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/remediation_consistency.json"));
            let report = remediation_consistency_check(&out)?;
            println!(
                "checked={} failed={} status={:?}",
                report.summary.total_conditions, report.summary.fail_count, report.summary.status
            );
            if !report.summary.top_mismatch_categories.is_empty() {
                println!(
                    "mismatch_categories={}",
                    report.summary.top_mismatch_categories.join(",")
                );
            }
            println!("out={}", out.display());
            if report.summary.fail_count > 0 {
                std::process::exit(2);
            }
        }
        "remediation-interop-check" => {
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/remediation_interop_check.json"));
            let report = remediation_interop_check(&out)?;
            println!(
                "conditions_checked={} mismatches_found={}",
                report.conditions_checked, report.mismatches_found
            );
            if !report.top_mismatch_categories.is_empty() {
                println!(
                    "top_mismatch_categories={}",
                    report.top_mismatch_categories.join(",")
                );
            }
            println!("out={}", out.display());
            if report.mismatches_found > 0 {
                let only_missing_surface = !report.top_mismatch_categories.is_empty()
                    && report
                        .top_mismatch_categories
                        .iter()
                        .all(|c| c.starts_with("MISSING_SURFACE:"));
                if only_missing_surface {
                    println!(
                        "remediation_interop_check=skip reason=required_surfaces_missing_in_bounded_smoke"
                    );
                } else {
                    std::process::exit(2);
                }
            }
        }
        "remediation-spine-check" => {
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/remediation_spine_check.json"));
            let report = remediation_spine_check(&out)?;
            println!(
                "conditions_checked={} mismatches_found={}",
                report.conditions_checked, report.mismatches_found
            );
            if !report.top_mismatch_categories.is_empty() {
                println!(
                    "top_mismatch_categories={}",
                    report.top_mismatch_categories.join(",")
                );
            }
            println!("out={}", out.display());
            if report.mismatches_found > 0 {
                std::process::exit(2);
            }
        }
        "scope" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "authority-check" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/scope_authority_check.json"));
                    let report = scope_authority_check(&workdir, &out)?;
                    println!("status={:?}", report.status);
                    println!("out={}", out.display());
                    if !matches!(report.status, ucf_ops::ScopeAuthorityOverallStatusV1::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops scope authority-check [--out <path>]".into()),
            }
        }

        "interop" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "consistency-matrix" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/interop_consistency_matrix.json"));
                    let report = interop_consistency_matrix(&workdir, &out)?;
                    println!("status={:?}", report.summary.overall_status);
                    if !report.match_rules.mismatch_categories.is_empty() {
                        println!(
                            "mismatch_categories={}",
                            report
                                .match_rules
                                .mismatch_categories
                                .iter()
                                .map(|v| format!("{:?}", v))
                                .collect::<Vec<_>>()
                                .join(",")
                        );
                    }
                    println!("out={}", out.display());
                    if !matches!(
                        report.summary.overall_status,
                        ucf_ops::InteropOverallStatusV1::Pass
                    ) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops interop consistency-matrix [--out <path>]".into()),
            }
        }
        "v0" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "gate" => {
                    let scenario = arg_value(&args, "--scenario")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("fixtures/e2e/v0_flow_a.json"));
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/v0_gate_report.json"));
                    let report = v0_gate(&workdir, &scenario, &out)?;
                    println!("overall={:?}", report.overall_status);
                    println!("schema_version={}", report.schema_version);
                    println!("out={}", out.display());
                    if !matches!(report.overall_status, ucf_ops::V0GateOverallStatus::Pass) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops v0 gate [--scenario <path>] [--out <path>]".into()),
            }
        }
        "goldens" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            let scenario = arg_value(&args, "--scenario").unwrap_or_else(|| "golden_a".to_string());
            let os = arg_value(&args, "--os").unwrap_or_else(|| std::env::consts::OS.to_string());
            let out_root = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("fixtures/goldens"));
            let workdir_root = arg_value(&args, "--workdir-root")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./.ucf_goldens"));
            match sub {
                "generate" => {
                    let out = goldens_generate(&GoldenGenerateArgs {
                        scenario,
                        os,
                        out_root,
                        workdir_root,
                    })?;
                    println!("out={}", out.display());
                }
                "verify" => {
                    if has_flag(&args, "--all") {
                        let scenarios = ["golden_a", "golden_b", "golden_c", "golden_d"];
                        let mut reports = Vec::new();
                        for scenario_id in scenarios {
                            let report = goldens_verify_detailed(&GoldenVerifyArgs {
                                scenario: scenario_id.to_string(),
                                os: os.clone(),
                                out_root: out_root.clone(),
                                workdir_root: workdir_root.clone(),
                            })?;
                            println!(
                                "scenario={} status={:?} refresh_candidate={} heuristic={:?}",
                                report.scenario, report.status, report.refresh_candidate, report.heuristic
                            );
                            reports.push(report);
                        }
                        reports.sort_by(|a, b| a.scenario.cmp(&b.scenario));
                        let overall = if reports.iter().all(|r| r.status == GateStatus::Pass) {
                            GateStatus::Pass
                        } else {
                            GateStatus::Fail
                        };
                        let bundle = GoldenVerifyReport {
                            os: os.clone(),
                            status: overall,
                            scenarios: reports,
                        };
                        if let Some(path) = arg_value(&args, "--report-out").map(PathBuf::from) {
                            if let Some(parent) = path.parent() {
                                std::fs::create_dir_all(parent)?;
                            }
                            std::fs::write(&path, serde_json::to_string_pretty(&bundle)?)?;
                            println!("report={}", path.display());
                        }
                        if overall != GateStatus::Pass {
                            std::process::exit(2);
                        }
                    } else {
                        goldens_verify(&GoldenVerifyArgs {
                            scenario,
                            os,
                            out_root,
                            workdir_root,
                        })?;
                        println!("status=PASS");
                    }
                }
                "update" => {
                    let out = goldens_update(&GoldenGenerateArgs {
                        scenario,
                        os,
                        out_root,
                        workdir_root,
                    })?;
                    println!("out={}", out.display());
                }
                _ => {
                    return Err("usage: ucf-ops goldens <generate|verify|update> --scenario <id> [--all] [--report-out <path>] [--os <linux|windows|macos>] [--out fixtures/goldens] [--workdir-root ./.ucf_goldens]".into())
                }
            }
        }
        "nightly" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "summarize" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/nightly_summary.json"));
                    let docs = arg_value(&args, "--docs")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/docs_lint_report.json"));
                    let gate = arg_value(&args, "--gate")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/gate_report.json"));
                    let adversarial = arg_value(&args, "--adversarial")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/adversarial_report.json"));
                    let goldens = arg_value(&args, "--goldens")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/goldens_report.json"));
                    let drift = arg_value(&args, "--drift").map(PathBuf::from);
                    let report = nightly_summarize(&NightlySummarizeArgs {
                        docs_lint_report: docs,
                        gate_report: gate,
                        adversarial_report: adversarial,
                        goldens_report: goldens,
                        drift_report: drift,
                        out: out.clone(),
                    })?;
                    println!("status={:?}", report.status);
                    println!("golden_refresh_suggested={}", report.golden_refresh_suggested);
                    println!("out={}", out.display());
                    if report.status != ucf_ops::NightlyOverallStatus::Pass {
                        std::process::exit(2);
                    }
                }
                _ => {
                    return Err("usage: ucf-ops nightly summarize [--docs <path>] [--gate <path>] [--adversarial <path>] [--goldens <path>] [--drift <path>] [--out <path>]".into())
                }
            }
        }
        "dev" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "loop" => {
                    let profile = arg_value(&args, "--profile").unwrap_or_else(|| "dev".to_string());
                    let scenario = arg_value(&args, "--scenario").unwrap_or_else(|| "golden_a".to_string());
                    let ticks = parse_u64(&args, "--ticks", 32);
                    let out_dir = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/dev_loop"));
                    let report = ucf_ops::dev_loop(
                        &workdir,
                        &DevLoopArgs {
                            profile,
                            scenario,
                            ticks,
                            out_dir: out_dir.clone(),
                            run_tests: !has_flag(&args, "--no-tests"),
                        },
                    )?;
                    println!("out={}", out_dir.display());
                    for step in report.steps {
                        println!("step={} status={:?} detail={}", step.step, step.status, step.detail);
                    }
                    for action in report.next_actions {
                        println!("next={action}");
                    }
                }
                _ => return Err("usage: ucf-ops dev loop [--profile <dev|test|prod>] [--scenario <id>] [--ticks <n>] [--out <path>] [--no-tests]".into()),
            }
        }
        "gateway" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "threat-test" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/gateway_threat.json"));
                    let report = gateway_threat_test(&out)?;
                    println!("out={}", out.display());
                    println!("ok={}", report.ok);
                    println!("abuse_log_total={}", report.abuse_log_total);
                }
                _ => return Err("usage: ucf-ops gateway threat-test --out <path>".into()),
            }
        }
        "troubleshoot" => {
            let Some(run_id) = arg_value(&args, "--run") else {
                return Err("usage: ucf-ops troubleshoot --run <id> --out <path>".into());
            };
            let out = arg_value(&args, "--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./out/troubleshoot.json"));
            let report = troubleshoot(&workdir, &run_id, &out)?;
            println!("run_id={}", report.run_id);
            println!("out={}", out.display());
            for issue in report.issues {
                println!(
                    "issue={} severity={} next={}",
                    issue.source, issue.severity, issue.next_command
                );
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
                "build-rc" => {
                    let Some(version) = arg_value(&args, "--version") else {
                        return Err("usage: ucf-ops release build-rc --version <vX.Y-rcZ> [--profile prod] [--out ./out/rc] [--fast]".into());
                    };
                    let profile =
                        arg_value(&args, "--profile").unwrap_or_else(|| "prod".to_string());
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/rc"));
                    let report = release_build_rc(
                        &workdir,
                        &ReleaseBuildRcArgs {
                            version,
                            profile,
                            out,
                            fast: has_flag(&args, "--fast"),
                        },
                    )?;
                    println!("version={}", report.version);
                    println!("rc_digest={}", report.rc_digest);
                    println!("rc_zip={}", report.rc_zip);
                }
                _ => return Err("usage: ucf-ops release <signoff|rc1-gate|build-rc> ...".into()),
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

        "soak" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "run" => {
                    let duration = arg_value(&args, "--duration").unwrap_or_else(|| "2h".to_string());
                    let duration_secs = parse_duration_secs(&duration)?;
                    let scenario = arg_value(&args, "--scenario").unwrap_or_else(|| "golden_a".to_string());
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from(format!("./out/soak_{}", std::process::id())));
                    let health_poll_secs = parse_u64(&args, "--health-poll", 5);
                    let memory_sample_secs = parse_u64(&args, "--memory-sample", 60);
                    let inject = args
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, v)| if v == "--inject" { args.get(idx + 1) } else { None })
                        .map(|v| parse_inject(v))
                        .collect::<Result<Vec<_>, _>>()?;
                    let report = soak_run(
                        &workdir,
                        &SoakRunArgs {
                            duration_secs,
                            scenario,
                            out: out.clone(),
                            health_poll_secs,
                            memory_sample_secs,
                            inject,
                            postmortem: has_flag(&args, "--postmortem"),
                        },
                    )?;
                    println!("run_id={}", report.run_id);
                    println!("status={:?}", report.status);
                    println!("out={}", out.join("soak_report.json").display());
                    if let Some(bundle) = report.postmortem_bundle {
                        println!("postmortem_bundle={bundle}");
                    }
                    if matches!(report.status, ucf_ops::SoakStatus::Fail) {
                        std::process::exit(2);
                    }
                }
                _ => return Err("usage: ucf-ops soak run [--duration 2h] [--scenario golden_a] [--out <dir>] [--inject <kind[:target]@t=N>]... [--postmortem]".into()),
            }
        }
        "strict" => {
            let sub = args.get(2).map(String::as_str).unwrap_or("help");
            match sub {
                "check" => {
                    if has_flag(&args, "--strict") {
                        std::env::set_var("UCF_STRICT_MODE", "1");
                    }
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/strict_check.json"));
                    let report = strict_check(&workdir, true, &out)?;
                    println!("out={}", out.display());
                    println!("strict_mode_enabled={}", report.strict_mode_enabled);
                    println!("overall={}", if report.ok { "PASS" } else { "FAIL" });
                }
                "explain" => {
                    let out = arg_value(&args, "--out")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("./out/strict_explain.json"));
                    let explain = strict_explain(
                        Path::new("./out"),
                        &StrictEvidenceContextV1 {
                            run_id: arg_value(&args, "--run"),
                            latest: has_flag(&args, "--latest"),
                            strict_required: true,
                            expected_policy_graph_digest_prefix: None,
                            expected_manifest_digest_prefix: None,
                            expected_supported_slot_set_digest_prefix: None,
                        },
                    );
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&out, serde_json::to_string_pretty(&explain)?)?;
                    println!("out={}", out.display());
                    println!("strict_status={:?}", explain.snapshot.strict_status);
                    println!("reason={}", explain.operator_blocking_view.primary_reason_code.unwrap_or_else(|| "none".to_string()));
                }
                _ => return Err("usage: ucf-ops strict <check|explain> [--strict] [--run <id>] [--latest] [--out <path>]".into()),
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
                "usage: ucf-ops <bringup|diag|health|diagnostics|export-bugreport|verify-bugreport|replay-bugreport|replay|metrics-snapshot|explain-tick|metrics|models|security|attest|repro|exports|readiness-gate|preflight|goldens|nightly|dev|troubleshoot|adversarial-run|out|release|bench|runs|status|strict|ess|ebm|drift|alerts|operator|policy|portability|spec|change-impact|soak|governance-surfaces-check|governance-entry-check|remediation-consistency-check|remediation-interop-check|remediation-spine-check|interop|v0|v1|v2|v3|v4|v5|v6|v7|version> [--workdir <path>] [--bundle <path>]"
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

fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    arg_value(args, name)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}
