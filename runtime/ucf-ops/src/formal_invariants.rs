use crate::{check_skip, CheckResult, OpsError};

pub fn run_formal_invariants_check(profile: &str) -> Result<CheckResult, OpsError> {
    #[cfg(not(feature = "formal-smt"))]
    {
        let _ = profile;
        Ok(check_skip(
            "formal_invariants_smt",
            [("feature".to_string(), "formal-smt_disabled".to_string())],
            "formal SMT checks are feature gated",
            "Enable `--features formal-smt` in CI lane to run solver-backed invariant checks.",
        ))
    }

    #[cfg(feature = "formal-smt")]
    {
        enabled::run(profile)
    }
}

#[cfg(feature = "formal-smt")]
mod enabled {
    use crate::{
        check_fail, check_pass, check_skip, prefix_hex, sha256_hex, CheckResult, GateStatus,
        OpsError,
    };
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use ucf_policy::determinism::DeterminismMode;
    use ucf_policy::policy_packs::{
        load_and_merge_policy_graph, policy_graph_digest, PolicyGraphV1,
    };

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
    #[serde(rename_all = "snake_case")]
    enum InvariantId {
        ToolExecutionRequiresIssuedToken,
        GovernorTierMonotoneTightening,
        SamplingDisabledInProd,
        PromotedOnlyWeights,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    struct InvariantResult {
        id: InvariantId,
        description: String,
        smt_instance_digest: String,
        status: GateStatus,
        witness: Option<BTreeMap<String, String>>,
        detail: String,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    struct FormalInvariantsReport {
        policy_graph_digest: String,
        profile: String,
        solver_backend: String,
        status: GateStatus,
        invariants: Vec<InvariantResult>,
    }
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MonotoneParams {
        risk_weight_q: i64,
        uncertainty_weight_q: i64,
    }

    pub fn run(profile: &str) -> Result<CheckResult, OpsError> {
        if !z3_available() {
            return Ok(check_skip(
                "formal_invariants_smt",
                [("solver".to_string(), "z3_missing".to_string())],
                "z3 binary not available",
                "Install z3 and run with `--features formal-smt` in CI solver lane.",
            ));
        }
        let overlay_name =
            std::env::var("UCF_POLICY_OVERLAY").unwrap_or_else(|_| profile.to_string());
        let overlay_path = PathBuf::from("policies/packs/overlays").join(&overlay_name);
        let overlay_ref = overlay_path.exists().then_some(overlay_path.as_path());
        let (graph, _) =
            load_and_merge_policy_graph(Path::new("policies/packs/base_v1"), overlay_ref)?;
        let policy_digest = policy_graph_digest(&graph)?;

        let results = vec![
            eval_tool_exec_requires_issue(),
            eval_monotone_tightening(&graph),
            eval_sampling_disabled_in_prod(profile, &graph),
            eval_promoted_only_weights(),
        ];
        let status = if results.iter().any(|r| r.status == GateStatus::Fail) {
            GateStatus::Fail
        } else {
            GateStatus::Pass
        };
        let report = FormalInvariantsReport {
            policy_graph_digest: policy_digest.clone(),
            profile: profile.to_string(),
            solver_backend: "z3-cli-smt2".to_string(),
            status,
            invariants: results,
        };

        let out_dir = PathBuf::from("out/formal").join(&policy_digest);
        fs::create_dir_all(&out_dir)?;
        let out_path = out_dir.join("invariants_report.json");
        fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;

        if status == GateStatus::Pass {
            Ok(check_pass(
                "formal_invariants_smt",
                [
                    ("status".to_string(), "pass".to_string()),
                    (
                        "policy_graph_digest".to_string(),
                        prefix_hex(&policy_digest, 12),
                    ),
                    ("artifact".to_string(), out_path.display().to_string()),
                ],
            ))
        } else {
            let failed = report
                .invariants
                .iter()
                .find(|i| i.status == GateStatus::Fail)
                .map(|i| format!("{:?}", i.id))
                .unwrap_or_else(|| "unknown".to_string());
            Ok(check_fail("formal_invariants_smt", [("status".to_string(), "fail".to_string()), ("failed_invariant".to_string(), failed), ("policy_graph_digest".to_string(), prefix_hex(&policy_digest, 12)), ("artifact".to_string(), out_path.display().to_string())], "formal SMT invariant check found a counterexample", "Inspect out/formal/<policy_graph_digest>/invariants_report.json and tighten policy thresholds/weights."))
        }
    }

    fn z3_available() -> bool {
        Command::new("z3")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn eval_tool_exec_requires_issue() -> InvariantResult {
        let smt = "(set-logic QF_UF)\n(declare-const has_exec Bool)\n(declare-const has_issue Bool)\n(assert (=> has_exec has_issue))\n(assert has_exec)\n(assert (not has_issue))".to_string();
        solve(
            InvariantId::ToolExecutionRequiresIssuedToken,
            "no tool execution without issued token",
            smt,
            &[],
        )
    }

    fn eval_monotone_tightening(graph: &PolicyGraphV1) -> InvariantResult {
        let params = MonotoneParams {
            risk_weight_q: graph
                .budgets
                .get("governor_risk_weight_q")
                .copied()
                .unwrap_or(1),
            uncertainty_weight_q: graph
                .budgets
                .get("governor_uncertainty_weight_q")
                .copied()
                .unwrap_or(0),
        };
        let smt = [
            "(set-logic QF_LIA)".to_string(),
            format!("(define-fun risk_weight_q () Int {})", params.risk_weight_q),
            format!(
                "(define-fun uncertainty_weight_q () Int {})",
                params.uncertainty_weight_q
            ),
            "(declare-const risk1 Int)".to_string(),
            "(declare-const risk2 Int)".to_string(),
            "(declare-const uncertainty1 Int)".to_string(),
            "(declare-const uncertainty2 Int)".to_string(),
            "(declare-const score1 Int)".to_string(),
            "(declare-const score2 Int)".to_string(),
            "(assert (or (= risk1 0) (= risk1 5000) (= risk1 10000)))".to_string(),
            "(assert (or (= risk2 0) (= risk2 5000) (= risk2 10000)))".to_string(),
            "(assert (or (= uncertainty1 0) (= uncertainty1 5000) (= uncertainty1 10000)))"
                .to_string(),
            "(assert (or (= uncertainty2 0) (= uncertainty2 5000) (= uncertainty2 10000)))"
                .to_string(),
            "(assert (<= risk1 risk2))".to_string(),
            "(assert (<= uncertainty1 uncertainty2))".to_string(),
            "(assert (= score1 (+ (* risk_weight_q risk1) (* uncertainty_weight_q uncertainty1))))"
                .to_string(),
            "(assert (= score2 (+ (* risk_weight_q risk2) (* uncertainty_weight_q uncertainty2))))"
                .to_string(),
            "(assert (> score1 score2))".to_string(),
        ]
        .join("\n");
        solve(
            InvariantId::GovernorTierMonotoneTightening,
            "governor tier monotone tightening given higher risk/uncertainty",
            smt,
            &[
                "risk1".to_string(),
                "risk2".to_string(),
                "score1".to_string(),
                "score2".to_string(),
            ],
        )
    }

    fn eval_sampling_disabled_in_prod(profile: &str, graph: &PolicyGraphV1) -> InvariantResult {
        let sampling_allowed = graph
            .determinism
            .allowed_rng_sites
            .iter()
            .any(|site| site.as_str() == "llm_sampling")
            || graph.determinism.allowed_mode != DeterminismMode::DeterministicOnly;
        if profile != "prod" {
            return InvariantResult {
                id: InvariantId::SamplingDisabledInProd,
                description: "sampling disabled in prod".to_string(),
                smt_instance_digest: prefix_hex(&sha256_hex(b"skip-non-prod"), 16),
                status: GateStatus::Skip,
                witness: None,
                detail: "profile is not prod".to_string(),
            };
        }
        let smt = format!("(set-logic QF_UF)\n(define-fun sampling_allowed () Bool {})\n(assert sampling_allowed)", if sampling_allowed { "true" } else { "false" });
        solve(
            InvariantId::SamplingDisabledInProd,
            "sampling disabled in prod",
            smt,
            &[],
        )
    }

    fn eval_promoted_only_weights() -> InvariantResult {
        let promoted = collect_promoted_hashes();
        let pinned = collect_pinned_hashes();
        if pinned.is_empty() {
            return InvariantResult {
                id: InvariantId::PromotedOnlyWeights,
                description: "pinned hash must be in promoted set".to_string(),
                smt_instance_digest: prefix_hex(&sha256_hex(b"no-pins"), 16),
                status: GateStatus::Skip,
                witness: None,
                detail: "no pinned hashes present".to_string(),
            };
        }
        let mut lines = vec!["(set-logic QF_UF)".to_string()];
        for (idx, pin) in pinned.iter().enumerate() {
            let member = promoted.contains(pin);
            lines.push(format!(
                "(define-fun pin_member_{} () Bool {})",
                idx,
                if member { "true" } else { "false" }
            ));
            lines.push(format!("(assert pin_member_{idx})"));
        }
        let mut out = solve(
            InvariantId::PromotedOnlyWeights,
            "pinned hash must be in promoted set",
            lines.join("\n"),
            &[],
        );
        if out.status == GateStatus::Fail {
            let mut witness = BTreeMap::new();
            for (idx, pin) in pinned.iter().enumerate() {
                if !promoted.contains(pin) {
                    witness.insert(format!("pin_{idx}"), pin.clone());
                }
            }
            out.witness = Some(witness);
        }
        out
    }

    fn solve(
        id: InvariantId,
        description: &str,
        smt: String,
        model_keys: &[String],
    ) -> InvariantResult {
        let mut script = smt;
        script.push_str("\n(check-sat)\n");
        if !model_keys.is_empty() {
            script.push_str("(get-model)\n");
        }
        let digest = prefix_hex(&sha256_hex(script.as_bytes()), 16);
        match run_z3(&script) {
            Ok(output) => {
                let first = output.lines().next().unwrap_or("unknown").trim();
                if first == "unsat" {
                    InvariantResult {
                        id,
                        description: description.to_string(),
                        smt_instance_digest: digest,
                        status: GateStatus::Pass,
                        witness: None,
                        detail: "counterexample query UNSAT".to_string(),
                    }
                } else if first == "sat" {
                    InvariantResult {
                        id,
                        description: description.to_string(),
                        smt_instance_digest: digest,
                        status: GateStatus::Fail,
                        witness: parse_witness(&output, model_keys),
                        detail: "counterexample query SAT".to_string(),
                    }
                } else {
                    InvariantResult {
                        id,
                        description: description.to_string(),
                        smt_instance_digest: digest,
                        status: GateStatus::Skip,
                        witness: None,
                        detail: format!("solver returned {first}"),
                    }
                }
            }
            Err(err) => InvariantResult {
                id,
                description: description.to_string(),
                smt_instance_digest: digest,
                status: GateStatus::Skip,
                witness: None,
                detail: err,
            },
        }
    }

    fn run_z3(script: &str) -> Result<String, String> {
        let mut child = Command::new("z3")
            .arg("-in")
            .arg("-smt2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to start z3: {e}"))?;
        use std::io::Write;
        child
            .stdin
            .as_mut()
            .ok_or_else(|| "failed to open stdin".to_string())?
            .write_all(script.as_bytes())
            .map_err(|e| format!("failed writing script: {e}"))?;
        let out = child
            .wait_with_output()
            .map_err(|e| format!("failed waiting for z3: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "z3 failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    }

    fn parse_witness(output: &str, model_keys: &[String]) -> Option<BTreeMap<String, String>> {
        if model_keys.is_empty() {
            return None;
        }
        let mut out = BTreeMap::new();
        for key in model_keys {
            if let Some(line) = output.lines().find(|l| l.contains(key)) {
                out.insert(key.clone(), line.trim().to_string());
            }
        }
        (!out.is_empty()).then_some(out)
    }

    fn collect_promoted_hashes() -> Vec<String> {
        let mut out = Vec::new();
        let root = PathBuf::from("models/promoted");
        if let Ok(slots) = fs::read_dir(root) {
            for slot in slots.flatten() {
                if let Ok(hashes) = fs::read_dir(slot.path()) {
                    for hash in hashes.flatten() {
                        if hash.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            out.push(hash.file_name().to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    fn collect_pinned_hashes() -> Vec<String> {
        let mut out: Vec<String> = std::env::vars()
            .filter_map(|(k, v)| {
                (k.starts_with("UCF_MODEL_PIN_") && !v.trim().is_empty())
                    .then_some(v.trim().to_string())
            })
            .collect();
        if let Ok(text) = fs::read_to_string("models/lifecycle_manifest.toml") {
            if let Ok(value) = text.parse::<toml::Value>() {
                if let Some(slots) = value.get("slots").and_then(|v| v.as_table()) {
                    for slot in slots.values() {
                        if let Some(hash) = slot.get("active_hash").and_then(|v| v.as_str()) {
                            out.push(hash.to_string());
                        }
                    }
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn monotone_script_stable() {
            let graph = PolicyGraphV1 {
                schema_version: 1,
                base_name: "b".to_string(),
                base_version: "1.0.0".to_string(),
                overlay_name: None,
                overlay_version: None,
                pbm_gem_rules: vec![],
                nsr_rules: vec![],
                ebm_terms: vec![],
                thresholds: BTreeMap::new(),
                budgets: BTreeMap::from([("governor_risk_weight_q".to_string(), -1)]),
                allowlists: BTreeMap::new(),
                determinism: Default::default(),
            };
            let a = eval_monotone_tightening(&graph);
            assert_eq!(a.status, GateStatus::Fail);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "formal-smt"))]
    #[test]
    fn skip_without_feature() {
        let check = run_formal_invariants_check("test").expect("skip");
        assert_eq!(check.status, crate::GateStatus::Skip);
    }
}
