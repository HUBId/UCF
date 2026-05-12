use std::fs;

use tempfile::tempdir;
use ucf_ops::{
    nightly_summarize, DocsLintMode, DocsLintReport, DocsLintStatus, GateStatus,
    GoldenRefreshHeuristic, GoldenVerifyReport, GoldenVerifyScenarioReport, NightlyOverallStatus,
    NightlySummarizeArgs, ReadinessGateReport, ReportFreshnessMetadata,
};

#[test]
fn nightly_summary_is_deterministic_and_actionable() {
    let dir = tempdir().expect("tempdir");
    let docs_path = dir.path().join("docs.json");
    let gate_path = dir.path().join("gate.json");
    let adversarial_path = dir.path().join("adversarial.json");
    let goldens_path = dir.path().join("goldens.json");
    let out_path = dir.path().join("nightly_summary.json");

    fs::write(
        &docs_path,
        serde_json::to_string_pretty(&DocsLintReport {
            metadata: ReportFreshnessMetadata::default(),
            ok: true,
            status: DocsLintStatus::Pass,
            mode: DocsLintMode::Strict,
            checks: Vec::new(),
        })
        .expect("docs json"),
    )
    .expect("write docs");
    fs::write(
        &gate_path,
        serde_json::to_string_pretty(&ReadinessGateReport {
            metadata: ReportFreshnessMetadata::default(),
            code_version_tag: "x".to_string(),
            profile: "test".to_string(),
            fixtures_digest_prefix: None,
            backend_pack_digest_prefix: None,
            timestamp: None,
            status: GateStatus::Pass,
            checks: Vec::new(),
            weights_lifecycle: None,
            world_vljepa_evidence: None,
            sae_real: None,
            ssm_opt: None,
            gpu_lane: None,
        })
        .expect("gate json"),
    )
    .expect("write gate");
    fs::write(
        &adversarial_path,
        r#"{"suite_version":"v1","code_version_tag":"x","policy_bundle_hash_prefix":"abc","pass":true,"cases":[]}"#,
    )
    .expect("write adversarial");
    fs::write(
        &goldens_path,
        serde_json::to_string_pretty(&GoldenVerifyReport {
            os: "linux".to_string(),
            status: GateStatus::Fail,
            scenarios: vec![GoldenVerifyScenarioReport {
                scenario: "golden_a".to_string(),
                status: GateStatus::Fail,
                refresh_candidate: true,
                heuristic: GoldenRefreshHeuristic::DigestPrefixOnly,
                detail: "digest-only change".to_string(),
                remediation: "run update".to_string(),
            }],
        })
        .expect("goldens json"),
    )
    .expect("write goldens");

    let report = nightly_summarize(&NightlySummarizeArgs {
        docs_lint_report: docs_path,
        gate_report: gate_path,
        adversarial_report: adversarial_path,
        goldens_report: goldens_path,
        drift_report: None,
        out: out_path.clone(),
    })
    .expect("summarize");

    assert_eq!(report.status, NightlyOverallStatus::Fail);
    assert!(report.golden_refresh_suggested);
    assert_eq!(report.components[0].name, "adversarial");
    assert!(out_path.exists());
}
