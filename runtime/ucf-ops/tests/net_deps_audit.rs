use std::path::Path;

use ucf_ops::{
    net_deps_audit_from_lockfile_toml, net_deps_audit_from_metadata_json, NetworkAllowlist,
};

fn allowlist() -> NetworkAllowlist {
    NetworkAllowlist {
        schema_version: 1,
        runtime_crates: vec!["ucf-runtime".to_string()],
        forbidden_crates: vec!["reqwest".to_string(), "hyper".to_string()],
        allowed_feature_notes: vec!["remote-compute".to_string()],
        exempt_runtime_edges: Vec::new(),
    }
}

#[test]
fn parses_allowlist_file() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let parsed = ucf_ops::load_network_allowlist(&repo_root.join("docs/network_allowlist.toml"))
        .expect("parse allowlist");
    assert_eq!(parsed.schema_version, 1);
    assert!(parsed
        .forbidden_crates
        .iter()
        .any(|crate_name| crate_name == "reqwest"));
}

#[test]
fn net_deps_audit_reports_forbidden_path_deterministically() {
    let metadata = r#"{
      "packages": [
        {"id":"root 0.1.0 (path+file:///root)","name":"ucf-runtime"},
        {"id":"mid 0.1.0 (path+file:///mid)","name":"ucf-mid"},
        {"id":"req 0.11.0 (registry+https://x)","name":"reqwest"}
      ],
      "resolve": {
        "nodes": [
          {"id":"root 0.1.0 (path+file:///root)","deps":[{"pkg":"mid 0.1.0 (path+file:///mid)"}]},
          {"id":"mid 0.1.0 (path+file:///mid)","deps":[{"pkg":"req 0.11.0 (registry+https://x)"}]},
          {"id":"req 0.11.0 (registry+https://x)","deps":[]}
        ]
      }
    }"#;
    let report = net_deps_audit_from_metadata_json(
        metadata,
        &allowlist(),
        Path::new("docs/network_allowlist.toml"),
    )
    .expect("audit");
    assert_eq!(report.violations.len(), 1);
    assert_eq!(report.violations[0].forbidden_crate, "reqwest");
    assert_eq!(
        report.violations[0].path,
        vec!["ucf-runtime", "ucf-mid", "reqwest"]
    );
}

#[test]
fn net_deps_lockfile_fallback_reports_forbidden_path() {
    let lockfile = r#"
version = 3

[[package]]
name = "ucf-runtime"
version = "0.1.0"
dependencies = ["ucf-mid 0.1.0"]

[[package]]
name = "ucf-mid"
version = "0.1.0"
dependencies = ["reqwest 0.11.0"]

[[package]]
name = "reqwest"
version = "0.11.0"
"#;

    let report = net_deps_audit_from_lockfile_toml(
        lockfile,
        &allowlist(),
        Path::new("docs/network_allowlist.toml"),
    )
    .expect("audit lockfile");
    assert_eq!(report.violations.len(), 1);
    assert_eq!(
        report.violations[0].path,
        vec!["ucf-runtime", "ucf-mid", "reqwest"]
    );
}
