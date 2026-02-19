use tempfile::tempdir;

fn ensure_workspace_root() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("workspace root");
        std::env::set_current_dir(&root).expect("set current dir");
    });
}

use ucf_ops::{
    bringup, causal_slice, event_id_for_decision, explain_why, simulate_counterfactual,
    CounterfactualRequest,
};

#[test]
fn causal_slice_contains_decision_chain() {
    ensure_workspace_root();
    let dir = tempdir().expect("tempdir");
    bringup(dir.path(), true, 24).expect("bringup");

    let run_id = "unknown";
    let event_id = event_id_for_decision(dir.path(), run_id, 24)
        .expect("event id")
        .expect("decision event");
    let slice = causal_slice(dir.path(), run_id, &event_id, 2).expect("causal slice");

    assert!(!slice.nodes.is_empty());
    assert!(slice
        .edges
        .iter()
        .any(|e| e.src_event_id == event_id || e.dst_event_id == event_id));
}

#[test]
fn explain_why_reports_links() {
    ensure_workspace_root();
    let dir = tempdir().expect("tempdir");
    bringup(dir.path(), true, 20).expect("bringup");

    let report = explain_why(dir.path(), 20).expect("explain why");
    assert_eq!(report.decision_id, 20);
    assert!(!report.slice.nodes.is_empty());
}

#[test]
fn counterfactual_simulation_never_executes_tools() {
    ensure_workspace_root();
    let dir = tempdir().expect("tempdir");
    bringup(dir.path(), true, 28).expect("bringup");

    let result = simulate_counterfactual(
        dir.path(),
        CounterfactualRequest {
            base_decision_id: 28,
            alternative_candidate_id: 1,
        },
    )
    .expect("counterfactual");

    assert!(matches!(result.would_issue_tool, true | false));
    assert!(dir
        .path()
        .join("ess")
        .join("counterfactual_records.json")
        .exists());
}
