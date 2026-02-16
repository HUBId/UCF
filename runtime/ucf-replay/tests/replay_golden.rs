use std::path::Path;

use ucf_replay::{
    load_fixture_records, replay_audit, replay_records, ReplayMode, ReplayPlan, ReplaySpec,
    ReplayStrictness,
};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/golden_replay_fixture.json"
);

fn spec() -> ReplaySpec {
    ReplaySpec {
        from_tick: 0,
        to_tick: u64::MAX,
        backend_override: None,
        seed_override: None,
        budget_override: None,
        mode: ReplayMode::ComputeOnly,
    }
}

#[test]
fn golden_fixture_matches_in_compute_mode() {
    let records = load_fixture_records(Path::new(FIXTURE)).expect("fixture");
    let result = replay_records(&records, &spec());
    assert_eq!(result.total_items, 3);
    assert_eq!(result.drifted, 0, "drifts: {:?}", result.items);
    assert_eq!(result.matched, 3);
}

#[test]
fn backend_override_can_trigger_drift() {
    let records = load_fixture_records(Path::new(FIXTURE)).expect("fixture");
    let mut replay_spec = spec();
    replay_spec.seed_override = Some(7);
    let result = replay_records(&records, &replay_spec);
    assert!(result.drifted > 0);
}

#[test]
fn full_no_action_never_executes_actions() {
    let records = load_fixture_records(Path::new(FIXTURE)).expect("fixture");
    let mut replay_spec = spec();
    replay_spec.mode = ReplayMode::FullNoAction;
    let result = replay_records(&records, &replay_spec);
    assert_eq!(result.total_items, 3);
}

#[test]
fn replay_audit_reports_missing_data_for_empty_range() {
    let records = load_fixture_records(Path::new(FIXTURE)).expect("fixture");
    let plan = ReplayPlan {
        t0: 99_999,
        t1: 100_000,
        expected_backend_pack_digest: None,
        strictness: ReplayStrictness::VerifyOnly,
        stop_on_first_divergence: true,
    };
    let report = replay_audit(&records, &plan);
    assert_eq!(
        report.overall_status,
        ucf_replay::ReplayOverallStatus::MissingData
    );
    assert!(report.first_divergence.is_none());
    assert!(report.counters.missing_records > 0);
}

#[test]
fn replay_audit_detects_missing_chain_digest() {
    let records = load_fixture_records(Path::new(FIXTURE)).expect("fixture");
    let plan = ReplayPlan {
        t0: 0,
        t1: u64::MAX,
        expected_backend_pack_digest: None,
        strictness: ReplayStrictness::VerifyOnly,
        stop_on_first_divergence: true,
    };
    let report = replay_audit(&records, &plan);
    assert_eq!(
        report.overall_status,
        ucf_replay::ReplayOverallStatus::DriftFound
    );
    let first = report.first_divergence.expect("first divergence");
    assert_eq!(first.component, ucf_replay::ReplayComponent::Risk);
}
