use ucf_demo::run_cycles;

#[test]
fn seed_1_three_cycles_final_summary_constants() {
    let summaries = run_cycles(3, 1);
    let final_summary = summaries
        .last()
        .expect("three cycles should produce summaries");

    assert_eq!(final_summary.cycle_id, 2);
    assert_eq!(final_summary.gamma_bucket, 14);
    assert_eq!(final_summary.plv, 8625);
    assert_eq!(final_summary.lock_window, 2);
    assert_eq!(final_summary.surprise, 2500);
    assert_eq!(final_summary.learn_rate, 6289);
    assert_eq!(final_summary.learn_mode, 2);
    assert_eq!(final_summary.delta_mass, 2578);
    assert_eq!(final_summary.delta_targets, [0, 0, 0, 0]);
    assert_eq!(final_summary.nsr_verdict, Some(0));
    assert_eq!(final_summary.nsr_hit_counts, [1, 0, 0]);
    assert_eq!(final_summary.violations.len(), 1);
    assert_eq!(&final_summary.violations[0].commit.as_bytes()[..4], &[0x9f, 0x5d, 0x67, 0x67]);
}
