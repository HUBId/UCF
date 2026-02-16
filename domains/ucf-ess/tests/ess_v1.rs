use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{
    EssError, ExperienceRecord, ExperienceStore, IdAllocator, InMemoryEss, LfmSummaryRecord,
    LfmWindowRecord,
};
use ucf_frames::v1::{
    ChannelCode, ControlFrame, CorrelationId, DecisionFrame, Intent, IntentId, IntentKind,
};

fn time(tick: u64, window: u64) -> SimTime {
    SimTime {
        tick: Tick::new(tick),
        window: WindowId::new(window),
    }
}

#[test]
fn append_with_increasing_ticks_succeeds() {
    let mut ess = InMemoryEss::new();
    let mut ids = IdAllocator::new(1);

    let a = ExperienceRecord::note(ids.next(), time(1, 0), CorrelationId(10), "a");
    let b = ExperienceRecord::note(ids.next(), time(2, 0), CorrelationId(11), "b");

    ess.append(a).expect("append a should succeed");
    ess.append(b).expect("append b should succeed");

    assert_eq!(ess.len(), 2);
}

#[test]
fn append_same_tick_increasing_window_succeeds() {
    let mut ess = InMemoryEss::new();
    let mut ids = IdAllocator::new(10);

    ess.append(ExperienceRecord::note(
        ids.next(),
        time(5, 1),
        CorrelationId(1),
        "first",
    ))
    .expect("first append should succeed");

    ess.append(ExperienceRecord::note(
        ids.next(),
        time(5, 2),
        CorrelationId(2),
        "second",
    ))
    .expect("second append should succeed");

    assert_eq!(ess.len(), 2);
}

#[test]
fn append_lower_tick_fails_and_len_unchanged() {
    let mut ess = InMemoryEss::new();
    let mut ids = IdAllocator::new(20);

    ess.append(ExperienceRecord::note(
        ids.next(),
        time(8, 0),
        CorrelationId(1),
        "ok",
    ))
    .expect("seed append should succeed");

    let before = ess.len();
    let err = ess
        .append(ExperienceRecord::note(
            ids.next(),
            time(7, 99),
            CorrelationId(2),
            "backwards",
        ))
        .expect_err("append should fail");

    assert_eq!(err, EssError::TimeWentBackwards);
    assert_eq!(ess.len(), before);
}

#[test]
fn append_lower_window_same_tick_fails_and_len_unchanged() {
    let mut ess = InMemoryEss::new();
    let mut ids = IdAllocator::new(30);

    ess.append(ExperienceRecord::note(
        ids.next(),
        time(12, 3),
        CorrelationId(1),
        "ok",
    ))
    .expect("seed append should succeed");

    let before = ess.len();
    let err = ess
        .append(ExperienceRecord::note(
            ids.next(),
            time(12, 2),
            CorrelationId(2),
            "backwards window",
        ))
        .expect_err("append should fail");

    assert_eq!(err, EssError::TimeWentBackwards);
    assert_eq!(ess.len(), before);
}

#[test]
fn tail_time_returns_last_appended_time() {
    let mut ess = InMemoryEss::new();
    let mut ids = IdAllocator::new(40);

    let first = time(3, 1);
    let second = time(3, 5);

    ess.append(ExperienceRecord::note(
        ids.next(),
        first,
        CorrelationId(10),
        "first",
    ))
    .expect("first append should succeed");
    ess.append(ExperienceRecord::note(
        ids.next(),
        second,
        CorrelationId(11),
        "second",
    ))
    .expect("second append should succeed");

    assert_eq!(ess.tail_time(), Some(second));
}

#[test]
fn decision_ledger_queries_are_indexed_by_correlation_id() {
    let mut ess = InMemoryEss::new();
    let mut ids = IdAllocator::new(50);
    let corr_x = CorrelationId(101);
    let corr_y = CorrelationId(202);

    let control = ControlFrame::new_text(
        time(20, 0),
        corr_x,
        ChannelCode::ExternalOutput,
        Intent::new(IntentId(1), IntentKind::Speak, "control"),
        "ctrl",
    );
    let decision = DecisionFrame::allow(time(20, 1), corr_x, "allowed");
    let note = ExperienceRecord::note(ids.next(), time(20, 2), corr_x, "note");

    ess.append(ExperienceRecord::from_control(ids.next(), control))
        .expect("control append should succeed");
    ess.append(ExperienceRecord::from_decision(
        ids.next(),
        decision.clone(),
    ))
    .expect("decision append should succeed");
    ess.append(note).expect("note append should succeed");

    let indices = ess.indices_by_corr(corr_x);
    assert_eq!(indices, &[0, 1, 2]);

    let trail = ess.trail_by_corr(corr_x);
    assert_eq!(trail.len(), 3);
    assert_eq!(trail[0].kind, ucf_ess::v1::ExperienceKind::ControlIn);
    assert_eq!(trail[1].kind, ucf_ess::v1::ExperienceKind::DecisionOut);
    assert_eq!(trail[2].kind, ucf_ess::v1::ExperienceKind::Note);

    let last_decision = ess
        .last_decision_for_corr(corr_x)
        .expect("decision should be present for corr_x");
    assert_eq!(last_decision, &decision);

    assert!(ess.indices_by_corr(corr_y).is_empty());
    assert!(ess.trail_by_corr(corr_y).is_empty());
    assert!(ess.last_decision_for_corr(corr_y).is_none());
}

#[test]
fn constructors_default_neuromod_to_none() {
    let mut ids = IdAllocator::new(1000);
    let corr = CorrelationId(303);
    let t = time(99, 1);

    let control = ControlFrame::new_text(
        t,
        corr,
        ChannelCode::ExternalOutput,
        Intent::new(IntentId(9), IntentKind::Speak, "control"),
        "ctrl",
    );
    let decision = DecisionFrame::allow(time(99, 2), corr, "allowed");

    assert_eq!(
        ExperienceRecord::from_control(ids.next(), control).neuromod,
        None
    );
    assert_eq!(
        ExperienceRecord::from_decision(ids.next(), decision).neuromod,
        None
    );
    assert_eq!(
        ExperienceRecord::note(ids.next(), time(99, 3), corr, "note").neuromod,
        None
    );
}

#[test]
fn lfm_summary_digest_is_stable() {
    let summary = LfmSummaryRecord {
        t: 42,
        decision_id: Some(7),
        evidence_chain_digest: [1; 32],
        backend_pack_digest: [2; 32],
        liquid_state_digest: [3; 32],
        liquid_readout_digest: [4; 32],
        uncertainty: 0.3,
        stability: 0.8,
        schema_version: 1,
        digest: [0; 32],
    };
    let a = summary.with_digest();
    let b = summary.with_digest();
    assert_eq!(a.digest, b.digest);
}

#[test]
fn lfm_records_can_be_appended() {
    let mut ess = InMemoryEss::new();
    let mut ids = IdAllocator::new(2000);
    let corr = CorrelationId(88);
    let t = time(55, 1);

    let summary = LfmSummaryRecord {
        t: 55,
        decision_id: Some(2001),
        evidence_chain_digest: [1; 32],
        backend_pack_digest: [2; 32],
        liquid_state_digest: [3; 32],
        liquid_readout_digest: [4; 32],
        uncertainty: 0.4,
        stability: 0.7,
        schema_version: 1,
        digest: [0; 32],
    }
    .with_digest();
    let window = LfmWindowRecord {
        t0: 50,
        t1: 55,
        sample_count: 4,
        mean_uncertainty: 0.35,
        mean_stability: 0.72,
        rolling_digest: [9; 32],
        schema_version: 1,
        digest: [0; 32],
    }
    .with_digest();

    ess.append(ExperienceRecord::from_lfm_summary(
        ids.next(),
        t,
        corr,
        summary,
    ))
    .expect("lfm summary append should succeed");
    ess.append(ExperienceRecord::from_lfm_window(
        ids.next(),
        time(56, 1),
        corr,
        window,
    ))
    .expect("lfm window append should succeed");

    assert_eq!(ess.len(), 2);
}
