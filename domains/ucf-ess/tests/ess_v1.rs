use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{EssError, ExperienceRecord, ExperienceStore, IdAllocator, InMemoryEss};
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
