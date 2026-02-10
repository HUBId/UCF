use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_ess::v1::{EssError, ExperienceRecord, ExperienceStore, IdAllocator, InMemoryEss};
use ucf_frames::v1::CorrelationId;

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
