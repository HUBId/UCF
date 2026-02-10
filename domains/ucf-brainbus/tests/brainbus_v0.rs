use ucf_brainbus::v0::{BrainBus, BrainBusError, BrainEvent, InMemoryBrainQueue, OscPhase, Spike};
use ucf_core::types::{SimTime, Tick, WindowId};
use ucf_frames::v1::CorrelationId;

fn time(tick: u64) -> SimTime {
    SimTime {
        tick: Tick::new(tick),
        window: WindowId::new(0),
    }
}

#[test]
fn phase_normalization() {
    let phase_over = OscPhase::new(40.0, 370.0);
    assert!((phase_over.theta_deg - 10.0).abs() < f32::EPSILON);
    assert!((0.0..360.0).contains(&phase_over.theta_deg));

    let phase_negative = OscPhase::new(40.0, -10.0);
    assert!((phase_negative.theta_deg - 350.0).abs() < f32::EPSILON);
    assert!((0.0..360.0).contains(&phase_negative.theta_deg));
}

#[test]
fn fifo_ordering_for_equal_time_spikes() {
    let mut queue = InMemoryBrainQueue::new(8);
    let t = time(5);
    let first = Spike::new(t, CorrelationId(1), 10, 20, 111);
    let second = Spike::new(t, CorrelationId(2), 11, 21, 222);

    queue.push(BrainEvent::Spike(first.clone())).unwrap();
    queue.push(BrainEvent::Spike(second.clone())).unwrap();

    assert_eq!(queue.pop_ready(t), Some(BrainEvent::Spike(first)));
    assert_eq!(queue.pop_ready(t), Some(BrainEvent::Spike(second)));
}

#[test]
fn time_gating_for_spike_events() {
    let mut queue = InMemoryBrainQueue::new(8);
    let spike = Spike::new(time(10), CorrelationId(1), 1, 2, 99);
    queue.push(BrainEvent::Spike(spike.clone())).unwrap();

    assert_eq!(queue.pop_ready(time(9)), None);
    assert_eq!(queue.pop_ready(time(10)), Some(BrainEvent::Spike(spike)));
}

#[test]
fn queue_full_error() {
    let mut queue = InMemoryBrainQueue::new(1);
    let first = Spike::new(time(1), CorrelationId(1), 1, 2, 10);
    let second = Spike::new(time(2), CorrelationId(2), 3, 4, 11);

    queue.push(BrainEvent::Spike(first)).unwrap();
    let err = queue.push(BrainEvent::Spike(second)).unwrap_err();

    assert_eq!(err, BrainBusError::QueueFull);
}
