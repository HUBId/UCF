use super::{compute_delta, NeuromodInputs, NeuromodScheduler, NeuromodulatorField};

#[test]
fn baseline_snapshot_is_half() {
    let snap = NeuromodulatorField::new_baseline().snapshot();
    assert_eq!(snap.dopamine, 0.5);
    assert_eq!(snap.serotonin, 0.5);
    assert_eq!(snap.norepinephrine, 0.5);
    assert_eq!(snap.acetylcholine, 0.5);
    assert_eq!(snap.oxytocin, 0.5);
    assert_eq!(snap.endorphin, 0.5);
    assert_eq!(snap.stress, 0.5);
}

#[test]
fn compute_delta_clamps_to_range() {
    let delta = compute_delta(NeuromodInputs {
        surprise: 10.0,
        reward: -10.0,
        threat: 10.0,
        social: 10.0,
    });

    for v in [
        delta.dopamine,
        delta.serotonin,
        delta.norepinephrine,
        delta.acetylcholine,
        delta.oxytocin,
        delta.endorphin,
        delta.stress,
    ] {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn scheduler_advances_deterministically_for_multiple_ticks() {
    let mut field = NeuromodulatorField::new_baseline();
    let mut scheduler = NeuromodScheduler::new(10);
    let inputs = NeuromodInputs {
        surprise: 1.0,
        reward: 1.0,
        threat: 0.0,
        social: 0.0,
    };

    scheduler.advance(25, &mut field, inputs);
    let after_two = field.snapshot();

    let mut expected: f32 = 0.5;
    let delta: f32 = 0.85;
    for _ in 0..2 {
        expected = (expected + (delta - 0.5_f32) * 0.1_f32).clamp(0.0_f32, 1.0_f32);
    }

    assert!((after_two.dopamine - expected).abs() < 1e-6);

    scheduler.advance(25, &mut field, inputs);
    let unchanged = field.snapshot();
    assert_eq!(unchanged, after_two);

    scheduler.advance(35, &mut field, inputs);
    let after_three = field.snapshot();
    let expected_three = (expected + (delta - 0.5_f32) * 0.1_f32).clamp(0.0_f32, 1.0_f32);
    assert!((after_three.dopamine - expected_three).abs() < 1e-6);
}
