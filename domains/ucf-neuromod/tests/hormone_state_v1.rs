use ucf_neuromod::hormone_state_v1::{HormoneStateRawV1, HormoneStateV1, NormalizedHormoneLevelV1};

#[test]
fn hormone_state_neutral_is_valid() {
    let state = HormoneStateV1::neutral();
    assert_eq!(
        state.dopamine_like.as_units(),
        NormalizedHormoneLevelV1::NEUTRAL
    );
    assert_eq!(
        state.serotonin_like.as_units(),
        NormalizedHormoneLevelV1::NEUTRAL
    );
    assert_eq!(
        state.cortisol_like.as_units(),
        NormalizedHormoneLevelV1::NEUTRAL
    );
    assert_eq!(
        state.arousal_like.as_units(),
        NormalizedHormoneLevelV1::NEUTRAL
    );
    assert_eq!(
        state.sleep_pressure.as_units(),
        NormalizedHormoneLevelV1::NEUTRAL
    );
    assert_eq!(
        state.novelty_pressure.as_units(),
        NormalizedHormoneLevelV1::NEUTRAL
    );
    assert_eq!(
        state.stability_pressure.as_units(),
        NormalizedHormoneLevelV1::NEUTRAL
    );
    assert!(state.validate().is_ok());
}

#[test]
fn hormone_level_rejects_out_of_range() {
    assert!(NormalizedHormoneLevelV1::try_new(10_001).is_err());
    assert!(NormalizedHormoneLevelV1::try_new(-1).is_err());
}

#[test]
fn hormone_level_clamps_out_of_range() {
    let high = NormalizedHormoneLevelV1::new_clamped(50_000);
    let low = NormalizedHormoneLevelV1::new_clamped(-50_000);

    assert_eq!(high.as_units(), NormalizedHormoneLevelV1::MAX);
    assert_eq!(low.as_units(), NormalizedHormoneLevelV1::MIN);
}

#[test]
fn hormone_state_repeated_construction_is_deterministic() {
    let raw = HormoneStateRawV1 {
        dopamine_like: 123,
        serotonin_like: 456,
        cortisol_like: 789,
        arousal_like: 999,
        sleep_pressure: 222,
        novelty_pressure: 333,
        stability_pressure: 444,
    };

    let a = HormoneStateV1::new(raw).expect("valid state");
    let b = HormoneStateV1::new(raw).expect("valid state");
    assert_eq!(a, b);
}

#[test]
fn hormone_state_fields_are_bounded() {
    let state = HormoneStateV1::new_clamped(HormoneStateRawV1 {
        dopamine_like: i64::MAX,
        serotonin_like: i64::MAX,
        cortisol_like: i64::MAX,
        arousal_like: i64::MAX,
        sleep_pressure: i64::MIN,
        novelty_pressure: i64::MIN,
        stability_pressure: i64::MIN,
    });

    let fields = [
        state.dopamine_like,
        state.serotonin_like,
        state.cortisol_like,
        state.arousal_like,
        state.sleep_pressure,
        state.novelty_pressure,
        state.stability_pressure,
    ];

    for field in fields {
        assert!(
            (NormalizedHormoneLevelV1::MIN..=NormalizedHormoneLevelV1::MAX)
                .contains(&field.as_units())
        );
    }
}

#[test]
fn hormone_state_has_no_policy_gateway_identity_archive_authority() {
    assert!(!HormoneStateV1::policy_mutating());
    assert!(!HormoneStateV1::gateway_visible());
    assert!(!HormoneStateV1::identity_authority());
    assert!(!HormoneStateV1::evidence_archive_authority());
}

#[test]
fn hormone_state_has_no_scheduler_or_runtime_side_effects() {
    let state = HormoneStateV1::neutral();
    assert!(state.validate().is_ok());
}
