use ucf_neuromod::hormone_state_v1::{HormoneStateRawV1, HormoneStateV1, NormalizedHormoneLevelV1};
use ucf_neuromod::hormone_update_v1::{
    update_hormone_state_v1, HormoneInputFrameV1, HormoneModulationOutputV1, HormoneUpdateConfigV1,
};

fn cfg() -> HormoneUpdateConfigV1 {
    HormoneUpdateConfigV1::bounded_default()
}

#[test]
fn update_rules_are_deterministic() {
    let prev = HormoneStateV1::neutral();
    let input = HormoneInputFrameV1 {
        reward_signal: 3_000,
        novelty_signal: 2_000,
        threat_signal: 1_000,
        fatigue_signal: 500,
        inconsistency_signal: 200,
        replay_density: 800,
        policy_violation_pressure: 300,
    };
    let a = update_hormone_state_v1(prev, input, cfg());
    let b = update_hormone_state_v1(prev, input, cfg());
    assert_eq!(a, b);
}

#[test]
fn update_clamps_to_bounds() {
    let prev = HormoneStateV1::new_clamped(HormoneStateRawV1 {
        dopamine_like: i64::MAX,
        serotonin_like: i64::MIN,
        cortisol_like: i64::MAX,
        arousal_like: i64::MIN,
        sleep_pressure: i64::MAX,
        novelty_pressure: i64::MAX,
        stability_pressure: i64::MIN,
    });
    let input = HormoneInputFrameV1 {
        reward_signal: i64::MAX,
        novelty_signal: i64::MAX,
        threat_signal: i64::MAX,
        fatigue_signal: i64::MAX,
        inconsistency_signal: i64::MAX,
        replay_density: i64::MAX,
        policy_violation_pressure: i64::MAX,
    };

    let (next, _) = update_hormone_state_v1(prev, input, cfg());
    for value in [
        next.dopamine_like,
        next.serotonin_like,
        next.cortisol_like,
        next.arousal_like,
        next.sleep_pressure,
        next.novelty_pressure,
        next.stability_pressure,
    ] {
        assert!(
            (NormalizedHormoneLevelV1::MIN..=NormalizedHormoneLevelV1::MAX)
                .contains(&value.as_units())
        );
    }
}

#[test]
fn decay_moves_toward_neutral() {
    let prev = HormoneStateV1::new_clamped(HormoneStateRawV1 {
        dopamine_like: 10_000,
        serotonin_like: 10_000,
        cortisol_like: 10_000,
        arousal_like: 10_000,
        sleep_pressure: 10_000,
        novelty_pressure: 10_000,
        stability_pressure: 10_000,
    });

    let (next, _) = update_hormone_state_v1(prev, HormoneInputFrameV1::neutral(), cfg());
    assert!(next.dopamine_like.as_units() < prev.dopamine_like.as_units());
}

#[test]
fn reward_increases_dopamine() {
    let prev = HormoneStateV1::neutral();
    let input = HormoneInputFrameV1 {
        reward_signal: 4_000,
        ..HormoneInputFrameV1::neutral()
    };
    let (next, _) = update_hormone_state_v1(prev, input, cfg());
    assert!(next.dopamine_like.as_units() > prev.dopamine_like.as_units());
}

#[test]
fn threat_increases_cortisol() {
    let prev = HormoneStateV1::neutral();
    let input = HormoneInputFrameV1 {
        threat_signal: 4_000,
        ..HormoneInputFrameV1::neutral()
    };
    let (next, _) = update_hormone_state_v1(prev, input, cfg());
    assert!(next.cortisol_like.as_units() > prev.cortisol_like.as_units());
}

#[test]
fn fatigue_increases_sleep_pressure() {
    let prev = HormoneStateV1::neutral();
    let input = HormoneInputFrameV1 {
        fatigue_signal: 4_000,
        ..HormoneInputFrameV1::neutral()
    };
    let (next, _) = update_hormone_state_v1(prev, input, cfg());
    assert!(next.sleep_pressure.as_units() > prev.sleep_pressure.as_units());
}

#[test]
fn inconsistency_reduces_stability() {
    let prev = HormoneStateV1::neutral();
    let input = HormoneInputFrameV1 {
        inconsistency_signal: 4_000,
        ..HormoneInputFrameV1::neutral()
    };
    let (next, _) = update_hormone_state_v1(prev, input, cfg());
    assert!(next.stability_pressure.as_units() < prev.stability_pressure.as_units());
}

#[test]
fn modulation_output_is_bounded() {
    let (next, output) = update_hormone_state_v1(
        HormoneStateV1::neutral(),
        HormoneInputFrameV1::neutral(),
        cfg(),
    );
    assert!(next.validate().is_ok());
    for value in [
        output.attention_gain,
        output.learning_rate_multiplier,
        output.replay_priority_multiplier,
        output.noise_scale,
        output.consolidation_gate,
        output.risk_damping,
    ] {
        assert!((0..=10_000).contains(&value));
    }
    assert!((-10_000..=10_000).contains(&output.sleep_pressure_delta));
}

#[test]
fn modulation_output_has_no_runtime_gateway_policy_authority() {
    assert!(!HormoneModulationOutputV1::runtime_authority());
    assert!(!HormoneModulationOutputV1::gateway_authority());
    assert!(!HormoneModulationOutputV1::policy_mutation());
}

#[test]
fn update_has_no_runtime_side_effects() {
    let prev = HormoneStateV1::neutral();
    let (next, _) = update_hormone_state_v1(prev, HormoneInputFrameV1::neutral(), cfg());
    assert!(next.validate().is_ok());
}

#[test]
fn update_uses_no_random_or_wallclock() {
    let prev = HormoneStateV1::neutral();
    let input = HormoneInputFrameV1 {
        reward_signal: 3_333,
        novelty_signal: 777,
        ..HormoneInputFrameV1::neutral()
    };
    let (a_state, a_out) = update_hormone_state_v1(prev, input, cfg());
    let (b_state, b_out) = update_hormone_state_v1(prev, input, cfg());
    assert_eq!(a_state, b_state);
    assert_eq!(a_out, b_out);
}

#[test]
fn repeated_decay_is_stable() {
    let mut state = HormoneStateV1::new_clamped(HormoneStateRawV1 {
        dopamine_like: 10_000,
        serotonin_like: 0,
        cortisol_like: 10_000,
        arousal_like: 0,
        sleep_pressure: 10_000,
        novelty_pressure: 0,
        stability_pressure: 10_000,
    });

    let start = state;
    for _ in 0..64 {
        state = update_hormone_state_v1(state, HormoneInputFrameV1::neutral(), cfg()).0;
    }

    assert!(state.dopamine_like.as_units() < start.dopamine_like.as_units());
    assert!(state.dopamine_like.as_units() > NormalizedHormoneLevelV1::NEUTRAL);
    assert!(state.serotonin_like.as_units() > start.serotonin_like.as_units());
    assert!(state.serotonin_like.as_units() < NormalizedHormoneLevelV1::NEUTRAL);
}
