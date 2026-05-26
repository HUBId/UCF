use ucf_neuromod::hormone_state_v1::{HormoneStateRawV1, HormoneStateV1, NormalizedHormoneLevelV1};
use ucf_neuromod::hormone_update_v1::{
    derive_hormone_modulation_output_v1, update_hormone_state_v1, HormoneInputFrameV1,
    HormoneModulationOutputV1, HormoneUpdateConfigV1,
};

fn state(raw: HormoneStateRawV1) -> HormoneStateV1 {
    HormoneStateV1::new_clamped(raw)
}

fn cfg() -> HormoneUpdateConfigV1 {
    HormoneUpdateConfigV1::bounded_default()
}

fn neutral_raw() -> HormoneStateRawV1 {
    HormoneStateRawV1 {
        dopamine_like: 5_000,
        serotonin_like: 5_000,
        cortisol_like: 5_000,
        arousal_like: 5_000,
        sleep_pressure: 5_000,
        novelty_pressure: 5_000,
        stability_pressure: 5_000,
    }
}

#[test]
fn modulation_mapping_is_deterministic() {
    let s = HormoneStateV1::neutral();
    assert_eq!(
        derive_hormone_modulation_output_v1(&s),
        derive_hormone_modulation_output_v1(&s)
    );
}

#[test]
fn modulation_outputs_are_bounded() {
    let out = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        dopamine_like: i64::MAX,
        serotonin_like: i64::MIN,
        cortisol_like: i64::MAX,
        arousal_like: i64::MAX,
        sleep_pressure: i64::MAX,
        novelty_pressure: i64::MAX,
        stability_pressure: i64::MIN,
    }));
    for v in [
        out.attention_gain,
        out.learning_rate_multiplier,
        out.replay_priority_multiplier,
        out.noise_scale,
        out.consolidation_gate,
        out.risk_damping,
    ] {
        assert!((0..=10_000).contains(&v));
    }
    assert!((-10_000..=10_000).contains(&out.sleep_pressure_delta));
}

#[test]
fn dopamine_novelty_raise_attention_gain() {
    let low = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        dopamine_like: 2_000,
        serotonin_like: 5_000,
        cortisol_like: 5_000,
        arousal_like: 5_000,
        sleep_pressure: 5_000,
        novelty_pressure: 2_000,
        stability_pressure: 5_000,
    }));
    let high = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        dopamine_like: 8_000,
        serotonin_like: 5_000,
        cortisol_like: 5_000,
        arousal_like: 5_000,
        sleep_pressure: 5_000,
        novelty_pressure: 8_000,
        stability_pressure: 5_000,
    }));
    assert!(high.attention_gain > low.attention_gain);
    assert!(high.replay_priority_multiplier > low.replay_priority_multiplier);
}

#[test]
fn cortisol_raises_noise_and_risk_damping() {
    let low = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        dopamine_like: 5_000,
        serotonin_like: 5_000,
        cortisol_like: 2_000,
        arousal_like: 5_000,
        sleep_pressure: 5_000,
        novelty_pressure: 5_000,
        stability_pressure: 5_000,
    }));
    let high = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        cortisol_like: 8_000,
        ..neutral_raw()
    }));
    assert!(high.noise_scale > low.noise_scale);
    assert!(high.risk_damping > low.risk_damping);
}

#[test]
fn stability_supports_consolidation_gate() {
    let low = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        stability_pressure: 2_000,
        ..neutral_raw()
    }));
    let high = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        stability_pressure: 8_000,
        ..neutral_raw()
    }));
    assert!(high.consolidation_gate > low.consolidation_gate);
}

#[test]
fn high_sleep_pressure_raises_sleep_pressure_delta() {
    let low = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        sleep_pressure: 2_000,
        ..neutral_raw()
    }));
    let high = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        sleep_pressure: 8_000,
        ..neutral_raw()
    }));
    assert!(high.sleep_pressure_delta > low.sleep_pressure_delta);
    assert!(high.sleep_pressure_delta > 0);
    assert!(low.sleep_pressure_delta < 0);
}

#[test]
fn modulation_is_advisory_only() {
    assert!(!HormoneModulationOutputV1::runtime_authority());
    assert!(!HormoneModulationOutputV1::gateway_authority());
    assert!(!HormoneModulationOutputV1::policy_mutation());
}

#[test]
fn modulation_mapping_has_no_replay_sleep_geist_side_effects() {
    let s = HormoneStateV1::neutral();
    let _ = derive_hormone_modulation_output_v1(&s);
    assert_eq!(s, HormoneStateV1::neutral());
}

#[test]
fn modulation_mapping_uses_no_random_or_wallclock() {
    let s = state(HormoneStateRawV1 {
        dopamine_like: 3_333,
        serotonin_like: 4_444,
        cortisol_like: 5_555,
        arousal_like: 6_666,
        sleep_pressure: 7_777,
        novelty_pressure: 2_222,
        stability_pressure: 1_111,
    });
    assert_eq!(
        derive_hormone_modulation_output_v1(&s),
        derive_hormone_modulation_output_v1(&s)
    );
}

#[test]
fn update_function_uses_same_modulation_mapping() {
    let prev = HormoneStateV1::new_clamped(HormoneStateRawV1 {
        dopamine_like: 4_000,
        serotonin_like: 5_000,
        cortisol_like: 6_000,
        arousal_like: 3_000,
        sleep_pressure: 7_000,
        novelty_pressure: 4_500,
        stability_pressure: 6_500,
    });
    let input = HormoneInputFrameV1 {
        reward_signal: 1_100,
        novelty_signal: 2_200,
        threat_signal: 700,
        fatigue_signal: 500,
        inconsistency_signal: 300,
        replay_density: 900,
        policy_violation_pressure: 100,
    };
    let (next_state, out) = update_hormone_state_v1(prev, input, cfg());
    assert_eq!(out, derive_hormone_modulation_output_v1(&next_state));
    assert!(
        (NormalizedHormoneLevelV1::MIN..=NormalizedHormoneLevelV1::MAX)
            .contains(&next_state.dopamine_like.as_units())
    );
}
