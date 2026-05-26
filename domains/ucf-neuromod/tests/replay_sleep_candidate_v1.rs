use ucf_neuromod::hormone_state_v1::{HormoneStateRawV1, HormoneStateV1};
use ucf_neuromod::hormone_update_v1::derive_hormone_modulation_output_v1;
use ucf_neuromod::replay_sleep_candidate_v1::{
    derive_replay_sleep_candidates_v1, MetabolicReplayPriorityCandidateV1,
    MetabolicSleepPressureCandidateV1,
};

fn state(raw: HormoneStateRawV1) -> HormoneStateV1 {
    HormoneStateV1::new_clamped(raw)
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
fn replay_sleep_candidate_mapping_is_deterministic() {
    let out = derive_hormone_modulation_output_v1(&HormoneStateV1::neutral());
    assert_eq!(
        derive_replay_sleep_candidates_v1(&out),
        derive_replay_sleep_candidates_v1(&out)
    );
}

#[test]
fn replay_priority_candidate_is_bounded() {
    let out = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        novelty_pressure: i64::MAX,
        stability_pressure: i64::MAX,
        arousal_like: i64::MAX,
        cortisol_like: i64::MIN,
        ..neutral_raw()
    }));
    let c = derive_replay_sleep_candidates_v1(&out);
    assert!((0..=10_000).contains(&c.replay.priority_hint));
}

#[test]
fn sleep_pressure_candidate_is_bounded() {
    let out = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        sleep_pressure: i64::MAX,
        cortisol_like: i64::MAX,
        ..neutral_raw()
    }));
    let c = derive_replay_sleep_candidates_v1(&out);
    assert!((0..=10_000).contains(&c.sleep.pressure_hint));
}

#[test]
fn high_replay_priority_multiplier_increases_replay_hint() {
    let low = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        novelty_pressure: 2_000,
        stability_pressure: 2_000,
        arousal_like: 2_000,
        ..neutral_raw()
    }));
    let high = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        novelty_pressure: 8_000,
        stability_pressure: 8_000,
        arousal_like: 8_000,
        ..neutral_raw()
    }));
    assert!(
        derive_replay_sleep_candidates_v1(&high)
            .replay
            .priority_hint
            > derive_replay_sleep_candidates_v1(&low).replay.priority_hint
    );
}

#[test]
fn high_sleep_pressure_delta_increases_sleep_hint() {
    let low = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        sleep_pressure: 2_000,
        ..neutral_raw()
    }));
    let high = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        sleep_pressure: 8_000,
        ..neutral_raw()
    }));
    assert!(
        derive_replay_sleep_candidates_v1(&high).sleep.pressure_hint
            > derive_replay_sleep_candidates_v1(&low).sleep.pressure_hint
    );
}

#[test]
fn risk_damping_reduces_or_bounds_candidate_effect() {
    let low_risk = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        cortisol_like: 1_000,
        ..neutral_raw()
    }));
    let high_risk = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        cortisol_like: 9_000,
        ..neutral_raw()
    }));
    let low = derive_replay_sleep_candidates_v1(&low_risk);
    let high = derive_replay_sleep_candidates_v1(&high_risk);
    assert!(high.replay.risk_damping_component >= low.replay.risk_damping_component);
    assert!(high.sleep.risk_damping_component >= low.sleep.risk_damping_component);
}

#[test]
fn candidates_are_advisory_only() {
    assert!(MetabolicReplayPriorityCandidateV1::advisory_only());
    assert!(MetabolicSleepPressureCandidateV1::advisory_only());
}

#[test]
fn candidates_do_not_create_replay_schedule_or_applied() {
    assert!(!MetabolicReplayPriorityCandidateV1::scheduler_authority());
    assert!(!MetabolicReplayPriorityCandidateV1::replay_applied());
}

#[test]
fn candidates_do_not_create_sleep_plan_or_completed() {
    assert!(!MetabolicSleepPressureCandidateV1::scheduler_authority());
    assert!(!MetabolicSleepPressureCandidateV1::sleep_completed());
}

#[test]
fn candidates_have_no_gateway_policy_identity_archive_authority() {
    assert!(!MetabolicReplayPriorityCandidateV1::gateway_visible());
    assert!(!MetabolicReplayPriorityCandidateV1::policy_mutation());
    assert!(!MetabolicReplayPriorityCandidateV1::identity_authority());
    assert!(!MetabolicReplayPriorityCandidateV1::evidence_archive_authority());

    assert!(!MetabolicSleepPressureCandidateV1::gateway_visible());
    assert!(!MetabolicSleepPressureCandidateV1::policy_mutation());
    assert!(!MetabolicSleepPressureCandidateV1::identity_authority());
    assert!(!MetabolicSleepPressureCandidateV1::evidence_archive_authority());
}

#[test]
fn mapping_uses_no_random_or_wallclock() {
    let out = derive_hormone_modulation_output_v1(&state(HormoneStateRawV1 {
        dopamine_like: 3_333,
        serotonin_like: 4_444,
        cortisol_like: 5_555,
        arousal_like: 6_666,
        sleep_pressure: 7_777,
        novelty_pressure: 2_222,
        stability_pressure: 1_111,
    }));
    assert_eq!(
        derive_replay_sleep_candidates_v1(&out),
        derive_replay_sleep_candidates_v1(&out)
    );
}

#[test]
fn mapping_does_not_depend_on_replay_or_sleep_crates_if_option_a_chosen() {
    let out = derive_hormone_modulation_output_v1(&HormoneStateV1::neutral());
    let _ = derive_replay_sleep_candidates_v1(&out);
}
