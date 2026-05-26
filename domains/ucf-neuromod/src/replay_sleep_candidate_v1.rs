use crate::hormone_state_v1::NormalizedHormoneLevelV1;
use crate::hormone_update_v1::HormoneModulationOutputV1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetabolicReplayPriorityCandidateV1 {
    pub priority_hint: i64,
    pub novelty_component: i64,
    pub stability_component: i64,
    pub arousal_component: i64,
    pub risk_damping_component: i64,
}

impl MetabolicReplayPriorityCandidateV1 {
    pub const fn advisory_only() -> bool {
        true
    }

    pub const fn scheduler_authority() -> bool {
        false
    }

    pub const fn replay_applied() -> bool {
        false
    }

    pub const fn gateway_visible() -> bool {
        false
    }

    pub const fn policy_mutation() -> bool {
        false
    }

    pub const fn evidence_archive_authority() -> bool {
        false
    }

    pub const fn identity_authority() -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetabolicSleepPressureCandidateV1 {
    pub pressure_hint: i64,
    pub sleep_delta_component: i64,
    pub risk_damping_component: i64,
}

impl MetabolicSleepPressureCandidateV1 {
    pub const fn advisory_only() -> bool {
        true
    }

    pub const fn scheduler_authority() -> bool {
        false
    }

    pub const fn sleep_completed() -> bool {
        false
    }

    pub const fn gateway_visible() -> bool {
        false
    }

    pub const fn policy_mutation() -> bool {
        false
    }

    pub const fn evidence_archive_authority() -> bool {
        false
    }

    pub const fn identity_authority() -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetabolicReplaySleepCandidatesV1 {
    pub replay: MetabolicReplayPriorityCandidateV1,
    pub sleep: MetabolicSleepPressureCandidateV1,
}

pub fn derive_replay_sleep_candidates_v1(
    output: &HormoneModulationOutputV1,
) -> MetabolicReplaySleepCandidatesV1 {
    let level_max = i64::from(NormalizedHormoneLevelV1::MAX);
    let level_min = i64::from(NormalizedHormoneLevelV1::MIN);

    let novelty_component = clamp(output.replay_priority_multiplier / 2, level_min, level_max);
    let stability_component = clamp(output.replay_priority_multiplier / 3, level_min, level_max);
    let arousal_component = clamp(output.attention_gain / 5, level_min, level_max);
    let risk_damping_component = clamp(output.risk_damping / 4, level_min, level_max);

    let replay_priority_base = novelty_component
        .saturating_add(stability_component)
        .saturating_add(arousal_component);
    let priority_hint = clamp(
        replay_priority_base.saturating_sub(risk_damping_component),
        level_min,
        level_max,
    );

    let sleep_delta_component = clamp(
        output.sleep_pressure_delta + level_max,
        level_min,
        level_max,
    );
    let sleep_risk_damping_component = clamp(output.risk_damping / 5, level_min, level_max);
    let pressure_hint = clamp(
        sleep_delta_component.saturating_sub(sleep_risk_damping_component),
        level_min,
        level_max,
    );

    MetabolicReplaySleepCandidatesV1 {
        replay: MetabolicReplayPriorityCandidateV1 {
            priority_hint,
            novelty_component,
            stability_component,
            arousal_component,
            risk_damping_component,
        },
        sleep: MetabolicSleepPressureCandidateV1 {
            pressure_hint,
            sleep_delta_component,
            risk_damping_component: sleep_risk_damping_component,
        },
    }
}

fn clamp(value: i64, min: i64, max: i64) -> i64 {
    value.clamp(min, max)
}
