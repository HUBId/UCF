use crate::hormone_state_v1::{HormoneStateRawV1, HormoneStateV1, NormalizedHormoneLevelV1};

const SCALE: i64 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HormoneInputFrameV1 {
    pub reward_signal: i64,
    pub novelty_signal: i64,
    pub threat_signal: i64,
    pub fatigue_signal: i64,
    pub inconsistency_signal: i64,
    pub replay_density: i64,
    pub policy_violation_pressure: i64,
}

impl HormoneInputFrameV1 {
    pub const fn neutral() -> Self {
        Self {
            reward_signal: 0,
            novelty_signal: 0,
            threat_signal: 0,
            fatigue_signal: 0,
            inconsistency_signal: 0,
            replay_density: 0,
            policy_violation_pressure: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HormoneUpdateConfigV1 {
    pub dopamine_gain: i64,
    pub serotonin_gain: i64,
    pub cortisol_gain: i64,
    pub arousal_gain: i64,
    pub sleep_gain: i64,
    pub novelty_gain: i64,
    pub stability_gain: i64,
    pub decay_rate: i64,
    pub clamp_min: i64,
    pub clamp_max: i64,
}

impl HormoneUpdateConfigV1 {
    pub const fn bounded_default() -> Self {
        Self {
            dopamine_gain: 2_000,
            serotonin_gain: 1_000,
            cortisol_gain: 2_000,
            arousal_gain: 1_200,
            sleep_gain: 1_500,
            novelty_gain: 1_300,
            stability_gain: 1_700,
            decay_rate: 600,
            clamp_min: 0,
            clamp_max: 10_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HormoneModulationOutputV1 {
    pub attention_gain: i64,
    pub learning_rate_multiplier: i64,
    pub replay_priority_multiplier: i64,
    pub noise_scale: i64,
    pub consolidation_gate: i64,
    pub sleep_pressure_delta: i64,
    pub risk_damping: i64,
}

impl HormoneModulationOutputV1 {
    pub const fn runtime_authority() -> bool {
        false
    }

    pub const fn gateway_authority() -> bool {
        false
    }

    pub const fn policy_mutation() -> bool {
        false
    }
}

pub fn update_hormone_state_v1(
    prev_state: HormoneStateV1,
    input_frame: HormoneInputFrameV1,
    config: HormoneUpdateConfigV1,
) -> (HormoneStateV1, HormoneModulationOutputV1) {
    let clamp_min = config
        .clamp_min
        .max(i64::from(NormalizedHormoneLevelV1::MIN));
    let clamp_max = config
        .clamp_max
        .min(i64::from(NormalizedHormoneLevelV1::MAX));

    let dopamine_delta = bounded_scale(
        saturating_add(input_frame.reward_signal, input_frame.novelty_signal),
        config.dopamine_gain,
    );
    let serotonin_delta = bounded_scale(
        saturating_sub(input_frame.reward_signal, input_frame.threat_signal),
        config.serotonin_gain,
    );
    let cortisol_delta = bounded_scale(
        saturating_add(
            input_frame.threat_signal,
            input_frame.policy_violation_pressure,
        ),
        config.cortisol_gain,
    );
    let arousal_delta = bounded_scale(
        saturating_add(input_frame.replay_density, input_frame.threat_signal),
        config.arousal_gain,
    );
    let sleep_delta = bounded_scale(input_frame.fatigue_signal, config.sleep_gain);
    let novelty_delta = bounded_scale(input_frame.novelty_signal, config.novelty_gain);
    let stability_delta = -bounded_scale(
        saturating_add(
            input_frame.inconsistency_signal,
            input_frame.policy_violation_pressure,
        ),
        config.stability_gain,
    );

    let next_raw = HormoneStateRawV1 {
        dopamine_like: decay_step(
            i64::from(prev_state.dopamine_like.as_units()),
            dopamine_delta,
            config.decay_rate,
            clamp_min,
            clamp_max,
        ),
        serotonin_like: decay_step(
            i64::from(prev_state.serotonin_like.as_units()),
            serotonin_delta,
            config.decay_rate,
            clamp_min,
            clamp_max,
        ),
        cortisol_like: decay_step(
            i64::from(prev_state.cortisol_like.as_units()),
            cortisol_delta,
            config.decay_rate,
            clamp_min,
            clamp_max,
        ),
        arousal_like: decay_step(
            i64::from(prev_state.arousal_like.as_units()),
            arousal_delta,
            config.decay_rate,
            clamp_min,
            clamp_max,
        ),
        sleep_pressure: decay_step(
            i64::from(prev_state.sleep_pressure.as_units()),
            sleep_delta,
            config.decay_rate,
            clamp_min,
            clamp_max,
        ),
        novelty_pressure: decay_step(
            i64::from(prev_state.novelty_pressure.as_units()),
            novelty_delta,
            config.decay_rate,
            clamp_min,
            clamp_max,
        ),
        stability_pressure: decay_step(
            i64::from(prev_state.stability_pressure.as_units()),
            stability_delta,
            config.decay_rate,
            clamp_min,
            clamp_max,
        ),
    };

    let next_state = HormoneStateV1::new_clamped(next_raw);
    let modulation_output = derive_hormone_modulation_output_v1(&next_state);
    (next_state, modulation_output)
}

/// Derives an advisory-only modulation vector from a bounded hormone state.
///
/// Semantics are deterministic and integer-only and must not be interpreted as
/// runtime, gateway, policy, identity, archive, or evidence authority.
pub fn derive_hormone_modulation_output_v1(state: &HormoneStateV1) -> HormoneModulationOutputV1 {
    let clamp_min = i64::from(NormalizedHormoneLevelV1::MIN);
    let clamp_max = i64::from(NormalizedHormoneLevelV1::MAX);

    let attention_gain = clamp(
        i64::from(state.dopamine_like.as_units())
            + (i64::from(state.novelty_pressure.as_units()) / 4)
            + (i64::from(state.arousal_like.as_units()) / 5)
            - (i64::from(state.cortisol_like.as_units()) / 5),
        clamp_min,
        clamp_max,
    );
    let learning_rate_multiplier = clamp(
        i64::from(state.dopamine_like.as_units())
            + (i64::from(state.novelty_pressure.as_units()) / 4)
            + (i64::from(state.serotonin_like.as_units()) / 6)
            - (i64::from(state.cortisol_like.as_units()) / 4),
        clamp_min,
        clamp_max,
    );
    let replay_priority_multiplier = clamp(
        i64::from(state.novelty_pressure.as_units())
            + (i64::from(state.stability_pressure.as_units()) / 5)
            + (i64::from(state.arousal_like.as_units()) / 5)
            - (i64::from(state.cortisol_like.as_units()) / 8),
        clamp_min,
        clamp_max,
    );
    let noise_scale = clamp(
        i64::from(state.cortisol_like.as_units()) + (i64::from(state.arousal_like.as_units()) / 6)
            - (i64::from(state.stability_pressure.as_units()) / 5)
            - (i64::from(state.serotonin_like.as_units()) / 8),
        clamp_min,
        clamp_max,
    );
    let consolidation_gate = clamp(
        i64::from(state.stability_pressure.as_units())
            + (i64::from(state.serotonin_like.as_units()) / 5)
            - (i64::from(state.cortisol_like.as_units()) / 5)
            - (i64::from(state.novelty_pressure.as_units()) / 8),
        clamp_min,
        clamp_max,
    );
    let sleep_pressure_delta = clamp(
        i64::from(state.sleep_pressure.as_units()) - i64::from(NormalizedHormoneLevelV1::NEUTRAL),
        -(clamp_max - clamp_min),
        clamp_max - clamp_min,
    );
    let risk_damping = clamp(
        i64::from(state.cortisol_like.as_units())
            + (i64::from(state.stability_pressure.as_units()) / 4)
            + (i64::from(state.serotonin_like.as_units()) / 8),
        clamp_min,
        clamp_max,
    );

    HormoneModulationOutputV1 {
        attention_gain,
        learning_rate_multiplier,
        replay_priority_multiplier,
        noise_scale,
        consolidation_gate,
        sleep_pressure_delta,
        risk_damping,
    }
}

fn saturating_add(a: i64, b: i64) -> i64 {
    a.saturating_add(b)
}

fn saturating_sub(a: i64, b: i64) -> i64 {
    a.saturating_sub(b)
}

fn bounded_scale(value: i64, gain: i64) -> i64 {
    value.saturating_mul(gain).div_euclid(SCALE)
}

fn decay_step(current: i64, delta: i64, decay_rate: i64, clamp_min: i64, clamp_max: i64) -> i64 {
    let with_delta = saturating_add(current, delta);
    let neutral = i64::from(NormalizedHormoneLevelV1::NEUTRAL);
    let drift = bounded_scale(saturating_sub(with_delta, neutral), decay_rate);
    clamp(saturating_sub(with_delta, drift), clamp_min, clamp_max)
}

fn clamp(value: i64, min_value: i64, max_value: i64) -> i64 {
    value.clamp(min_value, max_value)
}
