use super::{
    modulate_hh, FieldEvent, FieldEventKind, FieldUpdateCfg, HhParams, ModulationCfg,
    NeuromodulatorField, Unit01,
};

#[test]
fn unit01_clamps_nan_neg_and_overflow() {
    assert_eq!(Unit01::new(f32::NAN).get(), 0.0);
    assert_eq!(Unit01::new(-0.3).get(), 0.0);
    assert_eq!(Unit01::new(1.7).get(), 1.0);
}

#[test]
fn apply_event_obeys_clamp_and_rate_limit() {
    let cfg = FieldUpdateCfg {
        decay_per_s: 1.0,
        max_delta_per_tick: 0.1,
    };
    let start = NeuromodulatorField::default();
    let reward = FieldEvent {
        kind: FieldEventKind::Reward,
        magnitude: Unit01::new(1.0),
    };

    let next = start.apply_event(reward, cfg);
    assert_eq!(next.dopamine.get(), start.dopamine.get() + 0.1);
    assert_eq!(next.acetylcholine.get(), start.acetylcholine.get() + 0.1);

    let inhibit = FieldEvent {
        kind: FieldEventKind::Inhibit,
        magnitude: Unit01::new(1.0),
    };
    let saturated = NeuromodulatorField {
        gaba: Unit01::new(0.98),
        ..NeuromodulatorField::default()
    }
    .apply_event(inhibit, cfg);
    assert_eq!(saturated.gaba.get(), 1.0);
}

#[test]
fn decay_towards_moves_towards_baseline() {
    let cfg = FieldUpdateCfg {
        decay_per_s: 1.0,
        max_delta_per_tick: 1.0,
    };
    let current = NeuromodulatorField {
        dopamine: Unit01::new(1.0),
        ..NeuromodulatorField::default()
    };
    let baseline = NeuromodulatorField::default();
    let next = current.decay_towards(baseline, 0.5, cfg);

    assert!(next.dopamine.get() < current.dopamine.get());
    assert!(next.dopamine.get() > baseline.dopamine.get());
}

#[test]
fn modulate_hh_applies_expected_directional_effects() {
    let base = HhParams::default();
    let field = NeuromodulatorField {
        dopamine: Unit01::new(1.0),
        serotonin: Unit01::new(1.0),
        gaba: Unit01::new(1.0),
        glutamate: Unit01::new(1.0),
        endorphin: Unit01::new(1.0),
        ..NeuromodulatorField::default()
    };

    let out = modulate_hh(base, field, ModulationCfg::default());

    assert!(out.g_na > base.g_na);
    assert!(out.g_k > base.g_k);
    assert!(out.threshold_shift_mv > base.threshold_shift_mv);
    assert!(out.max_firing_hz < base.max_firing_hz);

    let field_excite_only = NeuromodulatorField {
        glutamate: Unit01::new(1.0),
        gaba: Unit01::new(0.0),
        oxytocin: Unit01::new(0.0),
        ..NeuromodulatorField::default()
    };
    let excite = modulate_hh(base, field_excite_only, ModulationCfg::default());
    assert!(excite.threshold_shift_mv < base.threshold_shift_mv);
}
