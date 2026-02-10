use super::{
    hpa_step, modulate_hh, FieldEvent, FieldEventKind, FieldUpdateCfg, HhParams, HpaCfg, HpaState,
    ModulationCfg, NeuromodulatorField, Unit01,
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

#[test]
fn hpa_step_increases_cortisol_under_sustained_stress() {
    let cfg = HpaCfg::default();
    let mut st = HpaState::default();

    for _ in 0..200 {
        st = hpa_step(st, 0.9, 0.1, cfg);
    }

    assert!(st.cortisol > 0.1);
}

#[test]
fn cortisol_negative_feedback_reduces_crh_when_cortisol_high() {
    let cfg = HpaCfg::default();
    let mut st = HpaState {
        crh: 0.9,
        acth: 0.6,
        cortisol: 1.0,
    };

    for _ in 0..50 {
        st = hpa_step(st, 0.1, 0.1, cfg);
    }

    assert!(st.crh < 0.9);
}

#[test]
fn with_hpa_lowers_serotonin_and_raises_gaba_when_cortisol_high() {
    let field = NeuromodulatorField::default();
    let out = field.with_hpa(HpaState {
        cortisol: 1.0,
        ..HpaState::default()
    });

    assert!(out.serotonin.get() < field.serotonin.get());
    assert!(out.gaba.get() > field.gaba.get());
}

#[test]
fn microcircuit_new_ring_has_expected_counts_and_adjacency() {
    let n = 32;
    let micro = super::Microcircuit::new_ring(n);

    assert_eq!(micro.neurons.len(), n);
    assert_eq!(micro.synapses.len(), n * 2);
    assert_eq!(micro.outgoing.len(), n);
    assert!(micro.outgoing.iter().all(|edges| !edges.is_empty()));
}

#[test]
fn microcircuit_spikes_under_high_glutamate_low_gaba() {
    let mut micro = super::Microcircuit::new_ring(32);
    let field = NeuromodulatorField {
        glutamate: Unit01::new(1.0),
        gaba: Unit01::new(0.0),
        ..NeuromodulatorField::default()
    };

    let mut total_spikes = 0_usize;
    for _ in 0..100 {
        let out = micro.step(field, 0.01);
        total_spikes += out.spikes.len();
    }

    assert!(total_spikes > 0);
}

#[test]
fn microcircuit_stays_quiet_under_high_gaba_low_glutamate() {
    let mut micro = super::Microcircuit::new_ring(32);
    let field = NeuromodulatorField {
        glutamate: Unit01::new(0.0),
        gaba: Unit01::new(1.0),
        ..NeuromodulatorField::default()
    };

    let mut total_spikes = 0_usize;
    for _ in 0..100 {
        let out = micro.step(field, 0.01);
        total_spikes += out.spikes.len();
    }

    assert!(total_spikes <= 2);
}

#[test]
fn wrap_phase_normalizes_to_tau_interval() {
    use super::{wrap_phase, TAU};

    let xs = [-9.0, -0.1, 0.0, 0.1, TAU, TAU + 0.2, 42.0];
    for x in xs {
        let w = wrap_phase(x);
        assert!((0.0..TAU).contains(&w));
    }
}

#[test]
fn phase_lock_identity_and_pi_offset() {
    use super::{phase_lock, Osc};

    let a = Osc {
        phase: 0.7,
        omega_hz: 40.0,
    };
    let same = phase_lock(a, a);
    assert!((same - 1.0).abs() <= 1e-6);

    let b = Osc {
        phase: a.phase + core::f32::consts::PI,
        omega_hz: a.omega_hz,
    };
    let opposite = phase_lock(a, b);
    assert!(opposite <= 1e-5);
}

#[test]
fn ttfs_phase_maps_window_bounds() {
    use super::{ttfs_phase, SpikeCodecCfg};

    let cfg = SpikeCodecCfg {
        max_phase: core::f32::consts::PI * 2.0 - 0.001,
        ..SpikeCodecCfg::default()
    };
    let start = ttfs_phase(0, cfg);
    let end = ttfs_phase(cfg.window_ms, cfg);

    assert!((start - cfg.min_phase).abs() <= 1e-6);
    assert!((end - cfg.max_phase).abs() <= 1e-6);
}
