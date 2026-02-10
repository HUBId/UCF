use ucf_onn::v0::{OnnCore, PhaseDeg};

use super::{IitConfig, IitMonitor, MOD_JEPA, MOD_SSM};

#[test]
fn identical_phases_produce_high_phi() {
    let mut onn = OnnCore::new(1.0, 0.0);
    onn.register(MOD_JEPA, PhaseDeg(90.0));
    onn.register(MOD_SSM, PhaseDeg(90.0));

    let monitor = IitMonitor::new(IitConfig::default());
    let snap = monitor.compute(&onn);

    assert_eq!(snap.n_pairs, 1);
    assert!((snap.coherence_mean - 1.0).abs() < 1e-6);
    assert!((snap.coherence_min - 1.0).abs() < 1e-6);
    assert!((snap.phi - 1.0).abs() < 1e-6);
}

#[test]
fn opposite_phases_produce_low_phi() {
    let mut onn = OnnCore::new(1.0, 0.0);
    onn.register(MOD_JEPA, PhaseDeg(0.0));
    onn.register(MOD_SSM, PhaseDeg(180.0));

    let monitor = IitMonitor::new(IitConfig::default());
    let snap = monitor.compute(&onn);

    assert_eq!(snap.n_pairs, 1);
    assert!(snap.coherence_mean < 1e-6);
    assert!(snap.coherence_min < 1e-6);
    assert!(snap.phi < 1e-6);
}
