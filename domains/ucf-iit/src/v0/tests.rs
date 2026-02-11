use super::{iit_push_and_eval, IitCfg, IitSample, IitState};

fn sample(
    now_ms: u64,
    tcf_lock: f32,
    ssm_gate: f32,
    nsr_risk: f32,
    cde_conf: f32,
    spike_rate: f32,
) -> IitSample {
    IitSample {
        now_ms,
        tcf_lock,
        ssm_gate,
        nsr_risk,
        cde_conf,
        spike_rate,
    }
}

#[test]
fn deterministic_for_same_sequence() {
    let cfg = IitCfg {
        min_samples: 3,
        ..IitCfg::default_v0()
    };
    let seq = [
        sample(1, 0.2, 0.2, 0.8, 0.1, 0.1),
        sample(2, 0.3, 0.3, 0.7, 0.2, 0.2),
        sample(3, 0.4, 0.4, 0.6, 0.3, 0.3),
        sample(4, 0.5, 0.5, 0.5, 0.4, 0.4),
    ];

    let mut a = IitState::new(&cfg);
    let mut b = IitState::new(&cfg);

    for s in seq {
        let ta = iit_push_and_eval(&cfg, &mut a, s.clone());
        let tb = iit_push_and_eval(&cfg, &mut b, s);
        assert_eq!(ta, tb);
    }
}

#[test]
fn high_coherence_raises_phi() {
    let cfg = IitCfg {
        min_samples: 2,
        ..IitCfg::default_v0()
    };

    let mut low = IitState::new(&cfg);
    iit_push_and_eval(&cfg, &mut low, sample(1, 0.1, 0.8, 0.1, 0.1, 0.9));
    let low_tick = iit_push_and_eval(&cfg, &mut low, sample(2, 0.1, 0.8, 0.1, 0.1, 0.9));

    let mut high = IitState::new(&cfg);
    iit_push_and_eval(&cfg, &mut high, sample(1, 0.9, 0.8, 0.1, 0.9, 0.9));
    let high_tick = iit_push_and_eval(&cfg, &mut high, sample(2, 0.9, 0.8, 0.1, 0.9, 0.9));

    assert!(high_tick.phi_proxy > low_tick.phi_proxy);
}

#[test]
fn enforce_triggers_below_threshold() {
    let cfg = IitCfg {
        min_samples: 2,
        enforce_threshold: 0.8,
        ..IitCfg::default_v0()
    };
    let mut st = IitState::new(&cfg);

    iit_push_and_eval(&cfg, &mut st, sample(1, 0.1, 0.0, 1.0, 0.0, 0.0));
    let tick = iit_push_and_eval(&cfg, &mut st, sample(2, 0.1, 0.0, 1.0, 0.0, 0.0));

    assert!(tick.enforce);
    assert!(tick.phi_proxy < cfg.enforce_threshold);
}
