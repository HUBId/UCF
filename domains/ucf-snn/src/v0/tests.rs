use ucf_onn::v0::PhaseDeg;

use crate::v0::{
    encode, encode_ttfsp, snn_emit, FeatureEvent, SnnCfg, SnnEncodeCfg, SpikePayload, SpikeSrc,
    TtfsMs,
};

#[test]
fn novelty_gate_filters_and_emits() {
    let cfg = SnnEncodeCfg {
        novelty_gate: 0.5,
        ..SnnEncodeCfg::default()
    };
    let feats = [
        FeatureEvent {
            chan: 1,
            intensity: 1.0,
            novelty: 0.49,
        },
        FeatureEvent {
            chan: 2,
            intensity: 1.0,
            novelty: 0.5,
        },
    ];

    let spikes = encode(10, None, cfg, &feats);

    assert_eq!(spikes.len(), 1);
    assert_eq!(spikes[0].chan, 2);
}

#[test]
fn ttfs_maps_intensity_extremes() {
    let cfg = SnnEncodeCfg {
        ttfs_max_ms: 77,
        ..SnnEncodeCfg::default()
    };

    let spikes = encode(
        5,
        None,
        cfg,
        &[
            FeatureEvent {
                chan: 1,
                intensity: 1.0,
                novelty: 1.0,
            },
            FeatureEvent {
                chan: 2,
                intensity: 0.0,
                novelty: 1.0,
            },
        ],
    );

    assert_eq!(spikes[0].payload, SpikePayload::Ttfs { ttfs_ms: TtfsMs(0) });
    assert_eq!(
        spikes[1].payload,
        SpikePayload::Ttfs {
            ttfs_ms: TtfsMs(77)
        }
    );
}

#[test]
fn encoder_sorts_by_time_then_chan() {
    let cfg = SnnEncodeCfg {
        ttfs_max_ms: 100,
        ..SnnEncodeCfg::default()
    };

    let spikes = encode(
        100,
        None,
        cfg,
        &[
            FeatureEvent {
                chan: 3,
                intensity: 0.0,
                novelty: 1.0,
            },
            FeatureEvent {
                chan: 1,
                intensity: 1.0,
                novelty: 1.0,
            },
            FeatureEvent {
                chan: 2,
                intensity: 1.0,
                novelty: 1.0,
            },
        ],
    );

    let order: Vec<(u64, u16)> = spikes.iter().map(|s| (s.t.0, s.chan)).collect();
    assert_eq!(order, vec![(100, 1), (100, 2), (200, 3)]);
}

#[test]
fn encoder_normalizes_phase() {
    let spikes = encode(
        0,
        Some(PhaseDeg(-30.0)),
        SnnEncodeCfg::default(),
        &[FeatureEvent {
            chan: 1,
            intensity: 1.0,
            novelty: 1.0,
        }],
    );

    assert_eq!(spikes[0].phase, Some(PhaseDeg(330.0)));
}

#[test]
fn encode_ttfsp_monotonic_by_intensity() {
    let low = encode_ttfsp(0.5, 0.2);
    let high = encode_ttfsp(0.5, 0.8);
    assert!(high < low);
}

#[test]
fn encode_ttfsp_phase_bias_earlier_phase_is_earlier_spike() {
    let early_phase = encode_ttfsp(0.1, 0.5);
    let late_phase = encode_ttfsp(0.9, 0.5);
    assert!(early_phase < late_phase);
}

#[test]
fn snn_emit_routing_thresholds_for_sae() {
    let cfg = SnnCfg::default_v0();
    let out = snn_emit(
        &cfg,
        42,
        0.25,
        SpikeSrc::Sae,
        &[(1, 0.70), (2, 0.50), (3, 0.31), (4, 0.2)],
    );
    assert_eq!(out.fired_count, 3);
    assert_eq!(out.suppressed_count, 1);
    assert_eq!(out.emitted[0].dst, crate::v0::SpikeDst::Nsr);
    assert_eq!(out.emitted[1].dst, crate::v0::SpikeDst::Cde);
    assert_eq!(out.emitted[2].dst, crate::v0::SpikeDst::Ssm);
}

#[test]
fn snn_emit_caps_and_keeps_top_intensities() {
    let mut cfg = SnnCfg::default_v0();
    cfg.max_events_per_tick = 2;
    let out = snn_emit(
        &cfg,
        7,
        0.5,
        SpikeSrc::Sae,
        &[(10, 0.70), (11, 0.90), (12, 0.80)],
    );

    let ids: Vec<u32> = out.emitted.iter().map(|e| e.feature_id).collect();
    assert_eq!(ids, vec![11, 12]);
    assert_eq!(out.suppressed_count, 1);
}

#[test]
fn snn_emit_is_deterministic_for_fixed_inputs() {
    let cfg = SnnCfg::default_v0();
    let candidates = vec![(10, 0.65), (11, 0.6), (12, 0.45)];
    let first = snn_emit(&cfg, 100, 0.33, SpikeSrc::Lens, &candidates);
    let second = snn_emit(&cfg, 100, 0.33, SpikeSrc::Lens, &candidates);
    assert_eq!(first, second);
}
