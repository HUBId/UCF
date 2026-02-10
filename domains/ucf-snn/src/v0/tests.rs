use ucf_onn::v0::PhaseDeg;

use crate::v0::{encode, FeatureEvent, SnnEncodeCfg, SpikePayload, TtfsMs};

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
