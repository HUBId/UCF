#![cfg(feature = "biophys")]

use biophys_core::{LifParams, LifState, NeuronId};
use biophys_runtime::BiophysRuntime;

fn make_runtime(max_spikes_per_step: usize) -> BiophysRuntime {
    let params = vec![
        LifParams {
            tau_ms: 2,
            v_rest: 0,
            v_reset: 0,
            v_threshold: 5,
        };
        10
    ];
    let states = vec![
        LifState {
            v: 0,
            refractory_steps: 0,
        };
        10
    ];
    BiophysRuntime::new(params, states, 1, max_spikes_per_step)
}

#[test]
fn lif_demo_spikes_and_digest_are_deterministic() {
    let mut runtime = make_runtime(10);
    let mut spike_steps = Vec::new();

    for (step, input) in [2, 4, 6, 8, 10].iter().enumerate() {
        let inputs = vec![*input; 10];
        let pop = runtime.step(&inputs);
        if !pop.spikes.is_empty() {
            spike_steps.push(step + 1);
            let expected: Vec<NeuronId> = (0..10).map(NeuronId).collect();
            assert_eq!(pop.spikes, expected);
        }
    }

    assert_eq!(spike_steps, vec![4, 5]);

    let mut runtime_repeat = make_runtime(10);
    for input in [2, 4, 6, 8, 10] {
        let inputs = vec![input; 10];
        runtime_repeat.step(&inputs);
    }

    assert_eq!(runtime.snapshot_digest(), runtime_repeat.snapshot_digest());
}

#[test]
fn spikes_are_bounded_and_deterministic() {
    let mut runtime = make_runtime(3);
    let inputs = [10; 10];
    let pop = runtime.step(&inputs);
    let expected: Vec<NeuronId> = (0..3).map(NeuronId).collect();
    assert_eq!(pop.spikes, expected);
}
