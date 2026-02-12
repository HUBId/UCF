use sha2::{Digest, Sha256};

use crate::v0::ode::clamp01;

pub const NEURON_COUNT: usize = 64;
pub const MAX_SPIKES_PER_TICK: usize = 32;
const V_MIN: f32 = -2.0;
const V_MAX: f32 = 2.0;
const W_MIN: f32 = -2.0;
const W_MAX: f32 = 2.0;
const DV_CAP: f32 = 1.0;
const DW_CAP: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuronPopState {
    pub t: u64,
    pub v: [f32; NEURON_COUNT],
    pub w: [f32; NEURON_COUNT],
    pub digest: [u8; 32],
}

impl Default for NeuronPopState {
    fn default() -> Self {
        let mut state = Self {
            t: 0,
            v: [0.0; NEURON_COUNT],
            w: [0.0; NEURON_COUNT],
            digest: [0; 32],
        };
        state.digest = digest_neuron_state(&state);
        state
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuroInput {
    pub t: u64,
    pub pressure: f32,
    pub surprise: f32,
    pub risk: f32,
    pub confidence: f32,
    pub cortisol: f32,
    pub drive: f32,
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuroStateSummary {
    pub t: u64,
    pub arousal: f32,
    pub attention_gain: f32,
    pub excitability: f32,
    pub spike_rate: f32,
    pub digest: [u8; 32],
    pub evidence_chain_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuroSpike {
    pub neuron_id: u16,
    pub magnitude: f32,
    pub t: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeuroSpikeBatch {
    pub t: u64,
    pub spikes: Vec<NeuroSpike>,
    pub digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiophysModulation {
    pub arousal: f32,
    pub attention_gain: f32,
    pub excitability: f32,
    pub digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuroCfg {
    pub dt: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub input_base: f32,
    pub input_pressure: f32,
    pub input_surprise: f32,
    pub input_confidence: f32,
    pub input_drive: f32,
    pub input_cortisol: f32,
    pub bias_scale: f32,
    pub spike_threshold_base: f32,
}

impl Default for NeuroCfg {
    fn default() -> Self {
        Self {
            dt: 0.5,
            a: 0.7,
            b: 0.8,
            c: 0.6,
            input_base: 0.25,
            input_pressure: 0.40,
            input_surprise: 0.35,
            input_confidence: 0.25,
            input_drive: 0.30,
            input_cortisol: 0.45,
            bias_scale: 0.08,
            spike_threshold_base: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeuroStepOutput {
    pub state: NeuronPopState,
    pub summary: NeuroStateSummary,
    pub modulation: BiophysModulation,
    pub spikes: NeuroSpikeBatch,
    pub degraded: bool,
}

pub fn neuro_step(cfg: &NeuroCfg, prev: NeuronPopState, input: &NeuroInput) -> NeuroStepOutput {
    let dt = cfg.dt.max(0.000_1);
    let excitability_factor = clamp01(1.0 - 0.6 * input.cortisol + 0.3 * input.drive);
    let gain_factor = clamp01(0.5 + 0.5 * input.drive - 0.2 * input.cortisol);
    let threshold =
        (cfg.spike_threshold_base + 0.5 * input.cortisol - 0.2 * input.drive).clamp(0.6, 1.6);

    let mut next = prev;
    next.t = input.t;

    let mut mean_abs_v = 0.0;
    let mut spike_candidates: Vec<NeuroSpike> = Vec::new();
    let mut degraded = false;

    for idx in 0..NEURON_COUNT {
        let bias = neuron_bias(idx as u16) * cfg.bias_scale;
        let i_cur = (cfg.input_base
            + cfg.input_pressure * input.pressure.clamp(0.0, 1.0)
            + cfg.input_surprise * input.surprise.clamp(0.0, 1.0)
            - cfg.input_confidence * input.confidence.clamp(0.0, 1.0)
            + cfg.input_drive * input.drive.clamp(0.0, 1.0)
            - cfg.input_cortisol * input.cortisol.clamp(0.0, 1.0)
            + bias)
            * (0.5 + 0.5 * excitability_factor)
            * (0.75 + 0.5 * gain_factor);

        let v0 = prev.v[idx].clamp(V_MIN, V_MAX);
        let w0 = prev.w[idx].clamp(W_MIN, W_MAX);

        let (k1v, k1w) = fhn_derivatives(cfg, v0, w0, i_cur);
        let v_mid = (v0 + 0.5 * dt * k1v).clamp(V_MIN, V_MAX);
        let w_mid = (w0 + 0.5 * dt * k1w).clamp(W_MIN, W_MAX);
        let (k2v, k2w) = fhn_derivatives(cfg, v_mid, w_mid, i_cur);

        let v1 = (v0 + dt * k2v).clamp(V_MIN, V_MAX);
        let w1 = (w0 + dt * k2w).clamp(W_MIN, W_MAX);

        if !(v1.is_finite() && w1.is_finite()) {
            degraded = true;
            next = NeuronPopState::default();
            next.t = input.t;
            break;
        }

        next.v[idx] = v1;
        next.w[idx] = w1;
        mean_abs_v += v1.abs();

        if v1 > threshold && v0 <= threshold {
            spike_candidates.push(NeuroSpike {
                neuron_id: idx as u16,
                magnitude: (v1 - threshold).abs(),
                t: input.t,
            });
        }
    }

    let spikes = if degraded {
        NeuroSpikeBatch {
            t: input.t,
            spikes: Vec::new(),
            digest: digest_spike_batch(input.t, &[]),
        }
    } else {
        spike_candidates.sort_by(|a, b| {
            b.magnitude
                .total_cmp(&a.magnitude)
                .then_with(|| a.neuron_id.cmp(&b.neuron_id))
        });
        spike_candidates.truncate(MAX_SPIKES_PER_TICK);
        spike_candidates.sort_by_key(|s| s.neuron_id);
        let digest = digest_spike_batch(input.t, &spike_candidates);
        NeuroSpikeBatch {
            t: input.t,
            spikes: spike_candidates,
            digest,
        }
    };

    next.digest = digest_neuron_state(&next);

    let spike_rate = clamp01((spikes.spikes.len() as f32) / (MAX_SPIKES_PER_TICK as f32));
    let arousal = clamp01(if degraded {
        0.0
    } else {
        0.35 * (mean_abs_v / (NEURON_COUNT as f32)).clamp(0.0, 1.0)
            + 0.3 * input.pressure.clamp(0.0, 1.0)
            + 0.4 * input.cortisol.clamp(0.0, 1.0)
    });
    let attention_gain = clamp01(
        gain_factor
            * (1.0 - input.risk.clamp(0.0, 1.0) * 0.5)
            * (input.confidence.clamp(0.0, 1.0) * 0.5 + 0.5),
    );

    let summary = NeuroStateSummary {
        t: input.t,
        arousal,
        attention_gain,
        excitability: excitability_factor,
        spike_rate,
        digest: digest_summary(
            input.t,
            [arousal, attention_gain, excitability_factor, spike_rate],
            [next.digest, input.evidence_chain_digest, spikes.digest],
        ),
        evidence_chain_digest: input.evidence_chain_digest,
    };

    let modulation = BiophysModulation {
        arousal,
        attention_gain,
        excitability: excitability_factor,
        digest: digest_modulation(arousal, attention_gain, excitability_factor, summary.digest),
    };

    NeuroStepOutput {
        state: next,
        summary,
        modulation,
        spikes,
        degraded,
    }
}

fn fhn_derivatives(cfg: &NeuroCfg, v: f32, w: f32, i_cur: f32) -> (f32, f32) {
    let dv = (v - (v * v * v) / 3.0 - w + i_cur).clamp(-DV_CAP, DV_CAP);
    let dw = (cfg.a * (v + cfg.b - cfg.c * w)).clamp(-DW_CAP, DW_CAP);
    (dv, dw)
}

fn neuron_bias(neuron_id: u16) -> f32 {
    let mut x = u32::from(neuron_id)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(0xA5A5_5A5A);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    let unit = (x as f32) / (u32::MAX as f32);
    (unit * 2.0) - 1.0
}

fn digest_neuron_state(state: &NeuronPopState) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(state.t.to_le_bytes());
    for v in state.v {
        hasher.update(v.to_le_bytes());
    }
    for w in state.w {
        hasher.update(w.to_le_bytes());
    }
    hasher.finalize().into()
}

fn digest_spike_batch(t: u64, spikes: &[NeuroSpike]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(t.to_le_bytes());
    for s in spikes {
        hasher.update(s.neuron_id.to_le_bytes());
        hasher.update(s.magnitude.to_le_bytes());
    }
    hasher.finalize().into()
}

fn digest_summary(t: u64, values: [f32; 4], digests: [[u8; 32]; 3]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(t.to_le_bytes());
    for v in values {
        hasher.update(v.to_le_bytes());
    }
    for d in digests {
        hasher.update(d);
    }
    hasher.finalize().into()
}

fn digest_modulation(
    arousal: f32,
    attention_gain: f32,
    excitability: f32,
    summary_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(arousal.to_le_bytes());
    hasher.update(attention_gain.to_le_bytes());
    hasher.update(excitability.to_le_bytes());
    hasher.update(summary_digest);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::{neuro_step, NeuroCfg, NeuroInput, NeuronPopState, MAX_SPIKES_PER_TICK};

    fn mk_input(t: u64, cortisol: f32, drive: f32) -> NeuroInput {
        NeuroInput {
            t,
            pressure: 0.6,
            surprise: 0.5,
            risk: 0.4,
            confidence: 0.7,
            cortisol,
            drive,
            evidence_chain_digest: [3; 32],
        }
    }

    #[test]
    fn deterministic_sequence_produces_identical_digests() {
        let cfg = NeuroCfg::default();
        let mut a = NeuronPopState::default();
        let mut b = NeuronPopState::default();

        let mut da = Vec::new();
        let mut db = Vec::new();

        for t in 0..40 {
            let inp = mk_input(t, 0.3, 0.6);
            let oa = neuro_step(&cfg, a, &inp);
            let ob = neuro_step(&cfg, b, &inp);
            a = oa.state;
            b = ob.state;
            da.push((oa.summary.digest, oa.spikes.digest));
            db.push((ob.summary.digest, ob.spikes.digest));
        }

        assert_eq!(da, db);
        assert_eq!(a.digest, b.digest);
    }

    #[test]
    fn bounded_under_sustained_stress() {
        let cfg = NeuroCfg::default();
        let mut state = NeuronPopState::default();

        for t in 0..120 {
            let out = neuro_step(&cfg, state, &mk_input(t, 1.0, 0.0));
            state = out.state;
            for v in state.v {
                assert!((-2.0..=2.0).contains(&v));
            }
            for w in state.w {
                assert!((-2.0..=2.0).contains(&w));
            }
            assert!(out.spikes.spikes.len() <= MAX_SPIKES_PER_TICK);
        }
    }

    #[test]
    fn higher_cortisol_reduces_excitability() {
        let cfg = NeuroCfg::default();
        let mut low = NeuronPopState::default();
        let mut high = NeuronPopState::default();
        let mut low_sum = 0.0;
        let mut high_sum = 0.0;

        for t in 0..50 {
            let low_out = neuro_step(&cfg, low, &mk_input(t, 0.1, 0.5));
            let high_out = neuro_step(&cfg, high, &mk_input(t, 0.9, 0.5));
            low = low_out.state;
            high = high_out.state;
            low_sum += low_out.summary.excitability;
            high_sum += high_out.summary.excitability;
        }

        assert!(high_sum < low_sum);
    }

    #[test]
    fn spike_cap_is_deterministic() {
        let cfg = NeuroCfg {
            input_base: 1.2,
            ..NeuroCfg::default()
        };
        let mut state = NeuronPopState::default();
        let mut sizes = Vec::new();

        for t in 0..10 {
            let out = neuro_step(&cfg, state, &mk_input(t, 0.0, 1.0));
            sizes.push(out.spikes.spikes.len());
            state = out.state;
        }

        assert!(sizes.iter().all(|s| *s <= MAX_SPIKES_PER_TICK));
        assert_eq!(sizes, {
            let mut s2 = Vec::new();
            let mut state = NeuronPopState::default();
            for t in 0..10 {
                let out = neuro_step(&cfg, state, &mk_input(t, 0.0, 1.0));
                s2.push(out.spikes.spikes.len());
                state = out.state;
            }
            s2
        });
    }
}
