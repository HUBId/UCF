use crate::chemistry::NeuromodState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeuroTx {
    Glu,
    Gaba,
    Ach,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modulator {
    Dopamine,
    Serotonin,
    Oxytocin,
    Endorphin,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HhParams {
    pub cm: f32,
    pub g_na: f32,
    pub e_na: f32,
    pub g_k: f32,
    pub e_k: f32,
    pub g_l: f32,
    pub e_l: f32,
    pub v_th: f32,
    pub max_fire_hz: f32,
}

impl HhParams {
    pub fn default_v0() -> Self {
        Self {
            cm: 1.0,
            g_na: 100.0,
            e_na: 50.0,
            g_k: 30.0,
            e_k: -77.0,
            g_l: 0.3,
            e_l: -54.4,
            v_th: -50.0,
            max_fire_hz: 120.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HhState {
    pub v: f32,
    pub m: f32,
    pub h: f32,
    pub n: f32,
    pub last_spike_ms: i64,
    pub spike_count: u32,
}

impl HhState {
    pub fn resting_v0() -> Self {
        let v = -65.0;
        Self {
            v,
            m: alpha_m(v) / (alpha_m(v) + beta_m(v)),
            h: alpha_h(v) / (alpha_h(v) + beta_h(v)),
            n: alpha_n(v) / (alpha_n(v) + beta_n(v)),
            last_spike_ms: i64::MIN / 4,
            spike_count: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Synapse {
    pub pre: u32,
    pub post: u32,
    pub tx: NeuroTx,
    pub w: f32,
    pub delay_ms: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HhNeuron {
    pub id: u32,
    pub p: HhParams,
    pub s: HhState,
    pub sens_dopa: f32,
    pub sens_5ht: f32,
    pub sens_oxy: f32,
    pub sens_end: f32,
}

impl HhNeuron {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            p: HhParams::default_v0(),
            s: HhState::resting_v0(),
            sens_dopa: 1.0,
            sens_5ht: 1.0,
            sens_oxy: 1.0,
            sens_end: 1.0,
        }
    }
}

fn safe_vtrap(x: f32, y: f32) -> f32 {
    if x.abs() < 1e-4 {
        y * (1.0 - x / (2.0 * y))
    } else {
        x / ((x / y).exp() - 1.0)
    }
}

fn alpha_m(v: f32) -> f32 {
    0.1 * safe_vtrap(25.0 - (v + 65.0), 10.0).max(0.0)
}
fn beta_m(v: f32) -> f32 {
    4.0 * (-(v + 65.0) / 18.0).exp().clamp(0.0, 1_000.0)
}
fn alpha_h(v: f32) -> f32 {
    0.07 * (-(v + 65.0) / 20.0).exp().clamp(0.0, 1_000.0)
}
fn beta_h(v: f32) -> f32 {
    (1.0 / (1.0 + ((30.0 - (v + 65.0)) / 10.0).exp())).clamp(0.0, 1_000.0)
}
fn alpha_n(v: f32) -> f32 {
    0.01 * safe_vtrap(10.0 - (v + 65.0), 10.0).max(0.0)
}
fn beta_n(v: f32) -> f32 {
    0.125 * (-(v + 65.0) / 80.0).exp().clamp(0.0, 1_000.0)
}

pub fn hh_step(
    now_ms: u64,
    dt_ms: f32,
    neuron: &mut HhNeuron,
    i_ext: f32,
    mods: &NeuromodState,
) -> bool {
    let dt = (dt_ms / 1000.0).max(1e-5);
    let p = &neuron.p;

    let g_na_eff = p.g_na * (1.0 + 0.25 * neuron.sens_dopa * mods.dopa).clamp(0.2, 3.0);
    let g_k_eff = p.g_k * (1.0 + 0.25 * neuron.sens_5ht * mods.serotonin).clamp(0.2, 3.0);
    let v_th_eff = p.v_th + 2.0 * (mods.serotonin * neuron.sens_5ht);
    let max_fire_hz_eff =
        (p.max_fire_hz * (1.0 - 0.35 * neuron.sens_end * mods.endorphin)).max(1.0);

    let v = neuron.s.v.clamp(-120.0, 80.0);
    let am = alpha_m(v);
    let bm = beta_m(v);
    let ah = alpha_h(v);
    let bh = beta_h(v);
    let an = alpha_n(v);
    let bn = beta_n(v);

    neuron.s.m = (neuron.s.m + dt * (am * (1.0 - neuron.s.m) - bm * neuron.s.m)).clamp(0.0, 1.0);
    neuron.s.h = (neuron.s.h + dt * (ah * (1.0 - neuron.s.h) - bh * neuron.s.h)).clamp(0.0, 1.0);
    neuron.s.n = (neuron.s.n + dt * (an * (1.0 - neuron.s.n) - bn * neuron.s.n)).clamp(0.0, 1.0);

    let i_na = g_na_eff * neuron.s.m.powi(3) * neuron.s.h * (v - p.e_na);
    let i_k = g_k_eff * neuron.s.n.powi(4) * (v - p.e_k);
    let i_l = p.g_l * (v - p.e_l);

    let dv_dt = (i_ext - i_na - i_k - i_l) / p.cm.max(0.01);
    neuron.s.v = (v + dt * dv_dt).clamp(-120.0, 80.0);

    let min_isi_ms = 1000.0 / max_fire_hz_eff;
    let elapsed = now_ms as i64 - neuron.s.last_spike_ms;
    let can_fire = (elapsed as f32) >= min_isi_ms;
    let spiked = neuron.s.v >= v_th_eff && can_fire;
    if spiked {
        neuron.s.last_spike_ms = now_ms as i64;
        neuron.s.spike_count = neuron.s.spike_count.saturating_add(1);
    }
    spiked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hh_rest_stable_no_nan() {
        let mods = NeuromodState::baseline();
        let mut n = HhNeuron::new(0);
        for t in 0..10_000 {
            let _ = hh_step(t, 1.0, &mut n, 0.0, &mods);
            assert!(n.s.v.is_finite());
            assert!(n.s.m.is_finite());
            assert!(n.s.h.is_finite());
            assert!(n.s.n.is_finite());
        }
    }

    fn spike_count(ext: f32, mods: NeuromodState) -> u32 {
        let mut n = HhNeuron::new(1);
        for t in 0..2_000 {
            let _ = hh_step(t, 1.0, &mut n, ext, &mods);
        }
        n.s.spike_count
    }

    #[test]
    fn dopamine_increases_spikes() {
        let base = spike_count(14.0, NeuromodState::baseline());
        let mut mods = NeuromodState::baseline();
        mods.dopa = 1.0;
        let high = spike_count(14.0, mods);
        assert!(high > base, "high={high} base={base}");
    }

    #[test]
    fn serotonin_not_more_spikes() {
        let base = spike_count(14.0, NeuromodState::baseline());
        let mut mods = NeuromodState::baseline();
        mods.serotonin = 1.0;
        let high = spike_count(14.0, mods);
        assert!(high <= base, "high={high} base={base}");
    }
}
