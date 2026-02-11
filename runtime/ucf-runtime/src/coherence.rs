#![forbid(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};

use blake3::Hasher;
use ucf_compute::Spike;

const DOMAIN_COHERENCE: &[u8] = b"ucf.runtime.coherence.v0";
const BUCKETS: u32 = 64;
const MAX_SPIKES_PER_TICK: usize = 256;
const MAX_EVENTS_PER_BATCH: usize = 128;
const MAX_SUBSCRIBERS: usize = 32;
const MAX_SELECTED: usize = 8;
const MAX_HISTORY: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct SpikeEvent {
    pub feature_id: u32,
    pub magnitude: f32,
    pub t: u64,
    pub source_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpikeBatch {
    pub t: u64,
    pub spikes: Vec<SpikeEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterestProfile {
    TopKFeatures(Vec<u32>),
    HashBuckets(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoherenceContext {
    pub pressure: f32,
    pub surprise: f32,
    pub risk: f32,
    pub confidence: f32,
    pub coherence: f32,
    pub instability: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubscriberOutput {
    pub attention_delta: f32,
    pub notes: Vec<String>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subscriber {
    pub module_id: u8,
    pub name: &'static str,
    pub interest: InterestProfile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeRoutingSummary {
    pub t: u64,
    pub spikes_in: usize,
    pub spikes_dispatched: usize,
    pub drops_count: usize,
    pub routing_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseState {
    pub phase: f32,
    pub freq: f32,
    pub amp: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseWindow {
    pub module: u8,
    pub open: bool,
    pub alignment: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    PhaseAligned,
    PhaseMisaligned,
    PressureHigh,
    BudgetLimited,
    CoherenceLow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleDecision {
    pub selected: Vec<u8>,
    pub reason_codes: Vec<Reason>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoherenceMetrics {
    pub coherence: f32,
    pub instability: f32,
    pub phi_proxy: f32,
    pub digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HistoryPoint {
    spike_count: usize,
    spike_energy: f32,
    alignment_mean: f32,
    pressure: f32,
    surprise: f32,
}

#[derive(Clone, Debug)]
pub struct CoherenceRuntime {
    subscribers: Vec<Subscriber>,
    phases: BTreeMap<u8, PhaseState>,
    history: VecDeque<HistoryPoint>,
    pub gated_total: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickInput {
    pub t: u64,
    pub source_digest: [u8; 32],
    pub pressure: f32,
    pub surprise: f32,
    pub risk: f32,
    pub confidence: f32,
    pub budget_limit: usize,
}

impl CoherenceRuntime {
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
            phases: BTreeMap::new(),
            history: VecDeque::new(),
            gated_total: 0,
        }
    }

    pub fn register_subscriber(&mut self, mut subscriber: Subscriber) {
        if self.subscribers.len() >= MAX_SUBSCRIBERS {
            return;
        }
        match &mut subscriber.interest {
            InterestProfile::TopKFeatures(features) => features.truncate(64),
            InterestProfile::HashBuckets(buckets) => buckets.truncate(16),
        }
        self.phases
            .entry(subscriber.module_id)
            .or_insert_with(|| seeded_phase(subscriber.module_id));
        self.subscribers.push(subscriber);
        self.subscribers.sort_by(|a, b| {
            a.name
                .cmp(b.name)
                .then_with(|| a.module_id.cmp(&b.module_id))
        });
    }

    pub fn tick(
        &mut self,
        spikes: &[Spike],
        input: TickInput,
    ) -> (
        SpikeRoutingSummary,
        Vec<PhaseWindow>,
        ScheduleDecision,
        CoherenceMetrics,
        Option<&'static str>,
    ) {
        let batch = make_batch(input.t, spikes, input.source_digest);
        let routing = route_batch(&batch, &self.subscribers);

        let mut windows = Vec::with_capacity(self.subscribers.len());
        for subscriber in &self.subscribers {
            let phase = self
                .phases
                .entry(subscriber.module_id)
                .or_insert_with(|| seeded_phase(subscriber.module_id));
            update_phase(
                phase,
                input.pressure,
                input.surprise,
                input.risk,
                input.confidence,
            );
            let alignment = alignment_to_ref(phase.phase, 0.0);
            windows.push(PhaseWindow {
                module: subscriber.module_id,
                open: alignment >= 0.45,
                alignment,
            });
        }

        let mean_alignment = if windows.is_empty() {
            0.0
        } else {
            windows.iter().map(|w| w.alignment).sum::<f32>() / windows.len() as f32
        };
        let spike_energy = batch
            .spikes
            .iter()
            .map(|s| s.magnitude.clamp(0.0, 1.0))
            .sum::<f32>();

        self.history.push_back(HistoryPoint {
            spike_count: batch.spikes.len(),
            spike_energy,
            alignment_mean: mean_alignment,
            pressure: input.pressure.clamp(0.0, 1.0),
            surprise: input.surprise.clamp(0.0, 1.0),
        });
        while self.history.len() > MAX_HISTORY {
            self.history.pop_front();
        }

        let metrics = compute_metrics(&self.history);
        let schedule = schedule_modules(
            &self.subscribers,
            &windows,
            input.pressure,
            metrics,
            input.budget_limit,
        );
        let gating_reason = gating_reason(metrics);
        if gating_reason.is_some() {
            self.gated_total = self.gated_total.saturating_add(1);
        }
        (routing, windows, schedule, metrics, gating_reason)
    }
}

impl Default for CoherenceRuntime {
    fn default() -> Self {
        Self::new()
    }
}

fn make_batch(t: u64, spikes: &[Spike], source_digest: [u8; 32]) -> SpikeBatch {
    let mut converted = spikes
        .iter()
        .map(|s| SpikeEvent {
            feature_id: s.feature_id,
            magnitude: s.magnitude.clamp(0.0, 1.0),
            t,
            source_digest,
        })
        .collect::<Vec<_>>();
    converted.sort_by(|a, b| {
        a.magnitude
            .total_cmp(&b.magnitude)
            .then_with(|| a.feature_id.cmp(&b.feature_id))
            .then_with(|| a.t.cmp(&b.t))
    });
    if converted.len() > MAX_SPIKES_PER_TICK {
        let split = converted.len() - MAX_SPIKES_PER_TICK;
        converted.drain(0..split);
    }
    SpikeBatch {
        t,
        spikes: converted,
    }
}

fn route_batch(batch: &SpikeBatch, subscribers: &[Subscriber]) -> SpikeRoutingSummary {
    let mut dispatched = 0usize;
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_COHERENCE);
    hasher.update(&batch.t.to_le_bytes());

    for subscriber in subscribers {
        let mut local = batch
            .spikes
            .iter()
            .filter(|spike| matches_interest(spike, &subscriber.interest))
            .collect::<Vec<_>>();
        if local.len() > MAX_EVENTS_PER_BATCH {
            local.truncate(MAX_EVENTS_PER_BATCH);
        }
        dispatched = dispatched.saturating_add(local.len());
        hasher.update(&[subscriber.module_id]);
        hasher.update(subscriber.name.as_bytes());
        hasher.update(&(local.len() as u64).to_le_bytes());
        for spike in local {
            hasher.update(&spike.feature_id.to_le_bytes());
            hasher.update(&spike.magnitude.to_le_bytes());
        }
    }

    SpikeRoutingSummary {
        t: batch.t,
        spikes_in: batch.spikes.len(),
        spikes_dispatched: dispatched,
        drops_count: batch
            .spikes
            .len()
            .saturating_sub(batch.spikes.len().min(MAX_SPIKES_PER_TICK)),
        routing_digest: *hasher.finalize().as_bytes(),
    }
}

fn matches_interest(spike: &SpikeEvent, interest: &InterestProfile) -> bool {
    match interest {
        InterestProfile::TopKFeatures(features) => features.contains(&spike.feature_id),
        InterestProfile::HashBuckets(buckets) => {
            let bucket = (spike.feature_id % BUCKETS) as u8;
            buckets.contains(&bucket)
        }
    }
}

fn seeded_phase(module_id: u8) -> PhaseState {
    let phase = (u32::from(module_id) * 37 % 100) as f32 / 100.0;
    let freq = 0.015 + (u32::from(module_id) % 7) as f32 * 0.002;
    PhaseState {
        phase,
        freq,
        amp: 0.8,
    }
}

fn update_phase(state: &mut PhaseState, pressure: f32, surprise: f32, risk: f32, confidence: f32) {
    let p = pressure.clamp(0.0, 1.0);
    let s = surprise.clamp(0.0, 1.0);
    let r = risk.clamp(0.0, 1.0);
    let c = confidence.clamp(0.0, 1.0);
    let coupling = (0.01 + 0.03 * (1.0 - p) + 0.01 * c - 0.015 * r + 0.01 * s).clamp(0.001, 0.05);
    state.freq = (0.95 * state.freq + 0.05 * coupling).clamp(0.001, 0.08);
    state.phase = (state.phase + state.freq).fract();
    state.amp = (0.9 * state.amp + 0.1 * (1.0 - p)).clamp(0.0, 1.0);
}

fn alignment_to_ref(phase: f32, reference: f32) -> f32 {
    let d = (phase - reference).abs();
    let wrapped = d.min(1.0 - d);
    (1.0 - 2.0 * wrapped).clamp(0.0, 1.0)
}

fn schedule_modules(
    subscribers: &[Subscriber],
    windows: &[PhaseWindow],
    pressure: f32,
    metrics: CoherenceMetrics,
    budget_limit: usize,
) -> ScheduleDecision {
    let k = budget_limit.min(MAX_SELECTED);
    let mut scored = Vec::with_capacity(subscribers.len());
    for subscriber in subscribers {
        let alignment = windows
            .iter()
            .find(|w| w.module == subscriber.module_id)
            .map(|w| w.alignment)
            .unwrap_or(0.0);
        let interest_match = match subscriber.interest {
            InterestProfile::TopKFeatures(ref f) => (f.len() as f32 / 64.0).clamp(0.0, 1.0),
            InterestProfile::HashBuckets(ref b) => (b.len() as f32 / 16.0).clamp(0.0, 1.0),
        };
        let score =
            0.45 * alignment + 0.3 * interest_match + 0.25 * (1.0 - pressure.clamp(0.0, 1.0))
                - 0.25 * metrics.instability
                - 0.2 * (1.0 - metrics.coherence);
        scored.push((subscriber.module_id, score, alignment));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let selected = scored.iter().take(k).map(|v| v.0).collect::<Vec<_>>();

    let mut reason_codes = Vec::new();
    if metrics.coherence < 0.35 {
        reason_codes.push(Reason::CoherenceLow);
    }
    if pressure > 0.8 {
        reason_codes.push(Reason::PressureHigh);
    }
    if k < subscribers.len() {
        reason_codes.push(Reason::BudgetLimited);
    }
    if scored.iter().any(|(_, _, alignment)| *alignment >= 0.45) {
        reason_codes.push(Reason::PhaseAligned);
    } else {
        reason_codes.push(Reason::PhaseMisaligned);
    }

    ScheduleDecision {
        selected,
        reason_codes,
    }
}

fn compute_metrics(history: &VecDeque<HistoryPoint>) -> CoherenceMetrics {
    if history.is_empty() {
        return CoherenceMetrics {
            coherence: 1.0,
            instability: 0.0,
            phi_proxy: 1.0,
            digest: digest_metrics(1.0, 0.0, 1.0),
        };
    }

    let n = history.len() as f32;
    let mean_align = history.iter().map(|h| h.alignment_mean).sum::<f32>() / n;
    let mean_energy = history.iter().map(|h| h.spike_energy).sum::<f32>() / n;
    let mean_pressure = history.iter().map(|h| h.pressure).sum::<f32>() / n;
    let mean_surprise = history.iter().map(|h| h.surprise).sum::<f32>() / n;
    let align_var = history
        .iter()
        .map(|h| {
            let d = h.alignment_mean - mean_align;
            d * d
        })
        .sum::<f32>()
        / n;
    let energy_var = history
        .iter()
        .map(|h| {
            let d = h.spike_energy - mean_energy;
            d * d
        })
        .sum::<f32>()
        / n;
    let timing_var = history
        .iter()
        .map(|h| {
            let density = h.spike_count as f32 / MAX_SPIKES_PER_TICK as f32;
            let d = density - 0.5;
            d * d
        })
        .sum::<f32>()
        / n;

    let coherence = (1.0 - (2.0 * align_var + timing_var)).clamp(0.0, 1.0);
    let instability =
        (0.5 * mean_pressure + 0.5 * mean_surprise + 0.2 * energy_var).clamp(0.0, 1.0);
    let phi_proxy = (coherence * (1.0 - instability)).clamp(0.0, 1.0);

    CoherenceMetrics {
        coherence,
        instability,
        phi_proxy,
        digest: digest_metrics(coherence, instability, phi_proxy),
    }
}

fn digest_metrics(coherence: f32, instability: f32, phi_proxy: f32) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_COHERENCE);
    hasher.update(&coherence.to_le_bytes());
    hasher.update(&instability.to_le_bytes());
    hasher.update(&phi_proxy.to_le_bytes());
    *hasher.finalize().as_bytes()
}

pub fn gating_reason(metrics: CoherenceMetrics) -> Option<&'static str> {
    if metrics.coherence < 0.30 {
        Some("coherence_low")
    } else if metrics.instability > 0.70 {
        Some("instability_high")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spikes() -> Vec<Spike> {
        (0..32)
            .map(|i| Spike {
                feature_id: i,
                magnitude: (i as f32 / 32.0).clamp(0.0, 1.0),
                timestamp: 1,
            })
            .collect()
    }

    #[test]
    fn routing_is_deterministic() {
        let mut runtime = CoherenceRuntime::new();
        runtime.register_subscriber(Subscriber {
            module_id: 1,
            name: "a",
            interest: InterestProfile::HashBuckets(vec![1, 2, 3]),
        });
        runtime.register_subscriber(Subscriber {
            module_id: 2,
            name: "b",
            interest: InterestProfile::TopKFeatures(vec![3, 7, 11]),
        });

        let input = TickInput {
            t: 1,
            source_digest: [9; 32],
            pressure: 0.2,
            surprise: 0.3,
            risk: 0.1,
            confidence: 0.8,
            budget_limit: 4,
        };
        let out_a = runtime.tick(&spikes(), input);
        let out_b = runtime.tick(&spikes(), input);
        assert_eq!(out_a.0.routing_digest, out_b.0.routing_digest);
    }

    #[test]
    fn metrics_are_bounded() {
        let mut runtime = CoherenceRuntime::new();
        runtime.register_subscriber(Subscriber {
            module_id: 5,
            name: "m",
            interest: InterestProfile::HashBuckets(vec![1]),
        });
        let input = TickInput {
            t: 1,
            source_digest: [1; 32],
            pressure: 0.9,
            surprise: 0.8,
            risk: 0.8,
            confidence: 0.2,
            budget_limit: 1,
        };
        let (_, windows, schedule, metrics, _) = runtime.tick(&spikes(), input);
        assert!(!windows.is_empty());
        assert!(schedule.selected.len() <= 1);
        assert!((0.0..=1.0).contains(&metrics.coherence));
        assert!((0.0..=1.0).contains(&metrics.instability));
        assert!((0.0..=1.0).contains(&metrics.phi_proxy));
    }
}
