#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrainRegion {
    Hypothalamus,
    Insula,
    NAcc,
    PFC,
    Thalamus,
    Hippocampus,
    Cerebellum,
    Brainstem,
    Unknown(u16),
}

impl BrainRegion {
    pub fn code(self) -> u16 {
        match self {
            Self::Hypothalamus => 1,
            Self::Insula => 2,
            Self::NAcc => 3,
            Self::PFC => 4,
            Self::Thalamus => 5,
            Self::Hippocampus => 6,
            Self::Cerebellum => 7,
            Self::Brainstem => 8,
            Self::Unknown(value) => value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spike {
    pub region: BrainRegion,
    pub amplitude: u16,
    pub width: u16,
    pub commit: Digest32,
}

impl Spike {
    pub fn new(region: BrainRegion, amplitude: u16, width: u16) -> Self {
        let commit = commit_spike(region, amplitude, width);
        Self {
            region,
            amplitude,
            width,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainStimulus {
    pub cycle_id: u64,
    pub spikes: Vec<Spike>,
    pub commit: Digest32,
}

impl BrainStimulus {
    pub fn new(cycle_id: u64, spikes: Vec<Spike>) -> Self {
        let commit = commit_stimulus(cycle_id, &spikes, None);
        Self {
            cycle_id,
            spikes,
            commit,
        }
    }

    pub fn with_seed(cycle_id: u64, spikes: Vec<Spike>, seed: Digest32) -> Self {
        let commit = commit_stimulus(cycle_id, &spikes, Some(&seed));
        Self {
            cycle_id,
            spikes,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeuromodDelta {
    pub dopamine: i16,
    pub serotonin: i16,
    pub norepi: i16,
    pub cortisol: i16,
    pub commit: Digest32,
}

impl NeuromodDelta {
    pub fn new(dopamine: i16, serotonin: i16, norepi: i16, cortisol: i16) -> Self {
        let commit = commit_delta(dopamine, serotonin, norepi, cortisol);
        Self {
            dopamine,
            serotonin,
            norepi,
            cortisol,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainResponse {
    pub arousal: u16,
    pub valence: i16,
    pub delta: NeuromodDelta,
    pub commit: Digest32,
}

impl BrainResponse {
    pub fn new(arousal: u16, valence: i16, delta: NeuromodDelta) -> Self {
        let commit = commit_response(arousal, valence, delta.commit);
        Self {
            arousal,
            valence,
            delta,
            commit,
        }
    }
}

pub trait BlueBrainPort {
    fn stimulate(&mut self, stim: &BrainStimulus) -> BrainResponse;
}

#[derive(Clone, Default)]
pub struct MockBlueBrainPort {
    last_response: Option<BrainResponse>,
}

impl MockBlueBrainPort {
    pub fn new() -> Self {
        Self {
            last_response: None,
        }
    }

    pub fn last_response(&self) -> Option<&BrainResponse> {
        self.last_response.as_ref()
    }
}

impl BlueBrainPort for MockBlueBrainPort {
    fn stimulate(&mut self, stim: &BrainStimulus) -> BrainResponse {
        let response = mock_response(stim);
        self.last_response = Some(response.clone());
        response
    }
}

fn commit_spike(region: BrainRegion, amplitude: u16, width: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.bluebrain.spike.v1");
    hasher.update(&region.code().to_be_bytes());
    hasher.update(&amplitude.to_be_bytes());
    hasher.update(&width.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_stimulus(cycle_id: u64, spikes: &[Spike], seed: Option<&Digest32>) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.bluebrain.stimulus.v1");
    hasher.update(&cycle_id.to_be_bytes());
    if let Some(seed) = seed {
        hasher.update(seed.as_bytes());
    }
    for spike in spikes {
        hasher.update(spike.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_delta(dopamine: i16, serotonin: i16, norepi: i16, cortisol: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.bluebrain.delta.v1");
    hasher.update(&dopamine.to_be_bytes());
    hasher.update(&serotonin.to_be_bytes());
    hasher.update(&norepi.to_be_bytes());
    hasher.update(&cortisol.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_response(arousal: u16, valence: i16, delta_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.bluebrain.response.v1");
    hasher.update(&arousal.to_be_bytes());
    hasher.update(&valence.to_be_bytes());
    hasher.update(delta_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn mock_response(stim: &BrainStimulus) -> BrainResponse {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.bluebrain.mock.v1");
    hasher.update(stim.commit.as_bytes());
    let digest = Digest32::new(*hasher.finalize().as_bytes());
    let bytes = digest.as_bytes();

    let total_amp: u32 = stim
        .spikes
        .iter()
        .map(|spike| u32::from(spike.amplitude))
        .sum();
    let arousal = total_amp.min(10_000) as u16;

    let magnitude = u16::from_be_bytes([bytes[0], bytes[1]]) % 2000;
    let sign_positive = bytes[2] & 1 == 0;
    let valence = if sign_positive {
        magnitude as i16
    } else {
        -(magnitude as i16)
    };

    let dopamine = bounded_delta(bytes[3], bytes[4], 50);
    let serotonin = bounded_delta(bytes[5], bytes[6], 50);
    let norepi = bounded_delta(bytes[7], bytes[8], 50);
    let cortisol = bounded_delta(bytes[9], bytes[10], 50);
    let delta = NeuromodDelta::new(dopamine, serotonin, norepi, cortisol);

    BrainResponse::new(arousal, valence, delta)
}

fn bounded_delta(lo: u8, hi: u8, span: i16) -> i16 {
    let raw = u16::from_be_bytes([lo, hi]) as i32;
    let span_i32 = i32::from(span);
    let modulo = span_i32.saturating_mul(2).saturating_add(1);
    let value = raw % modulo;
    (value - span_i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_port_is_deterministic() {
        let spikes = vec![Spike::new(BrainRegion::PFC, 1200, 80)];
        let stim = BrainStimulus::new(7, spikes);
        let mut port = MockBlueBrainPort::new();

        let first = port.stimulate(&stim);
        let second = port.stimulate(&stim);

        assert_eq!(first, second);
        assert_eq!(first.arousal, 1200);
    }
}
