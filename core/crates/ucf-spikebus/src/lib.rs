#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeMap;

use blake3::Hasher;
use ucf_onn::{accept_spike, PhaseLockDecision};
use ucf_types::Digest32;

pub mod producers;

const SPIKE_DOMAIN: &[u8] = b"ucf.spikebus.spike.v1";
const SPIKE_PARAMS_DOMAIN: &[u8] = b"ucf.spikebus.params.v1";
const SPIKE_INPUTS_DOMAIN: &[u8] = b"ucf.spikebus.inputs.v1";
const SPIKE_OUTPUTS_DOMAIN: &[u8] = b"ucf.spikebus.outputs.v1";
const SPIKE_BUS_DOMAIN: &[u8] = b"ucf.spikebus.core.v1";

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpikeKind {
    Feature = 0,
    Novelty = 1,
    Threat = 2,
    Reward = 3,
    CausalLink = 4,
    PolicySignal = 5,
    ThoughtOnly = 6,
    ReplayHint = 7,
    Unknown(u8),
}

impl SpikeKind {
    pub fn discriminant(self) -> u8 {
        match self {
            Self::Feature => 0,
            Self::Novelty => 1,
            Self::Threat => 2,
            Self::Reward => 3,
            Self::CausalLink => 4,
            Self::PolicySignal => 5,
            Self::ThoughtOnly => 6,
            Self::ReplayHint => 7,
            Self::Unknown(code) => code,
        }
    }

    pub fn as_u16(self) -> u16 {
        u16::from(self.discriminant())
    }
}

impl PartialOrd for SpikeKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpikeKind {
    fn cmp(&self, other: &Self) -> Ordering {
        self.discriminant().cmp(&other.discriminant())
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModuleId {
    Sae = 1,
    Lens = 2,
    Sle = 3,
    Iit = 4,
    Policy = 5,
    Cde = 6,
    Nsr = 7,
    Ssm = 8,
    Tcf = 9,
    Router = 10,
    Other(u8),
}

impl ModuleId {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Sae => 1,
            Self::Lens => 2,
            Self::Sle => 3,
            Self::Iit => 4,
            Self::Policy => 5,
            Self::Cde => 6,
            Self::Nsr => 7,
            Self::Ssm => 8,
            Self::Tcf => 9,
            Self::Router => 10,
            Self::Other(value) => value,
        }
    }
}

impl PartialOrd for ModuleId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ModuleId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_u8().cmp(&other.as_u8())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spike {
    pub cycle_id: u64,
    pub kind: SpikeKind,
    pub intensity: u16,
    pub bucket: u8,
    pub source: ModuleId,
    pub payload_commit: Digest32,
    pub commit: Digest32,
}

impl Spike {
    pub fn new(
        cycle_id: u64,
        kind: SpikeKind,
        intensity: u16,
        bucket: u8,
        source: ModuleId,
        payload_commit: Digest32,
    ) -> Self {
        let intensity = intensity.min(10_000);
        let bucket = bucket.min(15);
        let commit = commit_spike(cycle_id, kind, intensity, bucket, source, payload_commit);
        Self {
            cycle_id,
            kind,
            intensity,
            bucket,
            source,
            payload_commit,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeParams {
    pub thr_feature: u16,
    pub thr_novelty: u16,
    pub thr_threat: u16,
    pub thr_reward: u16,
    pub thr_causal: u16,
    pub thr_policy: u16,
    pub thr_thought: u16,
    pub thr_replay: u16,
    pub max_spikes: usize,
    pub commit: Digest32,
}

impl SpikeParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        thr_feature: u16,
        thr_novelty: u16,
        thr_threat: u16,
        thr_reward: u16,
        thr_causal: u16,
        thr_policy: u16,
        thr_thought: u16,
        thr_replay: u16,
        max_spikes: usize,
    ) -> Self {
        let max_spikes = max_spikes.clamp(1, 64);
        let params = Self {
            thr_feature,
            thr_novelty,
            thr_threat,
            thr_reward,
            thr_causal,
            thr_policy,
            thr_thought,
            thr_replay,
            max_spikes,
            commit: Digest32::new([0u8; 32]),
        };
        let commit = commit_params(&params);
        Self { commit, ..params }
    }
}

impl Default for SpikeParams {
    fn default() -> Self {
        Self::new(0, 0, 0, 0, 0, 0, 0, 0, 32)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeInputs {
    pub cycle_id: u64,
    pub lock: PhaseLockDecision,
    pub candidates: Vec<Spike>,
    pub commit: Digest32,
}

impl SpikeInputs {
    pub fn new(cycle_id: u64, lock: PhaseLockDecision, candidates: Vec<Spike>) -> Self {
        let mut inputs = Self {
            cycle_id,
            lock,
            candidates,
            commit: Digest32::new([0u8; 32]),
        };
        inputs.commit = commit_inputs(&inputs);
        inputs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeOutputs {
    pub cycle_id: u64,
    pub accepted_root: Digest32,
    pub accepted: Vec<Spike>,
    pub counts: Vec<(SpikeKind, u16)>,
    pub max_intensity: u16,
    pub commit: Digest32,
}

/// Spike router boundary for coherence gating.
///
/// # Commit formula
/// - `SpikeOutputs.commit = H(cycle_id, accepted_root, max_intensity, counts, params.commit, inputs.commit)`
pub trait SpikeRouter {
    fn tick(&self, inp: &SpikeInputs) -> SpikeOutputs;

    fn params(&self) -> SpikeParams;

    fn set_params(&mut self, params: SpikeParams);
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeBus {
    pub params: SpikeParams,
    pub commit: Digest32,
}

impl SpikeBus {
    pub fn new(params: SpikeParams) -> Self {
        let commit = commit_bus(params.commit);
        Self { params, commit }
    }

    pub fn tick(&self, inp: &SpikeInputs) -> SpikeOutputs {
        let mut accepted = inp
            .candidates
            .iter()
            .filter(|spike| {
                spike.intensity >= threshold(spike.kind, &self.params)
                    && accept_spike(&inp.lock, spike.bucket)
            })
            .cloned()
            .collect::<Vec<_>>();
        accepted.sort_by(compare_spikes);
        if accepted.len() > self.params.max_spikes {
            accepted.truncate(self.params.max_spikes);
        }
        let accepted_root = fold_root(&accepted);
        let max_intensity = accepted
            .iter()
            .map(|spike| spike.intensity)
            .max()
            .unwrap_or(0);
        let counts = spike_counts(&accepted);
        let commit = commit_outputs(
            inp.cycle_id,
            accepted_root,
            max_intensity,
            &counts,
            self.params.commit,
            inp.commit,
        );
        SpikeOutputs {
            cycle_id: inp.cycle_id,
            accepted_root,
            accepted,
            counts,
            max_intensity,
            commit,
        }
    }
}

impl Default for SpikeBus {
    fn default() -> Self {
        Self::new(SpikeParams::default())
    }
}

impl SpikeRouter for SpikeBus {
    fn tick(&self, inp: &SpikeInputs) -> SpikeOutputs {
        Self::tick(self, inp)
    }

    fn params(&self) -> SpikeParams {
        self.params
    }

    fn set_params(&mut self, params: SpikeParams) {
        self.params = params;
        self.commit = commit_bus(self.params.commit);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockSpikeRouter {
    params: SpikeParams,
    accept_all: bool,
}

impl MockSpikeRouter {
    pub fn new_accept_all() -> Self {
        Self {
            params: SpikeParams::default(),
            accept_all: true,
        }
    }

    pub fn new_accept_none() -> Self {
        Self {
            params: SpikeParams::default(),
            accept_all: false,
        }
    }
}

impl Default for MockSpikeRouter {
    fn default() -> Self {
        Self::new_accept_all()
    }
}

impl SpikeRouter for MockSpikeRouter {
    fn tick(&self, inp: &SpikeInputs) -> SpikeOutputs {
        let mut accepted = if self.accept_all {
            inp.candidates.clone()
        } else {
            Vec::new()
        };
        accepted.sort_by(compare_spikes);
        if accepted.len() > self.params.max_spikes {
            accepted.truncate(self.params.max_spikes);
        }
        let accepted_root = fold_root(&accepted);
        let max_intensity = accepted
            .iter()
            .map(|spike| spike.intensity)
            .max()
            .unwrap_or(0);
        let counts = spike_counts(&accepted);
        let commit = commit_outputs(
            inp.cycle_id,
            accepted_root,
            max_intensity,
            &counts,
            self.params.commit,
            inp.commit,
        );
        SpikeOutputs {
            cycle_id: inp.cycle_id,
            accepted_root,
            accepted,
            counts,
            max_intensity,
            commit,
        }
    }

    fn params(&self) -> SpikeParams {
        self.params
    }

    fn set_params(&mut self, params: SpikeParams) {
        self.params = params;
    }
}

pub fn threshold(kind: SpikeKind, p: &SpikeParams) -> u16 {
    match kind {
        SpikeKind::Feature => p.thr_feature,
        SpikeKind::Novelty => p.thr_novelty,
        SpikeKind::Threat => p.thr_threat,
        SpikeKind::Reward => p.thr_reward,
        SpikeKind::CausalLink => p.thr_causal,
        SpikeKind::PolicySignal => p.thr_policy,
        SpikeKind::ThoughtOnly => p.thr_thought,
        SpikeKind::ReplayHint => p.thr_replay,
        SpikeKind::Unknown(_) => 0,
    }
}

fn compare_spikes(a: &Spike, b: &Spike) -> Ordering {
    a.kind
        .discriminant()
        .cmp(&b.kind.discriminant())
        .then_with(|| b.intensity.cmp(&a.intensity))
        .then_with(|| a.source.as_u8().cmp(&b.source.as_u8()))
        .then_with(|| a.payload_commit.as_bytes().cmp(b.payload_commit.as_bytes()))
}

fn spike_counts(accepted: &[Spike]) -> Vec<(SpikeKind, u16)> {
    let mut counts: BTreeMap<SpikeKind, u16> = BTreeMap::new();
    for spike in accepted {
        let entry = counts.entry(spike.kind).or_insert(0);
        *entry = entry.saturating_add(1);
    }
    counts.into_iter().collect()
}

fn fold_root(accepted: &[Spike]) -> Digest32 {
    let mut root = Digest32::new([0u8; 32]);
    for spike in accepted {
        let mut hasher = Hasher::new();
        hasher.update(root.as_bytes());
        hasher.update(spike.commit.as_bytes());
        root = Digest32::new(*hasher.finalize().as_bytes());
    }
    root
}

fn commit_spike(
    cycle_id: u64,
    kind: SpikeKind,
    intensity: u16,
    bucket: u8,
    source: ModuleId,
    payload_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&[kind.discriminant()]);
    hasher.update(&intensity.to_be_bytes());
    hasher.update(&[bucket]);
    hasher.update(&[source.as_u8()]);
    hasher.update(payload_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_params(params: &SpikeParams) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_PARAMS_DOMAIN);
    hasher.update(&params.thr_feature.to_be_bytes());
    hasher.update(&params.thr_novelty.to_be_bytes());
    hasher.update(&params.thr_threat.to_be_bytes());
    hasher.update(&params.thr_reward.to_be_bytes());
    hasher.update(&params.thr_causal.to_be_bytes());
    hasher.update(&params.thr_policy.to_be_bytes());
    hasher.update(&params.thr_thought.to_be_bytes());
    hasher.update(&params.thr_replay.to_be_bytes());
    hasher.update(
        &u64::try_from(params.max_spikes)
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_inputs(inputs: &SpikeInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_INPUTS_DOMAIN);
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.lock.commit.as_bytes());
    hasher.update(
        &u64::try_from(inputs.candidates.len())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    for spike in &inputs.candidates {
        hasher.update(spike.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_outputs(
    cycle_id: u64,
    accepted_root: Digest32,
    max_intensity: u16,
    counts: &[(SpikeKind, u16)],
    params_commit: Digest32,
    inputs_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_OUTPUTS_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(accepted_root.as_bytes());
    hasher.update(&max_intensity.to_be_bytes());
    hasher.update(
        &u64::try_from(counts.len())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    for (kind, count) in counts {
        hasher.update(&[kind.discriminant()]);
        hasher.update(&count.to_be_bytes());
    }
    hasher.update(params_commit.as_bytes());
    hasher.update(inputs_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_bus(params_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_BUS_DOMAIN);
    hasher.update(params_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_onn::PhaseLockDecision;

    fn make_lock(cycle_id: u64, accept_center: u8, lock_window_buckets: u8) -> PhaseLockDecision {
        PhaseLockDecision {
            cycle_id,
            lock_window_buckets,
            accept_center,
            commit: Digest32::new([1u8; 32]),
        }
    }

    fn make_spike(kind: SpikeKind, intensity: u16, bucket: u8, source: ModuleId) -> Spike {
        Spike::new(
            1,
            kind,
            intensity,
            bucket,
            source,
            Digest32::new([bucket; 32]),
        )
    }

    #[test]
    fn phase_gating_rejects_outside_window() {
        let params = SpikeParams::new(0, 0, 0, 0, 0, 0, 0, 0, 8);
        let bus = SpikeBus::new(params);
        let lock = make_lock(1, 0, 1);
        let spike = make_spike(SpikeKind::Feature, 5000, 4, ModuleId::Sae);
        let inputs = SpikeInputs::new(1, lock, vec![spike]);
        let outputs = bus.tick(&inputs);
        assert!(outputs.accepted.is_empty());
        assert_eq!(outputs.max_intensity, 0);
    }

    #[test]
    fn thresholding_rejects_below_threshold() {
        let params = SpikeParams::new(6000, 0, 0, 0, 0, 0, 0, 0, 8);
        let bus = SpikeBus::new(params);
        let lock = make_lock(1, 0, 2);
        let spike = make_spike(SpikeKind::Feature, 5000, 0, ModuleId::Sae);
        let inputs = SpikeInputs::new(1, lock, vec![spike]);
        let outputs = bus.tick(&inputs);
        assert!(outputs.accepted.is_empty());
    }

    #[test]
    fn deterministic_sort_and_cap() {
        let params = SpikeParams::new(0, 0, 0, 0, 0, 0, 0, 0, 2);
        let bus = SpikeBus::new(params);
        let lock = make_lock(1, 0, 4);
        let spike_a = make_spike(SpikeKind::Novelty, 1000, 0, ModuleId::Lens);
        let spike_b = make_spike(SpikeKind::Feature, 2000, 0, ModuleId::Sae);
        let spike_c = make_spike(SpikeKind::Feature, 1500, 0, ModuleId::Sae);
        let inputs = SpikeInputs::new(
            1,
            lock,
            vec![spike_a.clone(), spike_c.clone(), spike_b.clone()],
        );
        let outputs = bus.tick(&inputs);
        assert_eq!(outputs.accepted.len(), 2);
        assert_eq!(outputs.accepted[0].kind, SpikeKind::Feature);
        assert_eq!(outputs.accepted[0].intensity, 2000);
        assert_eq!(outputs.accepted[1].kind, SpikeKind::Feature);
        assert_eq!(outputs.accepted[1].intensity, 1500);

        let inputs_again = SpikeInputs::new(1, lock, vec![spike_b, spike_a, spike_c]);
        let outputs_again = bus.tick(&inputs_again);
        assert_eq!(outputs.accepted, outputs_again.accepted);
        assert_eq!(outputs.accepted_root, outputs_again.accepted_root);
    }

    #[test]
    fn fold_root_matches_expected() {
        let spike_a = make_spike(SpikeKind::Feature, 1000, 0, ModuleId::Sae);
        let spike_b = make_spike(SpikeKind::Novelty, 900, 0, ModuleId::Lens);
        let mut root = Digest32::new([0u8; 32]);
        for spike in [&spike_a, &spike_b] {
            let mut hasher = Hasher::new();
            hasher.update(root.as_bytes());
            hasher.update(spike.commit.as_bytes());
            root = Digest32::new(*hasher.finalize().as_bytes());
        }
        let computed = fold_root(&[spike_a, spike_b]);
        assert_eq!(computed, root);
    }
}
