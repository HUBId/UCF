#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use blake3::Hasher;
use ucf_onn::OscId;
use ucf_types::Digest32;

pub use ucf_onn::OscId as SpikeModuleId;

const SPIKE_EVENT_DOMAIN: &[u8] = b"ucf.spikebus.event.v3";
const SPIKE_BATCH_ROOT_DOMAIN: &[u8] = b"ucf.spikebus.batch.root.v1";
const SPIKE_BATCH_COMMIT_DOMAIN: &[u8] = b"ucf.spikebus.batch.commit.v1";
const SPIKE_SEEN_ROOT_DOMAIN: &[u8] = b"ucf.spikebus.seen.root.v1";
const SPIKE_ACCEPTED_ROOT_DOMAIN: &[u8] = b"ucf.spikebus.accepted.root.v1";

pub const SPIKE_CYCLE_CAP: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpikeKind {
    Feature,
    Novelty,
    Threat,
    CausalLink,
    ConsistencyAlert,
    ThoughtOnly,
    MemoryCue,
    ReplayCue,
    OutputIntent,
    Unknown(u16),
}

impl SpikeKind {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::Feature => 1,
            Self::Novelty => 2,
            Self::Threat => 3,
            Self::CausalLink => 4,
            Self::ConsistencyAlert => 5,
            Self::ThoughtOnly => 6,
            Self::MemoryCue => 7,
            Self::ReplayCue => 8,
            Self::OutputIntent => 9,
            Self::Unknown(code) => code,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeEvent {
    pub cycle_id: u64,
    pub kind: SpikeKind,
    pub src: OscId,
    pub dst: OscId,
    pub phase_bucket: u8,
    pub ttfs: u16,
    pub phase_commit: Digest32,
    pub payload_commit: Digest32,
    pub commit: Digest32,
}

impl SpikeEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        kind: SpikeKind,
        src: OscId,
        dst: OscId,
        phase_bucket: u8,
        ttfs: u16,
        phase_commit: Digest32,
        payload_commit: Digest32,
    ) -> Self {
        if kind == SpikeKind::ThoughtOnly {
            assert!(
                matches!(dst, OscId::Reserved8 | OscId::Reserved9),
                "ThoughtOnly spikes must target Reserved8 or Reserved9"
            );
        }
        let commit = commit_spike_event(
            cycle_id,
            kind,
            src,
            dst,
            phase_bucket,
            ttfs,
            phase_commit,
            payload_commit,
        );
        Self {
            cycle_id,
            kind,
            src,
            dst,
            phase_bucket,
            ttfs,
            phase_commit,
            payload_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeBatch {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub events: Vec<SpikeEvent>,
    pub root: Digest32,
    pub commit: Digest32,
}

impl SpikeBatch {
    pub fn new(cycle_id: u64, phase_commit: Digest32, mut events: Vec<SpikeEvent>) -> Self {
        events.sort_by(compare_spikes);
        let root = spike_batch_root(&events);
        let commit = commit_spike_batch(cycle_id, phase_commit, root, events.len());
        Self {
            cycle_id,
            phase_commit,
            events,
            root,
            commit,
        }
    }

    pub fn truncate(&mut self, cap: usize) -> bool {
        if self.events.len() <= cap {
            return false;
        }
        self.events.truncate(cap);
        self.events.sort_by(compare_spikes);
        self.root = spike_batch_root(&self.events);
        self.commit = commit_spike_batch(
            self.cycle_id,
            self.phase_commit,
            self.root,
            self.events.len(),
        );
        true
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpikeSuppression {
    pub suppressed_by_onn: bool,
    pub suppressed_by_policy: bool,
    pub suppressed_by_phase: bool,
}

impl SpikeSuppression {
    pub fn is_suppressed(self) -> bool {
        self.suppressed_by_onn || self.suppressed_by_policy || self.suppressed_by_phase
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeBusSummary {
    pub seen_root: Digest32,
    pub accepted_root: Digest32,
    pub counts: Vec<(SpikeKind, u16)>,
    pub causal_link_count: u16,
    pub consistency_alert_count: u16,
    pub thought_only_count: u16,
    pub output_intent_count: u16,
    pub cap_hit: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeBusState {
    events: Vec<SpikeEvent>,
    seen_root: Digest32,
    accepted_root: Digest32,
    counts: BTreeMap<SpikeKind, u16>,
    cap_hit: bool,
    cycle_id: Option<u64>,
    seen_count: usize,
    seen_per_cycle: HashMap<u64, usize>,
}

impl Default for SpikeBusState {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            seen_root: Digest32::new([0u8; 32]),
            accepted_root: Digest32::new([0u8; 32]),
            counts: BTreeMap::new(),
            cap_hit: false,
            cycle_id: None,
            seen_count: 0,
            seen_per_cycle: HashMap::new(),
        }
    }
}

impl SpikeBusState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append_batch(
        &mut self,
        mut batch: SpikeBatch,
        mut suppressions: Vec<SpikeSuppression>,
    ) -> SpikeBusSummary {
        self.reset_for_cycle(batch.cycle_id);
        if suppressions.len() < batch.events.len() {
            suppressions.resize(batch.events.len(), SpikeSuppression::default());
        } else if suppressions.len() > batch.events.len() {
            suppressions.truncate(batch.events.len());
        }
        let seen_so_far = *self.seen_per_cycle.get(&batch.cycle_id).unwrap_or(&0);
        let available = SPIKE_CYCLE_CAP.saturating_sub(seen_so_far);
        if available == 0 {
            self.cap_hit = true;
            return self.summary();
        }
        if batch.truncate(available) {
            self.cap_hit = true;
        }
        suppressions.truncate(batch.events.len());
        for (event, suppression) in batch.events.iter().cloned().zip(suppressions) {
            self.seen_root = advance_root(self.seen_root, event.commit, SPIKE_SEEN_ROOT_DOMAIN);
            self.seen_count = self.seen_count.saturating_add(1);
            let entry = self.seen_per_cycle.entry(batch.cycle_id).or_insert(0);
            *entry = entry.saturating_add(1);
            if suppression.is_suppressed() {
                continue;
            }
            self.accepted_root =
                advance_root(self.accepted_root, event.commit, SPIKE_ACCEPTED_ROOT_DOMAIN);
            let count = self.counts.entry(event.kind).or_insert(0);
            *count = count.saturating_add(1);
            self.events.push(event);
        }
        self.summary()
    }

    pub fn drain_for(&mut self, dst: OscId, cycle_id: u64, limit: usize) -> Vec<SpikeEvent> {
        if limit == 0 {
            return Vec::new();
        }
        let mut matched: Vec<SpikeEvent> = self
            .events
            .iter()
            .filter(|ev| ev.dst == dst && ev.cycle_id <= cycle_id)
            .cloned()
            .collect();
        matched.sort_by(compare_spikes);
        let drained = matched.into_iter().take(limit).collect::<Vec<_>>();
        if drained.is_empty() {
            return drained;
        }
        let drained_commits = drained
            .iter()
            .map(|ev| *ev.commit.as_bytes())
            .collect::<std::collections::HashSet<_>>();
        let mut remaining = Vec::with_capacity(self.events.len());
        for ev in self.events.drain(..) {
            if drained_commits.contains(ev.commit.as_bytes())
                && ev.dst == dst
                && ev.cycle_id <= cycle_id
            {
                continue;
            }
            remaining.push(ev);
        }
        self.events = remaining;
        drained
    }

    pub fn summary(&self) -> SpikeBusSummary {
        let counts = self
            .counts
            .iter()
            .map(|(kind, count)| (*kind, *count))
            .collect::<Vec<_>>();
        SpikeBusSummary {
            seen_root: self.seen_root,
            accepted_root: self.accepted_root,
            causal_link_count: *self.counts.get(&SpikeKind::CausalLink).unwrap_or(&0),
            consistency_alert_count: *self.counts.get(&SpikeKind::ConsistencyAlert).unwrap_or(&0),
            thought_only_count: *self.counts.get(&SpikeKind::ThoughtOnly).unwrap_or(&0),
            output_intent_count: *self.counts.get(&SpikeKind::OutputIntent).unwrap_or(&0),
            counts,
            cap_hit: self.cap_hit,
        }
    }

    pub fn seen_root(&self) -> Digest32 {
        self.seen_root
    }

    pub fn accepted_root(&self) -> Digest32 {
        self.accepted_root
    }

    pub fn counts(&self) -> Vec<(SpikeKind, u16)> {
        self.counts
            .iter()
            .map(|(kind, count)| (*kind, *count))
            .collect()
    }

    pub fn cap_hit(&self) -> bool {
        self.cap_hit
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    fn reset_for_cycle(&mut self, cycle_id: u64) {
        if self.cycle_id == Some(cycle_id) {
            return;
        }
        self.cycle_id = Some(cycle_id);
        self.events.clear();
        self.seen_root = Digest32::new([0u8; 32]);
        self.accepted_root = Digest32::new([0u8; 32]);
        self.counts.clear();
        self.cap_hit = false;
        self.seen_count = 0;
        self.seen_per_cycle.clear();
    }
}

fn compare_spikes(a: &SpikeEvent, b: &SpikeEvent) -> Ordering {
    let cycle_cmp = a.cycle_id.cmp(&b.cycle_id);
    if cycle_cmp != Ordering::Equal {
        return cycle_cmp;
    }
    let dst_cmp = a.dst.as_u8().cmp(&b.dst.as_u8());
    if dst_cmp != Ordering::Equal {
        return dst_cmp;
    }
    let kind_cmp = a.kind.cmp(&b.kind);
    if kind_cmp != Ordering::Equal {
        return kind_cmp;
    }
    let bucket_cmp = a.phase_bucket.cmp(&b.phase_bucket);
    if bucket_cmp != Ordering::Equal {
        return bucket_cmp;
    }
    let ttfs_cmp = a.ttfs.cmp(&b.ttfs);
    if ttfs_cmp != Ordering::Equal {
        return ttfs_cmp;
    }
    a.commit.as_bytes().cmp(b.commit.as_bytes())
}

fn spike_batch_root(events: &[SpikeEvent]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_BATCH_ROOT_DOMAIN);
    hasher.update(
        &u64::try_from(events.len())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    for event in events {
        hasher.update(event.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_spike_batch(
    cycle_id: u64,
    phase_commit: Digest32,
    root: Digest32,
    event_count: usize,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_BATCH_COMMIT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(root.as_bytes());
    hasher.update(&u64::try_from(event_count).unwrap_or(u64::MAX).to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_spike_event(
    cycle_id: u64,
    kind: SpikeKind,
    src: OscId,
    dst: OscId,
    phase_bucket: u8,
    ttfs: u16,
    phase_commit: Digest32,
    payload_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_EVENT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&kind.as_u16().to_be_bytes());
    hasher.update(&[src.as_u8()]);
    hasher.update(&[dst.as_u8()]);
    hasher.update(&[phase_bucket]);
    hasher.update(&ttfs.to_be_bytes());
    hasher.update(phase_commit.as_bytes());
    hasher.update(payload_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn advance_root(root: Digest32, event_commit: Digest32, domain: &[u8]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(root.as_bytes());
    hasher.update(event_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(cycle_id: u64, dst: OscId, kind: SpikeKind, ttfs: u16, seed: u8) -> SpikeEvent {
        SpikeEvent::new(
            cycle_id,
            kind,
            OscId::Reserved7,
            dst,
            3,
            ttfs,
            Digest32::new([seed; 32]),
            Digest32::new([seed.wrapping_add(1); 32]),
        )
    }

    #[test]
    fn batch_commit_is_deterministic() {
        let events = vec![
            make_event(1, OscId::Nsr, SpikeKind::Threat, 3, 1),
            make_event(1, OscId::Nsr, SpikeKind::Feature, 2, 2),
        ];
        let batch_a = SpikeBatch::new(1, Digest32::new([9u8; 32]), events.clone());
        let batch_b = SpikeBatch::new(1, Digest32::new([9u8; 32]), events);
        assert_eq!(batch_a.root, batch_b.root);
        assert_eq!(batch_a.commit, batch_b.commit);
    }

    #[test]
    fn spike_commit_changes_with_ttfs() {
        let base = SpikeEvent::new(
            1,
            SpikeKind::Feature,
            OscId::Reserved7,
            OscId::Ssm,
            4,
            12,
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
        );
        let shifted = SpikeEvent::new(
            1,
            SpikeKind::Feature,
            OscId::Reserved7,
            OscId::Ssm,
            4,
            14,
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
        );
        assert_ne!(base.commit, shifted.commit);
    }

    #[test]
    fn drain_orders_by_cycle_then_dst_then_kind_then_ttfs_then_commit() {
        let mut bus = SpikeBusState::new();
        let ev_a = make_event(2, OscId::Nsr, SpikeKind::Threat, 30, 1);
        let ev_b = make_event(1, OscId::Nsr, SpikeKind::CausalLink, 25, 2);
        let ev_c = make_event(1, OscId::Nsr, SpikeKind::CausalLink, 10, 3);
        let ev_d = make_event(1, OscId::Cde, SpikeKind::Novelty, 5, 4);
        let batch = SpikeBatch::new(
            1,
            Digest32::new([1u8; 32]),
            vec![ev_a.clone(), ev_b.clone(), ev_c.clone(), ev_d.clone()],
        );
        bus.append_batch(batch, vec![SpikeSuppression::default(); 4]);

        let drained = bus.drain_for(OscId::Nsr, 2, 10);
        assert_eq!(drained, vec![ev_c, ev_b, ev_a]);
        assert_eq!(bus.len(), 1);
    }

    #[test]
    fn roots_diverge_when_suppressed() {
        let mut bus = SpikeBusState::new();
        let event = make_event(1, OscId::Nsr, SpikeKind::CausalLink, 10, 5);
        let batch = SpikeBatch::new(1, Digest32::new([2u8; 32]), vec![event]);
        let summary = bus.append_batch(
            batch,
            vec![SpikeSuppression {
                suppressed_by_onn: true,
                suppressed_by_policy: false,
                suppressed_by_phase: false,
            }],
        );
        assert_ne!(summary.seen_root, summary.accepted_root);
    }

    #[test]
    fn cap_enforced_per_cycle() {
        let mut bus = SpikeBusState::new();
        let events = (0..(SPIKE_CYCLE_CAP + 4))
            .map(|idx| make_event(1, OscId::Nsr, SpikeKind::Feature, idx as u16, idx as u8))
            .collect::<Vec<_>>();
        let batch = SpikeBatch::new(1, Digest32::new([3u8; 32]), events);
        let summary = bus.append_batch(
            batch,
            vec![SpikeSuppression::default(); SPIKE_CYCLE_CAP + 4],
        );
        assert!(summary.cap_hit);
        assert!(bus.len() <= SPIKE_CYCLE_CAP);
    }

    #[test]
    #[should_panic]
    fn thought_only_rejects_output_targets() {
        let _ = SpikeEvent::new(
            1,
            SpikeKind::ThoughtOnly,
            OscId::Reserved7,
            OscId::Output,
            2,
            1,
            Digest32::new([0u8; 32]),
            Digest32::new([1u8; 32]),
        );
    }
}
