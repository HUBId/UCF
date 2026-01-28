#![forbid(unsafe_code)]

use std::cmp::Ordering;

use blake3::Hasher;
use ucf_onn::OscId;
use ucf_types::Digest32;

pub use ucf_onn::OscId as SpikeModuleId;

const SPIKE_EVENT_DOMAIN: &[u8] = b"ucf.spikebus.event.v1";
const SPIKE_ROOT_DOMAIN: &[u8] = b"ucf.spikebus.root.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpikeKind {
    Novelty,
    Threat,
    CausalLink,
    ConsistencyAlert,
    ReplayTrigger,
    AttentionShift,
    Thought,
    Unknown(u16),
}

impl SpikeKind {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::Novelty => 1,
            Self::Threat => 2,
            Self::CausalLink => 3,
            Self::ConsistencyAlert => 4,
            Self::ReplayTrigger => 5,
            Self::AttentionShift => 6,
            Self::Thought => 7,
            Self::Unknown(code) => code,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeEvent {
    pub cycle_id: u64,
    pub src: OscId,
    pub dst: OscId,
    pub kind: SpikeKind,
    pub ttfs_code: u16,
    pub amplitude: u16,
    pub width: u16,
    pub phase_ref: Digest32,
    pub payload_commit: Digest32,
    pub commit: Digest32,
}

impl SpikeEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        src: OscId,
        dst: OscId,
        kind: SpikeKind,
        ttfs_code: u16,
        amplitude: u16,
        width: u16,
        phase_ref: Digest32,
        payload_commit: Digest32,
    ) -> Self {
        let commit = commit_spike_event(
            cycle_id,
            src,
            dst,
            kind,
            ttfs_code,
            amplitude,
            width,
            phase_ref,
            payload_commit,
        );
        Self {
            cycle_id,
            src,
            dst,
            kind,
            ttfs_code,
            amplitude,
            width,
            phase_ref,
            payload_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeBusState {
    events: Vec<SpikeEvent>,
    root_commit: Digest32,
}

impl Default for SpikeBusState {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            root_commit: Digest32::new([0u8; 32]),
        }
    }
}

impl SpikeBusState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, ev: SpikeEvent) {
        let mut hasher = Hasher::new();
        hasher.update(SPIKE_ROOT_DOMAIN);
        hasher.update(self.root_commit.as_bytes());
        hasher.update(ev.commit.as_bytes());
        self.root_commit = Digest32::new(*hasher.finalize().as_bytes());
        self.events.push(ev);
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

    pub fn root_commit(&self) -> Digest32 {
        self.root_commit
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

fn compare_spikes(a: &SpikeEvent, b: &SpikeEvent) -> Ordering {
    let cycle_cmp = a.cycle_id.cmp(&b.cycle_id);
    if cycle_cmp != Ordering::Equal {
        return cycle_cmp;
    }
    let dst_cmp = a.dst.as_u16().cmp(&b.dst.as_u16());
    if dst_cmp != Ordering::Equal {
        return dst_cmp;
    }
    let kind_cmp = a.kind.cmp(&b.kind);
    if kind_cmp != Ordering::Equal {
        return kind_cmp;
    }
    let ttfs_cmp = a.ttfs_code.cmp(&b.ttfs_code);
    if ttfs_cmp != Ordering::Equal {
        return ttfs_cmp;
    }
    a.commit.as_bytes().cmp(b.commit.as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_spike_event(
    cycle_id: u64,
    src: OscId,
    dst: OscId,
    kind: SpikeKind,
    ttfs_code: u16,
    amplitude: u16,
    width: u16,
    phase_ref: Digest32,
    payload_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SPIKE_EVENT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&src.as_u16().to_be_bytes());
    hasher.update(&dst.as_u16().to_be_bytes());
    hasher.update(&kind.as_u16().to_be_bytes());
    hasher.update(&ttfs_code.to_be_bytes());
    hasher.update(&amplitude.to_be_bytes());
    hasher.update(&width.to_be_bytes());
    hasher.update(phase_ref.as_bytes());
    hasher.update(payload_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(cycle_id: u64, dst: OscId, kind: SpikeKind, ttfs: u16, seed: u8) -> SpikeEvent {
        SpikeEvent::new(
            cycle_id,
            OscId::Jepa,
            dst,
            kind,
            ttfs,
            10,
            20,
            Digest32::new([seed; 32]),
            Digest32::new([seed.wrapping_add(1); 32]),
        )
    }

    #[test]
    fn drain_orders_by_cycle_then_dst_then_kind_then_ttfs_then_commit() {
        let mut bus = SpikeBusState::new();
        let ev_a = make_event(2, OscId::Nsr, SpikeKind::Threat, 30, 1);
        let ev_b = make_event(1, OscId::Nsr, SpikeKind::CausalLink, 25, 2);
        let ev_c = make_event(1, OscId::Nsr, SpikeKind::CausalLink, 10, 3);
        let ev_d = make_event(1, OscId::Cde, SpikeKind::Novelty, 5, 4);
        bus.append(ev_a.clone());
        bus.append(ev_b.clone());
        bus.append(ev_c.clone());
        bus.append(ev_d.clone());

        let drained = bus.drain_for(OscId::Nsr, 2, 10);
        assert_eq!(drained, vec![ev_c, ev_b, ev_a]);
        assert_eq!(bus.len(), 1);
    }

    #[test]
    fn root_commit_advances_on_append() {
        let mut bus = SpikeBusState::new();
        let root_before = bus.root_commit();
        bus.append(make_event(1, OscId::Nsr, SpikeKind::Threat, 12, 9));
        assert_ne!(root_before, bus.root_commit());
    }
}
