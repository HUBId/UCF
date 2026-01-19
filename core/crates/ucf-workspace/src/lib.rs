#![forbid(unsafe_code)]

use blake3::Hasher;
use std::cmp::Ordering;
use ucf_types::Digest32;

const SUMMARY_MAX_BYTES: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum SignalKind {
    World,
    Policy,
    Risk,
    Attention,
    Integration,
    Consistency,
    Output,
    Sleep,
}

impl SignalKind {
    const COUNT: usize = 8;

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceSignal {
    pub kind: SignalKind,
    pub priority: u16,
    pub digest: Digest32,
    pub summary: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceConfig {
    pub cap: usize,
    pub broadcast_cap: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceSnapshot {
    pub cycle_id: u64,
    pub broadcast: Vec<WorkspaceSignal>,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DropCounters {
    pub total: u64,
    pub by_kind: [u64; SignalKind::COUNT],
}

pub struct Workspace {
    config: WorkspaceConfig,
    signals: Vec<SignalEntry>,
    next_seq: u64,
    drops: DropCounters,
}

impl Workspace {
    pub fn new(config: WorkspaceConfig) -> Self {
        Self {
            config,
            signals: Vec::new(),
            next_seq: 0,
            drops: DropCounters::default(),
        }
    }

    pub fn drop_counters(&self) -> DropCounters {
        self.drops
    }

    pub fn publish(&mut self, mut sig: WorkspaceSignal) {
        sig.summary = sanitize_summary(&sig.summary);
        let entry = SignalEntry {
            signal: sig,
            seq: self.next_seq,
        };
        self.next_seq = self.next_seq.wrapping_add(1);
        self.signals.push(entry);
        if self.signals.len() > self.config.cap {
            self.drop_excess();
        }
    }

    pub fn arbitrate(&mut self, cycle_id: u64) -> WorkspaceSnapshot {
        let mut entries = std::mem::take(&mut self.signals);
        entries.sort_by(compare_for_broadcast);
        let broadcast_len = self.config.broadcast_cap.min(entries.len());
        let broadcast_entries: Vec<SignalEntry> = entries.drain(..broadcast_len).collect();
        for entry in entries {
            self.note_drop(entry.signal.kind);
        }
        let broadcast: Vec<WorkspaceSignal> = broadcast_entries
            .iter()
            .map(|entry| entry.signal.clone())
            .collect();
        let commit = commit_snapshot(cycle_id, &broadcast);
        WorkspaceSnapshot {
            cycle_id,
            broadcast,
            commit,
        }
    }

    fn drop_excess(&mut self) {
        if let Some(idx) = self
            .signals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| compare_for_drop(a, b))
            .map(|(idx, _)| idx)
        {
            let entry = self.signals.remove(idx);
            self.note_drop(entry.signal.kind);
        }
    }

    fn note_drop(&mut self, kind: SignalKind) {
        self.drops.total += 1;
        self.drops.by_kind[kind.index()] += 1;
    }
}

#[derive(Clone, Debug)]
struct SignalEntry {
    signal: WorkspaceSignal,
    seq: u64,
}

fn compare_for_drop(a: &SignalEntry, b: &SignalEntry) -> Ordering {
    let priority_cmp = a.signal.priority.cmp(&b.signal.priority);
    if priority_cmp != Ordering::Equal {
        return priority_cmp;
    }
    let kind_cmp = a.signal.kind.cmp(&b.signal.kind);
    if kind_cmp != Ordering::Equal {
        return kind_cmp;
    }
    let digest_cmp = a.signal.digest.as_bytes().cmp(b.signal.digest.as_bytes());
    if digest_cmp != Ordering::Equal {
        return digest_cmp;
    }
    a.seq.cmp(&b.seq)
}

fn compare_for_broadcast(a: &SignalEntry, b: &SignalEntry) -> Ordering {
    let priority_cmp = b.signal.priority.cmp(&a.signal.priority);
    if priority_cmp != Ordering::Equal {
        return priority_cmp;
    }
    let kind_cmp = a.signal.kind.cmp(&b.signal.kind);
    if kind_cmp != Ordering::Equal {
        return kind_cmp;
    }
    let digest_cmp = a.signal.digest.as_bytes().cmp(b.signal.digest.as_bytes());
    if digest_cmp != Ordering::Equal {
        return digest_cmp;
    }
    a.seq.cmp(&b.seq)
}

fn commit_snapshot(cycle_id: u64, broadcast: &[WorkspaceSignal]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&cycle_id.to_be_bytes());
    for signal in broadcast {
        hasher.update(&[signal.kind as u8]);
        hasher.update(&signal.priority.to_be_bytes());
        hasher.update(signal.digest.as_bytes());
        hasher.update(signal.summary.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn sanitize_summary(summary: &str) -> String {
    let mut cleaned = String::with_capacity(summary.len().min(SUMMARY_MAX_BYTES));
    for ch in summary.chars() {
        if ch.is_control() {
            continue;
        }
        cleaned.push(ch);
        if cleaned.len() >= SUMMARY_MAX_BYTES {
            break;
        }
    }
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "summary redacted".to_string()
    } else {
        trimmed.to_string()
    }
}
