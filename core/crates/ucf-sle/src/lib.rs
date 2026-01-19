#![forbid(unsafe_code)]

use std::collections::{HashSet, VecDeque};
use std::sync::Mutex;

use blake3::Hasher;
use ucf_cde_port::CdeHypothesis;
use ucf_geist::{encode_self_state, SelfState};
use ucf_nsr_port::NsrReport;
use ucf_types::{AiOutput, Digest32, OutputChannel};
use ucf_workspace::{SignalKind, WorkspaceSnapshot};

const DOMAIN_INPUT: &[u8] = b"ucf.sle.input.v1";
const DOMAIN_OUTPUT: &[u8] = b"ucf.sle.output.v1";
const DOMAIN_REPORT: &[u8] = b"ucf.sle.report.v1";
const DOMAIN_SELF_SYMBOL: &[u8] = b"ucf.sle.self_symbol.v1";
const DOMAIN_SELF_REFLEX: &[u8] = b"ucf.sle.self_reflex.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfReflex {
    pub loop_level: u8,
    pub self_symbol: Digest32,
    pub delta: i16,
    pub commit: Digest32,
}

#[derive(Debug)]
pub struct SleEngine {
    max_level: u8,
    history: Mutex<SleHistory>,
}

#[derive(Debug, Default)]
struct SleHistory {
    loop_level: u8,
    last_high_priority: Vec<Digest32>,
}

impl SleEngine {
    pub fn new(max_level: u8) -> Self {
        Self {
            max_level,
            history: Mutex::new(SleHistory::default()),
        }
    }

    pub fn evaluate(&self, snapshot: &WorkspaceSnapshot, last_state: &SelfState) -> SelfReflex {
        let self_symbol = hash_with_domain(DOMAIN_SELF_SYMBOL, |hasher| {
            hasher.update(&encode_self_state(last_state));
            hasher.update(ucf_workspace::encode_workspace_snapshot(snapshot).as_slice());
        });

        let (stable_high_priority, instability) = {
            let current = high_priority_digests(snapshot);
            let stable = self
                .history
                .lock()
                .ok()
                .map(|history| has_stable_overlap(&history.last_high_priority, &current))
                .unwrap_or(false);
            (stable, has_instability(snapshot))
        };

        let mut loop_level = self
            .history
            .lock()
            .map(|history| history.loop_level)
            .unwrap_or(0);
        let delta: i16 = if instability {
            loop_level = loop_level.saturating_sub(1);
            -1
        } else if stable_high_priority {
            loop_level = loop_level.saturating_add(1).min(self.max_level);
            1
        } else {
            0
        };

        if let Ok(mut history) = self.history.lock() {
            history.loop_level = loop_level;
            history.last_high_priority = high_priority_digests(snapshot);
        }

        let commit = hash_with_domain(DOMAIN_SELF_REFLEX, |hasher| {
            hasher.update(&[loop_level]);
            hasher.update(&delta.to_be_bytes());
            hasher.update(self_symbol.as_bytes());
        });

        SelfReflex {
            loop_level,
            self_symbol,
            delta,
            commit,
        }
    }
}

fn high_priority_digests(snapshot: &WorkspaceSnapshot) -> Vec<Digest32> {
    let mut digests: Vec<Digest32> = snapshot
        .broadcast
        .iter()
        .filter(|signal| signal.priority >= 8000)
        .map(|signal| signal.digest)
        .collect();
    digests.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    digests.dedup();
    digests
}

fn has_stable_overlap(previous: &[Digest32], current: &[Digest32]) -> bool {
    if previous.is_empty() || current.is_empty() {
        return false;
    }
    let prev: HashSet<Digest32> = previous.iter().copied().collect();
    let overlap = current
        .iter()
        .filter(|digest| prev.contains(digest))
        .count();
    overlap * 2 >= previous.len().max(1)
}

fn has_instability(snapshot: &WorkspaceSnapshot) -> bool {
    snapshot.broadcast.iter().any(|signal| {
        (matches!(signal.kind, SignalKind::Consistency)
            && (signal.summary.contains("NSR=DAMP") || signal.summary.contains("NSR=VIOL")))
            || (matches!(signal.kind, SignalKind::World)
                && (signal.summary.contains("SURPRISE=HIGH")
                    || signal.summary.contains("SURPRISE=CRIT")))
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopFrame {
    pub input_commit: Digest32,
    pub output_commit: Digest32,
    pub report_commit: Digest32,
}

impl LoopFrame {
    pub fn seed(digest: Digest32) -> Self {
        Self {
            input_commit: digest,
            output_commit: digest,
            report_commit: digest,
        }
    }
}

#[derive(Debug)]
pub struct StrangeLoopEngine {
    pub depth: u8,
    frames: Mutex<VecDeque<LoopFrame>>,
}

impl StrangeLoopEngine {
    pub fn new(depth: u8) -> Self {
        Self {
            depth,
            frames: Mutex::new(VecDeque::new()),
        }
    }

    pub fn latest(&self) -> Option<LoopFrame> {
        self.frames.lock().ok()?.back().cloned()
    }

    pub fn reflect(
        &self,
        prev: &LoopFrame,
        current_output: &AiOutput,
        nsr_report: Option<&NsrReport>,
        cde_hyp: Option<&CdeHypothesis>,
    ) -> LoopFrame {
        let input_commit = hash_with_domain(DOMAIN_INPUT, |hasher| {
            hasher.update(prev.input_commit.as_bytes());
            hasher.update(prev.output_commit.as_bytes());
            hasher.update(prev.report_commit.as_bytes());
            encode_output(hasher, current_output);
        });

        let output_commit = hash_with_domain(DOMAIN_OUTPUT, |hasher| {
            hasher.update(input_commit.as_bytes());
            encode_output(hasher, current_output);
        });

        let report_commit = hash_with_domain(DOMAIN_REPORT, |hasher| {
            hasher.update(output_commit.as_bytes());
            encode_nsr_report(hasher, nsr_report);
            encode_cde_hypothesis(hasher, cde_hyp);
        });

        let frame = LoopFrame {
            input_commit,
            output_commit,
            report_commit,
        };

        if self.depth > 0 {
            if let Ok(mut frames) = self.frames.lock() {
                frames.push_back(frame.clone());
                let max_len = usize::from(self.depth);
                while frames.len() > max_len {
                    frames.pop_front();
                }
            }
        }

        frame
    }
}

fn hash_with_domain(domain: &[u8], f: impl FnOnce(&mut Hasher)) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    f(&mut hasher);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn encode_output(hasher: &mut Hasher, output: &AiOutput) {
    let channel_tag: u8 = match output.channel {
        OutputChannel::Thought => 0,
        OutputChannel::Speech => 1,
    };
    hasher.update(&[channel_tag]);
    hasher.update(&output.confidence.to_be_bytes());
    hasher.update(
        &u64::try_from(output.content.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    hasher.update(output.content.as_bytes());
    match output.rationale_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match output.integration_score {
        Some(score) => {
            hasher.update(&[1]);
            hasher.update(&score.to_be_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn encode_nsr_report(hasher: &mut Hasher, report: Option<&NsrReport>) {
    match report {
        Some(report) => {
            hasher.update(&[1]);
            hasher.update(&[report.ok as u8]);
            hasher.update(
                &u64::try_from(report.violations.len())
                    .unwrap_or(0)
                    .to_be_bytes(),
            );
            for violation in &report.violations {
                hasher.update(violation.as_bytes());
            }
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn encode_cde_hypothesis(hasher: &mut Hasher, hyp: Option<&CdeHypothesis>) {
    match hyp {
        Some(hyp) => {
            hasher.update(&[1]);
            hasher.update(hyp.digest.as_bytes());
            hasher.update(&u64::try_from(hyp.nodes).unwrap_or(0).to_be_bytes());
            hasher.update(&u64::try_from(hyp.edges).unwrap_or(0).to_be_bytes());
            hasher.update(&hyp.confidence.to_be_bytes());
            hasher.update(
                &u64::try_from(hyp.interventions.len())
                    .unwrap_or(0)
                    .to_be_bytes(),
            );
            for intervention in &hyp.interventions {
                hasher.update(
                    &u64::try_from(intervention.kind.len())
                        .unwrap_or(0)
                        .to_be_bytes(),
                );
                hasher.update(intervention.kind.as_bytes());
            }
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_workspace::{SignalKind, WorkspaceSignal};

    #[test]
    fn reflect_is_deterministic() {
        let engine = StrangeLoopEngine::new(4);
        let prev = LoopFrame::seed(Digest32::new([1u8; 32]));
        let output = AiOutput {
            channel: OutputChannel::Thought,
            content: "ok".to_string(),
            confidence: 900,
            rationale_commit: Some(Digest32::new([2u8; 32])),
            integration_score: Some(1234),
        };
        let report = NsrReport {
            ok: true,
            violations: vec!["rule-a".to_string()],
        };
        let hyp = CdeHypothesis {
            digest: Digest32::new([3u8; 32]),
            nodes: 2,
            edges: 1,
            confidence: 9000,
            interventions: vec![ucf_cde_port::InterventionStub::new("none")],
        };

        let first = engine.reflect(&prev, &output, Some(&report), Some(&hyp));
        let second = engine.reflect(&prev, &output, Some(&report), Some(&hyp));

        assert_eq!(first, second);
        let latest = engine.latest().expect("frame stored");
        assert_eq!(latest, second);
    }

    #[test]
    fn loop_level_increases_on_stable_high_priority() {
        let engine = SleEngine::new(4);
        let state = SelfState::builder(1).build();
        let signal = WorkspaceSignal {
            kind: SignalKind::Risk,
            priority: 9000,
            digest: Digest32::new([9u8; 32]),
            summary: "RISK=9000 DENY".to_string(),
            slot: 0,
        };
        let snapshot_a = WorkspaceSnapshot {
            cycle_id: 1,
            broadcast: vec![signal.clone()],
            commit: Digest32::new([1u8; 32]),
        };
        let snapshot_b = WorkspaceSnapshot {
            cycle_id: 2,
            broadcast: vec![signal],
            commit: Digest32::new([2u8; 32]),
        };

        let _ = engine.evaluate(&snapshot_a, &state);
        let reflex = engine.evaluate(&snapshot_b, &state);

        assert!(reflex.loop_level >= 1);
        assert_eq!(reflex.delta, 1);
    }

    #[test]
    fn loop_level_decreases_on_surprise_or_instability() {
        let engine = SleEngine::new(4);
        let state = SelfState::builder(1).build();
        let stable_signal = WorkspaceSignal {
            kind: SignalKind::Risk,
            priority: 9000,
            digest: Digest32::new([8u8; 32]),
            summary: "RISK=9000 DENY".to_string(),
            slot: 0,
        };
        let snapshot_a = WorkspaceSnapshot {
            cycle_id: 1,
            broadcast: vec![stable_signal.clone()],
            commit: Digest32::new([1u8; 32]),
        };
        let snapshot_b = WorkspaceSnapshot {
            cycle_id: 2,
            broadcast: vec![stable_signal],
            commit: Digest32::new([2u8; 32]),
        };
        let _ = engine.evaluate(&snapshot_a, &state);
        let _ = engine.evaluate(&snapshot_b, &state);

        let surprise_signal = WorkspaceSignal {
            kind: SignalKind::World,
            priority: 9000,
            digest: Digest32::new([7u8; 32]),
            summary: "SURPRISE=CRIT BAND=CRIT".to_string(),
            slot: 0,
        };
        let snapshot_c = WorkspaceSnapshot {
            cycle_id: 3,
            broadcast: vec![surprise_signal],
            commit: Digest32::new([3u8; 32]),
        };

        let reflex = engine.evaluate(&snapshot_c, &state);
        assert_eq!(reflex.delta, -1);
    }
}
