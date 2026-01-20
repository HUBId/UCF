#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use blake3::Hasher;
use ucf_bus::BusPublisher;
use ucf_policy_ecology::{ConsistencyVerdict, SleepPhaseGate};
use ucf_predictive_coding::SurpriseBand;
use ucf_rsa::{RsaEngine, SleepCoordinator, SleepReportReady};
use ucf_types::EvidenceId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SleepHeuristics {
    pub min_records_since_last: u32,
    pub max_instability: u16,
    pub min_integration: u16,
    pub critical_surprise_threshold: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepState {
    pub last_cycle_id: u64,
    pub last_evidence: Option<EvidenceId>,
    pub records_since_last: u32,
    pub critical_surprise_count: u16,
    pub last_replay: Option<SleepReplaySummary>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepTrigger {
    None,
    Instability,
    Density,
    LowIntegration,
    Manual,
    SurpriseCritical,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecentMetrics {
    pub consistency_verdicts: Vec<ConsistencyVerdict>,
    pub integration_scores: Vec<u16>,
    pub records_since_last: u32,
}

impl RecentMetrics {
    pub fn instability_score(&self) -> u16 {
        if self.consistency_verdicts.is_empty() {
            return 0;
        }
        let unstable = self
            .consistency_verdicts
            .iter()
            .filter(|verdict| **verdict != ConsistencyVerdict::Accept)
            .count();
        let total = self.consistency_verdicts.len();
        let score = (unstable as u32 * 10_000) / total.max(1) as u32;
        u16::try_from(score.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
    }

    pub fn average_integration_score(&self) -> u16 {
        if self.integration_scores.is_empty() {
            return 0;
        }
        let sum: u32 = self
            .integration_scores
            .iter()
            .map(|score| u32::from(*score))
            .sum();
        let avg = sum / self.integration_scores.len().max(1) as u32;
        u16::try_from(avg.min(u32::from(u16::MAX))).unwrap_or(u16::MAX)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SleepReplaySummary {
    pub micro: u16,
    pub meso: u16,
    pub macro_: u16,
}

pub trait SleepStateUpdater {
    fn record_derived_record(&mut self, evidence_id: EvidenceId);
    fn record_consistency_verdict(&mut self, verdict: ConsistencyVerdict);
    fn record_integration_score(&mut self, score: u16);
    fn record_surprise_band(&mut self, _band: SurpriseBand) {}
    fn record_replay_summary(&mut self, _summary: SleepReplaySummary) {}
}

pub type SleepStateHandle = Arc<Mutex<WalSleepCoordinator>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepTriggered {
    pub cycle_id: u64,
    pub reason: SleepTrigger,
}

pub trait SleepPhaseRunner {
    fn run_sleep_phase(
        &self,
        cycle_id: u64,
        fixed_seed: [u8; 32],
        integration_score: u16,
        recent_evidence: &[EvidenceId],
    ) -> Option<SleepReportReady>;
}

impl<P, R, O, B> SleepPhaseRunner for SleepCoordinator<P, R, O, B>
where
    P: SleepPhaseGate,
    R: RsaEngine,
    O: ucf_openevolve_port::OpenEvolvePort,
    B: BusPublisher<SleepReportReady>,
{
    fn run_sleep_phase(
        &self,
        cycle_id: u64,
        fixed_seed: [u8; 32],
        integration_score: u16,
        recent_evidence: &[EvidenceId],
    ) -> Option<SleepReportReady> {
        SleepCoordinator::run_sleep_phase(
            self,
            cycle_id,
            fixed_seed,
            integration_score,
            recent_evidence,
        )
    }
}

pub struct WalSleepCoordinator {
    heuristics: SleepHeuristics,
    state: SleepState,
    window: usize,
    consistency_verdicts: VecDeque<ConsistencyVerdict>,
    integration_scores: VecDeque<u16>,
    recent_evidence: VecDeque<EvidenceId>,
}

impl WalSleepCoordinator {
    pub fn new(heuristics: SleepHeuristics, window: usize) -> Self {
        Self {
            heuristics,
            state: SleepState {
                last_cycle_id: 0,
                last_evidence: None,
                records_since_last: 0,
                critical_surprise_count: 0,
                last_replay: None,
            },
            window,
            consistency_verdicts: VecDeque::new(),
            integration_scores: VecDeque::new(),
            recent_evidence: VecDeque::new(),
        }
    }

    pub fn heuristics(&self) -> &SleepHeuristics {
        &self.heuristics
    }

    pub fn state(&self) -> &SleepState {
        &self.state
    }

    pub fn recent_metrics(&self) -> RecentMetrics {
        RecentMetrics {
            consistency_verdicts: self.consistency_verdicts.iter().copied().collect(),
            integration_scores: self.integration_scores.iter().copied().collect(),
            records_since_last: self.state.records_since_last,
        }
    }

    pub fn recent_evidence(&self) -> Vec<EvidenceId> {
        self.recent_evidence.iter().cloned().collect()
    }

    pub fn evaluate(&mut self, recent_metrics: &RecentMetrics) -> SleepTrigger {
        if self.heuristics.critical_surprise_threshold > 0
            && self.state.critical_surprise_count >= self.heuristics.critical_surprise_threshold
        {
            self.commit_trigger();
            return SleepTrigger::SurpriseCritical;
        }
        let instability = recent_metrics.instability_score();
        if instability > self.heuristics.max_instability {
            self.commit_trigger();
            return SleepTrigger::Instability;
        }

        let integration_score = recent_metrics.average_integration_score();
        if integration_score < self.heuristics.min_integration {
            self.commit_trigger();
            return SleepTrigger::LowIntegration;
        }

        if recent_metrics.records_since_last >= self.heuristics.min_records_since_last {
            self.commit_trigger();
            return SleepTrigger::Density;
        }

        SleepTrigger::None
    }

    pub fn maybe_trigger<R, B>(&mut self, runner: &R, bus: &B) -> Option<SleepReportReady>
    where
        R: SleepPhaseRunner + ?Sized,
        B: BusPublisher<SleepTriggered>,
    {
        let recent_metrics = self.recent_metrics();
        let recent_evidence = self.recent_evidence();
        let trigger = self.evaluate(&recent_metrics);
        if trigger == SleepTrigger::None {
            return None;
        }

        let cycle_id = self.state.last_cycle_id;
        let fixed_seed = derive_fixed_seed(cycle_id, self.state.last_evidence.as_ref());
        let integration_score = recent_metrics.average_integration_score();
        bus.publish(SleepTriggered {
            cycle_id,
            reason: trigger,
        });
        let result =
            runner.run_sleep_phase(cycle_id, fixed_seed, integration_score, &recent_evidence);
        self.reset_after_trigger();
        result
    }

    pub fn force_trigger(&mut self) -> SleepTrigger {
        self.commit_trigger();
        SleepTrigger::Manual
    }

    fn commit_trigger(&mut self) {
        self.state.last_cycle_id = self.state.last_cycle_id.saturating_add(1);
        self.state.records_since_last = 0;
        self.state.critical_surprise_count = 0;
    }

    fn reset_after_trigger(&mut self) {
        self.consistency_verdicts.clear();
        self.integration_scores.clear();
        self.recent_evidence.clear();
        self.state.critical_surprise_count = 0;
    }

    fn push_bounded<T>(queue: &mut VecDeque<T>, window: usize, value: T) {
        queue.push_back(value);
        if window > 0 {
            while queue.len() > window {
                queue.pop_front();
            }
        }
    }
}

impl SleepStateUpdater for WalSleepCoordinator {
    fn record_derived_record(&mut self, evidence_id: EvidenceId) {
        self.state.records_since_last = self.state.records_since_last.saturating_add(1);
        self.state.last_evidence = Some(evidence_id.clone());
        Self::push_bounded(&mut self.recent_evidence, self.window, evidence_id);
    }

    fn record_consistency_verdict(&mut self, verdict: ConsistencyVerdict) {
        Self::push_bounded(&mut self.consistency_verdicts, self.window, verdict);
    }

    fn record_integration_score(&mut self, score: u16) {
        Self::push_bounded(&mut self.integration_scores, self.window, score);
    }

    fn record_surprise_band(&mut self, band: SurpriseBand) {
        if band == SurpriseBand::Critical {
            self.state.critical_surprise_count =
                self.state.critical_surprise_count.saturating_add(1);
        } else {
            self.state.critical_surprise_count = 0;
        }
    }

    fn record_replay_summary(&mut self, summary: SleepReplaySummary) {
        self.state.last_replay = Some(summary);
    }
}

fn derive_fixed_seed(cycle_id: u64, last_evidence: Option<&EvidenceId>) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.sleep.wal.v1");
    hasher.update(&cycle_id.to_le_bytes());
    if let Some(evidence) = last_evidence {
        hasher.update(evidence.as_str().as_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use ucf_archive::InMemoryArchive;
    use ucf_bus::{BusSubscriber, InMemoryBus};
    use ucf_openevolve_port::MockOpenEvolvePort;
    use ucf_policy_ecology::{DefaultPolicyEcology, PolicyEcology, PolicyRule, PolicyWeights};
    use ucf_rsa::MockRsaEngine;

    #[test]
    fn sleep_trigger_density_increments_cycle() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 2,
            max_instability: 10_000,
            min_integration: 0,
            critical_surprise_threshold: 0,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_derived_record(EvidenceId::new("rec-1"));
        coordinator.record_derived_record(EvidenceId::new("rec-2"));
        coordinator.record_consistency_verdict(ConsistencyVerdict::Accept);
        coordinator.record_integration_score(9000);

        let metrics = coordinator.recent_metrics();
        let trigger = coordinator.evaluate(&metrics);

        assert_eq!(trigger, SleepTrigger::Density);
        assert_eq!(coordinator.state.last_cycle_id, 1);
        assert_eq!(coordinator.state.records_since_last, 0);
    }

    #[test]
    fn sleep_trigger_instability_is_deterministic() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 10,
            max_instability: 2000,
            min_integration: 0,
            critical_surprise_threshold: 0,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_consistency_verdict(ConsistencyVerdict::Reject);
        coordinator.record_consistency_verdict(ConsistencyVerdict::Accept);
        coordinator.record_integration_score(9000);

        let metrics = coordinator.recent_metrics();
        let trigger = coordinator.evaluate(&metrics);

        assert_eq!(trigger, SleepTrigger::Instability);
        assert_eq!(coordinator.state.last_cycle_id, 1);
    }

    #[test]
    fn policy_denial_prevents_sleep_report_append() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 1,
            max_instability: 10_000,
            min_integration: 0,
            critical_surprise_threshold: 0,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_derived_record(EvidenceId::new("rec-1"));
        coordinator.record_consistency_verdict(ConsistencyVerdict::Accept);
        coordinator.record_integration_score(9000);

        let archive = Arc::new(InMemoryArchive::new());
        let sleep_bus = InMemoryBus::new();
        let triggered_bus = InMemoryBus::new();
        let triggered_rx = triggered_bus.subscribe();
        let policy = DefaultPolicyEcology::new();
        let rsa = MockRsaEngine::new();
        let openevolve = MockOpenEvolvePort::default();
        let runner = SleepCoordinator::new(policy, rsa, openevolve, archive.clone(), sleep_bus);

        let result = coordinator.maybe_trigger(&runner, &triggered_bus);

        assert!(result.is_none());
        assert!(archive.list().is_empty());
        let triggered = triggered_rx.recv().expect("sleep triggered event");
        assert_eq!(triggered.reason, SleepTrigger::Density);
    }

    #[test]
    fn policy_allow_appends_sleep_report() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 1,
            max_instability: 10_000,
            min_integration: 0,
            critical_surprise_threshold: 0,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_derived_record(EvidenceId::new("rec-1"));
        coordinator.record_consistency_verdict(ConsistencyVerdict::Accept);
        coordinator.record_integration_score(9000);

        let archive = Arc::new(InMemoryArchive::new());
        let sleep_bus = InMemoryBus::new();
        let triggered_bus = InMemoryBus::new();
        let policy = PolicyEcology::new(1, vec![PolicyRule::AllowSleepPhase], PolicyWeights);
        let rsa = MockRsaEngine::new();
        let openevolve = MockOpenEvolvePort::default();
        let runner = SleepCoordinator::new(policy, rsa, openevolve, archive.clone(), sleep_bus);

        let result = coordinator.maybe_trigger(&runner, &triggered_bus);

        assert!(result.is_some());
        assert_eq!(archive.list().len(), 1);
    }

    #[test]
    fn repeated_critical_surprise_triggers_sleep() {
        let heuristics = SleepHeuristics {
            min_records_since_last: 10,
            max_instability: 10_000,
            min_integration: 0,
            critical_surprise_threshold: 2,
        };
        let mut coordinator = WalSleepCoordinator::new(heuristics, 4);
        coordinator.record_surprise_band(SurpriseBand::Critical);
        coordinator.record_surprise_band(SurpriseBand::Critical);
        coordinator.record_integration_score(9000);

        let metrics = coordinator.recent_metrics();
        let trigger = coordinator.evaluate(&metrics);

        assert_eq!(trigger, SleepTrigger::SurpriseCritical);
        assert_eq!(coordinator.state.last_cycle_id, 1);
    }
}
