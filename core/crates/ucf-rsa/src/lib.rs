#![forbid(unsafe_code)]

use std::sync::Arc;

use blake3::Hasher;
use ucf_archive::ExperienceAppender;
use ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind, RecordMeta};
use ucf_bus::BusPublisher;
use ucf_openevolve_port::{EvolutionProposal, OpenEvolvePort, SleepContext, SleepReport};
use ucf_policy_ecology::SleepPhaseGate;
use ucf_structural_store::{
    ReasonCode, StructuralCommitResult, StructuralCycleStats, StructuralDeltaProposal,
    StructuralGates, StructuralStore,
};
use ucf_types::v1::spec::ExperienceRecord;
use ucf_types::{Digest32, EvidenceId};

const STRUCTURAL_SEED_DOMAIN: &[u8] = b"ucf.rsa.structural.seed.v1";
const NSR_DENY_VERDICT: u8 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepReportReady {
    pub evidence_id: EvidenceId,
    pub cycle_id: u64,
}

pub trait RsaEngine {
    fn analyze(&self, ctx: &SleepContext, recent_evidence: &[EvidenceId]) -> SleepReport;
}

#[derive(Clone, Default)]
pub struct MockRsaEngine;

impl MockRsaEngine {
    pub fn new() -> Self {
        Self
    }

    fn proposal_from_evidence(
        &self,
        ctx: &SleepContext,
        evidence: &EvidenceId,
    ) -> EvolutionProposal {
        let mut hasher = Hasher::new();
        hasher.update(&ctx.fixed_seed);
        hasher.update(evidence.as_str().as_bytes());
        let digest = Digest32::new(*hasher.finalize().as_bytes());
        let expected_gain = (digest.as_bytes()[0] % 21) as i16 - 10;
        let risk = digest.as_bytes()[1] as u16;
        EvolutionProposal {
            id: digest,
            title: format!("RSA hint for {}", evidence.as_str()),
            rationale: format!("derived from evidence {}", evidence.as_str()),
            expected_gain,
            risk,
        }
    }
}

impl RsaEngine for MockRsaEngine {
    fn analyze(&self, ctx: &SleepContext, recent_evidence: &[EvidenceId]) -> SleepReport {
        let proposals = recent_evidence
            .iter()
            .map(|evidence| self.proposal_from_evidence(ctx, evidence))
            .collect::<Vec<_>>();
        let metrics = vec![
            ("rsa_evidence".to_string(), recent_evidence.len() as u32),
            (
                "integration_score".to_string(),
                ctx.integration_score as u32,
            ),
        ];
        SleepReport { proposals, metrics }
    }
}

pub struct SleepCoordinator<P, R, O, B>
where
    P: SleepPhaseGate,
    R: RsaEngine,
    O: OpenEvolvePort,
    B: BusPublisher<SleepReportReady>,
{
    policy: P,
    rsa: R,
    openevolve: O,
    archive: Arc<dyn ExperienceAppender + Send + Sync>,
    bus: B,
    structural_store: Option<Arc<std::sync::Mutex<StructuralStore>>>,
    structural_archive_store: Option<Arc<dyn ArchiveStore + Send + Sync>>,
    structural_archive_appender: std::sync::Mutex<ArchiveAppender>,
}

impl<P, R, O, B> SleepCoordinator<P, R, O, B>
where
    P: SleepPhaseGate,
    R: RsaEngine,
    O: OpenEvolvePort,
    B: BusPublisher<SleepReportReady>,
{
    pub fn new(
        policy: P,
        rsa: R,
        openevolve: O,
        archive: Arc<dyn ExperienceAppender + Send + Sync>,
        bus: B,
    ) -> Self {
        Self {
            policy,
            rsa,
            openevolve,
            archive,
            bus,
            structural_store: None,
            structural_archive_store: None,
            structural_archive_appender: std::sync::Mutex::new(ArchiveAppender::new()),
        }
    }

    pub fn with_structural_store(
        mut self,
        store: Arc<std::sync::Mutex<StructuralStore>>,
        archive_store: Arc<dyn ArchiveStore + Send + Sync>,
    ) -> Self {
        self.structural_store = Some(store);
        self.structural_archive_store = Some(archive_store);
        self
    }

    pub fn run_sleep_phase(
        &self,
        cycle_id: u64,
        fixed_seed: [u8; 32],
        integration_score: u16,
        recent_evidence: &[EvidenceId],
        structural_stats: Option<StructuralCycleStats>,
        structural_proposal: Option<StructuralDeltaProposal>,
    ) -> Option<SleepReportReady> {
        if !self.policy.allow_sleep() {
            return None;
        }

        let ctx = SleepContext {
            cycle_id,
            fixed_seed,
            integration_score,
        };
        let base_report = self.rsa.analyze(&ctx, recent_evidence);
        let proposals = self.openevolve.propose(&ctx, &base_report);
        let mut metrics = base_report.metrics.clone();
        metrics.push(("openevolve_selected".to_string(), proposals.len() as u32));
        let report = SleepReport { proposals, metrics };
        let record = build_sleep_record(&ctx, &report);
        let evidence_id = self.archive.append(record);
        self.maybe_commit_structural(cycle_id, structural_stats, structural_proposal);
        let event = SleepReportReady {
            evidence_id,
            cycle_id,
        };
        self.bus.publish(event.clone());
        Some(event)
    }

    fn maybe_commit_structural(
        &self,
        cycle_id: u64,
        structural_stats: Option<StructuralCycleStats>,
        structural_proposal: Option<StructuralDeltaProposal>,
    ) {
        let (Some(stats), Some(proposal)) = (structural_stats, structural_proposal) else {
            return;
        };
        let Some(store_handle) = self.structural_store.as_ref() else {
            return;
        };
        let mut store = match store_handle.lock() {
            Ok(store) => store,
            Err(err) => err.into_inner(),
        };
        let gates = StructuralGates::new(
            stats.nsr_verdict != NSR_DENY_VERDICT,
            stats.policy_ok,
            stats.consistency_ok,
            stats.coherence_plv,
            stats.drift,
            stats.surprise,
        );
        let result = store.evaluate_and_commit(&proposal, &gates);
        self.append_structural_records(cycle_id, &proposal, &result, &store);
    }

    fn append_structural_records(
        &self,
        cycle_id: u64,
        proposal: &StructuralDeltaProposal,
        result: &StructuralCommitResult,
        store: &StructuralStore,
    ) {
        let Some(archive_store) = self.structural_archive_store.as_ref() else {
            return;
        };
        let mut appender = self
            .structural_archive_appender
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let decision_flag = matches!(result, StructuralCommitResult::Committed { .. }) as u16;
        let meta = RecordMeta {
            cycle_id,
            tier: 0,
            flags: decision_flag,
            boundary_commit: Digest32::new([0u8; 32]),
        };
        let proposal_record = appender.build_record_with_commit(
            RecordKind::StructuralProposal,
            proposal.commit,
            meta,
        );
        archive_store.append(proposal_record);
        if let StructuralCommitResult::Committed { .. } = result {
            let params_record = appender.build_record_with_commit(
                RecordKind::StructuralParams,
                store.current.commit,
                meta,
            );
            archive_store.append(params_record);
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructuralProposalEngine {
    coherence_floor: u16,
    integration_floor: u16,
    drift_ceiling: u16,
    nsr_warn_repeats: u16,
}

impl Default for StructuralProposalEngine {
    fn default() -> Self {
        Self {
            coherence_floor: 5200,
            integration_floor: 4200,
            drift_ceiling: 6800,
            nsr_warn_repeats: 2,
        }
    }
}

impl StructuralProposalEngine {
    pub fn maybe_propose(
        &self,
        store: &StructuralStore,
        cycle_id: u64,
        stats: &StructuralCycleStats,
        nsr_warn_streak: u16,
        evidence: Vec<Digest32>,
    ) -> Option<StructuralDeltaProposal> {
        let mut reasons = Vec::new();
        if stats.coherence_plv < self.coherence_floor {
            reasons.push(ReasonCode::CoherenceLow);
        }
        if stats.integration_phi < self.integration_floor {
            reasons.push(ReasonCode::IntegrationLow);
        }
        if stats.drift > self.drift_ceiling {
            reasons.push(ReasonCode::DriftHigh);
        }
        if nsr_warn_streak >= self.nsr_warn_repeats {
            reasons.push(ReasonCode::NsrWarn);
        }
        if reasons.is_empty() {
            return None;
        }

        let seed = structural_seed(store.current.commit, stats.commit);
        Some(store.propose(cycle_id, reasons, evidence, seed))
    }
}

fn structural_seed(current_commit: Digest32, stats_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(STRUCTURAL_SEED_DOMAIN);
    hasher.update(current_commit.as_bytes());
    hasher.update(stats_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn build_sleep_record(ctx: &SleepContext, report: &SleepReport) -> ExperienceRecord {
    let mut metrics = report.metrics.clone();
    metrics.sort_by(|left, right| left.0.cmp(&right.0));
    let metric_payload = metrics
        .iter()
        .map(|(key, value)| format!("{key}:{value}"))
        .collect::<Vec<_>>()
        .join(",");
    let proposals_payload = report
        .proposals
        .iter()
        .map(|proposal| {
            format!(
                "{}|{}|{}|{}|{}",
                proposal.id,
                proposal.title,
                proposal.expected_gain,
                proposal.risk,
                proposal.rationale
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    let payload = format!(
        "derived=sleep_report;sleep_cycle={};integration_score={};proposal_count={};metrics={};proposals={}",
        ctx.cycle_id,
        ctx.integration_score,
        report.proposals.len(),
        metric_payload,
        proposals_payload
    )
    .into_bytes();
    ExperienceRecord {
        record_id: format!("sleep-report-{}", ctx.cycle_id),
        observed_at_ms: ctx.cycle_id,
        subject_id: "sleep-phase".to_string(),
        payload,
        digest: None,
        vrf_tag: None,
        proof_ref: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use prost::Message;
    use ucf_archive::InMemoryArchive;
    use ucf_bus::{BusSubscriber, InMemoryBus};
    use ucf_policy_ecology::{PolicyEcology, PolicyRule, PolicyWeights};
    use ucf_types::v1::spec::ExperienceRecord as ProtoExperienceRecord;

    #[test]
    fn sleep_phase_denied_by_default() {
        let policy = ucf_policy_ecology::DefaultPolicyEcology::new();
        let rsa = MockRsaEngine::new();
        let openevolve = ucf_openevolve_port::MockOpenEvolvePort::default();
        let archive = Arc::new(InMemoryArchive::new());
        let bus = InMemoryBus::new();
        let receiver = bus.subscribe();
        let coordinator = SleepCoordinator::new(policy, rsa, openevolve, archive.clone(), bus);

        let result = coordinator.run_sleep_phase(1, [1u8; 32], 9, &[], None, None);

        assert!(result.is_none());
        assert!(archive.list().is_empty());
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn sleep_phase_allowed_appends_report_and_emits_event() {
        let policy = PolicyEcology::new(1, vec![PolicyRule::AllowSleepPhase], PolicyWeights);
        let rsa = MockRsaEngine::new();
        let openevolve = ucf_openevolve_port::MockOpenEvolvePort::new(5);
        let archive = Arc::new(InMemoryArchive::new());
        let bus = InMemoryBus::new();
        let receiver = bus.subscribe();
        let coordinator = SleepCoordinator::new(policy, rsa, openevolve, archive.clone(), bus);
        let evidence = vec![EvidenceId::new("ev-1"), EvidenceId::new("ev-2")];

        let result = coordinator.run_sleep_phase(42, [7u8; 32], 12, &evidence, None, None);

        assert!(result.is_some());
        assert_eq!(archive.list().len(), 1);
        let envelope = &archive.list()[0];
        let proof = envelope.proof.as_ref().expect("proof envelope");
        let record =
            ProtoExperienceRecord::decode(proof.payload.as_slice()).expect("decode record");
        let payload = String::from_utf8(record.payload).expect("utf8 payload");
        assert!(payload.contains("derived=sleep_report"));
        assert!(payload.contains("proposal_count=2"));
        let event = receiver.recv().expect("sleep event");
        assert_eq!(event.cycle_id, 42);
    }

    #[test]
    fn sleep_report_proposals_are_deterministically_sorted() {
        let ctx = SleepContext {
            cycle_id: 5,
            fixed_seed: [2u8; 32],
            integration_score: 3,
        };
        let report = SleepReport {
            proposals: vec![
                EvolutionProposal {
                    id: Digest32::new([9u8; 32]),
                    title: "b".to_string(),
                    rationale: "b".to_string(),
                    expected_gain: 2,
                    risk: 4,
                },
                EvolutionProposal {
                    id: Digest32::new([1u8; 32]),
                    title: "a".to_string(),
                    rationale: "a".to_string(),
                    expected_gain: 4,
                    risk: 2,
                },
                EvolutionProposal {
                    id: Digest32::new([3u8; 32]),
                    title: "c".to_string(),
                    rationale: "c".to_string(),
                    expected_gain: 4,
                    risk: 8,
                },
            ],
            metrics: Vec::new(),
        };
        let openevolve = ucf_openevolve_port::MockOpenEvolvePort::new(3);

        let proposals = openevolve.propose(&ctx, &report);

        assert_eq!(proposals[0].id, Digest32::new([1u8; 32]));
        assert_eq!(proposals[1].id, Digest32::new([3u8; 32]));
        assert_eq!(proposals[2].id, Digest32::new([9u8; 32]));
    }
}
