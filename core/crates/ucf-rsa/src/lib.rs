#![forbid(unsafe_code)]

use std::sync::Arc;

use blake3::Hasher;
use ucf_archive::ExperienceAppender;
use ucf_bus::BusPublisher;
use ucf_openevolve_port::{EvolutionProposal, OpenEvolvePort, SleepContext, SleepReport};
use ucf_policy_ecology::SleepPhaseGate;
use ucf_types::v1::spec::ExperienceRecord;
use ucf_types::{Digest32, EvidenceId};

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
        }
    }

    pub fn run_sleep_phase(
        &self,
        cycle_id: u64,
        fixed_seed: [u8; 32],
        integration_score: u16,
        recent_evidence: &[EvidenceId],
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
        let event = SleepReportReady {
            evidence_id,
            cycle_id,
        };
        self.bus.publish(event.clone());
        Some(event)
    }
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

        let result = coordinator.run_sleep_phase(1, [1u8; 32], 9, &[]);

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

        let result = coordinator.run_sleep_phase(42, [7u8; 32], 12, &evidence);

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
