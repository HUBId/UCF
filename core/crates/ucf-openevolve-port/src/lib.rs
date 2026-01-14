#![forbid(unsafe_code)]

use ucf_types::Digest32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepContext {
    pub cycle_id: u64,
    pub fixed_seed: [u8; 32],
    pub integration_score: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvolutionProposal {
    pub id: Digest32,
    pub title: String,
    pub rationale: String,
    pub expected_gain: i16,
    pub risk: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SleepReport {
    pub proposals: Vec<EvolutionProposal>,
    pub metrics: Vec<(String, u32)>,
}

pub trait OpenEvolvePort {
    fn propose(&self, ctx: &SleepContext, report: &SleepReport) -> Vec<EvolutionProposal>;
}

#[derive(Clone, Debug)]
pub struct MockOpenEvolvePort {
    top_k: usize,
}

impl MockOpenEvolvePort {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }
}

impl Default for MockOpenEvolvePort {
    fn default() -> Self {
        Self { top_k: 3 }
    }
}

impl OpenEvolvePort for MockOpenEvolvePort {
    fn propose(&self, _ctx: &SleepContext, report: &SleepReport) -> Vec<EvolutionProposal> {
        let mut proposals = report.proposals.clone();
        proposals.sort_by(|left, right| {
            right
                .expected_gain
                .cmp(&left.expected_gain)
                .then_with(|| left.risk.cmp(&right.risk))
                .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
        });
        proposals.truncate(self.top_k.min(proposals.len()));
        proposals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_proposal(id_byte: u8, expected_gain: i16, risk: u16) -> EvolutionProposal {
        EvolutionProposal {
            id: Digest32::new([id_byte; 32]),
            title: format!("proposal-{id_byte}"),
            rationale: "mock".to_string(),
            expected_gain,
            risk,
        }
    }

    #[test]
    fn mock_openevolve_sorts_and_limits() {
        let ctx = SleepContext {
            cycle_id: 1,
            fixed_seed: [1u8; 32],
            integration_score: 5,
        };
        let report = SleepReport {
            proposals: vec![
                sample_proposal(1, 3, 7),
                sample_proposal(2, 5, 3),
                sample_proposal(3, 5, 9),
                sample_proposal(4, 1, 1),
            ],
            metrics: Vec::new(),
        };
        let port = MockOpenEvolvePort::new(2);

        let proposals = port.propose(&ctx, &report);

        assert_eq!(proposals.len(), 2);
        assert_eq!(proposals[0].id, Digest32::new([2u8; 32]));
        assert_eq!(proposals[1].id, Digest32::new([3u8; 32]));
    }
}
