#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

use crate::NsrVerdict;

const NSR_FACTS_ROOT_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.facts.root";
const NSR_INPUTS_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.inputs";
const NSR_TRACE_COMMIT_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.trace.commit";
const NSR_TRACE_ROOT_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.trace.root";
const NSR_OUTPUTS_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.outputs";
const NSR_MOCK_SOLVER_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.mock.smt";

const FACTS_MAX: usize = 64;
const RULES_MAX: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fact {
    Phi(u16),
    Plv(u16),
    Drift(u16),
    Surprise(u16),
    Risk(u16),
    OnnPhase {
        gamma_bucket: u8,
    },
    OnnLocked {
        global_plv: u16,
        lock_window_buckets: u8,
    },
    SpikeSummary {
        total: u16,
        threat: u16,
        thought_only: u16,
    },
    SpikeMaxIntensity(u16),
    JepaSurprise(u16),
    PhaseLocked,
    HighSurprise,
    SpikeThreatPresent,
    ThoughtOnlyPresent,
    TcfSleepActive,
    TcfReplayActive,
    CdeEdge {
        edge_commit: Digest32,
        score: u16,
    },
    CdeCounterfactualOk {
        commit: Digest32,
    },
    SsmNovelty(u16),
    SsmSalience(u16),
    NcdeEnergy(u16),
    IitHints {
        tighten_sync: bool,
        damp_output: bool,
        damp_learning: bool,
        request_replay: bool,
    },
    PolicyCommit {
        commit: Digest32,
    },
    ToolCallRequested,
    ThoughtOnlyRequested,
    Unknown(u16, Digest32),
}

impl Fact {
    fn discriminant(&self) -> u16 {
        match self {
            Self::Phi(_) => 1,
            Self::Plv(_) => 2,
            Self::Drift(_) => 3,
            Self::Surprise(_) => 4,
            Self::Risk(_) => 5,
            Self::OnnPhase { .. } => 6,
            Self::OnnLocked { .. } => 7,
            Self::SpikeSummary { .. } => 8,
            Self::SpikeMaxIntensity(_) => 9,
            Self::JepaSurprise(_) => 10,
            Self::PhaseLocked => 11,
            Self::HighSurprise => 12,
            Self::SpikeThreatPresent => 13,
            Self::ThoughtOnlyPresent => 14,
            Self::TcfSleepActive => 15,
            Self::TcfReplayActive => 16,
            Self::CdeEdge { .. } => 17,
            Self::CdeCounterfactualOk { .. } => 18,
            Self::SsmNovelty(_) => 19,
            Self::SsmSalience(_) => 20,
            Self::NcdeEnergy(_) => 21,
            Self::IitHints { .. } => 22,
            Self::PolicyCommit { .. } => 23,
            Self::ToolCallRequested => 24,
            Self::ThoughtOnlyRequested => 25,
            Self::Unknown(code, _) => *code,
        }
    }

    fn payload_bytes(&self) -> Vec<u8> {
        match self {
            Self::Phi(value)
            | Self::Plv(value)
            | Self::Drift(value)
            | Self::Surprise(value)
            | Self::Risk(value)
            | Self::SsmNovelty(value)
            | Self::SsmSalience(value)
            | Self::NcdeEnergy(value) => value.to_be_bytes().to_vec(),
            Self::OnnPhase { gamma_bucket } => vec![*gamma_bucket],
            Self::OnnLocked {
                global_plv,
                lock_window_buckets,
            } => {
                let mut bytes = Vec::with_capacity(3);
                bytes.extend_from_slice(&global_plv.to_be_bytes());
                bytes.push(*lock_window_buckets);
                bytes
            }
            Self::SpikeSummary {
                total,
                threat,
                thought_only,
            } => {
                let mut bytes = Vec::with_capacity(6);
                bytes.extend_from_slice(&total.to_be_bytes());
                bytes.extend_from_slice(&threat.to_be_bytes());
                bytes.extend_from_slice(&thought_only.to_be_bytes());
                bytes
            }
            Self::SpikeMaxIntensity(value) | Self::JepaSurprise(value) => {
                value.to_be_bytes().to_vec()
            }
            Self::TcfSleepActive
            | Self::TcfReplayActive
            | Self::ToolCallRequested
            | Self::ThoughtOnlyRequested
            | Self::PhaseLocked
            | Self::HighSurprise
            | Self::SpikeThreatPresent
            | Self::ThoughtOnlyPresent => vec![1],
            Self::CdeEdge { edge_commit, score } => {
                let mut bytes = Vec::with_capacity(34);
                bytes.extend_from_slice(edge_commit.as_bytes());
                bytes.extend_from_slice(&score.to_be_bytes());
                bytes
            }
            Self::CdeCounterfactualOk { commit } | Self::PolicyCommit { commit } => {
                commit.as_bytes().to_vec()
            }
            Self::IitHints {
                tighten_sync,
                damp_output,
                damp_learning,
                request_replay,
            } => vec![
                *tighten_sync as u8,
                *damp_output as u8,
                *damp_learning as u8,
                *request_replay as u8,
            ],
            Self::Unknown(_, commit) => commit.as_bytes().to_vec(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleId {
    Safety,
    Coherence,
    Causality,
    Output,
    Learning,
    RsaApply,
    Unknown(u16),
}

impl RuleId {
    fn code(self) -> u16 {
        match self {
            Self::Safety => 1,
            Self::Coherence => 2,
            Self::Causality => 3,
            Self::Output => 4,
            Self::Learning => 5,
            Self::RsaApply => 6,
            Self::Unknown(code) => code,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrTrace {
    pub cycle_id: u64,
    pub facts_root: Digest32,
    pub applied_rules: Vec<RuleId>,
    pub verdict: NsrVerdict,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrInputs {
    pub cycle_id: u64,
    pub phase_bus_commit: Digest32,
    pub policy_commit: Digest32,
    pub facts: Vec<Fact>,
    pub commit: Digest32,
}

impl NsrInputs {
    pub fn new(
        cycle_id: u64,
        phase_bus_commit: Digest32,
        policy_commit: Digest32,
        mut facts: Vec<Fact>,
    ) -> Self {
        if facts.len() > FACTS_MAX {
            facts.truncate(FACTS_MAX);
        }
        let facts_root = digest_facts_root(&facts);
        let commit = digest_inputs(cycle_id, phase_bus_commit, policy_commit, facts_root);
        Self {
            cycle_id,
            phase_bus_commit,
            policy_commit,
            facts,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrOutputs {
    pub cycle_id: u64,
    pub verdict: NsrVerdict,
    pub trace_root: Digest32,
    pub commit: Digest32,
}

pub trait NsrSolver {
    fn solve(&self, facts: &[Fact]) -> (NsrVerdict, Vec<RuleId>);
}

#[derive(Clone, Debug)]
pub struct MockSmtSolver {
    pub commit: Digest32,
}

impl MockSmtSolver {
    pub fn new() -> Self {
        let mut hasher = Hasher::new();
        hasher.update(NSR_MOCK_SOLVER_DOMAIN);
        let commit = Digest32::new(*hasher.finalize().as_bytes());
        Self { commit }
    }
}

impl Default for MockSmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl NsrSolver for MockSmtSolver {
    fn solve(&self, facts: &[Fact]) -> (NsrVerdict, Vec<RuleId>) {
        let mut phi = 0u16;
        let mut plv = 0u16;
        let mut drift = 0u16;
        let mut surprise = 0u16;
        let mut risk = 0u16;
        let mut phase_locked = false;
        let mut high_surprise = false;
        let mut spike_threat_present = false;
        let mut thought_only_present = false;
        let mut sleep_active = false;
        let mut replay_active = false;
        let mut tighten_sync = false;
        let mut damp_output = false;
        let mut damp_learning = false;
        let mut request_replay = false;
        let mut has_cde_edge = false;
        let mut has_cf_ok = false;
        let mut tool_req = false;
        let mut thought_only_requested = false;

        for fact in facts {
            match *fact {
                Fact::Phi(value) => phi = value,
                Fact::Plv(value) => plv = value,
                Fact::Drift(value) => drift = value,
                Fact::Surprise(value) => {
                    surprise = value;
                    if value >= 7_000 {
                        high_surprise = true;
                    }
                }
                Fact::Risk(value) => risk = value,
                Fact::TcfSleepActive => sleep_active = true,
                Fact::TcfReplayActive => replay_active = true,
                Fact::CdeEdge { .. } => has_cde_edge = true,
                Fact::CdeCounterfactualOk { .. } => has_cf_ok = true,
                Fact::IitHints {
                    tighten_sync: ts,
                    damp_output: doff,
                    damp_learning: dlearn,
                    request_replay: replay,
                } => {
                    tighten_sync = tighten_sync || ts;
                    damp_output = damp_output || doff;
                    damp_learning = damp_learning || dlearn;
                    request_replay = request_replay || replay;
                }
                Fact::OnnLocked {
                    global_plv,
                    lock_window_buckets: _,
                } => {
                    plv = global_plv;
                    if global_plv >= 7_000 {
                        phase_locked = true;
                    }
                }
                Fact::SpikeSummary {
                    total: _,
                    threat,
                    thought_only,
                } => {
                    if threat > 0 {
                        spike_threat_present = true;
                    }
                    if thought_only > 0 {
                        thought_only_present = true;
                    }
                }
                Fact::PhaseLocked => phase_locked = true,
                Fact::HighSurprise => high_surprise = true,
                Fact::SpikeThreatPresent => spike_threat_present = true,
                Fact::ThoughtOnlyPresent => thought_only_present = true,
                Fact::ToolCallRequested => tool_req = true,
                Fact::ThoughtOnlyRequested => thought_only_requested = true,
                _ => {}
            }
        }

        let mut verdict = NsrVerdict::Allow;
        let mut applied_rules = Vec::new();

        if risk >= 7500 && phi <= 2500 {
            applied_rules.push(RuleId::Safety);
            verdict = NsrVerdict::Deny;
        }

        if tool_req && (risk >= 6000 || drift >= 7000) {
            applied_rules.push(RuleId::Safety);
            verdict = NsrVerdict::Deny;
        }

        let outward_intent = tool_req;
        if thought_only_present && outward_intent {
            applied_rules.push(RuleId::Output);
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Restrict;
            }
        }

        if high_surprise && !phase_locked {
            applied_rules.push(RuleId::Coherence);
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Restrict;
            }
        }

        if tighten_sync && plv <= 3000 {
            applied_rules.push(RuleId::Coherence);
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Restrict;
            }
        }

        if drift >= 8000 && plv <= 4000 {
            applied_rules.push(RuleId::Coherence);
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Restrict;
            }
        }

        if sleep_active {
            applied_rules.push(RuleId::Coherence);
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Restrict;
            }
        }

        if surprise >= 7000 && !has_cde_edge {
            applied_rules.push(RuleId::Causality);
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Restrict;
            }
        }

        if has_cde_edge && !has_cf_ok && surprise >= 8000 {
            applied_rules.push(RuleId::Causality);
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Restrict;
            }
        }

        if damp_output {
            applied_rules.push(RuleId::Output);
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Restrict;
            }
        }

        if thought_only_requested {
            applied_rules.push(RuleId::Output);
        }

        let _ = (
            replay_active,
            damp_learning,
            request_replay,
            spike_threat_present,
        );

        if applied_rules.len() > RULES_MAX {
            applied_rules.truncate(RULES_MAX);
        }

        (verdict, applied_rules)
    }
}

pub struct NsrCore {
    solver: Box<dyn NsrSolver + Send + Sync>,
}

impl NsrCore {
    pub fn new(solver: Box<dyn NsrSolver + Send + Sync>) -> Self {
        Self { solver }
    }

    pub fn tick(&self, inputs: &NsrInputs) -> NsrOutputs {
        self.tick_with_trace(inputs).0
    }

    pub fn tick_with_trace(&self, inputs: &NsrInputs) -> (NsrOutputs, NsrTrace) {
        let facts_root = digest_facts_root(&inputs.facts);
        let (verdict, mut applied_rules) = self.solver.solve(&inputs.facts);
        if applied_rules.len() > RULES_MAX {
            applied_rules.truncate(RULES_MAX);
        }
        let trace_commit =
            digest_trace_commit(inputs.cycle_id, facts_root, verdict, &applied_rules);
        let trace_root = digest_trace_root(trace_commit);
        let trace = NsrTrace {
            cycle_id: inputs.cycle_id,
            facts_root,
            applied_rules,
            verdict,
            commit: trace_commit,
        };
        let outputs_commit = digest_outputs(inputs.cycle_id, verdict, trace_root);
        let outputs = NsrOutputs {
            cycle_id: inputs.cycle_id,
            verdict,
            trace_root,
            commit: outputs_commit,
        };
        (outputs, trace)
    }
}

impl Default for NsrCore {
    fn default() -> Self {
        Self::new(Box::new(MockSmtSolver::default()))
    }
}

fn digest_facts_root(facts: &[Fact]) -> Digest32 {
    let mut entries = facts
        .iter()
        .map(|fact| (fact.discriminant(), fact.payload_bytes()))
        .collect::<Vec<_>>();
    entries.sort_by(|(left_disc, left_bytes), (right_disc, right_bytes)| {
        left_disc
            .cmp(right_disc)
            .then_with(|| left_bytes.cmp(right_bytes))
    });
    let mut hasher = Hasher::new();
    hasher.update(NSR_FACTS_ROOT_DOMAIN);
    hasher.update(
        &u16::try_from(entries.len())
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    for (disc, bytes) in entries {
        hasher.update(&disc.to_be_bytes());
        hasher.update(&u16::try_from(bytes.len()).unwrap_or(u16::MAX).to_be_bytes());
        hasher.update(&bytes);
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_inputs(
    cycle_id: u64,
    phase_bus_commit: Digest32,
    policy_commit: Digest32,
    facts_root: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_INPUTS_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(policy_commit.as_bytes());
    hasher.update(facts_root.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_trace_commit(
    cycle_id: u64,
    facts_root: Digest32,
    verdict: NsrVerdict,
    applied_rules: &[RuleId],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_TRACE_COMMIT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(facts_root.as_bytes());
    hasher.update(&[verdict.as_u8()]);
    hasher.update(
        &u16::try_from(applied_rules.len())
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    for rule in applied_rules {
        hasher.update(&rule.code().to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_trace_root(trace_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_TRACE_ROOT_DOMAIN);
    hasher.update(trace_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_outputs(cycle_id: u64, verdict: NsrVerdict, trace_root: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_OUTPUTS_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&[verdict.as_u8()]);
    hasher.update(trace_root.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs(facts: Vec<Fact>) -> NsrInputs {
        NsrInputs::new(
            42,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            facts,
        )
    }

    #[test]
    fn mock_solver_denies_high_risk_low_phi() {
        let solver = MockSmtSolver::default();
        let facts = vec![Fact::Risk(7600), Fact::Phi(2400)];
        let (verdict, rules) = solver.solve(&facts);
        assert_eq!(verdict, NsrVerdict::Deny);
        assert!(rules.contains(&RuleId::Safety));
    }

    #[test]
    fn mock_solver_restricts_low_plv_with_tighten_sync() {
        let solver = MockSmtSolver::default();
        let facts = vec![
            Fact::Plv(2500),
            Fact::IitHints {
                tighten_sync: true,
                damp_output: false,
                damp_learning: false,
                request_replay: false,
            },
        ];
        let (verdict, rules) = solver.solve(&facts);
        assert_eq!(verdict, NsrVerdict::Restrict);
        assert!(rules.contains(&RuleId::Coherence));
    }

    #[test]
    fn mock_solver_restricts_surprise_without_cde_edge() {
        let solver = MockSmtSolver::default();
        let facts = vec![Fact::Surprise(7200)];
        let (verdict, rules) = solver.solve(&facts);
        assert_eq!(verdict, NsrVerdict::Restrict);
        assert!(rules.contains(&RuleId::Causality));
    }

    #[test]
    fn mock_solver_restricts_high_surprise_when_unlocked() {
        let solver = MockSmtSolver::default();
        let facts = vec![
            Fact::Surprise(7200),
            Fact::OnnLocked {
                global_plv: 2_000,
                lock_window_buckets: 1,
            },
        ];
        let (verdict, rules) = solver.solve(&facts);
        assert_eq!(verdict, NsrVerdict::Restrict);
        assert!(rules.contains(&RuleId::Coherence));
    }

    #[test]
    fn mock_solver_restricts_thought_only_with_outward_intent() {
        let solver = MockSmtSolver::default();
        let facts = vec![Fact::ThoughtOnlyPresent, Fact::ToolCallRequested];
        let (verdict, rules) = solver.solve(&facts);
        assert_eq!(verdict, NsrVerdict::Restrict);
        assert!(rules.contains(&RuleId::Output));
    }

    #[test]
    fn mock_solver_restricts_when_damp_output() {
        let solver = MockSmtSolver::default();
        let facts = vec![Fact::IitHints {
            tighten_sync: false,
            damp_output: true,
            damp_learning: false,
            request_replay: false,
        }];
        let (verdict, rules) = solver.solve(&facts);
        assert_eq!(verdict, NsrVerdict::Restrict);
        assert!(rules.contains(&RuleId::Output));
    }

    #[test]
    fn mock_solver_allows_thought_only() {
        let solver = MockSmtSolver::default();
        let facts = vec![Fact::ThoughtOnlyRequested];
        let (verdict, rules) = solver.solve(&facts);
        assert_eq!(verdict, NsrVerdict::Allow);
        assert!(rules.contains(&RuleId::Output));
    }

    #[test]
    fn facts_root_is_order_invariant() {
        let solver = MockSmtSolver::default();
        let facts_a = vec![Fact::Risk(1234), Fact::Phi(4321)];
        let facts_b = vec![Fact::Phi(4321), Fact::Risk(1234)];
        let inputs_a = base_inputs(facts_a);
        let inputs_b = base_inputs(facts_b);

        let (out_a, trace_a) = NsrCore::new(Box::new(solver.clone())).tick_with_trace(&inputs_a);
        let (out_b, trace_b) = NsrCore::new(Box::new(solver)).tick_with_trace(&inputs_b);

        assert_eq!(trace_a.facts_root, trace_b.facts_root);
        assert_eq!(out_a.trace_root, out_b.trace_root);
    }
}
