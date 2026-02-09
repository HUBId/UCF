#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

use crate::NsrVerdict;

const NSR_FACTS_ROOT_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.facts.root";
const NSR_INPUTS_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.inputs";
const NSR_FACTS_COMMIT_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.facts.commit";
const NSR_TRACE_COMMIT_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.trace.commit";
const NSR_TRACE_ROOT_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.trace.root";
const NSR_RULE_HIT_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.rule.hit";
const NSR_OUTPUTS_DOMAIN: &[u8] = b"ucf.nsr.v1.notar.outputs";

const FACTS_MAX: usize = 64;
const RULES_MAX: usize = 64;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuleId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleSeverity {
    Info = 0,
    Warn = 1,
    Block = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rule {
    pub id: RuleId,
    pub name: &'static str,
    pub severity: RuleSeverity,
    pub priority: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleHit {
    pub cycle_id: u64,
    pub id: RuleId,
    pub severity: RuleSeverity,
    pub reason: u16,
    pub aux: [Digest32; 2],
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrTrace {
    pub cycle_id: u64,
    pub hits: Vec<RuleHit>,
    pub trace_root: Digest32,
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
    pub nsr_trace_root: Digest32,
    pub commit: Digest32,
}

pub trait LogicHook {
    fn prove(&self, cycle_id: u64, facts_commit: Digest32) -> Vec<RuleHit>;
}

#[derive(Clone, Debug)]
pub struct NoopLogicHook;

impl LogicHook for NoopLogicHook {
    fn prove(&self, _cycle_id: u64, _facts_commit: Digest32) -> Vec<RuleHit> {
        Vec::new()
    }
}

const REASON_RISK_OVER_CAP: u16 = 100;
const REASON_POLICY_DENY: u16 = 110;
const REASON_THOUGHT_ONLY_LEAK: u16 = 120;
const REASON_HIGH_SURPRISE_LOW_COHERENCE: u16 = 130;
const REASON_TIGHTEN_SYNC_LOW_PLV: u16 = 140;
const REASON_DRIFT_OVER_CAP: u16 = 150;
const REASON_SLEEP_ACTIVE: u16 = 160;
const REASON_SURPRISE_NO_CDE: u16 = 170;
const REASON_SURPRISE_NO_COUNTERFACTUAL: u16 = 180;
const REASON_DAMP_OUTPUT: u16 = 190;
const REASON_THOUGHT_ONLY_REQUESTED: u16 = 200;

const RULES: &[Rule] = &[
    Rule {
        id: RuleId(1),
        name: "RiskOverCap",
        severity: RuleSeverity::Block,
        priority: 10,
    },
    Rule {
        id: RuleId(2),
        name: "PolicyDeny",
        severity: RuleSeverity::Block,
        priority: 20,
    },
    Rule {
        id: RuleId(3),
        name: "ThoughtOnlyLeakGuard",
        severity: RuleSeverity::Block,
        priority: 30,
    },
    Rule {
        id: RuleId(4),
        name: "HighSurpriseLowCoherence",
        severity: RuleSeverity::Block,
        priority: 40,
    },
    Rule {
        id: RuleId(5),
        name: "TightenSyncLowPlv",
        severity: RuleSeverity::Block,
        priority: 50,
    },
    Rule {
        id: RuleId(6),
        name: "DriftOverCap",
        severity: RuleSeverity::Block,
        priority: 60,
    },
    Rule {
        id: RuleId(7),
        name: "SleepActive",
        severity: RuleSeverity::Block,
        priority: 70,
    },
    Rule {
        id: RuleId(8),
        name: "SurpriseNoCdeEdge",
        severity: RuleSeverity::Block,
        priority: 80,
    },
    Rule {
        id: RuleId(9),
        name: "SurpriseNoCounterfactual",
        severity: RuleSeverity::Block,
        priority: 90,
    },
    Rule {
        id: RuleId(10),
        name: "DampOutput",
        severity: RuleSeverity::Block,
        priority: 100,
    },
    Rule {
        id: RuleId(11),
        name: "ThoughtOnlyRequested",
        severity: RuleSeverity::Info,
        priority: 110,
    },
];

pub fn rules() -> &'static [Rule] {
    RULES
}

#[derive(Default)]
struct FactsSummary {
    phi: u16,
    plv: u16,
    drift: u16,
    surprise: u16,
    risk: u16,
    phase_locked: bool,
    high_surprise: bool,
    thought_only_present: bool,
    sleep_active: bool,
    tighten_sync: bool,
    damp_output: bool,
    has_cde_edge: bool,
    has_cf_ok: bool,
    tool_req: bool,
    thought_only_requested: bool,
}

fn summarize_facts(facts: &[Fact]) -> FactsSummary {
    let mut summary = FactsSummary::default();
    for fact in facts {
        match *fact {
            Fact::Phi(value) => summary.phi = value,
            Fact::Plv(value) => summary.plv = value,
            Fact::Drift(value) => summary.drift = value,
            Fact::Surprise(value) => {
                summary.surprise = value;
                if value >= 7_000 {
                    summary.high_surprise = true;
                }
            }
            Fact::Risk(value) => summary.risk = value,
            Fact::TcfSleepActive => summary.sleep_active = true,
            Fact::CdeEdge { .. } => summary.has_cde_edge = true,
            Fact::CdeCounterfactualOk { .. } => summary.has_cf_ok = true,
            Fact::IitHints {
                tighten_sync: ts,
                damp_output: doff,
                damp_learning: _,
                request_replay: _,
            } => {
                summary.tighten_sync = summary.tighten_sync || ts;
                summary.damp_output = summary.damp_output || doff;
            }
            Fact::OnnLocked { global_plv, .. } => {
                summary.plv = global_plv;
                if global_plv >= 7_000 {
                    summary.phase_locked = true;
                }
            }
            Fact::SpikeSummary { thought_only, .. } => {
                if thought_only > 0 {
                    summary.thought_only_present = true;
                }
            }
            Fact::PhaseLocked => summary.phase_locked = true,
            Fact::HighSurprise => summary.high_surprise = true,
            Fact::ThoughtOnlyPresent => summary.thought_only_present = true,
            Fact::ToolCallRequested => summary.tool_req = true,
            Fact::ThoughtOnlyRequested => summary.thought_only_requested = true,
            _ => {}
        }
    }
    summary
}

fn evaluate_rules(cycle_id: u64, facts_commit: Digest32, summary: &FactsSummary) -> Vec<RuleHit> {
    let mut hits = Vec::new();
    for rule in RULES {
        match rule.id.0 {
            1 => {
                if summary.risk >= 7_500 && summary.phi <= 2_500 {
                    push_rule_hit(
                        &mut hits,
                        cycle_id,
                        facts_commit,
                        rule,
                        REASON_RISK_OVER_CAP,
                    );
                }
            }
            2 => {
                if summary.tool_req && (summary.risk >= 6_000 || summary.drift >= 7_000) {
                    push_rule_hit(&mut hits, cycle_id, facts_commit, rule, REASON_POLICY_DENY);
                }
            }
            3 => {
                if summary.thought_only_present && summary.tool_req {
                    push_rule_hit(
                        &mut hits,
                        cycle_id,
                        facts_commit,
                        rule,
                        REASON_THOUGHT_ONLY_LEAK,
                    );
                }
            }
            4 => {
                if summary.high_surprise && summary.plv <= 3_000 && !summary.phase_locked {
                    push_rule_hit(
                        &mut hits,
                        cycle_id,
                        facts_commit,
                        rule,
                        REASON_HIGH_SURPRISE_LOW_COHERENCE,
                    );
                }
            }
            5 => {
                if summary.tighten_sync && summary.plv <= 3_000 {
                    push_rule_hit(
                        &mut hits,
                        cycle_id,
                        facts_commit,
                        rule,
                        REASON_TIGHTEN_SYNC_LOW_PLV,
                    );
                }
            }
            6 => {
                if summary.drift >= 8_000 && summary.plv <= 4_000 {
                    push_rule_hit(
                        &mut hits,
                        cycle_id,
                        facts_commit,
                        rule,
                        REASON_DRIFT_OVER_CAP,
                    );
                }
            }
            7 => {
                if summary.sleep_active {
                    push_rule_hit(&mut hits, cycle_id, facts_commit, rule, REASON_SLEEP_ACTIVE);
                }
            }
            8 => {
                if summary.surprise >= 7_000 && !summary.has_cde_edge {
                    push_rule_hit(
                        &mut hits,
                        cycle_id,
                        facts_commit,
                        rule,
                        REASON_SURPRISE_NO_CDE,
                    );
                }
            }
            9 => {
                if summary.has_cde_edge && !summary.has_cf_ok && summary.surprise >= 8_000 {
                    push_rule_hit(
                        &mut hits,
                        cycle_id,
                        facts_commit,
                        rule,
                        REASON_SURPRISE_NO_COUNTERFACTUAL,
                    );
                }
            }
            10 => {
                if summary.damp_output {
                    push_rule_hit(&mut hits, cycle_id, facts_commit, rule, REASON_DAMP_OUTPUT);
                }
            }
            11 => {
                if summary.thought_only_requested {
                    push_rule_hit(
                        &mut hits,
                        cycle_id,
                        facts_commit,
                        rule,
                        REASON_THOUGHT_ONLY_REQUESTED,
                    );
                }
            }
            _ => {}
        }
    }
    hits
}

fn push_rule_hit(
    hits: &mut Vec<RuleHit>,
    cycle_id: u64,
    facts_commit: Digest32,
    rule: &Rule,
    reason: u16,
) {
    hits.push(build_rule_hit(
        cycle_id,
        rule.id,
        rule.severity,
        reason,
        [facts_commit, Digest32::new([0u8; 32])],
        facts_commit,
    ));
}

fn normalize_hook_hits(cycle_id: u64, facts_commit: Digest32, hits: &mut [RuleHit]) {
    for hit in hits {
        hit.cycle_id = cycle_id;
        hit.commit = digest_rule_hit(
            hit.cycle_id,
            hit.id,
            hit.severity,
            hit.reason,
            hit.aux,
            facts_commit,
        );
    }
}

fn verdict_from_hits(hits: &[RuleHit]) -> NsrVerdict {
    if hits.iter().any(|hit| hit.severity == RuleSeverity::Block) {
        return NsrVerdict::Restrict;
    }
    if hits.iter().any(|hit| hit.severity == RuleSeverity::Warn) {
        return NsrVerdict::Allow;
    }
    NsrVerdict::Allow
}

pub struct NsrCore {
    logic_hook: Box<dyn LogicHook + Send + Sync>,
}

impl NsrCore {
    pub fn new(logic_hook: Box<dyn LogicHook + Send + Sync>) -> Self {
        Self { logic_hook }
    }

    pub fn tick(&self, inputs: &NsrInputs) -> NsrOutputs {
        self.tick_with_trace(inputs).0
    }

    pub fn tick_with_trace(&self, inputs: &NsrInputs) -> (NsrOutputs, NsrTrace) {
        let facts_root = digest_facts_root(&inputs.facts);
        let facts_commit =
            digest_facts_commit(inputs.phase_bus_commit, inputs.policy_commit, facts_root);
        let summary = summarize_facts(&inputs.facts);
        let mut hits = evaluate_rules(inputs.cycle_id, facts_commit, &summary);

        let mut hook_hits = self.logic_hook.prove(inputs.cycle_id, facts_commit);
        normalize_hook_hits(inputs.cycle_id, facts_commit, &mut hook_hits);
        hook_hits.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.reason.cmp(&b.reason)));
        hits.extend(hook_hits);

        if hits.len() > RULES_MAX {
            hits.truncate(RULES_MAX);
        }

        let trace_root = digest_trace_root(&hits);
        let verdict = verdict_from_hits(&hits);
        let trace_commit = digest_trace_commit(inputs.cycle_id, facts_commit, trace_root, verdict);
        let trace = NsrTrace {
            cycle_id: inputs.cycle_id,
            hits,
            trace_root,
            commit: trace_commit,
        };
        let outputs_commit = digest_outputs(inputs.cycle_id, verdict, trace_root);
        let outputs = NsrOutputs {
            cycle_id: inputs.cycle_id,
            verdict,
            trace_root,
            nsr_trace_root: trace_root,
            commit: outputs_commit,
        };
        (outputs, trace)
    }
}

impl Default for NsrCore {
    fn default() -> Self {
        Self::new(Box::new(NoopLogicHook))
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

fn digest_facts_commit(
    phase_bus_commit: Digest32,
    policy_commit: Digest32,
    facts_root: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_FACTS_COMMIT_DOMAIN);
    hasher.update(phase_bus_commit.as_bytes());
    hasher.update(policy_commit.as_bytes());
    hasher.update(facts_root.as_bytes());
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
    facts_commit: Digest32,
    trace_root: Digest32,
    verdict: NsrVerdict,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_TRACE_COMMIT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(facts_commit.as_bytes());
    hasher.update(trace_root.as_bytes());
    hasher.update(&[verdict.as_u8()]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_trace_root(hits: &[RuleHit]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_TRACE_ROOT_DOMAIN);
    for hit in hits {
        hasher.update(hit.commit.as_bytes());
    }
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

fn digest_rule_hit(
    cycle_id: u64,
    id: RuleId,
    severity: RuleSeverity,
    reason: u16,
    aux: [Digest32; 2],
    facts_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_RULE_HIT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&id.0.to_be_bytes());
    hasher.update(&[severity as u8]);
    hasher.update(&reason.to_be_bytes());
    hasher.update(aux[0].as_bytes());
    hasher.update(aux[1].as_bytes());
    hasher.update(facts_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn build_rule_hit(
    cycle_id: u64,
    id: RuleId,
    severity: RuleSeverity,
    reason: u16,
    aux: [Digest32; 2],
    facts_commit: Digest32,
) -> RuleHit {
    let commit = digest_rule_hit(cycle_id, id, severity, reason, aux, facts_commit);
    RuleHit {
        cycle_id,
        id,
        severity,
        reason,
        aux,
        commit,
    }
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
    fn reasoning_trace_is_deterministic() {
        let core = NsrCore::default();
        let inputs = base_inputs(vec![Fact::Risk(7600), Fact::Phi(2400)]);

        let (out_a, trace_a) = core.tick_with_trace(&inputs);
        let (out_b, trace_b) = core.tick_with_trace(&inputs);

        assert_eq!(trace_a.trace_root, trace_b.trace_root);
        assert_eq!(out_a.trace_root, out_b.trace_root);
    }

    #[test]
    fn block_rule_triggers_on_high_surprise_low_plv() {
        let core = NsrCore::default();
        let inputs = base_inputs(vec![Fact::Surprise(7200), Fact::Plv(2000)]);

        let (out, trace) = core.tick_with_trace(&inputs);

        assert_eq!(out.verdict, NsrVerdict::Restrict);
        assert!(trace.hits.iter().any(|hit| hit.id == RuleId(4)));
    }

    #[test]
    fn logic_hook_block_flips_verdict() {
        #[derive(Default)]
        struct BlockingHook;

        impl LogicHook for BlockingHook {
            fn prove(&self, cycle_id: u64, _facts_commit: Digest32) -> Vec<RuleHit> {
                vec![RuleHit {
                    cycle_id,
                    id: RuleId(900),
                    severity: RuleSeverity::Block,
                    reason: 900,
                    aux: [Digest32::new([0u8; 32]); 2],
                    commit: Digest32::new([0u8; 32]),
                }]
            }
        }

        let core = NsrCore::new(Box::new(BlockingHook::default()));
        let inputs = base_inputs(vec![Fact::Phi(6000), Fact::Plv(6000)]);

        let (out, trace) = core.tick_with_trace(&inputs);

        assert_eq!(out.verdict, NsrVerdict::Restrict);
        assert!(trace.hits.iter().any(|hit| hit.id == RuleId(900)));
    }

    #[test]
    fn facts_root_is_order_invariant() {
        let facts_a = vec![Fact::Risk(1234), Fact::Phi(4321)];
        let facts_b = vec![Fact::Phi(4321), Fact::Risk(1234)];
        let inputs_a = base_inputs(facts_a);
        let inputs_b = base_inputs(facts_b);
        let core = NsrCore::default();

        let (out_a, trace_a) = core.tick_with_trace(&inputs_a);
        let (out_b, trace_b) = core.tick_with_trace(&inputs_b);

        assert_eq!(trace_a.trace_root, trace_b.trace_root);
        assert_eq!(out_a.trace_root, out_b.trace_root);
    }
}
