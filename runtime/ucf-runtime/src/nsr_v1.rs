use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use ucf_policy::candidate::DecisionCandidate;
use ucf_types::UQ0_16;

pub const MAX_FACTS: usize = 128;
pub const MAX_DERIVED_FACTS: usize = 256;
pub const MAX_REASONS: usize = 8;
pub const MAX_EVAL_STEPS: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NsrStatus {
    Ok = 0,
    BudgetExceeded = 1,
    ParseErrorFallback = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NsrReason {
    pub term_id: u16,
    pub contrib_q: UQ0_16,
    pub rule_id: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NsrOutput {
    pub status: NsrStatus,
    pub nsr_risk_q: UQ0_16,
    pub top_reasons: Vec<NsrReason>,
    pub rules_digest: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct NsrEngineV1 {
    rules: Vec<Rule>,
    rules_digest: [u8; 32],
    parse_ok: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateKindFact {
    Tool,
    Json,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextRiskLevel {
    Low,
    Med,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NsrEvalInput {
    pub candidate_kind: CandidateKindFact,
    pub tool_class: Option<u8>,
    pub context_risk_level: ContextRiskLevel,
    pub emergency_active: bool,
    pub budget_low: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Atom {
    pred: String,
    args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Rule {
    id: u16,
    head: Atom,
    body: Vec<Literal>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Literal {
    Pos(Atom),
    Neg(Atom),
}

impl NsrEngineV1 {
    pub fn from_policy_file(path: &Path) -> Self {
        let bytes = fs::read(path).unwrap_or_default();
        let rules_digest = Sha256::digest(&bytes).into();
        let text = String::from_utf8_lossy(&bytes);
        match parse_rules(&text) {
            Ok(rules) => Self {
                rules,
                rules_digest,
                parse_ok: true,
            },
            Err(_) => Self {
                rules: Vec::new(),
                rules_digest,
                parse_ok: false,
            },
        }
    }

    pub fn evaluate(&self, input: NsrEvalInput) -> NsrOutput {
        if !self.parse_ok {
            return conservative_fallback(input.candidate_kind, self.rules_digest);
        }
        let mut facts = BTreeSet::new();
        seed_facts(input, &mut facts);
        if facts.len() > MAX_FACTS {
            return budget_exceeded(self.rules_digest);
        }

        let mut derived = 0usize;
        let mut steps = 0usize;
        let mut changed = true;
        while changed {
            changed = false;
            for rule in &self.rules {
                steps += 1;
                if steps > MAX_EVAL_STEPS {
                    return budget_exceeded(self.rules_digest);
                }
                if body_holds(&facts, &rule.body) && !facts.contains(&rule.head) {
                    facts.insert(rule.head.clone());
                    derived += 1;
                    changed = true;
                    if facts.len() > MAX_FACTS || derived > MAX_DERIVED_FACTS {
                        return budget_exceeded(self.rules_digest);
                    }
                }
            }
        }

        let mut terms = BTreeMap::<u16, (UQ0_16, u16)>::new();
        for atom in &facts {
            if atom.pred == "risk_term" && atom.args.len() == 2 {
                if let (Ok(term_id), Ok(weight)) =
                    (atom.args[0].parse::<u16>(), atom.args[1].parse::<u16>())
                {
                    terms
                        .entry(term_id)
                        .or_insert((UQ0_16::from_raw(weight), 0));
                }
            }
        }
        for rule in &self.rules {
            if rule.head.pred == "risk_term" && rule.head.args.len() == 2 {
                if let Ok(term_id) = rule.head.args[0].parse::<u16>() {
                    if let Some(entry) = terms.get_mut(&term_id) {
                        entry.1 = rule.id;
                    }
                }
            }
        }

        let mut reasons = terms
            .into_iter()
            .map(|(term_id, (contrib_q, rule_id))| NsrReason {
                term_id,
                contrib_q,
                rule_id,
            })
            .collect::<Vec<_>>();
        reasons.sort_by(|a, b| {
            b.contrib_q
                .raw()
                .cmp(&a.contrib_q.raw())
                .then_with(|| a.term_id.cmp(&b.term_id))
        });
        reasons.truncate(MAX_REASONS);

        let risk_sum = reasons.iter().fold(0u32, |acc, r| {
            acc.saturating_add(u32::from(r.contrib_q.raw()))
        });
        NsrOutput {
            status: NsrStatus::Ok,
            nsr_risk_q: UQ0_16::from_raw(risk_sum.min(u32::from(u16::MAX)) as u16),
            top_reasons: reasons,
            rules_digest: self.rules_digest,
        }
    }
}

fn conservative_fallback(kind: CandidateKindFact, rules_digest: [u8; 32]) -> NsrOutput {
    let risk = if matches!(kind, CandidateKindFact::Tool) {
        UQ0_16::ONE
    } else {
        UQ0_16::from_raw(40_000)
    };
    NsrOutput {
        status: NsrStatus::ParseErrorFallback,
        nsr_risk_q: risk,
        top_reasons: vec![NsrReason {
            term_id: 0,
            contrib_q: risk,
            rule_id: 0,
        }],
        rules_digest,
    }
}

fn budget_exceeded(rules_digest: [u8; 32]) -> NsrOutput {
    NsrOutput {
        status: NsrStatus::BudgetExceeded,
        nsr_risk_q: UQ0_16::ONE,
        top_reasons: vec![NsrReason {
            term_id: 65_535,
            contrib_q: UQ0_16::ONE,
            rule_id: 0,
        }],
        rules_digest,
    }
}

fn seed_facts(input: NsrEvalInput, facts: &mut BTreeSet<Atom>) {
    let kind = match input.candidate_kind {
        CandidateKindFact::Tool => "tool",
        CandidateKindFact::Json => "json",
        CandidateKindFact::Text => "text",
    };
    facts.insert(atom("candidate_kind", &[kind]));
    if let Some(tool_class) = input.tool_class {
        let class = tool_class.to_string();
        facts.insert(atom("tool_class", &[&class]));
    }
    let level = match input.context_risk_level {
        ContextRiskLevel::Low => "low",
        ContextRiskLevel::Med => "med",
        ContextRiskLevel::High => "high",
    };
    facts.insert(atom("context_risk_level", &[level]));
    if input.emergency_active {
        facts.insert(atom("emergency", &["active"]));
    }
    if input.budget_low {
        facts.insert(atom("budget_low", &["true"]));
    }
}

fn body_holds(facts: &BTreeSet<Atom>, body: &[Literal]) -> bool {
    body.iter().all(|lit| match lit {
        Literal::Pos(atom) => facts.contains(atom),
        Literal::Neg(atom) => !facts.contains(atom),
    })
}

fn atom(pred: &str, args: &[&str]) -> Atom {
    Atom {
        pred: pred.to_string(),
        args: args.iter().map(|v| (*v).to_string()).collect(),
    }
}

fn parse_rules(input: &str) -> Result<Vec<Rule>, String> {
    let mut rules = Vec::new();
    for (line_no, raw) in input.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let line = line
            .strip_suffix('.')
            .ok_or_else(|| format!("line {} missing '.'", line_no + 1))?;
        if line.starts_with("fact(") {
            continue;
        }
        let (head_s, body_s) = line
            .split_once(":-")
            .ok_or_else(|| format!("line {} missing ':-'", line_no + 1))?;
        let head = parse_atom(head_s.trim())?;
        validate_atom(&head)?;
        let mut body = Vec::new();
        for part in body_s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some(rest) = part.strip_prefix("not ") {
                let atom = parse_atom(rest.trim())?;
                validate_atom(&atom)?;
                body.push(Literal::Neg(atom));
            } else {
                let atom = parse_atom(part)?;
                validate_atom(&atom)?;
                body.push(Literal::Pos(atom));
            }
        }
        rules.push(Rule {
            id: (rules.len() + 1) as u16,
            head,
            body,
        });
    }
    rules.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(rules)
}

fn parse_atom(s: &str) -> Result<Atom, String> {
    let open = s.find('(').ok_or_else(|| "missing '('".to_string())?;
    let close = s.rfind(')').ok_or_else(|| "missing ')'".to_string())?;
    let pred = s[..open].trim();
    let args_s = &s[open + 1..close];
    let args = if args_s.trim().is_empty() {
        Vec::new()
    } else {
        args_s
            .split(',')
            .map(|a| a.trim().to_string())
            .collect::<Vec<_>>()
    };
    Ok(Atom {
        pred: pred.to_string(),
        args,
    })
}

fn validate_atom(atom: &Atom) -> Result<(), String> {
    match atom.pred.as_str() {
        "candidate_kind" => {
            if atom.args.len() != 1 || !matches!(atom.args[0].as_str(), "tool" | "json" | "text") {
                return Err("invalid candidate_kind".to_string());
            }
        }
        "tool_class" => {
            if atom.args.len() != 1 || atom.args[0].parse::<u8>().is_err() {
                return Err("invalid tool_class".to_string());
            }
        }
        "context_risk_level" => {
            if atom.args.len() != 1 || !matches!(atom.args[0].as_str(), "low" | "med" | "high") {
                return Err("invalid context_risk_level".to_string());
            }
        }
        "emergency" => {
            if atom.args.as_slice() != ["active"] {
                return Err("invalid emergency".to_string());
            }
        }
        "budget_low" => {
            if atom.args.as_slice() != ["true"] {
                return Err("invalid budget_low".to_string());
            }
        }
        "risk_term" => {
            if atom.args.len() != 2
                || atom.args[0].parse::<u16>().is_err()
                || atom.args[1].parse::<u16>().is_err()
            {
                return Err("invalid risk_term".to_string());
            }
        }
        _ => return Err("predicate not allowed".to_string()),
    }
    Ok(())
}

pub fn eval_input_from_candidate(
    candidate: &DecisionCandidate,
    risk: f32,
    emergency_active: bool,
    governor_tier: u8,
) -> NsrEvalInput {
    let candidate_kind = if !candidate.tool_intents.is_empty() {
        CandidateKindFact::Tool
    } else if matches!(
        candidate.output_class,
        ucf_policy::candidate::OutputClass::ExternalIo
            | ucf_policy::candidate::OutputClass::Sensitive
    ) {
        CandidateKindFact::Json
    } else {
        CandidateKindFact::Text
    };
    let context_risk_level = if risk < 0.33 {
        ContextRiskLevel::Low
    } else if risk < 0.66 {
        ContextRiskLevel::Med
    } else {
        ContextRiskLevel::High
    };
    NsrEvalInput {
        candidate_kind,
        tool_class: candidate
            .tool_intents
            .first()
            .map(|t| t.kind.as_tag().bytes().next().unwrap_or(0)),
        context_risk_level,
        emergency_active,
        budget_low: governor_tier >= 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_unknown_predicate() {
        let rules = "risk_term(1,12000) :- evil(x).";
        assert!(parse_rules(rules).is_err());
    }

    #[test]
    fn deterministic_output_for_same_input() {
        let path = std::path::Path::new("policies/bundle_v1/nsr_rules_v1.dl");
        let engine = NsrEngineV1::from_policy_file(path);
        let input = NsrEvalInput {
            candidate_kind: CandidateKindFact::Tool,
            tool_class: Some(1),
            context_risk_level: ContextRiskLevel::High,
            emergency_active: false,
            budget_low: false,
        };
        let a = engine.evaluate(input);
        let b = engine.evaluate(input);
        assert_eq!(a, b);
    }

    #[test]
    fn parse_error_fallback_is_conservative_for_tool() {
        let engine = NsrEngineV1 {
            rules: Vec::new(),
            rules_digest: [7; 32],
            parse_ok: false,
        };
        let out = engine.evaluate(NsrEvalInput {
            candidate_kind: CandidateKindFact::Tool,
            tool_class: None,
            context_risk_level: ContextRiskLevel::Low,
            emergency_active: false,
            budget_low: false,
        });
        assert_eq!(out.status, NsrStatus::ParseErrorFallback);
        assert_eq!(out.nsr_risk_q, UQ0_16::ONE);
    }
}
