#[allow(clippy::module_name_repetitions)]
pub type RuleId = u16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Allow,
    Block,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckResult {
    pub verdict: Verdict,
    pub satisfied: u16,
    pub total: u16,
    pub blocked_by: Option<RuleId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Contains,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Field {
    Intent,
    Tool,
    Channel,
    Risk,
    Audience,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    pub id: RuleId,
    pub field: Field,
    pub op: Op,
    pub value: String,
    pub hard_block: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    pub intent: String,
    pub tool: String,
    pub channel: String,
    pub risk: String,
    pub audience: String,
}

#[derive(Clone, Debug, Default)]
pub struct NsrEngine {
    pub rules: Vec<Rule>,
}

impl NsrEngine {
    pub fn with_default_rules() -> Self {
        Self {
            rules: vec![
                Rule {
                    id: 1,
                    field: Field::Intent,
                    op: Op::Contains,
                    value: "harm".to_string(),
                    hard_block: true,
                },
                Rule {
                    id: 2,
                    field: Field::Risk,
                    op: Op::Eq,
                    value: "high".to_string(),
                    hard_block: true,
                },
                Rule {
                    id: 3,
                    field: Field::Channel,
                    op: Op::Eq,
                    value: "terminal".to_string(),
                    hard_block: false,
                },
                Rule {
                    id: 4,
                    field: Field::Tool,
                    op: Op::Contains,
                    value: "network".to_string(),
                    hard_block: true,
                },
                Rule {
                    id: 5,
                    field: Field::Audience,
                    op: Op::Eq,
                    value: "research".to_string(),
                    hard_block: false,
                },
            ],
        }
    }

    pub fn check(&self, claim: &Claim) -> CheckResult {
        let mut sorted_rules = self.rules.iter().collect::<Vec<_>>();
        sorted_rules.sort_by_key(|rule| rule.id);

        let total = sorted_rules.len() as u16;
        let mut satisfied = 0_u16;

        for rule in sorted_rules {
            let is_satisfied = evaluate_rule(rule, claim);
            if is_satisfied {
                satisfied = satisfied.saturating_add(1);
                if rule.hard_block {
                    return CheckResult {
                        verdict: Verdict::Block,
                        satisfied,
                        total,
                        blocked_by: Some(rule.id),
                    };
                }
            }
        }

        let verdict = if total > 0 && satisfied == total {
            Verdict::Allow
        } else {
            Verdict::Unknown
        };

        CheckResult {
            verdict,
            satisfied,
            total,
            blocked_by: None,
        }
    }
}

fn evaluate_rule(rule: &Rule, claim: &Claim) -> bool {
    let candidate = match rule.field {
        Field::Intent => &claim.intent,
        Field::Tool => &claim.tool,
        Field::Channel => &claim.channel,
        Field::Risk => &claim.risk,
        Field::Audience => &claim.audience,
    };

    match rule.op {
        Op::Eq => candidate == &rule.value,
        Op::Ne => candidate != &rule.value,
        Op::Contains => candidate.contains(&rule.value),
        Op::Lt => candidate < &rule.value,
        Op::Le => candidate <= &rule.value,
        Op::Gt => candidate > &rule.value,
        Op::Ge => candidate >= &rule.value,
    }
}

#[cfg(test)]
mod tests {
    use super::{Claim, NsrEngine, Verdict};

    fn base_claim() -> Claim {
        Claim {
            intent: "noop".to_string(),
            tool: "mock".to_string(),
            channel: "terminal".to_string(),
            risk: "low".to_string(),
            audience: "research".to_string(),
        }
    }

    #[test]
    fn high_risk_blocks_by_r2() {
        let engine = NsrEngine::with_default_rules();
        let mut claim = base_claim();
        claim.risk = "high".to_string();

        let result = engine.check(&claim);
        assert_eq!(result.verdict, Verdict::Block);
        assert_eq!(result.blocked_by, Some(2));
    }

    #[test]
    fn low_risk_terminal_research_is_unknown_deterministically() {
        let engine = NsrEngine::with_default_rules();
        let claim = base_claim();

        let result = engine.check(&claim);
        assert_eq!(result.verdict, Verdict::Unknown);
        assert_eq!(result.satisfied, 2);
        assert_eq!(result.total, 5);
        assert_eq!(result.blocked_by, None);
    }

    #[test]
    fn network_tool_blocks_by_r4() {
        let engine = NsrEngine::with_default_rules();
        let mut claim = base_claim();
        claim.tool = "mock-network-adapter".to_string();

        let result = engine.check(&claim);
        assert_eq!(result.verdict, Verdict::Block);
        assert_eq!(result.blocked_by, Some(4));
    }
}
