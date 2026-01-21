#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_sandbox::IntentSummary;
use ucf_types::Digest32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NsrVerdict {
    Ok,
    Warn,
    Deny,
}

impl NsrVerdict {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Ok => 0,
            Self::Warn => 1,
            Self::Deny => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionIntent {
    pub tag: String,
}

impl ActionIntent {
    pub fn new(tag: impl Into<String>) -> Self {
        Self { tag: tag.into() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrViolation {
    pub code: String,
    pub detail_digest: Digest32,
    pub severity: u16,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrReport {
    pub verdict: NsrVerdict,
    pub violations: Vec<NsrViolation>,
    pub proof_digest: Digest32,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrInput {
    pub cycle_id: u64,
    pub intent: IntentSummary,
    pub policy_class: u16,
    pub proposed_actions: Vec<ActionIntent>,
    pub world_state_commit: Digest32,
    pub commit: Digest32,
}

impl NsrInput {
    pub fn new(
        cycle_id: u64,
        intent: IntentSummary,
        policy_class: u16,
        proposed_actions: Vec<ActionIntent>,
        world_state_commit: Digest32,
    ) -> Self {
        let commit = digest_nsr_input(
            cycle_id,
            &intent,
            policy_class,
            &proposed_actions,
            world_state_commit,
        );
        Self {
            cycle_id,
            intent,
            policy_class,
            proposed_actions,
            world_state_commit,
            commit,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NsrEngine {
    rule_checker: RuleChecker,
    constraint_checker: ConstraintChecker,
}

impl NsrEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn evaluate(&self, input: &NsrInput) -> NsrReport {
        let rule_result = self.rule_checker.evaluate(input);
        let constraint_result = self.constraint_checker.evaluate(input);
        let mut violations = rule_result.violations;
        violations.extend(constraint_result.violations);
        let proof_digest = compute_proof_digest(
            input,
            &rule_result.rules_fired,
            &constraint_result.constraints_checked,
        );
        finalize_report(input, violations, proof_digest)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RuleChecker;

impl RuleChecker {
    pub fn evaluate(&self, input: &NsrInput) -> RuleCheckResult {
        let intent_kind = intent_kind_for_score(input.intent.intent);
        let tags = input
            .proposed_actions
            .iter()
            .map(|action| action.tag.as_str())
            .collect::<Vec<_>>();
        let mut violations = Vec::new();
        let mut rules_fired = Vec::new();

        for rule in rules_for_policy(input.policy_class) {
            if let Some(detail) = (rule.matcher)(intent_kind, &tags) {
                violations.push(build_violation(
                    rule.code,
                    &detail,
                    rule.severity,
                    input.commit,
                ));
                rules_fired.push(rule.code.to_string());
            }
        }

        RuleCheckResult {
            violations,
            rules_fired,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConstraintChecker;

impl ConstraintChecker {
    pub fn evaluate(&self, input: &NsrInput) -> ConstraintCheckResult {
        let thresholds = constraint_thresholds(input.policy_class);
        let risk_score = u64::from(input.intent.risk);
        let drift_score = u64::from(commit_to_u16(input.world_state_commit) % 10_000);
        let surprise_score = u64::from(commit_to_u16(input.commit) % 10_000);
        let ops_budget = 10_000u64
            .saturating_sub(u64::try_from(input.proposed_actions.len()).unwrap_or(0) * 700);

        let mut constraints_checked = Vec::new();
        let mut failed = Vec::new();

        let risk_ok = risk_score <= thresholds.risk_max;
        constraints_checked.push(format!("risk_score<={}:{risk_ok}", thresholds.risk_max));
        if !risk_ok {
            failed.push(format!("risk_score={risk_score}"));
        }

        let drift_ok = drift_score <= thresholds.drift_max;
        constraints_checked.push(format!("drift_score<={}:{drift_ok}", thresholds.drift_max));
        if !drift_ok {
            failed.push(format!("drift_score={drift_score}"));
        }

        let surprise_ok = surprise_score <= thresholds.surprise_max;
        constraints_checked.push(format!(
            "surprise_score<={}:{surprise_ok}",
            thresholds.surprise_max
        ));
        if !surprise_ok {
            failed.push(format!("surprise_score={surprise_score}"));
        }

        let ops_ok = ops_budget >= thresholds.ops_min;
        constraints_checked.push(format!("ops_budget>={}:{}", thresholds.ops_min, ops_ok));
        if !ops_ok {
            failed.push(format!("ops_budget={ops_budget}"));
        }

        let mut violations = Vec::new();
        if !failed.is_empty() {
            let detail = failed.join(";");
            violations.push(build_violation(
                "NSR_CONSTRAINT_FAIL",
                &detail,
                9000,
                input.commit,
            ));
        }

        ConstraintCheckResult {
            violations,
            constraints_checked,
        }
    }
}

pub struct RuleCheckResult {
    pub violations: Vec<NsrViolation>,
    pub rules_fired: Vec<String>,
}

pub struct ConstraintCheckResult {
    pub violations: Vec<NsrViolation>,
    pub constraints_checked: Vec<String>,
}

pub fn compute_proof_digest(
    input: &NsrInput,
    rules_fired: &[String],
    constraints_checked: &[String],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.proof.v1");
    hasher.update(input.commit.as_bytes());
    hasher.update(&u64::try_from(rules_fired.len()).unwrap_or(0).to_be_bytes());
    for rule in rules_fired {
        hasher.update(&u64::try_from(rule.len()).unwrap_or(0).to_be_bytes());
        hasher.update(rule.as_bytes());
    }
    hasher.update(
        &u64::try_from(constraints_checked.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for check in constraints_checked {
        hasher.update(&u64::try_from(check.len()).unwrap_or(0).to_be_bytes());
        hasher.update(check.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

pub fn finalize_report(
    input: &NsrInput,
    violations: Vec<NsrViolation>,
    proof_digest: Digest32,
) -> NsrReport {
    let verdict = verdict_from_violations(&violations);
    let commit = digest_nsr_report(&verdict, &violations, proof_digest, input.commit);
    NsrReport {
        verdict,
        violations,
        proof_digest,
        commit,
    }
}

fn verdict_from_violations(violations: &[NsrViolation]) -> NsrVerdict {
    if violations.is_empty() {
        return NsrVerdict::Ok;
    }
    if violations
        .iter()
        .any(|violation| violation.severity >= 9000)
    {
        return NsrVerdict::Deny;
    }
    NsrVerdict::Warn
}

fn build_violation(
    code: &str,
    detail: &str,
    severity: u16,
    input_commit: Digest32,
) -> NsrViolation {
    let detail_digest = digest_violation_detail(code, detail, input_commit);
    let commit = digest_violation_commit(code, detail_digest, severity, input_commit);
    NsrViolation {
        code: code.to_string(),
        detail_digest,
        severity,
        commit,
    }
}

fn digest_violation_detail(code: &str, detail: &str, input_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.violation.detail.v1");
    hasher.update(code.as_bytes());
    hasher.update(detail.as_bytes());
    hasher.update(input_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_violation_commit(
    code: &str,
    detail_digest: Digest32,
    severity: u16,
    input_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.violation.commit.v1");
    hasher.update(code.as_bytes());
    hasher.update(detail_digest.as_bytes());
    hasher.update(&severity.to_be_bytes());
    hasher.update(input_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_nsr_report(
    verdict: &NsrVerdict,
    violations: &[NsrViolation],
    proof_digest: Digest32,
    input_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.report.v1");
    hasher.update(&[verdict.as_u8()]);
    hasher.update(proof_digest.as_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(&u64::try_from(violations.len()).unwrap_or(0).to_be_bytes());
    for violation in violations {
        hasher.update(violation.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_nsr_input(
    cycle_id: u64,
    intent: &IntentSummary,
    policy_class: u16,
    proposed_actions: &[ActionIntent],
    world_state_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.input.v1");
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(&intent.intent.to_be_bytes());
    hasher.update(&intent.risk.to_be_bytes());
    hasher.update(intent.commit.as_bytes());
    hasher.update(&policy_class.to_be_bytes());
    hasher.update(
        &u64::try_from(proposed_actions.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for action in proposed_actions {
        hasher.update(&u64::try_from(action.tag.len()).unwrap_or(0).to_be_bytes());
        hasher.update(action.tag.as_bytes());
    }
    hasher.update(world_state_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_to_u16(commit: Digest32) -> u16 {
    let bytes = commit.as_bytes();
    u16::from_be_bytes([bytes[0], bytes[1]])
}

#[derive(Clone, Copy)]
struct ConstraintThresholds {
    risk_max: u64,
    drift_max: u64,
    surprise_max: u64,
    ops_min: u64,
}

fn constraint_thresholds(policy_class: u16) -> ConstraintThresholds {
    if policy_class >= 2 {
        ConstraintThresholds {
            risk_max: 6000,
            drift_max: 6000,
            surprise_max: 7000,
            ops_min: 2000,
        }
    } else {
        ConstraintThresholds {
            risk_max: 10_000,
            drift_max: 10_000,
            surprise_max: 10_000,
            ops_min: 500,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntentKind {
    Internal,
    ExternalEffect,
    DataExfil,
    SelfMod,
}

fn intent_kind_for_score(score: u16) -> IntentKind {
    if score >= 9000 {
        IntentKind::SelfMod
    } else if score >= 7000 {
        IntentKind::DataExfil
    } else if score >= 4000 {
        IntentKind::ExternalEffect
    } else {
        IntentKind::Internal
    }
}

struct Rule {
    code: &'static str,
    severity: u16,
    matcher: fn(IntentKind, &[&str]) -> Option<String>,
}

fn rules_for_policy(policy_class: u16) -> Vec<Rule> {
    let mut rules = vec![Rule {
        code: "NSR_RULE_FORBIDDEN_INTENT",
        severity: 9000,
        matcher: |intent, _| match intent {
            IntentKind::SelfMod => Some("intent=self_mod".to_string()),
            _ => None,
        },
    }];

    if policy_class >= 2 {
        rules.push(Rule {
            code: "NSR_RULE_FORBIDDEN_INTENT",
            severity: 9000,
            matcher: |intent, _| match intent {
                IntentKind::DataExfil => Some("intent=data_exfil".to_string()),
                _ => None,
            },
        });
    }

    rules.push(Rule {
        code: "NSR_RULE_UNSAFE_SEQUENCE",
        severity: 8000,
        matcher: |_, tags| {
            if has_sequence(tags, "self_mod", "external_effect") {
                Some("sequence=self_mod->external_effect".to_string())
            } else if has_sequence(tags, "data_exfil", "external_effect") {
                Some("sequence=data_exfil->external_effect".to_string())
            } else {
                None
            }
        },
    });

    rules
}

fn has_sequence(tags: &[&str], first: &str, second: &str) -> bool {
    let mut seen_first = false;
    for tag in tags {
        if *tag == first {
            seen_first = true;
            continue;
        }
        if seen_first && *tag == second {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> NsrInput {
        NsrInput::new(
            1,
            IntentSummary::new(100, 100),
            1,
            vec![ActionIntent::new("internal")],
            Digest32::new([3u8; 32]),
        )
    }

    #[test]
    fn evaluation_is_deterministic() {
        let engine = NsrEngine::new();
        let input = base_input();

        let first = engine.evaluate(&input);
        let second = engine.evaluate(&input);

        assert_eq!(first, second);
    }

    #[test]
    fn forbidden_intent_produces_deny() {
        let engine = NsrEngine::new();
        let input = NsrInput::new(
            9,
            IntentSummary::new(9500, 100),
            1,
            Vec::new(),
            Digest32::new([9u8; 32]),
        );

        let report = engine.evaluate(&input);

        assert_eq!(report.verdict, NsrVerdict::Deny);
        assert!(!report.violations.is_empty());
    }

    #[test]
    fn constraint_fail_produces_deny() {
        let engine = NsrEngine::new();
        let input = NsrInput::new(
            7,
            IntentSummary::new(100, 9000),
            2,
            vec![ActionIntent::new("internal")],
            Digest32::new([250u8; 32]),
        );

        let report = engine.evaluate(&input);

        assert_eq!(report.verdict, NsrVerdict::Deny);
        assert!(report
            .violations
            .iter()
            .any(|violation| violation.code == "NSR_CONSTRAINT_FAIL"));
    }

    #[test]
    fn ok_path_has_no_violations() {
        let engine = NsrEngine::new();
        let input = NsrInput::new(
            2,
            IntentSummary::new(100, 100),
            1,
            vec![ActionIntent::new("internal")],
            Digest32::new([0u8; 32]),
        );

        let report = engine.evaluate(&input);

        assert_eq!(report.verdict, NsrVerdict::Ok);
        assert!(report.violations.is_empty());
    }
}
