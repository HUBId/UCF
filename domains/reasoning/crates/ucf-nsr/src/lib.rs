#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_cde_scm::CounterfactualResult;
use ucf_sandbox::IntentSummary;
use ucf_types::Digest32;

const LIGHT_PROOF_DOMAIN: &[u8] = b"ucf.nsr.proof.light.v1";

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
    pub causal_report_commit: Digest32,
    pub violations: Vec<NsrViolation>,
    pub proof_digest: Digest32,
    pub commit: Digest32,
}

impl NsrReport {
    pub fn causal_verdict(&self) -> NsrVerdict {
        causal_verdict_from_violations(&self.violations)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrInput {
    pub cycle_id: u64,
    pub intent: IntentSummary,
    pub policy_class: u16,
    pub proposed_actions: Vec<ActionIntent>,
    pub world_state_commit: Digest32,
    pub causal_report_commit: Digest32,
    pub counterfactuals: Vec<CounterfactualResult>,
    pub nsr_warn_threshold: Option<u16>,
    pub nsr_deny_threshold: Option<u16>,
    pub commit: Digest32,
}

impl NsrInput {
    pub fn new(
        cycle_id: u64,
        intent: IntentSummary,
        policy_class: u16,
        proposed_actions: Vec<ActionIntent>,
        world_state_commit: Digest32,
        causal_report_commit: Digest32,
        counterfactuals: Vec<CounterfactualResult>,
    ) -> Self {
        let digest_fields = NsrInputDigestFields::new(
            cycle_id,
            &intent,
            policy_class,
            &proposed_actions,
            world_state_commit,
            causal_report_commit,
            &counterfactuals,
            None,
            None,
        );
        let commit = digest_nsr_input(&digest_fields);
        Self {
            cycle_id,
            intent,
            policy_class,
            proposed_actions,
            world_state_commit,
            causal_report_commit,
            counterfactuals,
            nsr_warn_threshold: None,
            nsr_deny_threshold: None,
            commit,
        }
    }

    pub fn with_nsr_thresholds(mut self, warn: u16, deny: u16) -> Self {
        let warn = warn.min(10_000);
        let deny = deny.min(10_000).max(warn);
        self.nsr_warn_threshold = Some(warn);
        self.nsr_deny_threshold = Some(deny);
        let digest_fields = NsrInputDigestFields::new(
            self.cycle_id,
            &self.intent,
            self.policy_class,
            &self.proposed_actions,
            self.world_state_commit,
            self.causal_report_commit,
            &self.counterfactuals,
            self.nsr_warn_threshold,
            self.nsr_deny_threshold,
        );
        self.commit = digest_nsr_input(&digest_fields);
        self
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
        let causal_result = evaluate_causal(input);
        let mut violations = rule_result.violations;
        violations.extend(constraint_result.violations);
        violations.extend(causal_result.violations);
        let proof_digest = compute_proof_digest(
            input,
            &rule_result.rules_fired,
            &constraint_result.constraints_checked,
            &causal_result.causal_checks,
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

pub struct CausalCheckResult {
    pub violations: Vec<NsrViolation>,
    pub causal_checks: Vec<String>,
}

pub fn compute_proof_digest(
    input: &NsrInput,
    rules_fired: &[String],
    constraints_checked: &[String],
    causal_checks: &[String],
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.proof.v2");
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
    hasher.update(
        &u64::try_from(causal_checks.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for check in causal_checks {
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
    let commit = digest_nsr_report(
        &verdict,
        input.causal_report_commit,
        &violations,
        proof_digest,
        input.commit,
    );
    NsrReport {
        verdict,
        causal_report_commit: input.causal_report_commit,
        violations,
        proof_digest,
        commit,
    }
}

pub fn light_report(input: &NsrInput) -> NsrReport {
    let mut hasher = Hasher::new();
    hasher.update(LIGHT_PROOF_DOMAIN);
    hasher.update(input.commit.as_bytes());
    let proof_digest = Digest32::new(*hasher.finalize().as_bytes());
    finalize_report(input, Vec::new(), proof_digest)
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

fn causal_verdict_from_violations(violations: &[NsrViolation]) -> NsrVerdict {
    if violations
        .iter()
        .any(|violation| is_causal_violation(&violation.code))
    {
        if violations
            .iter()
            .filter(|violation| is_causal_violation(&violation.code))
            .any(|violation| violation.severity >= 9000)
        {
            return NsrVerdict::Deny;
        }
        return NsrVerdict::Warn;
    }
    NsrVerdict::Ok
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
    causal_report_commit: Digest32,
    violations: &[NsrViolation],
    proof_digest: Digest32,
    input_commit: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.report.v2");
    hasher.update(&[verdict.as_u8()]);
    hasher.update(causal_report_commit.as_bytes());
    hasher.update(proof_digest.as_bytes());
    hasher.update(input_commit.as_bytes());
    hasher.update(&u64::try_from(violations.len()).unwrap_or(0).to_be_bytes());
    for violation in violations {
        hasher.update(violation.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

struct NsrInputDigestFields<'a> {
    cycle_id: u64,
    intent: &'a IntentSummary,
    policy_class: u16,
    proposed_actions: &'a [ActionIntent],
    world_state_commit: Digest32,
    causal_report_commit: Digest32,
    counterfactuals: &'a [CounterfactualResult],
    nsr_warn_threshold: Option<u16>,
    nsr_deny_threshold: Option<u16>,
}

impl<'a> NsrInputDigestFields<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cycle_id: u64,
        intent: &'a IntentSummary,
        policy_class: u16,
        proposed_actions: &'a [ActionIntent],
        world_state_commit: Digest32,
        causal_report_commit: Digest32,
        counterfactuals: &'a [CounterfactualResult],
        nsr_warn_threshold: Option<u16>,
        nsr_deny_threshold: Option<u16>,
    ) -> Self {
        Self {
            cycle_id,
            intent,
            policy_class,
            proposed_actions,
            world_state_commit,
            causal_report_commit,
            counterfactuals,
            nsr_warn_threshold,
            nsr_deny_threshold,
        }
    }
}

fn digest_nsr_input(fields: &NsrInputDigestFields<'_>) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.input.v2");
    hasher.update(&fields.cycle_id.to_be_bytes());
    hasher.update(&fields.intent.intent.to_be_bytes());
    hasher.update(&fields.intent.risk.to_be_bytes());
    hasher.update(fields.intent.commit.as_bytes());
    hasher.update(&fields.policy_class.to_be_bytes());
    hasher.update(
        &u64::try_from(fields.proposed_actions.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for action in fields.proposed_actions {
        hasher.update(&u64::try_from(action.tag.len()).unwrap_or(0).to_be_bytes());
        hasher.update(action.tag.as_bytes());
    }
    hasher.update(fields.world_state_commit.as_bytes());
    hasher.update(fields.causal_report_commit.as_bytes());
    hasher.update(&fields.nsr_warn_threshold.unwrap_or(0).to_be_bytes());
    hasher.update(&fields.nsr_deny_threshold.unwrap_or(0).to_be_bytes());
    hasher.update(
        &u64::try_from(fields.counterfactuals.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for counterfactual in fields.counterfactuals {
        hasher.update(&counterfactual.predicted_delta.to_be_bytes());
        hasher.update(&counterfactual.confidence.to_be_bytes());
        hasher.update(counterfactual.commit.as_bytes());
    }
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

#[derive(Clone, Copy)]
struct CausalThresholds {
    warn: u16,
    deny: u16,
}

fn causal_thresholds(policy_class: u16) -> CausalThresholds {
    if policy_class >= 2 {
        CausalThresholds {
            warn: 4000,
            deny: 7500,
        }
    } else {
        CausalThresholds {
            warn: 5000,
            deny: 8500,
        }
    }
}

fn input_causal_thresholds(input: &NsrInput) -> CausalThresholds {
    match (input.nsr_warn_threshold, input.nsr_deny_threshold) {
        (Some(warn), Some(deny)) => CausalThresholds {
            warn: warn.min(10_000),
            deny: deny.min(10_000).max(warn),
        },
        _ => causal_thresholds(input.policy_class),
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

fn evaluate_causal(input: &NsrInput) -> CausalCheckResult {
    let thresholds = input_causal_thresholds(input);
    let mut violations = Vec::new();
    let mut causal_checks = Vec::new();

    for counterfactual in &input.counterfactuals {
        let delta = counterfactual.predicted_delta;
        let confidence = counterfactual.confidence;
        let verdict = if delta > 0 {
            if confidence >= thresholds.deny {
                "deny"
            } else if confidence >= thresholds.warn {
                "warn"
            } else {
                "uncertain"
            }
        } else if delta < 0 {
            "decrease"
        } else {
            "flat"
        };
        let detail = format!(
            "delta={};confidence={};cf={}",
            delta, confidence, counterfactual.commit
        );
        causal_checks.push(format!(
            "causal:{}:{}:{}",
            verdict, detail, input.causal_report_commit
        ));
        if delta <= 0 {
            continue;
        }
        if confidence >= thresholds.deny {
            violations.push(build_violation(
                "NSR_CAUSAL_CONFIDENCE_HIGH_DENY",
                &detail,
                9500,
                input.commit,
            ));
        } else if confidence >= thresholds.warn {
            violations.push(build_violation(
                "NSR_CAUSAL_RISK_INCREASE",
                &detail,
                8200,
                input.commit,
            ));
        } else {
            violations.push(build_violation(
                "NSR_CAUSAL_UNCERTAIN_WARN",
                &detail,
                7000,
                input.commit,
            ));
        }
    }

    CausalCheckResult {
        violations,
        causal_checks,
    }
}

fn is_causal_violation(code: &str) -> bool {
    matches!(
        code,
        "NSR_CAUSAL_RISK_INCREASE"
            | "NSR_CAUSAL_CONFIDENCE_HIGH_DENY"
            | "NSR_CAUSAL_UNCERTAIN_WARN"
    )
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
            Digest32::new([0u8; 32]),
            Vec::new(),
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
            Digest32::new([0u8; 32]),
            Vec::new(),
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
            Digest32::new([0u8; 32]),
            Vec::new(),
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
            Digest32::new([0u8; 32]),
            Vec::new(),
        );

        let report = engine.evaluate(&input);

        assert_eq!(report.verdict, NsrVerdict::Ok);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn causal_high_confidence_delta_denies() {
        let engine = NsrEngine::new();
        let counterfactuals = vec![CounterfactualResult::new(
            12,
            9000,
            Digest32::new([1u8; 32]),
        )];
        let input = NsrInput::new(
            10,
            IntentSummary::new(100, 100),
            1,
            vec![ActionIntent::new("internal")],
            Digest32::new([5u8; 32]),
            Digest32::new([8u8; 32]),
            counterfactuals,
        );

        let report = engine.evaluate(&input);

        assert_eq!(report.verdict, NsrVerdict::Deny);
        assert!(report
            .violations
            .iter()
            .any(|violation| violation.code == "NSR_CAUSAL_CONFIDENCE_HIGH_DENY"));
    }

    #[test]
    fn causal_medium_confidence_delta_warns() {
        let engine = NsrEngine::new();
        let counterfactuals = vec![CounterfactualResult::new(4, 6000, Digest32::new([2u8; 32]))];
        let input = NsrInput::new(
            11,
            IntentSummary::new(100, 100),
            1,
            vec![ActionIntent::new("internal")],
            Digest32::new([6u8; 32]),
            Digest32::new([9u8; 32]),
            counterfactuals,
        );

        let report = engine.evaluate(&input);

        assert_eq!(report.verdict, NsrVerdict::Warn);
        assert!(report
            .violations
            .iter()
            .any(|violation| violation.code == "NSR_CAUSAL_RISK_INCREASE"));
    }

    #[test]
    fn causal_non_positive_delta_is_ignored() {
        let engine = NsrEngine::new();
        let counterfactuals = vec![
            CounterfactualResult::new(-3, 9000, Digest32::new([3u8; 32])),
            CounterfactualResult::new(0, 9000, Digest32::new([4u8; 32])),
        ];
        let input = NsrInput::new(
            12,
            IntentSummary::new(100, 100),
            1,
            vec![ActionIntent::new("internal")],
            Digest32::new([7u8; 32]),
            Digest32::new([0u8; 32]),
            counterfactuals,
        );

        let report = engine.evaluate(&input);

        assert_eq!(report.verdict, NsrVerdict::Ok);
        assert!(report
            .violations
            .iter()
            .all(|violation| !is_causal_violation(&violation.code)));
    }
}
