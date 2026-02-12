#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use blake3::Hasher;
use metrics::{counter, histogram};

pub const NSR_ENGINE_ID_V0: &str = "nsr_datalog_lite_v0";
pub const NSR_SCHEMA_VERSION_V0: u16 = 1;
pub const NSR_RULESET_ID_V0: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CapabilityKind {
    NetHttp,
    FileRead,
    FileWrite,
    Exec,
    Unknown(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OutputClass {
    SafeText,
    Code,
    ExecIntent,
    ExternalIo,
    Unknown(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PolicyTag {
    Sensitive,
    Pii,
    Network,
    Unknown(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PolicyHint {
    Block,
    SafeOnly,
    Normal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReasonCode {
    ViolatesDenyByDefault,
    CoherenceGateTriggered,
    HighRiskToolRequest,
    UntrustedTarget,
    BudgetStress,
    LowConfidenceContext,
    SensitiveOutputClass,
    PolicyRuleHit(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionType {
    Observe,
    Answer,
    ToolUse,
    Plan,
    Unknown(u8),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionIntentSummary {
    pub action_type: ActionType,
    pub tool_kinds: Vec<CapabilityKind>,
    pub target_domain_hashes: Vec<u64>,
    pub target_path_hashes: Vec<u64>,
    pub output_class: OutputClass,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NsrContext {
    pub risk: f32,
    pub confidence: f32,
    pub coherence: Option<f32>,
    pub instability: Option<f32>,
    pub pressure: Option<f32>,
    pub surprise: Option<f32>,
    pub cortisol: Option<f32>,
    pub arousal: Option<f32>,
    pub has_capability_token: bool,
    pub compute_degraded_ratio: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fact {
    RiskHigh,
    RiskLow,
    ConfidenceLow,
    ConfidenceHigh,
    CoherenceLow,
    InstabilityHigh,
    PressureHigh,
    SurpriseHigh,
    CortisolHigh,
    ArousalHigh,
    RequestsCapability(CapabilityKind),
    RequestsTool(CapabilityKind),
    TargetDomainHashed(u64),
    TargetPathHashed(u64),
    OutputClass(OutputClass),
    PolicyTag(PolicyTag),
    AnyToolRequest,
    DenyByDefault,
    BudgetExceededRecent,
    BlockAction,
    RequireHumanReview,
    AllowSafeTextOnly,
    HighRiskToolRequestFlag,
    SensitiveOutputClassFlag,
    UntrustedTargetFlag,
    LowConfidenceContextFlag,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    HasFact(Fact),
    Not(Fact),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub head: Fact,
    pub body: Vec<Predicate>,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NsrBudget {
    pub max_rules: usize,
    pub max_facts: usize,
    pub max_steps: usize,
    pub max_reasons: usize,
    pub fail_fast_on_unavailable: bool,
}

impl Default for NsrBudget {
    fn default() -> Self {
        Self {
            max_rules: 256,
            max_facts: 512,
            max_steps: 2_048,
            max_reasons: 16,
            fail_fast_on_unavailable: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NsrError {
    BudgetExceeded,
    BackendDisabled,
    NotImplemented,
    Unavailable(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub struct NsrAssessment {
    pub nsr_risk: f32,
    pub nsr_confidence: f32,
    pub reasons: Vec<ReasonCode>,
    pub facts_digest: [u8; 32],
    pub ruleset_id: u32,
    pub schema_version: u16,
    pub engine_id: &'static str,
    pub policy_hint: PolicyHint,
    pub digest: [u8; 32],
}

pub trait NsrEngine {
    fn assess(
        &self,
        ctx: &NsrContext,
        intent: &DecisionIntentSummary,
        policy_tags: &[PolicyTag],
        budget: NsrBudget,
    ) -> Result<NsrAssessment, NsrError>;
}

#[derive(Clone, Debug)]
pub struct NsrDatalogLiteEngine {
    ruleset_id: u32,
    rules: Vec<Rule>,
}

impl Default for NsrDatalogLiteEngine {
    fn default() -> Self {
        Self::new_default_ruleset()
    }
}

impl NsrDatalogLiteEngine {
    pub fn new_default_ruleset() -> Self {
        Self {
            ruleset_id: NSR_RULESET_ID_V0,
            rules: default_ruleset_v0(),
        }
    }

    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    fn derive_initial_facts(
        &self,
        ctx: &NsrContext,
        intent: &DecisionIntentSummary,
        policy_tags: &[PolicyTag],
    ) -> BTreeSet<Fact> {
        let mut facts = BTreeSet::new();
        if ctx.risk >= 0.7 {
            facts.insert(Fact::RiskHigh);
        } else {
            facts.insert(Fact::RiskLow);
        }
        if ctx.confidence <= 0.35 {
            facts.insert(Fact::ConfidenceLow);
        } else {
            facts.insert(Fact::ConfidenceHigh);
        }
        if ctx.coherence.unwrap_or(1.0) <= 0.35 {
            facts.insert(Fact::CoherenceLow);
        }
        if ctx.instability.unwrap_or(0.0) >= 0.65 {
            facts.insert(Fact::InstabilityHigh);
        }
        if ctx.pressure.unwrap_or(0.0) >= 0.7 {
            facts.insert(Fact::PressureHigh);
        }
        if ctx.surprise.unwrap_or(0.0) >= 0.7 {
            facts.insert(Fact::SurpriseHigh);
        }
        if ctx.cortisol.unwrap_or(0.0) >= 0.75 {
            facts.insert(Fact::CortisolHigh);
        }
        if ctx.arousal.unwrap_or(0.0) >= 0.75 {
            facts.insert(Fact::ArousalHigh);
        }
        if ctx.compute_degraded_ratio.unwrap_or(0.0) >= 0.5 {
            facts.insert(Fact::BudgetExceededRecent);
        }

        for cap in &intent.tool_kinds {
            facts.insert(Fact::RequestsCapability(*cap));
            facts.insert(Fact::RequestsTool(*cap));
        }
        if !intent.tool_kinds.is_empty() {
            facts.insert(Fact::AnyToolRequest);
        }
        for tag in policy_tags {
            facts.insert(Fact::PolicyTag(*tag));
        }
        for hash in &intent.target_domain_hashes {
            facts.insert(Fact::TargetDomainHashed(*hash));
        }
        for hash in &intent.target_path_hashes {
            facts.insert(Fact::TargetPathHashed(*hash));
        }
        facts.insert(Fact::OutputClass(intent.output_class));

        if !ctx.has_capability_token && !intent.tool_kinds.is_empty() {
            facts.insert(Fact::DenyByDefault);
        }
        facts
    }
}

impl NsrEngine for NsrDatalogLiteEngine {
    fn assess(
        &self,
        ctx: &NsrContext,
        intent: &DecisionIntentSummary,
        policy_tags: &[PolicyTag],
        budget: NsrBudget,
    ) -> Result<NsrAssessment, NsrError> {
        counter!("ucf_nsr_assess_total").increment(1);

        if self.rules.len() > budget.max_rules {
            counter!("ucf_nsr_unavailable_total", "reason" => "rules_over_budget").increment(1);
            if budget.fail_fast_on_unavailable {
                return Err(NsrError::BudgetExceeded);
            }
            return Ok(unavailable_assessment(
                self.ruleset_id,
                ReasonCode::BudgetStress,
            ));
        }

        let mut facts = self.derive_initial_facts(ctx, intent, policy_tags);
        let mut hit_rules: Vec<u16> = Vec::new();
        let mut step = 0usize;
        let mut budget_stressed = false;

        'outer: loop {
            let mut changed = false;
            for (idx, rule) in self.rules.iter().enumerate() {
                if rule.body.len() > 8 {
                    continue;
                }
                if step >= budget.max_steps {
                    budget_stressed = true;
                    break 'outer;
                }
                step += 1;
                if rule_matches(rule, &facts)
                    && !facts.contains(&rule.head)
                    && facts.len() < budget.max_facts
                {
                    facts.insert(rule.head.clone());
                    hit_rules.push(u16::try_from(idx).unwrap_or(u16::MAX));
                    changed = true;
                } else if facts.len() >= budget.max_facts {
                    budget_stressed = true;
                    break 'outer;
                }
            }
            if !changed {
                break;
            }
        }

        if budget_stressed {
            facts.insert(Fact::BudgetExceededRecent);
            facts.insert(Fact::RequireHumanReview);
        }

        let mut reasons = reasons_from_facts(&facts, &hit_rules);
        if budget_stressed {
            reasons.push(ReasonCode::BudgetStress);
        }
        if reasons.len() > budget.max_reasons {
            reasons.truncate(budget.max_reasons);
        }

        let policy_hint = if facts.contains(&Fact::BlockAction) {
            PolicyHint::Block
        } else if facts.contains(&Fact::AllowSafeTextOnly)
            || facts.contains(&Fact::RequireHumanReview)
        {
            PolicyHint::SafeOnly
        } else {
            PolicyHint::Normal
        };

        let mut risk = ctx.risk.clamp(0.0, 1.0);
        if facts.contains(&Fact::BlockAction) {
            risk = 1.0;
        }
        if facts.contains(&Fact::RequireHumanReview) {
            risk = risk.max(0.8);
        }
        if facts.contains(&Fact::AllowSafeTextOnly) {
            risk = risk.max(0.6);
        }
        if facts.contains(&Fact::CoherenceLow) {
            risk += 0.12;
        }
        if facts.contains(&Fact::InstabilityHigh) {
            risk += 0.12;
        }
        if facts.contains(&Fact::CortisolHigh) {
            risk += 0.05;
        }
        if facts.contains(&Fact::ArousalHigh) {
            risk += 0.05;
        }
        risk = risk.clamp(0.0, 1.0);

        let mut confidence = ctx.confidence.clamp(0.0, 1.0);
        if facts.contains(&Fact::CoherenceLow) {
            confidence -= 0.2;
        }
        if facts.contains(&Fact::BudgetExceededRecent) {
            confidence -= 0.2;
        }
        if budget_stressed {
            confidence -= 0.4;
        }
        confidence = confidence.clamp(0.0, 1.0);

        let facts_digest = digest_facts(&facts);
        let digest = digest_assessment(
            self.ruleset_id,
            NSR_SCHEMA_VERSION_V0,
            NSR_ENGINE_ID_V0,
            facts_digest,
            risk,
            confidence,
            &reasons,
            policy_hint,
        );

        if matches!(policy_hint, PolicyHint::Block) {
            counter!("ucf_nsr_block_total").increment(1);
        }
        if matches!(policy_hint, PolicyHint::SafeOnly) {
            counter!("ucf_nsr_safeonly_total").increment(1);
        }
        histogram!("ucf_nsr_risk").record(f64::from(risk));
        histogram!("ucf_nsr_confidence").record(f64::from(confidence));

        Ok(NsrAssessment {
            nsr_risk: risk,
            nsr_confidence: confidence,
            reasons,
            facts_digest,
            ruleset_id: self.ruleset_id,
            schema_version: NSR_SCHEMA_VERSION_V0,
            engine_id: NSR_ENGINE_ID_V0,
            policy_hint,
            digest,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct NsrSmtEngine;

impl NsrEngine for NsrSmtEngine {
    fn assess(
        &self,
        _ctx: &NsrContext,
        _intent: &DecisionIntentSummary,
        _policy_tags: &[PolicyTag],
        _budget: NsrBudget,
    ) -> Result<NsrAssessment, NsrError> {
        Err(NsrError::BackendDisabled)
    }
}

fn rule_matches(rule: &Rule, facts: &BTreeSet<Fact>) -> bool {
    rule.body.iter().all(|predicate| match predicate {
        Predicate::HasFact(fact) => facts.contains(fact),
        Predicate::Not(fact) => !facts.contains(fact),
    })
}

fn reasons_from_facts(facts: &BTreeSet<Fact>, hit_rules: &[u16]) -> Vec<ReasonCode> {
    let mut reasons = Vec::new();
    if facts.contains(&Fact::DenyByDefault) && facts.contains(&Fact::AnyToolRequest) {
        reasons.push(ReasonCode::ViolatesDenyByDefault);
    }
    if facts.contains(&Fact::CoherenceLow) || facts.contains(&Fact::InstabilityHigh) {
        reasons.push(ReasonCode::CoherenceGateTriggered);
    }
    if facts.contains(&Fact::HighRiskToolRequestFlag) {
        reasons.push(ReasonCode::HighRiskToolRequest);
    }
    if facts.contains(&Fact::UntrustedTargetFlag) {
        reasons.push(ReasonCode::UntrustedTarget);
    }
    if facts.contains(&Fact::BudgetExceededRecent) {
        reasons.push(ReasonCode::BudgetStress);
    }
    if facts.contains(&Fact::LowConfidenceContextFlag) || facts.contains(&Fact::ConfidenceLow) {
        reasons.push(ReasonCode::LowConfidenceContext);
    }
    if facts.contains(&Fact::SensitiveOutputClassFlag) {
        reasons.push(ReasonCode::SensitiveOutputClass);
    }
    for rule in hit_rules {
        reasons.push(ReasonCode::PolicyRuleHit(*rule));
    }
    reasons
}

fn unavailable_assessment(ruleset_id: u32, reason: ReasonCode) -> NsrAssessment {
    let reasons = vec![reason];
    let facts_digest = [0u8; 32];
    let digest = digest_assessment(
        ruleset_id,
        NSR_SCHEMA_VERSION_V0,
        NSR_ENGINE_ID_V0,
        facts_digest,
        1.0,
        0.0,
        &reasons,
        PolicyHint::Block,
    );
    NsrAssessment {
        nsr_risk: 1.0,
        nsr_confidence: 0.0,
        reasons,
        facts_digest,
        ruleset_id,
        schema_version: NSR_SCHEMA_VERSION_V0,
        engine_id: NSR_ENGINE_ID_V0,
        policy_hint: PolicyHint::Block,
        digest,
    }
}

fn digest_facts(facts: &BTreeSet<Fact>) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.v0.facts");
    hasher.update(&u64::try_from(facts.len()).unwrap_or(0).to_be_bytes());
    for fact in facts {
        encode_fact(&mut hasher, fact);
    }
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn digest_assessment(
    ruleset_id: u32,
    schema_version: u16,
    engine_id: &str,
    facts_digest: [u8; 32],
    risk: f32,
    confidence: f32,
    reasons: &[ReasonCode],
    policy_hint: PolicyHint,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.nsr.v0.assessment");
    hasher.update(&ruleset_id.to_be_bytes());
    hasher.update(&schema_version.to_be_bytes());
    hasher.update(engine_id.as_bytes());
    hasher.update(&facts_digest);
    hasher.update(&risk.to_bits().to_be_bytes());
    hasher.update(&confidence.to_bits().to_be_bytes());
    hasher.update(&u64::try_from(reasons.len()).unwrap_or(0).to_be_bytes());
    for reason in reasons {
        encode_reason(&mut hasher, *reason);
    }
    hasher.update(&[policy_hint as u8]);
    *hasher.finalize().as_bytes()
}

fn encode_reason(hasher: &mut Hasher, reason: ReasonCode) {
    match reason {
        ReasonCode::ViolatesDenyByDefault => {
            hasher.update(&[1]);
        }
        ReasonCode::CoherenceGateTriggered => {
            hasher.update(&[2]);
        }
        ReasonCode::HighRiskToolRequest => {
            hasher.update(&[3]);
        }
        ReasonCode::UntrustedTarget => {
            hasher.update(&[4]);
        }
        ReasonCode::BudgetStress => {
            hasher.update(&[5]);
        }
        ReasonCode::LowConfidenceContext => {
            hasher.update(&[6]);
        }
        ReasonCode::SensitiveOutputClass => {
            hasher.update(&[7]);
        }
        ReasonCode::PolicyRuleHit(idx) => {
            hasher.update(&[8]);
            hasher.update(&idx.to_be_bytes());
        }
    }
}

fn encode_fact(hasher: &mut Hasher, fact: &Fact) {
    match fact {
        Fact::RiskHigh => {
            hasher.update(&[1]);
        }
        Fact::RiskLow => {
            hasher.update(&[2]);
        }
        Fact::ConfidenceLow => {
            hasher.update(&[3]);
        }
        Fact::ConfidenceHigh => {
            hasher.update(&[4]);
        }
        Fact::CoherenceLow => {
            hasher.update(&[5]);
        }
        Fact::InstabilityHigh => {
            hasher.update(&[6]);
        }
        Fact::PressureHigh => {
            hasher.update(&[7]);
        }
        Fact::SurpriseHigh => {
            hasher.update(&[8]);
        }
        Fact::CortisolHigh => {
            hasher.update(&[9]);
        }
        Fact::ArousalHigh => {
            hasher.update(&[10]);
        }
        Fact::RequestsCapability(kind) => {
            hasher.update(&[11]);
            encode_capability(hasher, *kind);
        }
        Fact::RequestsTool(kind) => {
            hasher.update(&[12]);
            encode_capability(hasher, *kind);
        }
        Fact::TargetDomainHashed(hash) => {
            hasher.update(&[13]);
            hasher.update(&hash.to_be_bytes());
        }
        Fact::TargetPathHashed(hash) => {
            hasher.update(&[14]);
            hasher.update(&hash.to_be_bytes());
        }
        Fact::OutputClass(output_class) => {
            hasher.update(&[15]);
            encode_output_class(hasher, *output_class);
        }
        Fact::PolicyTag(tag) => {
            hasher.update(&[16]);
            encode_policy_tag(hasher, *tag);
        }
        Fact::AnyToolRequest => {
            hasher.update(&[17]);
        }
        Fact::DenyByDefault => {
            hasher.update(&[18]);
        }
        Fact::BudgetExceededRecent => {
            hasher.update(&[19]);
        }
        Fact::BlockAction => {
            hasher.update(&[20]);
        }
        Fact::RequireHumanReview => {
            hasher.update(&[21]);
        }
        Fact::AllowSafeTextOnly => {
            hasher.update(&[22]);
        }
        Fact::HighRiskToolRequestFlag => {
            hasher.update(&[23]);
        }
        Fact::SensitiveOutputClassFlag => {
            hasher.update(&[24]);
        }
        Fact::UntrustedTargetFlag => {
            hasher.update(&[25]);
        }
        Fact::LowConfidenceContextFlag => {
            hasher.update(&[26]);
        }
    }
}

fn encode_capability(hasher: &mut Hasher, capability: CapabilityKind) {
    match capability {
        CapabilityKind::NetHttp => {
            hasher.update(&[1]);
        }
        CapabilityKind::FileRead => {
            hasher.update(&[2]);
        }
        CapabilityKind::FileWrite => {
            hasher.update(&[3]);
        }
        CapabilityKind::Exec => {
            hasher.update(&[4]);
        }
        CapabilityKind::Unknown(code) => {
            hasher.update(&[255]);
            hasher.update(&code.to_be_bytes());
        }
    }
}

fn encode_output_class(hasher: &mut Hasher, output_class: OutputClass) {
    match output_class {
        OutputClass::SafeText => {
            hasher.update(&[1]);
        }
        OutputClass::Code => {
            hasher.update(&[2]);
        }
        OutputClass::ExecIntent => {
            hasher.update(&[3]);
        }
        OutputClass::ExternalIo => {
            hasher.update(&[4]);
        }
        OutputClass::Unknown(code) => {
            hasher.update(&[code]);
        }
    }
}

fn encode_policy_tag(hasher: &mut Hasher, tag: PolicyTag) {
    match tag {
        PolicyTag::Sensitive => {
            hasher.update(&[1]);
        }
        PolicyTag::Pii => {
            hasher.update(&[2]);
        }
        PolicyTag::Network => {
            hasher.update(&[3]);
        }
        PolicyTag::Unknown(code) => {
            hasher.update(&[255]);
            hasher.update(&code.to_be_bytes());
        }
    }
}

fn default_ruleset_v0() -> Vec<Rule> {
    vec![
        Rule {
            head: Fact::HighRiskToolRequestFlag,
            body: vec![
                Predicate::HasFact(Fact::RequestsTool(CapabilityKind::NetHttp)),
                Predicate::HasFact(Fact::RiskHigh),
            ],
            weight: 0.9,
        },
        Rule {
            head: Fact::PolicyTag(PolicyTag::Network),
            body: vec![Predicate::HasFact(Fact::HighRiskToolRequestFlag)],
            weight: 0.8,
        },
        Rule {
            head: Fact::BlockAction,
            body: vec![
                Predicate::HasFact(Fact::DenyByDefault),
                Predicate::HasFact(Fact::AnyToolRequest),
            ],
            weight: 1.0,
        },
        Rule {
            head: Fact::RequireHumanReview,
            body: vec![Predicate::HasFact(Fact::CoherenceLow)],
            weight: 0.8,
        },
        Rule {
            head: Fact::RequireHumanReview,
            body: vec![Predicate::HasFact(Fact::InstabilityHigh)],
            weight: 0.8,
        },
        Rule {
            head: Fact::SensitiveOutputClassFlag,
            body: vec![Predicate::HasFact(Fact::OutputClass(
                OutputClass::ExecIntent,
            ))],
            weight: 0.9,
        },
        Rule {
            head: Fact::AllowSafeTextOnly,
            body: vec![Predicate::HasFact(Fact::SensitiveOutputClassFlag)],
            weight: 0.9,
        },
        Rule {
            head: Fact::LowConfidenceContextFlag,
            body: vec![Predicate::HasFact(Fact::ConfidenceLow)],
            weight: 0.6,
        },
        Rule {
            head: Fact::RequireHumanReview,
            body: vec![Predicate::HasFact(Fact::BudgetExceededRecent)],
            weight: 0.7,
        },
        Rule {
            head: Fact::UntrustedTargetFlag,
            body: vec![
                Predicate::HasFact(Fact::PolicyTag(PolicyTag::Sensitive)),
                Predicate::HasFact(Fact::RequestsTool(CapabilityKind::FileWrite)),
            ],
            weight: 0.9,
        },
    ]
}

pub fn apply_nsr_to_fep(
    efe: f32,
    confidence: f32,
    lambda_nsr: f32,
    assessment: &NsrAssessment,
) -> (f32, f32) {
    (
        efe + lambda_nsr.max(0.0) * assessment.nsr_risk,
        (confidence * assessment.nsr_confidence).clamp(0.0, 1.0),
    )
}

pub fn fallback_assessment_fail_open() -> NsrAssessment {
    unavailable_assessment(NSR_RULESET_ID_V0, ReasonCode::BudgetStress)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_ctx() -> NsrContext {
        NsrContext {
            risk: 0.3,
            confidence: 0.9,
            coherence: Some(0.9),
            instability: Some(0.1),
            pressure: Some(0.1),
            surprise: Some(0.1),
            cortisol: Some(0.1),
            arousal: Some(0.1),
            has_capability_token: true,
            compute_degraded_ratio: Some(0.0),
        }
    }

    fn mk_intent() -> DecisionIntentSummary {
        DecisionIntentSummary {
            action_type: ActionType::ToolUse,
            tool_kinds: vec![CapabilityKind::NetHttp],
            target_domain_hashes: vec![42],
            target_path_hashes: vec![24],
            output_class: OutputClass::Code,
        }
    }

    #[test]
    fn deterministic_assessment_digest() {
        let engine = NsrDatalogLiteEngine::default();
        let ctx = mk_ctx();
        let intent = mk_intent();
        let out_a = engine
            .assess(&ctx, &intent, &[PolicyTag::Network], NsrBudget::default())
            .unwrap();
        let out_b = engine
            .assess(&ctx, &intent, &[PolicyTag::Network], NsrBudget::default())
            .unwrap();
        assert_eq!(out_a.facts_digest, out_b.facts_digest);
        assert_eq!(out_a.digest, out_b.digest);
    }

    #[test]
    fn deny_by_default_blocks_without_capability() {
        let engine = NsrDatalogLiteEngine::default();
        let mut ctx = mk_ctx();
        ctx.has_capability_token = false;
        let out = engine
            .assess(&ctx, &mk_intent(), &[], NsrBudget::default())
            .unwrap();
        assert!(out.reasons.contains(&ReasonCode::ViolatesDenyByDefault));
        assert_eq!(out.policy_hint, PolicyHint::Block);
        assert!(out.nsr_risk >= 0.8);
    }

    #[test]
    fn lower_coherence_monotonic_risk() {
        let engine = NsrDatalogLiteEngine::default();
        let mut high = mk_ctx();
        high.coherence = Some(0.95);
        let mut low = mk_ctx();
        low.coherence = Some(0.1);
        let hi = engine
            .assess(&high, &mk_intent(), &[], NsrBudget::default())
            .unwrap();
        let lo = engine
            .assess(&low, &mk_intent(), &[], NsrBudget::default())
            .unwrap();
        assert!(lo.nsr_risk >= hi.nsr_risk);
        assert!(lo.nsr_confidence <= hi.nsr_confidence);
    }

    #[test]
    fn reason_boundedness_and_clamps() {
        let engine = NsrDatalogLiteEngine::default();
        let mut ctx = mk_ctx();
        ctx.risk = 5.0;
        ctx.confidence = -4.0;
        ctx.coherence = Some(0.0);
        let out = engine
            .assess(
                &ctx,
                &mk_intent(),
                &[PolicyTag::Sensitive, PolicyTag::Network],
                NsrBudget {
                    max_reasons: 2,
                    ..NsrBudget::default()
                },
            )
            .unwrap();
        assert!((0.0..=1.0).contains(&out.nsr_risk));
        assert!((0.0..=1.0).contains(&out.nsr_confidence));
        assert!(out.reasons.len() <= 2);
    }

    #[test]
    fn fail_open_or_fast_is_deterministic() {
        let engine = NsrDatalogLiteEngine::default();
        let budget = NsrBudget {
            max_rules: 1,
            fail_fast_on_unavailable: false,
            ..NsrBudget::default()
        };
        let out = engine.assess(&mk_ctx(), &mk_intent(), &[], budget).unwrap();
        assert_eq!(out.nsr_risk, 1.0);
        assert_eq!(out.nsr_confidence, 0.0);

        let err = engine.assess(
            &mk_ctx(),
            &mk_intent(),
            &[],
            NsrBudget {
                max_rules: 1,
                fail_fast_on_unavailable: true,
                ..NsrBudget::default()
            },
        );
        assert_eq!(err, Err(NsrError::BudgetExceeded));
    }
}
