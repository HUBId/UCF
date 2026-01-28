#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_cde_scm::{edge_key, CdeNodeId, CounterfactualResult, NSR_ATOM_MIN};
use ucf_sandbox::IntentSummary;
use ucf_spikebus::SpikeKind;
use ucf_structural_store::NsrThresholds;
use ucf_types::Digest32;

#[cfg(feature = "nsr_datalog")]
mod backend_datalog;
#[cfg(feature = "nsr_smt")]
mod backend_smt;

const LIGHT_PROOF_DOMAIN: &[u8] = b"ucf.nsr.proof.light.v1";
const NSR_INPUTS_DOMAIN: &[u8] = b"ucf.nsr.inputs.v1";
const NSR_ATOM_DOMAIN: &[u8] = b"ucf.nsr.reasoning.atom.v1";
const NSR_TRACE_COMMIT_DOMAIN: &[u8] = b"ucf.nsr.reasoning.trace.commit.v1";
const NSR_TRACE_LEAF_DOMAIN: &[u8] = b"ucf.nsr.reasoning.trace.leaf.v1";
const NSR_TRACE_ROOT_DOMAIN: &[u8] = b"ucf.nsr.reasoning.trace.root.v1";
const NSR_BACKEND_DOMAIN: &[u8] = b"ucf.nsr.backend.config.v1";
const NSR_ENGINE_DOMAIN: &[u8] = b"ucf.nsr.engine.v1";
const NSR_ENGINE_MVP_DOMAIN: &[u8] = b"ucf.nsr.engine.mvp.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NsrVerdict {
    Allow,
    Warn,
    Deny,
}

impl NsrVerdict {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Allow => 0,
            Self::Warn => 1,
            Self::Deny => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NsrReasonCode {
    PolicyConflict,
    UnsafeToolRequest,
    LowCoherence,
    HighRisk,
    InconsistentSelfState,
    StructuralCommitBlocked,
    Unknown(u16),
}

impl NsrReasonCode {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::PolicyConflict => 1,
            Self::UnsafeToolRequest => 2,
            Self::LowCoherence => 3,
            Self::HighRisk => 4,
            Self::InconsistentSelfState => 5,
            Self::StructuralCommitBlocked => 6,
            Self::Unknown(code) => code,
        }
    }

    pub fn token(self) -> String {
        match self {
            Self::PolicyConflict => "policy_conflict".to_string(),
            Self::UnsafeToolRequest => "unsafe_tool_request".to_string(),
            Self::LowCoherence => "low_coherence".to_string(),
            Self::HighRisk => "high_risk".to_string(),
            Self::InconsistentSelfState => "inconsistent_self_state".to_string(),
            Self::StructuralCommitBlocked => "structural_commit_blocked".to_string(),
            Self::Unknown(code) => format!("unknown_{code}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Mvp,
    Smt,
    Datalog,
}

impl BackendKind {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Mvp => 0,
            Self::Smt => 1,
            Self::Datalog => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendConfig {
    pub kind: BackendKind,
    pub commit: Digest32,
}

impl BackendConfig {
    pub fn new(kind: BackendKind) -> Self {
        let commit = digest_backend_config(kind);
        Self { kind, commit }
    }

    pub fn mvp() -> Self {
        Self::new(BackendKind::Mvp)
    }
}

pub trait SymbolicBackend {
    fn check(&mut self, facts: &[ReasoningAtom], inp: &NsrInputs) -> SymbolicResult;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicResult {
    pub ok: bool,
    pub contradictions: Vec<(u16, NsrReasonCode, u16)>,
    pub derived_atoms: Vec<ReasoningAtom>,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningAtom {
    pub key: u16,
    pub value: i16,
    pub commit: Digest32,
}

impl ReasoningAtom {
    pub fn new(key: u16, value: i16, input_commit: Digest32) -> Self {
        let commit = digest_reasoning_atom(key, value, input_commit);
        Self { key, value, commit }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningTrace {
    pub cycle_id: u64,
    pub inputs_commit: Digest32,
    pub rule_hits: Vec<(u16, NsrReasonCode, u16)>,
    pub derived_atoms: Vec<ReasoningAtom>,
    pub verdict: NsrVerdict,
    pub prev_commit: Option<Digest32>,
    pub trace_root: Digest32,
    pub commit: Digest32,
}

impl ReasoningTrace {
    pub fn new(
        cycle_id: u64,
        inputs_commit: Digest32,
        mut rule_hits: Vec<(u16, NsrReasonCode, u16)>,
        mut derived_atoms: Vec<ReasoningAtom>,
        verdict: NsrVerdict,
        prev_commit: Option<Digest32>,
    ) -> Self {
        sort_rule_hits(&mut rule_hits);
        sort_reasoning_atoms(&mut derived_atoms);
        let trace_root = digest_trace_root(
            inputs_commit,
            &rule_hits,
            &derived_atoms,
            verdict,
            prev_commit,
        );
        let commit = digest_reasoning_trace_commit(cycle_id, trace_root, verdict);
        Self {
            cycle_id,
            inputs_commit,
            rule_hits,
            derived_atoms,
            verdict,
            prev_commit,
            trace_root,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsrInputs {
    pub cycle_id: u64,
    pub phase_commit: Digest32,
    pub coherence_plv: u16,
    pub phi_proxy: u16,
    pub risk: u16,
    pub drift: u16,
    pub surprise: u16,
    pub ssm_salience: u16,
    pub ncde_energy: u16,
    pub spike_counts: Vec<(SpikeKind, u16)>,
    pub policy_decision_commit: Option<Digest32>,
    pub self_consistency_ok: Option<bool>,
    pub reasoning_atoms: Vec<ReasoningAtom>,
    pub causal_context: Option<Digest32>,
    pub commit: Digest32,
}

impl NsrInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        phase_commit: Digest32,
        coherence_plv: u16,
        phi_proxy: u16,
        risk: u16,
        drift: u16,
        surprise: u16,
        ssm_salience: u16,
        ncde_energy: u16,
        spike_counts: Vec<(SpikeKind, u16)>,
        policy_decision_commit: Option<Digest32>,
        self_consistency_ok: Option<bool>,
        reasoning_atoms: Vec<(u16, i16)>,
        causal_context: Option<Digest32>,
    ) -> Self {
        let atoms = reasoning_atoms
            .iter()
            .map(|(key, value)| ReasoningAtom {
                key: *key,
                value: *value,
                commit: Digest32::new([0u8; 32]),
            })
            .collect::<Vec<_>>();
        let mut inputs = Self {
            cycle_id,
            phase_commit,
            coherence_plv,
            phi_proxy,
            risk,
            drift,
            surprise,
            ssm_salience,
            ncde_energy,
            spike_counts,
            policy_decision_commit,
            self_consistency_ok,
            reasoning_atoms: atoms,
            causal_context,
            commit: Digest32::new([0u8; 32]),
        };
        inputs.commit = digest_nsr_inputs(&inputs);
        inputs.reasoning_atoms = reasoning_atoms
            .into_iter()
            .map(|(key, value)| ReasoningAtom::new(key, value, inputs.commit))
            .collect();
        inputs
    }
}

pub trait ConstraintEngine {
    fn evaluate(&mut self, inp: &NsrInputs) -> ReasoningTrace;
}

#[derive(Clone, Debug)]
pub struct NsrEngineMvp {
    pub thresholds: NsrThresholds,
    pub commit: Digest32,
}

impl NsrEngineMvp {
    pub fn new(thresholds: NsrThresholds) -> Self {
        let commit = digest_nsr_engine_mvp(&thresholds);
        Self { thresholds, commit }
    }

    pub fn evaluate_components(
        &self,
        inp: &NsrInputs,
    ) -> (
        Vec<(u16, NsrReasonCode, u16)>,
        Vec<ReasoningAtom>,
        NsrVerdict,
    ) {
        const COHERENCE_MIN: u16 = 3_000;
        const PHI_MIN: u16 = 3_500;
        const THREAT_SPIKE_HIGH: u16 = 5;
        const CAUSAL_SEVERITY_WARN: u16 = 7_800;
        const SEVERITY_WARN: u16 = 4_000;
        const SEVERITY_DENY: u16 = 9_000;

        let mut rule_hits = Vec::new();
        let mut derived_atoms = Vec::new();
        let mut verdict = NsrVerdict::Allow;
        let warn_threshold = self.thresholds.warn.min(10_000);
        let deny_threshold = self.thresholds.deny.min(10_000).max(warn_threshold);

        if inp.risk >= deny_threshold {
            rule_hits.push((1, NsrReasonCode::HighRisk, SEVERITY_DENY));
            derived_atoms.push(ReasoningAtom::new(1, inp.risk as i16, inp.commit));
            verdict = NsrVerdict::Deny;
        }

        if inp.coherence_plv < COHERENCE_MIN && inp.phi_proxy < PHI_MIN {
            rule_hits.push((2, NsrReasonCode::LowCoherence, SEVERITY_WARN));
            derived_atoms.push(ReasoningAtom::new(2, inp.coherence_plv as i16, inp.commit));
            derived_atoms.push(ReasoningAtom::new(3, inp.phi_proxy as i16, inp.commit));
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Warn;
            }
        }

        if inp.drift > deny_threshold && inp.self_consistency_ok == Some(false) {
            rule_hits.push((3, NsrReasonCode::InconsistentSelfState, SEVERITY_DENY));
            derived_atoms.push(ReasoningAtom::new(4, inp.drift as i16, inp.commit));
            verdict = NsrVerdict::Deny;
        }

        let threat_count = inp
            .spike_counts
            .iter()
            .find(|(kind, _)| *kind == SpikeKind::Threat)
            .map(|(_, count)| *count)
            .unwrap_or(0);
        if threat_count >= THREAT_SPIKE_HIGH && inp.coherence_plv < COHERENCE_MIN {
            rule_hits.push((4, NsrReasonCode::LowCoherence, SEVERITY_WARN));
            derived_atoms.push(ReasoningAtom::new(5, threat_count as i16, inp.commit));
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Warn;
            }
        }

        if inp.policy_decision_commit.is_some() {
            rule_hits.push((5, NsrReasonCode::PolicyConflict, SEVERITY_DENY));
            derived_atoms.push(ReasoningAtom::new(6, 1, inp.commit));
            verdict = NsrVerdict::Deny;
        }

        let causal_key = edge_key(CdeNodeId::Risk, CdeNodeId::OutputSuppression);
        if let Some(atom) = inp
            .reasoning_atoms
            .iter()
            .find(|atom| atom.key == causal_key && atom.value >= NSR_ATOM_MIN as i16)
        {
            rule_hits.push((6, NsrReasonCode::HighRisk, CAUSAL_SEVERITY_WARN));
            derived_atoms.push(atom.clone());
            if verdict == NsrVerdict::Allow {
                verdict = NsrVerdict::Warn;
            }
        }

        (rule_hits, derived_atoms, verdict)
    }
}

impl ConstraintEngine for NsrEngineMvp {
    fn evaluate(&mut self, inp: &NsrInputs) -> ReasoningTrace {
        let (rule_hits, derived_atoms, verdict) = self.evaluate_components(inp);
        ReasoningTrace::new(
            inp.cycle_id,
            inp.commit,
            rule_hits,
            derived_atoms,
            verdict,
            None,
        )
    }
}

#[derive(Clone, Debug)]
pub struct NsrEngine {
    pub mvp: NsrEngineMvp,
    pub backend: BackendConfig,
    pub prev_trace_commit: Option<Digest32>,
    pub commit: Digest32,
}

impl NsrEngine {
    pub fn new(thresholds: NsrThresholds) -> Self {
        Self::with_backend(thresholds, BackendConfig::mvp())
    }

    pub fn with_backend(thresholds: NsrThresholds, backend: BackendConfig) -> Self {
        let mvp = NsrEngineMvp::new(thresholds);
        let commit = digest_nsr_engine(&mvp, &backend);
        Self {
            mvp,
            backend,
            prev_trace_commit: None,
            commit,
        }
    }

    pub fn set_thresholds(&mut self, thresholds: NsrThresholds) {
        self.mvp = NsrEngineMvp::new(thresholds);
        self.commit = digest_nsr_engine(&self.mvp, &self.backend);
    }

    pub fn set_backend(&mut self, backend: BackendConfig) {
        self.backend = backend;
        self.commit = digest_nsr_engine(&self.mvp, &self.backend);
    }
}

impl ConstraintEngine for NsrEngine {
    fn evaluate(&mut self, inp: &NsrInputs) -> ReasoningTrace {
        const MAX_FACTS: usize = 16;
        const MAX_DERIVED: usize = 16;

        let (mut rule_hits, mut derived_atoms, mut verdict) = self.mvp.evaluate_components(inp);
        let mut facts = Vec::new();
        let mut input_atoms = inp.reasoning_atoms.clone();
        sort_reasoning_atoms(&mut input_atoms);
        for atom in input_atoms {
            if facts.len() >= MAX_FACTS {
                break;
            }
            facts.push(atom);
        }
        let mut mvp_atoms = derived_atoms.clone();
        sort_reasoning_atoms(&mut mvp_atoms);
        for atom in mvp_atoms {
            if facts.len() >= MAX_FACTS {
                break;
            }
            facts.push(atom);
        }

        let backend_result: Option<SymbolicResult> = match self.backend.kind {
            BackendKind::Mvp => None,
            BackendKind::Smt => {
                #[cfg(feature = "nsr_smt")]
                {
                    let mut backend = backend_smt::SmtBackend::new(self.mvp.thresholds);
                    Some(backend.check(&facts, inp))
                }
                #[cfg(not(feature = "nsr_smt"))]
                {
                    None
                }
            }
            BackendKind::Datalog => {
                #[cfg(feature = "nsr_datalog")]
                {
                    let mut backend = backend_datalog::DatalogBackend::new(self.mvp.thresholds);
                    Some(backend.check(&facts, inp))
                }
                #[cfg(not(feature = "nsr_datalog"))]
                {
                    None
                }
            }
        };

        if let Some(result) = backend_result {
            rule_hits.extend(result.contradictions);
            derived_atoms.extend(result.derived_atoms);
            if !result.ok {
                verdict = NsrVerdict::Deny;
            }
        }

        sort_reasoning_atoms(&mut derived_atoms);
        if derived_atoms.len() > MAX_DERIVED {
            derived_atoms.truncate(MAX_DERIVED);
        }

        let trace = ReasoningTrace::new(
            inp.cycle_id,
            inp.commit,
            rule_hits,
            derived_atoms,
            verdict,
            self.prev_trace_commit,
        );
        self.prev_trace_commit = Some(trace.commit);
        trace
    }
}

#[derive(Clone, Debug)]
pub struct MockConstraintEngine {
    verdict: NsrVerdict,
}

impl MockConstraintEngine {
    pub fn new(verdict: NsrVerdict) -> Self {
        Self { verdict }
    }
}

impl Default for MockConstraintEngine {
    fn default() -> Self {
        Self::new(NsrVerdict::Allow)
    }
}

impl ConstraintEngine for MockConstraintEngine {
    fn evaluate(&mut self, inp: &NsrInputs) -> ReasoningTrace {
        let (rule_hits, derived_atoms) = match self.verdict {
            NsrVerdict::Allow => (Vec::new(), Vec::new()),
            NsrVerdict::Warn => (
                vec![(1, NsrReasonCode::LowCoherence, 4_000)],
                vec![ReasoningAtom::new(7, 1, inp.commit)],
            ),
            NsrVerdict::Deny => (
                vec![(1, NsrReasonCode::HighRisk, 9_000)],
                vec![ReasoningAtom::new(8, 1, inp.commit)],
            ),
        };
        ReasoningTrace::new(
            inp.cycle_id,
            inp.commit,
            rule_hits,
            derived_atoms,
            self.verdict,
            None,
        )
    }
}

#[derive(Clone, Debug)]
pub struct ConstraintEngineAdapter<E> {
    backend: E,
}

impl<E> ConstraintEngineAdapter<E> {
    pub fn new(backend: E) -> Self {
        Self { backend }
    }
}

impl<E: ConstraintEngine> ConstraintEngine for ConstraintEngineAdapter<E> {
    fn evaluate(&mut self, inp: &NsrInputs) -> ReasoningTrace {
        self.backend.evaluate(inp)
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
pub struct NsrPolicyEngine {
    rule_checker: RuleChecker,
    constraint_checker: ConstraintChecker,
}

impl NsrPolicyEngine {
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
        return NsrVerdict::Allow;
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
    NsrVerdict::Allow
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

fn digest_nsr_inputs(inputs: &NsrInputs) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_INPUTS_DOMAIN);
    hasher.update(&inputs.cycle_id.to_be_bytes());
    hasher.update(inputs.phase_commit.as_bytes());
    hasher.update(&inputs.coherence_plv.to_be_bytes());
    hasher.update(&inputs.phi_proxy.to_be_bytes());
    hasher.update(&inputs.risk.to_be_bytes());
    hasher.update(&inputs.drift.to_be_bytes());
    hasher.update(&inputs.surprise.to_be_bytes());
    hasher.update(&inputs.ssm_salience.to_be_bytes());
    hasher.update(&inputs.ncde_energy.to_be_bytes());
    hasher.update(
        &u64::try_from(inputs.spike_counts.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for (kind, count) in &inputs.spike_counts {
        hasher.update(&kind.as_u16().to_be_bytes());
        hasher.update(&count.to_be_bytes());
    }
    match inputs.policy_decision_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match inputs.self_consistency_ok {
        Some(ok) => {
            hasher.update(&[1]);
            hasher.update(&[ok as u8]);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match inputs.causal_context {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(
        &u64::try_from(inputs.reasoning_atoms.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for atom in &inputs.reasoning_atoms {
        hasher.update(&atom.key.to_be_bytes());
        hasher.update(&atom.value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_reasoning_atom(key: u16, value: i16, input_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_ATOM_DOMAIN);
    hasher.update(&key.to_be_bytes());
    hasher.update(&value.to_be_bytes());
    hasher.update(input_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn sort_rule_hits(rule_hits: &mut [(u16, NsrReasonCode, u16)]) {
    rule_hits.sort_by(|(rule_a, reason_a, sev_a), (rule_b, reason_b, sev_b)| {
        rule_a
            .cmp(rule_b)
            .then_with(|| sev_b.cmp(sev_a))
            .then_with(|| reason_a.as_u16().cmp(&reason_b.as_u16()))
    });
}

fn sort_reasoning_atoms(atoms: &mut [ReasoningAtom]) {
    atoms.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then_with(|| a.commit.as_bytes().cmp(b.commit.as_bytes()))
    });
}

fn digest_rule_hit(rule_id: u16, reason: NsrReasonCode, severity: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_TRACE_LEAF_DOMAIN);
    hasher.update(&[1]);
    hasher.update(&rule_id.to_be_bytes());
    hasher.update(&reason.as_u16().to_be_bytes());
    hasher.update(&severity.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_verdict_leaf(verdict: NsrVerdict) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_TRACE_LEAF_DOMAIN);
    hasher.update(&[2]);
    hasher.update(&[verdict.as_u8()]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_trace_root(
    inputs_commit: Digest32,
    rule_hits: &[(u16, NsrReasonCode, u16)],
    derived_atoms: &[ReasoningAtom],
    verdict: NsrVerdict,
    prev_commit: Option<Digest32>,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_TRACE_ROOT_DOMAIN);
    hasher.update(inputs_commit.as_bytes());
    for (rule_id, reason, severity) in rule_hits {
        let leaf = digest_rule_hit(*rule_id, *reason, *severity);
        hasher.update(leaf.as_bytes());
    }
    for atom in derived_atoms {
        hasher.update(atom.commit.as_bytes());
    }
    hasher.update(digest_verdict_leaf(verdict).as_bytes());
    hasher.update(
        prev_commit
            .unwrap_or_else(|| Digest32::new([0u8; 32]))
            .as_bytes(),
    );
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_reasoning_trace_commit(
    cycle_id: u64,
    trace_root: Digest32,
    verdict: NsrVerdict,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_TRACE_COMMIT_DOMAIN);
    hasher.update(trace_root.as_bytes());
    hasher.update(&[verdict.as_u8()]);
    hasher.update(&cycle_id.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_backend_config(kind: BackendKind) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_BACKEND_DOMAIN);
    hasher.update(&[kind.as_u8()]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_nsr_engine(mvp: &NsrEngineMvp, backend: &BackendConfig) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_ENGINE_DOMAIN);
    hasher.update(mvp.commit.as_bytes());
    hasher.update(backend.commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_nsr_engine_mvp(thresholds: &NsrThresholds) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(NSR_ENGINE_MVP_DOMAIN);
    hasher.update(thresholds.commit.as_bytes());
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
        let engine = NsrPolicyEngine::new();
        let input = base_input();

        let first = engine.evaluate(&input);
        let second = engine.evaluate(&input);

        assert_eq!(first, second);
    }

    #[test]
    fn forbidden_intent_produces_deny() {
        let engine = NsrPolicyEngine::new();
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
        let engine = NsrPolicyEngine::new();
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
        let engine = NsrPolicyEngine::new();
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

        assert_eq!(report.verdict, NsrVerdict::Allow);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn causal_high_confidence_delta_denies() {
        let engine = NsrPolicyEngine::new();
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
        let engine = NsrPolicyEngine::new();
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
        let engine = NsrPolicyEngine::new();
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

        assert_eq!(report.verdict, NsrVerdict::Allow);
        assert!(report
            .violations
            .iter()
            .all(|violation| !is_causal_violation(&violation.code)));
    }

    fn base_nsr_inputs(cycle_id: u64) -> NsrInputs {
        NsrInputs::new(
            cycle_id,
            Digest32::new([1u8; 32]),
            6_000,
            6_000,
            1_000,
            1_000,
            1_000,
            2_000,
            2_000,
            Vec::new(),
            None,
            Some(true),
            Vec::new(),
            None,
        )
    }

    #[test]
    fn reasoning_trace_is_deterministic() {
        let inputs = base_nsr_inputs(1);
        let mut engine_a = NsrEngine::new(NsrThresholds::new(5_000, 8_000));
        let mut engine_b = NsrEngine::new(NsrThresholds::new(5_000, 8_000));

        let first = engine_a.evaluate(&inputs);
        let second = engine_b.evaluate(&inputs);

        assert_eq!(first.trace_root, second.trace_root);
        assert_eq!(first.commit, second.commit);
    }

    #[test]
    fn reasoning_trace_merkle_chain_links_prev_commit() {
        let mut engine = NsrEngine::new(NsrThresholds::new(5_000, 8_000));
        let first = engine.evaluate(&base_nsr_inputs(1));
        let second = engine.evaluate(&base_nsr_inputs(2));

        assert_eq!(second.prev_commit, Some(first.commit));
    }

    #[test]
    fn causal_atoms_raise_risk_warning() {
        let thresholds = NsrThresholds::default();
        let atom_key = edge_key(CdeNodeId::Risk, CdeNodeId::OutputSuppression);
        let inputs = NsrInputs::new(
            1,
            Digest32::new([2u8; 32]),
            6_500,
            6_500,
            1_000,
            1_000,
            1_000,
            2_000,
            2_000,
            Vec::new(),
            None,
            Some(true),
            vec![(atom_key, NSR_ATOM_MIN as i16)],
            None,
        );

        let mut engine = NsrEngineMvp::new(thresholds);
        let trace = engine.evaluate(&inputs);

        assert_eq!(trace.verdict, NsrVerdict::Warn);
        assert!(trace
            .rule_hits
            .iter()
            .any(|(_, code, _)| *code == NsrReasonCode::HighRisk));
    }

    #[test]
    fn hard_constraint_trace_is_deterministic() {
        let mut engine = NsrEngineMvp::new(NsrThresholds::new(5_000, 8_000));
        let inputs = base_nsr_inputs(1);

        let first = engine.evaluate(&inputs);
        let mut engine_second = NsrEngineMvp::new(NsrThresholds::new(5_000, 8_000));
        let second = engine_second.evaluate(&inputs);

        assert_eq!(first.commit, second.commit);
        assert_eq!(first, second);
    }

    #[test]
    fn high_risk_denies_in_mvp() {
        let mut engine = NsrEngineMvp::new(NsrThresholds::new(5_000, 8_000));
        let mut inputs = base_nsr_inputs(1);
        inputs.risk = 9_000;
        inputs.commit = digest_nsr_inputs(&inputs);

        let trace = engine.evaluate(&inputs);

        assert_eq!(trace.verdict, NsrVerdict::Deny);
        assert!(!trace.rule_hits.is_empty());
    }

    #[test]
    fn low_coherence_low_phi_warns_in_mvp() {
        let mut engine = NsrEngineMvp::new(NsrThresholds::new(5_000, 8_000));
        let mut inputs = base_nsr_inputs(1);
        inputs.coherence_plv = 2_000;
        inputs.phi_proxy = 2_000;
        inputs.commit = digest_nsr_inputs(&inputs);

        let trace = engine.evaluate(&inputs);

        assert_eq!(trace.verdict, NsrVerdict::Warn);
        assert!(!trace.rule_hits.is_empty());
    }

    #[cfg(feature = "nsr_smt")]
    #[test]
    fn smt_backend_compiles() {
        use super::backend_smt::SmtBackend;
        use super::SymbolicBackend;

        let mut backend = SmtBackend::new(NsrThresholds::new(5_000, 8_000));
        let inputs = base_nsr_inputs(1);

        let _result = backend.check(&[], &inputs);
    }

    #[cfg(feature = "nsr_datalog")]
    #[test]
    fn datalog_backend_compiles() {
        use super::backend_datalog::DatalogBackend;
        use super::SymbolicBackend;

        let mut backend = DatalogBackend::new(NsrThresholds::new(5_000, 8_000));
        let inputs = base_nsr_inputs(1);

        let _result = backend.check(&[], &inputs);
    }
}
