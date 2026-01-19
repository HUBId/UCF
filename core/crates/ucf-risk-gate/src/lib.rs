#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_nsr_port::NsrReport;
use ucf_policy_ecology::{PolicyEcology, RiskDecision, RiskGateResult};
use ucf_sandbox::ControlFrameNormalized;
use ucf_scm_port::{digest_dag, CounterfactualQuery, Intervention, ScmDag};
use ucf_types::{AiOutput, Digest32, OutputChannel};

const BASE_RISK_THOUGHT: u16 = 400;
const BASE_RISK_SPEECH: u16 = 1200;
const SCM_UNSAFE_BAND: i32 = 100;
const SCM_UNSAFE_PENALTY: u16 = 2000;
const CDE_LOW_CONFIDENCE_THRESHOLD: u16 = 2000;
const CDE_LOW_CONFIDENCE_PENALTY: u16 = 800;

pub trait RiskGate {
    fn evaluate(
        &self,
        nsr: Option<&NsrReport>,
        scm: Option<&ScmDag>,
        output: &AiOutput,
        control_frame: &ControlFrameNormalized,
        cde_confidence: Option<u16>,
    ) -> RiskGateResult;
}

#[derive(Clone, Debug)]
pub struct PolicyRiskGate {
    policy: PolicyEcology,
}

impl PolicyRiskGate {
    pub fn new(policy: PolicyEcology) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> &PolicyEcology {
        &self.policy
    }
}

impl RiskGate for PolicyRiskGate {
    fn evaluate(
        &self,
        nsr: Option<&NsrReport>,
        scm: Option<&ScmDag>,
        output: &AiOutput,
        control_frame: &ControlFrameNormalized,
        cde_confidence: Option<u16>,
    ) -> RiskGateResult {
        evaluate_risk(
            &self.policy,
            nsr,
            scm,
            output,
            control_frame,
            cde_confidence,
        )
    }
}

pub fn evaluate_risk(
    policy: &PolicyEcology,
    nsr: Option<&NsrReport>,
    scm: Option<&ScmDag>,
    output: &AiOutput,
    control_frame: &ControlFrameNormalized,
    cde_confidence: Option<u16>,
) -> RiskGateResult {
    let risk_policy = policy.risk_policy();
    let mut risk = base_risk(output);
    let mut reasons = Vec::new();

    if let Some(confidence) = cde_confidence {
        if confidence < CDE_LOW_CONFIDENCE_THRESHOLD {
            risk = risk.saturating_add(CDE_LOW_CONFIDENCE_PENALTY);
            reasons.push(format!("cde_confidence_low:{confidence}"));
        }
    }

    if let Some(dag) = scm {
        for probe in fixed_counterfactuals(dag) {
            let predicted = probe_predicted(dag, &probe);
            if predicted.abs() >= SCM_UNSAFE_BAND {
                risk = risk.saturating_add(SCM_UNSAFE_PENALTY);
                reasons.push(format!("scm_unsafe_band:{predicted}"));
                break;
            }
        }
    }

    let mut decision = RiskDecision::Permit;
    if risk_policy.require_nsr_ok && nsr.is_some_and(|report| !report.ok) {
        decision = RiskDecision::Deny;
        reasons.push("nsr_not_ok".to_string());
    }

    if risk > risk_policy.max_risk {
        decision = RiskDecision::Deny;
        reasons.push("risk_above_max".to_string());
    }

    RiskGateResult {
        decision,
        risk,
        reasons,
        evidence: evidence_digest(policy, nsr, scm, output, control_frame, cde_confidence),
    }
}

pub fn digest_reasons(reasons: &[String]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.risk_gate.reasons.v1");
    hasher.update(&u64::try_from(reasons.len()).unwrap_or(0).to_be_bytes());
    for reason in reasons {
        hasher.update(&u64::try_from(reason.len()).unwrap_or(0).to_be_bytes());
        hasher.update(reason.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn base_risk(output: &AiOutput) -> u16 {
    match output.channel {
        OutputChannel::Thought => BASE_RISK_THOUGHT,
        OutputChannel::Speech => BASE_RISK_SPEECH,
    }
}

fn fixed_counterfactuals(dag: &ScmDag) -> Vec<CounterfactualQuery> {
    let mut probes = Vec::new();
    if dag.nodes.len() >= 2 {
        probes.push(CounterfactualQuery::new(
            vec![Intervention::new(dag.nodes[0].id, 3)],
            dag.nodes[1].id,
        ));
    }
    if dag.nodes.len() >= 3 {
        probes.push(CounterfactualQuery::new(
            vec![Intervention::new(dag.nodes[1].id, 7)],
            dag.nodes[2].id,
        ));
    }
    if dag.nodes.len() >= 4 {
        probes.push(CounterfactualQuery::new(
            vec![Intervention::new(dag.nodes[2].id, 11)],
            dag.nodes[3].id,
        ));
    }
    probes
}

fn probe_predicted(dag: &ScmDag, query: &CounterfactualQuery) -> i32 {
    let mut value = i32::try_from(dag.nodes.len()).unwrap_or(0);
    value = value.saturating_add(i32::try_from(query.target).unwrap_or(0));
    for intervention in &query.interventions {
        value = value.saturating_add(intervention.value);
        value = value.saturating_add(i32::try_from(intervention.node).unwrap_or(0));
    }
    value
}

fn evidence_digest(
    policy: &PolicyEcology,
    nsr: Option<&NsrReport>,
    scm: Option<&ScmDag>,
    output: &AiOutput,
    control_frame: &ControlFrameNormalized,
    cde_confidence: Option<u16>,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.risk_gate.v1");
    hasher.update(&policy.version().to_be_bytes());
    let risk_policy = policy.risk_policy();
    hasher.update(&risk_policy.max_risk.to_be_bytes());
    hasher.update(&[risk_policy.require_nsr_ok as u8]);
    hasher.update(control_frame.commitment().digest.as_bytes());
    hasher.update(output_digest(output).as_bytes());
    match nsr {
        Some(report) => {
            hasher.update(&[1]);
            hasher.update(nsr_digest(report).as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match scm {
        Some(dag) => {
            hasher.update(&[1]);
            hasher.update(digest_dag(dag).as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    if let Some(confidence) = cde_confidence {
        hasher.update(&[1]);
        hasher.update(&confidence.to_be_bytes());
    } else {
        hasher.update(&[0]);
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn output_digest(output: &AiOutput) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ai.output.v1");
    let channel_tag: u8 = match output.channel {
        OutputChannel::Thought => 0,
        OutputChannel::Speech => 1,
    };
    hasher.update(&[channel_tag]);
    hasher.update(&output.confidence.to_be_bytes());
    hasher.update(
        &u64::try_from(output.content.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    hasher.update(output.content.as_bytes());
    match output.rationale_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match output.integration_score {
        Some(score) => {
            hasher.update(&[1]);
            hasher.update(&score.to_be_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn nsr_digest(report: &NsrReport) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&[report.ok as u8]);
    hasher.update(
        &u64::try_from(report.violations.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for violation in &report.violations {
        hasher.update(violation.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    use ucf_policy_ecology::PolicyEcology;
    use ucf_sandbox::normalize;
    use ucf_scm_port::CausalNode;
    use ucf_types::v1::spec::ControlFrame;

    #[test]
    fn risk_gate_is_deterministic() {
        let policy = PolicyEcology::allow_all();
        let gate = PolicyRiskGate::new(policy);
        let nsr = NsrReport {
            ok: true,
            violations: Vec::new(),
        };
        let dag = ScmDag::new(
            vec![CausalNode::new(0, "a"), CausalNode::new(1, "b")],
            Vec::new(),
        );
        let output = AiOutput {
            channel: OutputChannel::Speech,
            content: "ok".to_string(),
            confidence: 900,
            rationale_commit: None,
            integration_score: None,
        };
        let cf = normalize(ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 1,
            decision: None,
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        });

        let result_a = gate.evaluate(Some(&nsr), Some(&dag), &output, &cf, Some(9000));
        let result_b = gate.evaluate(Some(&nsr), Some(&dag), &output, &cf, Some(9000));

        assert_eq!(result_a.decision, result_b.decision);
        assert_eq!(result_a.risk, result_b.risk);
        assert_eq!(result_a.evidence, result_b.evidence);
    }
}
