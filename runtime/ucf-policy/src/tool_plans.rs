use sha2::{Digest, Sha256};

use crate::{
    candidate::{EffectClass, ToolIntent},
    capability::{CapabilityDenyReason, CapabilityKind, CapabilityToken},
    gem::{GovernanceSignals, IssuanceTier, ToolRequest},
};

pub const MAX_PLAN_ARGS_BYTES: usize = 4 * 1024;
pub const MAX_PLAN_REQUIRED_CAPS: usize = 8;
pub const MAX_PLAN_STRING_CHARS: usize = 256;
pub const MAX_DENY_REASONS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolEffectClass {
    Read,
    Write,
    Compute,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPlanV1 {
    pub plan_id: [u8; 32],
    pub tool_id: String,
    pub tool_class_id: String,
    pub args_canonical: Vec<u8>,
    pub expected_effect_class: ToolEffectClass,
    pub required_caps: Vec<CapabilityKind>,
    pub created_from_candidate_id: u16,
    pub created_from_output_digest: [u8; 32],
    pub context_digest: [u8; 32],
    pub plan_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPlanRecord {
    pub plan_digest_prefix: [u8; 8],
    pub tool_id: String,
    pub tool_class_id: String,
    pub args_digest_prefix: [u8; 8],
    pub required_caps: Vec<String>,
    pub ebm_energy_q: Option<u16>,
    pub nsr_risk_q: Option<u16>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolIssueDecision {
    pub issued: bool,
    pub issued_token: Option<CapabilityToken>,
    pub deny_reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolIssueRecord {
    pub plan_digest_prefix: [u8; 8],
    pub issued_caps: Vec<[u8; 8]>,
    pub deny_reasons: Vec<String>,
    pub policy_graph_digest_prefix: [u8; 8],
    pub security_chain_digest_prefix: [u8; 8],
}

pub fn build_plan(
    intent: &ToolIntent,
    request: &ToolRequest,
    candidate_id: u16,
    output_digest: [u8; 32],
    policy_graph_digest: [u8; 32],
) -> ToolPlanV1 {
    let tool_id = request
        .kind
        .as_tag()
        .chars()
        .take(MAX_PLAN_STRING_CHARS)
        .collect::<String>();
    let tool_class_id = if matches!(request.kind, CapabilityKind::FileRead) {
        "memory_write".to_string()
    } else {
        request
            .target
            .chars()
            .take(MAX_PLAN_STRING_CHARS)
            .collect::<String>()
    };
    let args_canonical = canonical_args(request);
    let expected_effect_class = match intent.expected_effect {
        EffectClass::ReadOnly => ToolEffectClass::Read,
        EffectClass::Write | EffectClass::Network => ToolEffectClass::Write,
    };
    let mut required_caps = vec![request.kind.clone()];
    required_caps.truncate(MAX_PLAN_REQUIRED_CAPS);
    let plan_digest = digest_plan_bytes(
        &tool_id,
        &tool_class_id,
        &args_canonical,
        policy_graph_digest,
    );

    ToolPlanV1 {
        plan_id: plan_digest,
        tool_id,
        tool_class_id,
        args_canonical,
        expected_effect_class,
        required_caps,
        created_from_candidate_id: candidate_id,
        created_from_output_digest: output_digest,
        context_digest: request.evidence_chain_digest,
        plan_digest,
    }
}

fn digest_plan_bytes(
    tool_id: &str,
    tool_class_id: &str,
    args_canonical: &[u8],
    policy_graph_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"UCF:TOOL_PLAN:V1");
    hasher.update(tool_id.as_bytes());
    hasher.update([0]);
    hasher.update(tool_class_id.as_bytes());
    hasher.update([0]);
    hasher.update(args_canonical);
    hasher.update(policy_graph_digest);
    hasher.finalize().into()
}

pub fn canonical_args(req: &ToolRequest) -> Vec<u8> {
    if matches!(req.kind, CapabilityKind::FileRead) {
        let (root_id, rel_path) = req
            .target
            .split_once(':')
            .map(|(root, rel)| (root.trim(), rel.trim()))
            .unwrap_or(("", ""));
        let mut parts = [
            format!(
                "path={}",
                rel_path
                    .chars()
                    .take(MAX_PLAN_STRING_CHARS)
                    .collect::<String>()
            ),
            format!(
                "root_id={}",
                root_id
                    .chars()
                    .take(MAX_PLAN_STRING_CHARS)
                    .collect::<String>()
            ),
        ];
        parts.sort();
        let mut bytes = parts.join("\n").into_bytes();
        if bytes.len() > MAX_PLAN_ARGS_BYTES {
            bytes.truncate(MAX_PLAN_ARGS_BYTES);
        }
        return bytes;
    }
    let mut parts = [
        format!("bytes_in={}", req.payload_hint.bytes_in.unwrap_or(0)),
        format!("bytes_out={}", req.payload_hint.bytes_out.unwrap_or(0)),
        format!("decision_id={}", req.decision_id),
        format!(
            "target={}",
            req.target
                .chars()
                .take(MAX_PLAN_STRING_CHARS)
                .collect::<String>()
        ),
    ];
    parts.sort();
    let mut bytes = parts.join("\n").into_bytes();
    if bytes.len() > MAX_PLAN_ARGS_BYTES {
        bytes.truncate(MAX_PLAN_ARGS_BYTES);
    }
    bytes
}

pub fn make_plan_record(
    plan: &ToolPlanV1,
    ebm_energy_q: Option<u16>,
    nsr_risk_q: Option<u16>,
) -> ToolPlanRecord {
    let args_digest: [u8; 32] = Sha256::digest(&plan.args_canonical).into();
    ToolPlanRecord {
        plan_digest_prefix: prefix8(plan.plan_digest),
        tool_id: plan.tool_id.clone(),
        tool_class_id: plan.tool_class_id.clone(),
        args_digest_prefix: prefix8(args_digest),
        required_caps: plan
            .required_caps
            .iter()
            .map(|k| k.as_tag().to_string())
            .collect(),
        ebm_energy_q,
        nsr_risk_q,
    }
}

pub fn issue_for_plan(
    plan: &ToolPlanV1,
    request: &ToolRequest,
    token: Option<&CapabilityToken>,
    tier: IssuanceTier,
    signals: GovernanceSignals,
    policy_graph_digest: [u8; 32],
) -> (ToolIssueDecision, ToolIssueRecord) {
    let mut reasons = Vec::new();
    if signals.nsr_risk.is_some_and(|q| q > 0.70) {
        reasons.push("nsr_high_risk".to_string());
    }
    if signals
        .ebm_energy_mean_topk_q
        .is_some_and(|q| q.raw() > 45_000)
    {
        reasons.push("ebm_high_energy".to_string());
    }
    if matches!(tier, IssuanceTier::Tier3) {
        reasons.push("governor_tier_denied".to_string());
    }
    if plan.args_canonical != canonical_args(request) {
        reasons.push("plan_args_mismatch".to_string());
    }

    let issued_token = if reasons.is_empty() {
        token.cloned()
    } else {
        None
    };
    if issued_token.is_none() && reasons.is_empty() {
        reasons.push(format!("{:?}", CapabilityDenyReason::MissingToken));
    }
    reasons.truncate(MAX_DENY_REASONS);
    let issued = issued_token.is_some();
    let issued_caps = issued_token
        .as_ref()
        .map(|t| vec![prefix8(t.token_digest)])
        .unwrap_or_default();

    (
        ToolIssueDecision {
            issued,
            issued_token,
            deny_reasons: reasons.clone(),
        },
        ToolIssueRecord {
            plan_digest_prefix: prefix8(plan.plan_digest),
            issued_caps,
            deny_reasons: reasons,
            policy_graph_digest_prefix: prefix8(policy_graph_digest),
            security_chain_digest_prefix: prefix8(request.evidence_chain_digest),
        },
    )
}

pub fn prefix8(v: [u8; 32]) -> [u8; 8] {
    let mut p = [0u8; 8];
    p.copy_from_slice(&v[..8]);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        candidate::{PayloadHint, TargetRef},
        capability::{CapabilityLimits, CapabilityScope},
        gem::PayloadHint as GemPayloadHint,
    };

    #[test]
    fn canonical_digest_is_stable_across_arg_ordering() {
        let mut req = ToolRequest {
            id: 1,
            kind: CapabilityKind::FileRead,
            target: "x".to_string(),
            payload_hint: GemPayloadHint {
                bytes_out: Some(1),
                bytes_in: Some(2),
            },
            requested_at_t: 1,
            decision_id: 9,
            evidence_chain_digest: [3; 32],
            candidate_id: Some(1),
            tool_intent_digest: Some([4; 32]),
        };
        let a = canonical_args(&req);
        req.payload_hint = GemPayloadHint {
            bytes_in: Some(2),
            bytes_out: Some(1),
        };
        let b = canonical_args(&req);
        assert_eq!(a, b);
    }

    #[test]
    fn issue_tightens_on_risk_and_prevents_issue() {
        let intent = ToolIntent::new(
            CapabilityKind::FileRead,
            TargetRef::new(1, Some("x")),
            PayloadHint::default(),
            EffectClass::ReadOnly,
            false,
        );
        let req = ToolRequest {
            id: 1,
            kind: CapabilityKind::FileRead,
            target: "x".to_string(),
            payload_hint: GemPayloadHint::default(),
            requested_at_t: 1,
            decision_id: 9,
            evidence_chain_digest: [3; 32],
            candidate_id: Some(1),
            tool_intent_digest: Some([4; 32]),
        };
        let plan = build_plan(&intent, &req, 1, [5; 32], [7; 32]);
        let tok = CapabilityToken::issue(
            CapabilityKind::FileRead,
            CapabilityScope::Paths(vec!["x".into()]),
            CapabilityLimits {
                max_calls_per_window: 1,
                window_ticks: 1,
                max_bytes_out: None,
                max_bytes_in: None,
                max_concurrent: 1,
            },
            "t",
            1,
            Some(2),
        );
        let signals = GovernanceSignals::from_inputs(None, 1, Some(0.9), None);
        let (decision, _) = issue_for_plan(
            &plan,
            &req,
            Some(&tok),
            IssuanceTier::Tier0,
            signals,
            [9; 32],
        );
        assert!(!decision.issued);
        assert!(decision.deny_reasons.contains(&"nsr_high_risk".to_string()));
    }
}
