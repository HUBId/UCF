use blake3::Hasher;

use crate::{PolicyConstraintKindV1, PolicyFieldV1};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyContextV1 {
    pub context_id: &'static str,
    pub recursion_depth: u32,
    pub requests_gateway_action: bool,
    pub requests_policy_mutation: bool,
    pub requests_identity_finalization: bool,
    pub requests_evidence_archive_append: bool,
    pub requests_runtime_scheduler: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyViolationV1 {
    pub constraint_id: &'static str,
    pub kind: PolicyConstraintKindV1,
    pub detail: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyEvaluationCandidateStatusV1 {
    Pass,
    Constrained,
    RejectCandidate,
}

impl PolicyEvaluationCandidateStatusV1 {
    fn as_u8(self) -> u8 {
        match self {
            Self::Pass => 1,
            Self::Constrained => 2,
            Self::RejectCandidate => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyEvaluationCandidateV1 {
    pub status: PolicyEvaluationCandidateStatusV1,
    pub field_digest: [u8; 32],
    pub context_digest: [u8; 32],
    pub matched_constraints: Vec<&'static str>,
    pub violations: Vec<PolicyViolationV1>,
}

impl PolicyContextV1 {
    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ucf.policy_ecology.policy_context_v1\0");
        write_str(&mut out, self.context_id);
        write_u32(&mut out, self.recursion_depth);
        write_bool(&mut out, self.requests_gateway_action);
        write_bool(&mut out, self.requests_policy_mutation);
        write_bool(&mut out, self.requests_identity_finalization);
        write_bool(&mut out, self.requests_evidence_archive_append);
        write_bool(&mut out, self.requests_runtime_scheduler);
        out
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.deterministic_bytes());
        *hasher.finalize().as_bytes()
    }
}

impl PolicyEvaluationCandidateV1 {
    pub fn candidate_only(&self) -> bool {
        true
    }

    pub fn action_authority(&self) -> bool {
        false
    }

    pub fn gateway_authority(&self) -> bool {
        false
    }

    pub fn runtime_enforcement(&self) -> bool {
        false
    }

    pub fn policy_mutation(&self) -> bool {
        false
    }

    pub fn evidence_archive_authority(&self) -> bool {
        false
    }

    pub fn identity_authority(&self) -> bool {
        false
    }

    pub fn deterministic_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ucf.policy_ecology.policy_evaluation_candidate_v1\0");
        out.push(self.status.as_u8());
        out.extend_from_slice(&self.field_digest);
        out.extend_from_slice(&self.context_digest);
        write_u32(&mut out, self.matched_constraints.len() as u32);
        for constraint in &self.matched_constraints {
            write_str(&mut out, constraint);
        }
        write_u32(&mut out, self.violations.len() as u32);
        for violation in &self.violations {
            write_str(&mut out, violation.constraint_id);
            out.push(violation.kind as u8);
            write_str(&mut out, violation.detail);
        }
        out
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.deterministic_bytes());
        *hasher.finalize().as_bytes()
    }
}

pub fn evaluate_policy_constraints_v1(
    field: &PolicyFieldV1,
    ctx: &PolicyContextV1,
) -> PolicyEvaluationCandidateV1 {
    let mut matched_constraints = Vec::new();
    let mut violations = Vec::new();

    for constraint in &field.constraints {
        if !constraint.enabled {
            continue;
        }

        matched_constraints.push(constraint.id);

        match constraint.kind {
            PolicyConstraintKindV1::NoGatewayAction if ctx.requests_gateway_action => {
                violations.push(PolicyViolationV1 {
                    constraint_id: constraint.id,
                    kind: constraint.kind,
                    detail: "gateway action request is forbidden",
                });
            }
            PolicyConstraintKindV1::NoPolicyMutation if ctx.requests_policy_mutation => {
                violations.push(PolicyViolationV1 {
                    constraint_id: constraint.id,
                    kind: constraint.kind,
                    detail: "policy mutation request is forbidden",
                });
            }
            PolicyConstraintKindV1::NoIdentityFinalization
                if ctx.requests_identity_finalization =>
            {
                violations.push(PolicyViolationV1 {
                    constraint_id: constraint.id,
                    kind: constraint.kind,
                    detail: "identity finalization request is forbidden",
                });
            }
            PolicyConstraintKindV1::NoEvidenceArchiveAppend
                if ctx.requests_evidence_archive_append =>
            {
                violations.push(PolicyViolationV1 {
                    constraint_id: constraint.id,
                    kind: constraint.kind,
                    detail: "evidence archive append request is forbidden",
                });
            }
            PolicyConstraintKindV1::NoRuntimeScheduler if ctx.requests_runtime_scheduler => {
                violations.push(PolicyViolationV1 {
                    constraint_id: constraint.id,
                    kind: constraint.kind,
                    detail: "runtime scheduler request is forbidden",
                });
            }
            PolicyConstraintKindV1::MaxRecursionDepth => {
                if let Some(bound) = constraint.bound {
                    if ctx.recursion_depth > bound {
                        violations.push(PolicyViolationV1 {
                            constraint_id: constraint.id,
                            kind: constraint.kind,
                            detail: "recursion depth exceeds max bound",
                        });
                    }
                }
            }
            PolicyConstraintKindV1::ReadOnlyPolicyField if !field.is_read_only() => {
                violations.push(PolicyViolationV1 {
                    constraint_id: constraint.id,
                    kind: constraint.kind,
                    detail: "policy field must remain read-only",
                });
            }
            _ => {}
        }
    }

    let has_hard_reject = violations
        .iter()
        .any(|v| v.kind != PolicyConstraintKindV1::MaxRecursionDepth);
    let status = if has_hard_reject {
        PolicyEvaluationCandidateStatusV1::RejectCandidate
    } else if !violations.is_empty() {
        PolicyEvaluationCandidateStatusV1::Constrained
    } else {
        PolicyEvaluationCandidateStatusV1::Pass
    };

    PolicyEvaluationCandidateV1 {
        status,
        field_digest: field.digest(),
        context_digest: ctx.digest(),
        matched_constraints,
        violations,
    }
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn write_str(out: &mut Vec<u8>, value: &str) {
    write_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}
