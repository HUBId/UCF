use ucf_policy_ecology::{
    evaluate_policy_constraints_v1, PolicyConstraintKindV1, PolicyConstraintV1, PolicyContextV1,
    PolicyEvaluationCandidateStatusV1, PolicyFieldV1,
};

fn readonly_field() -> PolicyFieldV1 {
    PolicyFieldV1 {
        field_id: "policy-field-v1",
        version: 1,
        constraints: vec![
            PolicyConstraintV1 {
                id: "readonly",
                kind: PolicyConstraintKindV1::ReadOnlyPolicyField,
                bound: None,
                enabled: true,
            },
            PolicyConstraintV1 {
                id: "no-gateway",
                kind: PolicyConstraintKindV1::NoGatewayAction,
                bound: None,
                enabled: true,
            },
            PolicyConstraintV1 {
                id: "no-mutation",
                kind: PolicyConstraintKindV1::NoPolicyMutation,
                bound: None,
                enabled: true,
            },
            PolicyConstraintV1 {
                id: "no-identity",
                kind: PolicyConstraintKindV1::NoIdentityFinalization,
                bound: None,
                enabled: true,
            },
            PolicyConstraintV1 {
                id: "no-archive",
                kind: PolicyConstraintKindV1::NoEvidenceArchiveAppend,
                bound: None,
                enabled: true,
            },
            PolicyConstraintV1 {
                id: "no-runtime",
                kind: PolicyConstraintKindV1::NoRuntimeScheduler,
                bound: None,
                enabled: true,
            },
            PolicyConstraintV1 {
                id: "max-recursion",
                kind: PolicyConstraintKindV1::MaxRecursionDepth,
                bound: Some(3),
                enabled: true,
            },
        ],
        read_only: true,
        lower_layer_writable: false,
        gateway_authority: false,
        action_authority: false,
        identity_authority: false,
        evidence_archive_authority: false,
        runtime_enforcement: false,
    }
}

fn safe_context() -> PolicyContextV1 {
    PolicyContextV1 {
        context_id: "ctx-1",
        recursion_depth: 1,
        requests_gateway_action: false,
        requests_policy_mutation: false,
        requests_identity_finalization: false,
        requests_evidence_archive_append: false,
        requests_runtime_scheduler: false,
    }
}

#[test]
fn policy_evaluation_candidate_passes_safe_context() {
    let result = evaluate_policy_constraints_v1(&readonly_field(), &safe_context());
    assert_eq!(result.status, PolicyEvaluationCandidateStatusV1::Pass);
    assert!(result.violations.is_empty());
}

#[test]
fn policy_evaluation_candidate_rejects_gateway_action_request() {
    let mut ctx = safe_context();
    ctx.requests_gateway_action = true;
    let result = evaluate_policy_constraints_v1(&readonly_field(), &ctx);
    assert_eq!(
        result.status,
        PolicyEvaluationCandidateStatusV1::RejectCandidate
    );
}

#[test]
fn policy_evaluation_candidate_rejects_policy_mutation_request() {
    let mut ctx = safe_context();
    ctx.requests_policy_mutation = true;
    let result = evaluate_policy_constraints_v1(&readonly_field(), &ctx);
    assert_eq!(
        result.status,
        PolicyEvaluationCandidateStatusV1::RejectCandidate
    );
}

#[test]
fn policy_evaluation_candidate_rejects_identity_finalization_request() {
    let mut ctx = safe_context();
    ctx.requests_identity_finalization = true;
    let result = evaluate_policy_constraints_v1(&readonly_field(), &ctx);
    assert_eq!(
        result.status,
        PolicyEvaluationCandidateStatusV1::RejectCandidate
    );
}

#[test]
fn policy_evaluation_candidate_rejects_evidence_archive_append_request() {
    let mut ctx = safe_context();
    ctx.requests_evidence_archive_append = true;
    let result = evaluate_policy_constraints_v1(&readonly_field(), &ctx);
    assert_eq!(
        result.status,
        PolicyEvaluationCandidateStatusV1::RejectCandidate
    );
}

#[test]
fn policy_evaluation_candidate_rejects_runtime_scheduler_request() {
    let mut ctx = safe_context();
    ctx.requests_runtime_scheduler = true;
    let result = evaluate_policy_constraints_v1(&readonly_field(), &ctx);
    assert_eq!(
        result.status,
        PolicyEvaluationCandidateStatusV1::RejectCandidate
    );
}

#[test]
fn policy_evaluation_candidate_respects_max_recursion_depth() {
    let mut ctx = safe_context();
    ctx.recursion_depth = 4;
    let result = evaluate_policy_constraints_v1(&readonly_field(), &ctx);
    assert_eq!(
        result.status,
        PolicyEvaluationCandidateStatusV1::Constrained
    );
}

#[test]
fn policy_evaluation_candidate_digest_is_deterministic() {
    let field = readonly_field();
    let ctx = safe_context();
    let result = evaluate_policy_constraints_v1(&field, &ctx);
    assert_eq!(result.deterministic_bytes(), result.deterministic_bytes());
    assert_eq!(result.digest(), result.digest());
}

#[test]
fn policy_evaluation_candidate_digest_changes_when_context_changes() {
    let field = readonly_field();
    let result_a = evaluate_policy_constraints_v1(&field, &safe_context());
    let mut ctx_b = safe_context();
    ctx_b.recursion_depth = 2;
    let result_b = evaluate_policy_constraints_v1(&field, &ctx_b);
    assert_ne!(result_a.context_digest, result_b.context_digest);
    assert_ne!(result_a.digest(), result_b.digest());
}

#[test]
fn policy_evaluation_candidate_is_candidate_only_not_action_approval() {
    let result = evaluate_policy_constraints_v1(&readonly_field(), &safe_context());
    assert!(result.candidate_only());
    assert!(!result.action_authority());
}

#[test]
fn policy_evaluation_does_not_mutate_policy_field() {
    let field = readonly_field();
    let before = field.clone();
    let _ = evaluate_policy_constraints_v1(&field, &safe_context());
    assert_eq!(field, before);
}

#[test]
fn policy_evaluation_has_no_gateway_runtime_archive_identity_authority() {
    let result = evaluate_policy_constraints_v1(&readonly_field(), &safe_context());
    assert!(!result.gateway_authority());
    assert!(!result.runtime_enforcement());
    assert!(!result.evidence_archive_authority());
    assert!(!result.identity_authority());
    assert!(!result.policy_mutation());
}
