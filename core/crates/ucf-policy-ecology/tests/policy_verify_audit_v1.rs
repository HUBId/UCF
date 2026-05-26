use ucf_policy_ecology::{
    evaluate_policy_constraints_v1, verify_policy_evaluation_v1, PolicyConstraintKindV1,
    PolicyConstraintV1, PolicyContextV1, PolicyEvaluationCandidateStatusV1, PolicyFieldV1,
    PolicyVerifyAuditFailureV1, PolicyVerifyAuditStatusV1,
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
fn policy_verify_audit_passes_for_safe_candidate() {
    let field = readonly_field();
    let candidate = evaluate_policy_constraints_v1(&field, &safe_context());
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert_eq!(audit.status, PolicyVerifyAuditStatusV1::Pass);
    assert!(audit.failures.is_empty());
}

#[test]
fn policy_verify_audit_preserves_constrained_status() {
    let field = readonly_field();
    let mut ctx = safe_context();
    ctx.recursion_depth = 4;
    let candidate = evaluate_policy_constraints_v1(&field, &ctx);
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert_eq!(
        candidate.status,
        PolicyEvaluationCandidateStatusV1::Constrained
    );
    assert_eq!(audit.status, PolicyVerifyAuditStatusV1::Constrained);
}

#[test]
fn policy_verify_audit_preserves_reject_candidate_status() {
    let field = readonly_field();
    let mut ctx = safe_context();
    ctx.requests_gateway_action = true;
    let candidate = evaluate_policy_constraints_v1(&field, &ctx);
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert_eq!(
        candidate.status,
        PolicyEvaluationCandidateStatusV1::RejectCandidate
    );
    assert_eq!(audit.status, PolicyVerifyAuditStatusV1::RejectCandidate);
}

#[test]
fn policy_verify_audit_digest_is_deterministic() {
    let field = readonly_field();
    let candidate = evaluate_policy_constraints_v1(&field, &safe_context());
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert_eq!(audit.deterministic_bytes(), audit.deterministic_bytes());
    assert_eq!(audit.digest(), audit.digest());
    assert_eq!(audit.audit_digest, audit.digest());
}

#[test]
fn policy_verify_audit_digest_changes_when_candidate_changes() {
    let field = readonly_field();
    let candidate_a = evaluate_policy_constraints_v1(&field, &safe_context());
    let mut ctx = safe_context();
    ctx.recursion_depth = 4;
    let candidate_b = evaluate_policy_constraints_v1(&field, &ctx);
    let audit_a = verify_policy_evaluation_v1(&field, &candidate_a);
    let audit_b = verify_policy_evaluation_v1(&field, &candidate_b);
    assert_ne!(audit_a.candidate_digest, audit_b.candidate_digest);
    assert_ne!(audit_a.audit_digest, audit_b.audit_digest);
}

#[test]
fn policy_verify_audit_detects_field_digest_mismatch_if_constructible() {
    let field = readonly_field();
    let mut candidate = evaluate_policy_constraints_v1(&field, &safe_context());
    candidate.field_digest = [7u8; 32];
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert_eq!(audit.status, PolicyVerifyAuditStatusV1::Fail);
    assert!(audit
        .failures
        .contains(&PolicyVerifyAuditFailureV1::FieldDigestMismatch));
}

#[test]
fn policy_verify_audit_is_verify_only_not_enforcement() {
    let field = readonly_field();
    let candidate = evaluate_policy_constraints_v1(&field, &safe_context());
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert!(audit.metadata_only());
    assert!(audit.verify_only());
}

#[test]
fn policy_verify_audit_has_no_action_gateway_runtime_authority() {
    let field = readonly_field();
    let candidate = evaluate_policy_constraints_v1(&field, &safe_context());
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert!(!audit.action_authority());
    assert!(!audit.gateway_authority());
    assert!(!audit.runtime_enforcement());
    assert!(!audit.policy_mutation());
}

#[test]
fn policy_verify_audit_has_no_identity_or_archive_authority() {
    let field = readonly_field();
    let candidate = evaluate_policy_constraints_v1(&field, &safe_context());
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert!(!audit.identity_authority());
    assert!(!audit.evidence_archive_authority());
}

#[test]
fn policy_verify_audit_does_not_mutate_field_or_candidate() {
    let field = readonly_field();
    let candidate = evaluate_policy_constraints_v1(&field, &safe_context());
    let before_field = field.clone();
    let before_candidate = candidate.clone();
    let _audit = verify_policy_evaluation_v1(&field, &candidate);
    assert_eq!(field, before_field);
    assert_eq!(candidate, before_candidate);
}

#[test]
fn policy_verify_audit_failure_order_is_deterministic() {
    let mut field = readonly_field();
    field.read_only = false;
    let mut candidate = evaluate_policy_constraints_v1(&readonly_field(), &safe_context());
    candidate.field_digest = [1u8; 32];
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert_eq!(audit.status, PolicyVerifyAuditStatusV1::Fail);
    assert_eq!(
        audit.failures,
        vec![
            PolicyVerifyAuditFailureV1::InvalidPolicyField,
            PolicyVerifyAuditFailureV1::FieldDigestMismatch,
        ]
    );
}

#[test]
fn policy_verify_audit_is_not_action_approval() {
    let field = readonly_field();
    let candidate = evaluate_policy_constraints_v1(&field, &safe_context());
    let audit = verify_policy_evaluation_v1(&field, &candidate);
    assert!(!audit.is_action_approval());
}
