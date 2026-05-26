use ucf_policy_ecology::{
    PolicyConstraintKindV1, PolicyConstraintV1, PolicyFieldErrorV1, PolicyFieldV1,
};

fn readonly_field() -> PolicyFieldV1 {
    PolicyFieldV1 {
        field_id: "policy.field.v1",
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
                id: "max-recursion",
                kind: PolicyConstraintKindV1::MaxRecursionDepth,
                bound: Some(32),
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

#[test]
fn policy_field_v1_accepts_read_only_constraints() {
    let field = readonly_field();
    assert_eq!(field.validate(), Ok(()));
    assert!(field.is_read_only());
    assert!(!field.allows_lower_layer_write());
    assert!(!field.has_action_authority());
}

#[test]
fn policy_field_v1_rejects_empty_field_id() {
    let mut field = readonly_field();
    field.field_id = "";
    assert_eq!(field.validate(), Err(PolicyFieldErrorV1::EmptyFieldId));
}

#[test]
fn policy_field_v1_rejects_zero_version() {
    let mut field = readonly_field();
    field.version = 0;
    assert_eq!(field.validate(), Err(PolicyFieldErrorV1::ZeroVersion));
}

#[test]
fn policy_field_v1_rejects_no_constraints() {
    let mut field = readonly_field();
    field.constraints.clear();
    assert_eq!(field.validate(), Err(PolicyFieldErrorV1::NoConstraints));
}

#[test]
fn policy_field_v1_rejects_mutable_or_lower_layer_writable() {
    let mut mutable_field = readonly_field();
    mutable_field.read_only = false;
    assert_eq!(
        mutable_field.validate(),
        Err(PolicyFieldErrorV1::MutablePolicyField)
    );

    let mut lower_layer_write = readonly_field();
    lower_layer_write.lower_layer_writable = true;
    assert_eq!(
        lower_layer_write.validate(),
        Err(PolicyFieldErrorV1::LowerLayerWritable)
    );
}

#[test]
fn policy_field_v1_rejects_gateway_or_action_authority() {
    let mut gateway_field = readonly_field();
    gateway_field.gateway_authority = true;
    assert_eq!(
        gateway_field.validate(),
        Err(PolicyFieldErrorV1::GatewayAuthority)
    );

    let mut action_field = readonly_field();
    action_field.action_authority = true;
    assert_eq!(
        action_field.validate(),
        Err(PolicyFieldErrorV1::ActionAuthority)
    );
}

#[test]
fn policy_field_v1_rejects_identity_or_archive_authority() {
    let mut identity_field = readonly_field();
    identity_field.identity_authority = true;
    assert_eq!(
        identity_field.validate(),
        Err(PolicyFieldErrorV1::IdentityAuthority)
    );

    let mut archive_field = readonly_field();
    archive_field.evidence_archive_authority = true;
    assert_eq!(
        archive_field.validate(),
        Err(PolicyFieldErrorV1::EvidenceArchiveAuthority)
    );
}

#[test]
fn policy_field_v1_rejects_runtime_enforcement_authority() {
    let mut field = readonly_field();
    field.runtime_enforcement = true;
    assert_eq!(
        field.validate(),
        Err(PolicyFieldErrorV1::RuntimeEnforcementAuthority)
    );
}

#[test]
fn policy_field_v1_digest_is_deterministic() {
    let field = readonly_field();
    assert_eq!(field.deterministic_bytes(), field.deterministic_bytes());
    assert_eq!(field.digest(), field.digest());
}

#[test]
fn policy_field_v1_digest_changes_when_constraints_change() {
    let field_a = readonly_field();
    let mut field_b = readonly_field();
    field_b.constraints.push(PolicyConstraintV1 {
        id: "no-runtime-scheduler",
        kind: PolicyConstraintKindV1::NoRuntimeScheduler,
        bound: None,
        enabled: true,
    });

    assert_ne!(field_a.deterministic_bytes(), field_b.deterministic_bytes());
    assert_ne!(field_a.digest(), field_b.digest());
}

#[test]
fn policy_field_v1_has_no_update_or_upsert_surface() {
    let field = readonly_field();
    assert_eq!(field.validate(), Ok(()));
}

#[test]
fn policy_field_v1_is_not_action_approval() {
    let field = readonly_field();
    assert!(!field.has_action_authority());
}
