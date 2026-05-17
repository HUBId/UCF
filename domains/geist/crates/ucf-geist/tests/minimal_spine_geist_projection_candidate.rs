use ucf_geist::{
    build_geist_projection_candidate_from_sleep_audit,
    build_geist_projection_candidate_from_sleep_input, GeistProjectionError,
    MinimalSpineGeistProjectionInput, MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_SOURCE,
};
use ucf_sleep_coordinator::{
    build_sleep_applied_boundary_from_audit, build_sleep_plan_candidate_from_replay_boundary,
    verify_minimal_spine_sleep_plan_candidate, MinimalSpineSleepPlanAudit,
    MinimalSpineSleepPlanAuditStatus, MinimalSpineSleepPlanInput,
};
use ucf_types::Digest32;

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn sleep_input() -> MinimalSpineSleepPlanInput {
    MinimalSpineSleepPlanInput {
        replay_audit_digest: digest(11),
        replay_schedule_digest: digest(12),
        replay_applied_boundary_digest: Some(digest(13)),
        token_count: 7,
        source: "replay-audit-fixture",
    }
}

fn pass_sleep_audit() -> MinimalSpineSleepPlanAudit {
    let candidate = build_sleep_plan_candidate_from_replay_boundary(&sleep_input())
        .expect("valid sleep candidate");
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    audit
}

#[test]
fn geist_projection_candidate_from_sleep_boundary_is_deterministic() {
    let audit = pass_sleep_audit();
    let boundary = build_sleep_applied_boundary_from_audit(&audit).expect("valid sleep boundary");

    let first = build_geist_projection_candidate_from_sleep_audit(&audit, Some(&boundary))
        .expect("valid projection candidate");
    let second = build_geist_projection_candidate_from_sleep_audit(&audit, Some(&boundary))
        .expect("valid projection candidate");

    assert_eq!(first, second);
    assert_eq!(first.projection_digest, second.projection_digest);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.projection_digest, first.digest());
}

#[test]
fn geist_projection_candidate_changes_when_sleep_digest_changes() {
    let audit = pass_sleep_audit();
    let boundary = build_sleep_applied_boundary_from_audit(&audit).expect("valid sleep boundary");
    let baseline = build_geist_projection_candidate_from_sleep_audit(&audit, Some(&boundary))
        .expect("valid projection candidate");

    let changed_audit_digest =
        build_geist_projection_candidate_from_sleep_input(&MinimalSpineGeistProjectionInput {
            sleep_plan_audit_digest: digest(21),
            sleep_plan_candidate_digest: audit.sleep_plan_candidate_digest,
            sleep_applied_boundary_digest: Some(boundary.applied_boundary_digest),
            replay_audit_digest: audit.replay_audit_digest,
            replay_schedule_digest: audit.replay_schedule_digest,
            token_count: audit.token_count,
            source: audit.source,
        })
        .expect("valid projection input");
    assert_ne!(
        baseline.projection_digest,
        changed_audit_digest.projection_digest
    );

    let changed_candidate_digest =
        build_geist_projection_candidate_from_sleep_input(&MinimalSpineGeistProjectionInput {
            sleep_plan_audit_digest: audit.audit_digest,
            sleep_plan_candidate_digest: digest(22),
            sleep_applied_boundary_digest: Some(boundary.applied_boundary_digest),
            replay_audit_digest: audit.replay_audit_digest,
            replay_schedule_digest: audit.replay_schedule_digest,
            token_count: audit.token_count,
            source: audit.source,
        })
        .expect("valid projection input");
    assert_ne!(
        baseline.projection_digest,
        changed_candidate_digest.projection_digest
    );

    let changed_boundary_digest =
        build_geist_projection_candidate_from_sleep_input(&MinimalSpineGeistProjectionInput {
            sleep_plan_audit_digest: audit.audit_digest,
            sleep_plan_candidate_digest: audit.sleep_plan_candidate_digest,
            sleep_applied_boundary_digest: Some(digest(23)),
            replay_audit_digest: audit.replay_audit_digest,
            replay_schedule_digest: audit.replay_schedule_digest,
            token_count: audit.token_count,
            source: audit.source,
        })
        .expect("valid projection input");
    assert_ne!(
        baseline.projection_digest,
        changed_boundary_digest.projection_digest
    );

    let changed_token_count =
        build_geist_projection_candidate_from_sleep_input(&MinimalSpineGeistProjectionInput {
            sleep_plan_audit_digest: audit.audit_digest,
            sleep_plan_candidate_digest: audit.sleep_plan_candidate_digest,
            sleep_applied_boundary_digest: Some(boundary.applied_boundary_digest),
            replay_audit_digest: audit.replay_audit_digest,
            replay_schedule_digest: audit.replay_schedule_digest,
            token_count: audit.token_count + 1,
            source: audit.source,
        })
        .expect("valid projection input");
    assert_ne!(
        baseline.projection_digest,
        changed_token_count.projection_digest
    );
}

#[test]
fn geist_projection_candidate_rejects_failed_or_invalid_sleep_audit() {
    let invalid_input_result =
        build_sleep_plan_candidate_from_replay_boundary(&MinimalSpineSleepPlanInput {
            replay_audit_digest: Digest32::new([0; Digest32::LEN]),
            replay_schedule_digest: digest(12),
            replay_applied_boundary_digest: None,
            token_count: 7,
            source: "replay-audit-fixture",
        });
    assert!(invalid_input_result.is_err());

    let valid_candidate = build_sleep_plan_candidate_from_replay_boundary(&sleep_input())
        .expect("valid sleep candidate");
    let mut fail_audit = verify_minimal_spine_sleep_plan_candidate(&valid_candidate);
    fail_audit.status = MinimalSpineSleepPlanAuditStatus::Fail;
    fail_audit.audit_digest = fail_audit.digest();

    assert_eq!(
        build_geist_projection_candidate_from_sleep_audit(&fail_audit, None),
        Err(GeistProjectionError::AuditStatusNotPass)
    );

    assert_eq!(
        build_geist_projection_candidate_from_sleep_input(&MinimalSpineGeistProjectionInput {
            sleep_plan_audit_digest: Digest32::new([0; Digest32::LEN]),
            sleep_plan_candidate_digest: digest(31),
            sleep_applied_boundary_digest: None,
            replay_audit_digest: digest(32),
            replay_schedule_digest: digest(33),
            token_count: 1,
            source: "sleep-audit-fixture",
        }),
        Err(GeistProjectionError::ZeroSleepPlanAuditDigest)
    );
}

#[test]
fn geist_projection_candidate_validates_optional_sleep_boundary_match() {
    let audit = pass_sleep_audit();
    let boundary = build_sleep_applied_boundary_from_audit(&audit).expect("valid sleep boundary");

    let mut mismatched_audit_digest = boundary.clone();
    mismatched_audit_digest.sleep_plan_audit_digest = digest(41);
    mismatched_audit_digest.applied_boundary_digest = mismatched_audit_digest.digest();
    assert_eq!(
        build_geist_projection_candidate_from_sleep_audit(&audit, Some(&mismatched_audit_digest)),
        Err(GeistProjectionError::BoundaryAuditDigestMismatch)
    );

    let mut mismatched_candidate_digest = boundary.clone();
    mismatched_candidate_digest.sleep_plan_candidate_digest = digest(42);
    mismatched_candidate_digest.applied_boundary_digest = mismatched_candidate_digest.digest();
    assert_eq!(
        build_geist_projection_candidate_from_sleep_audit(
            &audit,
            Some(&mismatched_candidate_digest)
        ),
        Err(GeistProjectionError::BoundaryCandidateDigestMismatch)
    );

    let mut mismatched_token_count = boundary;
    mismatched_token_count.token_count += 1;
    mismatched_token_count.applied_boundary_digest = mismatched_token_count.digest();
    assert_eq!(
        build_geist_projection_candidate_from_sleep_audit(&audit, Some(&mismatched_token_count)),
        Err(GeistProjectionError::BoundaryTokenCountMismatch)
    );
}

#[test]
fn geist_projection_candidate_preserves_sleep_and_replay_provenance() {
    let audit = pass_sleep_audit();
    let boundary = build_sleep_applied_boundary_from_audit(&audit).expect("valid sleep boundary");
    let projection = build_geist_projection_candidate_from_sleep_audit(&audit, Some(&boundary))
        .expect("valid projection candidate");

    assert_eq!(projection.sleep_plan_audit_digest, audit.audit_digest);
    assert_eq!(
        projection.sleep_plan_candidate_digest,
        audit.sleep_plan_candidate_digest
    );
    assert_eq!(
        projection.sleep_applied_boundary_digest,
        Some(boundary.applied_boundary_digest)
    );
    assert_eq!(projection.replay_audit_digest, audit.replay_audit_digest);
    assert_eq!(
        projection.replay_schedule_digest,
        audit.replay_schedule_digest
    );
    assert_eq!(projection.token_count, audit.token_count);
    assert_eq!(projection.sleep_source, audit.source);
    assert_eq!(
        projection.source,
        MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_SOURCE
    );
}

#[test]
fn geist_projection_candidate_is_candidate_only() {
    let audit = pass_sleep_audit();
    let projection = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");

    assert!(projection.candidate_only);
    assert!(!projection.geist_applied);
    assert!(!projection.ism_written);
    assert!(!projection.identity_anchor);
    assert!(!projection.identity_finalized);
}

#[test]
fn geist_projection_candidate_does_not_mutate_policy_or_append() {
    let audit = pass_sleep_audit();
    let projection = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");

    assert!(!projection.policy_mutated);
    assert!(!projection.evidence_archive_appended);
}

#[test]
fn geist_projection_candidate_does_not_expose_gateway_or_runtime() {
    let audit = pass_sleep_audit();
    let projection = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");

    assert!(!projection.gateway_visible);
    assert!(!projection.geist_applied);
}

#[test]
fn geist_projection_candidate_does_not_activate_ism_or_identity() {
    let audit = pass_sleep_audit();
    let projection = build_geist_projection_candidate_from_sleep_audit(&audit, None)
        .expect("valid projection candidate");

    assert!(!projection.ism_written);
    assert!(!projection.identity_anchor);
    assert!(!projection.identity_finalized);
}
