use ucf_geist::{
    build_geist_projection_candidate_from_sleep_audit,
    build_geist_projection_candidate_from_sleep_input,
    build_ism_candidate_boundary_from_geist_audit, verify_minimal_spine_geist_projection_candidate,
    GeistProjectionError, IsmCandidateBoundaryError, MinimalSpineGeistProjectionAudit,
    MinimalSpineGeistProjectionAuditFailure, MinimalSpineGeistProjectionAuditStatus,
    MinimalSpineGeistProjectionCandidate, MinimalSpineGeistProjectionInput,
    MinimalSpineIsmCandidateBoundary, MINIMAL_SPINE_GEIST_PROJECTION_AUDIT_SOURCE,
    MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_SOURCE, MINIMAL_SPINE_ISM_CANDIDATE_BOUNDARY_SOURCE,
};
use ucf_sleep_coordinator::{
    build_sleep_applied_boundary_from_audit, build_sleep_plan_candidate_from_replay_boundary,
    verify_minimal_spine_sleep_plan_candidate, MinimalSpineSleepAppliedBoundary,
    MinimalSpineSleepPlanAudit, MinimalSpineSleepPlanAuditStatus, MinimalSpineSleepPlanInput,
};
use ucf_types::Digest32;

const REPLAY_SOURCE: &str = "minimal_spine_replay_audit_fixture_for_geist_ism_e2e";

#[derive(Clone, Debug)]
struct GeistIsmPipelineRun {
    sleep_audit: MinimalSpineSleepPlanAudit,
    sleep_boundary: MinimalSpineSleepAppliedBoundary,
    candidate: MinimalSpineGeistProjectionCandidate,
    audit: MinimalSpineGeistProjectionAudit,
    boundary: MinimalSpineIsmCandidateBoundary,
}

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn zero_digest() -> Digest32 {
    Digest32::new([0; Digest32::LEN])
}

fn sleep_input() -> MinimalSpineSleepPlanInput {
    MinimalSpineSleepPlanInput {
        replay_audit_digest: digest(61),
        replay_schedule_digest: digest(62),
        replay_applied_boundary_digest: Some(digest(63)),
        token_count: 13,
        source: REPLAY_SOURCE,
    }
}

fn alternate_sleep_input() -> MinimalSpineSleepPlanInput {
    MinimalSpineSleepPlanInput {
        replay_audit_digest: digest(71),
        replay_schedule_digest: digest(72),
        replay_applied_boundary_digest: Some(digest(73)),
        token_count: 17,
        source: "minimal_spine_replay_audit_fixture_for_geist_ism_mismatch",
    }
}

fn pass_sleep_audit_from_input(input: &MinimalSpineSleepPlanInput) -> MinimalSpineSleepPlanAudit {
    let candidate = build_sleep_plan_candidate_from_replay_boundary(input)
        .expect("valid sleep plan candidate fixture");
    let audit = verify_minimal_spine_sleep_plan_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineSleepPlanAuditStatus::Pass);
    assert!(audit.failure_reasons.is_empty());
    assert_eq!(audit.audit_digest, audit.digest());
    audit
}

fn run_pipeline() -> GeistIsmPipelineRun {
    let sleep_audit = pass_sleep_audit_from_input(&sleep_input());
    let sleep_boundary =
        build_sleep_applied_boundary_from_audit(&sleep_audit).expect("valid sleep boundary");

    let candidate =
        build_geist_projection_candidate_from_sleep_audit(&sleep_audit, Some(&sleep_boundary))
            .expect("valid Geist projection candidate");
    assert_candidate_boundary(&candidate);

    let audit = verify_minimal_spine_geist_projection_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineGeistProjectionAuditStatus::Pass);
    assert_audit_boundary(&audit);

    let boundary = build_ism_candidate_boundary_from_geist_audit(&audit)
        .expect("PASS audit builds local ISM candidate boundary");
    assert_ism_boundary(&boundary);

    GeistIsmPipelineRun {
        sleep_audit,
        sleep_boundary,
        candidate,
        audit,
        boundary,
    }
}

fn assert_candidate_boundary(candidate: &MinimalSpineGeistProjectionCandidate) {
    assert_eq!(
        candidate.source,
        MINIMAL_SPINE_GEIST_PROJECTION_CANDIDATE_SOURCE
    );
    assert_eq!(candidate.projection_digest, candidate.digest());
    assert!(candidate.candidate_only);
    assert!(!candidate.geist_applied);
    assert!(!candidate.ism_written);
    assert!(!candidate.identity_anchor);
    assert!(!candidate.identity_finalized);
    assert!(!candidate.policy_mutated);
    assert!(!candidate.evidence_archive_appended);
    assert!(!candidate.gateway_visible);
}

fn assert_audit_boundary(audit: &MinimalSpineGeistProjectionAudit) {
    assert_eq!(audit.source, MINIMAL_SPINE_GEIST_PROJECTION_AUDIT_SOURCE);
    assert_eq!(audit.audit_digest, audit.digest());
    assert_eq!(audit.status, MinimalSpineGeistProjectionAuditStatus::Pass);
    assert!(audit.failure_reasons.is_empty());
    assert!(audit.candidate_only);
    assert!(!audit.geist_applied);
    assert!(!audit.ism_written);
    assert!(!audit.identity_anchor);
    assert!(!audit.identity_finalized);
    assert!(!audit.policy_mutated);
    assert!(!audit.evidence_archive_appended);
    assert!(!audit.gateway_visible);
}

fn assert_ism_boundary(boundary: &MinimalSpineIsmCandidateBoundary) {
    assert_eq!(boundary.source, MINIMAL_SPINE_ISM_CANDIDATE_BOUNDARY_SOURCE);
    assert_eq!(boundary.ism_candidate_digest, boundary.digest());
    assert!(boundary.ism_candidate_only);
    assert!(!boundary.ism_written);
    assert!(!boundary.ism_upserted);
    assert!(!boundary.identity_anchor);
    assert!(!boundary.identity_finalized);
    assert!(!boundary.memory_stabilized);
    assert!(!boundary.policy_mutated);
    assert!(!boundary.evidence_archive_appended);
    assert!(!boundary.gateway_visible);
}

fn assert_no_forbidden_runtime_marker(value: &str) {
    for marker in [
        "GeistKernel::ingest_macro",
        "ingest_macro",
        "IsmStore",
        "InMemoryIsm",
        "upsert_anchor",
        "IdentityAnchor",
        "IdentityFinalization",
        "identity_anchor=true",
        "identity_finalized=true",
        "memory_stabilized=true",
        "policy_mutated=true",
        "PolicyMutation",
        "EvidenceStore",
        "ArchiveStore",
        "append_evidence",
        "append_archive",
        "evidence_archive_appended=true",
        "GatewayWrite",
        "GatewayAction",
        "gateway_visible=true",
        "runtime_apply=true",
        "RealCompute",
    ] {
        assert!(
            !value.contains(marker),
            "bounded Geist/ISM E2E path introduced forbidden marker {marker}"
        );
    }
}

fn assert_no_forbidden_runtime_byte_marker(bytes: &[u8]) {
    let rendered = String::from_utf8_lossy(bytes);
    assert_no_forbidden_runtime_marker(&rendered);
}

#[test]
fn geist_ism_pipeline_e2e_is_deterministic_across_fresh_runs() {
    let first = run_pipeline();
    let second = run_pipeline();

    assert_eq!(first.candidate, second.candidate);
    assert_eq!(first.audit, second.audit);
    assert_eq!(first.boundary, second.boundary);
    assert_eq!(
        first.candidate.projection_digest,
        second.candidate.projection_digest
    );
    assert_eq!(
        first.candidate.deterministic_bytes(),
        second.candidate.deterministic_bytes()
    );
    assert_eq!(first.audit.audit_digest, second.audit.audit_digest);
    assert_eq!(
        first.audit.deterministic_bytes(),
        second.audit.deterministic_bytes()
    );
    assert_eq!(
        first.boundary.ism_candidate_digest,
        second.boundary.ism_candidate_digest
    );
    assert_eq!(
        first.boundary.deterministic_bytes(),
        second.boundary.deterministic_bytes()
    );
}

#[test]
fn geist_ism_pipeline_preserves_sleep_to_ism_provenance() {
    let run = run_pipeline();

    assert_eq!(
        run.candidate.sleep_plan_audit_digest,
        run.sleep_audit.audit_digest
    );
    assert_eq!(
        run.candidate.sleep_plan_candidate_digest,
        run.sleep_audit.sleep_plan_candidate_digest
    );
    assert_eq!(
        run.candidate.sleep_applied_boundary_digest,
        Some(run.sleep_boundary.applied_boundary_digest)
    );
    assert_eq!(run.candidate.replay_audit_digest, digest(61));
    assert_eq!(run.candidate.replay_schedule_digest, digest(62));
    assert_eq!(run.candidate.token_count, 13);

    assert_eq!(run.audit.projection_digest, run.candidate.projection_digest);
    assert_eq!(
        run.audit.recomputed_projection_digest,
        run.candidate.digest()
    );
    assert_eq!(
        run.audit.sleep_plan_audit_digest,
        run.sleep_audit.audit_digest
    );
    assert_eq!(
        run.audit.sleep_plan_candidate_digest,
        run.sleep_audit.sleep_plan_candidate_digest
    );
    assert_eq!(
        run.audit.sleep_applied_boundary_digest,
        Some(run.sleep_boundary.applied_boundary_digest)
    );
    assert_eq!(run.audit.replay_audit_digest, digest(61));
    assert_eq!(run.audit.replay_schedule_digest, digest(62));
    assert_eq!(run.audit.token_count, 13);

    assert_eq!(
        run.boundary.geist_projection_audit_digest,
        run.audit.audit_digest
    );
    assert_eq!(
        run.boundary.geist_projection_digest,
        run.candidate.projection_digest
    );
    assert_eq!(
        run.boundary.sleep_plan_audit_digest,
        run.sleep_audit.audit_digest
    );
    assert_eq!(
        run.boundary.sleep_plan_candidate_digest,
        run.sleep_audit.sleep_plan_candidate_digest
    );
    assert_eq!(
        run.boundary.sleep_applied_boundary_digest,
        Some(run.sleep_boundary.applied_boundary_digest)
    );
    assert_eq!(run.boundary.replay_audit_digest, digest(61));
    assert_eq!(run.boundary.replay_schedule_digest, digest(62));
    assert_eq!(run.boundary.token_count, 13);
    assert_eq!(run.boundary.audit_source, run.audit.source);
    assert_eq!(run.boundary.candidate_source, run.candidate.source);
    assert_eq!(run.boundary.sleep_source, run.sleep_audit.source);
}

#[test]
fn geist_ism_pipeline_requires_pass_audit_before_ism_boundary() {
    let run = run_pipeline();

    assert!(build_ism_candidate_boundary_from_geist_audit(&run.audit).is_ok());

    let mut tampered_candidate = run.candidate.clone();
    tampered_candidate.geist_applied = true;
    let fail_audit = verify_minimal_spine_geist_projection_candidate(&tampered_candidate);
    assert_eq!(
        fail_audit.status,
        MinimalSpineGeistProjectionAuditStatus::Fail
    );
    assert!(fail_audit
        .failure_reasons
        .contains(&MinimalSpineGeistProjectionAuditFailure::GeistAppliedFlagSet));
    assert_eq!(
        build_ism_candidate_boundary_from_geist_audit(&fail_audit),
        Err(IsmCandidateBoundaryError::AuditStatusNotPass)
    );
}

#[test]
fn geist_ism_pipeline_has_no_runtime_identity_policy_archive_side_effects() {
    let run = run_pipeline();

    assert_candidate_boundary(&run.candidate);
    assert_audit_boundary(&run.audit);
    assert_ism_boundary(&run.boundary);

    for bytes in [
        run.candidate.deterministic_bytes(),
        run.audit.deterministic_bytes(),
        run.boundary.deterministic_bytes(),
    ] {
        assert_no_forbidden_runtime_byte_marker(&bytes);
    }
}

#[test]
fn geist_ism_pipeline_rejects_invalid_inputs() {
    let valid_sleep_audit = pass_sleep_audit_from_input(&sleep_input());
    let valid_sleep_boundary =
        build_sleep_applied_boundary_from_audit(&valid_sleep_audit).expect("valid sleep boundary");

    let zero_digest_input = MinimalSpineGeistProjectionInput {
        sleep_plan_audit_digest: zero_digest(),
        sleep_plan_candidate_digest: valid_sleep_audit.sleep_plan_candidate_digest,
        sleep_applied_boundary_digest: Some(valid_sleep_boundary.applied_boundary_digest),
        replay_audit_digest: valid_sleep_audit.replay_audit_digest,
        replay_schedule_digest: valid_sleep_audit.replay_schedule_digest,
        token_count: valid_sleep_audit.token_count,
        source: valid_sleep_audit.source,
    };
    assert_eq!(
        build_geist_projection_candidate_from_sleep_input(&zero_digest_input),
        Err(GeistProjectionError::ZeroSleepPlanAuditDigest)
    );

    let mismatched_sleep_audit = pass_sleep_audit_from_input(&alternate_sleep_input());
    let mismatched_sleep_boundary =
        build_sleep_applied_boundary_from_audit(&mismatched_sleep_audit)
            .expect("mismatched sleep boundary");
    assert_eq!(
        build_geist_projection_candidate_from_sleep_audit(
            &valid_sleep_audit,
            Some(&mismatched_sleep_boundary)
        ),
        Err(GeistProjectionError::BoundaryAuditDigestMismatch)
    );

    let candidate = build_geist_projection_candidate_from_sleep_audit(
        &valid_sleep_audit,
        Some(&valid_sleep_boundary),
    )
    .expect("valid projection candidate");
    let mut audit = verify_minimal_spine_geist_projection_candidate(&candidate);
    assert_eq!(audit.status, MinimalSpineGeistProjectionAuditStatus::Pass);
    audit.audit_digest = digest(99);
    assert_eq!(
        build_ism_candidate_boundary_from_geist_audit(&audit),
        Err(IsmCandidateBoundaryError::AuditDigestMismatch)
    );
}

#[test]
fn geist_ism_pipeline_does_not_append_or_activate_runtime() {
    let run = run_pipeline();

    assert_no_forbidden_runtime_marker(run.candidate.source);
    assert_no_forbidden_runtime_marker(run.candidate.sleep_source);
    assert_no_forbidden_runtime_marker(run.audit.source);
    assert_no_forbidden_runtime_marker(run.audit.candidate_source);
    assert_no_forbidden_runtime_marker(run.audit.sleep_source);
    assert_no_forbidden_runtime_marker(run.boundary.source);
    assert_no_forbidden_runtime_marker(run.boundary.audit_source);
    assert_no_forbidden_runtime_marker(run.boundary.candidate_source);
    assert_no_forbidden_runtime_marker(run.boundary.sleep_source);

    assert!(!run.candidate.geist_applied);
    assert!(!run.audit.geist_applied);
    assert!(!run.boundary.ism_written);
    assert!(!run.boundary.ism_upserted);
    assert!(!run.boundary.evidence_archive_appended);
    assert!(!run.boundary.gateway_visible);
}
