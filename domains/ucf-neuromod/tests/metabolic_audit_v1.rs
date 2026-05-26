use ucf_neuromod::hormone_state_v1::{HormoneStateRawV1, HormoneStateV1};
use ucf_neuromod::hormone_update_v1::derive_hormone_modulation_output_v1;
use ucf_neuromod::metabolic_audit_v1::{
    verify_metabolic_candidates_v1, MetabolicAuditFailureV1, MetabolicAuditStatusV1,
};
use ucf_neuromod::replay_sleep_candidate_v1::derive_replay_sleep_candidates_v1;

fn state(raw: HormoneStateRawV1) -> HormoneStateV1 {
    HormoneStateV1::new_clamped(raw)
}

#[test]
fn metabolic_audit_passes_for_valid_candidates() {
    let state = HormoneStateV1::neutral();
    let modulation = derive_hormone_modulation_output_v1(&state);
    let candidates = derive_replay_sleep_candidates_v1(&modulation);
    let audit = verify_metabolic_candidates_v1(&state, &modulation, &candidates);

    assert_eq!(audit.status, MetabolicAuditStatusV1::Pass);
    assert!(audit.failures.is_empty());
    assert!(audit.is_pass());
}

#[test]
fn metabolic_audit_is_deterministic() {
    let state = HormoneStateV1::neutral();
    let modulation = derive_hormone_modulation_output_v1(&state);
    let candidates = derive_replay_sleep_candidates_v1(&modulation);

    let a = verify_metabolic_candidates_v1(&state, &modulation, &candidates);
    let b = verify_metabolic_candidates_v1(&state, &modulation, &candidates);

    assert_eq!(a, b);
    assert_eq!(a.deterministic_bytes(), b.deterministic_bytes());
    assert_eq!(a.digest(), b.digest());
    assert_eq!(a.audit_digest, a.digest());
}

#[test]
fn metabolic_audit_digest_changes_when_state_changes() {
    let base_state = HormoneStateV1::neutral();
    let changed_state = state(HormoneStateRawV1 {
        dopamine_like: 7_500,
        ..HormoneStateRawV1 {
            dopamine_like: 5_000,
            serotonin_like: 5_000,
            cortisol_like: 5_000,
            arousal_like: 5_000,
            sleep_pressure: 5_000,
            novelty_pressure: 5_000,
            stability_pressure: 5_000,
        }
    });

    let base_mod = derive_hormone_modulation_output_v1(&base_state);
    let changed_mod = derive_hormone_modulation_output_v1(&changed_state);
    let base_candidates = derive_replay_sleep_candidates_v1(&base_mod);
    let changed_candidates = derive_replay_sleep_candidates_v1(&changed_mod);

    let base_audit = verify_metabolic_candidates_v1(&base_state, &base_mod, &base_candidates);
    let changed_audit =
        verify_metabolic_candidates_v1(&changed_state, &changed_mod, &changed_candidates);

    assert_ne!(base_audit.state_digest, changed_audit.state_digest);
    assert_ne!(base_audit.audit_digest, changed_audit.audit_digest);
}

#[test]
fn metabolic_audit_reports_candidate_digests_or_bytes_stably() {
    let state = HormoneStateV1::neutral();
    let modulation = derive_hormone_modulation_output_v1(&state);
    let candidates = derive_replay_sleep_candidates_v1(&modulation);

    let first = verify_metabolic_candidates_v1(&state, &modulation, &candidates);
    let second = verify_metabolic_candidates_v1(&state, &modulation, &candidates);

    assert_eq!(first.candidate_digest, second.candidate_digest);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
}

#[test]
fn metabolic_audit_is_verify_only() {
    let state = HormoneStateV1::neutral();
    let modulation = derive_hormone_modulation_output_v1(&state);
    let candidates = derive_replay_sleep_candidates_v1(&modulation);

    let before_state = state;
    let before_modulation = modulation;
    let before_candidates = candidates;

    let audit = verify_metabolic_candidates_v1(&state, &modulation, &candidates);

    assert!(audit.metadata_only());
    assert_eq!(state, before_state);
    assert_eq!(modulation, before_modulation);
    assert_eq!(candidates, before_candidates);
}

#[test]
fn metabolic_audit_has_no_scheduler_replay_sleep_side_effects() {
    let state = HormoneStateV1::neutral();
    let modulation = derive_hormone_modulation_output_v1(&state);
    let candidates = derive_replay_sleep_candidates_v1(&modulation);
    let audit = verify_metabolic_candidates_v1(&state, &modulation, &candidates);

    assert!(!audit.runtime_authority);
    assert!(!audit.scheduler_authority);
    assert!(audit.advisory_only);
}

#[test]
fn metabolic_audit_has_no_gateway_policy_identity_archive_authority() {
    let state = HormoneStateV1::neutral();
    let modulation = derive_hormone_modulation_output_v1(&state);
    let candidates = derive_replay_sleep_candidates_v1(&modulation);
    let audit = verify_metabolic_candidates_v1(&state, &modulation, &candidates);

    assert!(!audit.gateway_visible);
    assert!(!audit.policy_mutation);
    assert!(!audit.identity_authority);
    assert!(!audit.evidence_archive_authority);
}

#[test]
fn metabolic_audit_failure_reasons_are_deterministically_ordered() {
    let state = HormoneStateV1::neutral();
    let modulation = derive_hormone_modulation_output_v1(&state);
    let mut candidates = derive_replay_sleep_candidates_v1(&modulation);

    candidates.replay.priority_hint = -1;
    candidates.sleep.pressure_hint = 20_000;

    let audit = verify_metabolic_candidates_v1(&state, &modulation, &candidates);

    assert_eq!(audit.status, MetabolicAuditStatusV1::Fail);
    assert_eq!(
        audit.failures,
        vec![
            MetabolicAuditFailureV1::ReplayCandidateOutOfBounds,
            MetabolicAuditFailureV1::SleepCandidateOutOfBounds,
        ]
    );
}

#[test]
fn metabolic_audit_detects_invalid_state_if_constructible() {
    let unrepresentable = HormoneStateV1::new(HormoneStateRawV1 {
        dopamine_like: -1,
        serotonin_like: 5_000,
        cortisol_like: 5_000,
        arousal_like: 5_000,
        sleep_pressure: 5_000,
        novelty_pressure: 5_000,
        stability_pressure: 5_000,
    });
    assert!(unrepresentable.is_err());
}

#[test]
fn metabolic_audit_uses_no_random_or_wallclock() {
    let state = state(HormoneStateRawV1 {
        dopamine_like: 3_333,
        serotonin_like: 4_444,
        cortisol_like: 5_555,
        arousal_like: 6_666,
        sleep_pressure: 7_777,
        novelty_pressure: 2_222,
        stability_pressure: 1_111,
    });
    let modulation = derive_hormone_modulation_output_v1(&state);
    let candidates = derive_replay_sleep_candidates_v1(&modulation);

    assert_eq!(
        verify_metabolic_candidates_v1(&state, &modulation, &candidates),
        verify_metabolic_candidates_v1(&state, &modulation, &candidates)
    );
}
