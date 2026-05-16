use ucf_replay::{
    build_replay_token_from_minimal_spine_input, MinimalSpineReplayTokenInput, ReplayError,
    MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_types::consolidation::MilestoneTier;
use ucf_types::Digest32;

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn input() -> MinimalSpineReplayTokenInput {
    MinimalSpineReplayTokenInput {
        macro_candidate_digest: digest(1),
        macro_milestone_digest: digest(2),
        meso_aggregation_digest: digest(3),
        macro_finalization_digest: digest(4),
        meso_count: 3,
        source: MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
    }
}

#[test]
fn replay_token_builder_from_consolidation_artifact_is_deterministic() {
    let input = input();

    let first = build_replay_token_from_minimal_spine_input(&input).expect("token output");
    let second = build_replay_token_from_minimal_spine_input(&input).expect("token output");

    assert_eq!(first, second);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.digest(), second.digest());
    assert_eq!(first.replay_token_digest, second.replay_token_digest);
    assert_eq!(first.replay_token.commit, second.replay_token.commit);
    assert_eq!(first.replay_token.tier, MilestoneTier::Macro);
}

#[test]
fn replay_token_builder_changes_when_macro_digest_changes() {
    let baseline = build_replay_token_from_minimal_spine_input(&input()).expect("baseline");

    let mut changed_macro_candidate = input();
    changed_macro_candidate.macro_candidate_digest = digest(9);
    let changed_macro_candidate =
        build_replay_token_from_minimal_spine_input(&changed_macro_candidate)
            .expect("changed macro candidate");

    let mut changed_finalization = input();
    changed_finalization.macro_finalization_digest = digest(10);
    let changed_finalization = build_replay_token_from_minimal_spine_input(&changed_finalization)
        .expect("changed finalization");

    assert_ne!(
        baseline.replay_token_digest,
        changed_macro_candidate.replay_token_digest
    );
    assert_ne!(baseline.digest(), changed_macro_candidate.digest());
    assert_ne!(
        baseline.replay_token_digest,
        changed_finalization.replay_token_digest
    );
    assert_ne!(baseline.digest(), changed_finalization.digest());
}

#[test]
fn replay_token_builder_preserves_consolidation_provenance() {
    let input = input();
    let output = build_replay_token_from_minimal_spine_input(&input).expect("token output");

    assert_eq!(output.macro_candidate_digest, input.macro_candidate_digest);
    assert_eq!(output.macro_milestone_digest, input.macro_milestone_digest);
    assert_eq!(
        output.meso_aggregation_digest,
        input.meso_aggregation_digest
    );
    assert_eq!(
        output.macro_finalization_digest,
        input.macro_finalization_digest
    );
    assert_eq!(output.meso_count, input.meso_count);
    assert_eq!(output.source, input.source);
    assert_eq!(output.replay_token_digest, output.replay_token.commit);
}

#[test]
fn replay_token_is_intent_only_not_scheduled_or_applied() {
    let output = build_replay_token_from_minimal_spine_input(&input()).expect("token output");

    assert!(!output.scheduled);
    assert!(!output.applied);
    assert_eq!(output.replay_token.budget, 0);
    assert_eq!(output.replay_token.redaction, 0);
}

#[test]
fn replay_token_builder_rejects_invalid_zero_links() {
    for mutate in [
        |input: &mut MinimalSpineReplayTokenInput| input.macro_candidate_digest = digest(0),
        |input: &mut MinimalSpineReplayTokenInput| input.macro_milestone_digest = digest(0),
        |input: &mut MinimalSpineReplayTokenInput| input.meso_aggregation_digest = digest(0),
        |input: &mut MinimalSpineReplayTokenInput| input.macro_finalization_digest = digest(0),
    ] {
        let mut invalid = input();
        mutate(&mut invalid);
        assert!(matches!(
            build_replay_token_from_minimal_spine_input(&invalid),
            Err(ReplayError::InvalidMinimalSpineReplayTokenInput(_))
        ));
    }

    let mut invalid = input();
    invalid.meso_count = 0;
    assert!(matches!(
        build_replay_token_from_minimal_spine_input(&invalid),
        Err(ReplayError::InvalidMinimalSpineReplayTokenInput(_))
    ));
}

#[test]
fn replay_token_builder_has_no_sleep_geist_identity_side_effects() {
    let output = build_replay_token_from_minimal_spine_input(&input()).expect("token output");

    assert!(!output.sleep_cycle);
    assert!(!output.geist_ingested);
    assert!(!output.identity_anchor);
}

#[test]
fn replay_token_builder_does_not_append_to_evidence_archive() {
    let output = build_replay_token_from_minimal_spine_input(&input()).expect("token output");

    assert!(!output.evidence_archive_appended);
    assert!(!output.deterministic_bytes().is_empty());

    let replay_source = include_str!("../src/lib.rs");
    assert!(!replay_source.contains("ArchiveStore"));
    assert!(!replay_source.contains("EvidenceStore"));
    assert!(!replay_source.contains("append_evidence"));
    assert!(!replay_source.contains("append_archive"));
}
