use ucf_replay::{
    build_replay_schedule_from_minimal_spine_tokens, build_replay_token_from_minimal_spine_input,
    MinimalSpineReplayScheduleConfig, MinimalSpineReplayTokenBuildOutput,
    MinimalSpineReplayTokenInput, ReplayError, MINIMAL_SPINE_REPLAY_SCHEDULE_SOURCE,
    MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
};
use ucf_types::Digest32;

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; Digest32::LEN])
}

fn input(seed: u8) -> MinimalSpineReplayTokenInput {
    MinimalSpineReplayTokenInput {
        macro_candidate_digest: digest(seed),
        macro_milestone_digest: digest(seed.saturating_add(1)),
        meso_aggregation_digest: digest(seed.saturating_add(2)),
        macro_finalization_digest: digest(seed.saturating_add(3)),
        meso_count: u32::from(seed),
        source: MINIMAL_SPINE_REPLAY_TOKEN_SOURCE,
    }
}

fn token(seed: u8) -> MinimalSpineReplayTokenBuildOutput {
    build_replay_token_from_minimal_spine_input(&input(seed)).expect("token output")
}

fn three_tokens() -> Vec<MinimalSpineReplayTokenBuildOutput> {
    vec![token(10), token(20), token(30)]
}

#[test]
fn replay_schedule_builder_is_deterministic() {
    let tokens = three_tokens();
    let config = MinimalSpineReplayScheduleConfig::default();

    let first = build_replay_schedule_from_minimal_spine_tokens(&tokens, config).expect("schedule");
    let second =
        build_replay_schedule_from_minimal_spine_tokens(&tokens, config).expect("schedule");

    assert_eq!(first, second);
    assert_eq!(first.deterministic_bytes(), second.deterministic_bytes());
    assert_eq!(first.digest(), second.digest());
    assert_eq!(first.schedule_digest, second.schedule_digest);
    assert_eq!(first.schedule_digest, first.digest());
}

#[test]
fn replay_schedule_builder_normalizes_input_order() {
    let tokens = three_tokens();
    let mut reversed = tokens.clone();
    reversed.reverse();

    let sorted_schedule = build_replay_schedule_from_minimal_spine_tokens(
        &tokens,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("sorted schedule");
    let reversed_schedule = build_replay_schedule_from_minimal_spine_tokens(
        &reversed,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("reversed schedule");

    assert_eq!(sorted_schedule, reversed_schedule);
    assert_eq!(
        sorted_schedule.schedule_digest,
        reversed_schedule.schedule_digest
    );
}

#[test]
fn replay_schedule_builder_rejects_empty_or_duplicate_tokens() {
    assert!(matches!(
        build_replay_schedule_from_minimal_spine_tokens(
            &[],
            MinimalSpineReplayScheduleConfig::default()
        ),
        Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(_))
    ));

    let duplicate = token(10);
    let tokens = vec![duplicate, duplicate];
    assert!(matches!(
        build_replay_schedule_from_minimal_spine_tokens(
            &tokens,
            MinimalSpineReplayScheduleConfig::default()
        ),
        Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(_))
    ));
}

#[test]
fn replay_schedule_builder_changes_when_token_changes() {
    let baseline = vec![token(10), token(20)];
    let changed = vec![token(10), token(40)];

    let baseline_schedule = build_replay_schedule_from_minimal_spine_tokens(
        &baseline,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("baseline schedule");
    let changed_schedule = build_replay_schedule_from_minimal_spine_tokens(
        &changed,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("changed schedule");

    assert_ne!(
        baseline_schedule.schedule_digest,
        changed_schedule.schedule_digest
    );
    assert_ne!(
        baseline_schedule.deterministic_bytes(),
        changed_schedule.deterministic_bytes()
    );
}

#[test]
fn replay_schedule_preserves_token_order_and_provenance() {
    let tokens = three_tokens();
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &tokens,
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");

    let mut expected = tokens.clone();
    expected.sort_by_key(|token| *token.replay_token_digest.as_bytes());
    let expected_digests = expected
        .iter()
        .map(|token| token.replay_token_digest)
        .collect::<Vec<_>>();
    let expected_output_digests = expected
        .iter()
        .map(|token| token.digest())
        .collect::<Vec<_>>();

    assert_eq!(schedule.replay_token_digests, expected_digests);
    assert_eq!(schedule.token_build_output_digests, expected_output_digests);
    assert_eq!(schedule.scheduled_tokens.len(), expected.len());
    assert_eq!(schedule.scheduled_token_provenance.len(), expected.len());

    for (index, (provenance, expected_token)) in schedule
        .scheduled_token_provenance
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert_eq!(provenance.order, u32::try_from(index).expect("index fits"));
        assert_eq!(
            provenance.replay_token_digest,
            expected_token.replay_token_digest
        );
        assert_eq!(
            provenance.token_build_output_digest,
            expected_token.digest()
        );
        assert_eq!(
            provenance.macro_candidate_digest,
            expected_token.macro_candidate_digest
        );
        assert_eq!(
            provenance.macro_milestone_digest,
            expected_token.macro_milestone_digest
        );
        assert_eq!(
            provenance.meso_aggregation_digest,
            expected_token.meso_aggregation_digest
        );
        assert_eq!(
            provenance.macro_finalization_digest,
            expected_token.macro_finalization_digest
        );
        assert_eq!(provenance.meso_count, expected_token.meso_count);
        assert_eq!(provenance.source, expected_token.source);
    }
}

#[test]
fn replay_schedule_cap_is_deterministic_if_configured() {
    let tokens = three_tokens();
    let mut expected = tokens.clone();
    expected.sort_by_key(|token| *token.replay_token_digest.as_bytes());

    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &tokens,
        MinimalSpineReplayScheduleConfig {
            max_tokens: Some(1),
            source: MINIMAL_SPINE_REPLAY_SCHEDULE_SOURCE,
        },
    )
    .expect("capped schedule");

    assert_eq!(schedule.token_count, 1);
    assert!(schedule.truncated);
    assert_eq!(
        schedule.replay_token_digests,
        vec![expected[0].replay_token_digest]
    );
    assert_eq!(
        schedule.scheduled_tokens[0].commit,
        expected[0].replay_token.commit
    );

    assert!(matches!(
        build_replay_schedule_from_minimal_spine_tokens(
            &tokens,
            MinimalSpineReplayScheduleConfig {
                max_tokens: Some(0),
                source: MINIMAL_SPINE_REPLAY_SCHEDULE_SOURCE,
            }
        ),
        Err(ReplayError::InvalidMinimalSpineReplayScheduleInput(_))
    ));
}

#[test]
fn replay_schedule_is_not_applied_sleep_or_geist() {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");

    assert!(!schedule.applied);
    assert!(!schedule.sleep_cycle);
    assert!(!schedule.geist_ingested);
    assert!(!schedule.identity_anchor);
    assert_eq!(schedule.source, MINIMAL_SPINE_REPLAY_SCHEDULE_SOURCE);

    let replay_source = include_str!("../src/lib.rs");
    assert!(!replay_source.contains("ReplayApplied {"));
}

#[test]
fn replay_schedule_does_not_append_to_evidence_archive() {
    let schedule = build_replay_schedule_from_minimal_spine_tokens(
        &three_tokens(),
        MinimalSpineReplayScheduleConfig::default(),
    )
    .expect("schedule");

    assert!(!schedule.evidence_archive_appended);
    assert!(!schedule.deterministic_bytes().is_empty());

    let replay_source = include_str!("../src/lib.rs");
    assert!(!replay_source.contains("ArchiveStore"));
    assert!(!replay_source.contains("EvidenceStore"));
    assert!(!replay_source.contains("append_evidence"));
    assert!(!replay_source.contains("append_archive"));
}
