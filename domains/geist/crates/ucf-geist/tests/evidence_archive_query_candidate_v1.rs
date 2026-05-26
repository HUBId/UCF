use ucf_geist::{
    CrossLayerReadbackQueryCandidateStatusV1, CrossLayerReadbackQueryCandidateV1,
    EvidenceArchiveQueryRecordRefV1, EvidenceArchiveQueryableKindV1,
};
use ucf_types::Digest32;

fn d(v: u8) -> Digest32 {
    Digest32::new([v; 32])
}

fn sample_record(kind: EvidenceArchiveQueryableKindV1, v: u8) -> EvidenceArchiveQueryRecordRefV1 {
    EvidenceArchiveQueryRecordRefV1 {
        kind,
        archive_key_digest: d(v),
        evidence_id_digest: d(v.wrapping_add(1)),
        payload_digest: d(v.wrapping_add(2)),
        archive_record_digest: d(v.wrapping_add(3)),
        readback_digest: d(v.wrapping_add(4)),
        root_commit_digest: Some(d(v.wrapping_add(5))),
    }
}

#[test]
fn query_candidate_accepts_replay_sleep_geist_record_refs() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![
            sample_record(EvidenceArchiveQueryableKindV1::ReplayAppendV1, 1),
            sample_record(EvidenceArchiveQueryableKindV1::SleepAppendV1, 10),
            sample_record(EvidenceArchiveQueryableKindV1::GeistIsmAppendV1, 20),
        ],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert_eq!(c.records.len(), 3);
}

#[test]
fn query_candidate_digest_is_deterministic() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::ReplayAppendV1,
            1,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert_eq!(c.digest(), c.digest());
    assert_eq!(c.deterministic_bytes(), c.deterministic_bytes());
}

#[test]
fn query_candidate_digest_changes_when_record_changes() {
    let a = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::ReplayAppendV1,
            1,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let b = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::ReplayAppendV1,
            2,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn query_candidate_rejects_or_flags_missing_record() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::MissingRecord,
        vec!["missing replay record".to_string()],
    );
    assert_eq!(
        c.status,
        CrossLayerReadbackQueryCandidateStatusV1::MissingRecord
    );
    assert!(!c.failures.is_empty());
}

#[test]
fn query_candidate_rejects_or_flags_mismatched_kind() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::GeistIsmAppendV1,
            3,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Mismatch,
        vec!["kind mismatch".to_string()],
    );
    assert_eq!(c.status, CrossLayerReadbackQueryCandidateStatusV1::Mismatch);
}

#[test]
fn query_candidate_only_allows_bounded_kinds_65_66_67() {
    assert_eq!(
        EvidenceArchiveQueryableKindV1::ReplayAppendV1.record_kind(),
        ucf_archive_store::RecordKind::Other(65)
    );
    assert_eq!(
        EvidenceArchiveQueryableKindV1::SleepAppendV1.record_kind(),
        ucf_archive_store::RecordKind::Other(66)
    );
    assert_eq!(
        EvidenceArchiveQueryableKindV1::GeistIsmAppendV1.record_kind(),
        ucf_archive_store::RecordKind::Other(67)
    );
}

#[test]
fn query_candidate_is_read_model_only() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert!(c.read_model_only());
}

#[test]
fn query_candidate_has_no_append_write_authority() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert!(!c.append_authority());
    assert!(!c.evidence_archive_write_authority());
}

#[test]
fn query_candidate_has_no_gateway_action_authority() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert!(!c.gateway_authority());
}

#[test]
fn query_candidate_has_no_identity_or_ism_authority() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert!(!c.identity_authority());
}

#[test]
fn query_candidate_has_no_runtime_scheduler_authority() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert!(!c.runtime_authority());
}

#[test]
fn query_candidate_does_not_use_store_or_appender_handles() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::SleepAppendV1,
            7,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    assert!(!c.deterministic_bytes().is_empty());
}
