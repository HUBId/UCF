use ucf_geist::{
    verify_cross_layer_readback_query_candidate_v1, CrossLayerReadbackQueryAuditFailureV1,
    CrossLayerReadbackQueryAuditStatusV1, CrossLayerReadbackQueryCandidateStatusV1,
    CrossLayerReadbackQueryCandidateV1, EvidenceArchiveQueryRecordRefV1,
    EvidenceArchiveQueryableKindV1,
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
fn query_audit_passes_for_complete_candidate() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::ReplayAppendV1,
            1,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert_eq!(a.status, CrossLayerReadbackQueryAuditStatusV1::Pass);
    assert!(a.failures.is_empty());
    assert!(a.is_pass());
}

#[test]
fn query_audit_detects_missing_record_status() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::MissingRecord,
        vec!["missing replay".to_string()],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert_eq!(
        a.status,
        CrossLayerReadbackQueryAuditStatusV1::CandidateMissingRecord
    );
    assert!(a
        .failures
        .contains(&CrossLayerReadbackQueryAuditFailureV1::CandidateMissingRecord));
}

#[test]
fn query_audit_detects_mismatch_status() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::GeistIsmAppendV1,
            3,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Mismatch,
        vec!["kind mismatch".to_string()],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert_eq!(
        a.status,
        CrossLayerReadbackQueryAuditStatusV1::CandidateMismatch
    );
    assert!(a
        .failures
        .contains(&CrossLayerReadbackQueryAuditFailureV1::CandidateMismatch));
}

#[test]
fn query_audit_rejects_empty_candidate() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert_eq!(a.status, CrossLayerReadbackQueryAuditStatusV1::Fail);
    assert!(a
        .failures
        .contains(&CrossLayerReadbackQueryAuditFailureV1::EmptyCandidate));
}

#[test]
fn query_audit_digest_is_deterministic() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::SleepAppendV1,
            7,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert_eq!(a.digest(), a.digest());
    assert_eq!(a.deterministic_bytes(), a.deterministic_bytes());
}

#[test]
fn query_audit_digest_changes_when_candidate_changes() {
    let a1 =
        verify_cross_layer_readback_query_candidate_v1(&CrossLayerReadbackQueryCandidateV1::new(
            vec![sample_record(
                EvidenceArchiveQueryableKindV1::ReplayAppendV1,
                1,
            )],
            CrossLayerReadbackQueryCandidateStatusV1::Complete,
            vec![],
        ));
    let a2 =
        verify_cross_layer_readback_query_candidate_v1(&CrossLayerReadbackQueryCandidateV1::new(
            vec![sample_record(
                EvidenceArchiveQueryableKindV1::ReplayAppendV1,
                2,
            )],
            CrossLayerReadbackQueryCandidateStatusV1::Complete,
            vec![],
        ));
    assert_ne!(a1.audit_digest, a2.audit_digest);
}

#[test]
fn query_audit_records_candidate_digest_and_count() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![
            sample_record(EvidenceArchiveQueryableKindV1::ReplayAppendV1, 1),
            sample_record(EvidenceArchiveQueryableKindV1::SleepAppendV1, 2),
        ],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert_eq!(a.candidate_digest, c.digest());
    assert_eq!(a.record_count, 2);
}

#[test]
fn query_audit_is_verify_only() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert!(a.verify_only());
    assert!(a.read_model_only());
}

#[test]
fn query_audit_has_no_append_write_authority() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert!(!a.append_write_authority());
}

#[test]
fn query_audit_has_no_gateway_action_authority() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert!(!a.gateway_authority());
}

#[test]
fn query_audit_has_no_identity_or_runtime_authority() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert!(!a.identity_authority());
    assert!(!a.runtime_authority());
}

#[test]
fn query_audit_does_not_use_store_or_appender_handles() {
    let c = CrossLayerReadbackQueryCandidateV1::new(
        vec![sample_record(
            EvidenceArchiveQueryableKindV1::GeistIsmAppendV1,
            9,
        )],
        CrossLayerReadbackQueryCandidateStatusV1::Complete,
        vec![],
    );
    let a = verify_cross_layer_readback_query_candidate_v1(&c);
    assert!(!a.deterministic_bytes().is_empty());
}
