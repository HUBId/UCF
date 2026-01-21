use ucf::boundary::{v1, Digest32};
use ucf::boundary::{Envelope, MessageKind, ProtocolVersion};

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; 32])
}

#[test]
fn boundary_canonical_encoding_is_deterministic() {
    let message = v1::ExternalSpeechV1 {
        evidence_id: "ev-1".to_string(),
        text: "hello".to_string(),
        risk_bucket: 2,
        nsr_ok: true,
        tom_intent: "inform".to_string(),
    };

    let first = message.encode_canonical();
    let second = message.encode_canonical();

    assert_eq!(first, second);
    assert_eq!(message.digest(), message.digest());
}

#[test]
fn envelope_commit_changes_with_body() {
    let message_a = v1::ExternalSpeechV1 {
        evidence_id: "ev-1".to_string(),
        text: "hello".to_string(),
        risk_bucket: 2,
        nsr_ok: true,
        tom_intent: "inform".to_string(),
    };
    let message_b = v1::ExternalSpeechV1 {
        evidence_id: "ev-1".to_string(),
        text: "hello!".to_string(),
        risk_bucket: 2,
        nsr_ok: true,
        tom_intent: "inform".to_string(),
    };

    let env_a = Envelope::new(
        ProtocolVersion::V1,
        MessageKind::ExternalSpeech,
        message_a.digest(),
    );
    let env_b = Envelope::new(
        ProtocolVersion::V1,
        MessageKind::ExternalSpeech,
        message_b.digest(),
    );

    assert_ne!(env_a.commit, env_b.commit);
}

#[test]
fn boundary_messages_exclude_raw_content_by_default() {
    let broadcast = v1::WorkspaceBroadcastV1 {
        snapshot_commit: digest(1),
        top_signals: vec![v1::WorkspaceSignalV1 {
            kind: 1,
            digest: digest(2),
            priority: 9000,
        }],
    };
    let replay_notice = v1::ReplayNoticeV1 {
        token_commit: digest(3),
        tier: 2,
        target_digest: digest(4),
    };
    let audit_notice = v1::AuditNoticeV1 {
        event_kind: 7,
        evidence_digest: digest(5),
        reason_code: 42,
    };

    assert!(!broadcast.contains_raw_content());
    assert!(!replay_notice.contains_raw_content());
    assert!(!audit_notice.contains_raw_content());
}
