#![forbid(unsafe_code)]

use blake3::Hasher;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Digest32([u8; 32]);

impl Digest32 {
    pub const LEN: usize = 32;

    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl From<[u8; 32]> for Digest32 {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum ProtocolVersion {
    V1 = 1,
}

impl ProtocolVersion {
    pub fn id(self) -> u16 {
        self as u16
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageKind {
    ControlFrame,
    ExternalSpeech,
    ExternalThoughtSummary,
    WorkspaceBroadcast,
    ReplayNotice,
    AuditNotice,
    Unknown(u16),
}

impl MessageKind {
    pub fn code(self) -> u16 {
        match self {
            Self::ControlFrame => 1,
            Self::ExternalSpeech => 2,
            Self::ExternalThoughtSummary => 3,
            Self::WorkspaceBroadcast => 4,
            Self::ReplayNotice => 5,
            Self::AuditNotice => 6,
            Self::Unknown(code) => code,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Envelope {
    pub ver: ProtocolVersion,
    pub kind: MessageKind,
    pub body_commit: Digest32,
    pub commit: Digest32,
}

impl Envelope {
    pub fn new(ver: ProtocolVersion, kind: MessageKind, body_commit: Digest32) -> Self {
        let commit = digest_envelope(ver, kind, body_commit);
        Self {
            ver,
            kind,
            body_commit,
            commit,
        }
    }

    pub fn encode_canonical(&self) -> Vec<u8> {
        let mut enc = canonical::Encoder::new();
        enc.write_field(1, canonical::encode_u16(self.ver.id()));
        enc.write_field(2, canonical::encode_u16(self.kind.code()));
        enc.write_field(3, canonical::encode_digest32(&self.body_commit));
        enc.into_bytes()
    }
}

fn digest_envelope(ver: ProtocolVersion, kind: MessageKind, body_commit: Digest32) -> Digest32 {
    let mut enc = canonical::Encoder::new();
    enc.write_field(1, canonical::encode_u16(ver.id()));
    enc.write_field(2, canonical::encode_u16(kind.code()));
    enc.write_field(3, canonical::encode_digest32(&body_commit));
    digest_with_domain(b"ucf.protocol.envelope.v1", &enc.into_bytes())
}

fn digest_with_domain(domain: &[u8], payload: &[u8]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(payload);
    Digest32::new(*hasher.finalize().as_bytes())
}

pub mod v1 {
    use super::{canonical, digest_with_domain, Digest32};

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ControlFrameV1 {
        pub control_id: String,
        pub input_digest: Digest32,
        pub policy_class: u16,
        pub cycle_hint: u64,
        pub nonce: u64,
    }

    impl ControlFrameV1 {
        pub fn encode_canonical(&self) -> Vec<u8> {
            let mut enc = canonical::Encoder::new();
            enc.write_field(1, canonical::encode_string(&self.control_id));
            enc.write_field(2, canonical::encode_digest32(&self.input_digest));
            enc.write_field(3, canonical::encode_u16(self.policy_class));
            enc.write_field(4, canonical::encode_u64(self.cycle_hint));
            enc.write_field(5, canonical::encode_u64(self.nonce));
            enc.into_bytes()
        }

        pub fn digest(&self) -> Digest32 {
            digest_with_domain(b"ucf.protocol.v1.control_frame", &self.encode_canonical())
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ExternalSpeechV1 {
        pub evidence_id: String,
        pub text: String,
        pub risk_bucket: u16,
        pub nsr_ok: bool,
        pub tom_intent: String,
    }

    impl ExternalSpeechV1 {
        pub fn encode_canonical(&self) -> Vec<u8> {
            let mut enc = canonical::Encoder::new();
            enc.write_field(1, canonical::encode_string(&self.evidence_id));
            enc.write_field(2, canonical::encode_string(&self.text));
            enc.write_field(3, canonical::encode_u16(self.risk_bucket));
            enc.write_field(4, canonical::encode_bool(self.nsr_ok));
            enc.write_field(5, canonical::encode_string(&self.tom_intent));
            enc.into_bytes()
        }

        pub fn digest(&self) -> Digest32 {
            digest_with_domain(b"ucf.protocol.v1.external_speech", &self.encode_canonical())
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ExternalThoughtSummaryV1 {
        pub summary: String,
        pub thought_count: u32,
        pub evidence_count: u32,
        pub thought_commit: Digest32,
        pub evidence_commit: Digest32,
    }

    impl ExternalThoughtSummaryV1 {
        pub fn encode_canonical(&self) -> Vec<u8> {
            let mut enc = canonical::Encoder::new();
            enc.write_field(1, canonical::encode_string(&self.summary));
            enc.write_field(2, canonical::encode_u32(self.thought_count));
            enc.write_field(3, canonical::encode_u32(self.evidence_count));
            enc.write_field(4, canonical::encode_digest32(&self.thought_commit));
            enc.write_field(5, canonical::encode_digest32(&self.evidence_commit));
            enc.into_bytes()
        }

        pub fn digest(&self) -> Digest32 {
            digest_with_domain(
                b"ucf.protocol.v1.external_thought_summary",
                &self.encode_canonical(),
            )
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct WorkspaceSignalV1 {
        pub kind: u16,
        pub digest: Digest32,
        pub priority: u16,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct WorkspaceBroadcastV1 {
        pub snapshot_commit: Digest32,
        pub top_signals: Vec<WorkspaceSignalV1>,
    }

    impl WorkspaceBroadcastV1 {
        pub fn encode_canonical(&self) -> Vec<u8> {
            let mut enc = canonical::Encoder::new();
            enc.write_field(1, canonical::encode_digest32(&self.snapshot_commit));
            enc.write_field(2, canonical::encode_signal_list(&self.top_signals));
            enc.into_bytes()
        }

        pub fn digest(&self) -> Digest32 {
            digest_with_domain(
                b"ucf.protocol.v1.workspace_broadcast",
                &self.encode_canonical(),
            )
        }

        pub fn contains_raw_content(&self) -> bool {
            false
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ReplayNoticeV1 {
        pub token_commit: Digest32,
        pub tier: u8,
        pub target_digest: Digest32,
    }

    impl ReplayNoticeV1 {
        pub fn encode_canonical(&self) -> Vec<u8> {
            let mut enc = canonical::Encoder::new();
            enc.write_field(1, canonical::encode_digest32(&self.token_commit));
            enc.write_field(2, canonical::encode_u8(self.tier));
            enc.write_field(3, canonical::encode_digest32(&self.target_digest));
            enc.into_bytes()
        }

        pub fn digest(&self) -> Digest32 {
            digest_with_domain(b"ucf.protocol.v1.replay_notice", &self.encode_canonical())
        }

        pub fn contains_raw_content(&self) -> bool {
            false
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AuditNoticeV1 {
        pub event_kind: u16,
        pub evidence_digest: Digest32,
        pub reason_code: u16,
    }

    impl AuditNoticeV1 {
        pub fn encode_canonical(&self) -> Vec<u8> {
            let mut enc = canonical::Encoder::new();
            enc.write_field(1, canonical::encode_u16(self.event_kind));
            enc.write_field(2, canonical::encode_digest32(&self.evidence_digest));
            enc.write_field(3, canonical::encode_u16(self.reason_code));
            enc.into_bytes()
        }

        pub fn digest(&self) -> Digest32 {
            digest_with_domain(b"ucf.protocol.v1.audit_notice", &self.encode_canonical())
        }

        pub fn contains_raw_content(&self) -> bool {
            false
        }
    }
}

pub mod v2 {
    //! Reserved for future schema evolution.
}

pub mod canonical {
    use super::Digest32;

    pub fn encode_u8(value: u8) -> Vec<u8> {
        vec![value]
    }

    pub fn encode_u16(value: u16) -> Vec<u8> {
        value.to_be_bytes().to_vec()
    }

    pub fn encode_u32(value: u32) -> Vec<u8> {
        value.to_be_bytes().to_vec()
    }

    pub fn encode_u64(value: u64) -> Vec<u8> {
        value.to_be_bytes().to_vec()
    }

    pub fn encode_bool(value: bool) -> Vec<u8> {
        vec![u8::from(value)]
    }

    pub fn encode_string(value: &str) -> Vec<u8> {
        encode_bytes(value.as_bytes())
    }

    pub fn encode_bytes(value: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + value.len());
        out.extend_from_slice(&(value.len() as u32).to_be_bytes());
        out.extend_from_slice(value);
        out
    }

    pub fn encode_digest32(value: &Digest32) -> Vec<u8> {
        encode_bytes(value.as_bytes())
    }

    pub fn encode_signal_list(values: &[crate::boundary::v1::WorkspaceSignalV1]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(values.len() as u32).to_be_bytes());
        for signal in values {
            let mut enc = Encoder::new();
            enc.write_field(1, encode_u16(signal.kind));
            enc.write_field(2, encode_digest32(&signal.digest));
            enc.write_field(3, encode_u16(signal.priority));
            let payload = enc.into_bytes();
            out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            out.extend_from_slice(&payload);
        }
        out
    }

    #[derive(Default)]
    pub struct Encoder {
        bytes: Vec<u8>,
    }

    impl Encoder {
        pub fn new() -> Self {
            Self { bytes: Vec::new() }
        }

        pub fn write_field(&mut self, tag: u16, payload: Vec<u8>) {
            self.bytes.extend_from_slice(&tag.to_be_bytes());
            self.bytes
                .extend_from_slice(&(payload.len() as u32).to_be_bytes());
            self.bytes.extend_from_slice(&payload);
        }

        pub fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
    }
}
