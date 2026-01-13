#![forbid(unsafe_code)]

pub mod v1 {
    use prost::Enumeration;
    use prost::Message;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum DecisionKind {
        DecisionKindUnspecified = 0,
        DecisionKindAllow = 1,
        DecisionKindDeny = 2,
        DecisionKindEscalate = 3,
        DecisionKindObserve = 4,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum ActionCode {
        ActionCodeUnspecified = 0,
        ActionCodeContinue = 1,
        ActionCodePause = 2,
        ActionCodeTerminate = 3,
        ActionCodeRequireHuman = 4,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ControlFrame {
        #[prost(string, tag = "1")]
        pub frame_id: ::prost::alloc::string::String,
        #[prost(uint64, tag = "2")]
        pub issued_at_ms: u64,
        #[prost(message, optional, tag = "3")]
        pub decision: Option<PolicyDecision>,
        #[prost(string, repeated, tag = "4")]
        pub evidence_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
        #[prost(string, tag = "5")]
        pub policy_id: ::prost::alloc::string::String,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct PolicyDecision {
        #[prost(enumeration = "DecisionKind", tag = "1")]
        pub kind: i32,
        #[prost(enumeration = "ActionCode", tag = "2")]
        pub action: i32,
        #[prost(string, tag = "3")]
        pub rationale: ::prost::alloc::string::String,
        #[prost(uint32, tag = "4")]
        pub confidence_bp: u32,
        #[prost(string, repeated, tag = "5")]
        pub constraint_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ExperienceRecord {
        #[prost(string, tag = "1")]
        pub record_id: ::prost::alloc::string::String,
        #[prost(uint64, tag = "2")]
        pub observed_at_ms: u64,
        #[prost(string, tag = "3")]
        pub subject_id: ::prost::alloc::string::String,
        #[prost(bytes, tag = "4")]
        pub payload: ::prost::alloc::vec::Vec<u8>,
        #[prost(message, optional, tag = "5")]
        pub digest: Option<Digest>,
        #[prost(message, optional, tag = "6")]
        pub vrf_tag: Option<VrfTag>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct Digest {
        #[prost(string, tag = "1")]
        pub algorithm: ::prost::alloc::string::String,
        #[prost(bytes, tag = "2")]
        pub value: ::prost::alloc::vec::Vec<u8>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct VrfTag {
        #[prost(string, tag = "1")]
        pub algorithm: ::prost::alloc::string::String,
        #[prost(bytes, tag = "2")]
        pub proof: ::prost::alloc::vec::Vec<u8>,
        #[prost(bytes, tag = "3")]
        pub output: ::prost::alloc::vec::Vec<u8>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct MicroMilestone {
        #[prost(string, tag = "1")]
        pub milestone_id: ::prost::alloc::string::String,
        #[prost(uint64, tag = "2")]
        pub achieved_at_ms: u64,
        #[prost(string, tag = "3")]
        pub label: ::prost::alloc::string::String,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct MesoMilestone {
        #[prost(string, tag = "1")]
        pub milestone_id: ::prost::alloc::string::String,
        #[prost(uint64, tag = "2")]
        pub achieved_at_ms: u64,
        #[prost(string, tag = "3")]
        pub label: ::prost::alloc::string::String,
        #[prost(string, repeated, tag = "4")]
        pub micro_milestone_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct MacroMilestone {
        #[prost(string, tag = "1")]
        pub milestone_id: ::prost::alloc::string::String,
        #[prost(uint64, tag = "2")]
        pub achieved_at_ms: u64,
        #[prost(string, tag = "3")]
        pub label: ::prost::alloc::string::String,
        #[prost(string, repeated, tag = "4")]
        pub meso_milestone_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ProofEnvelope {
        #[prost(string, tag = "1")]
        pub envelope_id: ::prost::alloc::string::String,
        #[prost(bytes, tag = "2")]
        pub payload: ::prost::alloc::vec::Vec<u8>,
        #[prost(message, optional, tag = "3")]
        pub payload_digest: Option<Digest>,
        #[prost(message, repeated, tag = "4")]
        pub vrf_tags: ::prost::alloc::vec::Vec<VrfTag>,
        #[prost(string, repeated, tag = "5")]
        pub signature_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn serde_roundtrip_control_frame() {
            let decision = PolicyDecision {
                kind: DecisionKind::DecisionKindAllow as i32,
                action: ActionCode::ActionCodeContinue as i32,
                rationale: "ok".to_string(),
                confidence_bp: 8_500,
                constraint_ids: vec!["constraint-1".to_string()],
            };
            let frame = ControlFrame {
                frame_id: "frame-1".to_string(),
                issued_at_ms: 1_700_000_000_000,
                decision: Some(decision),
                evidence_ids: vec!["evidence-1".to_string(), "evidence-2".to_string()],
                policy_id: "policy-1".to_string(),
            };

            let encoded = serde_json::to_string(&frame).expect("serialize");
            let decoded: ControlFrame = serde_json::from_str(&encoded).expect("deserialize");

            assert_eq!(frame, decoded);
        }

        #[test]
        fn prost_roundtrip_experience_record() {
            let record = ExperienceRecord {
                record_id: "record-1".to_string(),
                observed_at_ms: 1_700_000_000_123,
                subject_id: "subject-1".to_string(),
                payload: vec![1, 2, 3, 4],
                digest: Some(Digest {
                    algorithm: "sha256".to_string(),
                    value: vec![9, 9, 9],
                }),
                vrf_tag: Some(VrfTag {
                    algorithm: "vrf".to_string(),
                    proof: vec![7, 8],
                    output: vec![6, 5],
                }),
            };

            let bytes = record.encode_to_vec();
            let decoded = ExperienceRecord::decode(bytes.as_slice()).expect("decode");

            assert_eq!(record, decoded);
        }
    }
}
