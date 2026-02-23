#![forbid(unsafe_code)]

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageRecordCanonicalMeta {
    pub contract_version: u16,
    pub backend_id: u16,
    pub validation_status: u8,
    pub policy_graph_digest_prefix: [u8; 8],
    pub evidence_chain_digest_prefix: [u8; 8],
}

impl StageRecordCanonicalMeta {
    pub fn new(
        contract_version: u16,
        backend_id: u16,
        validation_status: u8,
        policy_graph_digest_prefix: [u8; 8],
        evidence_chain_digest_prefix: [u8; 8],
    ) -> Self {
        Self {
            contract_version,
            backend_id,
            validation_status,
            policy_graph_digest_prefix,
            evidence_chain_digest_prefix,
        }
    }
}
