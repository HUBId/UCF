#[path = "../../../ucf-protocol/src/lib.rs"]
mod ucf_protocol;

use blake3::Hasher;
use prost::Message;

pub use ucf_protocol::*;

pub mod ucf {
    pub use crate::v1;
}

/// Canonically encode a protobuf message using deterministic field ordering.
pub fn canonical_bytes<M: Message>(message: &M) -> Vec<u8> {
    message.encode_to_vec()
}

/// Compute a 32-byte digest using BLAKE3 over DOMAIN || schema_id || schema_version || bytes.
pub fn digest32(domain: &str, schema_id: &str, schema_version: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(schema_id.as_bytes());
    hasher.update(schema_version.as_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}
