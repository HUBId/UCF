use crate::contracts::{BackendClass, BackendIdentity};
use sha2::{Digest, Sha256};

pub const COMPUTE_OUTPUT_LINK_VERSION: u32 = 1;
pub const COMPUTE_OUTPUT_LINK_PROVENANCE: &str = "derived-compute-output-link-v1-metadata-only";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComputeOutputLink {
    pub version: u32,
    pub output_record_digest: [u8; 32],
    pub output_record_id: Option<String>,
    pub output_record_bytes_digest: Option<[u8; 32]>,
    pub compute_result_digest: [u8; 32],
    pub backend_class: BackendClass,
    pub backend_name: String,
    pub source: String,
    pub provenance: String,
    pub no_real_runtime: bool,
    pub runtime_inference_supported: bool,
    pub production_claim: bool,
    pub external_service_required: bool,
    pub deterministic: bool,
    pub offline: bool,
    pub metadata_only: bool,
    pub output_record_authority: bool,
    pub minimal_spine_required: bool,
}

impl ComputeOutputLink {
    pub fn derived(
        output_record_digest: [u8; 32],
        compute_result_digest: [u8; 32],
        backend_identity: BackendIdentity,
        source: impl Into<String>,
    ) -> Self {
        Self {
            version: COMPUTE_OUTPUT_LINK_VERSION,
            output_record_digest,
            output_record_id: None,
            output_record_bytes_digest: None,
            compute_result_digest,
            backend_class: backend_identity.class,
            backend_name: backend_identity.name.to_string(),
            source: source.into(),
            provenance: COMPUTE_OUTPUT_LINK_PROVENANCE.to_string(),
            no_real_runtime: !backend_identity.claims_runtime_real_inference(),
            runtime_inference_supported: backend_identity.runtime_inference_supported,
            production_claim: backend_identity.production_claim,
            external_service_required: backend_identity.external_service_required,
            deterministic: backend_identity.deterministic,
            offline: backend_identity.offline,
            metadata_only: true,
            output_record_authority: false,
            minimal_spine_required: false,
        }
    }

    pub fn with_output_record_id(mut self, output_record_id: impl Into<String>) -> Self {
        self.output_record_id = Some(output_record_id.into());
        self
    }

    pub fn with_output_record_bytes_digest(mut self, bytes_digest: [u8; 32]) -> Self {
        self.output_record_bytes_digest = Some(bytes_digest);
        self
    }

    pub fn link_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.output_record_digest);
        hash_optional_string(&mut hasher, self.output_record_id.as_deref());
        hash_optional_digest(&mut hasher, self.output_record_bytes_digest);
        hasher.update(self.compute_result_digest);
        hasher.update([self.backend_class as u8]);
        hash_string(&mut hasher, &self.backend_name);
        hash_string(&mut hasher, &self.source);
        hash_string(&mut hasher, &self.provenance);
        hasher.update([self.no_real_runtime as u8]);
        hasher.update([self.runtime_inference_supported as u8]);
        hasher.update([self.production_claim as u8]);
        hasher.update([self.external_service_required as u8]);
        hasher.update([self.deterministic as u8]);
        hasher.update([self.offline as u8]);
        hasher.update([self.metadata_only as u8]);
        hasher.update([self.output_record_authority as u8]);
        hasher.update([self.minimal_spine_required as u8]);
        hasher.finalize().into()
    }
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u32).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_optional_string(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hash_string(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn hash_optional_digest(hasher: &mut Sha256, value: Option<[u8; 32]>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value);
        }
        None => hasher.update([0]),
    }
}
