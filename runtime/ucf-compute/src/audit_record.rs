use crate::contracts::{BackendClass, BackendIdentity};
use crate::output_link::ComputeOutputLink;
use sha2::{Digest, Sha256};

pub const COMPUTE_AUDIT_RECORD_VERSION: u32 = 1;
pub const COMPUTE_AUDIT_RECORD_SOURCE: &str = "compute-audit-record-v1-metadata-only";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ComputeAuditStatus {
    FixtureStub,
    GoldenToy,
    CompileOnly,
    RuntimeDeferred,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ComputeAuditRecord {
    pub version: u32,
    pub output_record_digest: [u8; 32],
    pub compute_output_link_digest: [u8; 32],
    pub compute_result_digest: [u8; 32],
    pub backend_class: BackendClass,
    pub backend_name: String,
    pub audit_status: ComputeAuditStatus,
    pub runtime_inference_claim: bool,
    pub production_claim: bool,
    pub evidence_authority: bool,
    pub output_authority: bool,
    pub minimal_spine_required: bool,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ComputeAuditRecordError {
    #[error("compute audit record requires a nonzero output_record_digest")]
    ZeroOutputRecordDigest,
    #[error("compute audit record requires a nonzero compute_output_link_digest")]
    ZeroComputeOutputLinkDigest,
    #[error("compute audit record requires a nonzero compute_result_digest")]
    ZeroComputeResultDigest,
}

impl ComputeAuditRecord {
    pub fn from_stub_link(link: &ComputeOutputLink) -> Result<Self, ComputeAuditRecordError> {
        Self::from_link(
            link,
            ComputeAuditStatus::FixtureStub,
            COMPUTE_AUDIT_RECORD_SOURCE,
        )
    }

    pub fn from_toy_link(link: &ComputeOutputLink) -> Result<Self, ComputeAuditRecordError> {
        Self::from_link(
            link,
            ComputeAuditStatus::GoldenToy,
            COMPUTE_AUDIT_RECORD_SOURCE,
        )
    }

    pub fn from_optional_real_compile_link(
        link: &ComputeOutputLink,
    ) -> Result<Self, ComputeAuditRecordError> {
        Self::from_link(
            link,
            ComputeAuditStatus::CompileOnly,
            COMPUTE_AUDIT_RECORD_SOURCE,
        )
    }

    pub fn from_link(
        link: &ComputeOutputLink,
        audit_status: ComputeAuditStatus,
        source: impl Into<String>,
    ) -> Result<Self, ComputeAuditRecordError> {
        let record = Self {
            version: COMPUTE_AUDIT_RECORD_VERSION,
            output_record_digest: link.output_record_digest,
            compute_output_link_digest: link.link_digest(),
            compute_result_digest: link.compute_result_digest,
            backend_class: link.backend_class,
            backend_name: link.backend_name.clone(),
            audit_status,
            runtime_inference_claim: false,
            production_claim: false,
            evidence_authority: false,
            output_authority: false,
            minimal_spine_required: false,
            source: source.into(),
        };
        record.validate()?;
        Ok(record)
    }

    pub fn from_backend_identity(
        output_record_digest: [u8; 32],
        compute_output_link_digest: [u8; 32],
        compute_result_digest: [u8; 32],
        backend_identity: BackendIdentity,
        audit_status: ComputeAuditStatus,
        source: impl Into<String>,
    ) -> Result<Self, ComputeAuditRecordError> {
        let record = Self {
            version: COMPUTE_AUDIT_RECORD_VERSION,
            output_record_digest,
            compute_output_link_digest,
            compute_result_digest,
            backend_class: backend_identity.class,
            backend_name: backend_identity.name.to_string(),
            audit_status,
            runtime_inference_claim: false,
            production_claim: false,
            evidence_authority: false,
            output_authority: false,
            minimal_spine_required: false,
            source: source.into(),
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), ComputeAuditRecordError> {
        if is_zero_digest(&self.output_record_digest) {
            return Err(ComputeAuditRecordError::ZeroOutputRecordDigest);
        }
        if is_zero_digest(&self.compute_output_link_digest) {
            return Err(ComputeAuditRecordError::ZeroComputeOutputLinkDigest);
        }
        if is_zero_digest(&self.compute_result_digest) {
            return Err(ComputeAuditRecordError::ZeroComputeResultDigest);
        }
        Ok(())
    }

    pub fn audit_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.version.to_le_bytes());
        hasher.update(self.output_record_digest);
        hasher.update(self.compute_output_link_digest);
        hasher.update(self.compute_result_digest);
        hasher.update([self.backend_class as u8]);
        hash_string(&mut hasher, &self.backend_name);
        hasher.update([self.audit_status as u8]);
        hasher.update([self.runtime_inference_claim as u8]);
        hasher.update([self.production_claim as u8]);
        hasher.update([self.evidence_authority as u8]);
        hasher.update([self.output_authority as u8]);
        hasher.update([self.minimal_spine_required as u8]);
        hash_string(&mut hasher, &self.source);
        hasher.finalize().into()
    }

    pub const fn metadata_only(&self) -> bool {
        !self.runtime_inference_claim
            && !self.production_claim
            && !self.evidence_authority
            && !self.output_authority
            && !self.minimal_spine_required
    }
}

fn is_zero_digest(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u32).to_le_bytes());
    hasher.update(value.as_bytes());
}
