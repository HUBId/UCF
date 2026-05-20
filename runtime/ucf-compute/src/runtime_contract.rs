use crate::{BackendClass, BackendIdentity};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionalRealRuntimeArtifactSpec {
    pub artifact_id: &'static str,
    pub artifact_kind: &'static str,
    pub artifact_digest: [u8; 32],
    pub artifact_size_bytes: u64,
    pub source_note: &'static str,
    pub license_note: &'static str,
    pub local_only: bool,
    pub network_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionalRealRuntimeFixtureSpec {
    pub fixture_id: &'static str,
    pub input_digest: [u8; 32],
    pub expected_output_digest: [u8; 32],
    pub deterministic: bool,
    pub max_runtime_ms: u64,
    pub max_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionalRealRuntimeCandidateContract {
    pub backend: BackendIdentity,
    pub artifact: OptionalRealRuntimeArtifactSpec,
    pub fixture: OptionalRealRuntimeFixtureSpec,
    pub offline_by_default: bool,
    pub external_service_required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionalRealRuntimeContractError {
    BackendClassMustBeOptionalRealRuntime,
    RuntimeInferenceMustBeSupported,
    RuntimeClaimMustBeTrue,
    ProductionClaimForbidden,
    ArtifactDigestMustBeNonZero,
    ArtifactMustBeLocalOnly,
    ArtifactNetworkMustBeDisabled,
    FixtureInputDigestMustBeNonZero,
    FixtureOutputDigestMustBeNonZero,
    FixtureMustBeDeterministic,
    BoundedRuntimeRequired,
    BoundedMemoryRequired,
    OfflineByDefaultRequired,
    ExternalServiceForbidden,
}

impl OptionalRealRuntimeCandidateContract {
    pub fn validate(&self) -> Result<(), OptionalRealRuntimeContractError> {
        if self.backend.class != BackendClass::OptionalRealRuntime {
            return Err(OptionalRealRuntimeContractError::BackendClassMustBeOptionalRealRuntime);
        }
        if !self.backend.runtime_inference_supported {
            return Err(OptionalRealRuntimeContractError::RuntimeInferenceMustBeSupported);
        }
        if !self.backend.claims_runtime_real_inference() {
            return Err(OptionalRealRuntimeContractError::RuntimeClaimMustBeTrue);
        }
        if self.backend.production_claim {
            return Err(OptionalRealRuntimeContractError::ProductionClaimForbidden);
        }
        if self.artifact.artifact_digest == [0_u8; 32] {
            return Err(OptionalRealRuntimeContractError::ArtifactDigestMustBeNonZero);
        }
        if !self.artifact.local_only {
            return Err(OptionalRealRuntimeContractError::ArtifactMustBeLocalOnly);
        }
        if self.artifact.network_required {
            return Err(OptionalRealRuntimeContractError::ArtifactNetworkMustBeDisabled);
        }
        if self.fixture.input_digest == [0_u8; 32] {
            return Err(OptionalRealRuntimeContractError::FixtureInputDigestMustBeNonZero);
        }
        if self.fixture.expected_output_digest == [0_u8; 32] {
            return Err(OptionalRealRuntimeContractError::FixtureOutputDigestMustBeNonZero);
        }
        if !self.fixture.deterministic {
            return Err(OptionalRealRuntimeContractError::FixtureMustBeDeterministic);
        }
        if self.fixture.max_runtime_ms == 0 {
            return Err(OptionalRealRuntimeContractError::BoundedRuntimeRequired);
        }
        if self.fixture.max_memory_bytes == 0 {
            return Err(OptionalRealRuntimeContractError::BoundedMemoryRequired);
        }
        if !self.offline_by_default {
            return Err(OptionalRealRuntimeContractError::OfflineByDefaultRequired);
        }
        if self.external_service_required {
            return Err(OptionalRealRuntimeContractError::ExternalServiceForbidden);
        }
        Ok(())
    }
}
