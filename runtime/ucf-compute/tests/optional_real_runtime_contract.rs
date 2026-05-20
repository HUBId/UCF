use ucf_compute::{
    BackendClass, BackendIdentity, BackendPackKind, OptionalRealRuntimeArtifactSpec,
    OptionalRealRuntimeCandidateContract, OptionalRealRuntimeContractError,
    OptionalRealRuntimeFixtureSpec,
};

fn contract() -> OptionalRealRuntimeCandidateContract {
    OptionalRealRuntimeCandidateContract {
        backend: BackendIdentity::optional_real_runtime("local_fixture_runtime"),
        artifact: OptionalRealRuntimeArtifactSpec {
            artifact_id: "tiny_local_artifact_v1",
            artifact_kind: "safetensors",
            artifact_digest: [0xAA; 32],
            artifact_size_bytes: 1024,
            source_note: "local fixture",
            license_note: "test-only",
            local_only: true,
            network_required: false,
        },
        fixture: OptionalRealRuntimeFixtureSpec {
            fixture_id: "optional_real_runtime_fixture_v1",
            input_digest: [0xBB; 32],
            expected_output_digest: [0xCC; 32],
            deterministic: true,
            max_runtime_ms: 200,
            max_memory_bytes: 8 * 1024 * 1024,
        },
        offline_by_default: true,
        external_service_required: false,
    }
}

#[test]
fn optional_real_runtime_contract_rejects_compile_only_backend() {
    let mut c = contract();
    c.backend = BackendIdentity::optional_real_compile("compile_only_fixture");
    assert_eq!(
        c.validate().expect_err("compile-only backend rejected"),
        OptionalRealRuntimeContractError::BackendClassMustBeOptionalRealRuntime
    );
}

#[test]
fn optional_real_runtime_contract_requires_nonzero_artifact_digest() {
    let mut c = contract();
    c.artifact.artifact_digest = [0; 32];
    assert_eq!(
        c.validate().expect_err("zero artifact digest rejected"),
        OptionalRealRuntimeContractError::ArtifactDigestMustBeNonZero
    );
}

#[test]
fn optional_real_runtime_contract_requires_nonzero_output_digest() {
    let mut c = contract();
    c.fixture.expected_output_digest = [0; 32];
    assert_eq!(
        c.validate().expect_err("zero output digest rejected"),
        OptionalRealRuntimeContractError::FixtureOutputDigestMustBeNonZero
    );
}

#[test]
fn optional_real_runtime_contract_requires_local_offline_artifact() {
    let mut network = contract();
    network.artifact.network_required = true;
    assert_eq!(
        network.validate().expect_err("network artifact rejected"),
        OptionalRealRuntimeContractError::ArtifactNetworkMustBeDisabled
    );

    let mut external = contract();
    external.external_service_required = true;
    assert_eq!(
        external
            .validate()
            .expect_err("external service requirement rejected"),
        OptionalRealRuntimeContractError::ExternalServiceForbidden
    );
}

#[test]
fn optional_real_runtime_contract_forbids_production_claim() {
    let mut c = contract();
    c.backend = BackendIdentity::new(
        "runtime_claim_prod",
        BackendClass::OptionalRealRuntime,
        true,
        true,
        false,
        true,
        true,
    );
    assert_eq!(
        c.validate().expect_err("production claim rejected"),
        OptionalRealRuntimeContractError::ProductionClaimForbidden
    );
}

#[test]
fn optional_real_runtime_contract_requires_bounded_cost() {
    let mut runtime = contract();
    runtime.fixture.max_runtime_ms = 0;
    assert_eq!(
        runtime.validate().expect_err("runtime bound required"),
        OptionalRealRuntimeContractError::BoundedRuntimeRequired
    );

    let mut memory = contract();
    memory.fixture.max_memory_bytes = 0;
    assert_eq!(
        memory.validate().expect_err("memory bound required"),
        OptionalRealRuntimeContractError::BoundedMemoryRequired
    );
}

#[test]
fn optional_real_runtime_contract_is_metadata_only() {
    let c = contract();
    c.validate().expect("metadata-only validation passes");
}

#[test]
fn no_current_backend_claims_optional_real_runtime() {
    let identities = [
        BackendPackKind::StubV0.identity(),
        BackendPackKind::ToyV1.identity(),
        BackendPackKind::CandleToyV1.identity(),
        BackendPackKind::CandleLiquidV1.identity(),
        BackendPackKind::BurnToyV1.identity(),
    ];
    for identity in identities {
        assert_ne!(identity.class, BackendClass::OptionalRealRuntime);
        assert!(!identity.claims_runtime_real_inference());
        assert!(!identity.production_claim);
    }
}
