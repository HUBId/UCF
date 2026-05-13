use ucf_compute::{
    AiComputeBackend, BackendClass, BackendIdentity, BackendPackKind, ComputeBackendKind,
    CpuStubBackend,
};

#[test]
fn stub_and_toy_paths_do_not_claim_real_runtime_inference() {
    let cpu_stub = CpuStubBackend;
    let stub_identity = cpu_stub.identity();
    assert_eq!(stub_identity.class, BackendClass::Stub);
    assert!(!stub_identity.runtime_inference_supported);
    assert!(!stub_identity.claims_runtime_real_inference());
    assert!(!stub_identity.production_claim);

    let toy_identity = BackendPackKind::ToyV1.identity();
    assert_eq!(toy_identity.class, BackendClass::Toy);
    assert!(toy_identity.deterministic);
    assert!(toy_identity.offline);
    assert!(!toy_identity.runtime_inference_supported);
    assert!(!toy_identity.claims_runtime_real_inference());
    assert!(!toy_identity.production_claim);
}

#[test]
fn mock_class_is_distinct_from_stub_toy_and_real() {
    let mock_identity = BackendIdentity::mock("mock_fixture");
    assert_eq!(mock_identity.class, BackendClass::Mock);
    assert_ne!(mock_identity.class, BackendClass::Stub);
    assert_ne!(mock_identity.class, BackendClass::Toy);
    assert!(!mock_identity.claims_runtime_real_inference());
    assert!(!mock_identity.production_claim);
}

#[test]
fn optional_real_compile_paths_do_not_claim_runtime_inference() {
    for identity in [
        ComputeBackendKind::Candle.identity(),
        ComputeBackendKind::Burn.identity(),
        BackendPackKind::CandleToyV1.identity(),
        BackendPackKind::CandleLiquidV1.identity(),
        BackendPackKind::BurnToyV1.identity(),
    ] {
        assert_eq!(identity.class, BackendClass::OptionalRealCompile);
        assert!(identity.offline);
        assert!(!identity.external_service_required);
        assert!(!identity.runtime_inference_supported);
        assert!(!identity.claims_runtime_real_inference());
        assert!(!identity.production_claim);
    }
}

#[test]
fn remote_external_identity_is_not_default_safe() {
    let identity = BackendIdentity::remote_external("remote_v1");
    assert_eq!(identity.class, BackendClass::RemoteExternal);
    assert!(identity.external_service_required);
    assert!(!identity.offline);
    assert!(!identity.default_safe());
    assert!(!identity.runtime_inference_supported);
    assert!(!identity.production_claim);
}

#[test]
fn optional_real_runtime_requires_explicit_runtime_identity() {
    let compile_identity = BackendIdentity::optional_real_compile("compile_only");
    assert_eq!(compile_identity.class, BackendClass::OptionalRealCompile);
    assert!(!compile_identity.claims_runtime_real_inference());

    let runtime_identity = BackendIdentity::optional_real_runtime("local_fixture_runtime");
    assert_eq!(runtime_identity.class, BackendClass::OptionalRealRuntime);
    assert!(runtime_identity.runtime_inference_supported);
    assert!(runtime_identity.claims_runtime_real_inference());
    assert!(runtime_identity.offline);
    assert!(!runtime_identity.external_service_required);
    assert!(!runtime_identity.production_claim);
}
