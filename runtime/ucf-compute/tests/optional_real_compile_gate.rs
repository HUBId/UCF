use ucf_compute::{BackendClass, BackendIdentity, BackendPackKind, ComputeBackendKind};

fn assert_no_runtime_or_production_claim(identity: BackendIdentity) {
    assert_ne!(identity.class, BackendClass::OptionalRealRuntime);
    assert!(!identity.runtime_inference_supported);
    assert!(!identity.claims_runtime_real_inference());
    assert!(!identity.production_claim);
}

#[test]
fn optional_real_compile_lanes_do_not_claim_runtime_inference() {
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
        assert_no_runtime_or_production_claim(identity);
    }
}

#[test]
fn optional_real_compile_lanes_are_not_default_backend_kind() {
    assert_eq!(ComputeBackendKind::default(), ComputeBackendKind::Stub);
    assert_ne!(ComputeBackendKind::default(), ComputeBackendKind::Candle);
    assert_ne!(ComputeBackendKind::default(), ComputeBackendKind::Burn);
}

#[test]
fn no_optional_real_compile_lane_claims_production() {
    for identity in [
        BackendIdentity::optional_real_compile("compile_only_fixture"),
        ComputeBackendKind::Candle.identity(),
        ComputeBackendKind::Burn.identity(),
        BackendPackKind::CandleToyV1.identity(),
        BackendPackKind::CandleLiquidV1.identity(),
        BackendPackKind::BurnToyV1.identity(),
    ] {
        assert_eq!(identity.class, BackendClass::OptionalRealCompile);
        assert_no_runtime_or_production_claim(identity);
    }
}

#[cfg(feature = "remote-compute")]
#[test]
fn remote_external_lanes_require_external_service() {
    let identity = BackendPackKind::RemoteV1.identity();
    assert_eq!(identity.class, BackendClass::RemoteExternal);
    assert!(identity.external_service_required);
    assert!(!identity.offline);
    assert!(!identity.default_safe());
    assert_no_runtime_or_production_claim(identity);
}
