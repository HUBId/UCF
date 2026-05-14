use ucf_compute::{
    run_stub_compute_fixture, stub_compute_fixture_digest, AiComputeBackend, BackendClass,
    BackendComponentId, ComputeBackendKind, ComputeBudget, ComputeInput, CpuStubBackend, FrameId,
    STUB_COMPUTE_FIXTURE_VERSION,
};

fn fixture_input() -> ComputeInput {
    ComputeInput {
        frame_id: FrameId(17),
        t: 9,
        context_digest: [0xA5; 32],
    }
}

fn fixture_budget() -> ComputeBudget {
    ComputeBudget::default()
}

#[test]
fn stub_compute_fixture_is_deterministic() {
    let first =
        run_stub_compute_fixture(&fixture_input(), fixture_budget()).expect("first fixture");
    let second =
        run_stub_compute_fixture(&fixture_input(), fixture_budget()).expect("second fixture");

    println!("stub_fixture_digest={}", hex::encode(first.digest));
    assert_eq!(first.signals, second.signals);
    assert_eq!(first.summary, second.summary);
    assert_eq!(first.digest, second.digest);
    assert_eq!(
        first.digest,
        stub_compute_fixture_digest(
            &first.provenance,
            &first.input,
            &first.signals,
            &first.summary
        )
    );
    assert_ne!(first.digest, [0_u8; 32]);
}

#[test]
fn stub_compute_fixture_reports_stub_provenance() {
    let backend = CpuStubBackend;
    let identity = backend.identity();
    let fixture = run_stub_compute_fixture(&fixture_input(), fixture_budget()).expect("fixture");

    assert_eq!(identity.class, BackendClass::Stub);
    assert_eq!(fixture.provenance.backend_class, BackendClass::Stub);
    assert_eq!(fixture.provenance.backend_name, "stub");
    assert_eq!(fixture.provenance.fixture_id, STUB_COMPUTE_FIXTURE_VERSION);
    assert!(fixture.provenance.no_real_inference);
    assert!(!fixture.provenance.production_claim);
    assert_eq!(fixture.summary.backend_profile, "stub:v1");
    assert_eq!(fixture.summary.backend_pack_id, 0);
}

#[test]
fn stub_compute_fixture_is_offline_no_external_artifacts() {
    let identity = ComputeBackendKind::Stub.identity();
    let fixture = run_stub_compute_fixture(&fixture_input(), fixture_budget()).expect("fixture");

    assert!(identity.offline);
    assert!(identity.deterministic);
    assert!(!identity.external_service_required);
    assert!(!fixture.provenance.external_service_required);
    assert_eq!(fixture.summary.model_hashes_digest, [0_u8; 32]);
    assert_eq!(
        fixture.summary.llm_backend,
        BackendComponentId::StubV0 as u8
    );
    assert_eq!(
        fixture.summary.world_backend,
        BackendComponentId::StubV0 as u8
    );
    assert_eq!(
        fixture.summary.sae_backend,
        BackendComponentId::StubV0 as u8
    );
    assert_eq!(
        fixture.summary.ssm_backend,
        BackendComponentId::StubV0 as u8
    );
    assert_eq!(
        fixture.summary.lfm_backend,
        BackendComponentId::StubV0 as u8
    );
}

#[test]
fn stub_compute_fixture_does_not_claim_real_runtime_inference() {
    let identity = ComputeBackendKind::Stub.identity();
    let fixture = run_stub_compute_fixture(&fixture_input(), fixture_budget()).expect("fixture");

    assert_ne!(identity.class, BackendClass::OptionalRealRuntime);
    assert!(!identity.runtime_inference_supported);
    assert!(!identity.claims_runtime_real_inference());
    assert!(!identity.production_claim);
    assert!(!fixture.provenance.runtime_inference_supported);
    assert!(fixture.provenance.no_real_inference);
    assert!(!fixture.provenance.production_claim);
}

#[test]
fn no_current_stub_backend_claims_production() {
    for identity in [
        ComputeBackendKind::Stub.identity(),
        CpuStubBackend.identity(),
    ] {
        assert_eq!(identity.class, BackendClass::Stub);
        assert!(identity.default_safe());
        assert!(!identity.production_claim);
        assert!(!identity.claims_runtime_real_inference());
    }
}
