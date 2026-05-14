use sha2::{Digest, Sha256};
use ucf_compute::{
    run_stub_compute_fixture, run_toy_compute_golden_fixture, BackendClass, BackendIdentity,
    BackendPackKind, ComputeBackendKind, ComputeBudget, ComputeInput, ComputeOutputLink, FrameId,
};
use ucf_protocol::{canonical_bytes, v1::spec::OutputRecord};

fn output_record_fixture() -> OutputRecord {
    OutputRecord {
        version: 1,
        input_digest: vec![0x11; 32],
        candidate_set_digest: vec![0x22; 32],
        selected_candidate_digest: vec![0x33; 32],
        output_digest: vec![0x44; 32],
        policy_status: "allow".to_string(),
        status: "materialized-test-output".to_string(),
        provenance: "compute-output-link-test-fixture".to_string(),
        evidence_id: None,
    }
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hasher.finalize().into()
}

fn output_record_digest(record: &OutputRecord) -> [u8; 32] {
    digest_bytes(b"ucf.protocol.output_record.v1", &canonical_bytes(record))
}

fn output_record_bytes_digest(record: &OutputRecord) -> [u8; 32] {
    digest_bytes(
        b"ucf.protocol.output_record.bytes.v1",
        &canonical_bytes(record),
    )
}

fn stub_fixture_input() -> ComputeInput {
    ComputeInput {
        frame_id: FrameId(20),
        t: 20,
        context_digest: [0x20; 32],
    }
}

#[test]
fn stub_link_is_derived_metadata_only_and_does_not_mutate_output_record() {
    let output_record = output_record_fixture();
    let before_bytes = canonical_bytes(&output_record);
    let fixture = run_stub_compute_fixture(&stub_fixture_input(), ComputeBudget::default())
        .expect("stub fixture");

    let link = ComputeOutputLink::derived(
        output_record_digest(&output_record),
        fixture.digest,
        ComputeBackendKind::Stub.identity(),
        fixture.provenance.fixture_id,
    )
    .with_output_record_id("output-record-fixture-v1")
    .with_output_record_bytes_digest(output_record_bytes_digest(&output_record));

    assert_eq!(canonical_bytes(&output_record), before_bytes);
    assert_eq!(link.backend_class, BackendClass::Stub);
    assert_eq!(link.backend_name, "stub");
    assert_eq!(link.compute_result_digest, fixture.digest);
    assert_eq!(
        link.output_record_digest,
        output_record_digest(&output_record)
    );
    assert!(link.metadata_only);
    assert!(!link.output_record_authority);
    assert!(!link.minimal_spine_required);
    assert!(link.no_real_runtime);
    assert!(!link.runtime_inference_supported);
    assert!(!link.production_claim);
    assert!(!link.external_service_required);
    assert!(link.offline);
}

#[test]
fn toy_golden_link_remains_toy_non_real_and_non_production() {
    let output_record = output_record_fixture();
    let golden = run_toy_compute_golden_fixture().expect("toy golden");

    let link = ComputeOutputLink::derived(
        output_record_digest(&output_record),
        golden.digest,
        BackendIdentity::toy(golden.provenance.backend_name),
        golden.provenance.golden_version,
    );

    assert_eq!(link.backend_class, BackendClass::Toy);
    assert_eq!(link.backend_name, "toy_v1");
    assert_eq!(link.compute_result_digest, golden.digest);
    assert!(link.no_real_runtime);
    assert!(!link.runtime_inference_supported);
    assert!(!link.production_claim);
    assert!(link.metadata_only);
    assert!(!link.output_record_authority);
}

#[test]
fn optional_real_compile_link_remains_compile_only() {
    let output_record = output_record_fixture();
    let compute_digest = digest_bytes(b"ucf.compute.optional_real_compile_probe.v1", b"candle");

    for identity in [
        ComputeBackendKind::Candle.identity(),
        ComputeBackendKind::Burn.identity(),
        BackendPackKind::CandleToyV1.identity(),
        BackendPackKind::CandleLiquidV1.identity(),
        BackendPackKind::BurnToyV1.identity(),
    ] {
        let link = ComputeOutputLink::derived(
            output_record_digest(&output_record),
            compute_digest,
            identity,
            "optional-real-compile-probe",
        );

        assert_eq!(link.backend_class, BackendClass::OptionalRealCompile);
        assert!(link.no_real_runtime);
        assert!(!link.runtime_inference_supported);
        assert!(!link.production_claim);
        assert!(!link.external_service_required);
        assert!(link.offline);
        assert!(link.metadata_only);
        assert!(!link.output_record_authority);
    }
}

#[test]
fn link_digest_is_deterministic_and_changes_with_references() {
    let output_record = output_record_fixture();
    let output_digest = output_record_digest(&output_record);
    let compute_digest = digest_bytes(b"ucf.compute.link.test", b"compute-result");
    let link = ComputeOutputLink::derived(
        output_digest,
        compute_digest,
        ComputeBackendKind::Stub.identity(),
        "stub_compute_fixture_v1",
    );

    assert_eq!(link.link_digest(), link.clone().link_digest());

    let changed_compute = ComputeOutputLink::derived(
        output_digest,
        digest_bytes(b"ucf.compute.link.test", b"other-compute-result"),
        ComputeBackendKind::Stub.identity(),
        "stub_compute_fixture_v1",
    );
    assert_ne!(link.link_digest(), changed_compute.link_digest());

    let changed_output = ComputeOutputLink::derived(
        digest_bytes(b"ucf.protocol.output_record.v1", b"other-output-record"),
        compute_digest,
        ComputeBackendKind::Stub.identity(),
        "stub_compute_fixture_v1",
    );
    assert_ne!(link.link_digest(), changed_output.link_digest());
}

#[test]
fn minimal_spine_e2e_test_does_not_import_compute() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let minimal_spine_test = manifest_dir
        .join("../../core/crates/ucf-router/tests/minimal_spine_e2e.rs")
        .canonicalize()
        .expect("minimal spine e2e path");
    let source = std::fs::read_to_string(minimal_spine_test).expect("minimal spine e2e source");

    assert!(!source.contains("ucf_compute"));
    assert!(!source.contains("ucf-compute"));
}
