use sha2::{Digest, Sha256};
use ucf_compute::{
    run_stub_compute_fixture, run_toy_compute_golden_fixture, BackendClass, BackendIdentity,
    BackendPackKind, ComputeAuditRecord, ComputeAuditRecordError, ComputeAuditStatus,
    ComputeBackendKind, ComputeBudget, ComputeInput, ComputeOutputLink, FrameId,
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
        provenance: "compute-audit-record-test-fixture".to_string(),
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

fn stub_fixture_input() -> ComputeInput {
    ComputeInput {
        frame_id: FrameId(21),
        t: 21,
        context_digest: [0x21; 32],
    }
}

fn assert_metadata_only(record: &ComputeAuditRecord) {
    assert!(!record.runtime_inference_claim);
    assert!(!record.production_claim);
    assert!(!record.evidence_authority);
    assert!(!record.output_authority);
    assert!(!record.minimal_spine_required);
    assert!(record.metadata_only());
}

#[test]
fn compute_audit_record_from_stub_link_is_metadata_only() {
    let output_record = output_record_fixture();
    let fixture = run_stub_compute_fixture(&stub_fixture_input(), ComputeBudget::default())
        .expect("stub fixture");
    let link = ComputeOutputLink::derived(
        output_record_digest(&output_record),
        fixture.digest,
        ComputeBackendKind::Stub.identity(),
        fixture.provenance.fixture_id,
    );

    let audit = ComputeAuditRecord::from_stub_link(&link).expect("stub audit record");

    assert_eq!(audit.audit_status, ComputeAuditStatus::FixtureStub);
    assert_eq!(audit.backend_class, BackendClass::Stub);
    assert_eq!(audit.backend_name, "stub");
    assert_eq!(audit.output_record_digest, link.output_record_digest);
    assert_eq!(audit.compute_output_link_digest, link.link_digest());
    assert_eq!(audit.compute_result_digest, fixture.digest);
    assert_metadata_only(&audit);
}

#[test]
fn compute_audit_record_from_toy_link_is_metadata_only() {
    let output_record = output_record_fixture();
    let golden = run_toy_compute_golden_fixture().expect("toy golden");
    let link = ComputeOutputLink::derived(
        output_record_digest(&output_record),
        golden.digest,
        BackendIdentity::toy(golden.provenance.backend_name),
        golden.provenance.golden_version,
    );

    let audit = ComputeAuditRecord::from_toy_link(&link).expect("toy audit record");

    assert_eq!(audit.audit_status, ComputeAuditStatus::GoldenToy);
    assert_eq!(audit.backend_class, BackendClass::Toy);
    assert_eq!(audit.backend_name, "toy_v1");
    assert_eq!(audit.compute_output_link_digest, link.link_digest());
    assert_eq!(audit.compute_result_digest, golden.digest);
    assert_metadata_only(&audit);
}

#[test]
fn compute_audit_record_from_optional_real_compile_is_compile_only() {
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

        let audit = ComputeAuditRecord::from_optional_real_compile_link(&link)
            .expect("optional-real compile audit record");

        assert_eq!(audit.audit_status, ComputeAuditStatus::CompileOnly);
        assert_eq!(audit.backend_class, BackendClass::OptionalRealCompile);
        assert_eq!(audit.compute_output_link_digest, link.link_digest());
        assert_metadata_only(&audit);
    }
}

#[test]
fn compute_audit_record_digest_is_deterministic() {
    let output_record = output_record_fixture();
    let output_digest = output_record_digest(&output_record);
    let compute_digest = digest_bytes(b"ucf.compute.audit.test", b"compute-result");
    let link = ComputeOutputLink::derived(
        output_digest,
        compute_digest,
        ComputeBackendKind::Stub.identity(),
        "stub_compute_fixture_v1",
    );
    let audit = ComputeAuditRecord::from_stub_link(&link).expect("audit record");

    assert_eq!(audit.audit_digest(), audit.clone().audit_digest());

    let changed_compute = ComputeOutputLink::derived(
        output_digest,
        digest_bytes(b"ucf.compute.audit.test", b"other-compute-result"),
        ComputeBackendKind::Stub.identity(),
        "stub_compute_fixture_v1",
    );
    let changed_audit = ComputeAuditRecord::from_stub_link(&changed_compute).expect("audit record");
    assert_ne!(audit.audit_digest(), changed_audit.audit_digest());
}

#[test]
fn compute_audit_record_rejects_zero_required_links() {
    let output_digest = digest_bytes(b"ucf.protocol.output_record.v1", b"output-record");
    let link_digest = digest_bytes(b"ucf.compute.output_link.v1", b"link");
    let compute_digest = digest_bytes(b"ucf.compute.result.v1", b"result");

    assert_eq!(
        ComputeAuditRecord::from_backend_identity(
            [0; 32],
            link_digest,
            compute_digest,
            ComputeBackendKind::Stub.identity(),
            ComputeAuditStatus::FixtureStub,
            "zero-output-record-digest-test",
        )
        .expect_err("zero output record digest rejected"),
        ComputeAuditRecordError::ZeroOutputRecordDigest
    );
    assert_eq!(
        ComputeAuditRecord::from_backend_identity(
            output_digest,
            [0; 32],
            compute_digest,
            ComputeBackendKind::Stub.identity(),
            ComputeAuditStatus::FixtureStub,
            "zero-link-digest-test",
        )
        .expect_err("zero link digest rejected"),
        ComputeAuditRecordError::ZeroComputeOutputLinkDigest
    );
    assert_eq!(
        ComputeAuditRecord::from_backend_identity(
            output_digest,
            link_digest,
            [0; 32],
            ComputeBackendKind::Stub.identity(),
            ComputeAuditStatus::FixtureStub,
            "zero-compute-result-digest-test",
        )
        .expect_err("zero compute result digest rejected"),
        ComputeAuditRecordError::ZeroComputeResultDigest
    );
}

#[test]
fn compute_audit_record_does_not_append_or_mutate_evidence_archive() {
    let output_record = output_record_fixture();
    let before_bytes = canonical_bytes(&output_record);
    let fixture = run_stub_compute_fixture(&stub_fixture_input(), ComputeBudget::default())
        .expect("stub fixture");
    let link = ComputeOutputLink::derived(
        output_record_digest(&output_record),
        fixture.digest,
        ComputeBackendKind::Stub.identity(),
        "stub_compute_fixture_v1",
    );

    let audit = ComputeAuditRecord::from_stub_link(&link).expect("audit record");

    assert_eq!(canonical_bytes(&output_record), before_bytes);
    assert_metadata_only(&audit);
    assert!(!audit.evidence_authority);
    assert!(!audit.output_authority);
}

#[test]
fn minimal_spine_e2e_still_has_no_compute_dependency() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let minimal_spine_test = manifest_dir
        .join("../../core/crates/ucf-router/tests/minimal_spine_e2e.rs")
        .canonicalize()
        .expect("minimal spine e2e path");
    let source = std::fs::read_to_string(minimal_spine_test).expect("minimal spine e2e source");

    assert!(!source.contains("ucf_compute"));
    assert!(!source.contains("ucf-compute"));
}
