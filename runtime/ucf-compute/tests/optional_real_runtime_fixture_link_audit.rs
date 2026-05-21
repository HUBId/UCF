use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use ucf_compute::{
    BackendClass, BackendIdentity, ComputeAuditRecord, ComputeAuditStatus, ComputeOutputLink,
};

const FIXTURE_DIR: &str = "tests/fixtures/optional_real_runtime";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn manifest_path() -> PathBuf {
    fixture_dir().join("fixture_manifest.json")
}

fn read_manifest() -> serde_json::Value {
    let bytes = fs::read(manifest_path()).expect("manifest file must exist");
    serde_json::from_slice(&bytes).expect("manifest json must parse")
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn sha256_file(path: &Path) -> [u8; 32] {
    let bytes = fs::read(path).expect("fixture file must exist");
    sha256_bytes(&bytes)
}

fn hex32(s: &str) -> [u8; 32] {
    let bytes = hex::decode(s).expect("sha256 must be valid lowercase hex");
    bytes.try_into().expect("sha256 must be 32 bytes")
}

#[test]
fn optional_real_runtime_fixture_metadata_links_into_output_and_audit_without_runtime() {
    let manifest = read_manifest();
    let manifest_bytes = fs::read(manifest_path()).expect("manifest bytes");
    let manifest_digest = sha256_bytes(&manifest_bytes);
    let manifest_digest_hex = hex::encode(manifest_digest);

    let artifact_digest = hex32(manifest["artifact"]["sha256"].as_str().unwrap());
    let input_digest = hex32(manifest["fixture"]["input_sha256"].as_str().unwrap());
    let planned_expected_output_digest = hex32(
        manifest["fixture"]["expected_output"]["sha256"]
            .as_str()
            .unwrap(),
    );

    let expected_output_path = fixture_dir().join(
        manifest["fixture"]["expected_output"]["path"]
            .as_str()
            .unwrap(),
    );
    assert_eq!(
        planned_expected_output_digest,
        sha256_file(&expected_output_path)
    );

    let fixture_output_reference_digest = sha256_bytes(
        format!(
            "ucf.optional_real_runtime.fixture_output_reference.v1:{}:{}",
            manifest_digest_hex,
            manifest["fixture"]["fixture_id"].as_str().unwrap()
        )
        .as_bytes(),
    );

    let backend_identity = BackendIdentity::optional_real_compile("local_fixture_runtime_metadata");
    let link_source = format!(
        "optional_real_runtime_fixture_metadata|manifest_sha256={}|artifact_sha256={}|input_sha256={}|planned_expected_output_sha256={}",
        manifest_digest_hex,
        hex::encode(artifact_digest),
        hex::encode(input_digest),
        hex::encode(planned_expected_output_digest),
    );

    let link = ComputeOutputLink::derived(
        fixture_output_reference_digest,
        planned_expected_output_digest,
        backend_identity,
        link_source,
    )
    .with_output_record_id("fixture-output-reference-only-non-authoritative");

    assert_eq!(link.backend_class, BackendClass::OptionalRealCompile);
    assert!(link.metadata_only);
    assert!(!link.output_record_authority);
    assert!(!link.runtime_inference_supported);
    assert!(link.no_real_runtime);
    assert!(!link.production_claim);
    assert!(!link.minimal_spine_required);
    assert_ne!(link.output_record_digest, [0; 32]);
    assert_eq!(link.compute_result_digest, planned_expected_output_digest);

    let link_digest_a = link.link_digest();
    let link_digest_b = link.clone().link_digest();
    assert_eq!(link_digest_a, link_digest_b);

    let audit = ComputeAuditRecord::from_link(
        &link,
        ComputeAuditStatus::RuntimeDeferred,
        "optional-real-runtime-fixture-planned-golden-audit-v1",
    )
    .expect("audit record");

    assert_eq!(audit.backend_class, BackendClass::OptionalRealCompile);
    assert_eq!(audit.audit_status, ComputeAuditStatus::RuntimeDeferred);
    assert_eq!(audit.output_record_digest, fixture_output_reference_digest);
    assert_eq!(audit.compute_output_link_digest, link_digest_a);
    assert_eq!(audit.compute_result_digest, planned_expected_output_digest);
    assert!(!audit.runtime_inference_claim);
    assert!(!audit.production_claim);
    assert!(!audit.evidence_authority);
    assert!(!audit.output_authority);
    assert!(!audit.minimal_spine_required);
    assert!(audit.metadata_only());

    let audit_digest_a = audit.audit_digest();
    let audit_digest_b = audit.clone().audit_digest();
    assert_eq!(audit_digest_a, audit_digest_b);
}
