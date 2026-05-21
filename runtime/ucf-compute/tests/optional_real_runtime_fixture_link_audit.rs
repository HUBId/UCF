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

fn read_manifest() -> serde_json::Value {
    let bytes =
        fs::read(fixture_dir().join("fixture_manifest.json")).expect("manifest file must exist");
    serde_json::from_slice(&bytes).expect("manifest json must parse")
}

fn sha256_hex(path: &Path) -> String {
    let bytes = fs::read(path).expect("fixture file must exist");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn hex32(s: &str) -> [u8; 32] {
    let bytes = hex::decode(s).expect("sha256 must be valid lowercase hex");
    bytes.try_into().expect("sha256 must be 32 bytes")
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hasher.finalize().into()
}

#[test]
fn optional_real_runtime_fixture_link_and_audit_are_metadata_only() {
    let manifest = read_manifest();
    let fixture_manifest_digest = digest_bytes(
        b"ucf.compute.optional_real_runtime.fixture_manifest.v1",
        &fs::read(fixture_dir().join("fixture_manifest.json")).expect("manifest bytes"),
    );
    let artifact_digest = hex32(
        manifest["artifact"]["sha256"]
            .as_str()
            .expect("artifact digest"),
    );
    let input_digest = hex32(
        manifest["fixture"]["input_sha256"]
            .as_str()
            .expect("input digest"),
    );
    let planned_expected_output_digest = hex32(
        manifest["fixture"]["expected_output"]["sha256"]
            .as_str()
            .expect("expected output digest"),
    );

    let expected_output_path = fixture_dir().join(
        manifest["fixture"]["expected_output"]["path"]
            .as_str()
            .expect("expected output path"),
    );
    assert_eq!(
        sha256_hex(&expected_output_path),
        hex::encode(planned_expected_output_digest)
    );

    let fixture_output_reference_digest = digest_bytes(
        b"ucf.compute.optional_real_runtime.fixture_output_reference.v1",
        &[
            &fixture_manifest_digest[..],
            &artifact_digest[..],
            &input_digest[..],
        ]
        .concat(),
    );
    assert_ne!(fixture_output_reference_digest, [0; 32]);

    let backend_identity =
        BackendIdentity::optional_real_compile("optional_real_runtime_fixture_metadata");
    assert_eq!(backend_identity.class, BackendClass::OptionalRealCompile);
    assert!(!backend_identity.claims_runtime_real_inference());
    assert!(!backend_identity.production_claim);

    let link = ComputeOutputLink::derived(
        fixture_output_reference_digest,
        planned_expected_output_digest,
        backend_identity,
        "optional_real_runtime_fixture_manifest_planned_golden_v1",
    )
    .with_output_record_id("fixture_output_reference_digest")
    .with_output_record_bytes_digest(fixture_manifest_digest);

    assert!(link.metadata_only);
    assert!(!link.output_record_authority);
    assert!(!link.minimal_spine_required);
    assert_eq!(link.compute_result_digest, planned_expected_output_digest);
    assert_eq!(link.output_record_digest, fixture_output_reference_digest);
    assert_eq!(link.backend_class, BackendClass::OptionalRealCompile);
    assert!(link.no_real_runtime);
    assert!(!link.runtime_inference_supported);
    assert!(!link.production_claim);

    let link_digest = link.link_digest();
    assert_eq!(link_digest, link.clone().link_digest());

    let audit = ComputeAuditRecord::from_link(
        &link,
        ComputeAuditStatus::RuntimeDeferred,
        "optional-real-runtime-fixture-planned-golden-metadata-only",
    )
    .expect("audit from planned-golden fixture link");

    assert_eq!(audit.audit_status, ComputeAuditStatus::RuntimeDeferred);
    assert_eq!(audit.compute_output_link_digest, link_digest);
    assert_eq!(audit.compute_result_digest, planned_expected_output_digest);
    assert!(!audit.runtime_inference_claim);
    assert!(!audit.production_claim);
    assert!(!audit.evidence_authority);
    assert!(!audit.output_authority);
    assert!(!audit.minimal_spine_required);
    assert!(audit.metadata_only());

    let audit_digest = audit.audit_digest();
    assert_eq!(audit_digest, audit.clone().audit_digest());

    let readme = fs::read_to_string(fixture_dir().join("README.md")).expect("readme must exist");
    assert!(readme.contains("No runtime inference execution"));
    assert!(readme.contains("No OptionalRealRuntime activation"));
    assert!(readme.contains("No production-readiness claim"));
}
