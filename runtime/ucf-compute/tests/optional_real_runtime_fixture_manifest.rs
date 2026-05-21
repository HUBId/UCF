use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use ucf_compute::{
    BackendClass, BackendPackKind, OptionalRealRuntimeArtifactSpec,
    OptionalRealRuntimeCandidateContract, OptionalRealRuntimeFixtureSpec,
};

const FIXTURE_DIR: &str = "tests/fixtures/optional_real_runtime";
const ARTIFACT_SIZE_CAP_BYTES: u64 = 256 * 1024;

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

#[test]
fn optional_real_runtime_fixture_manifest_is_well_formed() {
    let manifest = read_manifest();
    assert_eq!(manifest["schema_version"].as_u64(), Some(1));
    assert!(manifest["artifact"]["artifact_id"].as_str().is_some());
    assert!(manifest["artifact"]["artifact_kind"].as_str().is_some());
    assert!(manifest["artifact"]["sha256"].as_str().is_some());
    assert!(manifest["artifact"]["size_bytes"].as_u64().is_some());
    assert!(manifest["fixture"]["fixture_id"].as_str().is_some());
    assert!(manifest["fixture"]["input_sha256"].as_str().is_some());
    assert!(manifest["fixture"]["expected_output"]["sha256"]
        .as_str()
        .is_some());
}

#[test]
fn optional_real_runtime_fixture_artifact_hash_matches_file() {
    let manifest = read_manifest();
    let artifact_path = fixture_dir().join(manifest["artifact"]["path"].as_str().unwrap());
    let expected = manifest["artifact"]["sha256"].as_str().unwrap();
    assert_eq!(sha256_hex(&artifact_path), expected);
}

#[test]
fn optional_real_runtime_fixture_input_hash_matches_file() {
    let manifest = read_manifest();
    let input_path = fixture_dir().join(manifest["fixture"]["input_path"].as_str().unwrap());
    let expected = manifest["fixture"]["input_sha256"].as_str().unwrap();
    assert_eq!(sha256_hex(&input_path), expected);
}

#[test]
fn optional_real_runtime_fixture_expected_output_hash_matches_file() {
    let manifest = read_manifest();
    let path = fixture_dir().join(
        manifest["fixture"]["expected_output"]["path"]
            .as_str()
            .unwrap(),
    );
    let expected = manifest["fixture"]["expected_output"]["sha256"]
        .as_str()
        .unwrap();
    assert_eq!(sha256_hex(&path), expected);
    assert!(manifest["fixture"]["expected_output"]["note"]
        .as_str()
        .unwrap()
        .contains("not runtime proof"));
}

#[test]
fn optional_real_runtime_fixture_enforces_size_bound() {
    let manifest = read_manifest();
    let artifact_path = fixture_dir().join(manifest["artifact"]["path"].as_str().unwrap());
    let metadata = fs::metadata(artifact_path).expect("artifact metadata readable");
    let size = metadata.len();
    assert!(size <= ARTIFACT_SIZE_CAP_BYTES);
    assert_eq!(manifest["artifact"]["size_bytes"].as_u64(), Some(size));
}

#[test]
fn optional_real_runtime_fixture_is_local_offline_only() {
    let manifest = read_manifest();
    assert_eq!(manifest["artifact"]["local_only"].as_bool(), Some(true));
    assert_eq!(
        manifest["artifact"]["network_required"].as_bool(),
        Some(false)
    );
    assert_eq!(manifest["offline_by_default"].as_bool(), Some(true));
    assert_eq!(manifest["external_service_required"].as_bool(), Some(false));
}

#[test]
fn optional_real_runtime_fixture_forbids_prod_gateway_policy_claims() {
    let manifest = read_manifest();
    assert_eq!(manifest["production_claim"].as_bool(), Some(false));
    assert_eq!(manifest["gateway_visible"].as_bool(), Some(false));
    assert_eq!(manifest["policy_mutating"].as_bool(), Some(false));
}

#[test]
fn optional_real_runtime_fixture_contract_metadata_is_valid() {
    let manifest = read_manifest();
    let contract = OptionalRealRuntimeCandidateContract {
        backend: ucf_compute::BackendIdentity::optional_real_runtime("local_fixture_runtime"),
        artifact: OptionalRealRuntimeArtifactSpec {
            artifact_id: "tiny_local_artifact_v1",
            artifact_kind: "synthetic-binary",
            artifact_digest: hex32(manifest["artifact"]["sha256"].as_str().unwrap()),
            artifact_size_bytes: manifest["artifact"]["size_bytes"].as_u64().unwrap(),
            source_note:
                "synthetic local fixture artifact generated for metadata/manifest validation only",
            license_note: "test-only synthetic fixture",
            local_only: manifest["artifact"]["local_only"].as_bool().unwrap(),
            network_required: manifest["artifact"]["network_required"].as_bool().unwrap(),
        },
        fixture: OptionalRealRuntimeFixtureSpec {
            fixture_id: "optional_real_runtime_fixture_v1",
            input_digest: hex32(manifest["fixture"]["input_sha256"].as_str().unwrap()),
            expected_output_digest: hex32(
                manifest["fixture"]["expected_output"]["sha256"]
                    .as_str()
                    .unwrap(),
            ),
            deterministic: manifest["fixture"]["deterministic"].as_bool().unwrap(),
            max_runtime_ms: manifest["fixture"]["max_runtime_ms"].as_u64().unwrap(),
            max_memory_bytes: manifest["fixture"]["max_memory_bytes"].as_u64().unwrap(),
        },
        offline_by_default: manifest["offline_by_default"].as_bool().unwrap(),
        external_service_required: manifest["external_service_required"].as_bool().unwrap(),
    };
    contract
        .validate()
        .expect("manifest metadata must satisfy OptionalRealRuntime contract");
}

#[test]
fn optional_real_runtime_fixture_does_not_promote_current_backends() {
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

#[test]
fn optional_real_runtime_fixture_is_not_runtime_inference() {
    let readme = fs::read_to_string(fixture_dir().join("README.md")).expect("readme must exist");
    assert!(readme.contains("No runtime inference execution"));
    assert!(readme.contains("No OptionalRealRuntime activation"));
    assert!(readme.contains("No production-readiness claim"));
}
