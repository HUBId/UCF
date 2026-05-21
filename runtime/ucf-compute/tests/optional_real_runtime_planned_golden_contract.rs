use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use ucf_compute::{BackendClass, BackendPackKind};

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

#[test]
fn planned_golden_digest_matches_expected_output_fixture() {
    let manifest = read_manifest();
    let expected_output_path = fixture_dir().join(
        manifest["fixture"]["expected_output"]["path"]
            .as_str()
            .expect("expected output path must exist"),
    );
    let expected_digest = manifest["fixture"]["expected_output"]["sha256"]
        .as_str()
        .expect("expected output digest must exist");

    assert_eq!(sha256_hex(&expected_output_path), expected_digest);
}

#[test]
fn planned_golden_is_stable_across_repeated_reads() {
    let manifest = read_manifest();
    let expected_output_path = fixture_dir().join(
        manifest["fixture"]["expected_output"]["path"]
            .as_str()
            .expect("expected output path must exist"),
    );

    let first = sha256_hex(&expected_output_path);
    let second = sha256_hex(&expected_output_path);

    assert_eq!(first, second);
}

#[test]
fn planned_golden_is_not_runtime_inference() {
    let manifest = read_manifest();
    let readme = fs::read_to_string(fixture_dir().join("README.md")).expect("readme must exist");
    let output_note = manifest["fixture"]["expected_output"]["note"]
        .as_str()
        .expect("expected output note must exist")
        .to_ascii_lowercase();
    let readme_lower = readme.to_ascii_lowercase();

    assert!(output_note.contains("planned"));
    assert!(output_note.contains("not runtime proof"));
    assert!(readme_lower.contains("metadata-only"));
    assert!(readme_lower.contains("no runtime inference execution"));
    assert!(readme_lower.contains("no optionalrealruntime activation"));
}

#[test]
fn planned_golden_does_not_promote_current_backends() {
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
fn planned_golden_contract_has_no_prod_claim() {
    let manifest = read_manifest();

    assert_eq!(manifest["production_claim"].as_bool(), Some(false));
    assert_eq!(manifest["gateway_visible"].as_bool(), Some(false));
    assert_eq!(manifest["policy_mutating"].as_bool(), Some(false));
    assert_eq!(manifest["external_service_required"].as_bool(), Some(false));
}
