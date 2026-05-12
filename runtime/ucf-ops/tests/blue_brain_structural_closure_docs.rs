use std::{fs, path::Path};

fn read_doc(path: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path),
    )
    .unwrap_or_else(|err| panic!("failed to read {path}: {err}"))
}

fn assert_contains_all(doc: &str, expected: &[&str]) {
    for needle in expected {
        assert!(doc.contains(needle), "missing expected text: {needle}");
    }
}

#[test]
fn structural_closure_doc_pins_region_relation_model_statuses() {
    let doc = read_doc("docs/blue_brain_structural_closure_map_v1.md");

    assert_contains_all(
        &doc,
        &[
            "canonical active region",
            "canonical implemented relation",
            "canonical mediated relation",
            "canonical model boundary",
            "deferred/blocked/non-active",
            "non-canonical/internal-only",
            "Hippocampus",
            "Amygdala",
            "Thalamus",
            "Basal Ganglia",
            "Cerebellum",
            "Hypothalamus",
            "Amygdala ↔ Thalamus",
            "Amygdala ↔ Basal Ganglia",
            "bounded Kuramoto-like",
            "HH simulation-only/diagnostic-only",
            "later-HH/deferred",
        ],
    );
}

#[test]
fn structural_closure_doc_pins_no_direct_and_hh_readiness_decision() {
    let doc = read_doc("docs/blue_brain_structural_closure_map_v1.md");

    assert_contains_all(
        &doc,
        &[
            "no direct action trigger",
            "no direct execution trigger",
            "no direct retry trigger",
            "no direct memory commit",
            "no direct compute invocation",
            "no safety override",
            "no implicit new region",
            "no implicit new model-deepening candidate",
            "no global model platform",
            "no global region orchestration",
            "the Blue-Brain structure phase is closed enough for maintenance status and for a separate HH-Readiness block",
            "not HH implementation",
        ],
    );
}

#[test]
fn structural_closure_doc_is_indexed_by_authority_and_readme() {
    let readme = read_doc("docs/README.md");
    let authority = read_doc("docs/blue_brain_authority_chain_status_map.md");

    assert!(readme.contains("docs/blue_brain_structural_closure_map_v1.md"));
    assert!(authority.contains("docs/blue_brain_structural_closure_map_v1.md"));
}

#[test]
fn final_canonical_matrices_freeze_doc_is_indexed_and_pins_counts() {
    let doc = read_doc("docs/blue_brain_canonical_matrices_final_freeze_v1.md");
    let readme = read_doc("docs/README.md");
    let authority = read_doc("docs/blue_brain_authority_chain_status_map.md");

    assert_contains_all(
        &doc,
        &[
            "Finale kanonische Regionenmatrix",
            "Finale kanonische Relationsmatrix",
            "Finale kanonische Modellmatrix",
            "Exactly six canonical active regions",
            "exactly three implemented",
            "exactly four mediated",
            "exactly two bounded Kuramoto-like",
            "HH simulation-only/diagnostic-only",
            "later-HH/deferred",
            "architecture-lane-only is not implementation",
            "no new region",
            "no new model deepening",
        ],
    );
    assert!(readme.contains("docs/blue_brain_canonical_matrices_final_freeze_v1.md"));
    assert!(authority.contains("docs/blue_brain_canonical_matrices_final_freeze_v1.md"));
}
