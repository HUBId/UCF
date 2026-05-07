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
fn ir1_doc_pins_canonical_relation_classes_and_pair_map() {
    let doc = read_doc("docs/blue_brain_inter_region_architecture_serie_ir1_prompt1_v1.md");

    assert_contains_all(
        &doc,
        &[
            "direct bounded advisory relation",
            "reference-mediated relation",
            "selection-mediated relation",
            "execution-interface-mediated relation",
            "caveated inter-region relation",
            "deferred/not-yet-active relation",
            "blocked relation",
            "non-canonical/internal-only relation path",
            "Hippocampus ↔ Amygdala",
            "Hippocampus ↔ Thalamus",
            "Hippocampus ↔ Basal Ganglia",
            "Hippocampus ↔ Cerebellum",
            "Amygdala ↔ Thalamus",
            "Amygdala ↔ Basal Ganglia",
            "Amygdala ↔ Cerebellum",
            "Thalamus ↔ Basal Ganglia",
            "Thalamus ↔ Cerebellum",
            "Basal Ganglia ↔ Cerebellum",
        ],
    );
}

#[test]
fn ir1_doc_pins_no_direct_and_model_boundaries() {
    let doc = read_doc("docs/blue_brain_inter_region_architecture_serie_ir1_prompt1_v1.md");

    assert_contains_all(
        &doc,
        &[
            "advisory-only relation is not strong authority",
            "caveated relation is not stable relation",
            "deferred relation is not blocked relation",
            "blocked relation is not failed execution",
            "reference-mediated relation is not direct inter-region authority",
            "no direct action trigger",
            "no direct execution trigger",
            "no direct retry trigger",
            "no retry orchestration",
            "no direct memory commit",
            "no automatic memory persistence",
            "no direct compute invocation",
            "no safety override",
            "no implicit global region orchestration",
            "no new inter-region platform formation",
            "abstract functional current mode",
            "Hodgkin-Huxley production integration",
        ],
    );
}

#[test]
fn docs_readme_indexes_ir1_inter_region_architecture() {
    let readme = read_doc("docs/README.md");

    assert_contains_all(
        &readme,
        &[
            "Inter-region architecture consolidation (IR1)",
            "docs/blue_brain_inter_region_architecture_serie_ir1_prompt1_v1.md",
        ],
    );
}
