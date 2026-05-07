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
fn ir1_prompt2_doc_pins_exact_three_implemented_relations() {
    let doc = read_doc("docs/blue_brain_first_inter_region_implementation_serie_ir1_prompt2_v1.md");

    assert_contains_all(
        &doc,
        &[
            "exactly three implemented relations",
            "implemented direct bounded advisory relation",
            "implemented reference-mediated relation",
            "implemented selection-mediated relation",
            "Amygdala ↔ Thalamus",
            "Hippocampus ↔ Thalamus",
            "Amygdala ↔ Basal Ganglia",
            "DirectBoundedAdvisoryOnly",
            "ReferenceContextMediatedOnly",
            "SelectionContractMediatedOnly",
            "SalienceCaveatAdvisory",
            "RelayRoutingDiagnostic",
            "ContextReferenceDiagnostic",
            "SelectionReadinessDiagnostic",
        ],
    );
}

#[test]
fn ir1_prompt2_doc_keeps_deferred_blocked_and_no_direct_boundaries() {
    let doc = read_doc("docs/blue_brain_first_inter_region_implementation_serie_ir1_prompt2_v1.md");

    assert_contains_all(
        &doc,
        &[
            "Hippocampus ↔ Amygdala",
            "Hippocampus ↔ Basal Ganglia",
            "Hippocampus ↔ Cerebellum",
            "Amygdala ↔ Cerebellum",
            "Thalamus ↔ Basal Ganglia",
            "Thalamus ↔ Cerebellum",
            "Basal Ganglia ↔ Cerebellum",
            "deferred/not-yet-implemented relation",
            "blocked relation",
            "no direct action trigger",
            "no direct execution trigger",
            "no direct retry trigger",
            "no retry orchestration",
            "no direct memory commit",
            "no automatic memory persistence",
            "no direct compute invocation",
            "no safety override",
            "no new inter-region platform formation",
            "no model-mode change",
            "no Kuramoto expansion",
            "no Hodgkin-Huxley production integration",
        ],
    );
}

#[test]
fn docs_readme_indexes_ir1_first_inter_region_implementation() {
    let readme = read_doc("docs/README.md");
    let repo_map = read_doc("docs/roadmap/REPO_MAP.md");

    assert_contains_all(
        &readme,
        &[
            "Inter-region architecture consolidation (IR1)",
            "docs/blue_brain_first_inter_region_implementation_serie_ir1_prompt2_v1.md",
        ],
    );
    assert_contains_all(
        &repo_map,
        &[
            "Inter-region architecture consolidation (IR1)",
            "docs/blue_brain_first_inter_region_implementation_serie_ir1_prompt2_v1.md",
        ],
    );
}
