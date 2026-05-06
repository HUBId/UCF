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
fn br5_cerebellum_role_map_pins_canonical_roles_and_current_mode() {
    let doc = read_doc("docs/blue_brain_cerebellum_region_role_map_serie_br5_prompt1_v1.md");

    assert_contains_all(
        &doc,
        &[
            "cerebellum_like_region",
            "prediction-/timing-/correction-/mismatch-nah",
            "prediction role",
            "timing/coordination role",
            "error-correction or mismatch-shaping role",
            "bounded execution-support role",
            "non-role / out-of-scope biological detail",
            "abstract functional current mode",
            "bounded Kuramoto-like candidate",
            "Hodgkin-Huxley simulation-only/diagnostic-only",
            "later selective HH deepening",
            "keine implizite HH-Pflicht",
        ],
    );
}

#[test]
fn br5_cerebellum_bluebrain_attachment_stays_no_direct_and_bounded() {
    let doc = read_doc("docs/blue_brain_cerebellum_region_role_map_serie_br5_prompt1_v1.md");

    assert_contains_all(
        &doc,
        &[
            "Prediction/Timing/Correction",
            "Bounded dynamics",
            "Execution-interface/Eligibility/Safety",
            "Runtime/Selection",
            "Reference/Context",
            "bounded advisory/diagnostic Contract",
            "no direct action trigger",
            "no direct action selection",
            "no direct action execution",
            "no direct execution trigger",
            "no retry trigger",
            "no retry orchestration",
            "no direct memory commit",
            "no direct compute invocation",
            "no safety override",
        ],
    );
}

#[test]
fn br5_cerebellum_separates_all_established_region_roles_without_scope_expansion() {
    let doc = read_doc("docs/blue_brain_cerebellum_region_role_map_serie_br5_prompt1_v1.md");

    assert_contains_all(
        &doc,
        &[
            "hippocampus_like_region`: context/reference/episode/indexing-lastig",
            "amygdala_like_region`: salience/valence/caveat/priority-lastig",
            "thalamus_like_region`: relay/gating/routing-lastig",
            "basal_ganglia_like_region`: action-gating/suppression/channel-selection-lastig",
            "cerebellum_like_region`: prediction/timing/correction/mismatch-lastig",
            "Hypothalamus bleibt deferred",
            "keine semantische Dublette",
            "kein vollständiger biologischer Cerebellum-Nachbau",
            "keine HH-Produktivintegration",
            "keine neue Compute-Core-Arbeit",
            "keine globale Neurodynamikplattform",
        ],
    );
}

#[test]
fn docs_readme_indexes_br5_cerebellum_prompt1_role_map() {
    let readme = read_doc("docs/README.md");

    assert_contains_all(
        &readme,
        &[
            "Cerebellum-next role consolidation (BR5)",
            "docs/blue_brain_cerebellum_region_role_map_serie_br5_prompt1_v1.md",
        ],
    );
}
