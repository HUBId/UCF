use std::fs;
use std::path::{Path, PathBuf};

fn has_workspace_manifest(path: &Path) -> bool {
    let manifest = path.join("Cargo.toml");
    let Ok(contents) = fs::read_to_string(manifest) else {
        return false;
    };
    contents.contains("[workspace]")
}

fn repo_root() -> PathBuf {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate in start.ancestors() {
        if has_workspace_manifest(candidate) {
            return candidate.to_path_buf();
        }
    }

    panic!(
        "failed to locate workspace root from CARGO_MANIFEST_DIR={}",
        env!("CARGO_MANIFEST_DIR")
    );
}

fn read_doc(rel: &str) -> String {
    fs::read_to_string(repo_root().join(rel)).unwrap_or_else(|err| panic!("read {rel}: {err}"))
}

fn assert_contains_all(doc: &str, required: &[&str]) {
    for needle in required {
        assert!(doc.contains(needle), "missing `{needle}`");
    }
}

#[test]
fn br6_hypothalamus_readiness_map_keeps_surface_diagnostics_contract_and_model_classes_distinct() {
    let doc = read_doc(
        "docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "BR6-expansion-readiness map",
            "stable hypothalamus operational surface",
            "usable with caveats",
            "advisory-only",
            "deferred/blocked/insufficient/diagnostic-only/reference-only",
            "stable current model mode",
            "non-canonical/internal-only",
            "hypothalamus input surface",
            "hypothalamus state surface",
            "hypothalamus output/advisory surface",
            "hypothalamus reference surface",
            "hypothalamus diagnostics states",
            "hypothalamus contract signals",
            "abstract functional current mode",
        ],
    );

    assert_contains_all(
        &doc,
        &[
            "hypothalamus input surface` is not `hypothalamus state surface",
            "hypothalamus state surface` is not `hypothalamus output/advisory surface",
            "hypothalamus output/advisory surface` is not `hypothalamus reference surface",
            "hypothalamus diagnostics states` are not `hypothalamus contract signals",
            "hypothalamus bounded contract signal` is not an action, execution, retry, memory, compute, policy, planner, agent, or safety channel",
            "hypothalamus advisory-only diagnostic != hypothalamus caveated diagnostic",
            "hypothalamus deferred diagnostic != hypothalamus blocked diagnostic",
            "hypothalamus blocked diagnostic != hypothalamus insufficient diagnostic",
            "hypothalamus diagnostic-only state != hypothalamus advisory-only diagnostic",
            "reference-only` remains read-only and non-actionable",
        ],
    );
}

#[test]
fn br6_hypothalamus_closeout_preserves_no_direct_out_of_scope_and_bluebrain_contract_lines() {
    let doc = read_doc(
        "docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "no direct action execution",
            "no direct action trigger",
            "no direct action selection",
            "no direct execution trigger",
            "no retry orchestration or retry trigger",
            "no planner/agent logic",
            "no automatic memory persistence, mutation, or commit",
            "no safety override semantics",
            "no allowed-actions extension",
            "no direct compute invocation",
            "no new compute-core work",
            "no seventh anatomical region opened in this step",
            "no productive Hodgkin-Huxley integration",
        ],
    );

    assert_contains_all(
        &doc,
        &[
            "BB2 runtime transition/feedback",
            "BB4 selection/priority/deferral",
            "BB8 and BB17 context/memory/reference hardening",
            "BB12 bounded dynamics",
            "BB19 runtime/selection contract line",
            "BB21 execution/reference interaction",
            "finale Compute-Linie",
            "maintenance-only Core",
        ],
    );
}

#[test]
fn br6_hypothalamus_model_boundary_separates_abstract_kuramoto_hh_and_hh_later() {
    let doc = read_doc(
        "docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "abstract functional current mode` is not `bounded Kuramoto-like candidate",
            "bounded Kuramoto-like candidate` is not `Hodgkin-Huxley simulation-only/diagnostic-only",
            "Hodgkin-Huxley simulation-only/diagnostic-only` is not `later selective HH deepening` and not productive HH integration",
            "HH-later",
            "kein Kuramoto- oder Hodgkin-Huxley-Pfad wird produktiv",
        ],
    );
}

#[test]
fn br6_next_direction_prioritizes_system_audit_without_opening_region_seven() {
    let doc = read_doc(
        "docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "Priorisiert wird genau eine nächste Richtung: **System-Audit/Consolidation-Pass**.",
            "Sechs anatomische Regionen sind jetzt vorhanden",
            "Eine weitere Hirnregion muss warten",
            "HH-lastigere oder schwerere Modellschritte warten",
            "weitere anatomische Expansion ist bewusst gestoppt",
        ],
    );
}

#[test]
fn br6_prompt3_points_to_prompt4_closeout_and_readme_indexes_closure_reference() {
    let prompt3 = read_doc(
        "docs/blue_brain_hypothalamus_surface_diagnostics_contracts_hardening_serie_br6_prompt3_v1.md",
    );
    assert_contains_all(
        &prompt3,
        &[
            "docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md",
            "Prompt-4-Datei bleibt die kanonische BR6-Abschluss- und Expansionsgrenze",
            "Keine nächste echte Hirnregion aus Prompt 3 heraus öffnen",
        ],
    );

    let readme = read_doc("docs/README.md");
    assert_contains_all(
        &readme,
        &[
            "Hypothalamus-next integration line (BR6)",
            "docs/blue_brain_hypothalamus_region_role_map_serie_br6_prompt1_v1.md",
            "docs/blue_brain_hypothalamus_minimal_bounded_integration_serie_br6_prompt2_v1.md",
            "docs/blue_brain_hypothalamus_surface_diagnostics_contracts_hardening_serie_br6_prompt3_v1.md",
            "docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md",
        ],
    );
}
