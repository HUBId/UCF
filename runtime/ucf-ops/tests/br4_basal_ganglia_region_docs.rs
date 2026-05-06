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
fn br4_basal_ganglia_role_map_pins_canonical_roles_and_current_mode() {
    let doc = read_doc("docs/blue_brain_basal_ganglia_region_role_map_serie_br4_prompt1_v1.md");

    assert_contains_all(
        &doc,
        &[
            "basal_ganglia_like_region",
            "action gating role",
            "suppression/inhibition role",
            "bounded selection-channel arbitration role",
            "execution-readiness modulation role",
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
fn br4_basal_ganglia_bluebrain_attachment_stays_no_direct_and_bounded() {
    let doc = read_doc("docs/blue_brain_basal_ganglia_region_role_map_serie_br4_prompt1_v1.md");

    assert_contains_all(
        &doc,
        &[
            "Selection/Action-gating",
            "Priority/Deferral",
            "Execution-interface/Eligibility/Safety",
            "Reference/Context",
            "Runtime",
            "no direct action trigger",
            "no direct action selection",
            "no direct execution trigger",
            "no direct retry trigger",
            "no direct memory commit",
            "no direct compute invocation",
            "no safety override",
        ],
    );
}

#[test]
fn br4_basal_ganglia_separates_region_roles_without_scope_expansion() {
    let doc = read_doc("docs/blue_brain_basal_ganglia_region_role_map_serie_br4_prompt1_v1.md");

    assert_contains_all(
        &doc,
        &[
            "hippocampus_like_region`: context/reference/episode/indexing-lastig",
            "amygdala_like_region`: salience/valence/caveat/priority-lastig",
            "thalamus_like_region`: relay/gating/routing-lastig",
            "basal_ganglia_like_region`: action-gating/suppression/channel-selection-lastig",
            "semantische Dublette",
            "kein vollständiger biologischer Basal-Ganglia-Nachbau",
            "keine HH-Produktivintegration",
            "keine direkte Action-Auswahl",
            "keine Planner-/Agenten-/Policy-/Governance-/Retry-/Queue-/Orchestration-Plattform",
            "keine Retrieval-/Consolidation-/Reasoning-Plattform",
            "keine implizite Memory-Persistenz",
            "keine neue Compute-Core-Arbeit",
            "keine globale Neurodynamikplattform",
        ],
    );
}

#[test]
fn br4_basal_ganglia_handoff_prioritizes_basal_before_cerebellum_and_readme_indexes_doc() {
    let doc = read_doc("docs/blue_brain_basal_ganglia_region_role_map_serie_br4_prompt1_v1.md");
    assert_contains_all(
        &doc,
        &[
            "Basal Ganglia vor Cerebellum",
            "Cerebellum bleibt eher Kalibrierungs-/Timing-Kandidat",
            "BR4 baut keine neue Plattform",
            "BR4-Nächste Schritte",
        ],
    );

    let readme = read_doc("docs/README.md");
    assert_contains_all(
        &readme,
        &[
            "Basal-Ganglia-next role consolidation (BR4)",
            "docs/blue_brain_basal_ganglia_region_role_map_serie_br4_prompt1_v1.md",
        ],
    );
}

#[test]
fn br4_basal_ganglia_prompt3_hardens_diagnostics_contracts_and_reads() {
    let doc = read_doc(
        "docs/blue_brain_basal_ganglia_surface_diagnostics_contracts_hardening_serie_br4_prompt3_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "basal-ganglia input surface",
            "basal-ganglia state surface",
            "basal-ganglia output/advisory surface",
            "basal-ganglia reference surface",
            "basal-ganglia advisory-only diagnostic",
            "basal-ganglia caveated diagnostic",
            "basal-ganglia deferred diagnostic",
            "basal-ganglia blocked diagnostic",
            "basal-ganglia insufficient diagnostic",
            "basal-ganglia diagnostic-only state",
            "basal-ganglia bounded contract signal",
            "non-canonical/internal-only basal-ganglia path",
        ],
    );

    assert_contains_all(
        &doc,
        &[
            "Runtime, Selection und Reference lesen Basal Ganglia nur über denselben kanonischen bounded contract read",
            "basal-ganglia advisory-only diagnostic != basal-ganglia caveated diagnostic",
            "basal-ganglia deferred diagnostic != basal-ganglia blocked diagnostic",
            "basal-ganglia blocked diagnostic != basal-ganglia insufficient diagnostic",
            "reference-only/read-only Basal-Ganglia-Sichtbarkeit",
            "kein Action-Kanal, kein Execution-Kanal, kein Retry-Kanal und kein Memory-/Compute-Kanal",
        ],
    );
}

#[test]
fn br4_basal_ganglia_prompt3_preserves_model_region_and_no_direct_boundaries() {
    let doc = read_doc(
        "docs/blue_brain_basal_ganglia_surface_diagnostics_contracts_hardening_serie_br4_prompt3_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "current model mode remains unchanged",
            "abstract functional current mode",
            "bounded Kuramoto-like candidate",
            "Hodgkin-Huxley simulation-only/diagnostic-only",
            "later selective HH deepening",
            "HH-later",
            "keine Modell-Drift",
            "hippocampus_like_region` bleibt context/reference/episode/indexing-lastig",
            "amygdala_like_region` bleibt salience/valence/caveat/priority-lastig",
            "thalamus_like_region` bleibt relay/gating/routing-lastig",
            "basal_ganglia_like_region` bleibt action-gating/suppression/channel-selection-lastig",
            "keine semantische Dublette",
        ],
    );

    assert_contains_all(
        &doc,
        &[
            "no action request",
            "no action selection",
            "no execution trigger",
            "no retry trigger",
            "no memory commit",
            "no compute trigger",
            "no safety override",
            "keine allowed-actions-Flächen",
            "keine Compute-Core-Arbeit",
        ],
    );
}

#[test]
fn docs_readme_indexes_br4_basal_ganglia_prompt3_contract_hardening() {
    let readme = read_doc("docs/README.md");

    assert_contains_all(
        &readme,
        &[
            "Basal-Ganglia-next role consolidation (BR4)",
            "docs/blue_brain_basal_ganglia_region_role_map_serie_br4_prompt1_v1.md",
            "docs/blue_brain_basal_ganglia_minimal_bounded_integration_serie_br4_prompt2_v1.md",
            "docs/blue_brain_basal_ganglia_surface_diagnostics_contracts_hardening_serie_br4_prompt3_v1.md",
        ],
    );
}
