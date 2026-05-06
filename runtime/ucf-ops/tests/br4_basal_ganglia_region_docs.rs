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
