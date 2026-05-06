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
fn br3_thalamus_readiness_map_keeps_surface_diagnostics_contract_and_model_classes_distinct() {
    let doc = read_doc(
        "docs/blue_brain_br3_thalamus_readiness_sweep_expansion_boundary_serie_br3_prompt4_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "BR3-expansion-readiness map",
            "stable thalamus operational surface",
            "usable with caveats",
            "advisory-only",
            "deferred/blocked/insufficient",
            "diagnostic-only/reference-only",
            "stable current model mode",
            "non-canonical/internal-only",
            "thalamus input surface",
            "thalamus state surface",
            "thalamus output/advisory surface",
            "thalamus reference surface",
            "thalamus diagnostics states",
            "thalamus contract signals",
            "abstract functional current mode",
        ],
    );

    assert_contains_all(
        &doc,
        &[
            "thalamus input surface` is not `thalamus state surface",
            "thalamus state surface` is not `thalamus output/advisory surface",
            "thalamus diagnostics states` are not `thalamus contract signals",
            "thalamus bounded contract signal` is not an execution, retry, compute, or safety channel",
            "thalamus advisory-only diagnostic != thalamus caveated diagnostic",
            "thalamus deferred diagnostic != thalamus blocked diagnostic",
            "thalamus blocked diagnostic != thalamus insufficient diagnostic",
            "reference-only` remains read-only and non-actionable",
        ],
    );
}

#[test]
fn br3_thalamus_closeout_preserves_no_direct_out_of_scope_and_bluebrain_contract_lines() {
    let doc = read_doc(
        "docs/blue_brain_br3_thalamus_readiness_sweep_expansion_boundary_serie_br3_prompt4_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "no direct action execution",
            "no direct action selection",
            "no direct execution trigger",
            "no retry orchestration or retry trigger",
            "no planner/agent logic",
            "no automatic memory persistence, mutation, or commit",
            "no safety override semantics",
            "no direct compute invocation",
            "no new compute-core work",
            "no fourth anatomical region opened in this step",
            "no productive Hodgkin-Huxley integration",
        ],
    );

    assert_contains_all(
        &doc,
        &[
            "BB2 runtime transition/feedback",
            "BB4 selection/priority/deferral",
            "BB8 and BB17 context/memory/reference hardening",
            "BB19 runtime/selection contract line",
            "BB21 execution/reference interaction",
            "BB12 bounded dynamics",
            "final compute line",
            "maintenance-only core",
        ],
    );
}

#[test]
fn br3_next_region_decision_prioritizes_exactly_one_later_candidate_without_opening_it() {
    let doc = read_doc(
        "docs/blue_brain_br3_thalamus_readiness_sweep_expansion_boundary_serie_br3_prompt4_v1.md",
    );

    assert_contains_all(
        &doc,
        &[
            "Decision: **prioritize Cerebellum as the next single anatomical-region candidate; keep Basal Ganglia deferred.**",
            "Cerebellum can be evaluated as diagnostic/advisory calibration",
            "Basal Ganglia waits",
            "does not start Cerebellum implementation in BR3",
            "does not authorize concurrent Basal Ganglia work",
        ],
    );
}

#[test]
fn docs_readme_indexes_br3_thalamus_closure_reference() {
    let readme = read_doc("docs/README.md");

    assert_contains_all(
        &readme,
        &[
            "Thalamus-third role consolidation (BR3)",
            "docs/blue_brain_thalamus_region_role_map_serie_br3_prompt1_v1.md",
            "docs/blue_brain_thalamus_minimal_bounded_integration_serie_br3_prompt2_v1.md",
            "docs/blue_brain_thalamus_surface_diagnostics_contracts_hardening_serie_br3_prompt3_v1.md",
            "docs/blue_brain_br3_thalamus_readiness_sweep_expansion_boundary_serie_br3_prompt4_v1.md",
        ],
    );
}
