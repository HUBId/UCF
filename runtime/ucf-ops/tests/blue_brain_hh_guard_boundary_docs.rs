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
fn hh_guard_boundary_doc_pins_no_direct_barriers_at_hh_level() {
    let doc = read_doc("docs/blue_brain_hh_guard_boundary_map_v1.md");

    assert_contains_all(
        &doc,
        &[
            "HH guard boundary map",
            "Basal Ganglia ↔ Cerebellum",
            "HH-level no-direct barriers are pinned",
            "no direct action trigger",
            "no direct execution trigger",
            "no direct retry trigger",
            "no direct memory commit",
            "no direct compute invocation",
            "no safety override",
            "fail closed",
        ],
    );
}

#[test]
fn hh_guard_boundary_doc_separates_contract_state_and_authority() {
    let doc = read_doc("docs/blue_brain_hh_guard_boundary_map_v1.md");

    assert_contains_all(
        &doc,
        &[
            "HH-state is diagnostic/simulation state only",
            "HH-state is not contract-state",
            "HH diagnostic output is evidence only",
            "not operative authority",
            "no HH-based Runtime authority",
            "no HH-based Selection authority",
            "no HH-based Reference mutation authority",
            "no HH-based Execution authority",
            "scope drift",
        ],
    );
}

#[test]
fn hh_guard_boundary_doc_is_indexed_and_linked_from_scope_map() {
    let readme = read_doc("docs/README.md");
    let authority = read_doc("docs/blue_brain_authority_chain_status_map.md");
    let scope = read_doc("docs/blue_brain_hh_candidate_scope_map_v1.md");

    assert!(readme.contains("docs/blue_brain_hh_guard_boundary_map_v1.md"));
    assert!(authority.contains("docs/blue_brain_hh_guard_boundary_map_v1.md"));
    assert!(scope.contains("docs/blue_brain_hh_guard_boundary_map_v1.md"));
    assert!(scope.contains("HH-state is not Contract state"));
    assert!(scope.contains("HH diagnostic output is not operative authority"));
}
