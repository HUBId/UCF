# Blue-Brain maintenance verification findings map v1

Status: narrow maintenance-facing verification pass for the current post-BR6/IR1/MD2/MD3 UCF Blue-Brain state. This file is supporting evidence only. It does not add a brain region, model-deepening candidate, planner/agent layer, policy/governance platform, retry/orchestration lane, memory persistence, HH production integration, allowed-actions expansion, or compute-core work.

Code anchor: `CANONICAL_BLUE_BRAIN_MAINTENANCE_VERIFICATION_FINDINGS_MAP` and `CANONICAL_BLUE_BRAIN_MAINTENANCE_VERIFICATION_FINDING_CLASS_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Verification finding classes

| Class | Maintenance meaning |
| --- | --- |
| unclear test naming | A test name hides whether it checks a region surface, relation edge, model boundary, or no-direct guard. |
| ambiguous test scope | A test assertion is valid, but the name/context can be read as broader than its bounded maintenance target. |
| missing verification hint | Documentation names a surface but does not say which local or canonical check should be read with it. |
| duplicate check reference | The same check path is repeated in nearby docs without explaining local verification versus audit evidence. |
| stale verification wording | Check wording is still technically true but reflects an older prompt/audit phase more prominently than the current maintenance state. |
| no-change-needed finding | The surface was reviewed and remains clear enough; the entry records evidence only. |

## 2) Findings map

| Surface | Finding class | Maintenance response |
| --- | --- | --- |
| Region surface tests | unclear test naming | Region tests should name the maintained anatomical surface under check, especially for Hypothalamus BR6 surfaces. |
| Inter-region relation tests | ambiguous test scope | Relation tests should name bounded inter-region edge scope and no-direct guard expectations rather than only prompt ancestry. |
| Model-deepening boundary tests | missing verification hint | MD2 remains `Amygdala ↔ Thalamus`; MD3 remains `Amygdala ↔ Basal Ganglia`; read those docs with model-boundary assertions rather than as model-platform work. |
| No-direct guard tests | no-change-needed finding | Existing no-direct action, execution, retry, memory, compute, and safety-denial assertions remain the guard verification anchors. |
| Readiness/baseline check hints | duplicate check reference | Read local targeted checks first, canonical workspace/readiness/docs checks for handoff, and `out/` audit baselines as evidence artifacts. |
| Evidence-reference docs | stale verification wording | Verification wording now distinguishes local maintenance verification from audit evidence and keeps reports/tests in one reading order. |

## 3) Check-reading order

Use one verification reality:

1. **Local maintenance verification**: targeted Blue-Brain tests or doc-pinned assertions are used to understand the exact surface being changed.
2. **Canonical handoff checks**: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, docs lint, and readiness gate are the standard clean-state checks.
3. **Audit/baseline evidence**: reports under `out/`, especially `out/docs_lint_report.json`, `out/gate_report.json`, and run-specific `out/blue_brain_audit_baseline_*` folders, are evidence snapshots. They do not override current authority docs or tests.

Reports and tests should be read together: tests pin region/relation/model/guard boundaries, while reports show repository-level readiness and documentation consistency. Neither path creates a second Blue-Brain authority chain.

## 4) Scope lock

This pass only improves test readability, verification wording, and evidence discoverability. The real compute stack remains maintenance-only. The current bounded anatomical regions remain Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum, and Hypothalamus. IR1 remains bounded. MD2 and MD3 remain the only current selective model-deepening lines. No seventh region, third model deepening, direct action path, direct execution path, direct retry path, direct memory commit, direct compute invocation, or safety override is introduced.
