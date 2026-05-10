# Blue-Brain cross-surface maintenance cleanup pass v1

Status: completed narrow maintenance/bugfix/cleanup pass for the current post-BR6/IR1/MD2/MD3/SC1 Blue-Brain state. This note is supporting evidence only; it does not introduce a new region, relation lane, model-deepening candidate, compute-core path, planner/agent layer, policy/governance platform, retry/orchestration path, retrieval/consolidation/reasoning platform, memory persistence, HH production integration, allowed-action expansion, or new series.

## 1) Files changed

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`
- `docs/blue_brain_maintenance_findings_map_serie_maint_prompt1_v1.md`
- `docs/blue_brain_post_md3_maintenance_decision_pass_v1.md`
- `docs/blue_brain_cross_surface_maintenance_cleanup_pass_v1.md`
- `docs/blue_brain_maintenance_discoverability_map_v1.md`
- `docs/README.md`

## 2) Findings map update

The code-side maintenance findings taxonomy remains seven narrow buckets and now explicitly distinguishes:

1. `real_bug`
2. `semantic_inconsistency`
3. `guard_weakness`
4. `doc_test_drift`
5. `non_canonical_residual_path`
6. `cross_surface_ambiguity`
7. `no_change_needed`

The prior expansion-hook wording was replaced by `cross_surface_ambiguity` because an empty future-hook bucket was readable across docs/tests as a sanctioned later lane. The replacement is intentionally cleanup/evidence-only and keeps expansion review as **no active re-scope candidate**.

## 3) Bugs and inconsistencies found

| Class | Finding | Cleanup |
| --- | --- | --- |
| `semantic_inconsistency` | The maintenance taxonomy required cross-surface ambiguity handling, but the code-side class map used expansion-hook wording instead. | Added `CrossSurfaceAmbiguity` to the canonical class map and retagged the post-MD3 expansion-review entry as evidence-only ambiguity cleanup. |
| `doc_test_drift` | The post-MD3 maintenance doc described the old expansion-hook bucket, while current maintenance policy requires no implicit new lane. | Updated the post-MD3 doc and assertions to use cross-surface ambiguity and to keep `BLUE_BRAIN_POST_MD3_POSSIBLE_FUTURE_RE_SCOPE_CANDIDATE` empty. |
| `non_canonical_residual_path` | Expansion-hook language looked like a reusable future path despite no active candidate. | Reworded it as non-promoting ambiguity evidence, not a roadmap or scope-opening surface. |
| `no_change_needed` | Region surfaces, bounded inter-region relation maps, MD2/MD3 model-deepening boundaries, no-direct-* authority denial, and current model modes were checked. | No behavior expansion was needed; six regions, one bounded relation architecture, and exactly two selective model-deepening lines remain current. |

## 4) Cross-surface semantics retained

- Advisory-only remains advisory/read-only and never creates direct Action, Execution, Retry, Memory, Compute, Safety, selection, or promotion authority.
- Caveated, deferred, blocked, insufficient, diagnostic-only, reference-only, current-model-mode, and non-canonical/internal-only remain separately named states.
- Bounded inter-region relations remain bounded diagnostics/contract/reference surfaces; implemented, deferred, blocked and mediation-path wording is not collapsed.
- MD2 remains exactly `Amygdala ↔ Thalamus`; MD3 remains exactly `Amygdala ↔ Basal Ganglia`; no third model-deepening candidate is active.
- Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum and Hypothalamus remain the only current bounded anatomical regions.

## 5) Checks run for this pass

- Targeted source/document search for region, relation, model-deepening, runtime/selection/reference, authority, baseline and maintenance surfaces.
- Targeted regression test: `cargo test -p ucf-compute blue_brain_region_first_integration --lib`.
- Formatting: `cargo fmt --all -- --check`.
- Linting: `cargo clippy --workspace --all-targets -- -D warnings`.
- Workspace tests: `cargo test --workspace`.
- Documentation gate: `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`.
- Readiness gate: `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`.

## 6) Closure

The current state remains clean maintenance-ready for the audit anchor after this pass. Remaining work is normal maintenance only: bugfixes, deterministic cleanup, report refreshes, and terminology/test hardening inside the current authority envelope.

A new large expansion block is **not indicated by this pass**. It would only be vertretbar after an explicit future re-scope that names the intended expansion and reopens the relevant authority boundary; no such re-scope is active here.
