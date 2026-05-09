# Blue-Brain Maintenance Discoverability Map v1

Status: **maintenance-facing discoverability map** for the current post-BR6/IR1/MD2/MD3/SC1 Blue-Brain repository state. This file is a compact index/classification aid only; it does not replace the canonical authority chain in `docs/blue_brain_authority_chain_status_map.md` and does not create new operative authority.

## 1) Allowed discoverability classes

Maintainers should use only these classes when reading Blue-Brain docs and references:

| Class | Meaning | Operational reading |
| --- | --- | --- |
| `current operational authority` | Final/current authority docs that define the active six-region, bounded inter-region, bounded model-deepening and maintenance decision state. | Use first; wins conflicts. |
| `supporting current reference` | Current evidence, guard, audit, terminology or cleanup references that explain and verify the authority line. | Read with the current authority line; never as a second truth source. |
| `historical snapshot` | Preserved BB25/BB27/BB29 and early-stage transition snapshots. | Useful for traceability only; overridden by later current authority. |
| `stale discoverability pointer` | Older wording, shortened references, or transition-era pointers that can look current if read without the authority map. | Must be resolved through `docs/blue_brain_authority_chain_status_map.md`. |
| `non-canonical/internal-only shadow surface` | DBM, microcircuit, neuro, test, expert, or helper surfaces outside the six canonical regions and current authority chain. | No consumer authority; no implicit region/model/platform expansion. |

## 2) Current operational authority path

The single authority path remains the authority map plus the final current-authority docs listed there:

- `docs/blue_brain_authority_chain_status_map.md`
- `docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md`
- `docs/blue_brain_ir1_readiness_sweep_inter_region_closure_serie_ir1_prompt4_v1.md`
- `docs/blue_brain_md2_model_deepening_docs_tests_reference_cleanup_v1.md`
- `docs/blue_brain_md3_second_deepening_rescope_line_v1.md`
- `docs/blue_brain_md3_second_model_deepening_implementation_line_v1.md`
- `docs/blue_brain_md3_second_model_deepening_hardening_line_v1.md`
- `docs/blue_brain_md3_readiness_sweep_system_closure_v1.md`
- `docs/blue_brain_post_md3_maintenance_decision_pass_v1.md`
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md`
- `docs/blue_brain_sc1_prompt4_final_system_consolidation_sweep_v1.md`

Summary: current authority is six bounded anatomical regions; IR1 bounded relation semantics; MD2 exactly one first deepened pair; MD3 exactly one second bounded deepened pair; SC1 maintenance default. No seventh region, no additional model candidate, no inter-region platform, no global model platform, no planner/agent/policy/retry expansion, and no compute-core expansion are active.

## 3) Supporting current references

These docs and artifacts are current support, not competing authority:

- `docs/blue_brain_sc1_prompt2_post_br6_repro_baseline_refresh_v1.md` — current repro/test evidence line.
- `docs/blue_brain_sc1_prompt3_cross_line_terminology_guard_checklist_consolidation_v1.md` — compact terminology and no-direct-* guard checklist.
- `docs/blue_brain_audit_baseline_map_v1.md` — current audit baseline and artifact map.
- `docs/blue_brain_non_canonical_shadow_surface_inventory_v1.md` — non-canonical/internal-only shadow-surface inventory.
- `out/blue_brain_audit_baseline_2026-05-09/` — current Blue-Brain clean maintenance-ready baseline evidence bundle on HEAD `efbeec23b752744dc9f87a2e2e3eeb9efe25104f`.
- `out/docs_lint_report.json` and `out/gate_report.json` — standard root reports for canonical docs/readiness checks.

Supporting references can clarify evidence, terminology and maintenance caveats. They must not be used to override the authority map or infer new features.

## 4) Historical snapshots and stale pointers

Historical snapshots include BB25, BB27, BB29, two-region and three-region transition handoff docs, plus early implementation-stage relation docs where the repo intentionally preserves narrower intermediate states. These files are valuable for audit trail and design chronology, but their region counts, relation activation status, model-open wording, or maintenance locks can be stale relative to the current post-BR6/IR1/MD2/MD3/SC1 line.

Common stale readings to avoid:

- treating a two-region or three-region document as the current Blue-Brain state;
- treating early IR1 implementation text as the complete current relation set after later authority docs;
- treating pre-MD3 model-deepening language as leaving an open candidate after MD3/SC1 closure;
- treating old baseline folders as current audit evidence;
- treating non-canonical DBM/microcircuit/neuro crates as region expansion.

## 5) Maintenance caveat handling

The current state is a **clean maintenance-ready baseline**. Remaining items are normal discoverability and wording maintenance risks, not active expansion levers:

- historical docs remain searchable and require authority-map disambiguation;
- `selection-mediated` and `execution-interface-mediated` remain read/diagnostic labels only;
- current audit evidence is the 2026-05-09 baseline bundle, while 2026-05-08 and earlier bundles are historical;
- shadow surfaces remain non-canonical/internal-only/deferred unless a future explicit re-scope changes the authority chain;
- the former Cargo `workspace.features` warning has been removed at the root manifest instead of accepted as recurring maintenance noise.
