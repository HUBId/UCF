# Serie BB27 Prompt 4: Post-BB27 roadmap decision lock

Status: **decision locked** on top of the stabilized bounded two-region baseline from BB24–BB27.

This step adds no new implementation surface. It records the explicit roadmap decision after BB27 and keeps maintenance work clearly separated from any future re-scope discussion.

## 1) Repo-based decision check after BB27

Based on BB25/BB26/BB27 closure documents and their two-region guard/contract references:

- Region 1 and Region 2 are already stabilized as bounded, advisory/diagnostic-aware operational surfaces.
- The Region-1↔Region-2 relation is intentionally bounded and non-generalized.
- no-direct/freeze boundaries remain the active anti-drift mechanism.

Result: the two-region baseline is stable enough that **maintenance/bugfix/cleanup is the correct default mode**.

## 2) Region-3 leverage check (technical, not aspirational)

A Region-3 re-scope is currently **not** justified as default continuation, because:

- there is no unresolved load-bearing defect in the BB27 two-region closure that requires a third region class,
- existing open caveats are intentionally advisory/diagnostic/deferred classes, not hidden Region-3 requirements,
- opening Region 3 now would primarily add boundary and authority risk (class blur, platform drift) without immediate stabilization leverage.

A future Region-3 line remains possible only as an explicit later re-scope backed by new requirements and explicit boundary design, not as automatic follow-up.

## 3) Final decision (exactly one)

**Decision:** after BB27, stay in **maintenance/bugfix/cleanup without starting a new series by default**.

Interpretation rule:
- maintenance changes may continue when they preserve the two-region canonical boundary,
- Region-3 work requires a separate, explicit re-scope decision later.

## 4) Central reference alignment

For the post-BB27 roadmap stance, this file is the explicit decision reference and aligns with:

- `docs/blue_brain_bb27_final_two_region_stabilization_sweep_serie_bb27_prompt3_v1.md`
- `docs/blue_brain_bb26_readiness_sweep_second_region_expansion_boundary_serie_bb26_prompt8_v1.md`
- `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`

No contradictory expansion claim is introduced here.

## 5) Minimal consistency checks for this decision lock

1. Documentation states maintenance as post-BB27 default mode.
2. Documentation does not claim an implicit Region-3 opening.
3. Repo checks (fmt/clippy/tests/docs-lint/readiness-gate) remain green to avoid status-vs-repo drift.
