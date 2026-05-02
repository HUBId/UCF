# Serie BB27 Prompt 2: Two-region docs/tests/reference cleanup (maintenance-facing)

Status: **completed maintenance-facing cleanup pass** for the stabilized two-region baseline (Region 1 + Region 2 only).

This pass sharpens **findability**, **surface clarity**, and **drift-resistant tests** without opening any third-region or platform scope.

## 1) Canonical two-region maintenance reference map

The maintenance-facing reference surface is pinned to these categories:

1. `canonical two-region reference doc`
   - `docs/blue_brain_two_region_maintenance_stabilization_serie_bb27_prompt1_v1.md`
2. `canonical region-1 test surface`
   - `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` (`evaluate_blue_brain_first_region_attention_selection` + region-1 guard/contract tests)
3. `canonical region-2 test surface`
   - `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` (`evaluate_blue_brain_second_region_memory_context` + region-2 guard/contract tests)
4. `canonical bounded relation test surface`
   - `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` (`evaluate_blue_brain_inter_region_relation` + two-region consistency checks)
5. `maintenance-facing index/reference path`
   - `docs/README.md`
   - `docs/roadmap/REPO_MAP.md`
6. `non-canonical/internal-only or legacy two-region path`
   - any path explicitly marked `non-canonical/internal-only`, `test-only`, or `legacy` in BB26/BB27 docs and tests.

No second truth source is introduced; this map points back to already load-bearing docs/tests.

## 2) Operational two-region scope (frozen maintenance interpretation)

Within operational scope, exactly these surfaces remain canonical:

- Region 1: bounded advisory/diagnostic contract surface with no direct action/compute/retry/memory authority.
- Region 2: bounded advisory/diagnostic contract surface with no direct action/compute/retry/memory authority.
- Region-1↔Region-2 relation: exactly one bounded relation lane; no generalized multi-region coordination surface.

Still explicitly out of scope:

- third region class,
- planner/agent/policy governance buildout,
- retry orchestration platform,
- compute-core expansion,
- broad region platforming.

## 3) Guard/freeze markers that must stay visible

- `no-direct-*` boundaries remain hard and test-visible.
- `non-canonical/internal-only` remains excluded from canonical operational authority.
- Two-region baseline is maintenance-hardened and requires explicit re-scope for region expansion.
- Region 3 is **not open**.

## 4) Redundancy/ambiguity cleanup intent

This prompt consolidates reference wording for two-region maintenance operations:

- canonical pointers are centralized through docs index + repo map,
- overlapping wording is reduced where Region 1 / Region 2 / relation could be read as broader capability,
- legacy/internal-only mentions stay explicit and demoted.

## 5) Targeted regression checks

Maintenance-facing checks should keep drift visible in three places:

1. index discoverability (`docs/README.md`, `docs/roadmap/REPO_MAP.md`),
2. no-direct/freeze boundaries (`blue_brain_region_first_integration.rs` tests),
3. two-region-only semantics (no implied region-3 opening, no platform claims).

## 6) Guard/Semantic drift map (Pass-2 canonical classes)

Maintenance findings are intentionally narrow and fixed to:

1. `semantic drift risk`
2. `guard drift risk`
3. `ambiguous state meaning`
4. `weak test coverage`
5. `doc/code wording drift`
6. `no-change-needed finding`

The classes above are represented in code via
`CANONICAL_BLUE_BRAIN_TWO_REGION_MAINTENANCE_FINDINGS_MAP` and are used to keep maintenance reports
explicit without introducing a new platform layer.

## 7) Follow-up within BB27 (narrow)

1. Keep doc link integrity checks for canonical two-region map entries.
2. Prune stale duplicate two-region phrases in older non-canonical notes when touched.
3. Keep relation-boundary assertions aligned to current code semantics only.
4. Keep maintenance-only phrasing synchronized with BB23 freeze baseline.
5. Continue rejecting implicit third-region framing in maintenance edits.
