# Serie BB27 Prompt 1: Two-region maintenance stabilization line

Status: **maintenance-hardening pass complete** for the bounded two-region baseline (Region 1 + Region 2 only).

This pass intentionally focuses on **stability, drift prevention, and freeze-compatible maintenance semantics**.
It does **not** open Region 3, does **not** broaden inter-region platform authority, and does **not** alter compute-core boundaries.

## 1) Canonical two-region stabilization map

The canonical stabilization map is pinned to exactly five classes:

1. `stable two-region baseline`
2. `maintenance-hardened region-1 path`
3. `maintenance-hardened region-2 path`
4. `maintenance-hardened bounded relation path`
5. `non-canonical/internal-only residual path`

Interpretation constraints:
- Classes 1-4 are operationally relevant within maintenance-only scope.
- Class 5 remains explicitly excluded from operational authority and promotion-by-drift.

## 2) Surface stability contract (Region 1 + Region 2)

Across both regions, `input/state/output/reference` surfaces remain semantically separated:

- **input surface:** bounded consumption only from already-canonical runtime/selection/reference lanes.
- **state surface:** bounded diagnostics/contract state representation only (no implicit authority upgrade).
- **output surface:** advisory-only informational emission only.
- **reference surface:** reference-only and non-authoritative.

Hard boundary interpretation:
- advisory-only stays advisory-only,
- reference-only stays reference-only,
- no surface may drift into implicit action/execution/retry/memory/compute authority.

## 3) Diagnostics / contract / relation semantics lock

The two-region baseline keeps these states operationally distinguishable and non-interchangeable:

- `advisory-only`
- `caveated`
- `deferred`
- `blocked`
- `insufficient`
- `diagnostic-only`
- `reference-only`

The bounded Region-1↔Region-2 relation remains exactly one bounded relation lane:
- no generalized inter-region planner/orchestration semantics,
- no multi-region platform abstraction,
- no hidden authority tunnel across diagnostics/contract aliases.

## 4) Maintenance-hardened guard rails (unchanged authority limits)

The following boundaries remain hard and unchanged:

- no direct action trigger,
- no direct execution trigger,
- no direct retry trigger,
- no direct memory commit,
- no direct compute invocation,
- no safety override,
- no implicit third region class,
- no implicit broad inter-region platform.

These constraints remain aligned with BB16/BB18/BB19/BB21/BB23 guard and contract lines.

## 5) Residual cleanup posture (non-canonical/internal-only)

Maintenance posture for residual paths:

- keep non-canonical/internal-only paths explicitly marked and excluded,
- prevent shortcut aliases from becoming operational by naming drift,
- reject documentation wording that grants more authority than code/contracts provide,
- keep Region-3-adjacent language explicitly out-of-scope for BB27 Prompt 1.

## 6) Freeze/maintenance baseline alignment

This BB27 pass is constrained by the existing freeze/maintenance baseline:

- maintenance-only changes are allowed,
- semantic hardening/cleanup is allowed,
- capability expansion is out of scope without explicit re-scope,
- two-region stabilization is the prerequisite for any future region expansion decision.

## 7) References (canonical input set)

- `docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md`
- `docs/blue_brain_second_region_runtime_selection_reference_contract_serie_bb26_prompt3_v1.md`
- `docs/blue_brain_first_inter_region_relation_line_serie_bb26_prompt4_v1.md`
- `docs/blue_brain_second_region_diagnostics_caveat_deferred_semantics_serie_bb26_prompt5_v1.md`
- `docs/blue_brain_second_region_tests_guards_cleanup_serie_bb26_prompt6_v1.md`
- `docs/blue_brain_two_region_guard_contract_consistency_serie_bb26_prompt7_v1.md`
- `docs/blue_brain_bb26_readiness_sweep_second_region_expansion_boundary_serie_bb26_prompt8_v1.md`
- `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`
- `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`

## 8) Follow-up queue inside BB27 (bounded)

1. Add a compact two-region semantics regression checklist to prevent naming/state drift during maintenance patches.
2. Tighten doc-level cross-links where two-region terms can be misread as platform intent.
3. Add/refresh targeted guard assertions for no-direct-* invariants in touched modules when code changes occur.
4. Continue pruning stale internal-only shortcuts as they appear in maintenance edits.
5. Re-run readiness/docs gates after each bounded hardening increment.
