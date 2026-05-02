# Serie BB27 Prompt 5: Post-BB27 maintenance-default decision map

Status: **canonical post-BB27 default decision map locked** on the stabilized two-region baseline.

This map introduces no new functionality and no new series. It only fixes the explicit default mode after BB27 and keeps a possible Region-3 re-scope intentionally open-but-inactive.

## 1) Repo-based closure state entering Prompt 5

The closure state from BB24–BB27 is treated as settled for default operations:

- Region 1 is maintenance-hardened and part of the active baseline.
- Region 2 is maintenance-hardened and part of the active baseline.
- Region-1↔Region-2 relation remains bounded and non-platformizing.
- BB23 freeze/maintenance guard interpretation remains binding.

Therefore, no additional expansion step is implicit in the post-BB27 default lane.

## 2) Canonical post-BB27 decision map

Exactly these states are canonical:

1. `maintenance_default_active`
   - meaning: default work mode is **bugfix / cleanup / maintenance**.
2. `region1_active_stabilized_baseline`
   - meaning: Region 1 remains active within the bounded maintenance baseline.
3. `region2_active_stabilized_baseline`
   - meaning: Region 2 remains active within the bounded maintenance baseline.
4. `region3_not_active_requires_explicit_rescope`
   - meaning: Region 3 is not discarded, but not open; any activation needs a later explicit re-scope decision.
5. `deferred_non_canonical_out_of_scope_continuation`
   - meaning: implicit continuation patterns, platformizing expansions, or non-canonical buildout remain deferred/out-of-scope.

No additional canonical state is introduced in this prompt.

## 3) Operational interpretation (default after BB27)

- The standard path is maintenance-only work on existing bounded two-region semantics.
- There is **no automatic next series** after BB27.
- There is **no implicit functional opening** (no third region, no authority broadening, no hidden platform transition).

## 4) Region-3 stance: open for explicit re-scope, inactive by default

Region 3 is intentionally handled as:

- **possible later**, if a future requirement explicitly justifies it,
- **not active now**, and
- **not an automatic successor step** from BB27.

Any future Region-3 move must be a separate explicit prioritization and boundary decision outside this maintenance-default lock.

## 5) Unchanged guard and out-of-scope boundaries

Unchanged in the post-BB27 default lane:

- no multi-region expansion beyond Region 2,
- no direct HH production opening,
- no new allowed-actions expansion,
- no planner/agent platform buildout,
- no retrieval/consolidation/reasoning platform buildout,
- no new compute-core work,
- no retry/queue/orchestration platformization,
- no implicit memory-persistence expansion.

## 6) Single-source alignment references

This decision map is aligned with and subordinate to the existing canonical BB23/BB24/BB25/BB26/BB27 closure line:

- `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`
- `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`
- `docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md`
- `docs/blue_brain_bb26_readiness_sweep_second_region_expansion_boundary_serie_bb26_prompt8_v1.md`
- `docs/blue_brain_bb27_final_two_region_stabilization_sweep_serie_bb27_prompt3_v1.md`
- `docs/blue_brain_bb27_post_decision_serie_bb27_prompt4_v1.md`

