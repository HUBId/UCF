# Blue-Brain Post-MD3 maintenance-/decision-pass v1

Canonical code anchors: `CANONICAL_BLUE_BRAIN_POST_MD3_MAINTENANCE_FINDINGS_MAP`, `CANONICAL_BLUE_BRAIN_MAINTENANCE_FINDINGS_CLASS_MAP`, `CANONICAL_BLUE_BRAIN_MD3_READINESS_MAP`, `CANONICAL_BLUE_BRAIN_CROSS_LINE_TERMINOLOGY_GUARD_CHECKLIST`, and `BLUE_BRAIN_POST_MD3_POSSIBLE_FUTURE_RE_SCOPE_CANDIDATE` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

Status: maintenance-only pass after MD3 system closure. This file records the narrow bug/guard/doc/test decision sweep. It does not add a region, model candidate, planner/agent layer, policy/governance platform, retry/queue/orchestration platform, retrieval/consolidation/reasoning platform, memory persistence, HH production integration, allowed-actions expansion, or compute-core work.

## 1) Repo-based maintenance findings map

`CANONICAL_BLUE_BRAIN_POST_MD3_MAINTENANCE_FINDINGS_MAP` uses these maintenance finding classes:

| Class | Meaning in this pass |
| --- | --- |
| real bug | A bounded implementation defect that can be fixed without behavior expansion. |
| semantic inconsistency | A mismatch between advisory/caveated/deferred/blocked/insufficient/diagnostic-only/reference-only/current-model-mode wording. |
| guard weakness | A no-direct-* or bounded-consumer guard that is present but under-specified. |
| doc/test drift | Documentation or tests lag a canonical code/status surface. |
| non-canonical residual path | Historical, test-only, or internal-only path that must not become a consumer source of truth. |
| no-change-needed finding | Checked surface is already aligned and only records evidence. |
| cross-surface ambiguity | A review bucket whose wording can be interpreted differently across region, relation, model, guard, or maintenance-documentation surfaces. |

## 1a) Maintenance action map

| Action class | Maintenance target | Decision |
| --- | --- | --- |
| authority/discoverability cleanup | README entrypoint and authority-chain reading order | Keep the current authority chain primary; older entrypoints and compact maps remain supporting/historical only, not parallel authority. |
| relation wording cleanup | Implemented, mediated, deferred, blocked, caveated, and diagnostic-only relation wording | Preserve implemented vs mediated vs deferred vs blocked relation semantics; no implicit platform, action, execution, retry, or promotion authority. |
| model-boundary wording cleanup | Abstract current mode, bounded Kuramoto-like current mode, HH simulation-only/diagnostic-only, and later-HH/deferred wording | Keep current model wording bounded; no HH productive upgrade and no model mixing. |
| guard wording/visibility cleanup | Cross-line no-direct guard checklist | Keep no direct action, execution, retry, memory commit, compute invocation, and safety override denials visible together. |
| evidence/reference cleanup | Current reports, HEAD-qualified baseline folders, and supporting evidence docs | Refresh or cite evidence only; reports remain evidence for recorded runs/commits and do not change region, relation, model, policy, retry, or compute behavior. |
| no-change-needed finding | Region, relation, and model-deepening boundaries | Record evidence that the existing six regions, IR1 bounded relations, and exactly two selective model-deepening lines remain unchanged. |


Supporting action-map reference for the current maintenance pass: `docs/blue_brain_current_maintenance_action_map_v1.md`.

Findings by surface:

| Surface | Finding class | Decision |
| --- | --- | --- |
| Region surfaces | no-change-needed finding | Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum, and Hypothalamus remain bounded advisory/reference/diagnostic surfaces. |
| Inter-region relations | no-change-needed finding | IR1 remains bounded; relation reads stay direct-bounded-advisory, reference-mediated, selection-mediated, caveated, deferred/blocked, or non-canonical/internal-only as already defined. |
| First model deepening | no-change-needed finding | `Amygdala ↔ Thalamus` remains the MD1/MD2 maintenance-hardened advisory-only bounded Kuramoto-like line. |
| Second model deepening | no-change-needed finding | `Amygdala ↔ Basal Ganglia` remains the only MD3 advisory-only bounded second model-deepening line. |
| Runtime/selection/reference contracts | guard weakness | The cross-line direct-authority predicate now requires explicit memory-commit denial in addition to action, execution, retry, compute, and an explicit safety-override guard target. |
| Docs/tests/readiness references | doc/test drift | The maintenance finding class map now includes this post-MD3 reference and keeps taxonomy wording aligned with code-side tests. |
| Non-canonical residuals | non-canonical residual path | Historical and test-only paths remain marked non-canonical/internal-only with no consumer-operational authority. |
| Expansion lever review | cross-surface ambiguity | No active post-MD3 re-scope candidate remains; expansion-review wording is evidence-only, not a reusable future hook. |

## 2) Bugs, inconsistencies, and guard cleanups applied

- Real bug: none requiring behavioral repair was found in the checked region, relation, model-deepening, runtime/selection/reference, readiness, or docs/test surfaces.
- Guard weakness fixed: `blue_brain_cross_line_term_allows_direct_authority` now treats a term as non-authoritative only when the checklist denies no direct action, no direct execution, no direct retry, no direct memory commit, no direct compute invocation, and the explicit `forbids_safety_override` guard state denies safety override authority.
- Doc/test drift fixed: the maintenance finding taxonomy now uses `cross-surface ambiguity` instead of expansion-hook wording; tests pin that the post-MD3 re-scope candidate remains empty.
- Non-canonical residual cleanup: residual and test-only paths remain classified as non-canonical/internal-only, not promoted or deleted into a second truth source.

## 3) Semantic consistency decisions

- advisory-only remains bounded positive read only; it is never a direct action, execution, retry, memory, compute, safety, selection, or promotion authority.
- caveated remains visible uncertainty; caveated outputs are not strong support.
- deferred remains not-active-yet; it is not blocked, failed, or silently activated.
- blocked remains fail-closed unavailable/forbidden path.
- insufficient remains weak/absent evidence and never positive support.
- diagnostic-only remains explanatory state and does not steer transitions.
- reference-only remains read-only context/reference access and never persistence.
- current model mode remains descriptive and does not create a model platform.
- non-canonical/internal-only remains traceability/test/residual state and is not a consumer source of truth.

## 4) Expansion lever review

No active post-MD3 re-scope candidate remains. The repo-backed state already contains:

- six bounded anatomical regions;
- one bounded inter-region architecture;
- first model deepening: `Amygdala ↔ Thalamus`;
- second model deepening: `Amygdala ↔ Basal Ganglia`;
- maintenance-ready or maintenance-ready-with-caveats closure.

Further region work, a third model deepening, production HH integration, broader Kuramoto/HH platforms, planner/agent logic, policy/governance, retry orchestration, retrieval/consolidation/reasoning, memory persistence, or compute-core work would be scope drift unless a future task explicitly re-scopes it. Therefore `BLUE_BRAIN_POST_MD3_POSSIBLE_FUTURE_RE_SCOPE_CANDIDATE` is `None`.

## 5) Checks expected for this pass

Targeted checks should verify:

- no scope expansion;
- no implicit new region;
- no implicit new model-deepening candidate;
- no direct action, no direct execution, no direct retry, no direct memory commit, no direct compute invocation, and explicit no safety override / `forbids_safety_override` visibility;
- docs and tests match code.

Maintenance check-reading order is now pinned in `docs/blue_brain_maintenance_verification_findings_map_v1.md`: local targeted Blue-Brain tests explain the changed surface, canonical workspace/readiness/docs checks are the handoff checks, and `out/` reports are audit/baseline evidence. This clarification removes stale verification wording without creating a second verification authority.
