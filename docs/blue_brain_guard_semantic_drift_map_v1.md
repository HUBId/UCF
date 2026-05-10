# Blue-Brain Guard/Semantic Drift Map v1

Status: **completed narrow maintenance drift-control pass** for the current post-BR6/IR1/MD2/MD3/SC1 UCF Blue-Brain state. This map is a maintenance classifier only; it does not add a region, relation, model-deepening candidate, planner/agent lane, retry/orchestration lane, policy/governance lane, memory persistence, compute-core work, or HH productive integration.

Canonical code anchor: `CANONICAL_BLUE_BRAIN_GUARD_SEMANTIC_DRIFT_MAP`, `CANONICAL_BLUE_BRAIN_GUARD_SEMANTIC_DRIFT_CLASS_MAP`, `BlueBrainGuardSemanticDriftFindingClass`, `BlueBrainCrossLineSemanticTerm`, and `CANONICAL_BLUE_BRAIN_CROSS_LINE_TERMINOLOGY_GUARD_CHECKLIST` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Scope checked

This pass reviewed the currently active maintenance surfaces only:

- six bounded anatomical region surfaces: Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum, Hypothalamus;
- bounded inter-region architecture and relation wording;
- the first model deepening: `Amygdala ↔ Thalamus`, bounded Kuramoto-like advisory/diagnostic line;
- the second model deepening: `Amygdala ↔ Basal Ganglia`, bounded Kuramoto-like advisory/diagnostic line;
- runtime/selection/reference contract wording and reference-consumption status labels;
- no-direct-* guard wording;
- existing tests and assertions around maintenance/readiness/reference semantics.

## 2) Guard/semantic drift finding classes

Only these drift finding classes are valid for this maintenance map:

| Finding class | Meaning | Maintenance response |
| --- | --- | --- |
| `semantic drift risk` | A bounded state label can be read as stronger or weaker than intended. | Keep labels separate and assert read-only semantics. |
| `model-boundary drift risk` | A current model mode can be read as HH/productive, global, or as opening more model-deepening candidates. | Pin current model modes to relation-local bounded Kuramoto-like or abstract/deferred/diagnostic-only wording. |
| `guard drift risk` | no-direct-* wording can lose a denial or imply a direct trigger. | Keep action, execution, retry, memory commit, compute invocation, and safety override denials together. |
| `ambiguous state meaning` | caveated/deferred/blocked/insufficient/diagnostic-only/reference-only can collapse into one vague weak status. | Preserve separate consumer meaning for each state. |
| `weak test coverage` | Existing tests may not catch future status or guard wording drift. | Add focused regression assertions rather than new behavior. |
| `doc/code wording drift` | Docs and code can diverge on current model modes or authority boundaries. | Link this map into authority/discoverability docs and keep code/doc wording aligned. |
| `no-change-needed finding` | A checked surface is already aligned. | Record as no-change-needed; do not expand scope. |

## 3) Findings and cleanup

| Surface | Finding class | Drift risk found | Cleanup/hardening performed |
| --- | --- | --- | --- |
| Region surfaces | `semantic drift risk` | Advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only, and reference-only labels can slowly become implicit support/authority if compressed into generic positive wording. | Added a code-level drift map entry and regression assertion that all region-facing drift entries remain no-direct and no-new-region/no-new-model. |
| Inter-region relations | `guard drift risk` | Bounded relation wording can soften into direct action/execution/retry/memory/compute/safety wording. | Drift map now requires all no-direct denials for relation-sensitive findings. |
| First model deepening | `model-boundary drift risk` | `Amygdala ↔ Thalamus` bounded Kuramoto-like current mode can be misread as HH/productive or global model platform authority. | Drift map pins MD1/MD2 to relation-local bounded Kuramoto-like advisory/diagnostic support only. |
| Second model deepening | `model-boundary drift risk` | `Amygdala ↔ Basal Ganglia` second deepening can be misread as a third candidate, broader Kuramoto platform, or HH upgrade. | Drift map pins MD3 to exactly one second relation-local bounded Kuramoto-like advisory/diagnostic line. |
| Runtime/Selection/Reference contracts | `ambiguous state meaning` | `caveated`, `deferred`, `blocked`, `insufficient`, `diagnostic-only`, and `reference-only` can collapse into one ambiguous weak state. | Added assertions that each canonical semantic term retains distinct allowed-consumer-read wording. |
| Guard documentation | `doc/code wording drift` | Authority/discoverability docs did not yet name a dedicated guard/semantic drift map. | Added this map as a supporting current reference in the authority and discoverability surfaces. |
| Existing tests/assertions | `weak test coverage` | Future cleanup could remove direct-authority or model-boundary distinctions without targeted failure. | Added focused regression assertions for drift classes, current model mode boundaries, no-direct-* denials, and no new region/model-deepening flags. |
| Scope expansion review | `no-change-needed finding` | No active repo-backed scope expansion, seventh region, third model-deepening candidate, or direct-authority path was found in this pass. | Recorded as no-change-needed; future expansion still requires explicit re-scope. |

No checked active surface required a behavior change. No new active region, relation family, model-deepening candidate, allowed action, retry/orchestration path, memory persistence path, compute invocation path, or HH productive integration was introduced.

## 4) Status labels that must not be merged

The following labels remain intentionally distinct:

- `advisory-only`: bounded positive read only; not a trigger;
- `caveated`: bounded read with visible caveat only; not strong support;
- `deferred`: not-active-yet status; not blocked and not silently active;
- `blocked`: fail-closed unavailable/forbidden status;
- `insufficient`: weak or absent evidence; not support;
- `diagnostic-only`: observable diagnostic state; not advisory promotion;
- `reference-only`: read-only context/reference access; not persistence or execution.

## 5) Current model mode boundary

Current model wording remains bounded as follows:

- `abstract` / functional current modes stay abstract and do not imply model deepening;
- bounded `Kuramoto-like` modes are relation-local advisory/diagnostic surfaces only;
- `HH simulation-only` / `HH diagnostic-only` stays non-productive and non-authoritative;
- `HH-later` / `later selective HH deepening` remains deferred and requires explicit future re-scope;
- no wording in this pass opens a global Kuramoto/HH/model platform.

## 6) No-direct-* guard closure

Every entry in the code-level drift map is asserted to forbid:

- direct action trigger;
- direct execution trigger;
- direct retry trigger;
- direct memory commit;
- direct compute invocation;
- safety override;
- implicit new region;
- implicit new model-deepening candidate.

## 7) Closure note

Files changed by this pass:

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`
- `runtime/ucf-compute/src/lib.rs`
- `docs/README.md`
- `docs/blue_brain_authority_chain_status_map.md`
- `docs/blue_brain_maintenance_discoverability_map_v1.md`
- `docs/blue_brain_guard_semantic_drift_map_v1.md`

Maturity after this pass: **maintenance-ready with improved drift detection**. Remaining maintenance need is normal watchfulness: keep historical docs routed through the authority map, keep shadow surfaces non-canonical/internal-only, and re-run the standard docs/readiness/fmt/clippy gates on future edits.
