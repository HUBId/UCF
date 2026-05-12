# Blue-Brain canonical model-boundary map v1

Status: canonical structural-closure reference for system-wide model boundaries. This document mirrors `CANONICAL_BLUE_BRAIN_MODEL_BOUNDARY_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`; it does not create a second truth source, a new model platform, a new region, a third model-deepening candidate, productive Hodgkin-Huxley integration, planner/agent logic, policy governance, retry orchestration, or compute-core work.

## Canonical model-boundary vocabulary

| Boundary mode | Canonical meaning | Authority boundary |
| --- | --- | --- |
| canonical abstract functional/current mode | Current region or relation semantics are contract-/surface-sufficient without active model deepening. | Abstract mode is not a gap, not hidden HH, and not an implicit Kuramoto candidate. |
| canonical bounded Kuramoto-like mode | A narrowly named bounded dynamics surface can expose advisory/diagnostic coupling, synchrony, gating, or timing evidence. | Active only for explicitly listed model-deepened surfaces and the BB12 bounded advisory dynamics reference; it is not global region or relation logic. |
| canonical HH simulation-only/diagnostic-only mode | HH-like or membrane-near work can exist only as diagnostic/simulation reference. | HH output is not productive authority, not the basis of regional semantics, and not required by Runtime/Selection/Reference. |
| canonical later-HH/deferred mode | A possible later selective HH re-scope is named but inactive. | Deferred HH is not an implementation lane and cannot become active without a separate explicit re-scope. |
| non-canonical/internal-only model path | Helper, research, shadow, test-only, or residual model paths outside canonical reads. | No consumer read, no contract support, no direct authority. |

## Region model-boundary table

All six canonical regions keep region-surface semantics leading. Relation-local or dynamics-near model evidence does not rewrite regional roles.

| Surface | Surface kind | Current model mode | Active model deepening | Boundary |
| --- | --- | --- | --- | --- |
| Hippocampus | region | abstract functional/current mode | none | Context/reference/episode-indexing semantics remain surface-level. |
| Amygdala | region | abstract functional/current mode | none | Salience/valence/caveat semantics remain surface-level; relation-local deepening does not make Amygdala a model authority. |
| Thalamus | region | abstract functional/current mode | none | Relay/gating/routing semantics remain surface-level. |
| Basal Ganglia | region | abstract functional/current mode | none | Action-gating/selection-channel semantics remain advisory and do not execute actions. |
| Cerebellum | region | abstract functional/current mode | none | Prediction/timing/correction remains abstract; microcircuit/HH paths stay diagnostic-only or deferred. |
| Hypothalamus | region | abstract functional/current mode | none | Drive/homeostasis/urgency pressure remains bounded abstract contract semantics. |

## Relation and model-deepening table

| Surface | Surface kind | Current model mode | Relation closure status | Boundary |
| --- | --- | --- | --- | --- |
| Hippocampus ↔ Amygdala | relation | abstract functional/current mode | architectural lane only | No active model deepening. |
| Hippocampus ↔ Thalamus | relation | abstract functional/current mode | canonical mediated relation | Reference-mediated contract read; no dynamics platform. |
| Hippocampus ↔ Basal Ganglia | relation | abstract functional/current mode | canonical blocked relation | Blocked/unavailable, diagnostic-only, no model activation. |
| Hippocampus ↔ Cerebellum | relation | abstract functional/current mode | architectural lane only | No active model deepening. |
| Amygdala ↔ Thalamus | selective model deepening | bounded Kuramoto-like current mode | canonical implemented relation | First model-deepened surface; advisory/diagnostic only. |
| Amygdala ↔ Basal Ganglia | selective model deepening | bounded Kuramoto-like current mode | canonical mediated relation | Second model-deepened surface; selection-mediated, advisory/diagnostic only. |
| Amygdala ↔ Cerebellum | relation | abstract functional/current mode | canonical deferred relation | Deferred relation is not model deepening. |
| Thalamus ↔ Basal Ganglia | relation | abstract functional/current mode | architectural lane only | No active model deepening. |
| Thalamus ↔ Cerebellum | relation | abstract functional/current mode | architectural lane only | No active model deepening. |
| Basal Ganglia ↔ Cerebellum | relation | later-HH/deferred mode | architectural lane only | Later selective HH remains deferred, not productive. |
| Hippocampus ↔ Hypothalamus | relation | abstract functional/current mode | canonical mediated relation | Reference-mediated contract read only. |
| Amygdala ↔ Hypothalamus | relation | abstract functional/current mode | canonical implemented relation | Caveated/direct bounded advisory relation does not become model deepening. |
| Thalamus ↔ Hypothalamus | relation | abstract functional/current mode | canonical implemented relation | Direct bounded advisory relation does not become model deepening. |
| Basal Ganglia ↔ Hypothalamus | relation | abstract functional/current mode | canonical mediated relation | Selection-mediated contract read only. |
| Cerebellum ↔ Hypothalamus | relation | abstract functional/current mode | canonical deferred relation | Deferred relation is not model deepening. |

Canonical active model deepening is exactly two surfaces: `Amygdala ↔ Thalamus` and `Amygdala ↔ Basal Ganglia`. Completion-third-deepening closure reviews `Thalamus ↔ Cerebellum` first but leaves it unopened because it is still architecture-lane-only/NotYetImplemented. Relation diagnostics, caveats, mediated reads, deferred states, blocked states, and architecture lanes do not automatically become model-deepened surfaces.

## Dynamics-reference boundaries

| Surface | Surface kind | Current model mode | Boundary |
| --- | --- | --- | --- |
| BB12 bounded advisory dynamics surface | dynamics reference | bounded Kuramoto-like current mode | Shared vocabulary/reference only; no global Kuramoto platform. |
| BB10 HH diagnostic surface | dynamics reference | HH simulation-only/diagnostic-only | Diagnostic/simulation-only; no productive HH. |
| Cerebellum microcircuit HH diagnostic path | dynamics reference | HH simulation-only/diagnostic-only | Diagnostic-only or later selective re-scope; current Cerebellum region remains abstract. |
| Hypothalamus HH diagnostic path | dynamics reference | HH simulation-only/diagnostic-only | Diagnostic-only or later selective re-scope; current Hypothalamus region remains abstract. |
| Later selective HH deepening path | dynamics reference | later-HH/deferred mode | Future explicit re-scope only. |
| Non-canonical/internal-only model path | internal-only | non-canonical/internal-only model path | No Runtime/Selection/Reference/Execution consumer authority. |

## HH boundary

HH is nowhere a current productive mode. HH remains either HH simulation-only/diagnostic-only or later-HH/deferred. HH is not silently the basis of Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum, Hypothalamus, inter-region relations, Runtime, Selection, Reference, or Execution. No implicit HH duty follows from a region, relation, diagnostic, or dynamics-reference entry.

## Kuramoto-like boundary

Bounded Kuramoto-like mode is active only where explicitly listed: the first model-deepened `Amygdala ↔ Thalamus` surface, the second model-deepened `Amygdala ↔ Basal Ganglia` surface, and the BB12 bounded advisory dynamics reference. Completion-third-deepening closure does not add `Thalamus ↔ Cerebellum` to this active set. It remains bounded, advisory, and diagnostic. It is not a global region model, not a global relation model, not a planner, not action selection, not execution control, and not a compute-core platform.

## Contract, guard, and runtime authority separation

The model-boundary map is descriptive and lower authority than existing Contract, Guard, Runtime, Selection, Reference, and Execution boundaries:

- model state is not contract state;
- model output is not direct authority;
- diagnostic model output is not advisory support unless explicitly classed as bounded advisory support;
- region surface semantics remain leading for regions;
- relation contract semantics remain leading for relations;
- no-direct-action, no-direct-execution, no-direct-retry, no-direct-memory, no-direct-compute, and no-safety-override guards remain leading;
- Runtime, Selection, Reference, and Execution may read model-deepened surfaces only through bounded advisory/diagnostic contract reads;
- blocked, deferred, insufficient, diagnostic-only, and non-canonical/internal-only states create no fallback authority;
- no global model platform is created.

## Structural-closure handoff

This map provides the model-boundary basis for the Structural Closure decision. The remaining closure work should be limited to:

1. checking that Structural Closure references this map instead of restating model modes; and
2. verifying that final closure/readiness docs keep no-direct-* guards and out-of-scope boundaries aligned with this map.
