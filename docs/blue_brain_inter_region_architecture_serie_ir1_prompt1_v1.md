# Blue-Brain IR1 Prompt 1: bounded inter-region architecture map

Status: **bounded inter-region architecture line** over the already opened anatomical regions: `hippocampus_like_region`, `amygdala_like_region`, `thalamus_like_region`, `basal_ganglia_like_region`, and `cerebellum_like_region`.

This document is the canonical IR1 architecture map for these five regions. It consolidates the existing BR1-BR5 surfaces and the BB2/BB4/BB8/BB12/BB17/BB19/BB21 contract lines without creating a global region orchestration platform, a planner/agent layer, a retry platform, a policy-governance platform, or new compute-core behavior.

## 1) Region roles used by IR1

IR1 uses UCF functional roles, not neurobiological completeness:

| Region | Canonical IR1 role | Primary mediated surface |
|---|---|---|
| `hippocampus_like_region` | context/reference/episode/indexing-heavy | Reference/Context, BB8/BB17 |
| `amygdala_like_region` | salience/valence/caveat/priority-heavy | Runtime/Selection caveats, BB4/BB12/BB19 |
| `thalamus_like_region` | relay/gating/routing-heavy | Runtime/Selection contract reads, BB2/BB19 |
| `basal_ganglia_like_region` | action-channel/suppression/readiness-heavy | Selection/Execution-interface diagnostics, BB4/BB19/BB21 |
| `cerebellum_like_region` | timing/prediction/correction/mismatch-heavy | Bounded dynamics and Execution-interface diagnostics, BB12/BB16/BB21 |

The current model modes remain unchanged: `abstract functional current mode` stays current for Hippocampus, Thalamus, Basal Ganglia, and Cerebellum; Amygdala stays on its bounded current line. IR1 does not imply a Kuramoto expansion, a Hodgkin-Huxley production integration, or a new model-depth mandate.

## 2) Relation classes

IR1 recognizes only these relation classes:

- `direct bounded advisory relation`: a bounded advisory-only read between region surfaces; it is never strong authority.
- `reference-mediated relation`: Context/Reference is the mediator; this is not direct inter-region authority and cannot commit memory.
- `selection-mediated relation`: Selection/Contract mediates the read; this is not action selection authority.
- `execution-interface-mediated relation`: the Execution-interface may read diagnostics close to execution eligibility; this is not a direct execution trigger.
- `caveated inter-region relation`: quality-, salience-, or confidence-limited relation; caveated relation is not stable relation.
- `deferred/not-yet-active relation`: a deliberately delayed architecture edge; deferred relation is not blocked relation.
- `blocked relation`: an explicit unavailable path; blocked relation is not failed execution and does not start retry orchestration.
- `non-canonical/internal-only relation path`: implementation/test/internal-only lane; it has no operational authority.

No other relation class is canonical in IR1.

## 3) Canonical pair map

| Pair | Canonical relation class | Functional reason | Mediation / status |
|---|---|---|---|
| Hippocampus ↔ Amygdala | `caveated inter-region relation` | Context/reference signals can be annotated by salience/caveat, but not stabilized as authority. | Caveated Reference/Selection read only. |
| Hippocampus ↔ Thalamus | `reference-mediated relation` | Indexed context can inform relay/routing diagnostics only through canonical Reference/Context. | Reference/Context mediated only. |
| Hippocampus ↔ Basal Ganglia | `blocked relation` | Context/reference must not directly reach action-channel or suppression authority. | Blocked direct relation; any future read must be explicitly re-scoped. |
| Hippocampus ↔ Cerebellum | `reference-mediated relation` | Context/reference can provide bounded mismatch/timing context. | Reference/Context mediated only. |
| Amygdala ↔ Thalamus | `direct bounded advisory relation` | Salience/caveat can advisory-shape relay/routing diagnostics. | Direct bounded advisory-only relation. |
| Amygdala ↔ Basal Ganglia | `selection-mediated relation` | Salience/caveat may influence readiness diagnostics only through Selection/Contract. | Selection/Contract mediated only. |
| Amygdala ↔ Cerebellum | `deferred/not-yet-active relation` | Salience-to-timing/prediction coupling is not yet needed for the UCF surface. | Deferred; not blocked, not failed. |
| Thalamus ↔ Basal Ganglia | `selection-mediated relation` | Relay/routing and channel suppression meet only at bounded Selection/Contract reads. | Selection/Contract mediated only. |
| Thalamus ↔ Cerebellum | `direct bounded advisory relation` | Relay/routing can advisory-coordinate timing/mismatch diagnostics. | Direct bounded advisory-only relation. |
| Basal Ganglia ↔ Cerebellum | `execution-interface-mediated relation` | Action-channel/readiness and timing/correction intersect near execution eligibility diagnostics. | Execution-interface-mediated diagnostic read only. |

This map is intentionally not an all-to-all coupling engine. The pair labels identify canonical architecture edges, not autonomous region-to-region messages.

## 4) Mediation rules

- Direct bounded advisory-only relations: Amygdala ↔ Thalamus, Thalamus ↔ Cerebellum.
- Reference-mediated only relations: Hippocampus ↔ Thalamus, Hippocampus ↔ Cerebellum.
- Caveated relation: Hippocampus ↔ Amygdala.
- Selection/Contract mediated only relations: Amygdala ↔ Basal Ganglia, Thalamus ↔ Basal Ganglia.
- Execution-interface-mediated only relation: Basal Ganglia ↔ Cerebellum.
- Deferred/not-yet-active relation: Amygdala ↔ Cerebellum.
- Blocked relation: Hippocampus ↔ Basal Ganglia direct relation.
- Non-canonical/internal-only relation path: any unlisted direct path, any raw internal state path, and any test helper path outside the canonical map.

## 5) Relational state boundaries

IR1 keeps these distinctions hard:

- advisory-only relation is not strong authority.
- caveated relation is not stable relation.
- deferred relation is not blocked relation.
- blocked relation is not failed execution.
- reference-mediated relation is not direct inter-region authority.
- selection-mediated relation is not action selection.
- execution-interface-mediated relation is not action execution.
- non-canonical/internal-only relation path is not an operational relation.

## 6) No-direct-* and out-of-scope guards

The architecture map provides no authority for:

- no direct action trigger.
- no direct execution trigger.
- no direct retry trigger.
- no retry orchestration.
- no direct memory commit.
- no automatic memory persistence.
- no direct compute invocation.
- no safety override.
- no allowed-actions extension.
- no implicit global region orchestration.
- no new inter-region platform formation.
- no policy/governance platform.
- no planner/agent platform.
- no retrieval/consolidation/reasoning platform.
- no Hodgkin-Huxley production integration.

Blocked relation is represented as architectural unavailability, not as failed execution. Deferred relation is represented as not-yet-active architecture, not as a runtime error.

## 7) Cross-line consistency

- BB2 Runtime/Transition/Feedback remains the runtime reader, not an inter-region orchestrator.
- BB4 Selection/Priority/Deferral remains the bounded mediation lane for selection-heavy relations.
- BB8 and BB17 Context/Memory/Reference remain the only memory/reference mediation lanes; IR1 adds no direct memory commit.
- BB12/BB16 bounded dynamics remain advisory-only and do not become a global neurodynamics platform.
- BB19 Runtime/Selection contracts remain the common contract read layer.
- BB21 Execution/Reference interaction remains diagnostic and reference-bounded; IR1 adds no direct execution or retry trigger.
- Real Compute remains maintenance-only; IR1 performs no compute-core work.

## 8) IR1 next steps

1. Add narrow consumer-facing diagnostics that report the IR1 relation class without changing action/execution authority.
2. Add fixture/golden coverage for the ten canonical pair entries if an external artifact starts consuming the map.
3. Review whether Amygdala ↔ Cerebellum should stay deferred or become caveated after a separate model-depth decision.
4. Review whether Hippocampus ↔ Basal Ganglia needs a mediated future edge; direct coupling remains blocked.
5. Keep all future work behind explicit policy/spec intent rather than implicit all-to-all region coupling.
