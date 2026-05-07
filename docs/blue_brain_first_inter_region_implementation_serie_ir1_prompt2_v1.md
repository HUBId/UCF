# Blue-Brain IR1 Prompt 2: first bounded inter-region implementation line

Status: **first inter-region implementation line** over the bounded IR1 architecture map. This document is the implementation companion to `docs/blue_brain_inter_region_architecture_serie_ir1_prompt1_v1.md`; it does not replace that architecture map and does not create a second truth source for relation classes.

IR1 Prompt 2 implements exactly three implemented relations from the original five-region IR1 map. Every other original architecture-map relation remains deferred/not-yet-implemented or blocked for this step. BR6 Prompt 2 appends a bounded Hypothalamus adjunct relation set to the same code anchor without changing the original IR1 base-three decision.

## 1) Implementation classes

Only these implementation classes are canonical for this first line:

- `implemented direct bounded advisory relation`: direct bounded read between the two named region surfaces; advisory-only and never authority.
- `implemented reference-mediated relation`: Context/Reference mediates the relation; no direct inter-region authority and no memory commit.
- `implemented selection-mediated relation`: Selection/Contract mediates the relation; no action selection authority and no execution trigger.
- `deferred/not-yet-implemented relation`: architecture edge remains known but inactive in this implementation line.
- `blocked relation`: architecture edge is explicitly unavailable and is not a failed execution.
- `non-canonical/internal-only relation path`: raw internal or helper path with no operational authority.

There is no global inter-region platform, no broadcast/routing fabric, no planner/agent layer, and no free region-to-region decision authority.

## 2) Exactly three implemented relations

The implemented set is intentionally small and high-leverage:

| Pair | Directionality | Implementation class | Canonical mediation path | Source → target signal | Target → source signal | Boundary |
|---|---|---|---|---|---|---|
| `Amygdala ↔ Thalamus` | bidirectional pair label with bounded typed reads | `implemented direct bounded advisory relation` | `DirectBoundedAdvisoryOnly` | `SalienceCaveatAdvisory` | `RelayRoutingDiagnostic` | Salience/caveat can annotate relay/routing diagnostics; it cannot trigger actions, retries, execution, compute, or safety override. |
| `Hippocampus ↔ Thalamus` | bidirectional pair label with mediated typed reads | `implemented reference-mediated relation` | `ReferenceContextMediatedOnly` | `ContextReferenceDiagnostic` | `RelayRoutingDiagnostic` | Indexed context reaches relay/routing only through Reference/Context; it cannot commit memory or bypass Reference mediation. |
| `Amygdala ↔ Basal Ganglia` | bidirectional pair label with mediated typed reads | `implemented selection-mediated relation` | `SelectionContractMediatedOnly` | `SalienceCaveatAdvisory` | `SelectionReadinessDiagnostic` | Salience/caveat reaches readiness diagnostics only through Selection/Contract; it is not action-channel authority. |

These are exactly three implemented relations. Their pair labels are bidirectional only for bounded diagnostic/read semantics; they do not imply symmetric control, orchestration, or mutable region state exchange.

## 3) Deferred and blocked relations

All other canonical IR1 architecture-map pairs are not implemented in this first line:

| Pair | Architecture class from Prompt 1 | Prompt 2 implementation status | Reason |
|---|---|---|---|
| `Hippocampus ↔ Amygdala` | `caveated inter-region relation` | `deferred/not-yet-implemented relation` | Caveated context/salience coupling needs a later narrow caveat path before activation. |
| `Hippocampus ↔ Basal Ganglia` | `blocked relation` | `blocked relation` | Direct context/reference to action-channel or suppression authority remains unavailable. |
| `Hippocampus ↔ Cerebellum` | `reference-mediated relation` | `deferred/not-yet-implemented relation` | Reference-mediated timing/mismatch context waits until after the first three relations stabilize. |
| `Amygdala ↔ Cerebellum` | `deferred/not-yet-active relation` | `deferred/not-yet-implemented relation` | Salience-to-timing/prediction coupling remains delayed. |
| `Thalamus ↔ Basal Ganglia` | `selection-mediated relation` | `deferred/not-yet-implemented relation` | Selection-heavy relay/readiness coupling is intentionally not activated together with Amygdala ↔ Basal Ganglia. |
| `Thalamus ↔ Cerebellum` | `direct bounded advisory relation` | `deferred/not-yet-implemented relation` | Direct relay/timing advisory coupling waits; no all-to-all direct advisory layer is introduced. |
| `Basal Ganglia ↔ Cerebellum` | `execution-interface-mediated relation` | `deferred/not-yet-implemented relation` | Execution-interface-mediated diagnostics are deliberately not opened in this first implementation line. |

Deferred does not mean blocked, failed, retried, or automatically scheduled. Blocked does not mean failed execution.

## 4) No-direct-* and out-of-scope boundaries

The first implementation map is diagnostic/read-only and advisory-only. It preserves:

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
- no model-mode change.
- no Kuramoto expansion.
- no Hodgkin-Huxley production integration.

Reference-mediated remains Reference/Context-mediated; selection-mediated remains Selection/Contract-mediated; direct bounded advisory remains advisory-only. Non-canonical/internal-only paths cannot become operational relations.

## 5) Repo anchoring

The canonical code anchor for Prompt 2 is `CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`. It records relation class, source role, target role, signal types, mediation path, and no-direct guard booleans for the ten original Prompt 1 pairs while implementing only the three base pairs listed above. BR6 Prompt 2 extends that anchor to fifteen entries by adding the five Hypothalamus adjunct relations: Hippocampus ↔ Hypothalamus, Amygdala ↔ Hypothalamus, Thalamus ↔ Hypothalamus, Basal Ganglia ↔ Hypothalamus, and Cerebellum ↔ Hypothalamus. Those adjunct entries remain bounded/advisory-only and do not create direct action, execution, retry, memory, compute, or safety authority.

## 6) IR1 next steps

1. Add narrow consumer diagnostics that surface the three implemented relation statuses without changing runtime authority.
2. Add fixture/golden export only if an external artifact consumes the implementation map.
3. Decide whether the next implementation should be `Hippocampus ↔ Cerebellum` reference-mediated or `Thalamus ↔ Basal Ganglia` selection-mediated; do not activate both by default.
4. Keep `Basal Ganglia ↔ Cerebellum` execution-interface-mediated diagnostics deferred until a separate execution-interface guard review.
5. Keep `Hippocampus ↔ Basal Ganglia` direct relation blocked unless an explicit future mediated scope replaces the blocked direct edge.

## BR6 adjunct update

After BR6 Prompt 2, the original IR1 base-three implementation remains historically intact, but the runtime code map also carries the bounded Hypothalamus adjunct relation set. This update is not a new inter-region platform and not a rewrite of the original three high-leverage relations.
