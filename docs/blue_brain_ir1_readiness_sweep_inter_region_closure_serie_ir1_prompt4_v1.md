# Blue-Brain IR1 Prompt 4: inter-region readiness sweep and closeout

Status: **IR1 closeout reference** for the initial bounded inter-region architecture. This file closes the Prompt-1 map, Prompt-2 implementation line, and Prompt-3 diagnostics/contract hardening without creating a second truth source: the canonical code anchor remains `CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP`, and this document only records the readiness reading of that map.

Current-baseline note (2026-05-09): this document is current for the **initial IR1 closeout semantics**, not for collapsing every later BR6/SC1 relation into an active relation. Read the tables below as a separation between architecture-lane existence and implementation-lane activation; `docs/blue_brain_sc1_prompt4_final_system_consolidation_sweep_v1.md` carries the current all-six-region relation split.

Scope: only the already opened anatomical regions `hippocampus_like_region`, `amygdala_like_region`, `thalamus_like_region`, `basal_ganglia_like_region`, and `cerebellum_like_region`. Hypothalamus is considered only as a next-direction decision, not opened by IR1. BR6 Prompt 2 supersedes that historical waiting state by adding a separate bounded Hypothalamus adjunct integration line.

## 1) IR1-readiness map

Prompt-4 relation counts are scoped to the initial five-region IR1 line. Architecture-map labels are not implementation activation by themselves.

| Readiness state | Prompt-4 meaning | Pair assignment |
|---|---|---|
| Stable implemented relation | Implemented in Prompt 2, visible through Prompt-3 relation diagnostics, advisory/read-only, and pinned to a canonical mediation path. | `Amygdala ↔ Thalamus`, `Hippocampus ↔ Thalamus`, `Amygdala ↔ Basal Ganglia` |
| Usable with caveats | Would mean an implemented relation whose canonical relation state is caveated but still usable as bounded diagnostics. No Prompt-4 pair is operationally usable-with-caveats. | none |
| Advisory-only | All stable implemented IR1 relations are advisory-only. This state is a hard authority limit, not a separate action channel. | applies to the three stable implemented relations |
| Deferred/not-yet-active | Known architecture-map edge remains inactive; it is not failed, retried, scheduled, or blocked unless explicitly listed as blocked. | `Hippocampus ↔ Amygdala`, `Hippocampus ↔ Cerebellum`, `Amygdala ↔ Cerebellum`, `Thalamus ↔ Basal Ganglia`, `Thalamus ↔ Cerebellum`, `Basal Ganglia ↔ Cerebellum` |
| Blocked/insufficient/diagnostic-only | Relation is unavailable or only diagnostic. Blocked does not mean failed execution; insufficient does not become caveated; diagnostic-only is not authority. | blocked: `Hippocampus ↔ Basal Ganglia`; insufficient: no canonical pair currently; diagnostic-only: deferred and blocked reads only |
| Non-canonical/internal-only | Any unlisted direct path, raw internal state path, test helper path, or expert/internal lane outside the canonical IR1 pair map. | no operational pair |

There is no operative `usable with caveats` pair after IR1 Prompt 4. Caveated vocabulary remains available in the diagnostics/contract semantics, but the caveated `Hippocampus ↔ Amygdala` architecture edge is still deferred/not-yet-active because Prompt 2 did not implement it.

## 2) Implemented relations and canonical signal paths

Exactly these three relations are operational in the initial IR1 architecture:

| Implemented pair | Readiness | Canonical mediation path | Canonical signal types | Operational boundary |
|---|---|---|---|---|
| `Amygdala ↔ Thalamus` | stable implemented relation, advisory-only | `DirectBoundedAdvisoryOnly` | `SalienceCaveatAdvisory` ↔ `RelayRoutingDiagnostic` | Bounded salience/caveat read can annotate relay/routing diagnostics only. |
| `Hippocampus ↔ Thalamus` | stable implemented relation, advisory-only | `ReferenceContextMediatedOnly` | `ContextReferenceDiagnostic` ↔ `RelayRoutingDiagnostic` | Context/reference reaches relay/routing only through the BB8/BB17 Reference/Context line. |
| `Amygdala ↔ Basal Ganglia` | stable implemented relation, advisory-only | `SelectionContractMediatedOnly` | `SalienceCaveatAdvisory` ↔ `SelectionReadinessDiagnostic` | Salience/caveat reaches readiness diagnostics only through the BB4/BB19 Selection/Contract line. |

The bidirectional pair label is a typed diagnostic/read label only. It is not symmetric control, mutable region-state exchange, broadcasting, or orchestration.

## 3) Deferred, blocked, diagnostic-only, and internal-only readings

- `Hippocampus ↔ Amygdala` remains deferred/not-yet-active even though the Prompt-1 architecture class is caveated inter-region relation.
- `Hippocampus ↔ Cerebellum` remains deferred/not-yet-active; no reference-mediated timing/mismatch path is operational yet.
- `Amygdala ↔ Cerebellum` remains deferred/not-yet-active; salience-to-timing/prediction coupling is not opened.
- `Thalamus ↔ Basal Ganglia` remains deferred/not-yet-active; Selection/Contract coupling is not activated as a second readiness lane.
- `Thalamus ↔ Cerebellum` remains deferred/not-yet-active; no second direct advisory lane is opened.
- `Basal Ganglia ↔ Cerebellum` remains deferred/not-yet-active; execution-interface-mediated diagnostics are not opened.
- `Hippocampus ↔ Basal Ganglia` remains blocked/unavailable as a direct context/reference-to-action-channel relation.
- Non-canonical/internal-only paths remain non-operational even when they appear in internal tests, expert surfaces, or implementation helpers.

## 4) Diagnostics and contract semantics

The canonical relation states and contract signals are:

| Relation state | Contract/diagnostic reading |
|---|---|
| `AdvisoryOnlyActive` | bounded relation contract signal; stable implemented relations read this state. |
| `CaveatedNoStrongPositiveSignal` | caveated relation diagnostic vocabulary only in this closeout; no Prompt-4 pair uses it as an operative relation. |
| `DeferredNotYetUsable` | deferred relation diagnostic; inactive and diagnostic-only. |
| `BlockedByContractSafetyOrReference` | blocked relation diagnostic; unavailable and diagnostic-only. |
| `InsufficientRelationalBasis` | insufficient relation diagnostic; no current canonical pair maps here. |
| `DiagnosticOnlyVisible` | diagnostic-only relation state; not a contract authority. |
| `NonCanonicalInternalOnly` | non-canonical/internal-only relation path; never canonical IR1 architecture. |

Runtime, Selection, and Reference must read the same relation state, relation diagnostic class, mediation path, and contract signal for a given pair. A bounded relation contract signal is not an action request, not an execution trigger, not a retry trigger, not a memory commit, not a compute trigger, and not a safety override.

## 5) No-direct-* and out-of-scope boundary

IR1 Prompt 4 preserves these hard boundaries:

- no direct Action Execution;
- no direct execution trigger;
- no Retry-Orchestrierung;
- no planner, agent, policy, or governance platform;
- no automatic allowed-actions expansion;
- no direct memory commit;
- no automatische Memory-Persistenz;
- no direct compute invocation;
- no safety override semantics;
- keine implizite globale Inter-Region-Plattform;
- no free broadcasting, routing fabric, or region-to-region orchestration;
- no retrieval, consolidation, or reasoning platform;
- no direct Hodgkin-Huxley production integration;
- no new anatomical region opened in IR1.

## 6) Compatibility with existing BlueBrain lines

IR1 remains subordinate to the existing lines:

- BB2 runtime/transition/feedback remains the runtime state line; IR1 does not introduce a second runtime state machine.
- BB4 selection/priority/deferral remains the selection line; IR1 does not select actions.
- BB8 and BB17 context/memory/reference hardening remain the reference line; IR1 does not commit or persist memory.
- BB12/BB16 bounded dynamics remain advisory-only when referenced; IR1 does not make dynamics authoritative.
- BB19 runtime/selection contract remains the contract boundary for selection-mediated reads.
- BB21 execution/reference interaction remains the execution-adjacent boundary; IR1 does not trigger execution.
- Non-canonical/internal-only paths remain internal-only and cannot become a second operative inter-region architecture.

## 7) Compute-core closeout

IR1 opens no compute-core work. Compute bleibt maintenance-only: the final compute line, outward-facing contracts, and maintenance-only core remain unchanged. IR1 relation diagnostics are reads over existing bounded surfaces, not new compute jobs, backends, scheduling, model execution, or worker behavior.

## 8) Next-direction decision

Prompt 4 prioritizes **selektive Modellvertiefung** over Hypothalamus.

Technical rationale:

1. The current five-region basis now has exactly three stable implemented relations and six deferred relations. Adding Hypothalamus would increase the pair map and mediation pressure before the existing relation diagnostics have produced a stronger model-depth basis.
2. Selective model deepening can improve the already opened and bounded surfaces without creating a new region, a new global platform, or new inter-region authority.
3. Hypothalamus should wait because its likely homeostasis/drive/regulation semantics are high-scope-risk: they could be mistaken for policy, safety override, planner, or global orchestration authority unless a later repo spec narrows the surface first.
4. No broad expansion is justified: the highest-leverage next step is one narrow model-depth improvement on an existing region or relation surface, not an additional all-to-all relation stage.

This decision does not reopen Compute and does not authorize a Hypothalamus integration in IR1.

## BR6 Prompt 2 supersession note

The IR1 readiness counts remain the historical five-region closeout basis. BR6 Prompt 2 separately extends the code-level inter-region diagnostics map to include Hypothalamus adjunct relations. That later extension keeps the same no-direct guards and does not reopen Compute, policy/governance, planner/agent, retry/orchestration or memory persistence scope.

For the current 2026-05-09 baseline, the all-six-region relation surface must therefore be read in two lanes:

| Reading lane | Meaning |
|---|---|
| Architecture lane exists | The bounded architecture map names a relation class such as direct bounded advisory, reference-mediated, selection-mediated, execution-interface-mediated, caveated, deferred, or blocked. |
| Implemented active relation | The implementation map exposes advisory/read-only diagnostics for that pair. |
| Deferred/not-yet-implemented relation | The architecture lane exists, but the implementation lane is inactive. |
| Blocked relation | The implementation is fail-closed/unavailable and must not be treated as a failed execution. |
