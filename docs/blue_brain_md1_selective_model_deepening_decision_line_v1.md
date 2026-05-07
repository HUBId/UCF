# Architekturpaket MD1 Prompt 1: selective model-deepening decision line

Status: repo-basierte Entscheidungs- und Abgrenzungslinie nach IR1. Diese Linie entscheidet selektiv über Modellvertiefung für die fünf kontrolliert integrierten anatomischen Regionen und die IR1-Relationen. Sie baut keine globale Neurodynamikplattform, keine Gehirn-Vollsimulation und keine neue Compute-Core-Arbeit.

Canonical code anchor: `CANONICAL_BLUE_BRAIN_MD1_REGION_DEEPENING_DECISION_MAP` and `CANONICAL_BLUE_BRAIN_MD1_RELATION_DEEPENING_DECISION_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Canonical MD1 model-deepening classes

MD1 verwendet nur diese Vertiefungsklassen:

| Class | Meaning | Boundary |
|---|---|---|
| `abstract sufficient` | Current abstract functional surface is enough for UCF utility. | Keep current bounded contracts; no dynamics added. |
| `bounded Kuramoto-like candidate` | Coupling, synchrony, gating, or timing modulation has bounded UCF leverage. | Advisory-only/bounded; never direct action or execution control. |
| `HH simulation-only/diagnostic-only candidate` | Excitability/spiking/membrane-near modeling may produce diagnostic evidence. | Simulation/diagnostic only; not a productive default. |
| `later selective HH deepening` | HH-like depth is plausible only after a narrower implementation or diagnostic line proves value. | Explicit later rescope required. |
| `no-deepening-needed-now` | A surface is blocked, deferred, too early, or has no current system leverage. | Do not open model work now. |
| `non-canonical/internal-only model path` | Raw helper, research-only shortcut, or unlisted model path. | Not canonical and not consumable by Runtime, Selection, Reference, or Execution. |

These classes intentionally preserve the BB10/BB12/BB16 boundary: dynamics can modulate or diagnose only through bounded contracts, and HH is not a default production mode.

## 2) Region decision map

| Region | Current surface | MD1 deepening class | Reason | Next status |
|---|---|---|---|---|
| Hippocampus | abstract functional current mode | abstract sufficient | Context/reference/episode indexing already has bounded Reference and Selection surfaces; synchrony or membrane detail would not add near-term UCF utility. | Keep abstract. |
| Amygdala | bounded Kuramoto-like current/candidate lane | bounded Kuramoto-like candidate | Salience, caveat, and threat-priority timing can benefit from bounded coupling/synchrony modulation. | Candidate, but relation-level work is preferred first. |
| Thalamus | abstract functional current mode | abstract sufficient | Relay/routing integration is useful as a bounded relation endpoint; region-level dynamics would be premature. | Keep abstract. |
| Basal Ganglia | abstract functional current mode | no-deepening-needed-now | Action-gating mediation must remain contract-bound and non-authoritative; more dynamics risks action/execution scope confusion. | No model deepening now. |
| Cerebellum | abstract functional current mode | HH simulation-only/diagnostic-only candidate | Timing/mismatch/correction diagnostics could later use spiking/excitability evidence, but only offline or diagnostic-only. | Wait; not a next implementation candidate. |

Note: the broader anatomical map still keeps prefrontal cortex as a later selective HH deepening idea, but it is outside this MD1 pass because it is not one of the five currently controlled integrated regions.

## 3) Relation decision map

| Relation | IR1 implementation state | MD1 deepening class | Reason | Next status |
|---|---|---|---|---|
| Amygdala ↔ Thalamus | implemented direct bounded advisory relation | bounded Kuramoto-like candidate | Best current coupling/synchrony/gating target: salience-caveat pressure can modulate relay/routing posture without direct authority. | Priority 1. |
| Hippocampus ↔ Thalamus | implemented reference-mediated relation | abstract sufficient | Reference/context mediation is already the useful abstraction; synchrony would not improve the current contract. | Keep abstract. |
| Amygdala ↔ Basal Ganglia | implemented selection-mediated relation | bounded Kuramoto-like candidate | Salience-to-selection readiness can use bounded phase/coupling diagnostics while preserving action-gating boundaries. | Priority 2, after Priority 1 remains bounded. |
| Hippocampus ↔ Amygdala | deferred | no-deepening-needed-now | Caveated/deferred surface lacks enough implemented basis. | Wait. |
| Hippocampus ↔ Basal Ganglia | blocked | no-deepening-needed-now | Blocked relation; deepening would bypass the guard. | No model deepening. |
| Hippocampus ↔ Cerebellum | deferred | no-deepening-needed-now | Reference/timing relation is not implemented. | Wait. |
| Amygdala ↔ Cerebellum | deferred | no-deepening-needed-now | Salience/timing relation is too early. | Wait. |
| Thalamus ↔ Basal Ganglia | deferred | no-deepening-needed-now | Relay-to-action-channel path needs implementation scope before model depth. | Wait. |
| Thalamus ↔ Cerebellum | deferred | no-deepening-needed-now | Timing relay is plausible later, but not implemented now. | Wait. |
| Basal Ganglia ↔ Cerebellum | deferred execution-interface-mediated architecture | later selective HH deepening | Execution/timing diagnostics may eventually justify membrane-near simulation evidence, but only after explicit execution-interface scope hardening. | Candidate but wait. |

## 4) Kuramoto-like candidate boundary

Kuramoto-like depth is allowed only where bounded coupling/synchrony/gating/timing modulation has UCF leverage:

1. Priority 1: `Amygdala ↔ Thalamus` as a relation-level bounded Kuramoto-like candidate.
2. Priority 2: `Amygdala ↔ Basal Ganglia` as a relation-level bounded Kuramoto-like candidate after Priority 1 proves it stays bounded.
3. Region-level Amygdala remains a candidate, but MD1 does not prioritize standalone region dynamics over relation-level surfaces.

Kuramoto-like remains advisory-only. It may produce bounded phase/synchrony/caveat hints, but it is not direct Action control, not direct Execution control, not Retry orchestration, and not Memory mutation.

## 5) HH boundary

HH is heavier than Kuramoto-like and is not a productive default. MD1 permits HH only in these forms:

- `Cerebellum`: HH simulation-only/diagnostic-only candidate for timing/mismatch/correction evidence.
- `Basal Ganglia ↔ Cerebellum`: later selective HH deepening only after explicit execution-interface hardening.
- Broader prefrontal-cortex later-HH remains outside this pass.

HH outputs, if ever introduced, must down-map to diagnostics or caveats only. They must not directly select actions, trigger execution, orchestrate retries, mutate memory, invoke compute-core changes, or override safety.

## 6) Explicit out-of-scope boundaries

MD1 does not introduce:

- direct Action authority,
- direct Execution authority,
- Retry orchestration,
- Memory mutation or implicit persistence,
- Compute-Core expansion,
- a global model platform,
- a planner/agent platform,
- policy/governance platform work,
- retrieval/consolidation/reasoning platform work,
- global HH adoption.

Model depth must never undercut existing bounded contracts from Runtime, Selection, Reference, Execution, BB12, BB16, BB17, BB19, BB21, or IR1.

## 7) MD1 next steps

1. Specify the bounded Kuramoto-like relation input/output contract for `Amygdala ↔ Thalamus` only.
2. Add fixtures/goldens or narrow tests that prove the Kuramoto-like output remains advisory-only and cannot become Action, Execution, Retry, Memory, Compute, or Safety authority.
3. Only after that, evaluate whether `Amygdala ↔ Basal Ganglia` can reuse the same bounded modulation shape without promoting action-gating authority.
4. Keep Cerebellum HH as simulation-only/diagnostic-only research evidence; do not wire it into production Runtime or Execution.
5. Reconcile any future relation work against the IR1 diagnostics/contract map before adding model depth.
