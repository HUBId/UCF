# Architekturpaket MD1 Prompt 2: first model-deepening implementation line

Status: schmale, repo-basierte erste Modellvertiefung nach MD1 Prompt 1. Diese Linie vertieft genau einen Kandidaten minimal und kontrolliert. Sie baut keine allgemeine Neurodynamikplattform, keine globale HH/Kuramoto-Einführung, keine Planner-/Agentenlogik und keine neue Compute-Core-Arbeit.

Canonical code anchor: `BLUE_BRAIN_MD1_FIRST_DEEPENED_CANDIDATE_PAIR`, `CANONICAL_BLUE_BRAIN_MD1_FIRST_DEEPENING_INTEGRATION_MAP`, and `evaluate_blue_brain_md1_first_model_deepening` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Deepened candidate

The first deepened candidate is exactly:

- `Amygdala ↔ Thalamus`
- relation-level only
- `bounded Kuramoto-like candidate`
- direct bounded advisory relation from IR1
- minimal coupling/synchrony/gating/timing modulation only

`Amygdala ↔ Basal Ganglia` remains a prioritized-but-deferred candidate and is not deepened in this implementation line. Cerebellum and `Basal Ganglia ↔ Cerebellum` HH-like paths remain simulation-only/diagnostic-only or later-selective candidates and are not opened here.

## 2) First model-deepening integration map

MD1 Prompt 2 canonically separates these surfaces and paths:

| Integration path | Canonical meaning |
|---|---|
| `deepened candidate input surface` | The bounded Kuramoto-like input surface for the selected relation. It may read runtime posture, selection posture, bounded context/reference/evidence refs, memory caveats, allowed phase nodes, and bounded execution/reference feedback classes already accepted by BB10/BB12/BB16. |
| `deepened candidate state surface` | The relation identity, current model mode, MD1 deepening class, IR1 implementation class, mediation path, and leverage flags. |
| `deepened candidate output/advisory surface` | Advisory-only, caveated-advisory, deferred, blocked, insufficient, diagnostic-only, or non-canonical/internal-only result classification. |
| `deepened candidate diagnostic/model surface` | Kuramoto-like model diagnostic classification without action, execution, retry, memory, compute, or safety authority. |
| `blocked/deferred deepening path` | Prioritized-but-not-selected, abstract-sufficient, no-deepening-needed, or not-yet-implemented paths. |
| `non-canonical/internal-only deepening path` | Any unlisted/helper/research-only route; not consumable as canonical Runtime, Selection, Reference, or Execution authority. |

This map is intentionally small. It is not a meta-platform and does not define reusable model orchestration for additional regions.

## 3) Inputs and state boundaries

The deepened `Amygdala ↔ Thalamus` model may read only the existing bounded Kuramoto-like input classes:

- selection posture,
- runtime posture,
- selected context refs,
- selected evidence refs,
- memory caveat refs as caveats only,
- canonical phase nodes from runtime, selection, context/reference, memory-caveat, and evidence-derived advisory groups,
- canonical/caveated/blocked/insufficient/unavailable/diagnostic-only execution/reference feedback classes already bounded by the dynamics line.

It must not read or mutate raw action state, execution queues, retry queues, memory store internals, compute backends, policy decision authority, or safety override state.

The state surface remains relation-local:

- pair: `Amygdala ↔ Thalamus`,
- current model mode: bounded Kuramoto-like current mode,
- deepening class: bounded Kuramoto-like candidate,
- mediation path: direct bounded advisory only,
- leverage: coupling/synchrony/gating/timing only,
- no excitability/spiking/membrane leverage.

## 4) Output and diagnostics semantics

Allowed outputs are classification-only and bounded:

- `advisory-only`,
- `caveated advisory-only`,
- `deferred`,
- `blocked`,
- `insufficient`,
- `diagnostic-only`,
- `non-canonical/internal-only`.

The same categories remain distinct from the existing BB10/BB12/BB16/IR1 categories. `diagnostic-only` is not promoted to advisory authority. `caveated` is not promoted to successful/strong basis. `deferred`, `blocked`, and `insufficient` remain explicit non-authority states.

Runtime, Selection, and Reference may read the result only as bounded advisory/diagnostic information. The first-deepening result records this as bounded reads and does not create a separate authority layer.

## 5) No-direct-* and out-of-scope guard line

The first deepening explicitly forbids:

- direct action selection,
- direct execution trigger,
- direct retry trigger or retry orchestration,
- direct memory commit or automatic memory persistence,
- direct compute invocation,
- safety override,
- policy/governance authority,
- second simultaneous model deepening,
- global Kuramoto/HH platform formation.

Boundary guards on the underlying Kuramoto-like result remain false for action execution, retry orchestration, memory commit, compute invocation, and safety override.

## 6) MD1 follow-up steps

1. Add a narrow fixture/golden only for `Amygdala ↔ Thalamus` if future consumers need serialized evidence.
2. Validate whether `Amygdala ↔ Basal Ganglia` can remain deferred while sharing only documentation vocabulary, not runtime authority.
3. Keep HH lines simulation-only/diagnostic-only and add no productive HH path without a separate explicit MD1 rescope.
4. Re-check Runtime/Selection/Reference reads before any future model output is consumed outside diagnostics.
5. Re-run IR1 diagnostics/contract consistency tests before opening any second relation-level deepening.
