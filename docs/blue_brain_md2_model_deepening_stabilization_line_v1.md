# Architekturpaket MD2 Prompt 1: model-deepening stabilization line

Status: maintenance-seitige Stabilisierung der bereits eingeführten ersten selektiven Modellvertiefung. Diese Linie erzeugt keine zweite Modellvertiefung, keine allgemeine Modellplattform, keine neue Autoritätsschicht und keine Compute-Core-Arbeit.

Canonical code anchor: `CANONICAL_BLUE_BRAIN_MD2_MODEL_DEEPENING_STABILIZATION_MAP`, `CANONICAL_BLUE_BRAIN_MD1_READINESS_MAP`, `BLUE_BRAIN_MD1_FIRST_DEEPENED_CANDIDATE_PAIR`, `BLUE_BRAIN_MD1_NEXT_MODEL_DEEPENING_DIRECTION`, and `evaluate_blue_brain_md1_first_model_deepening` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Stabilized baseline

The only stabilized first model-deepening baseline remains:

- relation: `Amygdala ↔ Thalamus`,
- mode: bounded Kuramoto-like current mode,
- role: advisory-only / diagnostic bounded support,
- architecture anchor: existing inter-region architecture and existing region/relation contracts,
- maintenance status: maintenance-only with frozen semantics.

`Amygdala ↔ Basal Ganglia` remains deferred and is not opened as a second model-deepening candidate in MD2 Prompt 1. HH-like work remains simulation-only/diagnostic-only or later-selective and is not productive here.

## 2) Canonical model-deepening stabilization map

`CANONICAL_BLUE_BRAIN_MD2_MODEL_DEEPENING_STABILIZATION_MAP` uses exactly these stabilization states:

| Stabilization state | Canonical meaning | Boundary |
|---|---|---|
| `stable deepened baseline` | The `Amygdala ↔ Thalamus` bounded Kuramoto-like relation remains the stable first deepening. | No second candidate and no global mode are opened. |
| `maintenance-hardened model surface` | Input/state/output model surfaces remain relation-local and advisory-only. | Model state does not become contract state or region authority. |
| `maintenance-hardened diagnostics path` | Kuramoto-like, caveated, deferred, blocked, insufficient, diagnostic-only, and non-canonical diagnostics stay distinguishable. | Diagnostic output is not operational authority. |
| `maintenance-hardened contract path` | Contract support classes stay separate from model and diagnostic classes. | Diagnostic-only, blocked, deferred, insufficient, and non-canonical states create no advisory support. |
| `maintenance-hardened model boundary` | Abstract, bounded Kuramoto-like, HH simulation-only/diagnostic-only, and later-selective HH remain separate. | No broader model platform or global HH/Kuramoto adoption is implied. |
| `non-canonical/internal-only residual path` | Unlisted helper, test-only, research, shortcut, or residual paths remain non-canonical/internal-only. | No canonical consumer read, no contract support, and no operational authority. |

## 3) Frozen semantics

The current deepening mode keeps its canonical meaning: bounded Kuramoto-like coupling/synchrony/gating/timing for `Amygdala ↔ Thalamus` only. It is not a planner, not action selection, not execution control, not retry orchestration, not memory persistence, not policy governance, not safety override, and not compute invocation.

The following distinctions are frozen for maintenance:

- model state is not contract state,
- model state is not region authority,
- diagnostic model output is not operational authority,
- advisory-only is not direct action or execution authority,
- caveated advisory-only is not strong operational support,
- deferred / blocked / insufficient / diagnostic-only are not advisory support,
- non-canonical/internal-only is not a consumable runtime/selection/reference/execution surface.

## 4) Guard rail line

MD2 Prompt 1 keeps the existing no-direct-* guards intact:

- no direct action trigger,
- no direct execution trigger,
- no direct retry trigger,
- no direct memory commit,
- no direct compute invocation,
- no safety override,
- no implicit second model-deepening candidate,
- no implicit global model platform.

The bounded advisory dynamics line remains compatible with BB10/BB12; bounded dynamics ↔ execution remains non-triggering under BB16; context/memory/reference hardening remains intact under BB17; runtime/selection contract semantics remain leading under BB19; execution/reference interaction remains reference-only under BB21.

## 5) Residual cleanup policy

Residual references to helper, test-only, shortcut, or research paths must either point back to the canonical first deepening or be marked non-canonical/internal-only. Documentation must not describe broader authority than the code exposes. Any future second deepening requires explicit re-scope after this first deepening remains stable under maintenance.

## 6) MD2 maintenance reference handoff

The maintenance-facing Doku/Test/Referenz entrypoint for this stabilized baseline is `docs/blue_brain_md2_model_deepening_docs_tests_reference_cleanup_v1.md`. That map keeps this MD2 Prompt 1 stabilization line as the canonical frozen baseline and classifies MD1 decision/implementation/hardening/closure docs as supporting references, not competing current authority.

## 7) MD2 next steps

1. Keep targeted regression checks around model/diagnostic/contract/surface distinctions.
2. Keep docs lint anchored to this single stabilized first-deepening line and the MD2 Prompt 2 maintenance reference map.
3. Review future Runtime/Selection/Reference changes against the stabilization map before merging them.
4. If a second candidate is later proposed, require explicit re-scope and no-direct-* proof before implementation.
5. Continue reducing non-canonical/internal-only residual wording rather than adding new model surfaces.


## MD3 Prompt 1 supersession note

MD2 remains the canonical maintenance baseline for the first model-deepening surface (`Amygdala ↔ Thalamus`). The later MD3 re-scope line `docs/blue_brain_md3_second_deepening_rescope_line_v1.md` is the only current authority for whether to prioritize a second candidate. MD3 prioritizes exactly one decision-only second candidate (`Amygdala ↔ Basal Ganglia`, bounded Kuramoto-like) and still does not implement that candidate, open HH as productive default, create direct action/execution/retry/memory/compute/safety authority, or create a global model platform.
