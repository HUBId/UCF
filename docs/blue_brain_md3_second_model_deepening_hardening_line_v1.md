# Architekturpaket MD3 Prompt 3: second model-deepening hardening line

Canonical code anchor: `CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_HARDENING_MAP`, `CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_INTEGRATION_MAP`, `BLUE_BRAIN_MD3_SECOND_DEEPENED_CANDIDATE_PAIR`, and `evaluate_blue_brain_md3_second_model_deepening` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

Status: MD3 Prompt 3 hardens the already implemented second model-deepening candidate without widening it. The only second deepened candidate remains the `Amygdala ↔ Basal Ganglia` relation. The first deepened candidate remains `Amygdala ↔ Thalamus`.

## 1) Canonical second model-deepening hardening map

`CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_HARDENING_MAP` is the canonical hardening map for MD3 Prompt 3. It deliberately mirrors the existing integration surface, but names the guard boundaries that prevent model-state, diagnostics, contracts, and region/relation authority from collapsing into one another.

| Hardening class | Canonical boundary | Explicitly forbidden promotion |
| --- | --- | --- |
| `hardened second deepened input surface` | `Amygdala ↔ Basal Ganglia` may read only relation-local bounded Kuramoto-like inputs already accepted by the bounded dynamics surface. | No direct action, execution, retry, memory, compute, safety, or new region input path. |
| `hardened second deepened state surface` | The second model state is relation-local model state. | Model state is not contract state and is not region authority. |
| `hardened second deepened output/advisory surface` | Output may be advisory-only or caveated bounded support only when the bounded dynamics result allows it. | Caveated model signal is not strong operational input. |
| `hardened second diagnostic/model boundary` | Diagnostic model output remains diagnostic unless separately classed as bounded advisory support. | Diagnostic model output is not advisory support and is not operational authority. |
| `hardened second region/relation contract boundary` | Existing region/relation contracts remain leading for Runtime, Selection, and Reference reads. | The second deepening cannot rewrite inter-region architecture or create a model-led architecture. |
| `blocked forbidden authority path` | direct action, direct execution, direct retry, direct memory commit, automatic memory persistence, direct compute invocation, safety override, third deepening, and global model platform are fail-closed. | No fallback activation through blocked/deferred/insufficient/diagnostic-only status. |
| `non-canonical/internal-only second deepening path` | Internal-only residual paths have no canonical Runtime/Selection/Reference read. | No second operational model reality. |

## 2) Model state, diagnostics, and contract state stay separated

The second model-deepening result keeps separate state for:

- model state: relation-local `Amygdala ↔ Basal Ganglia` bounded Kuramoto-like state;
- diagnostic state: second-specific diagnostic classes for advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only, and non-canonical/internal-only reads;
- contract support state: second-specific support/no-support classes consumed by existing contract surfaces;
- consumer read state: one consistent Runtime/Selection/Reference read class, or no canonical second consumer read.

Canonical invariants:

- model state is not contract state;
- diagnostic model output is not advisory support;
- diagnostic model output is not operational authority;
- caveated model signal is not strong operational input;
- second model-deepening state is not region authority;
- first model-deepening state is not second model-deepening state.

## 3) Status vocabulary remains stable

The hardening line preserves the existing vocabulary without reinterpretation:

- `advisory-only` remains bounded support only;
- `caveated` remains caveated bounded support and never strong support;
- `deferred` remains inactive no-support;
- `blocked` remains fail-closed no-support;
- `insufficient` remains missing-basis no-support;
- `diagnostic-only` remains observable model feedback with no advisory support;
- `non-canonical/internal-only` remains outside canonical consumer reads.

## 4) Runtime, Selection, and Reference read the same boundary

Runtime, Selection, and Reference do not invent separate meanings for the second deepened state. The only canonical consumer read is the shared bounded advisory/diagnostic read class. If any second path is blocked, deferred, insufficient, diagnostic-only, or non-canonical/internal-only, it must not create an independent Runtime, Selection, or Reference authority.

## 5) Relation and region boundaries remain leading

The second deepening is embedded under the existing relation architecture. It does not turn the functional region/relation architecture into a model architecture. `Amygdala ↔ Basal Ganglia` is the only second deepened relation, and it remains bounded to coupling/synchrony/gating/timing evidence for salience/caveat and selection-readiness timing hints.

## 6) First and second deepening remain separate

- First deepening: `Amygdala ↔ Thalamus`, maintenance-hardened MD1/MD2 bounded Kuramoto-like advisory/diagnostic line.
- Second deepening: `Amygdala ↔ Basal Ganglia`, MD3 bounded Kuramoto-like advisory/diagnostic line.

They use separate candidate classes, output classes, diagnostic classes, contract-support classes, boundary states, and consumer-read classes. No hierarchy, roll-up score, hidden reconciliation layer, or global model layer is introduced.

## 7) Out-of-scope guard line

Still explicitly out of scope:

- no direct action trigger;
- no direct execution trigger;
- no direct retry trigger;
- no retry orchestration;
- no direct memory commit;
- no automatic memory persistence;
- no direct compute invocation;
- no safety override;
- no allowed-actions expansion;
- no third model-deepening candidate;
- no global Kuramoto or Hodgkin-Huxley platform;
- no planner, agent, policy-governance, retrieval, consolidation, or reasoning platform.

## 8) MD3 follow-up steps

1. Add narrow fixtures only when a concrete consumer-read ambiguity appears.
2. Keep MD1/MD2 first-deepening maintenance checks separate from MD3 second-deepening checks.
3. Keep Runtime/Selection/Reference docs synchronized with the shared consumer-read class.
4. Treat non-canonical/internal-only residuals as cleanup candidates, not as future platform hooks.
5. Revisit additional model candidates only through a separate explicit re-scope package.
