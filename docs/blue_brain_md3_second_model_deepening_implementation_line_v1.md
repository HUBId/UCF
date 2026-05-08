# Architekturpaket MD3 Prompt 2: second model-deepening implementation line

Canonical code anchor: `CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_INTEGRATION_MAP`, `CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_HARDENING_MAP`, `BLUE_BRAIN_MD3_SECOND_DEEPENED_CANDIDATE_PAIR`, and `evaluate_blue_brain_md3_second_model_deepening` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Candidate and model form

MD3 Prompt 2 implements exactly the Prompt 1 priority:

- **Candidate:** `Amygdala ↔ Basal Ganglia` relation.
- **Model form:** bounded Kuramoto-like.
- **Minimal deepening:** limited coupling/synchrony/gating/timing modulation for salience/caveat to selection-readiness timing.
- **Not implemented:** Hodgkin-Huxley productive behavior, global HH, global Kuramoto platform, full neural simulation, planner/agent logic, retry orchestration, policy governance, retrieval/consolidation/reasoning platform, or compute-core work.

MD1 first deepening remains Amygdala ↔ Thalamus. It is not reinterpreted as the MD3 second deepening, and the MD3 second state is not allowed to overwrite first-deepening state.

## 2) Canonical second-deepening integration map

`CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_INTEGRATION_MAP` is the only MD3 Prompt 2 integration map. It distinguishes:

| Map class | Canonical meaning | Boundary |
| --- | --- | --- |
| `second deepened input surface` | The relation-local bounded input read for `Amygdala ↔ Basal Ganglia`. | Reads selected context refs, evidence refs, caveats, and bounded phase-node groups only. |
| `second deepened state surface` | The candidate state for the second deepened relation. | Relation-local; no region authority and no first-deepening overwrite. |
| `second deepened output/advisory surface` | Advisory-only bounded Kuramoto-like modulation result. | May be read as a hint/caveat only. |
| `second deepened diagnostic/model surface` | Diagnostic classification for advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only, or non-canonical status. | Diagnostics do not become operational authority. |
| `second deepened region/relation contract surface` | Link back to existing inter-region relation contracts. | Existing Runtime/Selection/Reference contracts remain leading. |
| `blocked/deferred second deepening path` | Explicit no-support path for non-active or first-baseline/non-selected candidates. | No fallback activation. |
| `non-canonical/internal-only second deepening path` | Internal-only residual path. | No canonical Runtime/Selection/Reference read. |

No new meta-platform is introduced.

## 2a) MD3 Prompt 3 hardening overlay

MD3 Prompt 3 adds the **second model-deepening hardening line** without widening the implementation. `CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_HARDENING_MAP` pins exactly these hardening classes: hardened second deepened input surface, hardened second deepened state surface, hardened second deepened output/advisory surface, hardened second diagnostic/model boundary, hardened second region/relation contract boundary, blocked forbidden authority path, and non-canonical/internal-only second deepening path.

The overlay keeps the following boundaries canonical: model state is not contract state; diagnostic model output is not advisory support; diagnostic model output is not operational authority; caveated model signal is not strong operational input; second model-deepening state is not region authority; and first model-deepening state is not second model-deepening state. The blocked forbidden authority path keeps direct action, direct execution, direct retry, direct memory commit, automatic memory persistence, direct compute invocation, safety override, third model deepening, and global model platform paths closed.

## 3) Inputs and outputs

### Allowed inputs

The second deepening may read only bounded relation-local inputs already accepted by the bounded dynamics surface:

- `pair = Amygdala ↔ Basal Ganglia`;
- `BlueBrainKuramotoScopeState` values that remain advisory/diagnostic, especially selection-modulating, runtime-caveat-modulating, and diagnostic-only reads;
- selected context references;
- selected evidence references;
- memory caveats as caveat references, not commit authority;
- canonical phase-node groups for runtime state, selection attention, context/reference, memory-caveat reference, or evidence-derived advisory groups;
- explicit unsupported, blocked, insufficient, unavailable, failed, cancelled, or diagnostic-only feedback refs as diagnostics/caveats.

### Allowed outputs

The second deepening may produce only:

- `advisory-only` bounded second support;
- `caveated` bounded second support;
- `deferred` no-support status;
- `blocked` no-support status;
- `insufficient` no-support status;
- `diagnostic-only` no-advisory-support status;
- `non-canonical/internal-only` no-support status.

The output can inform Runtime, Selection, and Reference only as bounded reads. It does not select actions, execute work, retry work, commit memory, call compute, or override safety.

## 4) Diagnostics and contracts

The diagnostic/contract vocabulary remains separated:

- `advisory-only` is a bounded positive read, not direct authority.
- `caveated` is usable only with visible caveats and cannot become strong support.
- `deferred` is not active and is distinct from `blocked`.
- `blocked` is fail-closed and distinct from `insufficient`.
- `insufficient` reports missing basis and cannot be promoted to selection/execution.
- `diagnostic-only` is observable model feedback with no advisory support.
- `non-canonical/internal-only` is not a canonical consumer read.

The second deepening uses second-specific output, diagnostic, contract-support, and consumer-read classes so current model mode and deepened mode remain explicitly classed. Existing MD1/MD2 first-deepening classes remain separate.

## 5) Runtime, Selection, and Reference attachment

Runtime, Selection, and Reference may read the MD3 result only through bounded read classes:

- Runtime may receive a bounded caveat/readiness hint.
- Selection may receive a bounded readiness/timing hint.
- Reference/Context may receive bounded diagnostic context.

These reads do not introduce a new authority layer and do not alter existing relation contracts.

## 6) No-direct and out-of-scope boundaries

MD3 Prompt 2 explicitly keeps these paths forbidden:

- no direct action selection;
- no direct action trigger;
- no direct execution trigger;
- no direct retry trigger;
- no direct memory commit;
- no automatic memory persistence;
- no direct compute invocation;
- no safety override;
- no third model deepening;
- no global model platform;
- no global Kuramoto platform;
- no Hodgkin-Huxley productive integration.

## 7) Next MD3 steps

1. Exercise this second line with additional relation-local fixtures if future prompts need stronger evidence.
2. Keep MD1 `Amygdala ↔ Thalamus` maintenance tests separate from MD3 `Amygdala ↔ Basal Ganglia` tests.
3. Add only narrow consumer-facing diagnostics if Runtime/Selection/Reference docs need more examples.
4. Re-check non-canonical/internal-only paths before any future model-deepening candidate is considered.
5. Do not open a third candidate without an explicit re-scope prompt.
