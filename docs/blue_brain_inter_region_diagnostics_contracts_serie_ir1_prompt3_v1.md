# Blue-Brain IR1 Prompt 3: inter-region diagnostics and contract semantics

Status: **bounded diagnostics/contract hardening line** for exactly the three original Prompt 2 implemented relations. This document hardens how Runtime, Selection, and Reference read relation diagnostics without adding a global inter-region platform, a planner/agent layer, retry orchestration, policy governance, new memory persistence, or compute-core work.

Canonical code anchor: `CANONICAL_BLUE_BRAIN_INTER_REGION_DIAGNOSTICS_CONTRACT_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`. It is derived from `CANONICAL_BLUE_BRAIN_FIRST_INTER_REGION_IMPLEMENTATION_MAP`; it does not open additional relations.

## 1) Canonical diagnostics/contract classes

IR1 Prompt 3 recognizes only these diagnostics/contract classes for inter-region reads:

| Canonical class | Meaning | Operational boundary |
|---|---|---|
| `advisory-only relation diagnostic` | A currently implemented relation exposes a positive bounded read. | Positive read only; no direct authority. |
| `caveated relation diagnostic` | A relation has only weak/partial/caveated context, Reference, or Selection basis and therefore has no strong positive signal. | Must not be promoted to advisory-only unless a later explicit implementation line does so. |
| `deferred relation diagnostic` | A relation is known by the architecture map but not active/not yet usable in this implementation line. | Deferred is not blocked, not insufficient, not failed, and not automatically scheduled. |
| `blocked relation diagnostic` | A limiting contract, safety, Reference, or explicit architecture boundary makes the relation unavailable. | Blocked is not insufficient and not failed execution; it starts no retry path. |
| `insufficient relation diagnostic` | No durable relational basis exists for a consumer to treat the relation as usable. | Insufficient is not blocked and not deferred; it remains diagnostic-only. |
| `diagnostic-only relation state` | A relation state may be visible for diagnostics but cannot be consumed as a bounded positive contract signal. | Visibility does not imply usability. |
| `bounded relation contract signal` | The only positive contract signal for the currently implemented relations. | Bounded/advisory-only; no action, execution, retry, memory, compute, or safety authority. |
| `non-canonical/internal-only relation path` | Raw helper, test-only, shortcut, or unlisted relation path. | Not an operational relation and not promotable by consumers. |

No other diagnostics/contract class is canonical for IR1 Prompt 3.

## 2) Implemented relations and current state

Prompt 3 hardens exactly the three Prompt 2 implemented relations:

| Pair | Prompt 2 class | Canonical mediation path | Prompt 3 relation_state | Prompt 3 contract signal |
|---|---|---|---|---|
| `Amygdala ↔ Thalamus` | `implemented direct bounded advisory relation` | `DirectBoundedAdvisoryOnly` | `AdvisoryOnlyActive` | `BoundedRelationContractSignal` |
| `Hippocampus ↔ Thalamus` | `implemented reference-mediated relation` | `ReferenceContextMediatedOnly` | `AdvisoryOnlyActive` | `BoundedRelationContractSignal` |
| `Amygdala ↔ Basal Ganglia` | `implemented selection-mediated relation` | `SelectionContractMediatedOnly` | `AdvisoryOnlyActive` | `BoundedRelationContractSignal` |

`AdvisoryOnlyActive` means a positive bounded relation signal is present, but it remains advisory-only. It is not action selection, not relay command, not memory commit, not compute invocation, and not safety authority.

## 3) Deferred, blocked, insufficient, and diagnostic-only separation

The non-implemented Prompt 2 relations remain explicitly separated:

- `DeferredNotYetUsable` is used for architecture edges that remain known but inactive/not-yet-usable.
- `BlockedByContractSafetyOrReference` is used for the explicit blocked edge, currently `Hippocampus ↔ Basal Ganglia`.
- `InsufficientRelationalBasis` is reserved for reads that lack any durable relation basis; it is distinct from blocked and deferred.
- `DiagnosticOnlyVisible` is reserved for relation visibility that must not be consumed as a positive contract signal.
- `NonCanonicalInternalOnly` is reserved for raw internal/helper paths and cannot become a consumer-visible operational relation.

Therefore: deferred is not blocked; blocked is not insufficient; insufficient is not deferred; diagnostic-only visibility is not bounded positive usability.

## 4) Runtime/Selection/Reference consumption rule

Runtime, Selection, and Reference read the same relation_state, relation_diagnostic_class, contract_signal_class, and mediation_path for a given pair. There is no Runtime-specific, Selection-specific, or Reference-specific reinterpretation of the same relation.

Allowed consumer reads are limited to:

- observe the canonical `relation_state`,
- observe the canonical diagnostics/contract class,
- preserve the canonical mediation path,
- preserve no-direct-* boundaries.

Consumers must not infer extra authority from consumer-layer identity. A Runtime read of `Hippocampus ↔ Thalamus` remains `ReferenceContextMediatedOnly`; a Selection read of `Amygdala ↔ Basal Ganglia` remains `SelectionContractMediatedOnly`; a Reference read of `Amygdala ↔ Thalamus` remains `DirectBoundedAdvisoryOnly`.

## 5) No-direct-authority contract signal boundaries

A relation contract signal is not an action request, not an execution trigger, not a retry trigger, not a memory commit, not a compute trigger, and not a safety override.

The following remain explicitly out of scope:

- no direct action trigger,
- no direct execution trigger,
- no direct retry trigger,
- no retry orchestration,
- no direct memory commit,
- no automatic memory persistence,
- no direct compute invocation,
- no safety override,
- no allowed-actions extension,
- no policy/governance platform,
- no planner/agent platform,
- no retrieval/consolidation/reasoning platform,
- no global inter-region orchestration,
- no new inter-region platform formation,
- no new Compute Core work.

## 6) Mediation path preservation

Mediation paths are semantic boundaries, not shortcuts:

- `DirectBoundedAdvisoryOnly` remains direct bounded advisory-only.
- `ReferenceContextMediatedOnly` remains Reference/Context-mediated and cannot bypass Reference.
- `SelectionContractMediatedOnly` remains Selection/Contract-mediated and cannot become action-channel authority.
- `NotYetImplemented` remains deferred/not-yet-usable.
- `BlockedUnavailable` remains blocked and unavailable.
- `NonCanonicalInternalOnly` remains non-canonical/internal-only.

Execution-interface-mediated architecture edges remain deferred in Prompt 3 unless a later explicit prompt opens a narrow execution-interface diagnostic relation. Prompt 3 does not implement the `Basal Ganglia ↔ Cerebellum` execution-interface-mediated relation.

## 7) Non-canonical/internal-only handling

Loose helper paths, test-only signals, raw region internals, and unlisted relation shortcuts must be removed or marked as `non-canonical/internal-only relation path` before any consumer can see them. They cannot carry operational effects and cannot override the canonical Prompt 2 implementation map.

## 8) IR1 next steps

1. Add a narrow exported diagnostic artifact only if an external docs/spec consumer needs it.
2. Add consumer-side assertions in Runtime/Selection/Reference crates if they begin importing these reads directly.
3. Decide one next relation candidate in a later prompt; do not activate multiple deferred relations together.
4. Review execution-interface-mediated diagnostics separately before opening `Basal Ganglia ↔ Cerebellum`.
5. Keep non-canonical/internal-only paths non-operational and visible only as diagnostics.

## BR6 Hypothalamus adjunct note

BR6 Prompt 2 adds bounded Hypothalamus adjunct contract reads to the same diagnostics map. Runtime, Selection and Reference continue to read those entries as bounded Contract-/Diagnostic-Reads only; deferred, blocked, insufficient, diagnostic-only and non-canonical states remain distinct and no direct action, execution, retry, memory, compute or safety authority is added.
