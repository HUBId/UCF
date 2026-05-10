# Blue-Brain relation wording findings map v1

Status: narrow maintenance-facing relation wording cleanup for the current post-BR6/IR1/MD2/MD3/SC1 UCF Blue-Brain state. This document records wording and boundary findings only. It does not add a brain region, a model-deepening candidate, a global relation/model platform, planner/agent logic, policy/governance authority, retry/orchestration behavior, retrieval/consolidation/reasoning scope, memory persistence, allowed-actions expansion, HH production integration, or compute-core work.

Code anchor: `CANONICAL_BLUE_BRAIN_RELATION_WORDING_FINDINGS_MAP` and `CANONICAL_BLUE_BRAIN_RELATION_WORDING_FINDING_CLASS_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Finding classes

| Finding class | Maintenance meaning | Required wording response |
| --- | --- | --- |
| `implemented relation unclear` | A relation can be read as active merely because it exists in the architecture map, or as stronger than advisory/read-only because it is implemented. | Say that implemented means active only in the implementation/diagnostics lane, pinned to its canonical mediation path, advisory/read-only, and no-direct-authority. |
| `deferred relation unclear` | A deferred edge can be mistaken for blocked, failed, retried, scheduled, or silently active. | Say that deferred means architecture-known but inactive/not-yet-implemented, diagnostic-only, not failed, not blocked, not retried and not automatically scheduled. |
| `blocked relation unclear` | A blocked edge can be mistaken for failed execution or retry-eligible runtime state. | Say that blocked means fail-closed/unavailable diagnostics, not failed execution, and no action/execution/retry consequence. |
| `mediation-path ambiguity` | Mediation labels can be collapsed into full region-to-region coupling or bypassable shortcuts. | Preserve each named mediation path as the boundary; deny bypass, full coupling and authority escalation. |
| `contract wording drift` | Bounded contract signal wording can drift from diagnostic/read-only into operational authority. | Keep contract signals bounded advisory/read-only and explicitly deny action, execution, retry, memory, compute and safety authority. |
| `no-change-needed finding` | A reviewed surface is already clear enough; the entry records evidence and the current maintenance reading. | Keep wording stable and avoid creating a second truth source. |

## 2) Findings map

| Surface | Finding | Observed ambiguity | Maintenance cleanup |
| --- | --- | --- | --- |
| Relation status tables | `implemented relation unclear` | Implemented relation labels can be read as architecture-class activation unless the implementation lane is named. | Use **implemented active relation** only for entries exposed by the implementation/diagnostics map; implemented remains advisory/read-only and no-direct-authority. |
| Relation status tables | `deferred relation unclear` | Deferred architecture edges can be mistaken for blocked, failed, retried, or scheduled relations. | Use **deferred/not-yet-implemented** for inactive architecture-known edges; deferred is diagnostic-only, not blocked, not failed and not automatically scheduled. |
| Relation state maps | `blocked relation unclear` | Blocked relation wording can be misread as failed execution or retry-eligible runtime state. | Use **blocked/fail-closed/unavailable** for blocked diagnostics; blocked is not failed execution and starts no retry path. |
| Mediation path wording | `mediation-path ambiguity` | Reference-, selection-, execution-interface- and direct-advisory paths can be collapsed into broad inter-region coupling. | State that reference-mediated remains reference-mediated; selection-mediated remains selection-mediated; execution-interface-mediated remains bounded diagnostic/read-only; direct bounded advisory remains advisory-only. |
| Diagnostics/contract wording | `contract wording drift` | `BoundedRelationContractSignal` can be read as an action, execution, retry, memory, compute or safety signal. | Keep bounded contract signal as a relation diagnostic/read signal only; it is not action, execution, retry, memory mutation, compute or safety authority. |
| Relation authority boundaries | `no-change-needed finding` | No-direct guards were reviewed for relation-facing action, execution, retry, memory mutation and compute authority. | No code behavior change needed: relation maps and diagnostics already deny direct authority. |
| Region/model-deepening references | `no-change-needed finding` | Region and model-deepening references were checked for accidental seventh-region, third-deepening, platform, or compute-core readings. | No expansion wording needed: current six anatomical regions and exactly two selective model-deepening lines remain bounded and maintenance-only. |

## 3) Status separation rules

- **Implemented active relation** means the implementation map exposes an advisory/read-only diagnostic relation for that pair. It is active only in the bounded relation diagnostics/contract sense.
- **Deferred relation** means the architecture lane is known but the implementation lane is inactive. It is not a failed execution, not a retry state, not blocked by itself, and not automatically scheduled.
- **Blocked relation** means fail-closed/unavailable. It is diagnostic-only and must not be treated as failed execution.
- **Diagnostic-only relation state** means visible for diagnostics only. It is not a bounded positive contract signal and not an operative relation.
- **Caveated relation wording** means quality-, salience-, confidence- or context-limited signal vocabulary. It is not strong relation authority and does not activate an implementation lane by itself.

## 4) Mediation-path preservation rules

- `ReferenceContextMediatedOnly`: reference-mediated remains reference-mediated; no Reference/Context bypass and no memory mutation authority.
- `SelectionContractMediatedOnly`: selection-mediated remains selection-mediated; no action-channel authority and no action selection.
- `DirectBoundedAdvisoryOnly`: direct bounded advisory remains advisory-only; no command, routing fabric, broadcast, execution or compute authority.
- `BlockedUnavailable`: blocked remains fail-closed/unavailable; no execution failure interpretation and no retry.
- `NotYetImplemented`: deferred remains inactive; no implicit activation.
- Execution-interface-mediated architecture wording remains bounded diagnostic/read-only unless an explicit later implementation line opens a narrow relation. It is not execution authority.

## 5) Relation authority boundaries

All relation wording in this maintenance pass preserves:

- no relational Action authority;
- no relational Execution authority;
- no relational Retry authority;
- no relational Memory-mutation authority;
- no relational Compute authority;
- no safety override;
- no allowed-actions expansion;
- no global inter-region platform;
- no planner/agent, policy/governance, retry/queue/orchestration, retrieval/consolidation/reasoning or model-platform expansion.

## 6) Cleanup applied

This pass deliberately stayed narrow:

- introduced the code-backed relation wording findings map;
- added test assertions that the map contains implemented/deferred/blocked/mediation/contract/no-change buckets;
- pinned no-direct relation authority denial across the findings entries;
- added this maintenance-facing wording map to the docs index;
- left relation implementation behavior unchanged.

## 7) Checks and remaining maintenance need

Targeted checks should verify that relation states are easier to read, architecture-vs-implementation stays separated, mediation paths stay named, docs do not contradict tests, and no scope expansion occurs.

Remaining maintenance need: low. Continue to prefer this map when touching relation tables, diagnostics/contract wording, BR6 adjunct relation wording, or MD2/MD3 relation references. No new region, no third model-deepening candidate and no compute-core work is indicated by this pass.
