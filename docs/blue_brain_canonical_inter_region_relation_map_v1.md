# Blue-Brain canonical inter-region relation map v1

Status: supporting current reference for the bounded IR1 relation closure. This document mirrors `CANONICAL_BLUE_BRAIN_CANONICAL_INTER_REGION_RELATION_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`; it does not create a second truth source, a new region, a global inter-region platform, planner/agent logic, policy governance, retry orchestration, productive Hodgkin-Huxley integration, or compute-core work.

## Closure categories

The canonical closure vocabulary is intentionally small:

| Category | Meaning |
| --- | --- |
| canonical implemented relation | Currently implemented direct bounded advisory relation; readable only as advisory/diagnostic contract state, never as direct coupling. |
| canonical mediated relation | Currently implemented relation whose active read path remains reference-mediated or selection-mediated. |
| canonical deferred relation | Architecture and implementation both record an inactive deferred relation; not blocked and not nearly active. |
| canonical blocked relation | Explicit unavailable/fail-closed relation path; not failed execution. |
| architectural lane only | The architecture map names a bounded lane, but the implementation lane remains deferred/not-yet-implemented. |
| non-canonical/internal-only relation path | No consumer-readable operational relation authority. |

Architecture lane and implementation status are separate fields: an architecture lane names bounded design intent, while implementation status decides whether a relation is currently readable. Implemented status still means advisory/read-only diagnostics, not strong operational coupling.

## Canonical relation table

| Pair | Architecture lane | Canonical closure status | Current implementation status | Mediation/read path |
| --- | --- | --- | --- | --- |
| Hippocampus ↔ Amygdala | caveated inter-region relation | architectural lane only | deferred/not-yet-implemented relation | NotYetImplemented |
| Hippocampus ↔ Thalamus | reference-mediated relation | canonical mediated relation | implemented reference-mediated relation | ReferenceContextMediatedOnly |
| Hippocampus ↔ Basal Ganglia | blocked relation | canonical blocked relation | blocked relation | BlockedUnavailable |
| Hippocampus ↔ Cerebellum | reference-mediated relation | architectural lane only | deferred/not-yet-implemented relation | NotYetImplemented |
| Amygdala ↔ Thalamus | direct bounded advisory relation | canonical implemented relation | implemented direct bounded advisory relation | DirectBoundedAdvisoryOnly |
| Amygdala ↔ Basal Ganglia | selection-mediated relation | canonical mediated relation | implemented selection-mediated relation | SelectionContractMediatedOnly |
| Amygdala ↔ Cerebellum | deferred/not-yet-active relation | canonical deferred relation | deferred/not-yet-implemented relation | NotYetImplemented |
| Thalamus ↔ Basal Ganglia | selection-mediated relation | architectural lane only | deferred/not-yet-implemented relation | NotYetImplemented |
| Thalamus ↔ Cerebellum | direct bounded advisory relation | architectural lane only | deferred/not-yet-implemented relation | NotYetImplemented |
| Basal Ganglia ↔ Cerebellum | execution-interface-mediated relation | architectural lane only | deferred/not-yet-implemented relation | NotYetImplemented |
| Hippocampus ↔ Hypothalamus | reference-mediated relation | canonical mediated relation | implemented reference-mediated relation | ReferenceContextMediatedOnly |
| Amygdala ↔ Hypothalamus | caveated inter-region relation | canonical implemented relation | implemented direct bounded advisory relation carrying caveated architecture context | DirectBoundedAdvisoryOnly |
| Thalamus ↔ Hypothalamus | direct bounded advisory relation | canonical implemented relation | implemented direct bounded advisory relation | DirectBoundedAdvisoryOnly |
| Basal Ganglia ↔ Hypothalamus | selection-mediated relation | canonical mediated relation | implemented selection-mediated relation | SelectionContractMediatedOnly |
| Cerebellum ↔ Hypothalamus | deferred/not-yet-active relation | canonical deferred relation | deferred/not-yet-implemented relation | NotYetImplemented |

Canonical active relations are therefore exactly three canonical implemented direct bounded advisory relations plus four canonical mediated relations. Completion-third-deepening closure reviews `Thalamus ↔ Cerebellum` first for timing/relay leverage, but leaves it unopened because this row remains architecture-lane-only and `NotYetImplemented`. All other named lanes are deferred, blocked, or architecture-lane-only.

## Mediation boundaries

- Reference-mediated remains reference-mediated: Hippocampus-heavy context/reference reads do not become memory commits or direct routing triggers.
- Selection-mediated remains selection-mediated: Basal-Ganglia/action-channel selection diagnostics do not become action execution authority.
- Execution-interface-mediated remains an architecture lane only in the current closure: Basal Ganglia ↔ Cerebellum is not implemented and cannot trigger execution.
- Direct bounded advisory remains advisory-only: implemented direct bounded advisory reads are contract diagnostics, not action, execution, retry, memory, compute, or safety signals.

## Relation semantics that must not drift

- advisory-only relation is not an action signal.
- caveated relation is not strong positive support.
- deferred relation is not blocked relation.
- blocked relation is not failed execution.
- diagnostic-only relation is not operative relation.
- architectural lane only is not active implementation.
- implemented relation is not strong operative coupling.

## Region-role cross-check

- Hippocampus relations stay context/reference/episode-indexing heavy.
- Amygdala relations stay salience/caveat heavy.
- Thalamus relations stay relay/gating/routing heavy.
- Basal Ganglia relations stay action-channel/selection heavy without action execution.
- Cerebellum relations stay prediction/timing/correction or execution-interface-diagnostic only.
- Hypothalamus relations stay drive/homeostasis/urgency/state-pressure heavy.

## No-direct and platform boundaries

The closure map preserves:

- no direct action trigger.
- no direct execution trigger.
- no direct retry trigger.
- no direct memory commit.
- no direct compute invocation.
- no safety override.
- no implicit global region orchestration.
- no implicit platform formation.

## Structural Closure Pack next steps

1. Use this compact relation basis to harden model-boundary wording for MD2/MD3 and Completion-third-deepening closure without adding another model-deepening candidate.
2. Refresh maintenance evidence/reports against this closure map so future cleanup can detect relation-status drift without opening new region or compute scope.
