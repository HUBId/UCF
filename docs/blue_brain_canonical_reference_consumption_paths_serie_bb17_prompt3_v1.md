# BlueBrain Canonical Reference Consumption Paths — Serie BB17 Prompt 3 (v1)

Status: canonical consumption hardening across runtime, selection, bounded dynamics, minimal execution, and combined retrieval.

## Canonical consumption classes

The canonical reference classifier now feeds one bounded consumption decision for each operational layer:

- `RuntimeCanonicalReferenceConsumption`
- `SelectionCanonicalReferenceConsumption`
- `DynamicsCanonicalReferenceConsumption`
- `ExecutionCanonicalReferenceConsumption`
- `RetrievalCanonicalReferenceConsumption`
- `NonCanonicalInternalOnlyReferenceConsumptionPath`

`NonCanonicalInternalOnlyReferenceConsumptionPath` is explicitly excluded from operational consumption and remains diagnostics-only.

## Allowed canonical reference forms per layer

- Runtime and bounded dynamics consume only execution-result evidence and reference-only/diagnostic evidence lanes as bounded advisory input.
- Selection consumes context/memory/combined/reference-only lanes as bounded advisory/candidate-only input.
- Execution consumes execution-result references only.
- Retrieval consumes canonical context/memory/execution/combined/reference-only/diagnostic forms as bounded candidate/advisory basis.

No layer upgrades non-canonical/internal-only references into canonical authority.

## Consumption boundary hardening

- Router Kuramoto input building now checks each evidence reference against canonical **dynamics** consumption rules.
- Non-canonical/internal-only evidence paths are surfaced explicitly and marked as `non_canonical_internal_only_path` in modulation input.
- Combined retrieval now checks context/candidate/proposal references against canonical retrieval/selection consumption decisions and emits caveats when non-canonical consumption paths are attempted.

## Excluded non-canonical/internal-only paths

The following remain excluded from canonical operational lanes:

- internal-only/non-canonical reference paths,
- implicit internal shortcut consumption,
- duplicate alternate reference-consumption authority lanes.

They may appear only as explicit caveats/diagnostics and never as direct operational authority.

## no-direct-* boundaries (kept)

Canonical reference consumption remains bounded and does **not** provide:

- direct action execution,
- direct retry orchestration,
- direct compute invocation,
- implicit memory persistence,
- policy/agent authority expansion,
- neurodynamics authority expansion beyond advisory diagnostics.
