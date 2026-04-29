# Serie BB21 Prompt 3: runtime/selection/retrieval cross-line reference consumption consistency pass

Status: Dieser Pass konsolidiert **eine gemeinsame kanonische Referenzkonsumlinie** für Runtime, Selection und Retrieval, ohne neue Retrieval-/Planner-/Governance-/Compute-Plattformen einzuführen.

## Kanonische Cross-line-Consumption-Klassen

Die drei Linien Runtime, Selection und Retrieval konsumieren dieselben kanonischen Referenzformen:
- `ContextReference`
- `MemoryRecordReference`
- `ExecutionResultReference`
- `CombinedBoundedReference`
- `DiagnosticReference`
- `ReferenceOnlyNotMemoryOrResult`

`NonCanonicalInternalOnlyPath` bleibt als eigener non-canonical/internal-only Pfad fail-closed blockiert.

## Strong / Weak / Reference-only

Jede Consumption-Entscheidung wird zusätzlich als Stärke klassifiziert:
- `StrongReferenceConsumption`: current + kanonisch + nicht reference-only.
- `WeakReferenceConsumption`: caveated/stale/invalidated/blocked/insufficient oder weak execution basis.
- `ReferenceOnlyConsumption`: diagnostic/reference-only lanes; nie operative support basis.

Damit bleibt explizit erhalten:
- strong ≠ weak,
- weak ≠ reference-only,
- reference-only ist advisory/candidate-only und nicht execution-authoritative.

## Gültigkeitszustände (einheitlich über die Linien)

Die Zustände bleiben unverändert kanonisch und werden in allen drei Linien gleich gelesen:
- `current`
- `caveated`
- `stale`
- `invalidated`
- `blocked`
- `insufficient`
- `reference_only`
- `non_canonical_internal_only_path`

## Bounded / no-direct-* Grenzen

Die Konsolidierung ändert bewusst **nicht**:
- keine direkte Action-Execution aus Referenzkonsum,
- keine Retry-Orchestrierung,
- keine Compute-Invocation aus Reference-only/weak Pfaden,
- keine implizite Memory-Persistenz,
- keine Policy-/Reasoning-/Agentenplattform.

Execution-Layer bleibt separat streng: nur canonical, current execution result reference ist dort als operative Basis erlaubt.
