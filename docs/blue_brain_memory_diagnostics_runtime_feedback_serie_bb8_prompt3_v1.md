# Serie BB8 Prompt 3: Memory commit/retrieval diagnostics und runtime feedback bind-back

Status: Dieser Schritt ergänzt die BB8 Prompt 1/2 Memory-Basis um eine **kanonische Diagnostics- und Feedback-Schicht** für Commit und Retrieval. Die Linie bleibt minimal, deterministisch und klar getrennt von Monitoring/Explainability/Ranking/Reasoning.

## Scope und betroffene Flächen

- `runtime/ucf-compute/src/blue_brain_memory.rs`
- `runtime/ucf-compute/README.md`

## Kanonische Memory-Diagnostics-Surface

`CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP` ist die zentrale Klasse-/Lane-Liste für BB8 Prompt 3.

Enthaltene Diagnoseklassen:

- `commit_diagnostic`
- `committed_diagnostic`
- `committed_with_caveat_diagnostic`
- `rejected_commit_diagnostic`
- `blocked_commit_diagnostic`
- `failed_commit_diagnostic`
- `no_op_commit_diagnostic`
- `retrieval_diagnostic`
- `retrieved_diagnostic`
- `missing_memory_diagnostic`
- `stale_memory_diagnostic`
- `caveated_memory_diagnostic`
- `unavailable_memory_diagnostic`
- `non_canonical_internal_only_memory_diagnostic`

## Commit-Diagnostics: echte Store-Ergebnisse, keine erfundenen Outcomes

`BlueBrainMemoryCommitReport` trägt jetzt:

- `result_state`
- `diagnostic_class`
- `memory_record_id`
- `created_record`
- `diagnostic`
- `caveats`
- `feedback_backbind`

Bind-back-Regeln:

- `Committed` nur mit real erzeugter `memory_record_id`.
- `CommittedWithCaveat` konserviert Caveats in Record + Report.
- `Rejected` bei nicht commit-eligible oder fehlender Basis.
- `Blocked` bei stale/internal-only Guards.
- `Failed` bei echten Store-Fehlern.
- `NoOp` bei Duplikat (candidate bereits persisted).
- `Unavailable` wenn kanonischer Commit-Pfad nicht verfügbar ist.

## Retrieval-Diagnostics: echte Retrieval-Surface-Ergebnisse

`BlueBrainMemoryReadResult` trägt jetzt:

- `retrieval_state`
- `diagnostic_class`
- `reference`
- `diagnostic`
- Kontext-/Selection-Flags
- Auto-Trigger-Flags (bleiben `false`)
- `feedback_backbind`

Bind-back-Regeln:

- `RetrievedReferenceOnly` => `retrieved_diagnostic`
- `RetrievedWithCaveat` => `caveated_memory_diagnostic`
- `RetrievedStale` => `stale_memory_diagnostic`
- `Missing` => `missing_memory_diagnostic`
- `Blocked` (internal/non-canonical locator) => `non_canonical_internal_only_memory_diagnostic`
- `Unavailable` => `unavailable_memory_diagnostic`

## Runtime-/Context-/Selection-/Candidate-/Proposal-Feedback

`feedback_backbind` bindet Memory-Feedback in alle relevanten BB-Schichten zurück:

- Runtime: committed/retrieved/missing-stale-caveated/blocked/failed-unavailable + explizit no-auto-trigger.
- Context: committed/retrieved attach, caveat carry-over, stale/missing limits context update, keine automatische Candidate-Erzeugung.
- Selection/Candidate/Proposal: retrieved kann Candidate-Basis stützen, stale/missing schwächt Basis, committed kann future proposal basis stützen, caveated bleibt caveated, Retrieval selektiert/proposed/executed nichts automatisch.

## Harte Grenzen (weiterhin nicht Teil von BB8 Prompt 3)

Nicht implementiert in diesem Schritt:

- Monitoring-Plattformen
- Explainability-Engines
- Retrieval-Ranking / Semantic Search / Vector Search
- Knowledge Graph
- Memory Consolidation
- Reasoning-/Planning-/Action-Execution-Automatik
- automatische Compute-/Action-/Tool-Invocation aus Memory-Feedback

## Verwechslungs-Schutz

Die Memory-Surface bleibt klar getrennt von:

- History/Snapshot/Replay/Trace-Diagnostics
- Evidence-Retrieval
- Non-canonical/internal-only Expert-Hooks

Evidence-Referenzen in Memory-Records bleiben Referenzen; sie sind nicht selbst persisted-memory commits.
