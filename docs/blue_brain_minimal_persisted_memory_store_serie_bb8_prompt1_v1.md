# Serie BB8 Prompt 1: Minimal persisted memory store für commit-eligible Blue-Brain candidates

Status: BB8 Prompt 1 führt eine **echte minimale persisted-memory Lane** ein, die commit-eligible Blue-Brain-Candidates als eigenständige Memory-Records persistiert. Die Lane bleibt bewusst klein, deterministisch und klar getrennt von History/Snapshot/Evidence/Replay.

## 1) Minimaler Store (kanonisch)

Kanonischer Einstieg:
- `runtime/ucf-compute/src/blue_brain_memory.rs`
- `BlueBrainMemoryStore::open(path)`
- `BlueBrainMemoryStore::commit_candidate(candidate, committed_at_unix_ms)`
- `BlueBrainMemoryStore::get(memory_record_id)`
- `BlueBrainMemoryStore::get_by_candidate(candidate_id)`

Persistenzform:
- repo-native JSONL append-store,
- in-memory index via geordnete `BTreeMap`-Indizes,
- deterministische Canonicalisierung von Listenfeldern vor Commit.

## 2) Record-Shape (minimal, aber commit-fähig)

`PersistedBlueBrainMemoryRecord` enthält nur die minimalen Memory-Metadaten:
- `memory_record_id`
- `source_candidate_id`
- `origins` (`context|evidence|replay|reference|compute_result|selection|commit_feedback`)
- `evidence_refs`
- `reference_refs`
- `context_basis_refs`
- `selection_basis_refs`
- `freshness`
- `caveats`
- `committed_at_unix_ms`
- `commit_result_state`

Nicht enthalten:
- keine Compute-internen Rohdetails,
- keine Vector-/Embedding-/Ranking-Felder,
- keine Consolidation- oder Knowledge-Graph-Strukturen.

## 3) Commit-Result-Semantik (BB5 realisiert)

`BlueBrainMemoryCommitResultState` unterscheidet:
- `committed`
- `committed_with_caveat`
- `rejected`
- `blocked`
- `failed`
- `no_op`
- `unavailable`

`BlueBrainMemoryCommitReport` macht zur Laufzeit sichtbar:
- Candidate-ID,
- Result-State,
- erzeugte `memory_record_id` (oder `None`),
- `created_record` (`true/false`),
- kompakte Diagnose,
- erhaltene Caveats.

## 4) Harte Commit-Guards

Commit wird nur bei `commit_eligible` zugelassen.

Explizite Guards:
- non-eligible Klassen (`deferred|rejected|blocked|insufficient|reference_only`) werden nicht persisted,
- fehlende Basis (`evidence/reference/context`) wird ohne caveated allowance rejected,
- stale context ohne allowance wird blocked,
- internal/expert-only dependency wird blocked,
- unavailable store-path wird als `unavailable` gemeldet,
- doppelter Commit derselben Candidate-ID wird als `no_op` mit bestehender Record-ID gemeldet.

## 5) Trennung zu History/Snapshot/Evidence/Replay

Diese BB8-Lane ist eine **eigene Persistenzsemantik** für Memory-Commit. Sie ersetzt nicht:
- Job-History (`PersistedJobRecord`),
- Snapshot-Readiness-Flächen,
- Evidence-Bundle-Exports,
- Replay-Records oder Trace-Slices.

History/Snapshot/Evidence/Replay können weiter Referenzbasis liefern, sind aber selbst keine Memory-Records.

## 6) Weiterhin bewusst nicht implementiert

- kein Retrieval-Ranking,
- keine semantische Suche,
- keine Vector-DB,
- kein Knowledge Graph,
- keine Consolidation-Engine,
- keine neurodynamische Modellintegration.

Damit bleibt BB8 Prompt 1 ein minimaler realer Persistenzanker für spätere BB8-Folgeschritte.
