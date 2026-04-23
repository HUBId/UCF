# Serie BB8 Prompt 2: Minimal memory retrieval/reference surface ohne Ranking oder Vector Search

Status: BB8 Prompt 2 ergänzt die in Prompt 1 eingeführte persisted-memory Lane um eine **minimale Retrieval-/Reference-Surface**. Diese Surface liest persisted memory records stabil per ID oder source-candidate Referenz und führt sie als reine Reference-Basis in Context/Selection zurück.

## 1) Kanonische Retrieval-Einstiege

Kanonische Runtime-Fläche:
- `runtime/ucf-compute/src/blue_brain_memory.rs`
- `BlueBrainMemoryStore::read_reference(BlueBrainMemoryReadRequest)`
- `BlueBrainMemoryReferenceLocator::{MemoryRecordId, SourceCandidateId}`

Read-Pfad bleibt bewusst klein:
- kein query-basiertes Suchen,
- kein nearest-neighbor,
- kein Ranking,
- keine Embedding-/Vector-Layer.

## 2) Minimales Reference-Resultat

`BlueBrainMemoryReadResult` liefert nur Retrieval-/Reference-Semantik:
- `retrieval_state`
- `reference` (`BlueBrainMemoryReferenceRecord`) falls vorhanden
- kompakte `diagnostic`
- Context-Rückbindung (`context_attached`, `context_caveated`, `context_stale`, `context_insufficient_for_candidate_or_proposal`)
- Selection-Rückbindung (`selection_disposition`)
- harte No-Auto-Trigger-Flags:
  - `automatic_compute_triggered = false`
  - `automatic_action_or_planning_triggered = false`
  - `automatic_memory_commit_triggered = false`

## 3) Retrieval Result States (minimal, explizit)

`BlueBrainMemoryRetrievalState` unterscheidet:
- `retrieved_reference_only`
- `retrieved_with_caveat`
- `retrieved_stale`
- `missing`
- `blocked`
- `unavailable`

Diese States sind reine Read-/Reference-Zustände und **keine** Reasoning- oder Ranking-Entscheidung.

## 4) Context-Rückführung ohne Auto-Compute

Beim erfolgreichen Read wird der Memory-Record als Context-Reference-Basis markiert:
- memory reference observed,
- memory reference attached to current context,
- caveated/stale/insufficient sichtbar,
- aber kein automatischer Compute-/Planning-/Action-Trigger.

## 5) Selection/Priority-Einordnung ohne Ranking

`BlueBrainMemorySelectionDisposition` hält eine nicht-numerische, minimale Auswahlsemantik:
- `selected`
- `supporting`
- `deferred`
- `ignored`
- `insufficient`
- `caveated`

In Prompt 2 wird Read-Semantik primär als `supporting`, `deferred`, `insufficient` oder `caveated` zurückgeführt; keine Scoring-/Ranking-Plattform.

## 6) Trennung zu History/Snapshot/Evidence/Replay bleibt strikt

Memory retrieval bleibt getrennt von:
- Job-History retrieval,
- Snapshot/Replay retrieval,
- Evidence retrieval.

Evidence/Replay/Snapshot können Referenzbasis bleiben, werden aber nicht als persisted-memory retrieval umetikettiert.

## 7) Weiterhin nicht implementiert

- keine semantic search,
- keine vector search,
- keine embeddings,
- kein knowledge graph,
- keine memory consolidation,
- keine reasoning/planning engine aus Retrieval,
- keine neurodynamischen Spezialmodelle (z. B. Hodgkin-Huxley/Kuramoto).
