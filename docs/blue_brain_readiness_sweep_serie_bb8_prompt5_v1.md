# Serie BB8 Prompt 5: Readiness Sweep und finale minimale persisted-memory Linie

Status: Serie BB8 ist mit Prompt 5 als **harte minimale persisted-memory Linie** abgeschlossen.

Diese Abschlussprüfung bleibt strikt repo-basiert auf bestehenden BB8-Flächen (Prompt 1-4) und zieht
keine neue Retrieval-/Ranking-/Consolidation-/Reasoning-/Action- oder Compute-Core-Arbeit ein.

## Repo-basierte Abschlussmatrix

| Bereich | Einstufung | Harte Einordnung |
|---|---|---|
| Persisted memory store (`BlueBrainMemoryStore`) | stable minimal persisted memory line | Reale kanonische Persistenz via append-only JSONL + geordnete In-Memory-Indizes (`BTreeMap`) für deterministische ID-/Candidate-Lookups. |
| Persisted memory record shape (`PersistedBlueBrainMemoryRecord`) | stable minimal persisted memory line | Kanonisches Record-Schema inkl. commit-, maintenance- und caveat-refresh-Feldern (`schema_version = 2`). |
| Commit operation + guards (`commit_candidate`) | stable minimal persisted memory line | Commit nur für `commit_eligible`, mit expliziten Zuständen: `committed`, `committed_with_caveat`, `rejected`, `blocked`, `failed`, `no_op`, `unavailable`. |
| Retrieval/reference surface (`read_reference`) | stable minimal persisted memory line | Minimaler Read-Pfad via `memory_record_id`/`source_candidate_id`; Zustände: `retrieved_reference_only`, `retrieved_with_caveat`, `retrieved_stale`, `retrieved_invalidated`, `missing`, `blocked`, `unavailable`. |
| Commit/Retrieval diagnostics + feedback backbind | stable minimal persisted memory line | Diagnostik-Klassen bleiben explizit; Feedback informiert Runtime/Context/Selection/Candidate/Proposal, aber ohne automatische Ausführung. |
| Maintenance/invalidation/caveat refresh (`apply_maintenance`) | stable minimal persisted memory line | Explizite Pflegezustände: `current`, `stale`, `caveated`, `caveat_refreshed`, `invalidated`, `maintenance_blocked`, `refresh_unavailable`, `non_canonical_internal_only_path`. |
| Runtime/Context/Selection/Candidate/Proposal Rückbindung | usable with caveats | Feedback kann Qualität/Basis degradieren/stützen, bleibt aber diagnostisch/handoff-orientiert und nicht-exekutierend. |
| History/Snapshot/Evidence/Replay/Trace Surfaces | reference-only / not memory | Diese Surfaces sind Basis-/Referenzquellen; sie sind **nicht** persisted Blue-Brain memory und kein Commit-Beweis. |
| Internal/expert-only locator/dependency paths | non-canonical / internal-only | Interne Pfade bleiben explizit blockbar und nicht-kanonisch, sofern nicht explizit down-mapped. |
| Ranking/Vector Search/Embedding/Knowledge Graph | intentionally deferred | Kein Bestandteil der BB8-Minimallinie. |
| Memory consolidation / reasoning / planning / action execution | intentionally deferred | Kein Auto-Trigger, keine Engine, keine Ausführung in BB8. |

## Echte minimale persisted-memory Linie (kanonisch)

### 1) Kanonischer Store
- `runtime/ucf-compute/src/blue_brain_memory.rs::BlueBrainMemoryStore`
- Persistenz: append-only JSONL Datei (`open`/`upsert`)
- Determinismus: geordnete Canonicalisierung + `BTreeMap`-Indizes

### 2) Kanonische Record-Shape
- `PersistedBlueBrainMemoryRecord` ist die kanonische persisted-memory Form.
- Referenzoberfläche ist eine abgeleitete Read-Shape (`BlueBrainMemoryReferenceRecord`) und bleibt
  retrieval/reference-spezifisch.

### 3) Kanonische Commit-Zustände
- `committed`
- `committed_with_caveat`
- `rejected`
- `blocked`
- `failed`
- `no_op`
- `unavailable`

### 4) Kanonische Retrieval-Zustände
- `retrieved_reference_only`
- `retrieved_with_caveat`
- `retrieved_stale`
- `retrieved_invalidated`
- `missing`
- `blocked`
- `unavailable`

### 5) Kanonische Diagnostics-/Maintenance-Zustände
- Diagnostics über `CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP` und `BlueBrainMemoryDiagnosticClass`.
- Maintenance-Status über `BlueBrainMemoryMaintenanceStatus`.
- Maintenance-Resultate über `BlueBrainMemoryMaintenanceResultState`.
- Caveat-Refresh-Haltung über `BlueBrainMemoryCaveatRefreshState`.

### 6) Explizit ausgeschlossene Pfade
- Internal-/expert-only locator/dependency Pfade bleiben non-canonical.
- History/Snapshot/Evidence/Replay/Trace bleiben Referenzbasis, nicht Memory-Identität.
- Kein semantisches Retrieval, kein Ranking, keine Vectorsuche.
- Kein Consolidation-, Reasoning-, Planning- oder Action-Execution-Automatismus.

## Harte Grenzen: Memory vs. History/Snapshot/Evidence/Replay

Finale Abgrenzung in BB8:
- History ist keine Memory-Persistenz.
- Snapshot ist keine Memory-Persistenz.
- Evidence ist keine Memory-Persistenz.
- Replay/Trace ist keine Memory-Persistenz.
- Persisted Memory Records **dürfen** Evidence-/Reference-Basis enthalten, bleiben aber eine
  eigenständige persistierte Memory-Entität.

## Harte Grenzen: kein Retrieval-Ranking-/Vector-/Consolidation-/Reasoning-Claim

Die BB8-Minimallinie behauptet explizit **nicht**:
- semantische Suche,
- Vector Search / Embedding Retrieval,
- Ranking,
- Knowledge Graph,
- Memory Consolidation,
- Reasoning/Planning-Engine,
- Action-/Tool-Execution.

## Feedback-Grenzen (final abgesichert)

Commit-, Retrieval- und Maintenance-Feedback kann:
- Runtime informieren,
- Context-Updates caveaten/degraden,
- Selection/Candidate/Proposal-Basis stützen oder schwächen.

Commit-, Retrieval- und Maintenance-Feedback kann **nicht automatisch**:
- Compute invoken,
- Action/Tool ausführen,
- Reasoning/Planning ausführen,
- weiteren Memory-Commit auslösen.

## Compute-Core-Abschlusslinie

BB8 öffnet den Compute-Core nicht neu.

Compute bleibt:
- auf der finalen Exit-Linie,
- mit outward-facing Contracts,
- maintenance-only im Kern.

## Nächste Blue-Brain-Richtung (1 priorisierte Richtung)

Priorität #1: **Serie BB9 — Minimal action execution boundary / tool-safety prelayer**.

Technische Begründung:
- BB8 liefert jetzt eine echte minimale persisted-memory Grundlage.
- BB7 hat future-action-ready/Handoff bereits vorbereitet.
- Der höchste unmittelbare Hebel liegt daher in einer minimalen, sicheren,
  weiterhin strikt kontrollierten Ausführungsgrenze statt in Retrieval-Ausbau.

Nachrangig:
- Retrieval-/Consolidation-Ausbau (BB10) ist sinnvoll, aber nicht der engste nächste Integrationshebel.
- Hodgkin-Huxley/Kuramoto (BB11) bleiben weiterhin nachrangig; erst nach Action-/Retrieval-Erweiterung
  belastbar sinnvoll.
