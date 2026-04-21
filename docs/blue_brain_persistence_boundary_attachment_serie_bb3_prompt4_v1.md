# Serie BB3 Prompt 4: Persistenzgrenzen und spätere Memory-Subsystem-Anschlussstellen (repo-basiert, ohne Memory-Commit-Engine)

Status: BB3 Prompt 4 fixiert die Persistenzgrenzen zwischen Blue-Brain-Kontext, Evidence-/Replay-/Reference-Kontext, Memory Candidates und tatsächlicher Memory-Persistenz. Die Linie bleibt strikt repo-basiert und führt **kein** neues Memory-Subsystem ein.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_PERSISTENCE_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-runtime/src/orchestrator.rs`

Finale Referenzlinie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Repo-basierter Ist-Befund: echte Persistenz vs Context/Reference/Candidate

BB3 Prompt 4 trennt explizit:
- transient runtime context,
- evidence/reference-backed context,
- memory-adjacent candidate,
- future-memory-ready candidate,
- actual persisted memory,
- history/snapshot/reference but not memory,
- non-canonical/internal-only persistence-like path.

Wesentlicher Befund aus der aktuellen Baseline:
- BB3 implements no actual Blue-Brain memory persistence.
- History-/Snapshot-/Replay-/Evidence-Flächen existieren, sind aber Referenz-/Diagnostik-Basis und **kein** Memory-Store.
- Candidate-Lifecycle bleibt explizit und ohne impliziten Commit.

## 2) Persistence Boundary Map (kanonisch)

`CANONICAL_BLUE_BRAIN_PERSISTENCE_BOUNDARY_MAP` führt die kanonischen Klassen als belastbare Schnittgrenze:
- transient runtime context,
- evidence/reference-backed context,
- memory-adjacent candidate,
- future-memory-ready candidate,
- actual persisted memory,
- history/snapshot/reference but not memory,
- non-canonical/internal-only persistence-like path.

Damit sind diese Trennungen testbar und code-pinned:
- Context ≠ Candidate,
- Candidate ≠ Persisted Memory,
- Evidence/Replay/History/Snapshot/Reference ≠ Memory Persistence,
- Internal-/Expert-Pfade ≠ kanonische Blue-Brain-Memory-Autorität.

## 3) Actual persisted memory strikt repo-basiert

Die Lane `blue_brain_persistence_boundary_actual_persisted_memory_deferred` hält fest:
- actual persisted memory für Blue-Brain context/candidate ist bewusst deferred,
- es gibt in der aktuellen Baseline keinen realen Blue-Brain-Memory-Commit-Pfad,
- deshalb bleibt candidate->persisted-memory Übergang gesperrt.

Wichtig:
- Persistente Operativ-/History-/Snapshot-Daten im Repo sind nicht als Blue-Brain-Memory zu interpretieren.
- Keine Storage-/Commit-Engine wird hier eingeführt.

## 4) Future Memory Subsystem Attachment Points (nur Anschlussstellen)

`CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP` macht die späteren Anschlüsse sichtbar, ohne zu implementieren:
- candidate proposed,
- candidate future-memory-ready,
- candidate rejected,
- candidate stale/insufficient,
- persistence unavailable/deferred,
- commit only if real explicit path exists,
- history/snapshot/reference basis only (not commit authority).

Required fields für künftige Handoffs bleiben minimal und explizit:
- candidate_id + candidate digest,
- context digest,
- evidence/reference/replay basis inkl. quality posture,
- caveats (partial/stale/caveated/insufficient),
- explizite deferred/commit-boundary markers.

## 5) Candidate-to-Persistence Boundary sauber markiert

Die Candidate-Lifecycle-Semantik enthält jetzt zusätzlich:
- persistence unavailable/deferred,
- persistence performed only if real path exists,
- no persistence performed.

Das sichert:
- candidate proposed bleibt proposal,
- candidate future-memory-ready bleibt non-commit,
- candidate rejected und candidate stale/insufficient bleiben explizit,
- persistence unavailable/deferred bleibt sichtbar,
- commit nur bei realem explizitem Pfad (der aktuell nicht existiert).

## 6) History/Snapshot/Evidence/Replay sauber von Memory getrennt

Prompt 4 bindet History/Snapshot/Evidence/Replay als **reference basis** zurück:
- reference basis,
- evidence basis,
- replay/history basis,
- not memory persistence,
- usable for future memory candidate only with caveats.

Damit wird vermieden, dass diese Flächen als Memory-Ersatz bezeichnet werden.

## 7) Compute-Core-Grenze bleibt geschlossen

Compute-Core bleibt maintenance-only.

Prompt 4 stellt sicher:
- keine neuen compute-internen Hooks für Memory-Commit,
- internal/expert persistence-like paths bleiben non-canonical,
- Future-Attachment nur über outward-facing status/evidence/reference Kontinuität.

## 8) Ergebnis

BB3 Prompt 4 liefert eine belastbare Persistenzgrenze:
- klare Klassen für Context/Reference/Candidate/Persistenz,
- explizite Deferred- und No-Commit-Semantik,
- klare Anschlussstellen für spätere Memory-Subsysteme,
- keine implizite Persistenz,
- keine neue Memory-/Vector-DB-/Knowledge-Graph-/Consolidation-Plattform.

Compute-Core bleibt maintenance-only und ungeöffnet.
