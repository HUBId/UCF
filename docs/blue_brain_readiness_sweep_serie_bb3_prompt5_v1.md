# Serie BB3 Prompt 5: Readiness-Sweep und harte Context-/Memory-Grundlinie

Status: BB3 ist als repo-basierte Context-/Memory-Grundlinie abgeschlossen. Die Semantik für
Context, Evidence/Replay/Reference, Memory Candidates und Persistenzgrenzen ist jetzt als
technische Abschlusslinie konsolidiert, ohne neues Memory-Commit-System und ohne Compute-Core-Reopening.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_PERSISTENCE_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-runtime/src/orchestrator.rs`

Finale Compute-Referenzlinie (weiterhin verbindlich):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

Compute-Core-Posture bleibt unverändert:
- finale outward-facing Contracts,
- maintenance-only Core,
- keine neue Compute-Core-Arbeit in BB3 Prompt 5.

## 1) BB3-Kerncheck (hart, repo-basiert)

Real stabil und kanonisch:
- Context/Memory-Surface-Klassen (`transient_runtime_context`, `evidence_backed_context`,
  `replay_reference_backed_context`, `memory_adjacent_candidate`, `persisted_memory`,
  `non_canonical_internal_only_memory_like_path`).
- Context-Update-Lifecycle inklusive blocked/insufficient.
- Memory-Candidate-Lifecycle inklusive deferred/no-persistence.
- Evidence-/Replay-/Snapshot-/Trace-/Reference-Context-Klassen inklusive quality caveats.
- Persistenzgrenze zwischen Context, Candidate, Reference-Basis und actual persisted memory.

Nutzbar mit Caveats:
- evidence/replay/reference-backed Kontext ist load-bearing für Context-Update und
  Candidate-Backings, aber quality-gebunden (`partial|stale|caveated|insufficient`).

Vorbereitend / future-memory-ready:
- candidate future-memory-ready und Attachment-Lanes sind explizit proposal/handoff-only.

Non-canonical/internal-only:
- internal/expert/compat-Pfade bleiben nicht-kanonische Blue-Brain-Context-/Memory-Autorität
  und müssen vor Blue-Brain-facing Nutzung down-mapped werden.

Intentionally deferred:
- actual Blue-Brain persisted memory bleibt bewusst deferred.

## 2) Serie-BB3-Abschlussmatrix (repo-basiert, technisch)

| Bereich | BB3-Status | Technischer Befund |
| --- | --- | --- |
| Context/Memory Surface (`CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP`) | stable context/memory foundation | Transient Context, Evidence-/Replay-Context und Memory-Adjacent-Candidate sind klar getrennt; persisted-memory lane ist als aktuelle Null-Lane explizit codiert. |
| Context-Update-Lifecycle (`CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP`) | stable context/memory foundation | Update-only, update+candidate, candidate-only, blocked/insufficient sind als getrennte Zustände modelliert. |
| Memory-Candidate-Lifecycle (`CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`) | usable with caveats | Candidate-Klassen sind belastbar; accepted-for-future-handling bleibt non-commit, deferred/no-persistence bleibt explizit. |
| Evidence-/Replay-/Snapshot-/Trace-Reference-Kontext (`CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`) | usable with caveats | Referenzqualität bleibt sichtbar und steuert Context/Candidate-Caveats, ohne automatische Persistenzwirkung. |
| Future Attachment Lanes (`CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP`) | future-memory-ready / preparatory only | Anschlussstellen sind vorbereitet, aber proposal/handoff-only und ohne implizite Commit-Engine. |
| History/Snapshot/Trace/Evidence/Replay als Datenbasis | reference-only / not memory | Diese Surfaces bleiben Referenz-/Diagnostik-Basis und sind keine Blue-Brain-Memory-Persistenz. |
| Internal/Expert/Compat runtime paths (`run_operation_with_entry`, `replay_with_entry`, `build_backend(kind=stub\|candle\|worker)`, `domains/ai*`) | non-canonical / internal-only | Für kanonische Blue-Brain-Context-/Memory-Autorität ausgeschlossen; nur via outward down-mapping nutzbar. |
| Actual persisted memory in aktueller BB3-Baseline | intentionally deferred | Kein realer candidate->persisted-memory Commit-Pfad implementiert. |

## 3) Explizite Context-/Memory-Grundlinie ab BB3

Kanonische Context-Klassen:
- `transient_runtime_context`
- `evidence_backed_context`
- `replay_reference_backed_context`
- `memory_adjacent_candidate` (als Kontext-naher Kandidatenzustand, nicht als Persistenz)

Kanonische Context-Update-Zustände:
- context initialized,
- context updated from compute result,
- context updated from evidence/reference,
- context updated from replay/reference basis,
- context unchanged,
- context update blocked or insufficient.

Kanonische Memory-Candidate-Zustände:
- candidate proposed,
- candidate evidence-/replay-/reference-/trace-/snapshot-backed (qualitätsgebunden),
- candidate context-derived,
- candidate compute-result-derived,
- candidate accepted for future memory handling (non-commit),
- candidate rejected/stale/insufficient,
- persistence unavailable/deferred,
- persistence performed only if real path exists,
- no persistence performed.

Kanonische Evidence-/Replay-/Reference-Kontexte:
- evidence-backed context,
- replay-backed context,
- snapshot/reference-backed context,
- trace-backed context,
- caveated/insufficient reference context,
- non-canonical/internal-only reference path (explizit ausgeschlossen).

Persistenzgrenzen:
- Context Update ≠ Memory Commit.
- Candidate ≠ Persisted Memory.
- History/Snapshot/Replay/Trace/Evidence/Reference ≠ Memory Persistence.
- future-memory attachment lanes ≠ Commit-Engine.

Actual persisted memory:
- Im aktuellen BB3-Stand nicht vorhanden; bleibt bewusst deferred.

## 4) Final abgesicherte Boundaries

Verbindlich und testbar bleibt:
- Memory Candidates werden nicht automatisch persistiert.
- Evidence/Replay/Snapshot/Trace/History bleiben reference-only, solange kein realer
  Memory-Persistenzpfad implementiert ist.
- Future-Memory-Anschlussstellen kodieren nur Vorbereitung/Handoff und keinen impliziten Commit.
- Compute results können Context/Candidate beeinflussen, erzeugen aber nicht automatisch Memory.

## 5) Compute-Core-Abschlusslinie erneut bestätigt

BB3 Prompt 5 öffnet den Compute-Core nicht:
- keine neue Compute-Execution-Semantik,
- keine neue interne Commit-/Reasoning-/Audit-Engine im Compute-Kern,
- keine zweite Wahrheitsquelle neben der finalen Compute-Referenzlinie.

Es bleibt bei:
- finaler Compute-Linie,
- outward-facing Vertragskontinuität,
- maintenance-only Core.

## 6) Nächste Blue-Brain-Richtungen (1-3, repo-treu)

1. **Serie BB5: actual memory subsystem (auf den BB3-Attachment-Grenzen)**
   - Höchster Hebel, weil BB3 die candidate->persistence-Lücke exakt spezifiziert, aber bewusst offen lässt.
2. **Serie BB4: control/attention/selection layer über state/runtime/context**
   - Technisch sinnvoll nach/parallel zu klarer Memory-Commit-Basis, damit Auswahl nicht nur transient bleibt.
3. **Serie BB6: neural dynamics integration candidates (inkl. Hodgkin-Huxley/Kuramoto)**
   - Erst nachgelagert sinnvoll, wenn Control/Attention und/oder reale Memory-Persistenz belastbar stehen.

## 7) Priorisierte nächste Richtung

**Priorität 1: Serie BB5 (actual memory subsystem).**

Technische Begründung:
- BB3 hat die Semantik und Grenzen für Context/Candidate/Reference/Persistenz jetzt hart und testbar
  abgeschlossen; der größte offene technische Gap bleibt der reale Persistenzpfad.
- BB4 ist nachrangig, weil Control/Attention ohne tatsächliche Memory-Commit-Basis in wichtigen
  Fällen nur transient/caveated arbeiten kann.
- BB6 (Hodgkin-Huxley/Kuramoto) bleibt bewusst nicht zuerst, da neurodynamische Integration ohne
  robuste Commit-/Memory- oder mindestens stabilisierte Control-Basis architektonisch vorzeitig wäre.

## 8) Ergebnis

Die BB3-Context-/Memory-Grundlinie ist abgeschlossen:
- Context-, Candidate-, Reference- und Persistenzsemantik sind klar und repo-pinned getrennt,
- deferred actual persisted memory bleibt explizit und nicht verwischt,
- Future-Memory-Anschlussstellen sind vorbereitet, aber ohne implizite Commit-Semantik,
- Compute-Core bleibt geschlossen und maintenance-only,
- nächste priorisierte Richtung ist technisch klar: **BB5 zuerst**.
