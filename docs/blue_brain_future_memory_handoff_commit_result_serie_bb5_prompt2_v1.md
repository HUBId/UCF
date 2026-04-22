# Serie BB5 Prompt 2: Future-Memory-Handoff-Interface und Commit-Result-Semantik (repo-basiert)

Status: BB5 Prompt 2 konsolidiert die kanonische Future-Memory-Handoff-Schnittstelle und die Commit-Result-Semantik auf Basis von BB3/BB4/BB5 Prompt 1. Der **Compute-Kern bleibt maintenance-only** (`runtime/ucf-compute` auf der finalen technischen Exit-Linie), und es wird **keine Memory-Engine** gebaut.

Finale Ausführungsreferenz bleibt unverändert:
`submit -> compute_canonical -> result/fault/status -> execution_snapshot`.

## 1) Scope und Abgrenzung

Diese Prompt-2-Konsolidierung zieht nur die Interface-/Semantik-Linie:
- kanonische Handoff-Felder,
- kanonische Handoff-Zustände,
- kanonische Commit-Result-Zustände,
- Rückbindung in Runtime/Diagnostics,
- explizite Trennung von Reference/History/Snapshot/Evidence vs. Memory-Commit.

Explizit **nicht** Teil dieses Schritts:
- neue Persistenzengine,
- Vector-DB/Knowledge-Graph,
- globale Memory-Consolidation,
- neue Ranking-/Policy-/Reasoning-Engine,
- Hodgkin-Huxley/Kuramoto.

## 2) Code-pinned Prompt-2 Maps

Prompt 2 ist an folgende Code-Konstanten gebunden:
- `CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP`
- `CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP`

Sie ergänzen BB5 Prompt 1:
- `CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP`
- `CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP`
- `CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP`

## 3) Kanonische Future-Memory-Handoff-Felder

Das Handoff-Interface muss mindestens enthalten (oder eindeutig referenzieren):
- candidate identity,
- candidate origin (`context/evidence/replay/reference/compute-result/selection`),
- evidence/reference basis,
- selection/attention status,
- caveats,
- freshness/staleness,
- commit eligibility state,
- no persistence implied unless actual commit path exists.

Damit bleibt klar: handoff-ready ist nicht gleich committed memory.
Also explicit: handoff-ready is not a memory commit.

## 4) Kanonische Handoff-Zustände

`CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP` unterscheidet explizit:
- handoff ready,
- handoff deferred,
- handoff blocked,
- handoff rejected,
- handoff caveated,
- handoff unavailable,
- handoff internal-only/non-canonical.

Die Zustände bleiben deterministisch getrennt: deferred ≠ blocked ≠ rejected ≠ unavailable.

## 5) Kanonische Commit-Result-Semantik

`CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP` unterscheidet explizit:
- commit unavailable,
- commit deferred,
- committed,
- committed with caveats,
- commit rejected,
- commit blocked,
- commit failed,
- commit no-op,
- reference recorded only.

Zusätzlich gilt unverändert:
- **no actual memory commit is implemented in the current baseline**.
- Ohne realen Persistenzpfad sind `commit unavailable` / `commit deferred` + future-memory handoff die kanonischen Ergebnisse.
- `committed` und `committed with caveats` bleiben reservierte Klassen für den Fall eines später real implementierten Persistenzpfads.

## 6) Trennung von Commit-Result vs. Reference/History/Snapshot/Evidence

Prompt 2 fixiert die Trennung explizit:
- `reference recorded only` ist kein Commit,
- `evidence observed` ist kein Commit,
- `handoff prepared` ist kein Commit,
- `commit unavailable` bedeutet weiterhin no persisted memory.

Merksatz: **History/Snapshot/Evidence/Replay are reference-only**.

## 7) Runtime/Diagnostics Bind-back

Die Commit-/Handoff-Semantik bleibt in Runtime-Diagnostics sichtbar:
- candidate handoff-ready,
- handoff deferred/blocked/rejected/caveated/unavailable,
- commit unavailable/deferred,
- commit successful only if real path exists,
- caveats preserved (kein stilles Dropping).

Diese Rückbindung bleibt Diagnose-Semantik; sie ist keine Monitoring-/Explainability-Plattform.

## 8) Non-canonical Pfade

Internal-/Expert-/Legacy-/Compat-Pfade bleiben nicht-kanonisch für BB5 Future-Memory-Handoff:
- compute-interne hooks,
- expert/internal workflows,
- legacy/compat persistence-like traces.

Sie müssen auf outward candidate/evidence/selection references gemappt werden, bevor sie jemals kanonische Handoff-Autorität bekommen könnten.

## 9) Abschluss

BB5 Prompt 2 schafft eine klare, ehrliche Interface- und Ergebnis-Semantik:
- handoff-ready ist explizit non-commit,
- commit unavailable/deferred ist explizit,
- committed ist nur mit realem Persistenzpfad gültig,
- Evidence/Selection/Caveats bleiben durchgehend erhalten,
- der abgeschlossene Compute-Kern wird nicht wieder geöffnet.

Hodgkin-Huxley/Kuramoto bleiben außerhalb dieses Schritts.
