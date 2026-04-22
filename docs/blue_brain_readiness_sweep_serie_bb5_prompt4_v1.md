# Serie BB5 Prompt 4: Readiness-Sweep und minimale Memory-Commit-Linie (repo-basiert)

Status: BB5 ist mit Prompt 4 als **minimale Memory-Commit-Grenze** technisch abgeschlossen.
Die kanonische Linie bleibt bewusst minimal: Candidate-to-Commit/Handoff/Diagnostics sind eindeutig,
**actual memory commit ist im aktuellen Repo weiterhin nicht implementiert**, und der Compute-Kern bleibt
auf finaler Exit-Linie maintenance-only.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP`
  - `CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_HANDOFF_STATE_MAP`
  - `CANONICAL_BLUE_BRAIN_COMMIT_RESULT_SEMANTICS_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_PERSISTENCE_BOUNDARY_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-runtime/src/orchestrator.rs`

Finale Referenzlinie (unverändert):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) BB5-Abschlussmatrix (hart, repo-basiert)

| Bereich | Abschlussklasse | Repo-basierte Feststellung |
|---|---|---|
| Minimal Memory Commit Boundary + Eligibility Conditions | stable minimal memory-commit boundary | Candidate-Klassen und Eligibility-Gates sind explizit: proposed/deferred/rejected/stale/insufficient/commit-eligible/future-memory-ready/reference-only/internal-only + reale Path-Gate-Bedingung. |
| Future-Memory-Handoff + Commit-Result-Semantik | stable minimal memory-commit boundary | Handoff-Zustände und Commit-Result-Klassen sind getrennt modelliert; handoff-ready, commit-eligible und committed bleiben strikt nicht gleichgesetzt. |
| Commit-/Handoff-Diagnostics | stable minimal memory-commit boundary | Diagnostics binden kompakt auf candidate/selection/runtime zurück und halten rejected/blocked/deferred/unavailable/caveated/no-persistence getrennt. |
| Commit-usable with caveats | intentionally conditional | Nur gültig, **falls** ein realer kanonischer Persistenzpfad implementiert wird; dann sind committed/committed-with-caveats/failed/no-op semantisch vorbereitet. |
| Future-memory-ready Attachment/Handoff | future-memory-ready / preparatory only | Handoff/attachment ist explizit non-commit proposal path bis eine reale Persistenz-Implementierung existiert. |
| History/Snapshot/Evidence/Replay/Trace | reference-only / not memory | Diese Surfaces dürfen Candidate-Qualität stützen, bleiben aber explizit außerhalb von Memory-Commit-Beweis oder Persistenzautorität. |
| Internal/Expert/Legacy/Compat persistence-like hooks | non-canonical / internal-only | Können weder Commit-Eligibility noch Handoff/Commit-Autorität bilden, solange kein outward canonical remap vorliegt. |
| Actual Blue-Brain memory commit path | intentionally deferred | Aktuell keine kanonische reale Persistenzroute; `commit unavailable`/`commit deferred` bleiben baseline-konform. |

## 2) Explizite minimale Memory-Commit-Linie (kanonisch)

### 2.1 Candidate-to-Commit-Klassen
Kanonische Klassen (Boundary):
- `not a memory candidate`
- `memory candidate proposed`
- `memory candidate deferred`
- `memory candidate rejected`
- `memory candidate stale`
- `memory candidate insufficient`
- `commit-eligible candidate`
- `future-memory-ready candidate`
- `committed memory (only if real path exists)`
- `reference-only / not memory`
- `non-canonical/internal-only persistence path`

Harte Regel:
- `commit-eligible` ist ein Gate-Status, **kein** Commit-Resultat.
- `future-memory-ready` ist ein Handoff-Status, **kein** Persistenznachweis.

### 2.2 Commit-Eligibility-Bedingungen
Commit-Eligibility bleibt an diese kanonischen Bedingungen gebunden:
- ausreichende Evidence/Reference-Qualität,
- Selection/Attention-Gate erfüllt,
- nicht-stale Context/Freshness,
- keine blockierende Caveat/Fault-Lage,
- keine internal/expert-only Abhängigkeit,
- realer Persistenzpfad erforderlich für actual commit,
- ohne realen Pfad: expliziter `future-memory-ready` Handoff statt Commit.

### 2.3 Kanonische Future-Memory-Handoff-Zustände
- `handoff ready`
- `handoff deferred`
- `handoff blocked`
- `handoff rejected`
- `handoff caveated`
- `handoff unavailable`
- `handoff internal-only/non-canonical`

Harte Regel:
- `handoff ready` bedeutet nur intake-readiness, nicht committed memory.

### 2.4 Kanonische Commit-Result-Zustände
- `commit unavailable`
- `commit deferred`
- `committed`
- `committed with caveats`
- `commit rejected`
- `commit blocked`
- `commit failed`
- `commit no-op`
- `reference recorded only`

Harte Regel:
- Im aktuellen Baseline-Repo sind `commit unavailable`/`commit deferred` + `reference recorded only`
  die ehrliche Standardlinie.
- `committed`/`committed with caveats`/`failed` sind nur legitim mit realem Persistenzpfad.

### 2.5 Kanonische Commit-/Handoff-Diagnostics
- `handoff diagnostic`
- `commit eligibility diagnostic`
- `commit rejected diagnostic`
- `commit blocked diagnostic`
- `commit deferred diagnostic`
- `commit caveated diagnostic`
- `commit unavailable diagnostic`
- `committed-if-present diagnostic`
- `no-persistence diagnostic`
- `non-canonical/internal-only diagnostic`

Kompakte Gründe bleiben kanonisch (u. a.):
- weak/insufficient evidence oder candidate state,
- stale context / missing persistence path / internal-only dependency,
- no actual memory subsystem exists,
- partial reference basis.

## 3) Final abgesicherte Nicht-Gleichsetzungen

Diese Grenzen sind Abschlussbedingungen von BB5:
- `commit-eligible` ≠ `committed`
- `future-memory-ready` ≠ `persisted`
- `handoff-ready` ≠ `committed`
- `reference recorded only` ≠ `memory commit`
- `History/Snapshot/Evidence/Replay/Trace` ≠ `memory commit proof`
- `context updated but not persisted` bleibt explizit no-persistence

## 4) Selection/Evidence/Context/Candidate als Commit-Basis

Commit-Grenzen bleiben strikt auf vorhandenen BB2/BB3/BB4-Semantiken:
- Selection/Attention-Status gate’t Commit-Eligibility,
- Evidence/Reference-Qualität bleibt verpflichtende Basis,
- Context Freshness (nicht stale) bleibt verpflichtend,
- Candidate Lifecycle State bestimmt progression/hold/terminal,
- blocking caveats/faults verhindern Commit/Handoff-Fortschritt.

Damit bleibt BB5 technisch eng: kein Ranking-/Policy-/Reasoning-/Governance-Shift.

## 5) Compute-Core-Abschlusslinie (erneut gesichert)

BB5 öffnet keine neue Compute-Core-Arbeit:
- Compute-Kern bleibt maintenance-only.
- BB5 bleibt auf outward-facing candidate/handoff/diagnostics semantics.
- Keine Re-Öffnung von Core Execution-/Backend-/Runtime-Kernentwicklung.

## 6) Nächste Blue-Brain-Richtungen (1-3, repo-treu)

1. **Serie BB7: Actual Memory Subsystem Minimal Implementation (auf BB5-Minimallinie)**
   - Höchster Hebel: schließt die aktuell explizite Lücke zwischen `commit-eligible/future-memory-ready` und realem `committed` Pfad.
2. **Serie BB6: Planning/Reasoning Candidate Layer**
   - Sinnvoll nachrangig; profitiert von klarer tatsächlicher Memory-Commit-Rückkopplung statt nur no-persistence Semantik.
3. **Serie BB8: Neural dynamics integration candidates (inkl. Hodgkin-Huxley/Kuramoto)**
   - Bleibt nachrangig bis minimaler realer Memory-Persistenzpfad oder belastbarer Planning-Layer vorhanden ist.

### Priorisiert als nächster erster Schritt
**Priorität: Serie BB7 zuerst.**

Technische Begründung (knapp):
- BB5 hat die Grenze jetzt eindeutig gemacht, aber die reale Commit-Lücke bleibt bewusst offen.
- BB7 liefert den direkten Anschluss an die bereits kanonisch vorhandenen Candidate-/Eligibility-/Handoff-/Diagnostics-Verträge.
- BB6 ist nachrangig, solange persistente Commit-Rückkopplung fehlt.
- Hodgkin-Huxley/Kuramoto bleiben weiterhin nicht zuerst, weil ohne reale minimale Memory-Persistenz
  ihre Integration semantisch vorgezogen wäre.

## 7) Gezielte Konsistenz-Checkliste (BB5-Abschluss)

Die Abschlusslinie bleibt nur gültig, solange folgende Bedingungen erfüllt bleiben:
- `proposed/deferred/rejected/stale/insufficient/commit-eligible/future-memory-ready/committed-if-present/reference-only/no-persistence` bleiben getrennt.
- `commit-eligible`/`future-memory-ready`/`handoff-ready` erzeugen keinen Auto-Commit.
- Ohne realen Persistenzpfad wird `commit unavailable` (oder explizit `commit deferred`) diagnostisch sichtbar gehalten.
- History/Snapshot/Evidence/Replay/Trace bleiben reference-only, solange keine reale Memory-Persistenz implementiert ist.
- Commit-Diagnostics bleiben auf kanonischen candidate/selection/evidence/runtime Pfaden.
- BB5-Doku bleibt konsistent zu BB3/BB4 und zur Compute-Exit-/Maintenance-Linie.
- Internal/expert-only Pfade erscheinen nicht als kanonische Commit-/Handoff-Surface.
