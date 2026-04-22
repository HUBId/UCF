# Serie BB5 Prompt 3: Memory-commit diagnostics / rejection / caveat feedback (repo-basiert)

Status: BB5 Prompt 3 bindet Memory-commit diagnostics in Candidate Lifecycle, Selection/Deferral und Runtime/Context zurück. Der **Compute-Kern bleibt maintenance-only** und es wird **keine Monitoring- oder Explainability-Plattform** gebaut.

Diese Konsolidierung nutzt die bestehende finalisierte Referenzlinie:

`submit -> compute_canonical -> result/fault/status -> execution_snapshot`

und bleibt auf den vorhandenen Maps in `runtime/ucf-compute/src/reference_map.rs`.

## 1) Kanonische Commit-/Handoff-Diagnostics map

Für BB5 Prompt 3 ist `CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP` die kanonische Fläche.
Sie trennt exakt:

- handoff diagnostic,
- commit eligibility diagnostic,
- commit rejected diagnostic,
- commit blocked diagnostic,
- commit deferred diagnostic,
- commit caveated diagnostic,
- commit unavailable diagnostic,
- committed-if-present diagnostic,
- no-persistence diagnostic,
- non-canonical/internal-only diagnostic.

Diese Map ist bewusst kompakt und deterministisch; sie ist keine Audit-, Monitoring-, Reasoning- oder Memory-Engine.

## 2) Kompakte kanonische Gründe

Die Diagnostics tragen absichtlich kurze, kanonische Gründe statt freier Begründungsprosa.
Mindestens unterschieden werden:

- rejected due to weak or insufficient evidence,
- rejected due to candidate state,
- blocked due to stale context,
- blocked due to missing persistence path,
- blocked due to internal-only dependency,
- unavailable because no actual memory subsystem exists,
- caveated due to partial reference basis.

## 3) Bind-back in Candidate Lifecycle

Commit-/Handoff-Feedback wird direkt auf Candidate-Lifecycle-Semantik zurückgeführt:

- candidate remains future-memory-ready,
- candidate deferred after handoff,
- candidate rejected after handoff,
- candidate blocked from commit,
- candidate committed only if real path exists,
- no persistence performed.

Damit bleibt klar: Candidate-Zustand, Handoff-Zustand und Commit-Result sind getrennt, aber rückführbar.

## 4) Bind-back in Selection / Priority / Deferral

Commit-Feedback informiert BB4 Selection/Deferral technisch, ohne neue Policy-/Ranking-Plattform:

- selected candidate commit unavailable,
- deferred candidate remains deferred,
- rejected candidate removed from future consideration (für die aktuelle Candidate-Instanz),
- caveated candidate remains recheckable,
- insufficient candidate cannot become trigger/commit basis.

## 5) Bind-back in Runtime / Context Diagnostics

Runtime/Context können nun explizit sehen:

- memory handoff prepared,
- memory commit unavailable,
- commit rejected/blocked/caveated,
- committed-if-present,
- context updated but not persisted,
- evidence attached but not committed.

Auch hier gilt: runtime diagnostics bleiben technische Zustandsdiagnostik, keine Monitoring-Plattform.

## 6) Schutz gegen Reference-/History-Verwechslungen

BB5 Prompt 3 hält die Nicht-Gleichsetzungen weiterhin hart:

- History ≠ Memory Commit,
- Snapshot ≠ Memory Commit,
- Evidence Reference ≠ Memory Commit,
- Replay/Trace Reference ≠ Memory Commit.

## 7) Non-canonical Diagnostics bleiben ausgeschlossen

Internal-/Expert-/Legacy-/Compat-Flächen bleiben nicht-kanonisch für BB5 Commit-Diagnostics.
Wenn solche Informationen nötig werden, müssen sie zuerst auf outward candidate/evidence/selection/runtime references down-gemappt werden.

## 8) Architekturgrenzen

Diese Serie baut weiterhin **keine**:

- Monitoring- oder Explainability-Plattform,
- globale Memory-Consolidation,
- Reasoning- oder Audit-Engine,
- neue Memory-/Vector-DB-/Knowledge-Graph-Subsysteme,
- neurodynamische Spezialmodelle (Hodgkin-Huxley/Kuramoto bleiben out-of-scope).

