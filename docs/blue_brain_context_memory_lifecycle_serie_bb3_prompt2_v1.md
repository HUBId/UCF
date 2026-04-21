# Serie BB3 Prompt 2: Context-Update- und Memory-Candidate-Lifecycle (repo-basiert, ohne Memory-Commit-Engine)

Status: BB3 Prompt 2 konsolidiert die Lifecycle-Semantik für Blue-Brain-Context-Updates und
memory-nahe Kandidaten auf der bestehenden BB2/BB3-Prompt-1-Basis. Es wird **kein**
persistentes Memory-System eingeführt.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP`
- `runtime/ucf-runtime/src/orchestrator.rs`
- `runtime/ucf-compute/src/service_surface.rs`

Finale Referenzlinie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Repo-basierter Ist-Befund (Update vs Candidate vs Persistenz)

BB2/BB3 Prompt 1 hat bereits getrennt:
- transient runtime context,
- evidence-backed context,
- replay/reference-backed context,
- memory-adjacent candidate,
- persisted-memory null lane.

Prompt 2 schärft nun die Lifecycle-Sicht:
- context initialized,
- context updated from compute result,
- context updated from evidence/reference,
- context updated from replay/reference basis,
- context unchanged,
- context update blocked or insufficient.

Und für Kandidaten:
- candidate proposed,
- candidate evidence-backed,
- candidate context-derived,
- candidate compute-result-derived,
- candidate accepted for future memory handling,
- candidate rejected,
- candidate stale,
- candidate insufficient,
- persistence unavailable/deferred,
- persistence performed only if real path exists,
- no persistence performed.

## 2) Context-Update-Lifecycle (kanonisch)

`CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP` hält die Trennung explizit:
- **update only**: context update ohne Candidate-Zwang.
- **update plus candidate proposal**: explizite Folgeaktion, kein impliziter Nebeneffekt.
- **candidate without context mutation**: zulässiger Candidate-only Pfad bei stabiler Context-Lage.
- **rejected candidate with context preserved**: Candidate-Entscheid ohne stilles Context-Rewrite.
- **blocked/insufficient**: Caveat-/Insufficiency-Pfade bleiben sichtbar und nicht stillschweigend.

Damit bleiben separat unterscheidbar:
- runtime context mutation,
- reference attachment,
- candidate decision,
- persistence boundary.

## 3) Memory-Candidate-Lifecycle (kanonisch)

`CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP` kodiert explizite Candidate-Zustände:
- candidate proposed,
- candidate evidence-backed (inkl. replay/reference backing),
- candidate context-derived,
- candidate compute-result-derived,
- candidate accepted for future memory handling,
- candidate rejected,
- candidate stale,
- candidate insufficient,
- persistence unavailable/deferred,
- persistence performed only if real path exists,
- candidate no persistence performed.

Wichtig:
- accepted-for-future-handling ist **kein** Memory-Commit.
- rejected/stale/insufficient bleiben explizite End-/Zwischenzustände ohne Persistenzwirkung.
- persistence unavailable/deferred bleibt expliziter Marker und verhindert implizite Commit-Annahmen.
- persistence performed only if real path exists ist als Regel fixiert, ohne in Prompt 2 einen Realpfad zu bauen.
- compute-result-derived Kandidaten bleiben begrenzt und fault/caveat-sensitiv.

## 4) Evidence/Replay/Compute-Basis ohne Memory-Verwechslung

Kanonische Leitplanken:
- evidence refs und replay/reference context dürfen Kandidaten stützen,
- teilweise/stale/insufficient Basis muss als caveated/insufficient sichtbar bleiben,
- compute result uptake bleibt Context-Update; candidate formation ist optional und explizit,
- keine automatische Persistenz aus Candidate- oder Result-Pfaden.

## 5) Persistenzgrenze (explizit deferred)

Die Prompt-1 Null-Lane bleibt verbindlich:
- `blue_brain_persisted_memory_none_in_current_baseline`

Daraus folgt für Prompt 2:
- Candidate-Lifecycle ist real definiert,
- actual memory commit remains intentionally deferred,
- keine neue Memory-Commit-Engine, keine Vector-DB/Knowledge-Graph-Plattform.

## 6) Non-canonical/internal-only Pfade bleiben ausgegrenzt

Interne/Expert-/Compat-Pfade (`run_operation_with_entry`, `replay_with_entry`,
`build_backend(kind=stub|candle|worker)`, `domains/ai*`) bleiben:
- nicht kanonische Memory-/Context-Autorität,
- nur via Down-Mapping auf outward status/evidence references für Blue-Brain-facing Interpretation.

## 7) Ergebnis und BB3/BB4-Anschluss

BB3 Prompt 2 liefert belastbare Lifecycle-Grenzen:
- Context Update, Candidate Lifecycle und Persistence sind klar getrennt.
- Candidate-States sind strukturiert und testbar.
- Evidence-/Replay-/Compute-Bezüge sind explizit, ohne implizites Memory-System.

Nicht enthalten:
- kein Memory-Commit-System,
- keine Consolidation-/Reasoning-/Audit-Plattform,
- keine neurodynamischen Spezialmodelle.
