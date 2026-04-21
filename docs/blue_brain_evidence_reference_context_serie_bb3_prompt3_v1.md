# Serie BB3 Prompt 3: Evidence-backed und Replay/Reference-Context in Blue-Brain-Laufzeit (repo-basiert, ohne Memory-Commit)

Status: BB3 Prompt 3 integriert evidence-backed context und replay/reference-backed context tiefer in die bestehende Blue-Brain-Laufzeitlinie, ohne neue Audit-/Reasoning-/Memory-Commit-Plattform.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::status_evidence_export_surface`

Finale Referenzlinie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Kanonische Evidence-/Reference-Context-Klassen

`CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP` trennt explizit:
- evidence-backed context,
- replay-backed context,
- snapshot/reference-backed context,
- trace-backed context,
- caveated reference context,
- insufficient reference context,
- non-canonical/internal-only reference path.

Damit bleibt sichtbar, welche Basis Blue-Brain-seitig tatsächlich als kanonischer Context-Input gilt und welche Pfade nur intern/expert-only sind.

## 2) Reference Quality und Caveats (explizit)

Die Referenzqualität wird semantisch explizit geführt:
- sufficient reference basis,
- partial reference basis,
- stale reference basis,
- caveated reference basis,
- insufficient reference basis.

Diese Qualitäten informieren Runtime-Posture, Context-Update-Semantik und Candidate-Semantik ohne numerische Bewertungsmaschine.

## 3) Evidence-backed Context in Context Updates

Prompt-3-Semantik macht mindestens explizit:
- context updated with evidence reference,
- context update caveated by partial/stale evidence,
- context update blocked due to insufficient evidence,
- evidence observed without context update,
- no persistence implied.

Evidence bleibt damit Context-/Reference-Grundlage, aber nicht automatisch Memory.

## 4) Replay/Reference-Backed Context in der Laufzeit

Replay-/Reference-Kontext wird als Context-Basis explizit führbar:
- runtime context restored or informed by replay/reference basis,
- replay/reference context caveated,
- reference basis unavailable or insufficient,
- replay/reference used for context only, not memory commit.

Es wird keine neue Replay-/Rehydration-Plattform gebaut; bestehende `replay_preflight`/`replay_with_entry`-Flächen bleiben die Basis.

## 5) Evidence-/Reference-Basis für Memory Candidates (ohne Commit)

Candidate-Semantik bleibt getrennt von Persistenz:
- candidate evidence-backed,
- candidate replay/reference-backed,
- candidate trace/snapshot-backed,
- candidate caveated by weak reference,
- candidate insufficient due to missing/stale reference,
- no persistence performed.

Die bestehende Null-Lane für Persistenz bleibt autoritativ; es gibt weiterhin keinen Memory-Commit-Pfad.

## 6) Compute outward-facing vs interne Referenzen

Blue-Brain-kontextrelevante Nutzung bleibt auf outward-facing Status/Evidence-Exports:
- `status_evidence_export_surface` bleibt kanonischer Referenzanker.
- internal/expert-only Pfade bleiben als non-canonical/internal-only reference path markiert.
- Down-Mapping auf outward refs bleibt Pflicht vor Blue-Brain-facing Nutzung.

## 7) Ergebnis

BB3 Prompt 3 liefert eine belastbare Evidence-/Replay-/Reference-Context-Integration für Blue-Brain:
- Context Updates und Candidates sind klar auf Referenzbasis rückführbar,
- caveated/stale/partial/insufficient Basis bleibt explizit sichtbar,
- replay/reference und memory persistence bleiben getrennt,
- no persistence performed bleibt hart und testbar.
