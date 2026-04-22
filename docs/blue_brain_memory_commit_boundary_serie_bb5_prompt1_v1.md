# Serie BB5 Prompt 1: Minimal Memory Commit Boundary und commit-eligible candidate semantics (repo-basiert)

Status: BB5 Prompt 1 zieht die minimale Memory-Commit-Grenze auf Basis der vorhandenen BB3/BB4-Substanz fest. Es wird **keine Memory-Engine** eingeführt, kein neuer Storage-Stack gebaut, und der Compute-Kern bleibt unverändert.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_PERSISTENCE_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_FUTURE_MEMORY_ATTACHMENT_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-compute/src/lib.rs`

Finale Referenzlinie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Minimal Memory Commit Boundary Map (kanonisch)

`CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP` fixiert die commit-relevanten Klassen explizit:
- not a memory candidate,
- memory candidate proposed,
- memory candidate deferred,
- memory candidate rejected,
- memory candidate stale,
- memory candidate insufficient,
- commit-eligible candidate,
- future-memory-ready candidate,
- committed memory (only if real path exists),
- reference-only / not memory,
- non-canonical/internal-only persistence path.

Damit werden Candidate, Commit-Bereitschaft, Handoff und Nicht-Memory-Flächen sauber getrennt.

## 2) Commit Eligibility Conditions (minimal und repo-basiert)

`CANONICAL_BLUE_BRAIN_COMMIT_ELIGIBILITY_CONDITIONS_MAP` hält die minimalen Voraussetzungen fest:
- sufficient evidence/reference basis,
- selected or accepted candidate status,
- non-stale context basis,
- no blocking caveat,
- no internal/expert-only dependency,
- explicit persistence path exists,
- future-memory-ready handoff.

Semantik:
- Bedingungen müssen gemeinsam erfüllt sein, damit ein Kandidat als commit-eligible candidate gilt.
- Fällt eine Bedingung aus, bleibt der Kandidat deferred, rejected, stale, insufficient oder not-memory.

## 3) Selection/Attention und Evidence Quality als Commit-Grenze

BB4-Selection-Semantik bleibt wirksam, aber begrenzt:
- selected candidate can become commit-eligible,
- deferred candidate stays not commit-eligible,
- ignored/rejected candidate stays not commit-eligible,
- caveated basis can keep candidate blocked or future-memory-ready.

BB3-Evidence-/Reference-Qualität bleibt wirksam:
- sufficient evidence/reference basis erlaubt Eligibility-Prüfung,
- partial/caveated evidence bleibt caveated,
- stale/insufficient evidence blockt Commit-Eligibility,
- reference-only bleibt not memory.

## 4) Actual Commit vs Future-Memory-Handoff

Aktuelle Repo-Baseline:
- no actual memory commit is implemented.
- Commit ist nur als Bedingungsklasse modelliert: committed memory (only if real path exists).
- Ohne realen Persistenzpfad bleibt der Ausgang `future-memory-ready handoff`.

Das verhindert Implementierungsbehauptungen ohne realen Repo-Pfad.

## 5) History/Snapshot/Evidence/Replay/Trace erneut abgesichert

Die BB5-Grenze bestätigt explizit:
- History ≠ Memory,
- Snapshot ≠ Memory,
- Evidence ≠ Memory,
- Replay/Trace ≠ Memory.

Diese Flächen dürfen Candidate-Basis liefern, aber keine Commit-Autorität.

## 6) Non-canonical Commit-/Persistence-Pfade ausgrenzt

Nicht-kanonische Pfade (expert/internal/legacy/compat) bleiben explizit ausgeschlossen:
- non-canonical/internal-only persistence path,
- kein commit-eligible Status aus internen Hooks,
- nur outward candidate/evidence/selection references sind kanonische Basis.

## 7) Grenzen des Schritts

Unverändert außerhalb dieses Prompts:
- keine Policy-/Governance-/Ranking-Engine,
- keine Reasoning-/Audit-/Monitoring-Plattform,
- keine Vector-DB-/Knowledge-Graph-/Memory-Consolidation-Plattform,
- keine neue Compute-Core-Arbeit,
- keine neurodynamische Spezialintegration wie Hodgkin-Huxley/Kuramoto.

Compute-Kern bleibt maintenance-only.
