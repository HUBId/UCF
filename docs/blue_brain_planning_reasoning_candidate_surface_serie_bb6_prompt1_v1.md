# Serie BB6 Prompt 1: Planning-/Reasoning-Candidate Surface über Runtime, Context, Selection und Memory-Grenze

Status: BB6 Prompt 1 zieht eine **Planning-/Reasoning-candidate** Grundlage auf vorhandenen BB2/BB3/BB4/BB5-Signalen fest. Es wird **keine Planning-Engine**, **keine Reasoning-Engine**, **keine Policy-/RL-/Agentenplattform** gebaut.

## Scope und technische Leitplanke

- Compute-Kern bleibt maintenance-only und auf der finalen Linie: `submit -> compute_canonical -> result/fault/status -> execution_snapshot`.
- Canonical Code-Map für BB6 liegt in `runtime/ucf-compute/src/reference_map.rs`:
  - `CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP`
- Die BB6-Surface ist eine **Kandidatenbasis** für spätere Planungs-/Reasoning-Arbeit, nicht deren Ausführung.

## Canonical Planning-/Reasoning-Candidate Klassen (BB6)

Die folgende Klassenbildung ist verbindlich und code-pinned:

1. `runtime-derived planning candidate`
2. `context-derived reasoning candidate`
3. `evidence/reference-derived reasoning candidate`
4. `selection-derived action candidate`
5. `memory-candidate-derived reasoning candidate`
6. `commit-feedback-derived candidate`
7. `insufficient candidate basis`
8. `non-canonical/internal-only planning-like path`

## Candidate-Basis-Zustände

BB6 macht Candidate-Basis-Qualität explizit sichtbar:

- `candidate basis available`
- `partial/caveated`
- `stale`
- `insufficient`
- `deferred`
- `candidate proposed but unresolved`
- `evidence observed but no reasoning candidate`
- `blocked`

## Harte Trennung: Candidate-Basis vs Entscheidung/Ausführung

Die BB6-Surface kodiert explizit:

- Candidate-Basis bedeutet **nicht**: plan selected.
- Candidate-Basis bedeutet **nicht**: policy applied.
- Candidate-Basis bedeutet **nicht**: reasoning completed.
- Candidate-Basis bedeutet **nicht**: action executed (`no action execution implied`).
- Candidate-Basis bedeutet **nicht**: memory committed (`no memory commit implied`).

## Runtime-/Trigger-Einordnung (BB2 + BB4)

- Trigger können eine candidate basis informieren (`trigger suggests action candidate`).
- Blocked Trigger bleiben blockierte Kandidatenbasis (`blocked trigger yields blocked candidate basis`).
- Caveated Trigger bleiben caveated candidate (`caveated trigger yields caveated candidate`).
- Daraus folgt weiterhin: `no planner decision implied`.

## Context-/Evidence-/Reference-Einordnung (BB3)

- Sufficient context/evidence kann reasoning candidate basis liefern.
- Partial/caveated/stale/insufficient bleibt explizit sichtbar.
- `evidence observed but no reasoning candidate` bleibt eigener Zustand.
- `reasoning candidate proposed but not resolved` bleibt eigener Zustand.

## Selection-/Priority-/Deferral-Einordnung (BB4)

- `selected context yields candidate basis`.
- `deferred candidate remains unresolved`.
- `ignored item does not produce candidate`.
- `rejected item does not produce candidate`.
- `caveated selection produces caveated candidate`.

## Memory-Candidate-/Commit-Feedback-Einordnung (BB5)

- `future-memory-ready candidate may support later reasoning`.
- `rejected memory candidate weakens candidate basis`.
- `commit unavailable limits reasoning basis`.
- `committed-if-present may become stronger basis only if real path exists`.

## Non-canonical planning-/reasoning-like Pfade

Folgende Pfade bleiben für BB6 **non-canonical/internal-only** und sind nicht autoritativ:

- compute-interne/expert hooks (`run_operation_with_entry`, `replay_with_entry`)
- compat-/legacy-Lanes (`build_backend(kind=stub|candle|worker)`, `domains/ai*`)

Diese Pfade gelten nur nach explizitem Down-Mapping auf kanonische Runtime-/Context-/Evidence-/Selection-/Memory-Boundaries.

## Explizite Nicht-Ziele in BB6 Prompt 1

- keine Planning-Engine
- keine Reasoning-Engine
- keine Policy-Engine
- keine RL-Plattform
- keine autonome Agentenplattform
- keine Memory-Commit-Engine
- keine neurodynamische Integration (Hodgkin-Huxley/Kuramoto bleibt außerhalb)

## Ergebnis

BB6 Prompt 1 liefert eine belastbare, testbare Candidate-Surface für spätere Planung/Reasoning-Schritte, bleibt aber strikt auf Candidate-Basis-Semantik begrenzt und hält die Compute-Core-Maintenance-Grenze unverändert ein.
