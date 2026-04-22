# Serie BB6 Prompt 3: Reasoning-candidate diagnostics / insufficiency / caveat feedback zurückbinden

Status: BB6 Prompt 3 bindet **reasoning-candidate diagnostics** zurück in Runtime, Context, Selection, Memory-Boundary und Candidate-to-Proposal-Grenze. Es wird **keine Reasoning-Engine**, **keine Explainability-Plattform**, **keine Policy-/Audit-Plattform** und **keine Action-Execution-Engine** gebaut.

## Scope und technische Leitplanke

- Compute-Kern bleibt maintenance-only und auf der finalen Linie: `submit -> compute_canonical -> result/fault/status -> execution_snapshot`.
- Canonical Code-Map für BB6 Prompt 3 liegt in `runtime/ucf-compute/src/reference_map.rs`:
  - `CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP`
- BB6 Prompt 3 bleibt diagnostisch und feedback-orientiert; keine zweite Wahrheitsquelle.

## Canonical Reasoning-candidate Diagnostics-Klassen

Die folgende Klassenbildung ist verbindlich und code-pinned:

1. `candidate-basis diagnostic`
2. `sufficient candidate diagnostic`
3. `partial candidate diagnostic`
4. `caveated candidate diagnostic`
5. `stale candidate diagnostic`
6. `insufficient candidate diagnostic`
7. `deferred candidate diagnostic`
8. `rejected candidate diagnostic`
9. `proposal-ready diagnostic`
10. `non-canonical/internal-only diagnostic`

## Candidate-Basis kompakt und kanonisch

Candidate-Diagnostics referenzieren kompakte Basisquellen statt freier Begründungsprosa:

- runtime-derived,
- context-derived,
- evidence/reference-derived,
- selection-derived,
- memory-candidate-derived,
- commit-feedback-derived,
- proposal-derived.

## Insufficiency- und Caveat-Gründe (kanonisch)

Mindestens folgende Gründe sind explizit unterscheidbar:

- insufficient due to missing context,
- insufficient due to weak or missing evidence,
- insufficient due to stale reference basis,
- caveated due to partial evidence,
- caveated due to selection/attention caveat,
- caveated due to unavailable memory commit,
- blocked due to non-canonical/internal dependency.

## Rückbindung in Candidate-to-Proposal Boundary (BB6 Prompt 2)

Diagnostics informieren explizit die Übergänge:

- `candidate remains candidate`,
- `candidate becomes proposal-ready`,
- `candidate yields caveated proposal`,
- `candidate deferred before proposal`,
- `candidate rejected before proposal`.

Dabei bleibt load-bearing:

- proposal-ready ist **nicht** executed action,
- proposal-ready ist **nicht** memory-committed,
- proposal-ready ist **kein** reasoning-completed claim.

## Rückbindung in Selection / Priority / Deferral (BB4)

Candidate-Diagnostics informieren BB4 ohne Ranking-/Policy-Engine-Aufbau:

- sufficient candidate can be selected,
- caveated candidate may be deferred,
- stale candidate requires recheck,
- insufficient candidate cannot become selected/proposal-ready,
- rejected candidate excluded from current selection.

## Rückbindung in Memory-Boundary und Commit-Feedback (BB5)

Candidate-Diagnostics halten die BB5-Grenze explizit:

- commit unavailable limits candidate basis,
- future-memory-ready supports candidate but not commit,
- committed-if-present strengthens candidate only if real path exists,
- rejected memory candidate weakens or blocks candidate basis.

## Runtime-/Context-Feedback-Rückbindung (BB2/BB3)

Blue-Brain Runtime/Context kann explizit sehen:

- candidate basis observed,
- candidate partial/caveated/insufficient,
- candidate proposal-ready,
- candidate deferred or rejected,
- no action execution implied,
- no memory commit implied,
- no reasoning completed claim.

## Non-canonical Diagnostics ausgrenzen

Folgende Pfade sind nicht kanonisch für BB6 Candidate-Diagnostics:

- compute-interne Details,
- expert/internal hooks,
- legacy/compat objects,
- unstabile test/dev surfaces,
- implizite orchestration helpers ohne kanonische Down-Map.

Diese bleiben `non-canonical/internal-only diagnostic` bis explizites Down-Mapping auf Runtime-/Context-/Evidence-/Selection-/Memory-/Proposal-Referenzen vorliegt.

## Explizite Nicht-Ziele in BB6 Prompt 3

- keine vollständige Reasoning-Engine
- keine Explainability-, Audit- oder Policy-Plattform
- keine Planning- oder Agentenplattform
- keine automatische Action Execution
- keine automatische Compute Invocation
- keine automatische Memory Persistence
- keine neurodynamische Integration (Hodgkin-Huxley/Kuramoto bleibt außerhalb)

## Ergebnis

BB6 Prompt 3 etabliert eine belastbare, begrenzte Diagnostics-/Feedback-Schicht für planning/reasoning candidates. Die Semantik bleibt technisch kompakt, kanonisch rückgebunden und strikt getrennt von Execution, Memory Commit und Reasoning-Completion-Behauptungen.
