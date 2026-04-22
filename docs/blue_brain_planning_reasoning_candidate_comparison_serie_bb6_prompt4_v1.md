# Serie BB6 Prompt 4: Planning-/Reasoning-candidate comparison über Evidence, Context und Selection integrieren

Status: BB6 Prompt 4 etabliert eine **kanonische Candidate-Comparison-Schicht** für planning/reasoning candidates. Der Fokus ist Vergleichbarkeit, Basis-Transparenz und Diagnose. Es wird **keine Ranking-Engine**, **keine Planning-Engine**, **keine Reasoning-Engine**, **keine Policy-Schicht** und **keine Action-Execution-Plattform** gebaut.

## Scope und Bindung an den bestehenden Kern

- Finale technische Referenzlinie bleibt unverändert:
  - `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- Canonical Code-Map für BB6 Prompt 4 liegt in `runtime/ucf-compute/src/reference_map.rs`:
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP`
- BB6 Prompt 4 bleibt auf Candidate-Comparison begrenzt und trennt strikt zu Selection-Entscheidung, Proposal-Erzeugung, Action Execution und Memory Commit.
- Compute-Kern bleibt maintenance-only.

## Canonical Candidate-Comparison Klassen

`CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP` führt folgende Klassen:

1. `comparable candidates`
2. `comparison basis available`
3. `comparison meaningful`
4. `comparison caveated`
5. `comparison inconclusive`
6. `comparison not meaningful`
7. `comparison blocked`
8. `non-canonical/internal-only comparison`

Diese Klassen sind vergleichs- und diagnoseorientiert; sie liefern keine Ranking-/Winner-Semantik.

## Explizite Vergleichsbasis

Die Comparison-Layer macht die Basis pro Vergleich explizit:

- runtime basis
- context basis
- evidence/reference basis
- selection/attention basis
- memory-candidate or commit-feedback basis
- proposal-status basis
- caveats

Damit wird verhindert, dass comparison nur über implizite oder expert-only Pfade sichtbar bleibt.

## Meaningfulness-, Caveat- und Blocked-Semantik

Die kanonische Surface unterscheidet klar:

- meaningful because candidates share comparable basis,
- caveated because evidence/reference differs,
- inconclusive due to partial/stale basis,
- not meaningful due to incompatible context,
- blocked due to missing candidate basis,
- blocked due to non-canonical dependency.

Diese Zustände bleiben getrennt von Ranking, Entscheidung und Execution.

## Trennung zu Selection/Priority/Deferral

Candidate-Comparison bedeutet nicht automatische Auswahl:

- compared but not selected,
- compared and candidate remains deferred,
- compared and candidate remains proposal-ready,
- comparison informs selection, but does not decide.

Selection-/Deferral-Zustände bleiben BB4-semantisch und werden nicht durch comparison überschrieben.

## Trennung zu Proposal und Execution

Candidate-Comparison erzeugt nicht automatisch Proposals oder Ausführung:

- comparison only,
- comparison supports proposal caveat,
- comparison insufficient for proposal,
- no proposal generated,
- no action executed.

`proposal-ready` bleibt non-executing Status (BB6 Prompt 2).

## Rückbindung an Memory-Boundary

Comparison kann Memory-Signale nutzen, ohne Commit-Engine zu werden:

- candidate has future-memory-ready support,
- candidate has rejected memory basis,
- commit unavailable limits comparison,
- committed-if-present strengthens basis only if real path exists,
- no memory commit implied by comparison.

## Runtime-/Diagnostics-Rückführung

Runtime/Diagnostics sehen mindestens:

- which candidates were compared,
- why comparison was meaningful/caveated/inconclusive/not meaningful,
- which caveats remain,
- whether comparison affected selection/proposal state,
- no decision/execution/reasoning-completed claim.

## Non-canonical Vergleichspfade

Nicht kanonisch für BB6 Candidate-Comparison bleiben:

- compute-interne Details,
- expert/internal hooks,
- legacy/compat objects,
- unstabile test/dev surfaces,
- implizite orchestration helpers ohne Down-Mapping.

Solche Pfade gelten als `non-canonical/internal-only comparison` bis eine explizite Down-Mapping-Bindung auf die kanonischen Runtime-/Context-/Evidence-/Selection-/Memory-/Proposal-Referenzen existiert.

## Explizite Nicht-Ziele in BB6 Prompt 4

- keine Ranking-Engine,
- keine Planning-Engine,
- keine Reasoning-Engine,
- keine Policy-/RL-/Agentenplattform,
- keine automatische Selection,
- keine automatische Proposal-Erzeugung,
- keine automatische Action Execution,
- keine automatische Memory Persistence,
- keine Hodgkin-Huxley/Kuramoto-Integration.

BB6 Prompt 4 liefert eine belastbare Vergleichssemantik für planning/reasoning candidates, die Diagnose und spätere Arbeit vorbereitet, ohne Decision- oder Execution-Autorität zu behaupten.
