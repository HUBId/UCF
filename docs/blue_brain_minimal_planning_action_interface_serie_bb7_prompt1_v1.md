# Serie BB7 Prompt 1: Minimal planning/action interface und action-ready proposal semantics

Status: BB7 Prompt 1 fixiert eine **minimale planning/action interface boundary** auf Basis der BB6 non-executing Proposals. Es wird **keine Planning-Engine**, **keine Reasoning-Engine**, **keine Tool-Execution-Plattform**, **keine Policy-/Governance-Schicht** und **keine Agentenplattform** gebaut.

## Scope und technische Leitplanke

- Compute-Kern bleibt maintenance-only und auf der finalen Linie: `submit -> compute_canonical -> result/fault/status -> execution_snapshot`.
- Canonical BB7 Prompt 1 Code-Map in `runtime/ucf-compute/src/reference_map.rs`:
  - `CANONICAL_BLUE_BRAIN_MINIMAL_PLANNING_ACTION_INTERFACE_MAP`
- BB7 Prompt 1 bleibt bewusst bei Readiness-/Boundary-Semantik und baut keine Ausführungs-Engine.

## Kanonische minimale Planning-/Action-Interface-Klassen

Die folgenden Klassen sind code-pinned und bleiben explizit getrennt:

1. `diagnostic-only proposal`
2. `plan-ready proposal`
3. `action-ready proposal`
4. `deferred proposal`
5. `blocked proposal`
6. `rejected proposal`
7. `caveated proposal`
8. `insufficient proposal basis`
9. `executed action (canonical path only if explicit invocation exists)`
10. `non-canonical/internal-only action path`

## Action-ready Proposal-Semantik (minimal und non-executing)

Action-ready bedeutet in BB7 Prompt 1 ausschließlich:

- Proposal ist für spätere Action-Boundary vorbereitet,
- Context/Evidence/Selection/Caveats bleiben referenzierbar,
- keine automatische Ausführung.

Sichtbare action-ready Unterfälle:

- `action-ready but not executed`
- `action-ready with caveat`
- `action-ready blocked by missing boundary`
- `action-ready requires future action subsystem`

## Plan-ready Proposal-Semantik (minimal und non-planner)

Plan-ready bedeutet in BB7 Prompt 1 ausschließlich:

- Proposal ist für spätere Planverarbeitung vorbereitet,
- es wurde kein Plan erzeugt,
- es wurde kein Plan ausgeführt.

Sichtbare plan-ready Unterfälle:

- `plan-ready but no plan generated`
- `plan-ready with caveat`
- `plan-ready deferred`
- `plan-ready blocked due to insufficient basis`

## Rückbindung an BB6 Diagnostics und Candidate Comparison

Proposal-Readiness ist in BB7 Prompt 1 nicht frei spekulativ, sondern basisgebunden:

- `sufficient candidate basis permits readiness`
- `caveated candidate basis yields caveated readiness`
- `inconclusive comparison limits readiness`
- `insufficient candidate basis blocks readiness`
- `non-canonical basis blocks readiness`

Die Rückbindung erfolgt über:

- `CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP`
- `CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP`
- BB6 Candidate-/Proposal-Boundary-Maps.

## Rückbindung an Context/Evidence/Selection/Memory-Grenze

Readiness-Lanes referenzieren kanonisch:

- context basis,
- evidence/reference basis,
- selection/attention and deferral state,
- trigger origin,
- memory candidate and commit feedback (nur als Basis-/Caveat-Signal),
- caveats.

Damit bleibt die Trennung zu BB3/BB4/BB5 explizit erhalten.

## Harte Trennung: Readiness vs Execution/Tool/Compute/Memory

BB7 Prompt 1 kodiert explizit:

- `no action execution`
- `no plan generation`
- `no tool invocation`
- `no compute invocation`
- `no memory commit`

Readiness markiert maximal `future-action-ready` oder `future-plan-ready`, ohne automatische Seiteneffekte.

## Executed actions im aktuellen Repo

- Reale Action Execution bleibt ausschließlich auf expliziten kanonischen Compute-Invocationspfaden.
- BB7 Prompt 1 führt keine neue Action Execution ein.
- `executed action` wird nur als klar getrennte, explizite Handoff-Klasse anerkannt.

## Non-canonical planning/action paths

Folgende planning/action-nahe Pfade bleiben non-canonical/internal-only:

- `run_operation_with_entry`
- `replay_with_entry`
- `build_backend(kind=stub|candle|worker)`
- `domains/ai*` compatibility/internal seams

Diese Pfade können BB7-Readiness nicht autorisieren, solange kein explizites Down-Mapping auf kanonische Candidate-/Context-/Evidence-/Selection-Referenzen erfolgt.

## Explizite Nicht-Ziele in BB7 Prompt 1

- keine Planning-Engine
- keine Reasoning-Engine
- keine Policy-/Governance-Schicht
- keine RL-/Agentenplattform
- keine Tool-Execution-Plattform
- keine automatische Compute Invocation aus Readiness
- keine automatische Memory-Persistence aus Readiness
- keine neurodynamische Integration (Hodgkin-Huxley/Kuramoto bleibt außerhalb)

## Ergebnis

BB7 Prompt 1 liefert eine belastbare minimale Planning-/Action-Interface-Grenze, in der Proposal-Readiness klar klassifiziert und auf Candidate-/Diagnostics-/Comparison-/Context-Basis rückgebunden ist, ohne Execution-, Tool-, Compute- oder Memory-Commit-Semantik zu implizieren.
