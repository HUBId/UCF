# Serie BB13 Prompt 1: Minimal echte Tool/Action-Execution-Linie auf bestehender Safety-Boundary

Status: Diese BB13-Linie führt eine **kleinste reale Execution-Implementierung** ein, die ausschließlich auf der vorhandenen BB7/BB9-Handoff-, Eligibility- und Safety-Precheck-Boundary arbeitet.

## Kanonischer Scope (bewusst klein)

Die neue minimale Execution-Linie in `runtime/ucf-compute/src/blue_brain_minimal_execution.rs` erlaubt genau:

- Eingang über **kanonischen** `future-action-ready` + `execution-eligible` Handoff.
- Harte Bindung an BB9 Safety-Precheck.
- Eine einzige minimale Aktion: `emit_canonical_signal`.
- Explizite, getrennte Zustände:
  - `execution-eligible but not executed`
  - `execution-requested`
  - `execution-started`
  - `execution-completed`
  - `execution-failed`
  - `execution-blocked`
  - `execution-cancelled`
  - `execution-unavailable`
  - `non-canonical/internal-only execution path`

Nicht im Scope:
- keine Agentenplattform
- keine Tool-Orchestrierung
- keine autonome Multi-Step-Loop
- keine neue Policy-Sprache/Governance-Layer
- keine Compute-Core-Erweiterung

## Eintrittsgrenze und Bindung an BB9

Execution wird nur zugelassen, wenn gleichzeitig gilt:

1. `handoff_class == FutureActionReady`
2. `eligibility_class == ExecutionEligibleHandoff`
3. `safety_precheck in {Passed, Caveated}`
4. kein `cancelled`
5. kein `internal_only_path`

Alle anderen Fälle bleiben **nicht-ausführend** (`blocked`, `unavailable`, `cancelled`, `non-canonical`) und liefern kein echtes Action-Result.

## Safety-Precheck bleibt harte Vorbedingung

- `Failed`/`Blocked`/`Insufficient`/`NotApplicable` => `execution-blocked`, keine Ausführung.
- `Unavailable` => `execution-unavailable`, keine Ausführung.
- `Passed`/`Caveated` können eine echte Ausführung ermöglichen (bei explizitem Request).

Damit bleibt erhalten: `execution-eligible != executed action`.

## Result-Boundary (BB7/BB9 -> reale Execution)

Resultate sind explizit getrennt in:

- `PlaceholderOnly`
- `ExecutionRequested`
- `ActualExecutionResult`
- `FailedExecutionResult`
- `BlockedNoResult`
- `UnavailableExecutionPath`

Wichtig:
- Placeholder-only bleibt nicht-ausführend.
- Actual result entsteht nur im echten Execution-Path.
- Failed/blocked/unavailable/cancelled bleiben getrennt und werden nicht in Placeholder oder Policy-/Compute-/Memory-Feedback umgeschrieben.

## no-direct-* / Begrenzungen bleiben intakt

Die minimale Execution-Linie erhält explizit:

- keine Safety-Override-Logik
- keine implizite Memory-Persistenz
- keine Compute-Core-Mutation
- keine Policy-Engine-Erweiterung
- kein non-canonical Upgrade in kanonische Execution

## Testabdeckung (gezielt)

Unit-Tests decken mindestens ab:

- eligible aber nicht ausgeführt (PlaceholderOnly)
- completed bei passed precheck
- blocked/unavailable durch Safety-Precheck
- failed vs cancelled unterscheidbar
- non-canonical/internal-only bleibt unavailable/non-executing
