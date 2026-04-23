# Serie BB7 Prompt 2: Plan/action readiness diagnostics und blocked-action feedback zurückbindung

Status: BB7 Prompt 2 konsolidiert die kanonische Readiness-Diagnostics-Schicht über Candidate, Proposal, Selection, Context/Evidence, Runtime und Memory-Commit-Boundary. Es wird **keine Planning-Engine**, **keine Tool-Execution-Plattform**, **keine Policy-/Governance-Schicht** und **keine Agentenplattform** eingeführt.

## Scope und Leitplanke

- Compute-Kern bleibt maintenance-only auf der finalen Linie: `submit -> compute_canonical -> result/fault/status -> execution_snapshot`.
- Kanonische Prompt-2 Code-Map in `runtime/ucf-compute/src/reference_map.rs`:
  - `CANONICAL_BLUE_BRAIN_PLAN_ACTION_READINESS_DIAGNOSTICS_MAP`
- Prompt 2 bindet BB7 Prompt 1 (`CANONICAL_BLUE_BRAIN_MINIMAL_PLANNING_ACTION_INTERFACE_MAP`) an BB6 Candidate Diagnostics/Comparison, BB4 Selection/Deferral, BB3 Context/Evidence und BB5 Commit-Feedback zurück.

## Kanonische Plan/action readiness diagnostics map

Die kanonischen Klassen bleiben explizit getrennt:

1. `plan-ready diagnostic`
2. `action-ready diagnostic`
3. `diagnostic-only proposal diagnostic`
4. `deferred readiness diagnostic`
5. `blocked readiness diagnostic`
6. `rejected readiness diagnostic`
7. `caveated readiness diagnostic`
8. `insufficient readiness diagnostic`
9. `non-canonical/internal-only readiness diagnostic`

## Kompakte kanonische Readiness-Gründe

Prompt 2 verwendet kompakte, code-pinned Gründe statt freier Prosa:

- `ready due to sufficient candidate basis`
- `ready due to sufficient context/evidence`
- `ready due to selection/attention state`
- `deferred due to partial evidence`
- `blocked due to stale context`
- `blocked due to insufficient candidate basis`
- `blocked due to missing action boundary`
- `caveated due to memory/commit unavailability`
- `rejected due to candidate/proposal rejection`

Die Gründe sind ausdrücklich Diagnose-/Boundary-Signale und keine Planner-/Policy-/Tool-Urteile.

## Blocked-action feedback (kanonische Bedeutung)

`blocked-action feedback` bedeutet in BB7 Prompt 2 nur:

- Readiness-Übergang zu `action-ready` oder `future-action-ready` konnte nicht stattfinden,
- oder die action boundary fehlt,
- oder die Readiness-Basis ist unzureichend.

`blocked-action feedback` bedeutet **nicht**:

- `tool executed`,
- `action failed`,
- `policy denied`,
- `planner denied`.

Damit bleibt blocked-action feedback strikt auf Readiness-Diagnostics begrenzt.
blocked-action feedback means readiness transition could not occur.

## Rückbindung in Candidate-/Proposal-Boundary

Readiness-Diagnostics informieren kanonisch:

- candidate remains candidate,
- candidate becomes plan-ready proposal,
- candidate becomes action-ready proposal,
- proposal remains diagnostic-only,
- proposal deferred,
- proposal blocked/rejected.

Keine Klasse impliziert automatische Execution, Tool-Invocation, Compute-Invocation oder Memory-Commit.

## Rückbindung in Selection/Priority/Deferral

Readiness-Diagnostics tragen in BB4-kompatibler Form zurück:

- action-ready proposal can be selected for future boundary,
- caveated proposal remains deferred,
- insufficient proposal cannot become selected/action-ready,
- rejected proposal excluded from current selection,
- blocked proposal may require stronger context/evidence.

Dies bleibt Diagnose- und Boundary-Semantik; es wird keine Ranking-/Planning-/Policy-Plattform gebaut.

## Rückbindung in Context/Evidence/Memory-Boundary

Readiness-Diagnostics bewahren sichtbar:

- context basis,
- evidence/reference basis,
- selection/attention state,
- candidate comparison caveats,
- memory candidate or commit feedback,
- no memory commit implied.

Damit bleiben BB3/BB5-Grenzen explizit erhalten.

## Runtime-Feedback und harte Grenzen

Runtime sieht explizit:

- proposal readiness observed,
- plan-ready/action-ready status,
- deferred/blocked/rejected/caveated/insufficient status,
- no action execution,
- no tool invocation,
- no compute invocation,
- no memory commit.

## Non-canonical readiness diagnostics

Non-canonical/internal-only Pfade bleiben ausgeschlossen, bis Down-Mapping erfolgt:

- compute-interne Details,
- expert/internal hooks,
- legacy/compat objects,
- unstabile dev/test surfaces,
- implizite tool/orchestration helpers.

Ohne Down-Mapping auf kanonische Candidate-/Proposal-/Context-/Evidence-/Selection-Referenzen gibt es keine kanonische BB7-Readiness-Autorität.

## Explizite Nicht-Ziele

- keine Planning-Engine
- keine Action-Execution-Engine
- keine Tool-Execution-Plattform
- keine Policy-/Governance-Entscheidungsschicht
- keine automatische Compute Invocation aus Readiness-Diagnostics
- keine automatische Memory-Persistence aus Readiness-Diagnostics
- keine neurodynamische Integration (Hodgkin-Huxley/Kuramoto außerhalb dieses Schritts)

## Ergebnis

BB7 Prompt 2 stellt eine belastbare Readiness-Diagnostics- und blocked-action-feedback Schicht bereit, die plan-ready/action-ready/deferred/blocked/rejected/caveated/insufficient/non-canonical Zustände kompakt, kanonisch und integrationsfähig rückbindet, ohne Execution-/Tool-/Compute-/Policy-/Memory-Commit-Semantik zu vermischen.
