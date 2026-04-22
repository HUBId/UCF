# Serie BB6 Prompt 2: Candidate-to-action boundary und non-executing action proposals

Status: BB6 Prompt 2 zieht die Grenze zwischen **planning/reasoning candidate**, **action proposal (non-executing)** und **executed action** explizit. Es wird **keine Planning-Engine**, **keine Agentenplattform**, **keine Tool-Execution-Plattform** und **keine neue Policy-Schicht** gebaut.

## Scope und technische Leitplanke

- Compute-Kern bleibt maintenance-only und auf der finalen Linie: `submit -> compute_canonical -> result/fault/status -> execution_snapshot`.
- Canonical Code-Maps für BB6 Prompt 2 liegen in `runtime/ucf-compute/src/reference_map.rs`:
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP`
  - `CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP`
- BB6 Prompt 2 bleibt boundary-semantisch und baut keine Execution Engine.

## Canonical Candidate-/Proposal-/Action-Klassen

Die folgende Klassenbildung ist verbindlich und code-pinned:

1. `planning/reasoning candidate`
2. `action proposal (non-executing)`
3. `selected proposal`
4. `deferred proposal`
5. `rejected proposal`
6. `blocked proposal`
7. `caveated proposal`
8. `insufficient proposal basis`
9. `executed action (canonical path only if explicit invocation exists)`
10. `non-canonical/internal-only action-like path`

## Canonical non-executing Proposal-Zustände

Action Proposals bleiben ausdrücklich non-executing und nutzen folgende Zustände:

- `proposal created`
- `proposal selected for possible future action`
- `proposal deferred`
- `proposal rejected`
- `proposal blocked`
- `proposal caveated`
- `proposal insufficient basis`
- `no execution performed`

## Candidate-to-proposal Übergänge

Nicht jeder Candidate wird ein Proposal. Canonical Übergänge:

- `candidate remains candidate`
- `candidate yields proposal`
- `candidate insufficient for proposal`
- `candidate yields caveated proposal`
- `candidate rejected before proposal`
- `candidate deferred before proposal`

## Proposal-Basis-Rückbindung (BB2/BB3/BB4/BB5)

Jedes Proposal referenziert seine Basis explizit über:

- context basis,
- evidence/reference basis,
- selection/attention state,
- trigger/candidate origin,
- memory-candidate and commit-feedback basis (nur als Basis-/Diagnostiksignal),
- caveats.

Diese Rückbindung bleibt technisch/kompakt und wird nicht als freie Begründungsprosa modelliert.

## Harte Trennung: Proposal vs Execution / Compute / Tool / Memory

BB6 Prompt 2 kodiert explizit:

- `no automatic action execution`
- `no automatic compute invocation`
- `no automatic memory commit`
- `no automatic tool execution`

Auch `selected proposal` bedeutet nur mögliche zukünftige Aktion (future-action-ready / trigger-candidate), **nicht** bereits ausgeführte Aktion.

## Executed actions im aktuellen Repo

- Reale Ausführung existiert weiterhin nur auf dem kanonischen Compute-Pfad (`CanonicalComputeEntryPoint::{submit,status,drain_scheduler}`).
- BB6 Prompt 2 führt **keine** neue Execution-Fläche ein.
- Proposal-Semantik bleibt davon getrennt; ein Proposal impliziert keine Ausführung.

## Non-canonical action-like Pfade

Folgende action-nahe Pfade bleiben non-canonical/internal-only und sind nicht autoritativ für BB6:

- `run_operation_with_entry`
- `replay_with_entry`
- `build_backend(kind=stub|candle|worker)`
- `domains/ai*` compatibility/internal seams

Sie sind nur nach explizitem Down-Mapping auf kanonische Context-/Evidence-/Selection-/Candidate-Referenzen nutzbar.

## Explizite Nicht-Ziele in BB6 Prompt 2

- keine Planning-/Reasoning-Engine
- keine Policy-/Governance-Plattform
- keine autonome Agentenarchitektur
- keine Tool-Execution-Engine
- keine automatische Compute-Invocation aus Proposals
- keine automatische Memory-Persistence aus Proposals
- keine neurodynamische Integration (Hodgkin-Huxley/Kuramoto bleibt außerhalb)

## Ergebnis

BB6 Prompt 2 etabliert eine belastbare Candidate-to-action boundary mit expliziter non-executing Action-Proposal-Semantik. Dadurch bleiben Candidate, Proposal und executed action sauber getrennt, ohne automatische Ausführung, Compute-Trigger oder Memory-Commit-Seiteneffekte.
