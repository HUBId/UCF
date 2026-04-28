# Serie BB19 Prompt 3: deferred/blocked/priority contract boundary cleanup

Status: Die BB19 Runtime/Selection-Contract-Linie bleibt **operativ bounded** und **advisory-only**. Diese Stufe trennt priority hints, deferred state und blocked state explizit und schließt non-canonical/internal-only coupling paths als operative Shortcut-Pfade aus.

## Kanonische deferred/blocked/priority Boundary-Map

Kanonische Boundary-States (ohne neue Priority-/Planner-Engine):

- `priority_advisory_hint`
- `deferred_contract_state`
- `blocked_contract_state`
- `caveated_priority_deferred_blocked_signal`
- `insufficient_contract_basis_boundary_state`
- `non_canonical_internal_only_coupling_path_boundary_state`

## Runtime/Selection Contract-Signale (bereinigt)

Explizite Contract-Signale:

- `runtime_to_selection_advisory_signal`
- `runtime_to_selection_deferred_signal`
- `runtime_to_selection_blocked_signal`
- `selection_to_runtime_advisory_state`
- `selection_to_runtime_deferred_state`
- `selection_to_runtime_blocked_state`
- `caveated_contract_signal`
- `insufficient_contract_basis`
- `non_canonical_internal_only_contract_path`

## Harte Boundary-Regeln

- priority bleibt advisory-only (`priority_advisory_hint_only_no_direct_selection_authority`).
- deferred ist bounded Aufschub und bleibt getrennt von blocked.
- blocked ist Contract-/Safety-/Reference-Grenzzustand und bleibt getrennt von niedriger Priorität.
- non-canonical/internal-only coupling paths bleiben ausgeschlossen.
- Dynamics-/Execution-/Reference-Signale liefern nur bounded Contract-Basis.

## Explizit ausgeschlossene Scope-Erweiterungen (no-direct-*)

- kein direct action execution
- kein direct retry orchestration
- kein direct compute invocation
- keine implizite memory persistenz
- keine Planner-/Policy-/Agenten-Autoritätserweiterung

## Ergebnis

Die Runtime/Selection-Kopplung hat jetzt eine explizite, kanonische Boundary zwischen priority advisory hint, deferred und blocked. Dadurch bleibt der Abschluss-Sweep in BB19 auf einer bereinigten Contract-Grenze aufsetzbar, ohne zweite operative Contract-Wirklichkeit.

## BB20 Anschluss

Diese BB19-Contract-Boundary bleibt in BB20 unverändert wirksam und wird dort nur als repo-weite Readiness-Klasse konsolidiert (`usable with caveats`).

