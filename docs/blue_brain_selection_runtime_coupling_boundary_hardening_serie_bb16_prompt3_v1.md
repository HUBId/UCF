# Serie BB16 Prompt 3: selection/runtime coupling boundary hardening (advisory-only)

Diese Härtung schärft die Kopplungsgrenze zwischen bounded dynamics, Runtime, Selection und echter Execution weiter, ohne Autoritätsausweitung.

## Kanonische advisory coupling states

- `runtime_advisory_coupling`
- `selection_advisory_coupling`
- `caveated_advisory_coupling`
- `insufficient_advisory_coupling`
- `blocked_advisory_coupling`
- `ignored_advisory_coupling`
- `non_canonical_internal_only_coupling_path`

Diese Zustände sind diagnostisch/advisory-only und bleiben getrennt von realen Execution-Requests.

## Runtime/Selection Kopplung bleibt advisory-only

- Runtime darf dynamics-informed Hinweise als caveat/modulation hints beobachten.
- Selection darf dynamics-informed Hinweise als bounded advisory input beobachten.
- Caveated/insufficient/blocked/ignored bleiben explizit unterscheidbar.
- Unsuccessful execution basis (failed/cancelled/blocked/unavailable) wird nicht als successful execution-informed coupling behandelt.

## Harte Trennung gegen echte Execution

Explizit ausgeschlossen bleiben:

- kein direct action selection
- kein direct re-execute trigger
- kein direct retry trigger
- kein direct compute invocation
- kein direct proposal generation
- kein direct memory commit
- kein safety override

Damit bleibt execution-integrity intakt: dynamics-informed coupling informiert Runtime/Selection ausschließlich advisory-only.
