# Serie BB21 Prompt 1: execution/reference interaction hardening

Status: **narrow cross-line hardening** zwischen echter Execution, kanonischer Result-Referenz und bounded downstream consumption.

## Kanonische Execution→Reference→Consumption-Linie

1. **Execution result only**
   - Rohes Execution-Ergebnis ist noch keine starke Konsumbasis.
2. **Canonical result reference**
   - Nur `execution result: completed` + `validity=current` gilt als kanonische starke Result-Referenz.
3. **Bounded reference consumption**
   - Downstream-Konsum bleibt advisory/candidate-bounded und ohne direkte Autoritätsausweitung.

## Explizite Nicht-Erfolgsbasis

Diese Klassen bleiben getrennt von kanonischer starker Result-Basis:
- `failed`
- `cancelled`
- `blocked`
- `unavailable`
- `unsupported`
- `placeholder_only` / `not_execution_result`

Für diese Basis gilt:
- nur caveated/candidate-only Konsum,
- kein direkter Execution-Folgepfad,
- keine implizite Retry-/Memory-/Compute-/Action-Autorität.

## Non-canonical/internal-only Pfade

`non_canonical_internal_only_*` bleibt ein dedizierter Transition-Pfad und ist kein kanonischer bounded consumption path.

## No-direct-* Grenzen bleiben unverändert bindend

- kein direct retry
- keine direkte Folge-Execution
- keine implizite Memory-Persistenz
- keine Compute Invocation aus Referenzkonsum
- keine Reasoning-/Agent-/Policy-Autorität aus dieser Linie
