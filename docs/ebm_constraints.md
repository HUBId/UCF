# EBM Constraint Library v1

Die EBM-Constraint-Library liegt in `policies/bundle_v1/ebm_constraints.toml` und ist über den Bundle-Hash fixiert.

## Bearbeiten

1. `[[terms]]` in aufsteigender `id`-Reihenfolge pflegen.
2. Nur bekannte `kind`-Werte verwenden:
   - `ToolIntentPenalty`
   - `CapabilityForbidden`
   - `CapabilityHighRisk`
   - `ContextRiskAmplifier`
   - `EmergencyDenyAllBias`
   - `OutputClassMismatch`
   - `BudgetExhaustedBias`
3. Danach `policies/manifest.toml` neu hashen.

## Explain-Tick Interpretation

`ucf-ops explain-tick` zeigt im EBM-Block:
- `aggregate_energy_q`
- `base_energy_q`
- `top_term_contributions`: Tupel aus `(term_id, label, contribution_q)`.

Damit sind Constraint-Beiträge begrenzt (Top-8) und nachvollziehbar.
