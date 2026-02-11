# Compute Budget Profiles (v0)

`UCF_COMPUTE_BUDGET_PROFILE` steuert deterministische Work-Unit-Budgets im Compute-Pipeline-Backend.

## Profile

- `default` (`profile_id=1`): großzügig, normalerweise `VerifiedPipeline`.
- `tight` (`profile_id=2`): mildes Degrade-Verhalten möglich.
- `stress` (`profile_id=3`): löst reproduzierbar Budget-Überschreitungen (aktuell `sae/extract`) aus.

## Semantik

- Budgeting ist **deterministisch** über Work Units (`WorkMeter`) implementiert.
- Stage-Labels für Audit und Tests:
  - `world_model/predict`
  - `sae/extract`
  - `ssm/step`
  - `candle/forward` (falls verwendet)
- `DegradePolicy::DegradeStages`:
  - Stage-Fallbacks werden verwendet,
  - `RiskSignal.quality = DegradedFallback`.
- `DegradePolicy::FailFast`:
  - Compute wird `Unavailable` (`risk=1`, `confidence=0`).

## Audit / Persistenz

`Decision.compute_summary` enthält:

- `budget_profile_id`
- `budget_exceeded_stage` (falls Budget überschritten)
- `risk_quality`

Zusätzlich wird ein deterministisches Backpressure-Signal im Orchestrator berechnet (`bp` in `[0,1]`).
Bei `bp > 0.8` wird auf `compute_backpressure` defer-gated.

## Stress-Mode ausführen

```bash
UCF_COMPUTE_BUDGET_PROFILE=stress cargo test -p ucf-runtime runtime_v0::stress_profile_sets_budget_stage_and_backpressure_gating
```
