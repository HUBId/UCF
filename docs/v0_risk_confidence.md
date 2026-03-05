# v0 Risk/Confidence Contract

`RiskConfidenceV1` stabilizes decision-facing risk and confidence as fixed-point UQ0.16 in `[0,1]`.

## Contract

- `risk_q: UQ0_16`
- `confidence_q: UQ0_16`
- `update_digest: [u8;32]`

Update is deterministic per tick:

- `risk_q = clamp(base_risk_q + alpha_q*surprise_q + beta_q*pressure_q)`
- `confidence_q = clamp(1 - gamma_q*uncertainty_q)`

All coefficients are loaded from policy thresholds:

- `risk_confidence_base_risk_q`
- `risk_confidence_alpha_q`
- `risk_confidence_beta_q`
- `risk_confidence_gamma_q`

Defaults are conservative and bounded if keys are absent.

## FEP wiring

FEP/Active Inference now consumes `SignalBundleV1` and `RiskConfidenceV1` through `FepInputs`, using bundle signals (`surprise_q`, `pressure_q`, etc.) as primary sensory summary.

## ESS artifact

Each tick emits `DecisionInputsRecordV1` before `DecisionFrame` append. The record is bounded/redaction-safe:

- digest prefixes only
- quantized risk/confidence
- governor tier and compact gating status
- tool-intent booleans without plan payload
