# Model Governance v1

Model governance evaluates slot-local health in deterministic windows and only **tightens** runtime behavior.

## Window stats

Per governance window (default `512` ticks), per slot (`llm`, `world`, `sae`, `ssm`, `lfm`, `ebm`):

- active mode (`toy`/`shadow`/`active`) from slot enablement config.
- timeout rate.
- invalid rate.
- envelope violation rate.
- mean uncertainty delta vs baseline.
- mean pressure delta vs baseline.

Baselines are updated only while slot mode is `toy`.

## Policy thresholds

Thresholds are loaded from policy packs as:

- `model_governance_<slot>_max_timeout_rate_q`
- `model_governance_<slot>_max_invalid_rate_q`
- `model_governance_<slot>_max_envelope_violation_rate_q`
- `model_governance_<slot>_max_delta_uncertainty_q`
- `model_governance_<slot>_max_delta_pressure_q`

Quantized `q` values are deterministic (`u16` / unit interval).

## Runtime actions (tightening only)

On threshold breach:

1. slot status becomes `Degraded`.
2. `model_governance_alarm` note is emitted with bounded reason codes.
3. governor tier is tightened for critical slots (`llm`, `ebm`).
4. tool issuance is tightened (critical degraded => deny issuance).

On persistent breach streak (N windows):

- emit `model_rollback_recommendation` note.
- recommendation is advisory in prod; human operator executes rollback.

## Rollback safety

- No automatic promoted-weight rollback is performed.
- Runtime may tighten and fallback behavior, but promoted rollback remains operator-driven.

## Explainability / audit

- Tool issue audit carries model governance digest prefix and reason codes.
- ESS notes include alarm and recommendation reasons for forensic review.
