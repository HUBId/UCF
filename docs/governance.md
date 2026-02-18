# Liquid Governance v0

This document defines deterministic capability issuance governance for tool access.

## Governor score

Per tick, a bounded `GovernanceSignals` bundle is computed from compute, NSR, coherence, LFM,
and hormone stress signals. Missing signals are fail-safe defaults.

`G = clamp(0.35*nsr_risk_or_risk + 0.20*(1-coherence) + 0.20*instability + 0.15*lfm_uncertainty + 0.10*hormone_stress, 0..1)`

## Issuance tiers

- Tier 0 (`G < 0.25`): normal least-privilege issuance.
- Tier 1 (`0.25 <= G < 0.5`): reduced quotas and narrower issuance.
- Tier 2 (`0.5 <= G < 0.75`): safe-only issuance (internal-safe capabilities).
- Tier 3 (`G >= 0.75`): deny-by-default issuance for tool capabilities.

`ProcessExec` remains denied in v0.

## Deterministic throttling

`ToolGovernor` maintains per-capability bounded token buckets and cooldown counters:

- bounded capacity and refill per tick,
- escalating cooldown on repeated denies,
- bounded deny counters,
- deterministic tick update and digest.

This state is updated once per tick and is replayable from issuance decisions.

## Audit and replay

Issuance emits auditable ESS records:

- `CapabilityIssuance` with requested/granted/denied kinds, tier, governor score quantization,
  governance signal digest, throttle digest, evidence chain digest, schema version.
- `Throttle` snapshots per capability kind with tokens/cooldown/deny count and digest.

All tool actions still require explicit issuance and ToolGate authorization.

## Safe tuning guidance

- Keep all risk-like inputs clamped to `[0,1]`.
- Keep token capacities/cooldowns bounded and small.
- Use conservative defaults when inputs are absent (`risk=1`, `confidence=0`, no optional trust signal).
- Never allow `All` scopes or unbounded quotas.

## Fixed-point governor path
Governor input signals carry parallel fixed-point fields (`risk_q`, `coherence_q`, `instability_q`, `lfm_uncertainty_q`, `hormone_stress_q`), digest commits these quantized values, and governor scoring/tiering uses fixed-point arithmetic.


## EBM penalty (tightening-only)

Der Governor-Score enthält optional eine EBM-basierte Zusatzstrafe (`ebm_penalty_q`).

- Aktiv nur oberhalb eines hohen Energieschwellwerts (`E_HIGH`).
- Additiv und fixed-point quantisiert.
- Sicherheitsdominanz: EBM erhöht nur den Score/Tier (strenger) oder wirkt neutral.
- In Emergency wird das Signal ignoriert (deny-all bleibt dominant).
