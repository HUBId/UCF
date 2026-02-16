# Emergency Mode (Liquid Stability v0)

Emergency Mode provides deterministic fail-safe behavior when LFM/LNN stability guards detect runaway dynamics.

## Stability metrics

Per tick, the runtime computes from LFM summary:

- `state_norm = mean_abs(x)`
- `deriv_norm = mean_abs(dx/dt)` (LNN) or `mean_abs(delta_x)` (toy)
- `V = state_norm^2 + 0.25 * deriv_norm^2`
- `dV = V - V_prev`
- `saturation_ratio = fraction(|x_i| near clamp)`

Thresholds:

- `V > 1.10` => runaway
- `dV > 0.06` for 3 consecutive ticks => trend instability
- `saturation_ratio > 0.20` => instability
- `NaN/Inf` => immediate emergency

All thresholds are fixed constants and replay-deterministic.

## State machine

States:

- `Inactive`
- `Armed { since_t, reason }`
- `Active { since_t, reason, cool_down_remaining }`

Rules:

- Runaway (`V`) and `NaN/Inf`: immediate `Active`
- Trend/saturation: `Armed` first, then `Active` after deterministic arm window
- `Active` persists for fixed cooldown ticks (32), then transitions `Off` only if stable

Transitions are persisted to ESS `EmergencyRecord` entries for audit/replay.

## Overrides while Active

When `Active`:

- Plasticity is force-disabled (`emergency_disabled=true` on plasticity record)
- Tool issuance is force-overridden to effective tier `3` (deny-all)
- Candidate selection is forced toward no-op/defer path
- Output policy is forced to safe-only and `max_tokens_eff <= 64`
- Emergency telemetry is emitted (`ucf_emergency_*`)

## Audit/replay interpretation

Use ESS records:

- `EmergencyRecord` for transitions (`armed`, `active`, `off`)
- `CapabilityIssuanceRecord.effective_tier` and `emergency_override`
- `PlasticityRecord.emergency_disabled`

These records include quantized stability values and digests (`lfm`, backend-pack, evidence chain) to allow deterministic replay comparison.
