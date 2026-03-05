# v0 Mock Dynamics (JEPA/SAE/SSM/LFM) and SignalBundleV1

This document defines the deterministic v0 mock coupling path:

`JEPA(world) -> SAE(spikes) -> SSM(pressure) -> LFM(uncertainty/stability) -> SignalBundleV1`

## Deterministic equations and bounds

- All scalar signals are `UQ0_16` compatible (`[0, 1]` domain).
- SAE spike count is capped (`MAX_SAE_SPIKES`).
- JEPA mock prediction vector is bounded to fixed dimension (`D=16`).
- No online learning occurs; updates are deterministic and input/state bound.

### JEPA / world mock

Inputs:
- `context_digest`
- `previous_world_state_digest` (optional)
- `signal_q`

Outputs:
- `prediction_q: [i16; 16]`
- `prediction_error_q`: mean absolute delta against previous digest-derived values
- `surprise_q`: weighted coupling of `prediction_error_q` and context novelty
- `prediction_digest`

### SAE mock

Inputs:
- `context_digest`
- `prediction_digest`
- `top_k`

Outputs:
- deterministic feature spikes derived from digest bytes
- sorted ascending by `feature_id`
- stable tie handling through deterministic sorting
- `spikes_digest`

### SSM mock

Inputs:
- `spikes_digest`
- `spike_count`
- `surprise_q`
- previous pressure and previous state digest

Update:
- deterministic bounded blend of previous pressure, surprise, and normalized spike count

Outputs:
- `pressure_q`
- `state_digest`

### LFM mock

Inputs:
- `pressure_q`, `surprise_q`, `risk_q`, `previous_lfm_digest`

Outputs:
- `uncertainty_q = weighted_sum(pressure, surprise, risk)`
- `stability_q = 1 - k * uncertainty`
- `lfm_digest`

## SignalBundleV1 contract

`SignalBundleV1` is the canonical v0 bundle for FEP / active inference and later decision flow consumption.

Fields:
- `risk_q`
- `confidence_q`
- `surprise_q`
- `pressure_q`
- `uncertainty_q`
- `stability_q`
- `coherence_q` (optional)
- component digests:
  - `world_prediction_digest`
  - `sae_spikes_digest`
  - `ssm_state_digest`
  - `lfm_state_digest`

Digest:
- `signals_digest = SHA256(canonical ordered encoding of all fields + t + policy_graph_digest)`
- fixed ordering and explicit option marker for `coherence_q`
- big-endian scalar encoding in digest path for canonical stability

## Persistence

In the runtime control loop, a per-tick `SignalBundleRecordV1` is persisted with:
- quantized scalars
- component digest prefixes
- `signals_digest_prefix`
- policy/evidence digest prefixes

This enables deterministic replay checks without payload bloat.

## FEP consumption and stabilized risk/confidence

Runtime tick wiring now updates `RiskConfidenceV1` from `SignalBundleV1` before decisioning and feeds that pair into FEP/active-inference inputs. `DecisionInputsRecordV1` is appended to ESS immediately before `DecisionFrame` so replay/postmortem can inspect exactly what decision logic saw, without raw payloads.
