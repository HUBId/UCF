# Compute Backends v1

## Scope
CPU-only real compute adapters for `world_jepa`, `sae`, and `ssm` in backend packs (`candle_toy_v1`/`burn_toy_v1`) using existing traits and model slots.

## Module contracts
- **World/JEPA** (`WorldModelPredictor`):
  - input: `WorldModelInput { obs_features[16], context_digest, t, seed }`
  - output: `WorldModelOutput { prediction_digest, state_digest, prediction_error, surprise, state_norm }`
  - forward (v1): `h = tanh(x·W1 + b1)`, `y = h·W2 + b2` (CPU deterministic loops).
  - error: `mean_abs(y-x)` clamped to `[0,1]`.
  - digesting: `prediction_digest = H(q_i16(y_clamped) || model_hash || t || context_digest)` and `prediction_error_q = quantize_unit_u16(error)`.
- **SAE** (`SaeExtractor`):
  - input: `SaeInput { context_features[32], world_state_digest, t, seed }`
  - forward: `y = W x + b` (CPU deterministic loops).
  - sparsify: deterministic top-k by `|y_i|`, tie-break by `feature_id`.
  - output: spikes + `spikes_digest`, `spike_count`, `energy`, `sparsity` where digest uses canonical sorted spikes + quantized magnitudes + model/context metadata.
- **SSM** (`SsmKernel`):
  - input: `SsmInput { spikes_digest, spike_count, sae_energy, world_surprise, ... }`
  - forward: structured scan update in ascending index order: `s[i] = A[i]*s[i] + B[i]*inp`, clamped to `[-1,1]`.
  - output: `pressure`, `state_digest`, `readout_digest`, `state_norm`, `readout`, with state digest over quantized signed state + model hash + tick/context digests.

## Model provisioning (offline/hash-locked)
Use `models/manifest.toml` with slots:
- `world_jepa`
- `sae`
- `ssm`

Each slot is verified via `ModelStore::verify_slot` (allowlist path + SHA-256 lock + max bytes + cpu-only device).

## Enablement / shadow rollout
- Global mode: `UCF_REAL_ENABLEMENT_MODE=off|shadow|compare|active`
- Per-slot mode:
  - `UCF_SLOT_WORLD_JEPA_MODE=toy|shadow|active`
  - `UCF_SLOT_SAE_MODE=toy|shadow|active`
  - `UCF_SLOT_SSM_MODE=toy|shadow|active`
- Shadow cadence: `UCF_SHADOW_EVERY_N_TICKS`

## Probes / checks
Run deterministic compute probe paths by building ucf-compute tests or `ucf-ops models probe` flow with fixed seed + manifest.

## Budget tuning
- Budgets stay enforced by stage (`world_model/step`, `sae/extract`, `ssm/step`) through `ComputeBudget` and work meters.
- Suggested rollout: shadow first, compare envelopes, then active slot-by-slot.

## Candle safetensors loader behavior
- Loader API: `load_safetensors(slot, bytes, spec)` (CPU only).
- Input is local bytes only (typically from `ModelStore::verify_slot` + `read_verified_bytes`).
- Strict validation: all required tensors must exist with exact shape and dtype.
- Deterministic loading order: tensors are kept in `BTreeMap<String, Tensor>` (name sorted).
- Bounded memory: bytes larger than slot `max_bytes` are rejected before parsing.
- Stable error codes:
  - `WEIGHT_MISSING_TENSOR`
  - `WEIGHT_SHAPE_MISMATCH`
  - `WEIGHT_DTYPE_MISMATCH`
  - `WEIGHT_TOO_LARGE`
  - `WEIGHT_PARSE_ERROR`
  - `WEIGHT_HASH_MISMATCH`
- Any validation failure must fail safe and map to `ComputeError::BackendDisabled`.

## Offline fixtures
Golden/negative safetensors fixtures are generated in unit tests at runtime (offline, deterministic) to keep the repository text-only and VCS-compatible.


## Envelope checks / fail-safe
- Any `NaN`/`Inf` during JEPA/SAE/SSM yields deterministic degraded fallback (`StageQuality::DegradedFallback`).
- Degraded outputs are conservative: JEPA error/surprise = 1, SAE empty spikes + energy 0, SSM pressure/readout = 1.
- Degraded digests use deterministic markers and include `(t, context/evidence digest)` for replay compatibility.
- Outputs are bounded in v1 (`spike_count <= SAE_TOP_K`, state values clamped to `[-1,1]`, unit metrics clamped to `[0,1]`).
