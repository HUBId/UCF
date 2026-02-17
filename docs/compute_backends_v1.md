# Compute Backends v1

## Scope
CPU-only real compute adapters for `world_jepa`, `sae`, and `ssm` in backend packs (`candle_toy_v1`/`burn_toy_v1`) using existing traits and model slots.

## Module contracts
- **World/JEPA** (`WorldModelPredictor`):
  - input: `WorldModelInput { obs_features[16], context_digest, t, seed }`
  - output: `WorldModelOutput { prediction_digest, state_digest, prediction_error, surprise, state_norm }`
  - digesting: quantized signed latent/state values.
- **SAE** (`SaeExtractor`):
  - input: `SaeInput { context_features[32], world_state_digest, t, seed }`
  - forward: `y = W x + b`, ReLU, deterministic top-k sparsify.
  - output: spikes + `spikes_digest`, `spike_count`, `energy`, `sparsity`.
- **SSM** (`SsmKernel`):
  - input: `SsmInput { spikes_digest, spike_count, sae_energy, world_surprise, ... }`
  - forward: selective scan update over state (O(T*N) minimal path).
  - output: `pressure`, `state_digest`, `readout_digest`, `state_norm`, `readout`.

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
