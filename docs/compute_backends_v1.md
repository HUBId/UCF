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
- Loader API: `load_safetensors_raw(slot, bytes, spec)` (CPU only, backend-agnostic).
- Input is local bytes only (typically from `ModelStore::verify_slot` + `read_verified_bytes`).
- Strict validation: all required tensors must exist with exact shape and dtype.
- Deterministic loading order: tensors are kept in `BTreeMap<String, LoadedTensorRaw>` (name sorted) and optionally converted to Candle tensors.
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

## Minimal artifact warmup/readiness semantics
- `ModelStore::warmup_slot_paths(slot)` now provides a narrow readiness view per path (`active`, `candidate`, `compare`, `shadow`, `blocked`).
- Canonical warmup states:
  - `cold`: not prefetched yet (or optional path not configured).
  - `prepared`: hash/path verified and available for rollout path probing.
  - `warm`: active slot bytes were verified and prefetched via `read_verified_bytes`.
  - `blocked`: warmup/readiness cannot proceed (verification/path/hash failure).
  - `stale`: reserved for future refresh signaling (not yet promoted to policy gating).
- Backend pack provenance includes rollout + warmup detail in a single deterministic detail string (`rollout=...;warmup=...`), so admission/placement/ops surfaces can distinguish cold vs prepared/warm vs blocked without introducing a separate lifecycle platform.
- Runtime ops snapshots expose a compact warmup view per slot (`cold`, `preparing`, `ready`, `blocked`, `unknown`) derived from that canonical provenance detail.
- Job history persists per-slot warmup state (best-effort extraction) so cold-start vs warm/ready effects can be correlated with execution timing/hotspot summaries.
- Intentional boundary: this is a slim readiness seam for active/candidate/compare/shadow paths, not a global cache manager or preload orchestrator.

## Activation / fallback / rollback transition semantics (Serie B hardening)
- Canonical activation assessment is now explicit via `ModelStore::assess_slot_activation(slot, target_hash, contract_version)`.
- Canonical activation outcomes are intentionally narrow:
  - `pending`: target is technically valid, but not yet the active path.
  - `succeeded`: target hash is already the active verified path (or active reference).
  - `degraded`: activation is technically possible, but compare/shadow baseline diagnostics are incomplete or degraded.
  - `blocked`: activation preconditions are explicitly blocked (verification/contract/path/promotion blockers).
  - `failed_technically`: activation preconditions passed, but target artifact warmup/prefetch failed.
- Fallback and rollback are explicitly distinct:
  - `fallback_to_prior_active`: runtime-near safety return to the prior active verified hash when activation is blocked/failed/degraded.
  - `rollback`: explicit return assessment (`ModelStore::assess_slot_rollback`) to a prior/target hash with outcomes `completed|unavailable|failed`.
- Prior active reference is carried in both assessments (`prior_active_hash`, `replaced_hash`, `resulting_active_hash`) to keep rollback/fallback traceability load-bearing without introducing a release-management state machine.
- Backend slot provenance keeps rollout transition detail in one deterministic string (`rollout=...;activation={...};rollback={...}`) so ops/history/promotion consumers can observe:
  - attempted activation target
  - activation outcome (success/degraded/blocked/failed/pending)
  - fallback usage
  - rollback readiness/resulting active hash.
- Intentional boundary: still no auto-promotion and no automatic operator workflow; this remains a slim technical transition seam.


## Envelope checks / fail-safe
- Any `NaN`/`Inf` during JEPA/SAE/SSM yields deterministic degraded fallback (`StageQuality::DegradedFallback`).
- Degraded outputs are conservative: JEPA error/surprise = 1, SAE empty spikes + energy 0, SSM pressure/readout = 1.
- Degraded digests use deterministic markers and include `(t, context/evidence digest)` for replay compatibility.
- Outputs are bounded in v1 (`spike_count <= SAE_TOP_K`, state values clamped to `[-1,1]`, unit metrics clamped to `[0,1]`).
