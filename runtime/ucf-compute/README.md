# ucf-compute v0 pipeline

Deterministic offline compute pipeline used by the runtime.

## Capability model

The top-level runtime contract stays `AiComputeBackend`, but concrete backends are now composed from stable subtraits in `src/capabilities.rs`:

- `WorldModelPredictor` (JEPA/world prediction)
- `FeatureExtractor` (SAE/sparse spike extraction)
- `WorkingMemoryModel` (SSM/selective scan memory)
- `LlmInference` + `LlmOutput` placeholder (defined for future policy/model integration)

`ComputePipelineBackend` orchestrates these capabilities with bounded deterministic degradation.

## Profile mapping (factory)

`build_backend` wires `ComputeBackendKind` into runtime packs:

- `stub`: `BackendPackKind::ToyV1`
- `candle` (`--features compute-candle`): `BackendPackKind::CandleToyV1`
- `burn` (`--features compute-burn,backend-burn`): `BackendPackKind::BurnToyV1`

### Canonical onboarding lane (single reference path)

- The canonical onboarding entrypoint is now pinned to Burn via
  `build_onboarding_reference_backend` (`CANONICAL_ONBOARDING_PACK = BurnToyV1`).
- Canonical request/result/failure contract stays:
  `CanonicalPipelineRequest -> ComputePipelineBackend::compute_canonical -> CanonicalPipelineResult|CanonicalPipelineFailure`.
- Canonical stage sequence is fixed as `World -> SAE -> SSM -> LFM`
  (`CANONICAL_STAGE_SEQUENCE`), with honest runtime state per stage:
  - required productive core: `World`, `SAE`, `SSM`;
  - `LFM` runs when Burn LFM runtime is enabled (`lfm-burn`), otherwise backend init is
    explicitly blocked (`BackendDisabled`) — no silent fallback lane.
- `NSR` remains an optional attachment and is surfaced explicitly in `CanonicalPipelineResult.nsr_stage`
  (`disabled`, `used`, `contract_mismatch`, `verification_failed`, etc.).
- `Candle` remains a compatibility seam and is **not** a second onboarding default path.

See also `docs/compute_onboarding_reference_path.md` for the compact readiness matrix.

## Backend selection (runtime)

The orchestrator can be bootstrapped from env config via `RuntimeOrchestrator::try_new_from_env`.

- `UCF_COMPUTE_BACKEND=stub|candle|burn`
- `UCF_COMPUTE_SEED=<u64>`
- `UCF_COMPUTE_MAX_MICROS=<u64>`
- `UCF_COMPUTE_HARD_TIMEOUT_MICROS=<u64>`

Default remains `stub` when env vars are unset.

## Candle feature extractor v0 (offline dummy weights)

`compute-candle` enables `CandleFeatureExtractor`, which performs a deterministic forward pass (`32 -> 64`) on CPU-only candle tensors using inline dummy weights.

- No HTTP, no model downloads, no external fixture pulls.
- Input vector is derived from `ComputeInput.context_digest` + world prediction digest.
- Reductions (`top-k`, sparsity, energy) are done in Rust over `Vec<f32>` for deterministic ordering.

## Offline fixture policy and constraints

- No network and no model-weight download.
- Output deterministic from `(context_digest, seed, t)`.
- Bounded outputs: capped spikes/notes and digest-only persistence for large vectors/state.

## Model manifest source

- Canonical manifest path: `models/manifest.toml`.
- Override path only via `UCF_MODEL_MANIFEST` when explicit compatibility behavior is required.

## Adding future backends

To add a real backend later without refactoring orchestrator/frame contracts:

1. Implement one or more capability traits (`WorldModelPredictor`, `FeatureExtractor`, `WorkingMemoryModel`).
2. Register capability wiring in `build_backend` for a profile.
3. Keep `AiComputeBackend` entrypoint unchanged by returning `ComputePipelineBackend`.
