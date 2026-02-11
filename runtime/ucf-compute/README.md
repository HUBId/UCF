# ucf-compute v0 pipeline

Deterministic offline compute pipeline used by the runtime:

1. `world_model` (`MockJepaPredictor`) produces surprise and prediction digest.
2. `feature_extractor` (`MockSaeExtractor`) produces sparse spikes, sparsity, and energy.
3. `ssm` (`MockSsmSelectiveScan`) performs a selective-scan memory step and pressure/readout.
4. `fuse_signals` maps surprise + pressure + energy to scalar `risk`/`confidence` in `[0,1]`.

## Backend selection (runtime)

The orchestrator can be bootstrapped from env config via `RuntimeOrchestrator::try_new_from_env`.

- `UCF_COMPUTE_BACKEND=stub|candle|burn`
- `UCF_COMPUTE_SEED=<u64>`
- `UCF_COMPUTE_MAX_MICROS=<u64>`
- `UCF_COMPUTE_HARD_TIMEOUT_MICROS=<u64>`

Default remains `stub` when env vars are unset.

## Candle backend v0 (offline dummy weights)

`compute-candle` enables `CandleBackend`, which performs a tiny deterministic forward pass (`32 -> 16`) on CPU-only candle tensors using inline dummy weights in source (`src/backends/candle_backend.rs`).

- No HTTP, no model downloads, no external fixture pulls.
- Input vector is derived from `ComputeInput.context_digest` bytes.
- Reductions (`mean`, top-k spike selection) are executed in Rust over `Vec<f32>` after tensor extraction for stable deterministic ordering across runs.

## Constraints

- No network and no model weights download.
- Output is deterministic from `(context_digest, seed, t)`.
- Bounded outputs: capped spikes/notes and digest-only persistence for large vectors/state.

## Future backends

`compute-candle` and `compute-burn` features keep the same backend trait/summaries, so real
implementations can replace mock stages without changing runtime frame contracts.
