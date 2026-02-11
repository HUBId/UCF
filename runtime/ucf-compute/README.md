# ucf-compute v0 pipeline

Deterministic offline compute pipeline used by the runtime:

1. `world_model` (`MockJepaPredictor`) produces surprise and prediction digest.
2. `feature_extractor` (`MockSaeExtractor`) produces sparse spikes, sparsity, and energy.
3. `ssm` (`MockSsmSelectiveScan`) performs a selective-scan memory step and pressure/readout.
4. `fuse_signals` maps surprise + pressure + energy to scalar `risk`/`confidence` in `[0,1]`.

## Constraints

- No network and no model weights.
- Output is deterministic from `(context_digest, seed, t)`.
- Bounded outputs: capped spikes/notes and digest-only persistence for large vectors/state.

## Future backends

`compute-candle` and `compute-burn` features keep the same backend trait/summaries, so real
implementations can replace mock stages without changing runtime frame contracts.
