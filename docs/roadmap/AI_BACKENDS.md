# AI backend roadmap (T102 next phase)

This repository now uses a strict `ModelHost` ABI via `ucf-ai-host-abi` so runtime backends can be swapped without destabilizing the coherence loop.

## Current state

- Default runtime path uses `MockBackend` (`ucf-ai-host-abi`) through `AiHostRuntime` (`domains/ai`).
- `ucf-ai-backends` contains feature-gated adapter placeholders for Candle and Burn.
- No Candle/Burn dependencies are enabled by default.

## Feature flags

At the workspace root, opt into future backends with:

- `ai-candle`
- `ai-burn`

Example:

```bash
cargo test -q --features ai-candle
```

## What Candle/Burn adapters must implement

1. **Tensor I/O boundary**
   - Map `AiHostAbiInput` scalars + commit references to model input tensors.
   - Convert model outputs into bounded ABI vectors (`feature_events`, `output_candidates`, `internal_thoughts`).
2. **SAE hooks**
   - Surface sparse-activation pathways and commit traces required by downstream diagnostics.
3. **Lens hooks**
   - Expose interpretability checkpoints for lens readouts across LFM/RLM blocks.
4. **Commit coherence**
   - Preserve deterministic `abi_commit` semantics so router/coherence paths remain auditable.

## Integration guideline

Until tensor runtimes are wired, keep adapters compile-only and deterministic stubs so default CI (`cargo test -q`) remains stable.
