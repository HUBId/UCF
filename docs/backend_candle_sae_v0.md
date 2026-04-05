# Backend Candle SAE V0 (canonical runtime seam)

Status: active in `runtime/ucf-compute` as a canonical SAE-stage adapter.

## Runtime binding

- Candle SAE execution is wired through the same canonical pipeline used by Burn:
  `backend_pack.rs` -> `backends/candle_backend.rs` -> `pipeline.rs`.
- SAE execution is backed by `stage_v1_candle::CandleSaeAdapterV0`.
- No Candle-only result schema exists; output is reported through
  `ComputeSignals` and `CanonicalPipelineResult`.

## Contract and failure semantics

- Candle SAE uses stage contract v1 and emits canonical failures on the top-level
  pipeline contract.
- Distinguishable runtime outcomes include:
  - Candle backend disabled (feature/runtime gate)
  - artifact unavailable / verification failure / incompatible
  - stage unavailable (adapter cannot execute despite verified slot)
  - contract mismatch
  - hard execution error
  - degraded-but-usable output path

## Honest readiness

- Real path available: SAE stage can execute deterministic top-k from verified
  candle safetensors via `CandleSaeAdapterV0`.
- Scope intentionally narrow: only canonical SAE stage behavior is integrated;
  no second parallel Candle pipeline is introduced.
