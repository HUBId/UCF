# Backend Candle World V0 (canonical runtime seam)


Status note: this Candle world document describes a bounded feature seam. It must not be read as a production-ready or general runtime-inference claim without a pinned local artifact-backed fixture and deterministic runtime golden test.
Status: active in `runtime/ucf-compute` as a canonical world-stage adapter.

## Runtime binding

- Candle world execution is wired through the canonical runtime backend path:
  `backend_pack.rs` -> `backends/candle_backend.rs` -> `pipeline.rs`.
- The world stage uses `stage_v1_candle::CandleWorldAdapterV0` (contract v1), not a
  separate pipeline contract.
- Candle and Burn both return through `CanonicalPipelineResult` and
  `CanonicalPipelineFailure`.

## Artifact and compatibility rules

- Slot verification still goes through `ModelStore::verify_slot`.
- Candle world requires `world_jepa` to be `format = "candle_safetensors"` and
  `contract_version = "v1"` (or `"1"`), enforced in `backend_pack`.
- Failures remain structured through canonical categories:
  - backend disabled
  - artifact unavailable / verification failed / incompatible
  - stage unavailable (adapter cannot execute even with verified slot)
  - contract mismatch
  - hard execution error

## Honest readiness

- Real path available: world stage can execute with verified candle weights via
  `CandleWorldAdapterV0`.
- Remaining blocker: if a verified slot cannot be executed by the adapter
  (unexpected tensor payload), runtime emits `stage_unavailable` instead of
  silently falling back.
