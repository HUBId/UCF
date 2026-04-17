# AI backend roadmap (compatibility boundary)

## Canonical runtime boundary

- `runtime/ucf-compute` is the canonical backend execution path for real compute onboarding.
- `domains/ai-host-abi` defines host ABI contracts.
- `domains/ai` wraps ABI contracts for host-facing runtime usage.
- `domains/ai-backends` remains compile-time adapter scaffolding for host ABI compatibility.
- Canonical lane classification and shared-core invariants are pinned in
  `runtime/ucf-compute/src/reference_map.rs` (`CANONICAL_COMPUTE_REFERENCE_MAP`).

## Current implementation status

- `domains/ai-backends` Candle/Burn backends are placeholder adapters returning bounded empty ABI outputs.
- Real compute-oriented Candle/Burn stage wiring currently exists in `runtime/ucf-compute` (deterministic CPU paths with guarded degradation behavior).
- `build_backend(kind=stub|candle)` remains a compatibility/dev constructor lane and is not a
  canonical production entry.
- `build_backend(kind=worker)` remains an internal worker lane, not a second production truth.

## Immediate rule

Treat `domains/ai*` as compatibility layer. Put canonical model-pipeline expansion work in `runtime/ucf-compute`.

## Alignment with status/readiness surfaces

- Status checkpoints and transition framing live in:
  - `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md`
  - `docs/roadmap/REAL_COMPUTE_TRANSITION.md`
- Readiness classification (stable core / production-usable but constrained / partial/diagnostic /
  intentionally deferred) lives in:
  - `docs/real_compute_readiness_sweep_v26.md`

This file stays a backend-boundary roadmap context; it must not redefine canonical runtime contracts.
