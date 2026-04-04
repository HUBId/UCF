# AI backend roadmap (compatibility boundary)

## Canonical runtime boundary

- `runtime/ucf-compute` is the canonical backend execution path for real compute onboarding.
- `domains/ai-host-abi` defines host ABI contracts.
- `domains/ai` wraps ABI contracts for host-facing runtime usage.
- `domains/ai-backends` remains compile-time adapter scaffolding for host ABI compatibility.

## Current implementation status

- `domains/ai-backends` Candle/Burn backends are placeholder adapters returning bounded empty ABI outputs.
- Real compute-oriented Candle/Burn stage wiring currently exists in `runtime/ucf-compute` (deterministic CPU paths with guarded degradation behavior).

## Immediate rule

Treat `domains/ai*` as compatibility layer. Put canonical model-pipeline expansion work in `runtime/ucf-compute`.
