# GPU Lane v1.1

## Feature flags
- `gpu-cuda` (Linux) enables optional CUDA capability plumbing.
- `gpu-metal` (macOS) enables optional Metal capability plumbing.
- CPU-only runs remain default and fully functional.

## Runtime modes
Set `UCF_GPU_MODE=off|shadow|active`.
- `off` (default): no GPU path.
- `shadow`: CPU stays decision source; GPU result is compared for parity.
- `active`: GPU can become primary only when parity checks pass.

## Parity gate (EBM stage v1)
- CPU reference output is always computed.
- GPU output is compared on fixed-point `energies_q` scalars.
- Allowed tolerance: `±1` LSB.
- On mismatch, runtime falls back to CPU and records parity evidence.
- Policy path remains fixed-point; no float digest gating.

## Resource caps
Environment knobs:
- `UCF_GPU_MAX_VRAM_BYTES` (default `536870912`)
- `UCF_GPU_MAX_KERNEL_MICROS` (default `100000`)

If estimated caps are exceeded, GPU is disabled and CPU fallback is enforced.

## Audit records
ESS receives GPU-specific records:
- `GpuUnavailableRecord`
- `GpuParityRecord`
- `GpuResourceViolationRecord`

These can be inspected with existing ESS/audit tooling.

## Local run (shadow)
```bash
UCF_GPU_MODE=shadow UCF_GPU_AVAILABLE=1 cargo test -p ucf-runtime --features "compute-candle,gpu-cuda"
```

Without hardware, lane falls back safely and emits unavailability records.
