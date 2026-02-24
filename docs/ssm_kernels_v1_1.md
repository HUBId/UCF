# SSM Kernels v1.1

## Modes

`UCF_SSM_KERNEL` controls kernel execution:

- `ref`: canonical reference kernel
- `shadow`: reference primary + optimized shadow drift check
- `opt`: optimized primary (with fail-safe fallback to ref once drift alarm triggers)

## Determinism

- Reference kernel is canonical for contract parity.
- Optimized kernel keeps update/readout loop ordering deterministic.
- Pressure is digest/policy-safe via deterministic `UQ0_16` quantization (`pressure_q`) and converted back to `pressure` for output.
- State digest remains bound to quantized signed state values.

## Drift gating

In `shadow` mode the runtime computes:

- `pressure_delta_q = |pressure_q(opt) - pressure_q(ref)|`
- digest equality between quantized ref/opt state digests

If envelope fails (`pressure_delta_q` above threshold or digest mismatch), optimized kernel is disabled and runtime continues with ref.

## Notes / records

SSM output notes include kernel execution metadata:

- `kernel_mode=...`
- `kernel_id=...`
- `kernel_id=shadow_ref_opt delta_q=...`
- `drift_alarm=1` (if drift envelope violated)
