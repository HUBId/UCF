# SSM Kernels v1.1

## Kernel modes

`UCF_SSM_KERNEL=ref|shadow|opt`

- `ref`: canonical baseline kernel (`ssm_ref_kernel`).
- `shadow`: runs ref + opt, keeps ref as primary, computes drift (`pressure_delta_q`, digest equality), can disable opt.
- `opt`: runs optimized kernel, but automatically falls back to ref if a previous drift alarm disabled opt.

## Determinism and reduction ordering

- Both kernels keep identical loop/index order over the bounded state dimension (`SSM_STATE_DIM=32`).
- Optimized path only unrolls element-wise state updates; reductions (`readout`, absolute sum) remain deterministic.
- Pressure path is quantized deterministically via fixed-point-like accumulation (`abs_sum_q15`) and final `quantize_unit_u16`.
- State digest is always computed from quantized state (`quantize_signed_unit`) with fixed hasher field ordering.

## Drift gating and alarms

Shadow mode computes:

- `pressure_delta_q = |pressure_q(opt) - pressure_q(ref)|`
- `digest_mismatch = state_digest(opt) != state_digest(ref)`

Policy threshold:

- `SSM_DRIFT_PRESSURE_DELTA_MAX_Q` (current default: `2`)
- digest mismatch treated as immediate drift

If drift is exceeded:

- optimized kernel is disabled (`opt_enabled=false`)
- runtime falls back to ref in `opt` mode
- `SsmKernelDriftAlarmRecord` is persisted (if log path configured)

## Records and interpretation

Set paths:

- `UCF_SSM_KERNEL_RECORDS_LOG=/path/ssm_records.jsonl`
- `UCF_SSM_KERNEL_ALARMS_LOG=/path/ssm_alarms.jsonl`

Record schema:

- `SsmKernelRecord`
  - `kernel_id`: `ref|opt|shadow_ref_opt|ref_fallback`
  - `pressure_q`: quantized pressure
  - `state_digest_prefix`: first 8 bytes of digest
  - `drift_delta_q`: present for shadow samples

Alarm schema:

- `SsmKernelDriftAlarmRecord`
  - `pressure_delta_q`
  - `digest_mismatch`
  - `action=disable_opt_fallback_to_ref`

## Bench harness

Run:

```bash
cargo run -p ucf-compute --bin ucf-ssm-bench
```

The harness prints small/med/large iteration cases and `speedup = ref_ns / opt_ns`.
