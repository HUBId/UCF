# Determinism Policy

## Canonical float policy

Digest-critical floating point values MUST be canonicalized/quantized before hashing.

- `canonicalize_f32`:
  - normalizes `-0.0` to `+0.0`
  - maps all `NaN` values to one canonical quiet NaN bit pattern
- Unit-domain metrics (`[0,1]`) are encoded as `u16` via `quantize_unit(x, 65535)`.
- Signed unit-domain state values (`[-1,1]`) are encoded as `i16` via `quantize_signed_unit_i16`.

Raw floats can still be stored in records for readability, but digest inputs use quantized integer forms.

## Quantization by signal type

- Unit signals (`risk`, `confidence`, `surprise`, `pressure`, `coherence`, `instability`, `uncertainty`, `stability`, spike magnitude): `u16` quantization.
- State vectors (`world`, `ssm`, `lfm` latent state): `i16` quantization.

## Ordering and reductions

- All digested collections must use explicit deterministic ordering.
- Spike ordering is stable and key-driven (`timestamp`, `feature_id`, then canonicalized magnitude when needed).
- Reduction loops for digest-critical values are sequential and index-ordered.
- No digest-critical algorithm uses parallel reductions.

## Schema/versioning guidance

Schema/version SHOULD be bumped when digest semantics change.

In this hardening pass:
- compute summary chain encoding moved to quantized float encoding (`COMPUTE_SUMMARY_SCHEMA_VERSION=2`)
- LFM summary/window records use quantized digest encoding and were bumped to `schema_version=2` at emission.

## Drift debugging workflow

1. Re-run replay fixture checks.
2. Compare digests at stage boundaries (`world`, `spikes`, `ssm`, `lfm`, `compute_chain`).
3. Confirm float-bearing digest fields are quantized at the callsite.
4. Validate spike ordering and seed stability.

## Build/CI constraints

- `cargo fmt --check` and strict `clippy -D warnings` are required.
- Property-style and mini-fuzz deterministic tests run offline.
- CI also runs replay checks to catch deterministic drift early.

## Fixed-point safety scalars (v2)
Safety-critical scalar path now has integer contract values (`*_q`, `u16` UQ0.16) for digest/policy/replay. Float values remain display-only and non-normative for these signals.
