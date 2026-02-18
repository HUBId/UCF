# LFM × LNN ODE Core v1.1

This document describes the deterministic LFM ODE kernel (`LnnOdeLfmKernel`) used in compute.

## Equation

State dynamics per substep:

- `x ∈ R^N`, `u ∈ R^M`
- `dx/dt = -alpha ⊙ x + tanh(Wx*x + Wu*u + b)`

v1.1 bounds:

- `N <= 32`
- `M <= 8`
- fixed `dt`
- fixed `steps` per tick
- clamp `x` to `[-clamp_state, clamp_state]` each substep

## Integrator

v1.1 uses deterministic RK2 (midpoint), fixed-step only.

For each substep:

1. `k1 = f(x, u)`
2. `k2 = f(x + 0.5*dt*k1, u)`
3. `x <- clamp(x + dt*k2)`

No adaptive solver is used.

## Weight tensors (hash-locked, offline)

The LFM model slot (`ModelSlot::Lfm`) can provide safetensors with strict names and shapes:

- required `lfm.alpha`: `[N]` `f32`
- required `lfm.wx`: `[N,N]` `f32`
- required `lfm.wu`: `[N,M]` `f32`
- required `lfm.b`: `[N]` `f32`
- optional `lfm.x0`: `[N]` `f32`

Validation rules:

- exact tensor names
- exact shapes and dtype (`f32` only in v1.1)
- `N <= 32`, `M <= 8`
- slot hash must already be verified by `ModelStore::verify_slot` before load

Reference fixtures (hex-encoded safetensors bytes, to keep the repo text-only for PR tooling):

- `fixtures/weights/lfm_ode_v1_small.safetensors.hex`
- negative: `fixtures/weights/lfm_ode_v1_bad_shape.safetensors.hex`
- negative: `fixtures/weights/lfm_ode_v1_missing_tensor.safetensors.hex`

## Deterministic x0 init

If `lfm.x0` is not provided, `x0` is generated deterministically from:

- `UCF_RUN_ID`
- `UCF_POLICY_HASH`
- session seed
- domain separator `"lfm_x0"`

A deterministic xorshift stream maps entries into `[-0.1, 0.1]`, then clamps by `clamp_state`.

## Safety envelope and fallback

The kernel tracks rolling instability metrics:

- clamp saturation ratio
- delta norm `mean(|x_t - x_{t-1}|)`
- sign flip rate

Fallback trigger (consecutive streaks):

- saturation ratio `> 0.98` for 3 ticks, or
- delta norm `> 0.95` for 3 ticks.

On trigger (or NaN/Inf), output degrades to deterministic fallback (`StageQuality::DegradedFallback`).

## Audit / digests

The kernel surfaces digest prefixes in notes:

- fixture digest prefix
- model slot hash prefix (if loaded)
- parameter digest prefix
- `x0` digest prefix

Parameter digest canonicalization uses fixed tensor name ordering and raw `f32` little-endian bytes with slot hash + dims.

## Enablement modes

Slot rollout remains controlled by:

- `UCF_SLOT_LFM_MODE=toy|shadow|active`

`shadow` computes with real slot params without replacing primary outputs, while `active` uses real params as primary.
`toy` keeps policy fixture params.

## Probing commands

- validate/test kernel: `cargo test -p ucf-compute lnn_ --features lfm-lnn`
- full checks: `cargo fmt --all --check && cargo clippy --workspace --all-targets --all-features -- -D warnings`
