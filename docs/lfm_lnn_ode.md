# LFM × LNN ODE Core v1

This document describes the deterministic LFM ODE kernel (`LnnOdeLfmKernel`) used in compute.

## Equation

State dynamics per substep:

- `x ∈ R^N`, `u ∈ R^M`
- `dx/dt = -alpha ⊙ x + tanh(Wx*x + Wu*u + b)`

v1 bounds:

- `N <= 32`
- `M <= 8`
- fixed `dt`
- fixed `steps` per tick
- clamp `x` to `[-1, 1]` each substep

## Integrator

v1 uses deterministic RK2 (midpoint), fixed-step only.

For each substep:

1. `k1 = f(x, u)`
2. `k2 = f(x + 0.5*dt*k1, u)`
3. `x <- clamp(x + dt*k2)`

No adaptive solver is used.

## Inputs

Canonical `u` uses bounded compute signals only (no raw text):

- pressure
- surprise
- risk
- prior uncertainty
- confidence
- spike density
- sae energy
- stress composite

All components are mapped deterministically in `[0,1]` and then consumed as signed inputs in the ODE.

## Contracted outputs

Derived scalars:

- `uncertainty`
- `stability`
- `homeostasis_error = |uncertainty - homeostasis_target|`

Each scalar is quantized to `UQ0_16` and persisted in the LFM output contract.

Digest binding includes:

- model fixture digest
- optional plasticity digest
- `t`
- `context/world/spikes` digests
- quantized state
- quantized contracted scalars

## Safety and fallback

If NaN/Inf is detected, kernel returns `DegradedFallback` with:

- `uncertainty=1`
- `stability=0`
- `homeostasis_error=1`

and marks `nan_inf_detected=true`.

## Config knobs

Current fixture fields:

- `n`, `m`
- `dt`, `steps`
- `clamp_state`, `clamp_deriv`
- `homeostasis_target`
- `alpha`, `Wx`, `Wu`, `b`

Recommended defaults for v1:

- `n=16`
- `m=8`
- `dt=0.05`
- `steps=4`

## Weights (future)

v1 remains policy-fixture first for determinism envelope validation. A later revision can add
`lfm-weights` model-slot loading without changing the deterministic contract behavior.
