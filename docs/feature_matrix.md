# Canonical feature matrix (post-rc1)

Only the combinations below are supported and CI-covered.

## Runtime lanes

1. `default (toy)`
   - `cargo test --workspace --all-targets`
2. `candle-cpu`
   - `cargo test --workspace --all-targets --features "compute-candle,llm-candle,lfm-candle"`
3. `burn-cpu`
   - `cargo test --workspace --all-targets --features "compute-burn,backend-burn,llm-burn,lfm-burn"`
4. `stage-isolation`
   - `cargo test --workspace --all-targets --features "sandbox-wasm,stage-isolation"`

## Tools-only lane

5. `ebm-train`
   - `cargo test -p ucf-ebm-train --features "ebm-train"`

## Compile-time guards

`ucf-compute` enforces mutual exclusion:
- `compute-candle` vs `compute-burn`
- `llm-candle` vs `llm-burn`
- `lfm-candle` vs `lfm-burn`

## CI policy

CI must run only these canonical lanes and reject unsupported feature-combination sprawl.
