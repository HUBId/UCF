# Backend Packs


Status note: backend names are feature/backend-family names. Current docs must distinguish stub fixture, toy golden, optional-real compile-only, remote/external compile-only, optional-real runtime deferred, and production claim forbidden.
`BackendPack` is the unified compute surface for LLM + world model + SAE + SSM.

## Pack selection

Use environment variables:

- `UCF_BACKEND_PACK=toy_v1|stub_v0|candle_toy_v1|candle_liquid_v1|burn_toy_v1`
- `UCF_BACKEND_SEED=<u64>`

## Fixtures

Fixtures are committed and loaded offline via `include_bytes!`:

- `runtime/ucf-compute/fixtures/toy_weights_v1.json`
- `runtime/ucf-compute/fixtures/jepa_dyn_v1.json`
- `runtime/ucf-compute/fixtures/sae_proj_v1.json`
- `runtime/ucf-compute/fixtures/ssm_toy_v1.json`
- `runtime/ucf-compute/fixtures/lfm_params_v1.json`

`FixtureManager` computes per-fixture digests and an overall digest in canonical order.

## Evidence and replay

EvidenceChain carries backend pack identity and fixture digest metadata (pack id + component ids + fixtures digest), so replay can verify exactly which pack/fixtures produced outputs.

## Hot swap

Hot-swap is intended only at safe tick boundaries (never mid-tick). Backend pack changes are audited with a dedicated ESS backend-pack record.
