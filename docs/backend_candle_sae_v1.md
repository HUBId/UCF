# Backend Candle SAE v1

Scope for v3 Prompt 202: extend real-backend support to the already selected second real slot (`sae`) with Candle only.

## Semantics in scope

- Feature-gated behind `backend-candle`.
- Probe-first lifecycle (`stage -> verify -> probe -> promote -> probe`).
- Shadow-only runtime stage for SAE in this prompt (no decision impact).
- No contract/schema semantic changes; SAE output contract remains:
  - `spike_count`
  - top-k spikes (`feature_id`, `magnitude_q`)
  - `spikes_digest`
- Active enablement for SAE remains denied (`ACTIVE_NOT_ENABLED_FOR_SLOT_STAGE`).

## WeightSpec (unchanged)

Candle SAE validates encoder tensors only:

- `sae.w_enc`: shape `[F, D]`, dtype `f32`
- `sae.b_enc`: shape `[F]`, dtype `f32`

Missing/invalid weights degrade safely:

- missing promoted weights -> `BACKEND_DISABLED`
- invalid/mismatched spec during load -> `VALIDATION_FAILED`

## Deterministic behavior

- Candle executes encoder tensor load/forward.
- Deterministic top-k ranking remains Rust-side with tie-break on `feature_id`.
- Quantization and digesting stay deterministic and bound to model/input/spike payload.

## Offline command sequence

```bash
cargo run -p ucf-ops -- models stage --slot sae --src fixtures/models_real/sae_real_tiny
cargo run -p ucf-ops -- models verify --manifest models/manifest.toml
cargo run -p ucf-ops --features backend-candle -- models probe --slot sae --hash <staged_hash> --out ./out/probe_sae_staged.json
cargo run -p ucf-ops -- models promote --slot sae --hash <staged_hash>
cargo run -p ucf-ops --features backend-candle -- models probe --slot sae --out ./out/probe_sae_active.json
cargo run -p ucf-ops -- models shadow-ready --out ./out/shadow_ready_report.json
cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json
```

Expected: `probe_ready=true`, `shadow_ready=true` once compare/no-impact evidence exists; `active_eligible=false` for SAE at this stage.
