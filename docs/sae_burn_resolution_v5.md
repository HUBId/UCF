# SAE Burn Resolution v5

## Scope

This phase stays strictly on the already chosen second real slot from `docs/series_state_snapshot.md` (current: `sae`).
No new slot is introduced.

## Canonical Resolution

Burn status is now emitted as `BurnSupportResolutionV1` with a binary resolution:

- `BURN_SUPPORTED_FOR_SHADOW_COMPARE`
- `BURN_CLOSED_UNSUPPORTED`

The record also carries:

- `support_state` (`SUPPORTED|UNSUPPORTED|NOT_BUILT|NOT_CONFIGURED`)
- bounded `rationale_codes`
- deterministic `evidence_digest`

## Inspect Command

```bash
cargo run -p ucf-ops -- models backend-resolution --slot sae --workdir . --out ./out/backend_resolution_sae.json
```

Interpretation:

- `burn_resolution=BURN_SUPPORTED_FOR_SHADOW_COMPARE` means optional Burn probe/shadow compare is available.
- `burn_resolution=BURN_CLOSED_UNSUPPORTED` means Burn is formally closed for this phase and not assumed by default.

## Strict/Gate behavior

- Default remains fail-closed and honest.
- If Burn is closed unsupported, optional Burn checks stay `SKIP` unless explicitly required.
- If config explicitly requires Burn while closure is unsupported, strict fails with `OPTIONAL_BACKEND_CLOSED_UNSUPPORTED`.

## Safety note

Even when Burn is supported, it remains **shadow-only** and has **no decision impact**.
