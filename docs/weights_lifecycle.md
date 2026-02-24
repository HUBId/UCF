# Weights Lifecycle v1.1

Operational flow (offline-first):

1. `ucf-ops models stage --slot <slot> --path <local_dir>`
2. `ucf-ops models probe --manifest models/manifest.toml --out ./out/probe_report.json`
3. `ucf-ops readiness-gate --profile test --out ./out/gate_report.json`
4. `ucf-ops models promote --slot <slot> --hash <sha256> --probe-report ./out/probe_report.json --gate-report ./out/gate_report.json`
5. Runtime loads promoted artifact referenced by `models/MANIFEST.toml`.

Rollback:

- `ucf-ops models rollback --slot <slot> --to <sha256>`

Inventory:

- `ucf-ops models list --slot <slot>`

Emergency pinning:

- `UCF_MODEL_PIN_<SLOT>=<sha256>` (promoted-only).
- Example: `UCF_MODEL_PIN_WORLD_JEPA=<sha256>`.

## WorldVljepa promotion gate evidence

`world_vljepa` promotion is shadow-evidence gated when `UCF_WORLD_VLJEPA_REQUIRE_SHADOW_EVIDENCE=1` (default).

Required for `ucf-ops models promote --slot world_vljepa`:
- probe report PASS
- readiness gate PASS
- shadow report PASS with minimum soak ticks (`UCF_WORLD_VLJEPA_PROMOTION_MIN_TICKS`, default `10000`)

Example:

```bash
ucf-ops models promote \
  --slot world_vljepa \
  --hash <artifact_hash> \
  --probe-report ./out/probe_report.json \
  --gate-report ./out/gate_report.json \
  --shadow-report ./out/world_shadow_report.json
```
