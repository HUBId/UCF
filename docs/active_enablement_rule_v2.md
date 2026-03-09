# Active Enablement Rule v2

Real slots MUST remain in `shadow` unless an explicit bounded evidence bundle is available.

## Active evidence contract (`ActiveEnablementEvidenceV1`)

Required fields:
- `slot_id`
- `target_hash`
- `latest_probe_report_digest_prefix`
- `latest_probe_status`
- `latest_compare_window_digest_prefix`
- `shadow_no_impact_verified`
- `latest_drift_status` (`OK|WARN|SEVERE|UNKNOWN`)
- `evidence_window_ticks`
- `evidence_digest`

`evidence_digest` is computed deterministically from canonical field order and bounded values.

## Guard rule

`can_enable_active(slot_id, target_hash, ctx)` denies by default (fail-closed).

Denial codes:
- `ACTIVE_DENIED_NO_PROBE`
- `ACTIVE_DENIED_NO_SHADOW_EVIDENCE`
- `ACTIVE_DENIED_DRIFT`
- `ACTIVE_DENIED_HASH_MISMATCH`
- `ACTIVE_DENIED_STRICT_MODE`

Checks:
1. promoted active hash exists and matches requested target
2. latest probe report exists for slot/hash and is `PASS`
3. latest compare window exists and is fresh within bounded window ticks
4. shadow no-impact check succeeded
5. no severe drift in evidence window (WARN only if explicitly allowed)

## Strict mode behavior

When strict mode is enabled, any configured active real slot requires `can_enable_active == Ok`.
If not satisfied, strict checks fail with deterministic evidence references.

## Ops workflow

Run:

```bash
cargo run -p ucf-ops -- models active-check --slot <slot> --out ./out/active_check_<slot>.json
```

Also visible via:

```bash
cargo run -p ucf-ops -- models list --slot <slot>
```

`models list` now surfaces:
- `current_mode`
- `active_eligible`
- `last_evidence_digest_prefix`

## Important

Promotion (`stage -> promote`) does **not** imply activation.
Activation remains blocked until the active evidence bundle passes.


## Shadow-Ready distinction

`Shadow-Ready` is a separate gate (`models shadow-ready`) that certifies shadow evidence completeness for the two currently supported real slots.

- `Shadow-Ready` indicates probe/compare/no-impact/drift evidence is sufficient for shadow scaffolding.
- `Active-Eligible` indicates active enablement evidence passed for a slot/hash.

These are intentionally different:

- `Shadow-Ready != Active-Eligible`
- A slot can be shadow-ready and still be denied for active mode by policy/stage rules.
