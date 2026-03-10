# Strict Mode v3 Refresh

Strict mode v3 enforces one unified evidence path over the currently supported real-slot set only:

- `world_jepa`
- plus exactly one second slot from `docs/series_state_snapshot.md` (`sae` or `ssm`)

No new slot/backends are introduced by this check.

## Unified checks

`ucf-ops strict check` and runtime startup strict validation both execute the same v3 check family:

- `STRICT_MANIFEST_VALID`
- `STRICT_PROBE_READY`
- `STRICT_SHADOW_READY`
- `STRICT_ACTIVE_ELIGIBLE`
- `STRICT_COMPARE_FRESH`
- `STRICT_DRIFT_OK`
- `STRICT_HASH_CONSISTENT`

Behavior is fail-closed: stale/missing evidence or hash mismatch denies strict validation.

## One-report semantics

Strict failures are consolidated into one deterministic report artifact:

- `./out/strict_failure.json`

The v3 section (`v3`) is bounded and redaction-safe:

- `schema_version`
- `strict_mode_enabled`
- `overall_status`
- fixed-order `checks[]` with:
  - `check_id`
  - `slot_id` (optional)
  - `status` (`PASS|FAIL|SKIP`)
  - `denial_code`
  - bounded `evidence_digest_prefixes`
  - `remediation_code`

Ordering is deterministic:

1. global checks
2. per-slot checks sorted by `slot_id`

## Operator command

```bash
cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json
```

Interpretation:

- `STRICT_COMPARE_WINDOW_STALE` → compare evidence is outside configured freshness bounds.
- `STRICT_DRIFT_DENY` → severe/warn drift evidence denies strict pass under current strict defaults.
- `STRICT_HASH_MISMATCH` → probe/active evidence is not hash-consistent with target hash.
- `STRICT_MANIFEST_INVALID` → strict-required manifest/hash state is incomplete or invalid.
