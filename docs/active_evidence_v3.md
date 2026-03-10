# Active Evidence v3 (Supported Real Slots Set)

## Supported real-slot set (current stage)
- `world_jepa`
- plus exactly one second slot declared in `docs/series_state_snapshot.md` (`sae` in current repo state)

This check is bounded to the currently supported real-slot set (max 2 slots in v3 stage slice).

## Unified policy contract
`UnifiedActiveEvidencePolicyV1` centralizes freshness and behavior:
- `freshness_probe_max_age_ticks`
- `freshness_compare_max_age_ticks`
- `freshness_no_impact_max_age_ticks`
- `freshness_drift_status_max_age_ticks`
- `allow_warn_drift_for_active` (default `false`)
- `require_matching_target_hash` (default `true`)

Values are sourced from profile config (`[active_evidence]`) and evaluated deterministically.

## Denial codes
- `ACTIVE_DENIED_NO_PROBE`
- `ACTIVE_DENIED_STALE_PROBE`
- `ACTIVE_DENIED_NO_COMPARE`
- `ACTIVE_DENIED_STALE_COMPARE`
- `ACTIVE_DENIED_NO_NOIMPACT`
- `ACTIVE_DENIED_STALE_NOIMPACT`
- `ACTIVE_DENIED_DRIFT_SEVERE`
- `ACTIVE_DENIED_DRIFT_WARN`
- `ACTIVE_DENIED_HASH_MISMATCH`

## Guard semantics
- No automatic activation.
- Active remains human-controlled.
- Fail-closed on missing/stale evidence.
- Shadow mode keeps no decision-impact semantics.
- `active-eligible` is readiness-only, not an activation command.

## Operator command
```bash
cargo run -p ucf-ops -- models active-evidence --out ./out/active_evidence_report.json
```

The report provides per-slot eligibility, denial code (if any), evidence digest prefix, and freshness ages.
