# Compare Window Semantics v3

Scope: supported real-slot set only in current stage:
- `world_jepa`
- configured second slot from `docs/series_state_snapshot.md` (current: `sae`)

## UnifiedCompareWindowSemanticsV1
Shared rules for world + second slot:
- fixed tick windows with inclusive bounds `[t0, t1]`
- deterministic `window_id = u64_prefix(sha256(run_id:slot_id:t0:t1))`
- explicit primary backend id
- compared backend ids sorted lexicographically (`<=2`, bounded)
- sample digest prefixes bounded (`<=4`) with deterministic source ordering
- freshness: `current_tick - t1 <= max_age_ticks` => `FRESH`, else `STALE_COMPARE`

## CompareWindowMetaV1
Shared metadata shell embedded in parity records:
- `slot_id`
- `run_id`
- `window_id`
- `t0`, `t1`
- `primary_backend_id`
- `compared_backend_ids`
- `compare_window_digest`
- `policy_graph_digest_prefix`

## Status normalization
Backend parity statuses use one enum:
- `OK`
- `WARN`
- `SEVERE`
- `SKIP`

`SKIP` is used for optional compare dimensions (e.g., Burn backend not scaffolded/enabled).

## Drift + evidence consumption
Drift inputs are derived through shared normalization from compare windows:
- `invalid_rate_q`
- `digest_mismatch_rate_q`
- `latency_p95_ms_q` (if available)
- slot-specific bounded scalar deltas

Shadow-ready and active evidence reuse the same compare freshness semantics (`NO_COMPARE` / `STALE_COMPARE`) for world and second slot.

## Compatibility
Parity records include additive shared metadata (`CompareWindowMetaV1`) with serde defaults,
so legacy records without metadata remain readable via compatibility defaults.
