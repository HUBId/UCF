# Slot Modes v1

## Modes

`SlotModeV1` defines hardware-neutral slot rollout states:

- `off`: slot ignored.
- `shadow`: slot executes on deterministic sampled ticks for diagnostics only.
- `active`: slot output can influence primary path.

## Deterministic shadow scheduler

Shadow ticks are selected with a phase-offset schedule:

- `offset = H(run_id || slot_id) % shadow_rate`
- run shadow on tick `t` when `t % shadow_rate == offset`

This keeps sampling deterministic and avoids per-tick hashing overhead.

## No decision impact guarantee

The compute wrapper enforces type separation:

- `PrimaryOutput` is the only value returned to decision code.
- `ShadowOutput` is consumed only by compare-window aggregation.
- Shadow events are emitted through a side-channel (`drain_shadow_events`) and persisted to ESS.

Shadow failures never fail the primary path. In active mode, primary failure still degrades to unavailable fallback.

## Compare records

Every `compare_window` ticks, ESS receives bounded `SlotCompareWindowRecordV1`:

- window bounds `t0..t1`
- primary/shadow scalar stats (`mean`, `p95`) in fixed-point
- mismatch counters (`digest_mismatch_count`, `invalid_shadow_count`)
- digest sample prefixes (max 4)
- status: `Ok`, `DriftWarn`, `ShadowDisabled`

Repeated shadow failures emit `ShadowDisableRecord`; runtime also emits `SlotModeChangeRecordV1` on observed mode changes.

## Enable shadow safely

Environment knobs for v1 wiring:

- `UCF_REAL_ENABLEMENT_MODE=off|shadow|compare|active`
- `UCF_SLOT_SHADOW_RATE=<n>`
- `UCF_SLOT_COMPARE_WINDOW=<ticks>`

To verify no decision impact, run the same scenario twice with identical inputs:

1. shadow disabled (`off`)
2. shadow enabled (`shadow`)

Decision outputs (`DecisionInputs` / decision digests) must match; only side-channel slot compare records may differ.
