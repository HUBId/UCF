# LFM Memory Coupling v0

## Liquid context semantics

`Liquid context` is a bounded, windowed summary over recent LFM outputs. It is **not** a state dump and never stores raw vectors.

## ESS records

### `LfmSummaryRecord`

Small per-cadence digest/scalar record:
- `t`
- `decision_id`
- `evidence_chain_digest`
- `backend_pack_digest`
- `liquid_state_digest`
- `liquid_readout_digest`
- `uncertainty`
- `stability`
- `schema_version`
- `digest`

### `LfmWindowRecord`

Optional macro summary for consolidation-friendly access:
- `[t0, t1]`
- `sample_count`
- `mean_uncertainty`
- `mean_stability`
- `rolling_digest`
- `schema_version`
- `digest`

## Index cache and rebuild semantics

`LiquidTimelineIndex` is a deterministic local cache rebuilt from ESS `LfmSummaryRecord` entries.

- Sort order is stable: `t`, then `digest`.
- `get_window(t0, t1)` is bounded (`<=256`).
- `get_last(n)` is bounded (`<=128`).
- Missing/corrupt cache is recoverable by deterministic rebuild from ESS.

## Decision/LLM/consolidation usage

- Decision context receives a bounded `LiquidContextWindowSummary` with aggregate scalars + rolling digest.
- LLM prompt includes only aggregate scalars and digest prefixes.
- Consolidation/geist compatibility is preserved by `LfmWindowRecord` aggregates and bounded timeline queries.

## Bloat-avoidance rules

- Never persist raw LFM vectors.
- Persist only compact digests/scalars.
- Keep cadence bounded (`UCF_LFM_PERSIST_EVERY`, default 2).
- Use bounded retrieval windows and aggregate summaries.
