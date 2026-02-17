# Perf & Cost Envelope Playbook (v1)

## Bench command

Run offline deterministic bench:

```bash
ucf-ops bench --scenario fixtures/e2e_scenario_a.json --ticks 256 --out ./out/bench_report.json
```

Optional knobs:

- `--rss-sample-every <K>` RSS sample period in ticks (default: 16)
- `--rss-cap-mb <N>` fail run if max RSS exceeds cap

## What is measured

- Throughput: `throughput_ticks_per_sec`
- Tick latency: `tick_time_ms` (p50/p95/p99/max)
- Per-stage latency:
  - world
  - sae
  - ssm
  - lfm
  - risk
  - governor
  - llm
- Backpressure / budget envelope:
  - `counters.backpressure_ticks`
  - `counters.budget_exceeded_events`
  - `counters.degraded_stages`
- Memory envelope:
  - `memory.min_rss_mb`
  - `memory.mean_rss_mb`
  - `memory.max_rss_mb`
  - `memory.cap_exceeded`

## Tuning knobs

- Compute budgets per stage (`UCF_COMPUTE_BUDGET_PROFILE`, `UCF_COMPUTE_MAX_MICROS`)
- LFM kernel profile (`UCF_BACKEND_PACK`: toy/candle/burn/lnn variants)
- Plasticity on/off (depending on chosen pack/profile)
- Shadow mode sampling rate (reduce overhead in prod-like validation)
- LLM decode caps (`UCF_LLM_MAX_TOKENS`, uncertainty-driven token scaling in runtime)

## Interpretation quick guide

- If `lfm` p95 > budget:
  - raise stage budget or switch to lighter LFM kernel/profile.
- If `llm` dominates p95:
  - reduce `max_tokens_eff` and tighten uncertainty scaling.
- If backpressure active for many ticks:
  - tune governor thresholds or lower workload.
- If memory max breaches cap:
  - lower model footprint, reduce buffer sizes, or reduce tick burst size.

## Envelope targets (NUC Ultra 7 class nodes)

Use these as placeholders and calibrate on hardware:

- Tick p95: `< target_ms_placeholder >`
- LFM p95: `< target_lfm_ms_placeholder >`
- RSS max: `< target_rss_mb_placeholder >`
- Backpressure ratio: `< target_bp_ratio_placeholder >`

