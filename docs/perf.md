# UCF Benchmarks & Latency Budgets (v0)

`ucf-bench` provides deterministic, offline benchmark runs for:

- control loop end-to-end (`ControlFrame -> Decision -> ESS append`)
- compute backend micro benchmark (`AiComputeBackend::compute`)
- sandbox runtime micro benchmark (`IsolationRuntime::call`)

## Determinism rules

- fixed fixtures (`runtime/ucf-compute/fixtures/compute_inputs.json`)
- deterministic synthetic control frames (tick/correlation-id sequence)
- no network access and no external model weights
- no sleeps in workload paths

## Budget profile (initial baseline targets)

- Control loop: `p50 <= 2ms`, `p95 <= 5ms`, `p99 <= 10ms` (stub baseline target)
- ESS append path proxy: `p95 <= 1ms` (derived from control loop per tick)
- Regression guard thresholds:
  - throughput must stay above `baseline * 0.7`
  - p95 must stay below `baseline * 1.5`

These are conservative starting points. They are intended for relative regression detection.

## Commands

```bash
cargo run -p ucf-bench -- control-loop --ticks 2000 --backend stub --isolation inproc --out bench/control-loop.json
cargo run -p ucf-bench -- compute --cases fixtures --backend stub --out bench/compute.json
cargo run -p ucf-bench -- sandbox --runtime inproc --cases echo --n 2000 --out bench/sandbox-inproc.json
```

Optional features:

```bash
cargo run -p ucf-bench --features compute-candle -- compute --cases fixtures --backend candle --out bench/compute-candle.json
cargo run -p ucf-bench --features compute-burn -- compute --cases fixtures --backend burn --out bench/compute-burn.json
cargo run -p ucf-bench --features sandbox-wasm -- sandbox --runtime wasm --cases echo --out bench/sandbox-wasm.json
cargo run -p ucf-bench --features sandbox-proc -- sandbox --runtime proc --cases echo --out bench/sandbox-proc.json
```

## Baseline guard workflow

1. Produce a baseline JSON and commit it under `bench/baselines/<platform>/baseline.json`.
2. Run a fresh benchmark in CI/local smoke mode.
3. Compare:

```bash
cargo run -p ucf-bench -- compare-baseline \
  --baseline bench/baselines/linux-x86_64/baseline.json \
  --current bench/control-loop.json \
  --out bench/regression-report.json
```

The compare command exits non-zero if thresholds fail.

## Output JSON fields

Each benchmark JSON includes:

- `build_tag`
- `command` and selected config
- `stats` (`n`, `mean/p50/p95/p99/min/max`, `throughput_ops_sec`)
- allocation proxies (`alloc_count`, `alloc_bytes_total`, dealloc counters)
- optional benchmark-specific extras

## Variability caveats

- absolute timing still varies across hosts and CPU governors
- use relative baseline checks in CI
- compare like-for-like feature profiles (`stub/candle/burn`, `inproc/wasm/proc`)
