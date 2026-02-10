# UCF Demo CLI

The pause-point demo provides a deterministic coherence loop checkpoint without any external data.

## Run

```bash
cargo run -q -p ucf-demo -- --cycles 12 --seed 42
```

## Output shape

One line per cycle with the coherence summary fields:

- `cycle`
- `gamma_bucket`
- `plv`
- `lock_window`
- `surprise`
- `novelty` / `salience` / `attention_gain`
- `learn_rate` / `mode`
- `delta_mass` + `targets` bitmap
- `nsr_verdict` + `nsr_hits`
- `violations` (`none` or comma-separated rule tuples)

Example shape:

```text
cycle=1 gamma_bucket=... plv=... lock_window=... surprise=... novelty=... salience=... attention_gain=... learn_rate=... mode=... delta_mass=... targets=.... nsr_verdict=... nsr_hits=[..., ..., ...] violations=...
```

## Determinism

For a fixed `--seed` and `--cycles`, the demo is deterministic:

- each cycle derives deterministic `external_commit` and `policy_snapshot_commit` from `(seed, cycle)`
- those deterministic values are embedded in the control-frame inputs
- runtime summaries are derived from the same router/workspace runtime path used in tests
