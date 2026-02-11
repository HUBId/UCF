# Compute Replay Harness v0

`ucf-replay` re-executes persisted decisions offline and deterministically.

## Run

```bash
cargo run -p ucf-replay -- replay \
  --fixture runtime/ucf-replay/fixtures/golden_replay_fixture.json \
  --from 1 --to 10 --mode compute
```

Optional flags:

- `--mode compute|score|full`
- `--backend stub|candle|burn`
- `--seed <u64>`
- `--report <path>`

## Interpreting drift

- `match`: persisted and recomputed summaries are within epsilon.
- `drift`: at least one float or digest differs beyond policy.
- `unreplayable`: missing fields or unavailable backend.

JSON diagnostics include:

- `float_mismatch`
- `digest_mismatch`
- `missing_persisted_field`
- `backend_unavailable`

## Updating the golden fixture

1. Keep the fixture tiny (3 control frames + 3 decisions).
2. Recompute with the same backend/seed/budget profile.
3. Update persisted summary fields together in one commit.
4. Run full workspace tests before merge.
