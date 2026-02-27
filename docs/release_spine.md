# Release Spine v0

## Feature flags

Single source of truth lives in `runtime/ucf-compute/src/feature_matrix.rs`.

- `backend-stub`
- `backend-toy`
- `llm-candle`
- `lfm-candle`
- `backend-burn`
- `lfm-lnn`
- `plasticity`
- `replay`
- `ops-explain`

Pack selection is validated at build/start time. If a selected backend pack requires a missing feature, the runtime fails fast with a clear error.

## Offline runtime profiles

`UCF_PROFILE=dev|test|prod` is resolved by `ucf-ops` and merged with env overrides.

Defaults:
- `dev`: offline, shadow governance lane, no tools by default, debug logs.
- `test`: deterministic, deny-by-default, shadow modes, offline on.
- `prod`: deterministic strict, deny-by-default, no sampling, policy-hash locked flow.

## One-command bringup

```bash
cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out
```

Behavior:
1. Enforces offline test defaults.
2. Starts runtime in-process.
3. Runs bounded deterministic ticks.
4. Emits artifacts:
   - `out/metrics_summary.json`
   - `out/explain_tick_last.json`
   - `out/replay_verify.json` (unless `--no-replay`)
   - `out/run_metadata_record.json`

## RunMetadataRecord

Persisted under:
- `.ucf/ess/run_metadata_record.json`
- `out/run_metadata_record.json`

Fields:
- `run_id`
- `started_at_tick`
- `code_version_tag`
- `backend_pack_meta_digest`
- `fixtures_digest`
- `enabled_features_bitmap`
- `profile`
- `schema_versions`

## Versioned artifacts

Reproducibility uses:
- git code tag (`git rev-parse HEAD`)
- backend pack digest
- fixtures digest
- deterministic seed/profile
