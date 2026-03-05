# Strict Mode v1

Strict Mode enables a single additive guard rail switch for runtime and ops checks.

## Enable

- Env: `UCF_STRICT_MODE=1`
- CLI: `--strict`

## Enforced checks

- Determinism lock checks (sampling disabled + RNG scan)
- Policy checks (digest required + policy pack validation)
- Models checks (manifest digest required, promoted-only paths, slot verify)
- Tooling checks (deny-default tool policy / governed path)
- Sandbox checks (runtime path scan)
- Ops-only release checks (`ucf-ops strict check` also runs docs lint strict)

## Failure report

On strict failure, a single consolidated report is written to:

- `./out/strict_failure.json` (runtime/startup path)
- custom `--out` path for `ucf-ops strict check`

Report fields are bounded and redaction-safe:

- `check_id`
- `status` (`pass` / `fail`)
- `error_codes`
- `remediation`

## Run metadata

`RunMetadataRecord` persists:

- `strict_mode_enabled`
- `strict_mode_digest`

## Recommended use

- **test/prod**: always enable strict mode
- **dev**: optional, but recommended before promotion


## Panic policy

- Runtime panic handling is structured:
  - Stage-boundary panics are caught and converted to runtime panic errors (`runtime.panic`, code `1004`).
  - A `PanicRecordV1` is emitted to local panic diagnostics (`out/panic_records.jsonl`).
- Strict mode supports optional fail-fast panic shutdown:
  - `UCF_STRICT_MODE=1`
  - `UCF_STRICT_PANIC_FAIL_FAST=1`
- Without fail-fast, runtime uses deterministic degraded fallback semantics.
