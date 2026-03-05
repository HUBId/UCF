# Failure Semantics (v1)

- **Stage failure**
  - Runtime stage boundaries are panic-guarded.
  - Panic in compute/tool stages records `PanicRecordV1` and degrades stage output to fallback when fail-fast is disabled.
  - Stable runtime panic code: `runtime.panic` (`1004`).

- **Tool failure**
  - Tool execution failure is persisted in `ToolExecutionRecord` with status `Failed` and deterministic `error_code`.
  - Tool failure does not escalate to process panic.

- **Gateway failure**
  - Gateway wraps handler execution in panic guard and always returns safe error payload.
  - Internal failures map to `ERR_INTERNAL (1500)` and include only `request_id`, never internal stack or paths.

- **Strict mode failure**
  - Strict checks emit `out/strict_failure.json` and fail with non-zero exit.
  - Optional panic fail-fast can be enabled with `UCF_STRICT_PANIC_FAIL_FAST=1` in strict mode.

- **Crash-dump/backtrace policy**
  - Set `RUST_BACKTRACE=0` by default in deployment environments.
  - Core/crash dumps are disabled best-effort (Linux `RLIMIT_CORE=0`); recorded as `crash_dumps_disabled` in run metadata.
  - Backtrace material is diagnostics-only (`ucf-ops diagnostics collect --include_backtrace`) and path-redacted.
