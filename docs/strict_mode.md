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

### v1 strict checks summary

Strict mode evaluates v1 controls in deterministic order:

1. model manifest + promoted-path integrity checks
2. active-slot probe evidence checks (when probe enforcement is enabled)
3. shadow preconditions (`drift_budget` present, compare-window wiring set)
4. observational-only guarantee checks for shadow outputs
5. strict-failure artifact emission for operator remediation

### v1 scaffold checks (strict extension)

When strict mode is enabled, v1 scaffold invariants are enforced additively:

- Active slot modes require a present manifest digest (`models/MANIFEST.toml`).
- Active slot hashes must remain promoted-only and verifiable.
- Optional probe enforcement (`UCF_STRICT_ENFORCE_ACTIVE_PROBES=1`) requires PASS probe evidence for promoted active slots.
- Shadow enablement requires drift budget availability from merged policy graph.
- Shadow enablement requires compare-window emission wiring (`UCF_SLOT_COMPARE_WINDOW > 0`).
- Shadow outputs are guarded to remain observational-only and not alter decision outputs.

## Failure report

On strict failure, a single consolidated report is written to:

- `./out/strict_failure.json` (runtime/startup path)
- custom `--out` path for `ucf-ops strict check`

Report fields are bounded and redaction-safe:

- `check_id`
- `status` (`pass` / `fail`)
- `error_codes`
- `remediation`

The report now also includes:

- `v1_checks` (fixed check-id ordering)
- `evidence_digest_prefixes` (bounded digest prefixes only)

## Run metadata

`RunMetadataRecord` persists:

- `strict_mode_enabled`
- `strict_mode_digest`
- `probe_report_digest_prefix` (when probe report is available)

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


### v3 strict refresh

Strict mode now also exposes a unified `v3` report section in `out/strict_failure.json` for the supported real-slot set (`world_jepa` + declared second slot).

Runtime startup strict validation and `ucf-ops strict check` share the same v3 denial semantics/check IDs.

See details in `docs/strict_mode_v3.md`.


## Operator first-stop summary

Before running isolated strict checks, operators should first run:

```bash
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
```

Then use `strict_section` to see whether strict is `OK|FAIL|MISSING` and continue with `ucf-ops strict check` for detail remediation.


## v4 consistency note
Strict v3 checks now use the shared supported-slot set resolution path to align with eligibility/operator/gate evidence interpretation.

## Canonical remediation registry v1

Strict report checks continue emitting specialized denial codes, and now additionally carry canonical remediation codes from the shared registry mapping.

## Operator interplay (v4 hardening)

Strict evidence consumed by operator-facing surfaces is unified as `StrictEvidenceSnapshotV1`. Consolidated operator report and operator signoff read the same strict snapshot/mapping path, including explicit `MISSING` handling when strict evidence is required.
See `docs/strict_operator_interplay_v4.md`.
