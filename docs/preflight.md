# Preflight Checklist (Final Pre-Release Gate)

`ucf-ops preflight` runs a strict, portable, offline-first shipment checklist against a bundle.

## Command

```bash
ucf-ops preflight --bundle <path> --out ./out/preflight.json
```

## What it checks (fixed order)

1. `bundle_integrity` (critical)
   - Required bundle layout/files (`bin`, `configs`, `policies`, `models`, `VERSION.txt`)
   - `VERSION.txt` consistency (`manifest_digest` check)
2. `strict_check` (critical)
   - Runs strict invariant checks equivalent to strict-mode guardrails.
3. `docs_lint` (optional)
   - Runs only if bundle has a `docs/` directory; otherwise `SKIP`.
4. `gate_status`
   - Uses `out/gate_latest.json` or `out/gate_report.json` if available.
   - Falls back to a readiness smoke run when no gate report exists.
5. `runtime_status` (optional)
   - Includes runtime evidence presence (`health`, `alerts`, `drift`) if available.
6. `rc_manifest` (critical if RC artifacts are present)
   - Validates `RC_MANIFEST.json` digest, `RC_MANIFEST.sig`, and `SHA256SUMS.txt`.

## Exit codes

- `0`: PASS
- `2`: FAIL (non-critical)
- `3`: FAIL (critical integrity failure)

## Reading output

The JSON report contains:
- `overall`: `PASS`/`FAIL`
- `exit_code`
- ordered `checks[]` with `status`, `critical`, evidence, and remediation text
- deduplicated `remediation_hints`

## Common remediation

- Rebuild portable bundle:

```bash
python deploy/scripts/build_bundle.py --target <bundle> --profile <dev|test|prod>
```

- Re-run strict checks:

```bash
ucf-ops strict check --bundle <path> --strict --out ./out/strict_check.json
```

- Re-run readiness gate:

```bash
ucf-ops readiness-gate --bundle <path> --profile test --out ./out/gate_report.json
```
