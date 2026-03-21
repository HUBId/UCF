# remediation spine consistency v8

`ucf-ops remediation-spine-check` is the stronger v8 consistency proof for canonical blocking/remediation alignment.

It verifies that the same canonical condition emits the same primary blocking/remediation semantics across:
- applied scope authority
- canonical governance entry
- canonical readiness spine
- canonical bundle spine
- interop matrix
- operator/export authority chain
- gate family (v4-v8 coverage with explicit MISSING/SKIP where unsupported)
- operator signoff/review packet
- export roundtrip / bundle spine check

## Command

```bash
cargo run -p ucf-ops -- remediation-spine-check --out ./out/remediation_spine_check.json
```

## What it detects beyond v7 remediation-interop-check

- spine-level condition drift (`AppliedScopeMismatch`, `GovernanceEntryMismatch`, `ReadinessSpineMismatch`, `BundleSpineMismatch`)
- explicit `UNKNOWN_CONDITION_MAPPING` when a surface mismatch category cannot map to canonical conditions
- explicit `MISSING_SURFACE` / `SKIP` semantics per surface rather than silent omission
- gate-family consistency checks for canonical gate-fail conditions with deterministic fail-on-drift behavior

Final v9 authority proof is consolidated in `docs/primary_semantics_sweep_v9.md` via `ucf-ops primary-semantics-sweep`. v10 finalizes universal canonical consumer authority in `docs/final_primary_semantics_sweep_v10.md` via `ucf-ops final-primary-semantics-sweep`.

## v11 residual-sweep note

v11 note: residual primary-semantics reconstruction in canonical flows is now explicitly swept/blocked by `primary-semantics-residual-sweep`.
