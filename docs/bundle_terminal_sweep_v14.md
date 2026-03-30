# Bundle Terminal Sweep v14

`ucf-ops bundle-terminal-sweep` emits `AbsoluteFinalBundleTerminalSweepV1`.

## What this proves

The sweep proves that covered canonical export consumers derive bundle truth from terminal, residual-free inputs only:

1. `CanonicalExportArtifactRefV1`
2. `CanonicalExportContextV1`
3. `CanonicalBundleConsumptionContextV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`
6. `FinalBundleConsumerAuthorityV1`
7. `FinalBundleResidualSweepV1`
8. `ResidualFreeBundleConsumerAuthorityV1`
9. `ResidualFreeBundleAbsoluteSweepV1`

Any manifest echo, lineage memory, export summary/hint, or report-summary based reconstruction in canonical flows is treated as residual and fails the sweep.

## Covered consumers

- Repro manifest outputs
- BugKit manifest outputs
- Repro verify/report surfaces
- Export roundtrip and bundle-spine checks
- Continuity chain artifacts
- Interop/export matrix helpers
- v14 prep gate helper surfaces

## Command

```bash
cargo run -p ucf-ops -- bundle-terminal-sweep --out ./out/bundle_terminal_sweep.json
```

`PASS` means no canonical consumer path relies on bundle echo/summary/history reconstruction as primary bundle truth.


## v15 hardening

v15 (`bundle-ultimate-sweep`) removes remaining canonical bundle cache/mirror/snapshot traces by requiring `bundle_ultimate_sweep_digest_prefix` and terminal digest continuity on covered consumers.

## v16 convergence note

Canonical export consumers are now converged through `bundle-convergence-sweep` and no longer allow memoized/copied/derived bundle authority as primary truth.

## v17 stabilization note

v17 finalizes canonical export consumer stabilization and blocks adapter/translation/projection bundle truth paths in canonical flows.
