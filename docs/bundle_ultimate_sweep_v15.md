# Bundle ultimate sweep (v15)

`ucf-ops bundle-ultimate-sweep` emits `TerminalBundleUltimateSweepV1` as the terminal proof
that canonical export/build/verify/inspect/continuity consumers derive bundle authority only from
terminal absolute residual-free final bundle inputs.

## What this proves

The sweep fail-closes unless all covered consumers are aligned to:

1. `CanonicalExportArtifactRefV1`
2. `CanonicalExportContextV1`
3. `CanonicalBundleConsumptionContextV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`
6. `FinalBundleConsumerAuthorityV1`
7. `FinalBundleResidualSweepV1`
8. `ResidualFreeBundleConsumerAuthorityV1`
9. `ResidualFreeBundleAbsoluteSweepV1`
10. `AbsoluteFinalBundleTerminalSweepV1`

It additionally verifies that canonical flows do **not** treat bundle echo caches, manifest mirrors,
embedded export snapshots, bundle-lineage history, or inspect/report summaries as primary bundle truth.

## Covered consumers

The v15 sweep checks canonical consumer surfaces for:

- repro manifest outputs
- bugkit manifest outputs
- repro verify output
- export roundtrip and bundle spine reports
- continuity artifact surface (`terminal_absolute_final_input_continuity_sweep`)
- portability/interop report surface
- v15-prep helper (`v14_gate_report`)

## Command

```bash
cargo run -p ucf-ops -- bundle-ultimate-sweep --out ./out/bundle_ultimate_sweep.json
```

`PASS` means no covered canonical consumer path uses cache/mirror/snapshot reconstruction as primary bundle input substrate.

## v16 convergence note

v16 adds `bundle-convergence-sweep` to remove the final canonical dependencies on memoized/copied/derived bundle truth in canonical export consumers.
