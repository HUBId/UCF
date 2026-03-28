# Bundle Absolute Sweep v13

`ucf-ops bundle-absolute-sweep` emits `ResidualFreeBundleAbsoluteSweepV1`.

## What it proves

The sweep is a bounded, deterministic proof that covered canonical export consumers derive bundle truth from the residual-free final bundle input chain only:

1. `CanonicalExportArtifactRefV1`
2. `CanonicalExportContextV1`
3. `CanonicalBundleConsumptionContextV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`
6. `FinalBundleConsumerAuthorityV1`
7. `FinalBundleResidualSweepV1`
8. `ResidualFreeBundleConsumerAuthorityV1`

Historical manifest lineage, bundle-local history, embedded export hints, and ad-hoc inspect/roundtrip summaries are no longer valid primary authority in canonical flows.

## Covered consumers

- repro manifest/build surface
- bugkit manifest/build surface
- repro verify report path
- export roundtrip check output
- export bundle spine check output
- continuity chain artifact path
- v13 prep helper gate path

## Command

```bash
cargo run -p ucf-ops -- bundle-absolute-sweep --out ./out/bundle_absolute_sweep.json
```

A non-`PASS` sweep status fails closed.

## v14 terminal note

v14 adds `bundle-terminal-sweep` to remove remaining canonical consumer dependence on bundle echoes, summaries, lineage memory, and embedded export-summary traces.


## v15 note

Final canonical cache/mirror/snapshot cleanup is enforced by `bundle-ultimate-sweep` (`TerminalBundleUltimateSweepV1`).

## v16 convergence note

`bundle-convergence-sweep` finalizes canonical export convergence and blocks remaining memoized/copied/derived bundle residues in canonical flows.
