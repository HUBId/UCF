# Bundle Stabilization Sweep v17

`BundleStabilizationSweepV1` proves that canonical export consumers derive authoritative bundle truth only from the converged canonical bundle chain (v16) and no longer from adapter/translation/projection paths.

## What it proves

- Canonical bundle consumers are validated against:
  - `CanonicalExportArtifactRefV1`
  - `CanonicalExportContextV1`
  - `CanonicalBundleConsumptionContextV1`
  - `CanonicalBundleSpineV1`
  - `CanonicalBundleAuthorityV2`
  - `FinalBundleConsumerAuthorityV1`
  - `FinalBundleResidualSweepV1`
  - `ResidualFreeBundleConsumerAuthorityV1`
  - `ResidualFreeBundleAbsoluteSweepV1`
  - `AbsoluteFinalBundleTerminalSweepV1`
  - `TerminalBundleUltimateSweepV1`
  - `BundleConvergenceSweepV1`
- Any remaining bundle-adapter, translation, or projection residue is surfaced as FAIL / LEGACY_PRESENT.

## Covered canonical export consumers

- repro manifest output
- bugkit manifest output
- repro verify output
- export roundtrip-check output
- export inspect summary output
- continuity sweep output
- v16 gate helper output

## Why adapters/translations/projections are blocked

These paths can provide competing bundle truth. v17 enforces fail-closed behavior with deterministic denial paths, so canonical export flow authority remains unique and auditable.

## Command

```bash
cargo run -p ucf-ops -- bundle-stabilization-sweep --out ./out/bundle_stabilization_sweep.json
```


## v18 follow-up

v18 (`bundle-final-consolidation-sweep`) removes remaining canonical facade/alias/shadow bundle residues from canonical export consumers and binds them to the stabilized chain.

## v19 closure follow-up

v19 (`bundle-closure-sweep`) performs the final closure sweep so canonical export flows no longer accept bundle wrapper/crosswalk/secondary-rendering paths as competing authority substrates.


## v20 seal-sweep note

v20 (`bundle-seal-sweep`) removes remaining canonical dependence on bundle compatibility shells, bridge layers, and auxiliary export views across canonical export consumers.
