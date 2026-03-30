# Terminal Absolute Final-Input Continuity v14

`TerminalAbsoluteFinalInputContinuityAuthorityV1` is the single top-level terminal proof for canonical operator/export/build/verify continuity after v14 hardening.

## What it proves
- One unique terminal absolute residual-free final-input chain from applied governance inputs to bundle artifacts.
- Terminal authorities are all included and aligned:
  - `AbsoluteFinalGovernanceTerminalSweepV1`
  - `AbsoluteFinalReadinessTerminalSweepV1`
  - `AbsoluteFinalBundleTerminalSweepV1`
  - `AbsoluteFinalPrimarySemanticsTerminalSweepV1`
- `OperatorReviewPacketV1`, `OperatorSignoffDecisionV1`, `OperatorWorkflowChainV1`, and `CanonicalRoundTripChainV1` are subordinate contributors whose digests are bound into the same terminal authority.
- Any residual-path dependency, hidden prerequisite, or legacy parallel top-level continuity surface is fail-closed.

## Sole top-level command

```bash
cargo run -p ucf-ops -- terminal-absolute-final-input-continuity-sweep --bundle <path> --out ./out/terminal_absolute_final_input_continuity_sweep.json
```

## Canonical mismatch categories
- `TERMINAL_FINAL_INPUT_GOVERNANCE_MISMATCH`
- `TERMINAL_FINAL_INPUT_SCOPE_MISMATCH`
- `TERMINAL_FINAL_INPUT_READINESS_MISMATCH`
- `TERMINAL_FINAL_INPUT_PRIMARY_SEMANTICS_MISMATCH`
- `TERMINAL_FINAL_INPUT_WORKFLOW_MISMATCH`
- `TERMINAL_FINAL_INPUT_BUNDLE_MISMATCH`
- `RESIDUAL_PATH_DEPENDENCY_PRESENT`
- `LEGACY_TOP_LEVEL_CONTINUITY_PRESENT`

## Canonical operator/export/build/verify sequence
1. `operator review-packet`
2. `operator signoff`
3. `operator workflow`
4. Build/export bundle artifacts
5. `terminal-absolute-final-input-continuity-sweep`
6. Optional subordinate diagnostics (`absolute-final-input-continuity-sweep`, `final-input-continuity-sweep`, `residual-free-continuity-sweep`, `operator roundtrip-chain-check`)


## v15 update
`TerminalAbsoluteFinalInputContinuityAuthorityV1` is now a **SUBORDINATE_CONTINUITY_CONTRIBUTOR**.
The sole top-level continuity proof is `UltimateTerminalAbsoluteFinalInputContinuityAuthorityV1` via `ultimate-terminal-absolute-final-input-continuity-sweep`.

> v16 update: canonical top-level continuity proof is now only `canonical-convergence-continuity-sweep` (`CanonicalConvergenceContinuityAuthorityV1`). This surface is subordinate continuity evidence.

## v17 update
`TerminalAbsoluteFinalInputContinuityAuthorityV1` remains subordinate continuity evidence.
Canonical top-level continuity PASS/FAIL is delegated only to `canonical-stabilization-continuity-sweep` (`CanonicalStabilizationContinuityAuthorityV1`).
