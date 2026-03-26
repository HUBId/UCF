# Primary Semantics Terminal Sweep (v14)

`ucf-ops primary-semantics-terminal-sweep` is the terminal v14 proof that canonical governance/readiness/bundle/review/export/interop/gate surfaces derive primary blocking/remediation semantics only from absolute residual-free final primary-semantics inputs.

## What this sweep proves

The sweep fails closed unless all terminal inputs are present and aligned:

1. canonical condition model
2. canonical remediation registry
3. `CanonicalPrimarySemanticsAuthorityV1`
4. `FinalPrimarySemanticsConsumerAuthorityV1`
5. `FinalPrimarySemanticsResidualSweepV1`
6. `ResidualFreePrimarySemanticsConsumerAuthorityV1`
7. `ResidualFreePrimarySemanticsAbsoluteSweepV1`

The emitted proof artifact is `AbsoluteFinalPrimarySemanticsTerminalSweepV1` and includes the terminal `sweep_digest`.

## Covered canonical surfaces

- governance terminal sweep output
- readiness terminal sweep output
- bundle terminal sweep output
- operator signoff output
- operator review packet output
- operator workflow output
- interop consistency matrix output
- v14 prep gate helper output (`v13_gate_report`)

## Why echoes/summaries/caches/overrides are disallowed

Canonical flows are no longer allowed to reconstruct top-level primary blocking/remediation semantics from:

- priority echoes or historical precedence snapshots
- embedded action-hint summaries
- surface-local precedence caches
- override residue / implicit fallback order
- ad-hoc mismatch-to-canonical reconstruction

These may remain only as secondary diagnostics; canonical primary semantics must come from the terminal absolute authority chain.

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-terminal-sweep --out ./out/primary_semantics_terminal_sweep.json
```
