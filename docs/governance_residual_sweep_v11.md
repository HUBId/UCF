# Governance Residual Sweep v11

`governance-residual-sweep` is the canonical residual-governance cleanup authority for v11.

## What changed for scope execution

`SupportedScopeExecutionV6` now consumes `FinalGovernanceResidualSweepV1` directly. Expansion is denied unless residual sweep status is `PASS` and digest prefixes align with current applied context + canonical governance authorities.

## Command

```bash
cargo run -p ucf-ops -- governance-residual-sweep --out ./out/governance_residual_sweep.json
```

`REAFFIRM_FREEZE` remains a valid success result when residual paths remain or expansion prerequisites are not uniquely satisfied.

## v12 update

v12 adds `residual-free-governance-sweep` and requires canonical governance consumers to carry residual-free authority references (`ResidualFreeGovernanceConsumerAuthorityV1`) instead of historical reconstruction hints.

## v13 follow-up

v13 finalizes canonical consumer cleanup with `governance-absolute-sweep`, requiring residual-free final governance inputs across covered governance/review/export/gate consumers.

## v14 follow-up

v14 adds `governance-terminal-sweep` to remove the final governance echo/summary/lineage traces from canonical consumers and enforce terminal absolute governance inputs.
