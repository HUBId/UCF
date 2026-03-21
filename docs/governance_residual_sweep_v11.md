# Governance Residual Sweep v11

`governance-residual-sweep` is the canonical residual-governance cleanup authority for v11.

## What changed for scope execution

`SupportedScopeExecutionV6` now consumes `FinalGovernanceResidualSweepV1` directly. Expansion is denied unless residual sweep status is `PASS` and digest prefixes align with current applied context + canonical governance authorities.

## Command

```bash
cargo run -p ucf-ops -- governance-residual-sweep --out ./out/governance_residual_sweep.json
```

`REAFFIRM_FREEZE` remains a valid success result when residual paths remain or expansion prerequisites are not uniquely satisfied.
