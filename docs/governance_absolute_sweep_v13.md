# Governance Absolute Sweep v13

`governance-absolute-sweep` is mandatory input for v13 supported-scope execution.

`SupportedScopeExecutionV8` is fail-closed unless `ResidualFreeGovernanceAbsoluteSweepV1` is PASS and digest-aligned with current applied/final governance inputs.

## Command

```bash
cargo run -p ucf-ops -- governance-absolute-sweep --out ./out/governance_absolute_sweep.json
```
