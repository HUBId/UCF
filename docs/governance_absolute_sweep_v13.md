# Governance Absolute Sweep v13

`governance-absolute-sweep` is mandatory input for v13 supported-scope execution.

`SupportedScopeExecutionV8` is fail-closed unless `ResidualFreeGovernanceAbsoluteSweepV1` is PASS and digest-aligned with current applied/final governance inputs.

## Command

```bash
cargo run -p ucf-ops -- governance-absolute-sweep --out ./out/governance_absolute_sweep.json
```

v14 ergänzt darauf den terminalen Consumer-Sweep (`governance-terminal-sweep`) und blockiert die letzten Echo-/Summary-/Lineage-Reste in kanonischen Governance-Consumer-Flows.

## v15 note

v15 finalizes consumer-side cleanup: canonical flows no longer admit governance cache/mirror/embedded snapshot traces as primary authority once ultimate sweep enforcement is active.


## v16 convergence note

v16 closes remaining governance residue by requiring canonical consumers to source governance truth only from the terminal canonical governance authority chain and by rejecting memoized governance mirrors in canonical flow checks.
