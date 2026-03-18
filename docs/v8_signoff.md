# v8 Signoff Gate (Governance/Scope/Readiness/Bundle Continuity)

## Purpose
The v8 gate is the final hardening checkpoint for the v8 phase. It certifies governance, supported-scope execution coherence, readiness spine consistency, bundle spine canon, remediation spine proof, and full operator→export→bundle continuity.

This gate is **offline**, **hardware-neutral**, and **bounded**. It does not certify broader compute capability.

## Command
```bash
cargo run -p ucf-ops -- v8 gate --out ./out/v8_gate_report.json
```

## Exit codes
- `0`: PASS
- `2`: FAIL

## PASS guarantees
A PASS means all required v8 governance/scope/readiness/export continuity checks are coherent:
- canonical governance entry is authoritative and consumed consistently
- supported-scope execution artifact is present, explicit, and current against applied scope
- readiness spine is shared and consistent across operator surfaces
- bundle spine is canonical
- remediation spine proof has no mismatches
- operator round-trip chain is coherent from operator surfaces through export bundle artifacts
- artifact schema snapshots and portability/docs checks are clean

## PASS does **not** guarantee
A PASS does not imply:
- wider runtime capability
- more slots/backends are production-ready
- automatic slot/backend activation
- GPU readiness, remote compute readiness, or training readiness

## PASS / FAIL / SKIP interpretation
- **PASS**: check is required and coherent, or optional path is present and coherent.
- **FAIL**: required surface missing or inconsistent; optional surface present but inconsistent.
- **SKIP**: optional path is unsupported or absent under current applied scope.

## Scope authority note
The authoritative scope for this gate is the **current applied supported scope**, derived from applied-scope context and the latest authoritative supported-scope execution artifact.

## Phase framing
v8 is a governance/scope/readiness/export-continuity hardening phase. It is not a compute-feature expansion phase.

## Post-v8 continuation note
After `ucf-ops v8 gate` PASS, continue at Prompt 260 via `docs/next_10_prompts.md`.
