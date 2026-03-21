# Continuity Authority v9

`CanonicalContinuityAuthorityV1` is the final top-level continuity proof from governance entry through operator workflow/export to built bundle continuity.

## What it proves

It binds one canonical chain with digest prefixes for:

1. applied scope
2. canonical governance entry + governance authority
3. canonical readiness spine + readiness authority
4. operator review/signoff/workflow
5. canonical bundle spine + bundle authority
6. canonical roundtrip chain

Status is bounded to `PASS | FAIL | LEGACY_PRESENT` and fail-closed mismatch categories.

## Difference vs existing checks

- `operator workflow`: operator-side readiness/export stage projection.
- `operator roundtrip-chain-check`: operator-to-bundle continuity validation.
- `continuity-authority-check`: final authority object that unifies both into one canonical proof surface.

## Canonical sequence

1. `cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json`
2. `cargo run -p ucf-ops -- readiness-spine-sweep --out ./out/readiness_spine_sweep.json`
3. `cargo run -p ucf-ops -- operator workflow --latest --out ./out/operator_workflow_chain.json`
4. `cargo run -p ucf-ops -- repro-pack --run <id> --out ./out/repro_<id>.zip` (or bugkit build)
5. `cargo run -p ucf-ops -- continuity-authority-check --bundle ./out/repro_<id>.zip --out ./out/continuity_authority_check.json`

## Mismatch categories emitted via blocking codes

- `CONTINUITY_GOVERNANCE_MISMATCH`
- `CONTINUITY_SCOPE_MISMATCH`
- `CONTINUITY_READINESS_MISMATCH`
- `CONTINUITY_WORKFLOW_MISMATCH`
- `CONTINUITY_EXPORT_READY_MISMATCH`
- `CONTINUITY_BUNDLE_MISMATCH`
- `LEGACY_CONTINUITY_PATH_PRESENT`

## v10 Demotion

`continuity-authority-check` bleibt als **SUBORDINATE_CONTINUITY_CONTRIBUTOR** erhalten.
Die einzige top-level Kontinuitätsautorität ist `final-continuity-sweep` (`FinalContinuityAuthorityV2`).


## v11 note

`continuity-authority-check` is a subordinate contributor; it is not the top-level continuity proof.
The sole top-level proof is `residual-free-continuity-sweep`.
