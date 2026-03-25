# Final Continuity Sweep v10

`FinalContinuityAuthorityV2` ist ab v10 der **einzige** top-level Kontinuitätsbeweis für den kanonischen Operator/Export/Build/Verify-Flow.

## Was `FinalContinuityAuthorityV2` beweist

`ucf-ops final-continuity-sweep` verankert eine einzige Autoritätskette von Governance-Inputs bis zum gebauten Bundle:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1` + `FinalGovernanceConsumerAuthorityV1`
3. `CanonicalReadinessSpineV1` + `FinalReadinessConsumerAuthorityV1`
4. `CanonicalBundleSpineV1` + `FinalBundleConsumerAuthorityV1`
5. `CanonicalPrimarySemanticsAuthorityV1` + `FinalPrimarySemanticsConsumerAuthorityV1`
6. Operator review/signoff/workflow digests
7. `CanonicalRoundTripChainV1` (subordinate contributor)
8. `CanonicalContinuityAuthorityV1` (subordinate contributor)

Der Report ist offline, deterministisch, bounded und fail-closed.

## Einziger Top-Level Proof

- **Top-level (einzig):** `final-continuity-sweep`
- **Subordinate contributors / diagnostics:**
  - `operator workflow`
  - `operator roundtrip-chain-check`
  - `continuity-authority-check`
  - `exports roundtrip-check`
  - `exports bundle-spine-check`

Diese Sub-Surfaces liefern Detaildiagnostik, aber nicht mehr die finale Autorität.

## Command

```bash
cargo run -p ucf-ops -- final-continuity-sweep --bundle <path> --out ./out/final_continuity_sweep.json
```

## Mismatch Kategorien

- `FINAL_CONTINUITY_GOVERNANCE_MISMATCH`
- `FINAL_CONTINUITY_SCOPE_MISMATCH`
- `FINAL_CONTINUITY_READINESS_MISMATCH`
- `FINAL_CONTINUITY_PRIMARY_SEMANTICS_MISMATCH`
- `FINAL_CONTINUITY_WORKFLOW_MISMATCH`
- `FINAL_CONTINUITY_BUNDLE_MISMATCH`
- `LEGACY_CONTINUITY_PROOF_PRESENT`

## Canonical Sequence

1. `cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json`
2. `cargo run -p ucf-ops -- final-readiness-consumer-sweep --out ./out/final_readiness_consumer_sweep.json`
3. `cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json`
4. `cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json`
5. `cargo run -p ucf-ops -- final-continuity-sweep --bundle <path> --out ./out/final_continuity_sweep.json`


## v11 residual-free finalization

`final-continuity-sweep` is now a legacy subordinate continuity surface.
Top-level authority moved to `residual-free-continuity-sweep` (`ResidualFreeContinuityAuthorityV1`).

## v12 update
`final-continuity-sweep` ist ab v12 explizit ein **LEGACY_TOP_LEVEL_CONTINUITY_PROOF** und nur noch diagnostisch/subordinate. Die einzige top-level Autorität ist `final-input-continuity-sweep` (`FinalInputContinuityAuthorityV1`).



## v13 update
`final-continuity-sweep` is retained only as `LEGACY_TOP_LEVEL_CONTINUITY_PROOF` diagnostics.
