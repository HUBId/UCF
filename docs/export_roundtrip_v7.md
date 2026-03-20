# Export Round-Trip Consistency v7

`ucf-ops exports roundtrip-check` validates that canonical export bundles reconstruct one shared consumption context for verify/inspect flows.

## Command

```bash
cargo run -p ucf-ops -- exports roundtrip-check --in <bundle.zip> --out ./out/export_roundtrip_check.json
```

The command auto-detects Repro Pack vs BugKit, rebuilds `CanonicalBundleConsumptionContextV1`, and emits `BundleRoundTripConsistencyV1`.

## What “round-trip consistency” means

For the same exported bundle bytes, bundle consumption must deterministically recover the same:

- applied scope digest prefix
- policy graph digest prefix
- model manifest digest prefix
- related artifact reference semantics and included-state interpretation
- governance/review/signoff reference consistency

## Mismatch categories

- `BUNDLE_SCOPE_MISMATCH`: applied scope digest context differs.
- `BUNDLE_POLICY_MISMATCH`: policy digest context differs.
- `BUNDLE_MANIFEST_MISMATCH`: manifest/context digest does not reconstruct canonically.
- `BUNDLE_ARTIFACT_REF_MISMATCH`: artifact reference digest/state mismatch.
- `BUNDLE_INCLUDED_STATE_MISMATCH`: included artifact semantics are inconsistent.
- `LEGACY_BUNDLE_LAYOUT`: legacy export layout detected.
- `LEGACY_BUNDLE_TRANSLATED`: deterministic legacy translation path was used.
- `LEGACY_BUNDLE_UNSUPPORTED`: legacy bundle could not be safely translated (fail closed).

All checks are offline, deterministic, bounded, and read-only.


## Canonical remediation linkage

`BundleRoundTripConsistencyV1` now also includes:

- `canonical_condition_codes`
- `primary_remediation_codes`

so roundtrip mismatch codes are explicitly normalized back into canonical condition/remediation semantics. The v7 cross-surface proof command is:

```bash
cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json
```

See also: `docs/operator_export_authority_chain_v7.md` and `ucf-ops operator export-chain-check` for applied-scope authority validation across review/signoff/workflow/export chain.

## Bundle Spine (v8)

Als kanonischer End-to-End-Nachweis für Bundle-Kohärenz wird zusätzlich `exports bundle-spine-check` verwendet:

```bash
cargo run -p ucf-ops -- exports bundle-spine-check --in <bundle.zip> --out ./out/bundle_spine_check.json
```


## v8 continuity
Use `operator roundtrip-chain-check` as the top-level proof that bundle roundtrip aligns with operator governance/readiness/workflow state.


## v9 update
Roundtrip validation is now part of the canonical bundle spine sweep (`exports bundle-spine-sweep`) and cannot act as an alternate bundle authority path.


## v10 finalization

v10 finalizes universal bundle-input authority for canonical export consumers via `ucf-ops final-bundle-consumer-sweep`.
