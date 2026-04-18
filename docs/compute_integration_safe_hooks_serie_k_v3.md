# Serie K: Integration-safe Expert/Runtime Hooks v3

Status: narrow technical hook classification for compute-facing integration.

This view sharpens which expert/runtime-near hooks are safe for broader UCF surfaces without
leaking expert/internal mutation semantics.

## 1) Source anchors

- `runtime/ucf-compute/src/service_surface.rs`
  - `RuntimeOpsSnapshot::integration_hook_view()`
  - `CanonicalComputeEntryPoint::integration_hook_view()`
  - `ComputeIntegrationHook{Class,Exposure,MutationSemantics,Descriptor,View}`
- `docs/compute_facing_integration_contracts_serie_k_v1.md`
- `docs/compute_status_evidence_export_surface_serie_k_v2.md`

## 2) Hook classes (minimal, explicit)

- `read_only_integration_safe`
  - outward-facing
  - read-only only
  - canonical anchors: `status_evidence_export_surface`, `integration_signals`
- `caveated_conditional`
  - outward-facing
  - read-only only
  - active only when runtime status indicates constrained/caveated or degraded semantics
- `expert_only`
  - expert-only
  - mutating/high-trust semantics remain non-generic
  - anchor: `run_operation_with_entry(..., ExpertHighTrust)`
- `internal_dev_test_only`
  - internal-only
  - mutating semantics for dev/test path
  - anchor: `InternalClearReplayRegression`

## 3) Deliberate boundaries

- No plugin/hook control plane is introduced.
- No second diagnostics/status language is introduced.
- Outward hooks remain tied to canonical status/evidence export semantics and final reference-line
  invariants.
- Expert/internal paths remain explicit and do not become generic cross-subsystem contracts.
