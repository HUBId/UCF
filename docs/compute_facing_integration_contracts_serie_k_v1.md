# Serie K: Compute-facing Integration Contracts v1

Status: technical integration contract view over the current canonical compute core.

This document is intentionally narrow: it does **not** define product APIs, tenant/auth/billing,
or a platform control plane. It pins compute-facing integration contracts for adjacent UCF surfaces
onto the same final production line from Serie J.

## 1) Canonical contract classes (minimal)

Code source of truth:
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW`
  - `CANONICAL_FINAL_REFERENCE_LINE`

The integration view keeps four classes only:

1. **compute execution contract** (`outward-facing`)
   - anchor: `CanonicalComputeEntryPoint::{submit,status,drain_scheduler}`
   - semantic scope: request/job/run execution on canonical `result/fault/status` core
2. **compute diagnostics/status contract** (`outward-facing`)
   - anchor: `operations_snapshot` + `workflow_view`
   - semantic scope: top-level runtime state/freshness/drift/diagnostics signals
3. **compute evidence/reference contract** (`outward-facing`)
   - anchor: `service_surface + evidence + job_history`
   - semantic scope: evidence/snapshot/history references extending canonical run truth
4. **compute expert/internal-only contract** (`internal or high-trust only`)
   - anchor: `replay_with_entry`, `run_operation_with_entry`, compatibility backends,
     and `domains/ai*` compatibility lane
   - semantic scope: expert/internal extensions that must not become generic outward contracts

## 2) Outward-facing vs internal-only boundaries

Outward-facing integration is constrained to the three outward classes above and remains pinned to:

- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- shared invariants (`blocked!=failed!=no_op`, explicit partial/stale/caveated/degraded split)

Internal/expert-only boundaries remain explicit:

- `build_backend(kind=stub|candle)` is compatibility/dev only
- worker/internal legacy lane and `domains/ai*` stay compatibility/internal
- internal-dev operation path stays non-generic (`InternalClearReplayRegression`)

## 3) Integration signals for adjacent UCF surfaces

`CanonicalComputeEntryPoint::status_evidence_export_surface()` now provides the canonical
status/evidence export envelope for adjacent UCF subsystems.

`RuntimeOpsSnapshot::integration_signals()` remains available as a compact compatibility signal
frame and is derived from the canonical export surfaces.

- current service/runtime state (`service_state`, `runtime_mode`, `state_signal`)
- active production path context (`active_path_context`)
- constrained/caveated and degraded/unavailable top-level flags
- diagnostics availability + snapshot consistency
- evidence bundle references (`evidence_bundle_refs`)
- latest action outcomes (`latest_actions`) as controlled top-level action signals

The export/envelope and signal frame are contract adapters over canonical runtime data; they do not
introduce a separate diagnostics or monitoring platform.

## 4) Binding to Serie-J final production line

The contract view remains explicitly bound to the same final reference line and invariants from Serie J
(`CANONICAL_FINAL_REFERENCE_LINE`, `CROSS_CUTTING_PRODUCTION_INVARIANTS_V1`,
`CANONICAL_RUNTIME_HANDOFF_SEMANTICS_V1`).

No parallel integration semantics are introduced.

## 5) Deliberate limits

Kept intentionally out-of-scope:

- productized public API family
- auth/tenant/billing/governance systems
- message-bus/orchestration redesign
- second contract world beside canonical compute core

## 6) Integration-safe hook view (Serie K Prompt 3)

Source anchors:

- `runtime/ucf-compute/src/service_surface.rs`
  - `RuntimeOpsSnapshot::integration_hook_view`
  - `CanonicalComputeEntryPoint::integration_hook_view`
  - `ComputeIntegrationHook{Class,Exposure,MutationSemantics}`

Minimal hook classes are now explicit and intentionally narrow:

1. `read_only_integration_safe`
   - outward-facing
   - read-only only
   - anchored on status/evidence export adapters
2. `caveated_conditional`
   - outward-facing
   - read-only only
   - emitted only when snapshot/trust semantics are constrained or degraded
3. `expert_only`
   - expert-only boundary
   - mutating/high-trust path
   - anchored on `run_operation_with_entry(..., ExpertHighTrust)`
4. `internal_dev_test_only`
   - internal-only boundary
   - mutating path
   - anchored on `InternalClearReplayRegression`

This hook view is an integration classifier over existing runtime surfaces, not a new plugin/control-plane.

## 7) Real domain-facing consumer map (Serie M follow-up)

The compute-facing contract classes above remain the canonical boundary model. Serie M binds this
boundary to real consumers in-repo via a narrow consumer map:

- `docs/compute_consumer_integration_map_serie_m_v1.md`
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`
  - `DomainFacingConsumerAlignment`

The map keeps exactly four alignment classes:

- `aligned_canonical_outward`
- `legacy_compat_path`
- `needs_final_integration_adjustment`
- `internal_dev_test_only`

This is a small integration-tracking surface only. It does not introduce a second contract world.
