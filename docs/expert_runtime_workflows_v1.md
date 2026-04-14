# Expert Runtime Workflows v1 (Replay / Rollout / Ops)

This document defines the **canonical expert workflow surface** for `runtime/ucf-compute` without adding a workflow engine.

## Canonical high-trust workflow classes

The runtime snapshot now exposes one explicit workflow view (`workflow_view`) with four classes:

1. `inspect_diagnose_act`
   - canonical path: `operations_snapshot -> diagnostics assessment -> runtime operation`
2. `replay_oriented`
   - canonical path: `operations_snapshot -> replay_preflight -> replay_with_entry`
3. `rollout_oriented`
   - canonical path: `operations_snapshot.rollout diagnostics -> activation/fallback/rollback action`
4. `internal_dev_test_only`
   - canonical path: `run_operation_with_entry(..., InternalDevTest)`

Each class reports transition state as one of:
- `supported`
- `partial`
- `blocked`
- `internal_only`

## Common transition semantics across replay / rollout / ops

All classes bind to existing expert contracts (no parallel contract world):

- **Entry contract** (`RuntimeEntryClass`)
- **Diagnostics contract** (`ExpertDiagnosticsAvailability`)
- **Action contract** (`RuntimeContractShape`)
- **Resulting state contract** (`CanonicalSnapshotConsistency`)

Load-bearing transitions are explicit in the snapshot:

- `snapshot_diagnostics_before_mutating_action`
- `replay_preflight_before_replay_action`
- `rollout_diagnostics_before_activation_fallback_rollback`

## Mutation boundary and safety-rail semantics (Serie F / Prompt 5)

Runtime operations now expose explicit mutation semantics per operation outcome:

- `mutation_boundary`
  - `read_only`
  - `controlled_mutable`
  - `high_impact_mutable`
  - `internal_dev_test_only`
- `mutation_result`
  - `no_mutation_read_only`
  - `state_changed`
  - `no_op`
  - `guarded_mutation`
  - `partial_effect`
  - `blocked_by_safety_rail`
  - `unsupported_in_runtime_context`
- `blocked_by` (optional safety-rail blocker)
  - `stale_diagnostic_basis`
  - `conflicting_runtime_state`
  - `subsystem_constrained_or_busy`

For mutating high-trust actions, operations additionally report:
- `intended_state_change`
- `resulting_state_change`

This keeps replay/rollout/worker/warmup context constraints explicit without
introducing a separate approval or governance platform.

## Intentional boundaries

- No workflow orchestration engine.
- No governance/admin/control-plane platform.
- No mixing with standard canonical compute submission path.
- Internal/dev/test paths remain explicit and isolated (`internal_only`).
