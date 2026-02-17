# Operator Run Registry & Session Graph

## Session graph

Each run persists `RunMetadataRecord` under `.ucf/ess/runs/<run_id>.json` and contains:

- `run_id`
- `parent_run_id` (optional parent link)
- `resume_reason` (`operator_resume`, `crash_recovery`, `fallback`, `upgrade`)
- `compat_digest` (hash over policy bundle hash + backend pack digest + model hashes digest + schema versions)

This forms a parent/child graph for resumes, fallback and upgrade transitions.

## Resume safety

Resume compatibility is strict:

1. `policy_bundle_hash` must match.
2. `backend_pack_meta_digest` must match.
3. `model_hashes_digest` must match when enabled features indicate real slots are enabled.
4. Schema versions must match for required schema keys.

If any check fails, a new child run is created with `resume_reason=upgrade` and parent pointer set.

## Registry workflows

- `ucf-ops runs list --last 50`
- `ucf-ops runs show --run <id>`
- `ucf-ops runs search --pack <prefix> --policy <prefix> --model <prefix>`

Registry is rebuildable from ESS run metadata files under `.ucf/ess/runs/`.

## Status workflow

Use `ucf-ops status --run <id>` to inspect:

- active slots
- governor tier and score
- emergency state
- last 8 ticks for pressure/surprise/uncertainty/risk
- bounded issuance deny kinds + reasons

## Common operations

### Resume after crash

1. Restart `ucf-ops bringup ...`.
2. The new run links to previous via `parent_run_id`.
3. If compatibility fails, new child session is forced.

### Enablement rollout

When policy/pack/model hashes change for rollout, new child runs are created with mismatch reasons.

### Fallback execution

Fallback/upgrade transitions are represented by creating child runs from the previous run.
