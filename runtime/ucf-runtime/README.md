# ucf-runtime

## Optional Consolidation/Geist hooks (v0)

The runtime orchestrator keeps ESS append as source-of-truth and can optionally run derived hooks after decision append.

### Flags

- `UCF_ENABLE_CONSOLIDATION_HOOK=true` enables bounded compute milestone aggregation.
- `UCF_ENABLE_GEIST_HOOK=true` enables conservative Geist macro-baseline updates from milestones.

Both hooks are disabled by default.

### Behavior

- Consolidation hook consumes decision ESS records and projects bounded `ComputeSignalSummaryView` values.
- Milestones are bounded aggregates (windowed means/counters/top-k digests only).
- Geist hook consumes milestones and persists only accepted stable updates (drift/degraded gating).
- Hook failures are counted and skipped without affecting the control loop.
