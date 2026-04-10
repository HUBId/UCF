# Distributed Capacity Pressure / Backpressure v1

This document defines the canonical, minimal pressure semantics used by `runtime/ucf-compute` distributed scheduling.

## Canonical pressure levels

`CapacityPressure` is interpreted per execution unit and at service level:

- `healthy`: normal headroom, no active pressure.
- `constrained`: unit is busy with reduced free capacity but still dispatchable.
- `saturated`: high utilization or backlog-induced saturation.
- `backpressured`: effective capacity exhausted under active demand.
- `temporarily_unschedulable`: admissible work exists, but no currently placeable unit can run now.

## Queueing and deferral behavior

`CapacityQueueDisposition` captures pressure-driven decisions:

- `none`: immediate placement/run.
- `queued_due_to_capacity`: local scheduler queue used due to active capacity usage.
- `deferred_due_to_capacity`: distributed placement cannot run immediately, retry later.
- `degraded_placement_due_to_pressure`: degraded/local fallback selected due to pressure or worker failure under load.
- `rejected_due_to_capacity`: no viable placement under current constraints.

## Admission / placement / retry coupling

- Admission remains canonical and technical; pressure does not create a second policy world.
- Placement evaluates a single input view that combines:
  - worker/runtime health and availability (`runtime_status`, dispatch eligibility),
  - capacity/pressure (`free_capacity_units`, `capacity_pressure`),
  - backend + device capability/suitability,
  - warmup/readiness (`warm|prepared|cold|stale|blocked`),
  - work class pressure (`light|standard|heavy` via capacity weight).
- Placement uses small deterministic heuristics (not scoring/optimization):
  - prefer technically suitable healthy candidates over marginal ones,
  - prefer warm-ready over equivalent cold/prepared paths,
  - avoid saturated/backpressured paths when comparable alternatives exist,
  - use local/remote fallback only with explicit technical cause.
- Retry/redispatch uses existing coordination signals and can mark degraded pressure-driven fallback explicitly.

## Visibility surface

- Per-worker snapshots include used/free units and derived `capacity_pressure`.
- Service-level pressure can be queried via distributed pressure snapshot (`pressure_snapshot`).
- Distributed degradation/recovery can be queried with `distributed_recovery_snapshot()`:
  - state: `healthy|partially_degraded|constrained_but_serviceable|recovery_in_progress|unrecoverable_unavailable`
  - unit sets: `placement_eligible_units`, `excluded_units`, `recovered_units`
  - impact counters: `queued_jobs`, `uncertain_jobs`, `recovery_required_jobs`
- Job records/history retain pressure and queue disposition fields for provenance.
- Placement provenance now exposes:
  - selected outcome (`optimal|constrained_valid|degraded_valid|queued/deferred constrained|hard_incompatible`),
  - up to 1-3 decisive signals (for example warmup, runtime pressure, locality/backend fit),
  - explicit reason text for queue/defer/fallback/degraded decisions.

## Queue / defer / fallback / degraded semantics

- `queued_due_to_capacity`: admissible work is held to wait for a better placement path.
- `deferred_due_to_capacity`: all currently suitable paths are constrained/temporarily unavailable.
- `degraded_placement_due_to_pressure`: degraded-but-valid path was selected (e.g. local redispatch after worker pressure/failure).
- `rejected_due_to_capacity`: no acceptable placement was available under current hard constraints.

## Non-goals

- No global optimization scheduler.
- No autoscaling/quota/pricing/governance layer.
- No advanced queueing-theory engine.
