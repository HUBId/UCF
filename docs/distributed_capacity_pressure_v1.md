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
- Placement evaluates worker suitability plus free-vs-used capacity units.
- Retry/redispatch uses existing coordination signals and can mark degraded pressure-driven fallback explicitly.

## Visibility surface

- Per-worker snapshots include used/free units and derived `capacity_pressure`.
- Service-level pressure can be queried via distributed pressure snapshot (`pressure_snapshot`).
- Job records/history retain pressure and queue disposition fields for provenance.

## Non-goals

- No global optimization scheduler.
- No autoscaling/quota/pricing/governance layer.
- No advanced queueing-theory engine.
