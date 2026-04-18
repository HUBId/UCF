# Serie K: Canonical Compute Status/Evidence Export Surface v2

Status: narrow technical export view for adjacent UCF subsystems.

This document consolidates how other UCF subsystems should consume compute status and evidence
without coupling to compute-internal expert diagnostics structures.

## 1) Canonical export anchors

Source of truth:

- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::status_evidence_export_surface()`
  - `RuntimeOpsSnapshot::{status_export_surface,evidence_export_surface,status_evidence_export_surface}`
  - `ComputeStatusEvidenceExportSurface`

The export layer is an adapter over canonical runtime snapshot/evidence state. It does **not**
introduce a second runtime semantics world.

## 2) Canonical status export surface

`ComputeStatusExportSurface` is the outward-facing status view and is pinned to canonical runtime
snapshot/trust semantics:

- state and mode: `service_state`, `runtime_mode`, `deployment_profile`, `state_signal`
- snapshot/trust/hardening basis:
  - `snapshot_consistency`
  - `diagnostics_availability`
  - `service_trust`
  - `hardening_state`
  - `recovery_recommendation`
- top-level production context:
  - `active_path_context`
  - `active_production_line`
  - high-level health context: `worker_health`, `placement_health`, `runtime_health`
- outward caveat/degradation flags:
  - `constrained_or_caveated`
  - `degraded_or_unavailable`
  - `top_level_caveats`

## 3) Canonical evidence export surface

`ComputeEvidenceExportSurface` is the outward-facing evidence/reference view:

- `bundle_refs`: canonical evidence bundle references (`id`, `kind`, `status`, summary, trace refs)
- `trace_slice_refs`: trace references only (`slice_id`, `kind`, `status`)
- `comparison_refs`: evidence comparison references (comparison id/class + evidence/caveat refs)
- `caveat_refs`: deduplicated top-level evidence caveat references

This surface intentionally exports reference-grade evidence metadata, not full internal diagnostic
objects.

## 4) Outward-facing vs internal-only boundary

Outward-facing consumers should bind to `ComputeStatusEvidenceExportSurface` (or its status/evidence
subviews) instead of depending on deep internals such as:

- full `RuntimeOpsSnapshot` internals (`specialization`, `queue_hygiene`, `workflow_view` internals)
- full `CanonicalTraceSlice` diagnostic detail payloads
- expert/internal replay/runtime control internals

Those deeper structures remain available for expert workflows, but are not the canonical
cross-subsystem integration baseline.
