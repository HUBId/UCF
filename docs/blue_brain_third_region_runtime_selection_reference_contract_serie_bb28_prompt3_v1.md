# Blue Brain Third-Region Runtime/Selection/Reference Contract Line (Serie BB28 Prompt 3)

## 1) Scope and intent

This pass hardens exactly one bounded third-region interface for the class selected in BB28 prompt 1 and surfaced in BB28 prompt 2:

- `Runtime-feedback-integration-related`

No planner/agent platform, no governance expansion, no retry orchestration, no fourth region, and no compute-core extension are introduced.

## 2) Canonical third-region contract map

The canonical third-region contract map is intentionally narrow and includes exactly:

- `region-3-to-runtime advisory signal`
- `runtime-to-region-3 bounded input`
- `region-3-to-selection advisory signal`
- `selection-to-region-3 bounded state input`
- `region-3-reference signal`
- `caveated/deferred/blocked region-3 contract signal`
- `reference-only region-3 contract signal`
- `non-canonical/internal-only region-3 contract path`

These classes are the only canonical contract classes for the Region-3 runtime/selection/reference seam.

## 3) Runtime semantics (Region 3)

Runtime may consume only bounded Region-3 contract inputs.

Allowed read semantics:
- advisory-only runtime posture hints,
- caveated/deferred/blocked boundary markers,
- bounded reference validity hints.

Explicitly excluded semantics:
- no direct execution authority,
- no direct compute authority,
- no action-trigger authority,
- no retry-trigger authority.

Region-3 complements Region-1 and Region-2 by carrying runtime-feedback integration hints; it does not replace Region-1 attention/selection advisory semantics nor Region-2 context/reference quality semantics.

## 4) Selection semantics (Region 3)

Selection may read bounded Region-3 advisory and bounded state inputs only.

Allowed read semantics:
- advisory priority/deferral/caveat hints,
- deferred/blocked separation,
- reference-aware caveat posture.

Explicitly excluded semantics:
- no direct action selection authority,
- no proposal-authority escalation,
- no planner/autonomous policy behavior.

## 5) Reference/context semantics (Region 3)

Region-3 reference consumption remains canonical only via bounded reference signals.

Rules:
- reference signals can inform bounded context/reference basis,
- stale/caveated references remain caveated and cannot be promoted,
- reference-only remains reference-only and not execution-support authority,
- no implicit memory persistence is created from Region-3 references.

No second reference reality is introduced.

## 6) Deferred/blocked/caveat boundary sharpening

For Region-3 contract signals the distinctions are explicit and preserved:

- `deferred != blocked`
- `blocked != failed execution`
- `caveated != strong region-3 signal`
- `reference-only != operative support basis`

These distinctions are shared across runtime/selection/reference readings.

## 7) Bounded dynamics coupling status

Region-3 remains unlinked from any direct production dynamics-control authority.

If advisory-only bounded dynamics hints appear, they remain distinct from contract authority and cannot drive direct control.

No HH productive integration is opened.

## 8) No-direct-* and out-of-scope guards

The Region-3 contract seam keeps these hard boundaries:

- no direct action trigger,
- no direct execution trigger,
- no direct retry trigger,
- no direct memory commit,
- no direct compute invocation,
- no safety override,
- no fourth-region opening,
- no broad inter-region platform.

## 9) Functional complement role

Region-3 is embedded as a complementary third region:

- Region 1: attention/selection advisory lane,
- Region 2: context/reference quality lane,
- Region 3: runtime-feedback integration lane.

This is bounded and canonical-path-only by design.
