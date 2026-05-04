# Serie BB30 Prompt 3: First Anatomical Region Integration Line (minimal, bounded)

Status: **first anatomical region integration established** as a bounded mapping onto the existing three-region baseline.

## 1) Chosen region and minimal integration surface

The selected first anatomical region from BB30 Prompt 2 is:

- `hippocampus_like_region`

It is integrated as a narrow mapping to existing functional lines, without introducing direct authority or a new region class.

## 2) Canonical first-anatomical-region integration map

This prompt pins exactly these classes:

1. `anatomical region input surface`
2. `anatomical region state surface`
3. `anatomical region output/advisory surface`
4. `anatomical region reference surface`
5. `anatomical region to existing functional mapping`
6. `blocked/deferred anatomical region path`
7. `non-canonical/internal-only anatomical region path`

No additional anatomical platform classes are introduced.

## 3) Minimal input/state/output/reference surfaces

### 3.1 anatomical region input surface

Allowed bounded inputs:

- runtime/selection/context signals that are already canonical in BB2/BB4/BB19 lines,
- advisory reference signals via BB8/BB17/BB21-hardened reference semantics.

Explicitly blocked inputs:

- direct tool/action control signals,
- compute-internal raw states,
- direct safety override inputs,
- implicit or explicit memory mutation inputs.

### 3.2 anatomical region state surface

`hippocampus_like_region` state remains bounded to advisory context/reference shaping states only.

It must not carry execution authority state, retry authority state, direct memory commit state, or compute invocation authority.

### 3.3 anatomical region output/advisory surface

Allowed bounded outputs:

- advisory salience hint,
- advisory gating-hint,
- advisory memory-context hint,
- advisory reference-bounded signal.

Explicitly forbidden outputs:

- direct action selection,
- direct execution trigger,
- direct retry trigger,
- direct memory commit,
- direct compute invocation,
- safety override.

### 3.4 anatomical region reference surface

Reference interaction is bounded and read/interpretation-only at this layer.
It can shape caveat/salience interpretation but cannot promote reference data into direct authority.

## 4) Mapping onto existing three-region baseline

`hippocampus_like_region` complements existing Region 1/2/3 semantics by adding explicit anatomical naming for a context/reference-heavy advisory lane.

- Runtime sees the region only through bounded advisory signals.
- Selection sees the region only as non-authoritative hints.
- Reference/context layers may support the region, but only through existing bounded reference semantics.

This extends the three-region baseline without replacing it and without creating a second operational reality.

## 5) Guard/scope boundaries (unchanged)

This prompt preserves strict boundaries:

- no direct action/retry/memory/compute authority,
- no safety override semantics,
- no reopening of deferred/non-canonical paths,
- no parallel opening of a second anatomical region,
- no broad inter-region platform,
- no fourth-region opening,
- no HH production integration.

## 6) Out of scope (explicit)

Still out of scope after BB30 Prompt 3:

- full neuroanatomical simulation,
- generalized anatomical orchestration platform,
- planner/agent authority changes,
- policy/governance expansion,
- retry orchestration,
- memory persistence automation,
- direct compute-core work.
