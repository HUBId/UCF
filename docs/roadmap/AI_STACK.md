# AI Stack Roadmap (canonicalized for Real Compute Onboarding)

## Canonical architecture decision (Phase A)

- `runtime/ucf-compute` is the canonical runtime model pipeline for Real Compute Onboarding.
- `domains/ai`, `domains/ai-host-abi`, and `domains/ai-backends` are retained as ABI/compatibility layers and are **not** the primary runtime pipeline path.
- Canonical model manifest path for runtime bootstrap is `models/manifest.toml`.

## Canonical docs/status/readiness split (load-bearing only)

Use exactly this split to avoid competing truth surfaces:

1. **Technical reference surface (authoritative for runtime semantics)**
   - `docs/real_compute_reference_surface_v1.md`
   - Code-pinned map: `runtime/ucf-compute/src/reference_map.rs`
2. **Status + transition surface (authoritative for current repo status and transition framing)**
   - `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md`
   - `docs/roadmap/REAL_COMPUTE_TRANSITION.md`
3. **Readiness classification surface (authoritative for stable/constrained/deferred framing)**
   - `docs/real_compute_readiness_sweep_v26.md`
4. **Roadmap context surface (this file + backend roadmap)**
   - `docs/roadmap/AI_STACK.md`
   - `docs/roadmap/AI_BACKENDS.md`

Roadmap files are context and prioritization only; they must not redefine runtime contracts,
reference lanes, or readiness authority semantics from the canonical surfaces above.

## Repository-truth status

- World-model, feature extraction, SSM, LFM, model store, capability wiring, and stage orchestration are implemented under `runtime/ucf-compute`.
- `domains/ai*` provides host-facing ABI types, adapter boundaries, and mock/placeholder backends.
- Detailed inventory and gap matrix is maintained in `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md`.

## Scope boundary for this roadmap view

- Keep this roadmap constrained to sequencing and focus areas.
- Do not duplicate detailed stage/failure/readiness semantics here.
- If wording in this file conflicts with the canonical reference/status/readiness surfaces,
  those canonical surfaces win and this file must be adjusted.
