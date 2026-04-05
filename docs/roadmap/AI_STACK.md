# AI Stack Roadmap (canonicalized for Real Compute Onboarding)

## Canonical architecture decision (Phase A)

- `runtime/ucf-compute` is the canonical runtime model pipeline for Real Compute Onboarding.
- `domains/ai`, `domains/ai-host-abi`, and `domains/ai-backends` are retained as ABI/compatibility layers and are **not** the primary runtime pipeline path.
- Canonical model manifest path for runtime bootstrap is `models/manifest.toml`.

## Repository-truth status

- World-model, feature extraction, SSM, LFM, model store, capability wiring, and stage orchestration are implemented under `runtime/ucf-compute`.
- `domains/ai*` provides host-facing ABI types, adapter boundaries, and mock/placeholder backends.
- Detailed inventory and gap matrix is maintained in `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md`.
