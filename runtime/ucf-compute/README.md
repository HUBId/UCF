# ucf-compute v0 pipeline

Deterministic offline compute pipeline used by the runtime.

## Real-compute reference layer

For a compact, repo-based reference map of the canonical kernel, extension
paths, and diagnostic/test-near seams, see
`docs/real_compute_reference_surface_v1.md`.

## Capability model

The top-level runtime contract stays `AiComputeBackend`, but concrete backends are now composed from stable subtraits in `src/capabilities.rs`:

- `WorldModelPredictor` (JEPA/world prediction)
- `FeatureExtractor` (SAE/sparse spike extraction)
- `WorkingMemoryModel` (SSM/selective scan memory)
- `LlmInference` + `LlmOutput` placeholder (defined for future policy/model integration)

`ComputePipelineBackend` orchestrates these capabilities with bounded deterministic degradation.

## Profile mapping (factory)

`build_backend` wires `ComputeBackendKind` into runtime packs:

- `stub` (**compat/dev lane**): `BackendPackKind::ToyV1`
- `candle` (**compat seam**, `--features compute-candle`): `BackendPackKind::CandleToyV1`
- `burn` (**canonical production lane**, `--features compute-burn,backend-burn`): `BackendPackKind::BurnToyV1`
- `worker` (**internal execution lane**): `BackendPackKind::WorkerV1`

Legacy aliases (`cpu_stub`, `candle_dummy`, `burn_dummy`, `worker_v1`) are no longer accepted in
`UCF_COMPUTE_BACKEND`; use canonical names only.

### Canonical onboarding lane (single reference path)

- The canonical onboarding entrypoint is now pinned to Burn via
  `build_onboarding_reference_backend`
  / `build_canonical_production_backend` (`CANONICAL_ONBOARDING_PACK = BurnToyV1`).
- Canonical request/result/failure contract stays:
  `CanonicalPipelineRequest -> ComputePipelineBackend::compute_canonical -> CanonicalPipelineResult|CanonicalPipelineFailure`.
- Canonical stage sequence is fixed as `World -> SAE -> SSM -> LFM`
  (`CANONICAL_STAGE_SEQUENCE`), with honest runtime state per stage:
  - required productive core: `World`, `SAE`, `SSM`;
  - `LFM` runs when Burn LFM runtime is enabled (`lfm-burn`), otherwise backend init is
    explicitly blocked (`BackendDisabled`) — no silent fallback lane.
- `NSR` remains an optional attachment and is surfaced explicitly in `CanonicalPipelineResult.nsr_stage`
  (`disabled`, `used`, `contract_mismatch`, `verification_failed`, etc.).
- `Candle` remains a compatibility seam and is **not** a second onboarding default path.

See also `docs/compute_onboarding_reference_path.md` for the compact readiness matrix.

## Execution-device classes (bounded service placement)

`MultiWorkerComputeService` now keeps a narrow execution-device layer for placement:

- `cpu`: in-process/local execution units.
- `worker`: isolated worker execution units (`worker_ipc` path).

Per candidate, placement tracks backend suitability and device suitability separately:

- backend: `suitable|incompatible|disabled|unavailable`
- device: `suitable|unsuitable|disabled|unavailable`

This is intentionally technical and minimal. The repo does **not** introduce GPU vendor/driver
inventory or hardware orchestration in this layer.

Worker snapshots now expose a narrow registry/health signal set:
- registry identity: worker id + class (`local_primary|remote_secondary`) + role (`primary|secondary`);
- runtime health status:
  `known|ready|busy|saturated|degraded|unavailable|stale|unknown|unhealthy`;
- last health-contact timestamp, optional cooldown/quarantine-until timestamp, and
  last dispatch/error metadata.

Dispatch candidacy is tied to those health states: `degraded`, `stale`, `unknown`,
`unavailable`, `saturated`, and `unhealthy` units are explicitly skipped and reflected in
placement candidate diagnostics.

Multi-worker scheduling remains intentionally compact: jobs are either placed immediately, kept
queued as currently-unschedulable (capacity/device temporarily unavailable), or rejected when no
technical backend/device placement is possible.

## Resource classes and capacity accounting (runtime scope)

Capacity is modeled as a narrow runtime signal (not a cluster manager):

- Resource classes: `light`, `standard`, `heavy` (derived from canonical `global_work_units`).
- Class weights: `1`, `2`, `3` capacity units respectively.
- Each execution unit exposes `max_parallel_jobs * 2` capacity units.

Scheduler/admission behavior uses these signals to distinguish:

- admitted + queued due to capacity pressure,
- deferred due to transient capacity saturation,
- rejected as currently not supportable under class/capacity constraints,
- placement fallback/degradation decisions under capacity pressure.

Runtime/job provenance now includes resource class, queue/reject capacity disposition, and
capacity pressure (`nominal|saturated|overloaded`) so ops/history can separate scheduling-capacity
decisions from execution failures.

## Backend selection (runtime)

The orchestrator can be bootstrapped from env config via `RuntimeOrchestrator::try_new_from_env`.

- `UCF_COMPUTE_BACKEND=stub|candle|burn|worker`
- `UCF_COMPUTE_SEED=<u64>`
- `UCF_COMPUTE_MAX_MICROS=<u64>`
- `UCF_COMPUTE_HARD_TIMEOUT_MICROS=<u64>`

Default remains `stub` when env vars are unset (compatibility/dev-safe default).
Production callers should set `UCF_COMPUTE_BACKEND=burn` explicitly or call
`build_canonical_production_backend`.

## Candle feature extractor v0 (offline dummy weights)

`compute-candle` enables `CandleFeatureExtractor`, which performs a deterministic forward pass (`32 -> 64`) on CPU-only candle tensors using inline dummy weights.

- No HTTP, no model downloads, no external fixture pulls.
- Input vector is derived from `ComputeInput.context_digest` + world prediction digest.
- Reductions (`top-k`, sparsity, energy) are done in Rust over `Vec<f32>` for deterministic ordering.

## Offline fixture policy and constraints

- No network and no model-weight download.
- Output deterministic from `(context_digest, seed, t)`.
- Bounded outputs: capped spikes/notes and digest-only persistence for large vectors/state.

## Model manifest source

- Canonical manifest path: `models/manifest.toml`.
- Override path only via `UCF_MODEL_MANIFEST` when explicit compatibility behavior is required.

## Rollout path semantics (candidate/compare/shadow/promotion)

Runtime rollout diagnostics use a narrow canonical state set per slot path:

- `active`: selected primary hash/path for the slot.
- `candidate`: staged hash under `UCF_MODEL_CANDIDATE_<SLOT>`.
- `compare`: side-by-side compare hash under `UCF_MODEL_COMPARE_<SLOT>`.
- `shadow`: observational shadow hash under `UCF_MODEL_SHADOW_<SLOT>`.
- `disabled`: slot not configured or disabled by manifest/env.
- `blocked`: required rollout path is configured but cannot be verified.

`ModelStore::slot_path_statuses` verifies these paths against promoted artifacts and surfaces
whether a candidate/compare/shadow path is technically comparable (`verified + comparable`) or
blocked with explicit reason text.

`BackendPack` slot provenance carries this rollout digest in lifecycle details so ops/history views
can distinguish:

- active path reference,
- candidate/compare/shadow side paths,
- compare/shadow availability failures that block activation for required slots.

Boundaries (intentional):

- no approval workflow/governance engine in runtime;
- no experiment/statistics suite in rollout paths;
- promotion still uses existing compatibility gates and artifact verification, now with richer
  blocked diagnostics.

## Adding future backends

To add a real backend later without refactoring orchestrator/frame contracts:

1. Implement one or more capability traits (`WorldModelPredictor`, `FeatureExtractor`, `WorkingMemoryModel`).
2. Register capability wiring in `build_backend` for a profile.
3. Keep `AiComputeBackend` entrypoint unchanged by returning `ComputePipelineBackend`.
