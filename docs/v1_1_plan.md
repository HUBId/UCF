# v1.1 Plan: Real Models, Optional GPU Lane, and Compatibility Guardrails

Status: planning/scaffolding only (no training rollout in this document).

## Scope and Constraints

This document extends the stabilized `v1.0-rc1` baseline to `v1.1` with real-model integration and operational maturity while preserving determinism and offline-core guarantees.

### Preconditions

- `v1.0-rc1` is stabilized and tagged.
- `ModelSlot` + `WeightSpec` strict validation is already active.
- Compute contracts and policy graph versioning are in place.

### Hard Requirements (Non-negotiable)

1. No contract break in-place. If semantics change, introduce explicit V2 contracts.
2. Weight promotion and rollback are first-class operational workflows.
3. GPU is optional and must not be required for core correctness.
4. Offline core remains functional without network dependency.
5. Security/sandboxing and determinism controls remain enforced.

## Architecture Principles for v1.1

- **Stable default path:** keep CPU deterministic paths as default/reference.
- **Layered rollout:** stage all model/backend changes via shadow + readiness gates before promotion.
- **Separation of concerns:** contracts define policy-facing semantics; compute backends can evolve behind traits.
- **Evidence-first ops:** every promotion/rollback is logged with digests and gate evidence.

---

## Phase A — Weights Strategy v1.1

### A1) Artifact Layout

Canonical model artifact directories:

- `models/staging/` — candidate weights under evaluation.
- `models/promoted/` — approved weights eligible for manifest references.

Manifest policy:

- Runtime manifests reference **only** artifacts under `models/promoted/`.
- Staging hashes are never directly used for production decisions.

### A2) Promotion Workflow

Promotion state machine (per slot/hash):

1. **Probe** — integrity + shape/spec checks (`WeightSpec` strictness).
2. **Shadow** — run in mirrored path without policy control impact.
3. **Readiness Gate** — latency/error envelope and stability checks.
4. **Signoff** — explicit operator approval.
5. **Promote** — atomically update promoted manifest pointer.

Required evidence bundle for promotion:

- Model hash and manifest digest.
- Probe report.
- Shadow evaluation report.
- Readiness gate output.
- Signoff record (operator + timestamp + rationale).

### A3) Rollback Strategy

- Keep last `N` promoted manifests per slot (configurable retention).
- Support emergency pinning via environment override to a known hash.
- Rollback operation must be atomic and auditable.

Emergency pinning guardrails:

- Pin must reference a known promoted hash.
- Startup logs must show active pin state.
- Readiness gate should report pinned mode in output metadata.

### A4) Ops Tooling (CLI)

Planned commands:

- `ucf-ops models promote --slot <slot> --hash <hash>`
- `ucf-ops models rollback --slot <slot> --to <hash>`

Recommended related commands (same epic):

- `ucf-ops models list --slot <slot>`
- `ucf-ops models history --slot <slot>`
- `ucf-ops models pin --slot <slot> --hash <hash>`
- `ucf-ops models unpin --slot <slot>`

### A5) Model Incident Runbook (Required)

See: `docs/runbooks/weights_incident_response.md`.

---

## Phase B — VL-JEPA Adapter (World Model v1.1)

### B1) Contract Continuity

Preferred path: keep `WorldModelPredictor` V1 output semantics unchanged:

- `prediction_error_q`
- `prediction_digest`

If output semantics remain equivalent, do **not** introduce V2.

### B2) Slot Addition

Add slot:

- `ModelSlot::WorldVljepa`

### B3) WeightSpec Definition

Define VL-JEPA `WeightSpec` including:

- Minimal required tensor set.
- Required tensor dimensions/ranges.
- Input encoding contract expectations.
- Quantization/path compatibility expectations for policy path.

### B4) Rollout Discipline

- Start in shadow mode only.
- Must satisfy envelope and latency budgets before promotion.
- Readiness gate must include VL-JEPA-specific checks.

### B5) Contract V2 Trigger Rule

Introduce V2 only if policy-visible semantics must change, never for implementation convenience.

---

## Phase C — SAE Real (v1.1)

### C1) Backend Evolution

Replace linear SAE placeholder with real sparse autoencoder backend while preserving policy-facing shape:

- Output spikes remain `(id, magnitude_q)`.

### C2) Scale Envelope

- Allow larger feature space `F`.
- Keep top-`K` spike cap bounded for deterministic policy cost.

### C3) Performance Option

- Evaluate precomputed quantization tables to reduce runtime overhead.

### C4) Gate Criteria

- Latency within configured budget.
- Sparsity profile stability.
- No contract-semantic drift without explicit V2.

---

## Phase D — SSM Kernels (v1.1)

### D1) Contract Stability

Keep SSM outputs stable:

- `pressure_q`
- `state_digest`

### D2) Backend Pluggability

Implement kernel strategy behind trait boundary:

- Baseline CPU scan (reference).
- Optimized CPU kernels (SIMD).
- Optional GPU kernel.

### D3) Parity/Correctness Testing

- Optimized backends must stay within accepted numeric envelope.
- Deterministic modes should preserve digest equality when required.
- Non-deterministic accelerators use envelope checks for readiness decisions.

---

## Phase E — GPU Lane (Optional, Non-blocking)

### E1) Feature Flags and CI Separation

Introduce optional features:

- `gpu-cuda`
- `gpu-metal`

CI policy:

- GPU lane is separate and non-blocking for core correctness lanes.
- CPU/offline deterministic lane remains required gate.

### E2) Determinism Policy

- Expect minor drift on GPU outputs.
- Use envelope comparison instead of exact digest equality for GPU parity.
- Quantize to fixed-point before policy path usage.

### E3) Sandbox/Isolation

GPU lane must still respect:

- Resource limits.
- No external IO requirement for core checks.
- Same security posture as CPU lane where applicable.

---

## Phase F — Backwards Compatibility Guardrails

### F1) Contracts

- `StageContractVersion::V1` remains default in v1.1.
- Add `V2` only for explicit semantic changes.

### F2) Frames and Schemas

For `ControlFrame`, `DecisionFrame`, `ExperienceRecord`:

- Version increments only with explicit compat strategy.
- Document migration path and dual-read/write where needed.

### F3) Ops Pinning for Reproducibility

Per run, persist and expose:

- Policy graph digest.
- Model hash digests (all active slots).
- Pin overrides (if any).

---

## Compatibility Risk Table

| Module / Area | Planned v1.1 Change | Contract Impact | Default Version | V2 Needed? | Mitigation / Guardrail |
|---|---|---|---|---|---|
| Weights manifests & ops | staging/promoted + promote/rollback/pin workflows | None to compute output contract | V1 | No | Atomic promotion, signed evidence, manifest retention |
| World model (VL-JEPA slot) | Add `ModelSlot::WorldVljepa`; shadow rollout | Prefer no output semantic change (`prediction_error_q`, `prediction_digest`) | V1 | Only if semantics change | Shadow-first, latency+envelope gates |
| SAE backend | Linear -> real sparse AE | Keep spike tuple format `(id, magnitude_q)` | V1 | Only if spike semantics change | Top-K bound, quantized outputs |
| SSM kernels | Baseline + SIMD + optional GPU backend | Keep `pressure_q`, `state_digest` shape/meaning | V1 | No (unless semantics drift) | Trait boundary + parity tests |
| GPU optional lane | New CI lane + feature flags | No required core-contract change | V1 | No | Non-blocking CI, envelope parity, fixed-point quantization |
| Frame schemas | Potential metadata extension for digests/pins | Schema/versioned compatibility path | V1 | Maybe (schema-specific) | Explicit migration docs, compat read strategy |

---

## Test and Benchmark Requirements

### Required for v1.1 readiness

1. **Weights ops tests**
   - Promotion/rollback atomicity.
   - Manifest retention (`N`) and history integrity.
   - Pin/unpin behavior and audit metadata.

2. **Shadow rollout tests**
   - VL-JEPA slot shadow execution and probe evidence.
   - Readiness gate inclusion and threshold behavior.

3. **Contract stability tests**
   - V1 output snapshot/golden checks for unchanged semantics.
   - V2 introduction tests only when explicitly justified.

4. **SSM parity tests**
   - Baseline CPU vs SIMD envelope checks.
   - Optional GPU vs CPU envelope checks.

5. **Determinism/security checks**
   - CPU deterministic digest reproducibility.
   - Sandbox/resource policy adherence.

### Benchmarks (minimum)

- Per-slot latency p50/p95/p99 (CPU baseline and optimized backends).
- Memory footprint impact per model slot.
- Throughput comparison in shadow and promoted modes.

---

## Phase G — Prompt Sequence (Implementation-Ready)

1. **Prompt 121** — Weights promotion/rollback ops tooling.
2. **Prompt 122** — VL-JEPA slot scaffolding + `WeightSpec`.
3. **Prompt 123** — VL-JEPA shadow rollout + probes.
4. **Prompt 124** — SAE real slot spec + backend.
5. **Prompt 125** — SSM optimized kernel lane + parity tests.
6. **Prompt 126** — GPU lane scaffolding (non-blocking CI).
7. **Prompt 127** — v1.1 readiness gate extension + signoff flow.

Each prompt must preserve V1 contract behavior unless it explicitly introduces and justifies V2.

---

## Acceptance Criteria Checklist (Plan Completion)

- [x] `docs/v1_1_plan.md` includes weight strategy.
- [x] Slot-by-slot milestones are defined.
- [x] Contract/version impact is assessed.
- [x] Test/bench requirements are defined.
- [x] Backwards compatibility guardrails documented.
- [x] GPU lane is optional/non-blocking.

---

## Appendix A — Weights Incident Response

Reference runbook: `docs/runbooks/weights_incident_response.md`.

Incident classes:

- Bad promotion (quality regression).
- Spec mismatch (tensor/shape mismatch).
- Latency budget breach.
- Determinism drift in policy-facing quantized path.

Immediate actions:

1. Freeze further promotions for affected slot.
2. Pin to last known-good promoted hash.
3. Roll back promoted pointer if pinning insufficient.
4. Re-run readiness gate with incident profile.
5. Collect evidence bundle and update incident ticket.

Exit criteria:

- Stable gate pass on known-good hash.
- Clear root cause and preventative action recorded.
- Incident runbook artifacts attached and archived.


## Operational references
- Weights lifecycle: `docs/weights_lifecycle.md`
- Incident runbook: `docs/runbooks/model_incident.md`
