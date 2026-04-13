# Model Slots (Local-only, hash-locked)

`ucf-compute` supports local model slots (`llm`, `world_jepa`, `world_vljepa`, `sae`, `lfm`, `ssm`, `ebm_reasoner`) via `models/manifest.toml`.

`ebm_reasoner` is also the current compatibility seam used by the optional NSR post-inference hook in the canonical compute pipeline.

## Guarantees
- no network fetch path (filesystem only)
- allowlisted root (`allowlist_root`, default `models/`)
- canonicalized path checks (reject traversal / outside root)
- max-bytes cap per slot
- SHA-256 must match expected hash
- mismatch disables slot (safe fallback to toy/stub)

## Manifest
Use `models/manifest.toml`:

```toml
allowlist_root = "models"

[slots.llm]
enabled = true
path = "llm.bin"
expected_sha256 = "<64 hex chars>"
max_bytes = 67108864
format = "candle_bin"
device = "cpu_only"
```

## Env overrides
- Canonical manifest source remains `models/manifest.toml`.
- `UCF_MODEL_MANIFEST` is a **legacy/explicit compatibility override only**; production bootstrap should keep the canonical path.

Per-slot overrides:
- `UCF_MODEL_<SLOT>_PATH`
- `UCF_MODEL_<SLOT>_SHA256`
- `UCF_MODEL_<SLOT>_MAX_BYTES`
- `UCF_MODEL_<SLOT>_ENABLED`

`<SLOT>`: `LLM`, `WORLD_JEPA`, `SAE`, `LFM`, `SSM`.

## Verify
```bash
cargo run -p ucf-ops -- models verify --manifest models/manifest.toml
```

## SHA-256
```bash
sha256sum models/llm.bin
```

Only the hash + metadata are persisted in records (not model bytes).

## Candle safetensors weight specs (v1)
For `format = "candle_safetensors"`, `ucf-compute` validates required tensor names, exact shapes, and dtypes before enabling a slot.

### `world_jepa` (JEPA v1)
Required tensors:
- `W1: [D,H] f32`
- `b1: [H] f32`
- `W2: [H,D] f32`
- `b2: [D] f32`

### `sae` (SAE v1)
Required tensors:
- `W: [F,D] f32`
- `b: [F] f32`

### `ssm` (SSM v1)
Required tensors:
- `A: [N,N] f32` (v1 uses deterministic diagonal/structured scan path)
- `B: [N] f32`
- `C: [N] f32`

### `lfm` (LFM LNN v1)
Required tensors:
- `alpha: [N] f32`
- `Wx: [N,N] f32`
- `Wu: [N] f32`
- `b: [N] f32`

### `llm` (Candle CPU v1 tiny)
Required tensors:
- `tok_emb: [32,64] f32`
- `lm_head: [64,32] f32`

Tokenizer asset (hash-locked, offline) is required for active LLM slot loading:
- default path: `runtime/ucf-compute/fixtures/llm_v1_tiny_vocab.json`
- override: `UCF_LLM_TOKENIZER_PATH`
- hash override: `UCF_LLM_TOKENIZER_SHA256`

If tokenizer hash verification fails, slot creation falls back safely to stub/toy backend.

Dimension symbols (`D/H/F/N`) are slot-local bind variables and must stay consistent across tensors in a slot.

## Runtime compatibility + failure semantics

`runtime/ucf-compute` resolves each slot into a structured runtime status:

- `used`
- `disabled`
- `unavailable`
- `verification_failed`
- `incompatible`

Failure codes are emitted per slot and distinguish:

- `disabled`
- `missing_path`
- `missing_expected_hash`
- `hash_mismatch`
- `oversized`
- `path_violation`
- `artifact_unavailable`
- `artifact_incompatible`
- `activation_blocked`

Canonical pipeline failures map these slot outcomes into explicit failure kinds:

- `artifact_unavailable`
- `artifact_verification_failed`
- `artifact_incompatible`
- `backend_disabled`
- `stage_contract_mismatch`
- `degraded_fallback`

## Canonical pack/slot lifecycle (runtime-facing)

`ucf-compute` keeps one canonical slot path and classifies it deterministically:

- `disabled`: slot disabled by manifest/env.
- `discovered`: slot enabled but artifact path is unavailable.
- `verified`: artifact path is present but failed verification constraints (hash/max/path).
- `active`: promoted artifact hash resolved + verified + compatible with selected pack/contract.
- `incompatible`: artifact verified, but format or `contract_version` is not pack-compatible.

Primary activation is always promoted-hash based (`active_hash` or `UCF_MODEL_PIN_<SLOT>`).
`ModelStore::plan_slot_activation(...)` validates the activation intent explicitly (slot, hash, contract compatibility, pin conflicts) before use.

Diagnostic side paths are intentionally non-primary:

- `UCF_MODEL_CANDIDATE_<SLOT>`
- `UCF_MODEL_COMPARE_<SLOT>`
- `UCF_MODEL_SHADOW_<SLOT>`

They are surfaced in slot provenance detail strings (hash prefixes + slot mode), but must not replace the primary active path implicitly.

## Activation / promotion failure classes (runtime mapping)

Activation planning distinguishes:

- artifact not verified (`artifact_not_verified`)
- incompatible pack/contract/backend (`incompatible_pack_contract_backend`)
- activation rejected (`activation_rejected`, e.g. pin conflict)
- active slot missing (`active_slot_missing`)
- compare/shadow path unavailable (`compare_shadow_path_unavailable`)

Pipeline admission/failure mapping continues to use the existing canonical runtime failure taxonomy (`artifact_unavailable`, `artifact_verification_failed`, `artifact_incompatible`, etc.).

## Canonical production compatibility gates

`ModelSlotProvenance` now includes a canonical `gate` payload that makes productive-use readiness explicit per slot:

- `contract_compatible`
- `slot_compatible`
- `backend_compatible`
- `placement_device_compatible`
- `promotable`
- `activatable`
- `blocked_reason`

`blocked_reason` is normalized to one of:

- `verification_failed`
- `contract_incompatible`
- `slot_incompatible`
- `backend_incompatible`
- `placement_device_worker_incompatible`
- `activation_blocked`
- `blocked_from_production_use`

Runtime rules stay narrow and technical:

- verified + compatible slots are only treated as active/usable when activation is technically valid on the canonical promoted-hash path.
- required slots that fail activation planning are marked `incompatible` with `activation_blocked` and are rejected for production pack use.
- placement/worker/device suitability remains evaluated by compute-service placement (`placement_failure` / candidate assessments), not by a second capability system.

## Canonical promotion decision semantics (technical only)

`ModelStore::slot_promotion_decision(slot)` now provides a narrow rollout/promotion classification for
the existing slot-path model (`active/candidate/compare/shadow`) without introducing approval workflows.

Canonical states:

- `known`
- `candidate`
- `comparable`
- `promotable`
- `blocked_for_promotion`
- `active`

Canonical blockers (when present):

- `not_comparable_yet`
- `insufficient_baseline_signal`
- `runtime_path_not_production_usable`
- `gate_blocked`
- `degraded_beyond_acceptable_threshold`

Signals are explicitly technical and bounded:

- baseline/compare readiness (`compare` path verified with configured hash),
- runtime-path usability (`active` path verified + warm),
- warmup/readiness state,
- degraded flag derived from configured compare path that remains blocked,
- compare/shadow diagnostic readiness (`compare_or_shadow_diagnostic_ready`),
- strict same-effective-config comparability (`comparable_under_same_effective_configuration`).

Constrained backend/device support is now surfaced as a narrow rollout view (still technical-only):

- `fully_supported`
- `supported_with_backend_device_caveat`
- `supported_only_under_guardrails`
- `blocked_for_rollout`

The same decision now carries a compact backend/device path provenance string
(`constrained_backend_device_path`, e.g. `active=warm;candidate=prepared;...`) so rollout
diagnostics can distinguish:

- candidate promotable but caveated on the current backend/device warmup path,
- guarded activation required due to backend/device readiness caveats,
- activation/promotion blocked by backend/device path constraints.

Canonical compare/shadow evaluation terms (promotion-adjacent, technical only):

- context:
  - `comparable_same_effective_configuration`
  - `comparable_with_caveats`
  - `not_comparable_different_runtime_context`
  - `blocked_missing_signals`
- compare outcome:
  - `compared_successfully`
  - `comparison_inconclusive`
  - `comparison_blocked`
  - `comparison_failed_technically`
  - `not_comparable`
- shadow outcome:
  - `shadowed_successfully`
  - `shadow_inconclusive`
  - `shadow_blocked`
  - `shadow_failed_technically`
  - `not_comparable`

Promotion linkage remains intentionally narrow and non-governance:

- `candidate_remains_blocked`
- `candidate_more_promotable`
- `candidate_comparison_inconclusive`
- `active_path_remains_preferred`

`backend_pack` provenance detail now carries this decision summary (`promotion_state`,
`promotion_transition`, `promotion_blockers`, compare/shadow context + outcomes, promotion disposition)
so runtime status/history/ops surfaces can distinguish:

- candidate became comparable,
- comparable became promotable,
- comparable but blocked,
- active from prior promotion,
- compare/shadow completed vs blocked/inconclusive/not comparable with explicit diagnostics.

Intentional boundary: this is not an approval/governance/MLOps workflow; it is a technical decision
surface over existing compatibility, rollout, and readiness signals.
