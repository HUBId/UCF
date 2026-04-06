# Real Compute Onboarding: Canonical E2E Reference Path

Status: active, load-bearing, repo-backed.

## Single canonical onboarding path

- Request type: `CanonicalPipelineRequest`.
- Runtime entrypoint: `build_onboarding_reference_backend(seed)` + `compute_canonical(request)`.
- Fixed stage order: `World -> SAE -> SSM -> LFM`.
- Typed output: `CanonicalPipelineResult` with structured state/failure and stage/backend/artifact provenance.

This is intentionally one path only. It is pinned to Burn (`BurnToyV1`) and does not silently
promote Candle/Stub/Worker as alternative onboarding defaults.

## Honest readiness matrix (for the canonical onboarding path)

| Surface | Status | Evidence in code |
|---|---|---|
| World/JEPA | real (required) | Burn pack wires `BurnWorldPredictor` for `BurnToyV1` and blocks on missing/incompatible slot provenance. |
| SAE | real (required) | Burn pack wires `BurnSaeExtractor` and enforces slot/contract compatibility. |
| SSM | real (required) | Burn pack wires `BurnSsmKernel` and enforces slot/contract compatibility. |
| LFM | real when `lfm-burn`; otherwise blocked | `BurnToyV1` requires Burn LFM kernel behind `lfm-burn`; without it backend build returns `BackendDisabled`. |
| NSR | optional hook (default disabled) | Canonical result always includes `nsr_stage`; default mode is disabled unless explicitly enabled via env. |
| Burn | primary | `CANONICAL_ONBOARDING_BACKEND = Burn`, `CANONICAL_ONBOARDING_PACK = BurnToyV1`. |
| Candle | compatibility seam | available through `build_backend(kind=candle)`; not used by onboarding reference builder. |

## Failure semantics required on the reference path

The canonical path reports structured failure categories through `CanonicalPipelineFailure.kind`,
including:

- invalid request
- artifact unavailable
- artifact verification failed
- incompatible slot/contract/backend
- stage unavailable / backend disabled
- execution error
- degraded but usable completion

## Why this path is the reference

It is the smallest real load-bearing E2E path in the current repo that:

1. is stage-typed and deterministic,
2. carries model/backend provenance in the top-level typed result,
3. keeps optional/blocked surfaces explicit instead of claiming full multi-path parity.
