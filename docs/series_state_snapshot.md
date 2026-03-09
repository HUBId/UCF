# Prompt Series State Snapshot

## Current anchor
- **Current anchor milestone:** `Real Compute Onboarding v2`.
- **Anchor definition:** v2 planning queue in `docs/next_10_prompts.md` (entry set `189-198`).
- **Prompt index source:** `docs/prompt_series_index.md`.

## Status against v2 anchor (v2 gate-controlled)
- **v1 signoff prerequisite:** `ucf-ops v1 gate` overall PASS.
- **v2 completion rule:** v2 is complete only when `ucf-ops v2 gate` overall PASS.
- **Transition state:** v1 is complete; v2 execution is gated by v2 signoff.
- **Queue policy:** immediate queue remains capped to 10 prompts.

## Last executed prompt / resume point
- Last executed prompt ID: **198**.
- Next prompt ID: **199**.
- Resume from: **`PROMPT 199`**.
- Numbering remains monotonic and append-only.

## Immediate next prompts (capped to 10)
Reference: `docs/next_10_prompts.md`

| Prompt ID | Title (short) | v2 class | Status |
|---:|---|---|---|
| 189 | Adapter trait hardening for Candle/Burn optional slots | MUST | complete |
| 190 | Optional feature wiring for one or two real backend slots | MUST | complete |
| 191 | Tiny real-weights fixture contract for one slot (probe-first) | MUST | complete |
| 192 | Probe-only execution path for real backend fixture | MUST | complete |
| 193 | Shadow-only rollout guard for real backend activation | MUST | complete |
| 194 | Drift/evidence parity checks for stub vs optional real backend | MUST | complete |
| 195 | Operator runbook update for optional real-backend probes | MUST | complete |
| 196 | Deterministic benchmark harness for optional backend paths | NICE | complete |
| 197 | Extended drift dashboard docs for probe/shadow comparison | NICE | complete |
| 198 | v2 phase-1 signoff gate for optional real-backend readiness | MUST | complete |

## Series control notes
- v1 completion requirement: **`ucf-ops v1 gate` overall PASS**.
- v2 completion requirement: **`ucf-ops v2 gate` overall PASS**.
- Next anchor movement is blocked until v2 gate PASS is recorded.
- v2 real backend work is optional, probe-first, and shadow-only by default.
- Prompts are classified as MUST/NICE/DEFERRED at authoring time.

## Tiny real fixture coverage (v2)

- First supported slot: `world_jepa` (from previous prompt chain).
- Second supported slot in this stage: `sae` (chosen over `ssm` for lower-invasiveness and already-auditable probe contract).
- `sae` remains **shadow-only** at this stage; active requests are denied with `ACTIVE_NOT_ENABLED_FOR_SLOT_STAGE`.
