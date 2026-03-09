# Prompt Series State Snapshot

## Current anchor
- **Current anchor milestone:** `Real Compute Onboarding v2`.
- **Anchor definition:** v2 planning queue in `docs/next_10_prompts.md` (entry set `189-198`).
- **Prompt index source:** `docs/prompt_series_index.md`.

## Status against v2 anchor (v1 wrap complete)
- **v1 signoff prerequisite:** `ucf-ops v1 gate` overall PASS.
- **Transition state:** v1 is complete; v2 planning is active.
- **Queue policy:** immediate queue remains capped to 10 prompts.

## Last executed prompt / resume point
- Last executed prompt ID: **188**.
- Next prompt ID: **189**.
- Resume from: **`PROMPT 189`**.
- Numbering remains monotonic and append-only.

## Immediate next prompts (capped to 10)
Reference: `docs/next_10_prompts.md`

| Prompt ID | Title (short) | v2 class | Status |
|---:|---|---|---|
| 189 | Adapter trait hardening for Candle/Burn optional slots | MUST | pending |
| 190 | Optional feature wiring for one or two real backend slots | MUST | pending |
| 191 | Tiny real-weights fixture contract for one slot (probe-first) | MUST | pending |
| 192 | Probe-only execution path for real backend fixture | MUST | pending |
| 193 | Shadow-only rollout guard for real backend activation | MUST | pending |
| 194 | Drift/evidence parity checks for stub vs optional real backend | MUST | pending |
| 195 | Operator runbook update for optional real-backend probes | MUST | pending |
| 196 | Deterministic benchmark harness for optional backend paths | NICE | pending |
| 197 | Extended drift dashboard docs for probe/shadow comparison | NICE | pending |
| 198 | v2 phase-1 signoff gate for optional real-backend readiness | MUST | pending |

## Series control notes
- v1 completion requirement: **`ucf-ops v1 gate` overall PASS**.
- v2 real backend work is optional, probe-first, and shadow-only by default.
- Prompts are classified as MUST/NICE/DEFERRED at authoring time.
