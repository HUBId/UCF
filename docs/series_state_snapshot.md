# Prompt Series State Snapshot

## Current anchor
- **Current anchor milestone:** `Real Compute Onboarding v1`.
- **Anchor definition:** v1 transition queue in `docs/next_10_prompts.md` (entry set `178-187`).
- **Prompt index source:** `docs/prompt_series_index.md`.

## Status against v1 anchor (transition from completed v0)
- **v0 completion prerequisite:** `ucf-ops v0 gate` overall PASS in CI.
- **Transition state:** v0 signoff is treated as complete; planning focus moved to v1 scaffolding.
- **Queue policy:** immediate queue remains capped to 10 MUST prompts.

## Last executed prompt / resume point
- Last executed prompt ID: **177**.
- Next prompt ID: **178**.
- Resume from: **`PROMPT 178`**.
- Numbering remains monotonic and append-only.

## Immediate next prompts (capped to 10, MUST-only)
Reference: `docs/next_10_prompts.md`

| Prompt ID | Title (short) | v1 class | Status |
|---:|---|---|---|
| 178 | Weights lifecycle scaffold (staging/promoted, no real weights required) | MUST | pending |
| 179 | Hardware-neutral backend adapter traits (Candle/Burn optional, no real compute) | MUST | pending |
| 180 | Probe infrastructure per model slot with dummy fixtures | MUST | pending |
| 181 | Slot-level rollout state machine (shadow/compare/active) | MUST | pending |
| 182 | Drift budget schema and evaluator for shadow outputs | MUST | pending |
| 183 | Minimal alerts rules and report format | MUST | pending |
| 184 | Portability gate integration (Linux/Windows lanes) | MUST | pending |
| 185 | Strict-mode wiring for v1 scaffold features | MUST | pending |
| 186 | Operator docs and end-state update for v1 onboarding | MUST | pending |
| 187 | v1 scaffolding signoff gate (PASS/FAIL) | MUST | pending |

## Series control notes
- v0 completion requirement remains: **`ucf-ops v0 gate` overall PASS**.
- Prompts are classified as MUST/NICE/DEFERRED at authoring time.
- Only MUST-aligned prompts are included in the immediate capped queue.
