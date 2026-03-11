# Prompt Series State Snapshot

## Current anchor
- **Current anchor milestone:** `Real Compute Onboarding v3`.
- **Anchor definition:** v3 planning queue in `docs/next_10_prompts.md` (entry set `200-209`).
- **Prompt index source:** `docs/prompt_series_index.md`.

## Status against anchor milestones
- **v0:** complete.
- **v1:** complete.
- **v2:** complete (`ucf-ops v2 gate` overall PASS recorded at Prompt 198).
- **v3:** completes when `ucf-ops v3 gate` reports overall PASS.
- **Queue policy:** immediate queue remains capped to 10 prompts.

## Last executed prompt / resume point
- Last executed prompt ID: **208**.
- Next prompt ID: **209**.
- Resume from: **`PROMPT 209`**.
- Numbering remains monotonic and append-only.

## Immediate next prompts (capped to 10)
Reference: `docs/next_10_prompts.md`

| Prompt ID | Title (short) | v3 class | Status |
|---:|---|---|---|
| 200 | Active evidence expansion to supported real slots | MUST | complete |
| 201 | Unified eligibility report for Probe/Shadow/Active | MUST | complete |
| 202 | Candle second-slot adapter parity beyond fixture smoke | MUST | planned |
| 203 | Burn or second-slot backend parity extension | NICE | planned |
| 204 | Real-slot compare window normalization | MUST | planned |
| 205 | v3 strict-mode evidence broadening | MUST | planned |
| 206 | Operator/signoff report consolidation for real slots | MUST | planned |
| 207 | Portability and docs checks refresh | NICE | planned |
| 208 | v3 gate schema and orchestration | MUST | complete |
| 209 | v3 wrap and next-anchor governance | MUST | planned |

## Series control notes
- v0 completion requirement: historical signoff recorded.
- v1 completion requirement: **`ucf-ops v1 gate` overall PASS**.
- v2 completion requirement: **`ucf-ops v2 gate` overall PASS**.
- v3 remains hardware-neutral, offline-first, probe-first, and shadow-first.
- v3 signoff requirement: **`ucf-ops v3 gate` overall PASS** before advancing anchor.
- Prompts are classified as MUST/NICE/DEFERRED at authoring time.

## Supported real-slot baseline carried from v2
- First supported slot: `world_jepa`.
- Second supported slot declaration: `sae` (scope is fixed to `world_jepa` + exactly one second slot).
- Candle real-backend support is now available for both declared slots (`world_jepa`, `sae`) under probe-first + shadow-first operation.
- v3 planning may broaden support, but this declaration remains available for compatibility checks until superseded by explicit schema/docs updates.
