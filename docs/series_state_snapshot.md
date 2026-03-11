# Prompt Series State Snapshot

## Current anchor
- **Current anchor milestone:** `Real Compute Onboarding v4`.
- **Anchor definition:** v4 planning queue in `docs/next_10_prompts.md` (entry set `210-219`).
- **Prompt index source:** `docs/prompt_series_index.md`.

## Status against anchor milestones
- **v0:** complete.
- **v1:** complete.
- **v2:** complete (`ucf-ops v2 gate` overall PASS recorded at Prompt 198).
- **v3:** complete (`ucf-ops v3 gate` overall PASS recorded at Prompt 208).
- **v4:** not started yet.
- **Queue policy:** immediate queue remains capped to 10 prompts.

## Last executed prompt / resume point
- Last executed prompt ID: **209**.
- Next prompt ID: **210**.
- Resume from: **`PROMPT 210`**.
- Numbering remains monotonic and append-only.

## Immediate next prompts (capped to 10)
Reference: `docs/next_10_prompts.md`

| Prompt ID | Title (short) | v4 class | Status |
|---:|---|---|---|
| 210 | Active evidence/signoff consistency for supported real slots | MUST | planned |
| 211 | Optional second-slot backend parity extension | NICE | planned |
| 212 | Unified backend evidence snapshot/spec export refresh | MUST | planned |
| 213 | Stricter operator signoff automation from consolidated reports | MUST | planned |
| 214 | Normalized remediation-code registry across reports/gates | MUST | planned |
| 215 | Report/schema snapshot checks for v4 artifacts | MUST | planned |
| 216 | Portability/docs refresh for expanded evidence paths | NICE | planned |
| 217 | Strict-mode/operator interplay hardening | MUST | planned |
| 218 | v4 gate schema and orchestration | MUST | planned |
| 219 | v4 wrap and next-anchor governance | MUST | planned |

## Series control notes
- v0 completion requirement: historical signoff recorded.
- v1 completion requirement: **`ucf-ops v1 gate` overall PASS**.
- v2 completion requirement: **`ucf-ops v2 gate` overall PASS**.
- v3 completion requirement: **`ucf-ops v3 gate` overall PASS at Prompt 208**.
- v4 remains hardware-neutral, offline-first, probe-first, shadow-first, and fail-closed.
- Prompts are classified as MUST/NICE/DEFERRED at authoring time.

## Supported real-slot baseline carried into v4
- First supported slot: `world_jepa`.
- Second supported slot declaration: `sae` (scope remains fixed to `world_jepa` + exactly one second slot unless explicitly superseded).
- Active-evidence expansion in v4 remains conservative and evidence-bound for supported slots only.


## Archived v3 queue reference

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
| 209 | v3 wrap and next-anchor governance | MUST | complete |
