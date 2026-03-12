# Prompt Series State Snapshot

## Current anchor
- **Current anchor milestone:** `Real Compute Onboarding v5`.
- **Anchor definition:** v5 planning queue in `docs/next_10_prompts.md` (entry set `220-229`).
- **Prompt index source:** `docs/prompt_series_index.md`.

## Status against anchor milestones
- **v0:** complete.
- **v1:** complete.
- **v2:** complete (`ucf-ops v2 gate` overall PASS recorded at Prompt 198).
- **v3:** complete (`ucf-ops v3 gate` overall PASS recorded at Prompt 208).
- **v4:** complete (`ucf-ops v4 gate` overall PASS recorded at Prompt 218).
- **v5:** not started yet.
- **Queue policy:** immediate queue remains capped to 10 prompts.

## Last executed prompt / resume point
- Last executed prompt ID: **219**.
- Next prompt ID: **220**.
- Resume from: **`PROMPT 220`**.
- Numbering remains monotonic and append-only.

## Immediate next prompts (capped to 10)
Reference: `docs/next_10_prompts.md`

| Prompt ID | Title (short) | v5 class | Status |
|---:|---|---|---|
| 220 | Supported real-slot governance expansion (cautious) | MUST | planned |
| 221 | Active-review evidence export unification | MUST | planned |
| 222 | Optional second-slot Burn parity closure | NICE | planned |
| 223 | Backend evidence/signoff reuse in repro exports | MUST | planned |
| 224 | Gate/report remediation consistency hardening | MUST | planned |
| 225 | v5 schema snapshot refresh | MUST | planned |
| 226 | v5 portability/docs refresh for evidence/export | NICE | planned |
| 227 | Read-only operator review workflow hardening | MUST | planned |
| 228 | v5 gate schema and orchestration | MUST | planned |
| 229 | v5 wrap and next-anchor governance | MUST | planned |

## Series control notes
- v0 completion requirement: historical signoff recorded.
- v1 completion requirement: **`ucf-ops v1 gate` overall PASS**.
- v2 completion requirement: **`ucf-ops v2 gate` overall PASS**.
- v3 completion requirement: **`ucf-ops v3 gate` overall PASS at Prompt 208**.
- v4 completion requirement: **`ucf-ops v4 gate` overall PASS at Prompt 218**.
- v5 remains hardware-neutral, offline-first, probe-first, shadow-first, and fail-closed.
- Prompts are classified as MUST/NICE/DEFERRED at authoring time.

## Supported real-slot baseline carried into v5
- First supported slot: `world_jepa`.
- Second supported slot declaration: `sae` (scope remains fixed to `world_jepa` + exactly one second slot unless explicitly superseded by evidence-bound governance).
- v5 scope remains conservative: governance/evidence/signoff hardening first, optional parity depth only under existing probe/shadow evidence.

## Archived v4 queue reference

| Prompt ID | Title (short) | v4 class | Status |
|---:|---|---|---|
| 210 | Active evidence/signoff consistency for supported real slots | MUST | complete |
| 211 | Optional second-slot backend parity extension | NICE | complete |
| 212 | Unified backend evidence snapshot/spec export refresh | MUST | complete |
| 213 | Stricter operator signoff automation from consolidated reports | MUST | complete |
| 214 | Normalized remediation-code registry across reports/gates | MUST | complete |
| 215 | Report/schema snapshot checks for v4 artifacts | MUST | complete |
| 216 | Portability/docs refresh for expanded evidence paths | NICE | complete |
| 217 | Strict-mode/operator interplay hardening | MUST | complete |
| 218 | v4 gate schema and orchestration | MUST | complete |
| 219 | v4 wrap and next-anchor governance | MUST | complete |


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
