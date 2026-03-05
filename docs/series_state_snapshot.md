# Prompt Series State Snapshot

## Current anchor
- **Current anchor milestone:** `Real Compute Onboarding v0`.
- **Anchor definition:** `docs/roadmap_anchor_v0.md`.
- **Prompt index source:** `docs/prompt_series_index.md`.

## Status against v0 anchor (best-effort)
- **MUST coverage already completed (historical prompts):**
  - Real compute onboarding block (`38-67`) is indexed as delivered historical work.
  - Existing v0 onboarding E2E documentation is present.
- **MUST governance now active:**
  - Future prompt generation is constrained to MUST-aligned items until v0 closure.

## Last executed prompt / resume point
- Last executed prompt ID: **128**.
- Resume from: **`PROMPT 129`**.
- Numbering remains monotonic and append-only.

## Immediate next prompts (capped to 10, MUST-only)
Reference: `docs/next_10_prompts.md`

| Prompt ID | Title (short) | v0 class | Status |
|---:|---|---|---|
| 129 | Backend trait contract freeze for v0 stubs | MUST | pending |
| 130 | Deterministic CPU backend stub conformance tests | MUST | pending |
| 131 | JEPA mock deterministic signal mapping | MUST | pending |
| 132 | SAE mock deterministic signal mapping | MUST | pending |
| 133 | SSM mock deterministic state update mapping | MUST | pending |
| 134 | Compute summary wiring for spikes/surprise/pressure | MUST | pending |
| 135 | E2E ControlFrame->Decision->ESS canonical fixture pass | MUST | pending |
| 136 | Policy gate assertion: no decision, no action | MUST | pending |
| 137 | Minimal explain-tick observability acceptance checks | MUST | pending |
| 138 | v0 onboarding signoff bundle (must-only evidence) | MUST | pending |

## Series control notes
- Prompts are classified as MUST/NICE/DEFERRED at authoring time.
- Only MUST-aligned prompts are included in the immediate capped queue.
