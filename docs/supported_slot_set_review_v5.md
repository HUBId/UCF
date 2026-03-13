# Supported Slot Set Review v5

## Purpose

`supported-set-review` performs a **read-only governance review** to decide whether v5 should keep the current supported real-slot scope frozen or mark exactly one candidate as expansion-ready for later prompts.

The command does **not** activate runtime slot support and does not mutate manifest, policy, or slot mode state.

## Decision model

The report emits a deterministic `SupportedRealSlotSetPolicyV2` with:

- `current_supported_slots`
- `candidate_slots_considered`
- `decision` (`FREEZE` or `EXPAND_BY_ONE`)
- optional `chosen_candidate_slot`
- `rationale_codes`
- `policy_digest`

Default posture is fail-closed:

- If evidence scaffolding is insufficient, decision is `FREEZE`.
- If more than one candidate is equally ready, decision is `FREEZE` with ambiguity rationale.
- `EXPAND_BY_ONE` is allowed only when exactly one candidate satisfies all required eligibility booleans.

## Expansion eligibility criteria

Per candidate slot, `SlotExpansionEligibilityV1` checks:

- trait/contract availability
- probe path availability/reuse
- shadow path availability or trivial attachability
- compare-window normalization fit
- strict/evidence plumbing representability without architecture fork
- tiny fixture path feasibility

All required checks must be true for `expansion_ready=true`.

## Command

```bash
cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json
```

## How to interpret output

- `decision=FREEZE`: current supported real-slot set remains authoritative; no expansion is justified yet.
- `decision=EXPAND_BY_ONE`: exactly one candidate is governance-approved for potential later implementation work, but runtime support remains unchanged until explicitly implemented.


## Application handoff (v6)

`SupportedRealSlotSetPolicyV2` is a review policy artifact, not the applied state.
Use `supported-set-apply` to produce `SupportedRealSlotSetV2` as the canonical applied scope artifact consumed by downstream governance surfaces.
