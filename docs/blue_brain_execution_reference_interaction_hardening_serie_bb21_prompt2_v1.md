# BlueBrain Serie BB21 — Prompt 2: weak reference-consumption boundary hardening

Prompt 2 hardens the weak reference-consumption boundary introduced in BB21 Prompt 1.

## Canonical weak reference-consumption states

Execution result references are now consumed with strict separation:

- `failed_execution_feedback_basis`
- `cancelled_execution_feedback_basis`
- `blocked_dynamics_feedback_basis`
- `unavailable_dynamics_feedback_basis`
- `insufficient_dynamics_feedback_basis`
- `non_canonical_internal_only_feedback_path`

Strong basis remains only:

- `execution_informed_dynamics_input` (completed/current canonical execution references)

## Boundary rules

- Failed, cancelled, blocked, and unavailable are never collapsed into a strong/current basis.
- Failed is not cancelled; blocked is not failed; unavailable is not blocked.
- Weak basis can only inform caveated/insufficient/diagnostic awareness.
- Weak basis cannot imply positive execution outcome or re-use as strong operational support.

## Cross-line alignment (runtime/selection/retrieval)

- Runtime and selection consume the same canonical feedback state machine from `ucf-compute`.
- Retrieval-origin execution references are classified before modulation input assembly in `ucf-router`.
- Non-canonical/internal-only paths stay explicitly bounded and non-promotable.

## No-direct-* boundaries (unchanged and explicit)

Weak reference consumption remains advisory-only:

- no direct retry orchestration,
- no direct follow-up execution trigger,
- no implicit memory persistence,
- no compute invocation authority uplift,
- no policy/reasoning/agent expansion.
