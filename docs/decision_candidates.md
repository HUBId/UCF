# Decision Candidate Contracts v0

This document defines deterministic, bounded candidate planning and gating.

## Taxonomy

- `IntentKind`: `Respond`, `Plan`, `QueryEss`, `Consolidate`, `RequestTool`, `Defer`.
- `OutputClass`: `SafeText`, `Code`, `ExternalIo`, `ExecIntent`, `Sensitive`.
- `ToolIntent`: explicit capability request contract with target hash, bounded preview,
  payload-size hint, expected effect class, and digest.

## Gating path

1. `CandidateGenerator` emits a bounded candidate set (max 8), stable sorted by `candidate_id`.
2. NSR + policy produce deterministic candidate assessments (`allowed`, hint, score, reasons).
3. Selection chooses highest score among allowed candidates, stable tie-break by `candidate_id`.
4. Tool requests are derived only from selected candidate intents with capability checks.

## Safety invariants

- No decision, no action.
- Tool execution only after candidate selection and policy assessment.
- Tool requests carry `decision_id`, `evidence_chain_digest`, `candidate_id`, `tool_intent_digest`.
- Candidate selection and summaries are persisted in ESS audit chain.

## Extending generators

Implement `CandidateGenerator` in `runtime/ucf-policy/src/candidate.rs`, keep:

- deterministic output ordering,
- bounded counts and string lengths,
- canonical digest stability,
- explicit `OutputClass` and `ToolIntent` declarations.
