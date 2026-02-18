# Tool Plans v1

`ToolIntent` is now handled as a strict three-phase flow:

1. **ToolPlan (dry run)**
   - Canonical `ToolPlanV1` is generated from intent + request context.
   - Canonical args are key-sorted and bounded (`<=4KB`).
   - `plan_digest = H(canonical_plan_bytes || policy_graph_digest)`.
   - ESS stores `ToolPlan` audit record with digest prefixes, required caps, and optional EBM/NSR scalars.

2. **ToolIssue (issuance decision)**
   - Governor tier + existing capability set + EBM/NSR signals are evaluated.
   - EBM/NSR are **tightening only**: they can deny issuance, never expand privileges.
   - ESS stores `ToolIssue` audit record with issued capability digest prefixes or bounded deny reasons.

3. **ToolExecute**
   - Execution proceeds only if `ToolIssueDecision.issued=true`.
   - Replay protection is enforced by single-use token tracking in orchestrator (`spent_tool_tokens`).
   - Any mismatch or replay is denied and persisted as denied `ToolExecution`.

## Canonicalization

- Canonical args are generated from a stable key set (`bytes_in`, `bytes_out`, `decision_id`, `target`).
- Keys are sorted lexicographically before encoding.
- String values are truncated to bounded length.
- Digesting uses SHA-256 with domain separation (`UCF:TOOL_PLAN:V1`).

## Audit trail interpretation

For each successful or denied tool attempt, ESS now contains:

- `ToolRequest`
- `ToolPlan`
- `ToolIssue`
- `SandboxCall`/`ToolAuth`/`ToolExecution`/`SandboxReply` (if execution attempted)

This provides phase-by-phase forensic reconstruction with digest links and reason codes.

## Failure modes

- `tool_plan_missing_intent`: request could not be tied to a declared tool intent.
- `nsr_high_risk` / `ebm_high_energy`: tightened denial at issue phase.
- `plan_args_mismatch`: request mutated relative to planned canonical args.
- `token_replay`: single-use token replay was detected.
