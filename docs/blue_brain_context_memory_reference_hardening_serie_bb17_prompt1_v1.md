# BlueBrain Context/Memory/Reference Hardening — Serie BB17 Prompt 1 (v1)

Status: integrated hardening line across context, memory, execution, runtime/dynamics inputs, and bounded retrieval/reference.

## Canonical reference map (single runtime language)

BB17 keeps one shared runtime reference language with these canonical kinds:

1. `context_reference`
   - examples: `bb3:context:*`, `ctx:*`, `lens_feature:*`, `workspace_signal:*`
2. `memory_record_reference`
   - examples: `bb8:memory_record:*`
3. `execution_result_reference`
   - examples: `bb14:execution:*:result:*`
4. `combined_bounded_reference`
   - examples: `bb15:combined:*`
5. `diagnostic_reference`
   - examples: `diag:*`
6. `reference_only_not_memory_or_result`
   - advisory/reference path without memory or execution-result authority
7. `non_canonical_internal_only_reference_path`
   - any internal-only / non-canonical lane that must not be promoted

No second global reference language is introduced.

## Hard separation rules

- Context reference is never treated as memory record or execution result.
- Memory record reference is never treated as execution result.
- Execution result reference is never treated as context update.
- Diagnostic references stay diagnostics-only and are not operative result references.
- Combined bounded references (`bb15`) remain bounded candidate/basis references and are **not** consolidation engines.

## Validity/lifecycle basis kept explicit

Reference validity remains explicit and bounded:

- `current`
- `caveated`
- `stale`
- `invalidated`

Execution-result basis remains explicit and distinct:

- `successful`
- `failed`
- `cancelled`
- `blocked`
- `unavailable`
- `unsupported`
- `placeholder_only`

No global lifecycle engine is added.

## Runtime / Selection / Dynamics / Execution consumption posture

- Runtime/dynamics evidence ingestion now classifies references by canonical kind before usage.
- Kuramoto advisory input uses only canonical execution-result outcome buckets for execution-derived feedback.
- Diagnostic/reference-only/placeholder inputs remain diagnostics-only.
- Non-evidence kinds (context/memory/combined inside evidence slots) are flagged as unsupported evidence basis and not auto-promoted.
- Non-canonical/internal paths are explicitly flagged and never normalized to canonical evidence.

## Bounded retrieval/reference hardening

Combined retrieval basis now enforces boundary caveats:

- context reference must stay context-scoped,
- candidate reference must not be memory/result reference,
- proposal reference must not collapse into memory/result/diagnostic references.

These are caveated boundaries only; they do not trigger compute/action/memory mutation.

## No-direct-* boundaries (explicitly preserved)

Reference processing in BB17 hardening remains advisory-only:

- no direct action execution,
- no retry orchestration,
- no compute invocation,
- no implicit memory persistence,
- no policy/agent authority expansion,
- no neurodynamics authority escalation.

The line remains deterministic, bounded, and maintenance-compatible with BB8/BB14/BB15/BB16 constraints.
