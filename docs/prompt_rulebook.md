# Next Prompt Generator Rulebook

## Naming and numbering
- Use the canonical header form: `PROMPT <N> — <Title>`.
- Keep `<N>` strictly monotonic (`N = previous + 1`).
- Never recycle or rename historical prompt IDs after publication.

## Series governance guardrails
- Generate at most the **next 10 prompts** in planning docs by default.
- Do not generate more than 10 queued prompts unless explicitly requested.
- Treat queue generation as classification/planning only; do not delete historical prompts.
- Keep queue documents bounded and deterministic (stable order by prompt ID).

## Prompt classification requirement
- Every prompt must declare one class: **MUST**, **NICE**, or **DEFERRED**.
- Prompt body must align requirements/acceptance criteria with its declared class.
- Immediate execution queues must prioritize MUST items for the active anchor milestone.

## Required prompt structure
Every new prompt must include the following sections, in this order:
1. **Goal**
2. **Preconditions**
3. **Hard requirements**
4. **Phased plan**
5. **Acceptance criteria**
6. **Deliverables**
7. **Execution instructions**

## Hard requirements template (must always be present)
Include all of these invariants explicitly:
- **Offline**: no internet dependency in runtime paths unless explicitly and narrowly allowlisted.
- **Determinism**: same inputs/fixtures/config produce byte-stable outputs.
- **Safety**: policy-first execution and deny-by-default tool behavior.
- **Budget**: bounded compute, storage, and runtime costs.
- **Boundedness**: finite queues/retries/state growth and explicit failure exits.
- **Hardware-neutral**: prompts must not assume specific hardware models, machine classes, clusters, or vendor platforms in core requirements.
- **Probe-first / shadow-first / fail-closed**: prompts must preserve conservative progression and explicit fail-closed behavior for unsupported or missing evidence paths.

## Safety invariants (must remain true)
- **No decision, no action**.
- **Deny-by-default tools/capabilities**.
- **Hash-locked artifacts** (weights/config/contracts/evidence-bound assets).
- **Fixed-point safety signals** for gate-critical checks.

## Determinism invariants (must remain true)
- Canonical encoding for persisted artifacts and evidence payloads.
- No RNG in production paths unless policy-allowlisted and auditable.
- Replay compatibility for all acceptance-critical flows.

## Evidence invariants (must remain true)
- ESS records include digests and chain linkage.
- Explain-tick output remains available for operator/audit flows.
- Replay harness compatibility is preserved for modified execution paths.

## Phased plan template
Use explicit phases in new prompts:
1. Repo discovery
2. Types/contracts
3. Implementation
4. Wiring/integration
5. Tests/verification
6. Docs/release notes

## Acceptance criteria writing rules
- Use observable outcomes (files, commands, reports, checks).
- Keep each criterion testable and deterministic.
- Include at least one criterion each for safety, determinism, and evidence.
- Avoid speculative language; tie to concrete repo artifacts.

## Deliverables writing rules
- List exact file paths to be created/updated.
- Separate code, docs, fixtures, and release artifacts.
- Include any required checklist or signoff outputs.

## Execution instructions writing rules
- Provide ordered, non-interactive steps.
- Include required validation commands.
- Require final summary to include changed files and verification results.

## Canonical template usage
- Use `docs/codex_prompt_template.txt` as the default copy/paste super-prompt skeleton.
- Put task-specific details only between `START_TASK_SPECIFIC` and `END_TASK_SPECIFIC`.
- Keep all standard sections/checklists intact (context, discovery, implementation, tests, docs, invariants, final summary).
- For concrete guidance, see `docs/codex_prompt_template_example.txt`.

## Hardware-neutral guidance
- Express target environments using `DeviceProfile` classes (`small`, `medium`, `large`) rather than machine names.
- Encode performance/throughput expectations as explicit budget envelopes (latency, memory, compute), not vendor or node-family references.
- Restrict hardware-specific wording to deployment templates/history notes when historically required.
