# LLM Backend v0 (offline)

## Contracts
- `LlmRequest` carries deterministic inputs (`decision_id`, `candidate_id`, `output_class`, bounded `prompt`, digests, seed, token budget).
- `LlmResponse` carries bounded text + audit fields (`status`, `finish_reason`, `token_count`, canonical `digest`).
- `LlmInference` is text-only (`infer(req, budget)`), no tool execution and no IO access.

## Stub semantics
- `LlmStubBackend` is deterministic and offline.
- Text generation uses fixed vocabulary + index selection from prompt/context digest + seed.
- Hard bounds:
  - prompt bytes capped to 8 KiB
  - response bytes capped to 16 KiB
  - max tokens capped to 1024
- `ExternalIo` / `ExecIntent` are refused by stub with `PolicyRefusal`.

## Output-class enforcement
- Runtime validates final output in one choke point:
  - `SafeText`: rejects code fences.
  - `Code`: allows fenced output.
  - `ExternalIo`/`ExecIntent`: no actionable generated text path; runtime stores plan summary and relies on tool-intent + ToolGate.

## Wiring & persistence
- Selected candidate now always creates an ESS `OutputRecord`.
- For `SafeText`/`Code`: runtime calls LLM backend, persists request/response digests and bounded text.
- For `ExternalIo`/`ExecIntent`: runtime does not auto-infer, persists a plan/intent summary instead.

## Future adapters
- Add Candle/Burn by implementing `LlmInference` adapter only.
- Select backend via env:
  - `UCF_LLM_BACKEND=stub|candle|burn`
  - `UCF_LLM_SEED`
  - `UCF_LLM_MAX_TOKENS`
- Without features, candle/burn fail fast as `BackendDisabled`.
