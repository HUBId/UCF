# LLM Backend v1 (offline, deterministic)

## Contracts
- `LlmRequest` carries deterministic inputs (`decision_id`, `candidate_id`, `output_class`, bounded `prompt`, digests, seed, token budget) plus bounded liquid conditioning (`lfm_readout_digest`, `lfm_uncertainty`, `lfm_stability`, `coherence`, `instability`, `risk`, `confidence`).
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


## Liquid-conditioned prompt template v0
- Prompt assembly is deterministic and fixed-order:
  1) system constraint line
  2) bounded context summary bullets
  3) fixed-order signals header (`risk`, `confidence`, `surprise`, `pressure`, `uncertainty`, `coherence`, `instability`)
  4) digest prefixes (`evidence_chain_digest`, optional `lfm_readout_digest`)
  5) output-class instruction + do/don't rules
- Prompt is capped at 8 KiB with deterministic truncation.
- Raw liquid/ESS payload vectors are never included; only bounded scalars and digest prefixes are injected.

## Uncertainty-aware decoding policy
- Effective output length is deterministic:
  - `max_tokens_eff = clamp(base * (1 - 0.6 * uncertainty), min=64, max=base)`
- NSR `SafeOnly`/`Block` hints force `SafeText` output class at generation time.
- High uncertainty / low stability trigger a deterministic short-output override path.
- Runtime persists the policy decision in `OutputRecord`:
  - `lfm_readout_digest`, `lfm_uncertainty`, `lfm_stability`
  - `max_tokens_eff`
  - `output_override` + bounded `override_reasons` (auditable reason codes)

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

## Candle CPU backend v1
- Enable with `--features llm-candle` on crates depending on `ucf-compute`.
- Backend selection remains env-driven:
  - `UCF_LLM_BACKEND=stub|candle|burn`
  - `candle` resolves to `candle:llm_v1` when `ModelSlot::Llm` + tokenizer asset verify; otherwise safe fallback (`candle:toy_v1` or stub).
- Active Candle v1 path loads safetensors via ModelSlot verification (hash-locked, local only):
  - required tensors: `tok_emb[32,64]`, `lm_head[64,32]`
  - tokenizer vocab JSON hash-locked via `UCF_LLM_TOKENIZER_PATH` + `UCF_LLM_TOKENIZER_SHA256`
- Decoding policy is deterministic greedy only:
  - no temperature sampling, no top-k randomness, no RNG draw in decode loop
  - `next = argmax(logits[-1])`
- Boundedness remains enforced:
  - prompt cap 8 KiB
  - max tokens bounded
  - output text cap 16 KiB
- NaN/Inf logits and timeout are treated as backend errors and trigger deterministic safe fallback response.

## Burn adapter status
- `burn` is feature-gated (`--features llm-burn`) and currently a skeleton returning `NotImplemented`.
- Without feature flags, selecting `candle` or `burn` fails fast with `BackendDisabled`.

## Determinism caveats
- CPU-only execution and fixed operation ordering are used.
- Logits are quantized before argmax to reduce backend-specific float drift.

## Safety invariants (non-negotiable)
- LLM does not execute tools.
- OutputClass validation is final gate in runtime:
  - invalid class/text pairing is converted to refusal
  - `ExecIntent` / `ExternalIo` stay tool-gated and non-automatic.
- Backend errors degrade safely (bounded refusal text / busy response).
