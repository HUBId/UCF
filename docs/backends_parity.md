# Backend Parity (Candle ↔ Burn) v1

`ucf-compute` supports CPU-only parity across Candle and Burn backend packs behind feature flags.

## Enablement
- Candle: `--features compute-candle`
- Burn: `--features backend-burn` (enables `compute-burn`)
- Combined parity test: `--features compute-candle,backend-burn`

Runtime selection:
- `UCF_BACKEND_PACK=candle_toy_v1`
- `UCF_BACKEND_PACK=burn_toy_v1`

## What parity means in v1
Parity is **envelope-level**, not byte-for-byte tensor/runtime identity:
- same deterministic loop ordering
- same quantization (`quantize_signed_unit`, `quantize_unit_u16`)
- same bounded outputs (`[0,1]` / `[-1,1]` clamps)
- same degraded fallback semantics on NaN/Inf
- same digest construction inputs for stage outputs

This is enough for stable `EvidenceChain` behavior and explain-tick backend attribution while avoiding backend lock-in.

## Unified weight loader (`LoadedWeightsRaw`)
WeightSpec validation is shared for Candle and Burn:
- parse safetensors offline only
- enforce slot/hash lock via `ModelStore`
- enforce required tensor names + exact shape + dtype
- decode contiguous payload to canonical `Vec<f32>` (`LoadedWeightsRaw`)

Then:
- Candle converts `LoadedWeightsRaw -> Tensor`
- Burn consumes `LoadedWeightsRaw` buffers directly (CPU deterministic loops in v1)

## Determinism cautions
- top-k in SAE must break ties by feature id
- structured scan/state updates must run in strict index order
- digest-critical reductions should stay in Rust loops until backend kernel equivalence is proven

## Compare/smoke strategy
A parity smoke check compares Candle vs Burn envelopes:
- both outputs finite and bounded
- scalar drift under threshold for toy inputs (example: `|pressure_candle-pressure_burn| <= 0.15`)
- no requirement for digest equality across implementations
