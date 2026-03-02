# Spec Snapshot

Deterministic snapshot generated from code registries + policy pack manifests.

## A) Frames / Records

- `compute.evidence_chain`: schema_version=`2`; key fields: schema_version, spikes_digest, chain_digest
- `compute.ipc`: schema_version=`1`; key fields: schema_version, request_id, stage
- `compute.remote`: schema_version=`1`; key fields: schema_version, nonce, payload_digest
- `policy.graph`: schema_version=`1`; key fields: schema_version, base_version, overlay_version
- `ess.governance_state`: schema_version=`2`; key fields: schema_version, cooldown_until_tick, flags

## B) Stage contracts

- `world`: version(s) `1`; output fields: surprise, prediction_error, state_norm, prediction_digest; quantization: f32 scalars in [0,1], digest bytes as fixed [u8;32]
- `sae`: version(s) `1`; output fields: spike_count, spikes, sparsity, energy, spikes_digest; quantization: spike_count u16, magnitudes f32 in [0,1], digest bytes as fixed [u8;32]
- `ssm`: version(s) `1`; output fields: pressure, readout, state_norm, state_digest, readout_digest; quantization: f32 scalars in [0,1], digest bytes as fixed [u8;32]
- `lfm`: version(s) `1`; output fields: uncertainty, stability, energy, state_digest; quantization: f32 scalars in [0,1], digest bytes as fixed [u8;32]
- `llm`: version(s) `1`; output fields: risk, confidence, reason_codes, reason_digest; quantization: risk/confidence are f32 in [0,1], reason codes bounded (MAX_REASON_CODES)

## C) Backends

### BackendComponentId

- `StubV0` (`0`)
- `ToyV1` (`1`)
- `CandleToyV1` (`2`)
- `BurnToyV1` (`3`)
- `LnnOdeV1` (`4`)
- `RemoteProxyV1` (`5`)
- `CandleJepaV1` (`10`)
- `CandleSaeV1` (`11`)
- `CandleSsmV1` (`12`)
- `CandleEbmV1` (`13`)
- `VljepaAdapterV0` (`14`)
- `CandleVljepaV1` (`15`)
- `BurnJepaV1` (`20`)
- `BurnSaeV1` (`21`)
- `BurnSsmV1` (`22`)
- `Disabled` (`255`)

### BackendPackMeta schema

- `schema_version`
- `pack_name`
- `pack_id`
- `llm_backend`
- `world_backend`
- `sae_backend`
- `ssm_backend`
- `lfm_backend`
- `fixtures_digest`
- `model_hashes_digest`
- `code_version`
- `digest`

## D) Policy digests

- base_pack_digest: `42e8a59369a29faa…`
- overlay_pack_digest: `9f86cf9136aee6ae…`
- policy_graph_digest: `464943c52bb222f0…`
- determinism_policy_digest: `690b28933c1bb65f…`

## E) Model slots

### `ebm_reasoner`
- active_hash: `n/a`
- max_bytes: `67108864`
- required_tensors:
  - `ebm.w1` shape=`[d,h]` dtype=`f32`
  - `ebm.b1` shape=`[h]` dtype=`f32`
  - `ebm.w2` shape=`[h,1]` dtype=`f32`
  - `ebm.b2` shape=`[1]` dtype=`f32`
### `lfm`
- active_hash: `n/a`
- max_bytes: `67108864`
- required_tensors:
  - `alpha` shape=`[N]` dtype=`f32`
  - `Wx` shape=`[N,N]` dtype=`f32`
  - `Wu` shape=`[N]` dtype=`f32`
  - `b` shape=`[N]` dtype=`f32`
### `llm`
- active_hash: `n/a`
- max_bytes: `67108864`
- required_tensors:
  - `tok_emb` shape=`[32,64]` dtype=`f32`
  - `lm_head` shape=`[64,32]` dtype=`f32`
### `sae`
- active_hash: `n/a`
- max_bytes: `67108864`
- required_tensors:
  - `sae.w_enc` shape=`[F,D]` dtype=`f32`
  - `sae.b_enc` shape=`[F]` dtype=`f32`
### `ssm`
- active_hash: `n/a`
- max_bytes: `67108864`
- required_tensors:
  - `A` shape=`[N,N]` dtype=`f32`
  - `B` shape=`[N]` dtype=`f32`
  - `C` shape=`[N]` dtype=`f32`
### `world_jepa`
- active_hash: `n/a`
- max_bytes: `67108864`
- required_tensors:
  - `W1` shape=`[D,H]` dtype=`f32`
  - `b1` shape=`[H]` dtype=`f32`
  - `W2` shape=`[H,D]` dtype=`f32`
  - `b2` shape=`[D]` dtype=`f32`
### `world_vljepa`
- active_hash: `n/a`
- max_bytes: `67108864`
- required_tensors:
  - `vljepa.w1` shape=`[D,H]` dtype=`f32`
  - `vljepa.b1` shape=`[H]` dtype=`f32`
  - `vljepa.w2` shape=`[H,D]` dtype=`f32`
  - `vljepa.b2` shape=`[D]` dtype=`f32`

## F) Safety invariants

- no inferable invariant flags exported by registries; section intentionally bounded.
