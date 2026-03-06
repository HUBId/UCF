# Probes v1 (offline, deterministic)

`ucf-ops models probe` validates per-slot model plumbing before real weights become active.

## Purpose

- Validate loader/manifest path and slot wiring, even with dummy fixtures.
- Produce deterministic contracted outputs (digest prefixes + fixed-point scalars).
- Run bounded envelope checks and emit `ProbeReportV1` JSON.
- Never execute tools; probe logic is local/offline only.

## Command

Active (promoted hash from lifecycle manifest):

```bash
cargo run -p ucf-ops -- models probe --slot llm --out ./out/probe_llm.json
```

Staging hash (pre-promotion, without activation):

```bash
cargo run -p ucf-ops -- models probe --slot llm --hash <staged_hash> --out ./out/probe_llm_staging.json
```

## Report schema (`ProbeReportV1`)

- `schema_version`
- `slot_id`
- `mode` (`active|hash|stub`)
- `manifest_digest_prefix`
- `model_hash_prefix` (optional)
- `backend_id`
- `contract_version`
- `outputs`:
  - `digests[]` (prefixes)
  - `scalars[]` (`*_q` in `[0, 10000]`)
  - `counters[]`
- `latency_ms` (best effort)
- `envelope_checks[]` (`PASS|FAIL` with stable codes)
- `status` (`PASS|FAIL`)

## Envelope checks

- `PROBE_SCALAR_BOUNDS`
- `PROBE_DIGEST_NON_ZERO`
- `PROBE_MODEL_BYTES_NON_ZERO`
- `PROBE_OUTPUT_CAP`
- `PROBE_SAE_SPIKE_COUNT_BOUNDED` (SAE)

## Dummy fixtures

Dummy model artifacts are provided under:

- `fixtures/models_dummy/<slot>/model.safetensors`

They are sufficient to validate `stage -> promote -> probe` offline.
