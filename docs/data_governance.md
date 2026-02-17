# Data Governance v1

## Data classes

`ucf-ess` defines bounded classes:

- `DigestOnly`
- `ScalarSummary`
- `TextPayload`
- `BinaryPayload`

Current mapping:

- `CandidateSetRecord`: `ScalarSummary`
- `LfmSummaryRecord`: `ScalarSummary`
- `CapabilityIssuanceRecord`: `ScalarSummary`
- `OutputRecord`: `TextPayload`
- `PolicyProvenanceRecord` / checkpoint digests: `DigestOnly`

## Retention policy

Policy file: `policies/bundle_v1/retention_v1.json`.

Fields:

- `keep_full_for_ticks`
- `keep_full_for_days`
- `keep_digests_forever`
- `max_ess_bytes`
- `policy_marker`

Retention is deterministic from `(policy, now_tick, record.tick)`.

## Redaction model (audit-safe)

`OutputRecord` now stores `content_digest` computed from canonical content bytes (`UCF:ESS:OUTPUT:CONTENT:v1` domain separator).

When a text payload crosses the retention horizon:

- `text` is removed
- `redacted=true`
- `payload_len` is stored
- `payload_classification` is stored
- `redaction_policy_marker` is stored
- `content_digest` remains stable and verifiable

This keeps digest-linking paths verifiable while pruning raw text.

## Snapshot / compaction workflow

Commands:

```bash
ucf-ops ess snapshot --workdir .ucf --out ./snapshots/run_001.snap
ucf-ops ess compact --workdir .ucf --policy policies/bundle_v1/retention_v1.json --apply
```

Compaction emits a manifest with:

- covered tick range
- total records
- redaction / pruned-byte counters
- policy hash
- snapshot digest
- manifest digest

## Verification after compaction

Run replay/security verification as usual (verify-only mode) to confirm digest chains:

```bash
ucf-ops replay --workdir .ucf --strict verify
ucf-ops security verify-chain --workdir .ucf --from 0 --to 18446744073709551615
```

Explain tick works with redacted outputs by exposing digest/length metadata instead of payload preview.
