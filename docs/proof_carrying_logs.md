# Proof-Carrying Logs v1 (local)

This document specifies deterministic Merkleized log segments for ESS fixture records and local proof verification.

## Segmenting rules

- Segment boundaries are deterministic: records are split in append order into fixed-size chunks (`segment_size`, default `1024`).
- Segment ID is `(run_id, segment_index)`.
- Each segment computes a deterministic binary Merkle tree over canonical leaf digests.
- For odd node counts, the last node is duplicated when hashing parent nodes.

## Canonical leaf digest

Each leaf digest is computed from a canonical JSON object containing:

- `id`
- `tick`
- `window`
- `corr`
- `kind`
- `audit_digest` (if present)

The canonical JSON bytes are SHA-256 hashed.

## Segment record fields

`MerkleSegmentRecord` contains:

- `segment_id`
- `first_t`, `last_t`
- `record_count`
- `merkle_root`
- `prev_segment_root` (segment chain)
- `segment_digest`

`segment_digest` is domain-separated SHA-256 over segment metadata.

## Proof format

`ucf-ops logs prove --record-digest <hex> --out ./out/proof.json`

Proof JSON includes:

- `segment_id`
- `leaf_index`
- `siblings[]` with:
  - `sibling_hash`
  - `sibling_on_left`
- `segment_root`
- `leaf_hash`
- `proof_digest`

Verification command:

`ucf-ops logs verify-proof --proof ./out/proof.json`

Proof size is `O(log n)` in segment record count.

## verify-chain integration

`ucf-ops security verify-chain` now:

1. verifies existing security-chain linkage,
2. builds deterministic Merkle segments,
3. validates segment-chain linkage via `prev_segment_root`,
4. samples one inclusion proof per segment and verifies it.

## Checkpoints and future anchoring

v1 remains local-only. Segment records and segment roots are designed to be anchor-ready for future external checkpoint publication.
