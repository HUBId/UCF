# Compute Evidence Chain (v0)

Die Compute-Pipeline erzeugt pro Lauf eine **Evidence Chain** mit ausschließlich bounded Referenzen (Digests/IDs), keine Payloads.

## Bestandteile

- `compute_schema_version` (aktuell `1`)
- `compute_code_version` (build-time `UCF_GIT_COMMIT` Prefix oder crate version)
- `backend_profile`, `budget_profile_id`, `seed`
- `context_digest`, optional `world_digest`/`spikes_digest`/`ssm_digest`
- `risk_digest` (canonical digest des `RiskSignal`)
- `compute_chain_digest` (canonical digest über alle Felder oben)

## Canonical Hash Discipline

- Hash-Funktion: SHA-256.
- Encoding: manuelles canonical byte encoding (little-endian primitives).
- Float-Daten: über `to_bits()` serialisiert.
- Spikes: canonical sortiert nach `(timestamp, feature_id, magnitude_bits)` vor Digest.

## Replay/Audit

- Replay behandelt identische `compute_chain_digest` als **strong match**.
- Bei Abweichung werden komponentenweise Drift-Gründe inkl. Digest-Prefixen ausgegeben.
- Telemetriezähler:
  - `ucf_compute_chain_digest_emitted_total`
  - `ucf_compute_chain_mismatch_total`
