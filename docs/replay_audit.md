# Replay Audit (Liquid Safety)

`ucf-ops replay` bietet einen deterministischen, offline Replay-/Audit-Modus für ESS-Fixtures.

## Modi

- `verify` (`ReplayStrictness::VerifyOnly`):
  - prüft Record-Integrität, Evidence-Links und Digest-Referenzen
  - verifiziert BackendPack-Konsistenz im Bereich `[t0..t1]`
  - verifiziert Capability-Issuance Referenzen und Governance-Signal-Digest
- `recompute` (`ReplayStrictness::RecomputeStages`):
  - enthält alle `verify`-Checks
  - re-simuliert Compute-Stages für Decision-Records (stub/toy) und vergleicht Chain-Digests
  - verifiziert Governor-Score/Tier Ableitung gegen persistierte Issuance-Felder

## Benötigte Records

Im ESS-Slice werden für audit-grade Replays erwartet:

- `DecisionOut` mit `compute_summary.compute_chain_digest`
- `BackendPack` mit `backend_pack_record.meta_digest`
- `CapabilityIssuance` (`AuditPayload::CapabilityIssuance`)
- optional (wenn aktiv): `Nsr`, `LfmSummary`, `Output`, `Hormone`

## Drift Report

Der Replay Report ist bounded:

- `range`: `(t0, t1)`
- `overall_status`: `ok | drift_found | missing_data`
- `first_divergence`: erste Divergenz
- `counters`: `missing_records`, `mismatched_digests`, `degraded_steps`
- `details`: max. 64 Divergenzen

Jede Divergenz enthält:

- `t`
- `component` (`backend_pack`, `world`, `sae`, `ssm`, `lfm`, `risk`, `nsr`, `coherence`, `governor`, `issuance`, `output`)
- `expected_digest`
- `observed_digest`
- `hint`

## Determinismus-Hinweise

- Immer offline aus lokalen, committed Fixtures ausführen.
- Canonical Float-Encoding und Option-Encoding für Digest-Bildung beibehalten.
- Seed-/Backend-Auswahl aus persistierten Summaries konsistent verwenden.
- Replay range immer begrenzen (`--from`, `--to`).

## CLI

```bash
ucf-ops replay --workdir .ucf --from 1 --to 500 --strict verify --report replay_report.json
ucf-ops replay --workdir .ucf --from 1 --to 500 --strict recompute --report replay_report.json
```

Optional:

- `--continue`: sammelt mehrere Divergenzen (bis 64), statt beim ersten Fund zu stoppen.
