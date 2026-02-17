# Ops Explain Tick

`ucf-ops explain-tick` erzeugt einen deterministischen, offline Report aus ESS-Records.

## Commands

```bash
cargo run -p ucf-ops -- explain-tick --workdir .ucf --t 123 --json
cargo run -p ucf-ops -- explain-tick --workdir .ucf --decision-id 42 --detail-level 2 --json
cargo run -p ucf-ops -- metrics summary --workdir .ucf --last 64 --json
cargo run -p ucf-ops -- metrics trend --workdir .ucf --from 0 --to 256 --json
```

## Report sections

- **header**: Tick, Decision-ID, Backend-Pack/Evidence-Chain Digest Prefix.
- **compute**: world/sae/ssm/lfm/coherence/risk aus persistierten Summaries.
- **governance**: Governor-Tier/Score, Emergency-Status, Capability-Issuance (bounded, sorted).
- **decision**: CandidateSet-Auswahl, Policy-Hints, NSR-Reason-Codes.
- **output**: Output-Class, Backend, Request/Response-Digest Prefixe, Status.
- **links**: deterministische Liste verknüpfter Record IDs/Kinds.
- **warnings**: fehlende oder inkonsistente Datensätze.

## Failure patterns

- **Missing records**: `warnings` enthält Einträge wie `CandidateSetRecord missing`.
- **Tier 3 / emergency**: in `governance.tier` bzw. `governance.emergency_active` sichtbar.
- **Output gap**: fehlender Output erscheint als Warning und `output.* = null`.

Alle Ausgaben sind read-only, bounded und aus ESS abgeleitet.
