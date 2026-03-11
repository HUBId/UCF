# SAE Parity v4

Scope bleibt exakt auf dem konfigurierten zweiten Real-Slot aus `docs/series_state_snapshot.md` (aktuell: `sae`).

## Backends und Optional-Status
- Decision source bleibt `stub_sae_v1` (keine Decision-Änderung).
- Pflicht-Shadow bleibt `candle_sae_v1`.
- Optionaler Shadow `burn_sae_v1` wird über `OptionalBackendSupportStateV1` explizit ausgewiesen:
  - `SUPPORTED`
  - `UNSUPPORTED`
  - `NOT_BUILT`
  - `NOT_CONFIGURED`

Zusätzlich zeigt der Report:
- `burn_parity_status` (`OK|WARN|SEVERE|SKIP`)
- `burn_parity_present` (`true|false`)

## SKIP vs FAIL
- Standardprofil: Burn ist optional.
  - Kein Burn-Scaffold/Feature oder kein Burn-Shadow-Enablement => `SKIP`.
- Wenn explizit gefordert (`UCF_SECOND_SLOT_BURN_PARITY_REQUIRED=1`):
  - fehlende Burn-Parität führt zu `FAIL` mit stabilem Code:
  - `OPTIONAL_BACKEND_REQUIRED_BUT_MISSING`

## Compare/Window-Semantik
SAE-Parität nutzt weiterhin die normalisierte `CompareWindowMetaV1`-Semantik (Prompt 204):
- deterministische Window-ID
- sortierte compared backend IDs
- bounded Window-Menge
- keine Active-Enablement-Wirkung

## Beobachtung / Command
```bash
cargo run -p ucf-ops -- models parity --slot sae --run <id> --out ./out/sae_parity_report.json
```

Interpretation:
- Candle vorhanden + Burn `SKIP`: gültiger v4-Default (optional).
- Burn erforderlich + nicht vorhanden: strict check FAIL (`OPTIONAL_BACKEND_REQUIRED_BUT_MISSING`).
