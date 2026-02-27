# Config Migration v2 — Minimal Viable Governance

## Zielbild
Die Runtime nutzt jetzt eine sichere Profile-Ladder (`dev -> test -> prod`) über versionierte Dateien in `configs/`.

- `configs/dev.toml`
- `configs/test.toml`
- `configs/prod.toml`

`UCF_PROFILE` wählt das Profil aus. Unbekannte Konfigurationsschlüssel werden beim Laden strikt abgelehnt.

## Runtime Surface (v2)
Zur Laufzeit bleiben nur diese Schalter übrig:

- Profilauswahl: `UCF_PROFILE=dev|test|prod`
- Overlay-Auswahl (allowlist): `UCF_POLICY_OVERLAY`
- Slot-Modi: `UCF_SLOT_EBM_MODE` (`shadow`/`active`)
- Stage-Isolation-Laufzeit: `UCF_STAGE_ISOLATION`
- Emergency-Pin-Override: `UCF_EMERGENCY_POLICY_PIN`

## Verschobene Schlüssel (in Policy Packs)
Folgende Grenzwerte liegen jetzt zentral in `policies/packs/*/thresholds.toml`:

- Governor-Tier Schwellwerte (`governor_tier_*`)
- EBM Risk Schwellwerte (`ebm_*`)
- Drift/Gate Schwellwerte (`world_vljepa_*`, `ssm_opt_*`)

## Legacy Runtime-Overrides (deprecated)
Diese Env-Overrides sollten entfernt werden, da sie Policy-Werte umgangen haben:

- `UCF_WORLD_VLJEPA_GATE_MIN_WINDOWS`
- `UCF_WORLD_VLJEPA_DRIFT_ALARM_RATE_MAX`

## Safe Override Workflow
1. Profil wählen (`UCF_PROFILE`).
2. Falls nötig, Overlay in Policy Packs anpassen.
3. Policy-Validierung und Readiness-Gates laufen lassen.
4. Nur explizite Emergency-Pins dokumentiert einsetzen.

## Prod Verbote
In `prod` gilt:

- kein Sampling (`sampling_enabled=false`)
- determinism lock strict
- deny-by-default tool issuance
- docs lint als Pflicht-Gate (`docs_lint_required=true`)

## Auditability
`RunMetadataRecord` enthält zusätzlich:

- `profile`
- `policy_overlay`
- `config_digest`

Damit ist die effektive Konfiguration pro Lauf nachvollziehbar und hashbar.
