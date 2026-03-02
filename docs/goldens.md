# Golden Scenarios v1

`ucf-ops goldens` erzeugt und verifiziert kleine, deterministische Toy/Stub-Baselines für Regressionsschutz.

## Szenarien

Szenarien liegen in `fixtures/goldens/scenarios/` und sind bewusst klein (12 Ticks):

- `golden_a`: baseline stabil
- `golden_b`: höherer Druck/Unsicherheit (ohne EBM)
- `golden_c`: EBM shadow
- `golden_d`: Tool-Plan-Demo mit EBM active

Alle Szenarien sind offline, auf `test` Overlay gepinnt und nutzen feste Konfiguration (`ConfigV1` via `configs/test.toml`).

## Artefakte

Pro Szenario/OS werden Artefakte unter `fixtures/goldens/<os>/<scenario>/` geschrieben:

- `golden_manifest.json`
  - Policy- und Config-Digest-Prefixes
  - Tick-Digest-Samples (bounded)
  - Fixed-point Scalar-Summary (`*_q`)
  - erwarteter Gate-Status (`PASS`)
- `explain_tick_last.json`
- `gate_report.json`
- `spec_snapshot.md`

## Befehle

Generieren:

```bash
cargo run -p ucf-ops -- goldens generate --scenario golden_a --os linux
```

Verifizieren:

```bash
cargo run -p ucf-ops -- goldens verify --scenario golden_a --os linux
```

Explizites Update (review-pflichtig):

```bash
cargo run -p ucf-ops -- goldens update --scenario golden_a --os linux
```

## CI/Regression-Regel

- Linux-Job verifiziert Linux-Goldens.
- Windows-Job verifiziert Windows-Goldens.
- Bei absichtlichen Änderungen nur via `goldens update` und committed Artifacts.
- Verifikation ist konservativ:
  - gleiches OS: strikter Vergleich der Digest-Prefixes + Scalar/Struktur
  - cross-OS: Scalar/Struktur bleibt verbindlich; Digests können OS-spezifisch sein
