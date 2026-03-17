# Bundle Spine v8

`CanonicalBundleSpineV1` ist der kanonische Nachweis, dass Repro-Pack- und BugKit-Bundles dieselbe Bundle-Consumption-Wahrheit für Build, Verify und Inspect verwenden.

## Was CanonicalBundleSpineV1 beweist

- Bundle-Typ (`REPRO` oder `BUGKIT`)
- Applied-Supported-Set-Digest-Prefix
- Kanonischer Governance-Entry-Digest-Prefix
- Optionaler Canonical-Readiness-Spine-Digest-Prefix
- Bundle-Consumption-Context-Digest-Prefix
- Artifact-Refs-Digest-Prefix
- Roundtrip-Consistency-Digest-Prefix
- Gesamtstatus (`PASS`/`FAIL`) und stabiler Spine-Digest

Damit wird überprüfbar, dass Bundle-Kontext, Scope, Governance-Referenzen und Readiness-Referenzen deterministisch zusammenpassen.

## Beziehung zu `exports roundtrip-check`

- `exports roundtrip-check` validiert primär Manifest-/Roundtrip-Konsistenz.
- `exports bundle-spine-check` ist der End-to-End-Beweis über dieselbe Consumption-Spine inklusive Scope/Governance/Readiness-Kohärenz.

## Warum ein gemeinsamer Spine für Build/Verify/Inspect

Ein gemeinsamer Spine verhindert Drift:

- zwischen Build und Verify (gleiche Rekonstruktion),
- zwischen Verify und Inspect (gleiche semantische Auswertung),
- zwischen Bundle-Inhalt und Governance-/Readiness-Kontext.

## Befehl

```bash
cargo run -p ucf-ops -- exports bundle-spine-check --in <bundle.zip> --out ./out/bundle_spine_check.json
```

Die Ausgabe enthält `PASS`/`FAIL`, Mismatch-Kategorien und den kanonischen Bundle-Spine.
