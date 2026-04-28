# Serie BB18 Prompt 3: Production-facing Guard Rails & non-canonical execution path cleanup

Ziel: Die minimale echte Execution-Linie bleibt bewusst schmal und wird production-facing explizit dokumentiert und testseitig gebunden.

## Kanonische production-facing Guard-Rail-Map

Die kanonische Guard-Rail-Map für die minimale Execution-Linie besteht aus genau fünf Klassen:

1. **canonical production guard rail**
   - Canonical surface bleibt auf `emit_canonical_signal` und den vorhandenen Result-/Reference-Semantiken begrenzt.
2. **scope guard rail**
   - Nur `allowed canonical action` ist operativ; kein Scope-Upgrade auf Agenten-, Plattform- oder Orchestrierungslogik.
3. **no-direct-* guard rail**
   - Kein direkter Action-/Compute-/Memory-/Retry-Orchestrierungs-Bypass.
4. **terminal-state guard rail**
   - Terminalzustände (`completed`, `failed`, `cancelled`, `blocked`, `unavailable`, `unsupported`, non-canonical) bleiben explizit und kollabieren nicht.
5. **non-canonical/internal-only execution path exclusion**
   - Internal-only/test-only Pfade bleiben sichtbar ausgeschlossen und nicht operativ.

## Minimal echte Execution (production-facing)

Die minimal echte Execution ist weiterhin ausschließlich die bestehende kanonische Linie:

- `allowed canonical action` auf `emit_canonical_signal`.
- Eligibility + Safety bleiben Eintrittsbedingung.
- Result-/Reference-Integrität bleibt bindend für terminale Zustände.

**Allowed canonical tool call remains deferred** und ist weiterhin nicht als zusätzliche operative Execution-Fläche aktiviert.

## Explizite Ausschlüsse (non-canonical / intern-only)

Folgende Pfade sind production-facing weiterhin ausgeschlossen:

- interne/non-canonical execution paths,
- doppelte operative Execution-Zugänge außerhalb der kanonischen Linie,
- implizite Folge-Execution,
- Retry-/Queue-Orchestrierung,
- automatische Memory-Persistenz.

Kurzform:

- **No implizite Folge-Execution**
- **No Retry-/Queue-Orchestrierung**
- **No automatische Memory-Persistenz**

## Bindung an Eligibility / Safety / Result / Reference

Die Guard-Rails sind nicht isoliert, sondern an die bestehende Linie gebunden:

- Eligibility/Safety entscheiden weiterhin über den Eintritt in die minimale Execution.
- Terminal-State-Integrität bleibt fail-closed bei inkonsistenten Übergängen.
- Result-/Reference-Pfade bleiben kanonisch getrennt von non-canonical/internal-only.

## Unveränderte no-direct-* Grenzen

Bewusst unverändert bleiben:

- keine Agentenlogik,
- keine Policy-/Governance-Plattform,
- keine autonome Multi-Step-Orchestrierung,
- keine implizite Compute-Core-Ausweitung,
- keine implizite Memory-Persistenz.

Damit bleibt die Linie maintenance-only, schmal und production-facing klar begrenzt.
