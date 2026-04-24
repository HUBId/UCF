# Serie BB11 Prompt 1: Deterministischer Neuromod-Delta-Downstream-Pfad

Status: **minimal operational geschlossen** für advisory-only Delta-Weitergabe im Router-Pfad.

## Kanonischer Pfad

1. **Quelle**: `BlueBrainPort::stimulate` erzeugt `BrainResponse { delta: NeuromodDelta, ... }`.
2. **Übergabe**: Router setzt `pending_neuromod_delta` beim BlueBrain-Stimulus-Schritt.
3. **Kanonischer Konsumpunkt**: im selben Verify-Puls konsumiert Router das Pending-Delta deterministisch (`take`-Semantik).
4. **Produktiver Downstream**:
   - Veröffentlichung als Workspace-Signal `BRAIN_NEUROMOD_HINT=...` (sichtbar im Snapshot/Broadcast),
   - Append eines dedizierten Archive-Records (`RecordKind::Other(166)`) mit Delta-Commit.
5. **Lebensdauer**:
   - Delta ist pro Zyklus **ephemeral**,
   - wird bei Konsum aus `pending_neuromod_delta` entfernt,
   - ein späteres Delta ersetzt frühere Pending-Werte nur bis zum Konsumpunkt.

## Semantik

Die Downstream-Semantik ist **Runtime-Caveat/Modulationshinweis als Workspace-Broadcast**, nicht direkte Entscheidungs- oder Ausführungsautorität.

- Kein direkter Tool-/Action-Call.
- Keine direkte Policy-Entscheidung.
- Keine direkte Compute-Invocation.
- Keine Safety-Override-Semantik.
- Keine direkte Memory-Persistenz.

## Determinismus

- Konsum läuft als deterministischer read-once Schritt im Verify-Puls.
- Workspace-Signal-Text ist kanonisch formatiert (DA/SE/NE/CO, fester Schlüsselraum).
- Archive-Record nutzt den Delta-Commit als stabile Grenze.

## Hinweis zu Runtime/Policy/Bridge

- `ucf-bluebrain-bridge` bleibt zuständig für deterministische Stimulus-Kodierung.
- `ucf-policy`/`ucf-runtime` no-direct-* und advisory-only Grenzen bleiben unverändert.
- Es wurde keine zweite Delta-Sprache eingeführt.
