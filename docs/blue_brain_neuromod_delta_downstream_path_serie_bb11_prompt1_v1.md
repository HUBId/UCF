# Serie BB11 Prompt 1: Deterministischer Neuromod-Delta-Downstream-Pfad

Status: **minimal operational geschlossen** für advisory-only Delta-Weitergabe im Router-Pfad.

## Kanonischer Pfad

1. **Quelle**: `BlueBrainPort::stimulate` erzeugt `BrainResponse { delta: NeuromodDelta, ... }`.
2. **Übergabe**: Router setzt `pending_neuromod_delta` beim BlueBrain-Stimulus-Schritt.
3. **Kanonischer Konsumpunkt**: im selben Verify-Puls konsumiert Router das Pending-Delta deterministisch (`take`-Semantik).
4. **Produktiver Downstream**:
   - Veröffentlichung als Workspace-Signal `BRAIN_NEUROMOD_HINT=...` (sichtbar im Snapshot/Broadcast),
   - Append eines dedizierten Archive-Records (`RecordKind::Other(166)`) mit Delta-Commit.
   - Seit BB11 Prompt 2 wird an diesem **gleichen** Konsumpunkt zusätzlich eine
     deterministische Kuramoto-Runtime-Caveat-Auswertung gerechnet und als
     `KURAMOTO_RUNTIME=... KURAMOTO_COHERENCE=...` im gleichen
     `BRAIN_NEUROMOD_HINT`-Signal mitgeführt.
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

Kuramoto bleibt dabei **advisory-only**:

- Eingänge stammen aus vorhandenem Flow (Evidence-Refs, Lens-/Workspace-Kontext,
  Attention, Runtime-Snapshot, vorhandenes Neuromod-Delta).
- Ausgänge bleiben auf Runtime-Caveat-Hinweise im bestehenden Delta-Downstream
  begrenzt (keine zweite Modulationssprache, kein zusätzlicher Autoritätskanal).

## Determinismus

- Konsum läuft als deterministischer read-once Schritt im Verify-Puls.
- Workspace-Signal-Text ist kanonisch formatiert (DA/SE/NE/CO, fester Schlüsselraum).
- Archive-Record nutzt den Delta-Commit als stabile Grenze.

## Hinweis zu Runtime/Policy/Bridge

- `ucf-bluebrain-bridge` bleibt zuständig für deterministische Stimulus-Kodierung.
- `ucf-policy`/`ucf-runtime` no-direct-* und advisory-only Grenzen bleiben unverändert.
- Es wurde keine zweite Delta-Sprache eingeführt.
