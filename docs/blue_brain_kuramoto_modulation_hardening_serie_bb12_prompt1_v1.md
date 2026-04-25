# Serie BB12 Prompt 1: Bounded Kuramoto modulation hardening (operative Linie)

Status: **gehärtet** auf der bestehenden BB11-Operationslinie (kein neuer Dynamics-Stack).

## 1) Kanonische operative Aufrufstelle bleibt unverändert

- Der Router ruft Kuramoto weiterhin ausschließlich beim Konsum von `pending_neuromod_delta` auf.
- Die Wirkungskette bleibt: `NeuromodDelta -> evaluate_blue_brain_kuramoto_modulation -> BRAIN_NEUROMOD_HINT`.
- Es gibt weiterhin **keine** direkte Action-/Tool-/Memory-/Compute-/Policy-/Safety-Autorität.

## 2) Kanonische bounded modulation states

Die Kuramoto-Linie führt jetzt explizit genau diese Zustände:

1. `applied_advisory_only`
2. `caveated`
3. `insufficient`
4. `ignored`
5. `no_op`
6. `blocked`
7. `unavailable`
8. `non_canonical_internal_only_path`

Damit sind Erfolgs-, Grenz- und Nichtanwendungsfälle deterministisch und diagnostisch unterscheidbar.

## 3) Deterministische Semantik

- Input-Kanonisierung bleibt aktiv (sort/dedup für refs/caveats, deterministische Node-Reihenfolge).
- `insufficient` ist strikt an zu wenige Phase-Nodes gebunden.
- `unavailable` und `ignored` sind strikt scope-basiert.
- `blocked` ist strikt posture-basiert.
- `caveated` ist strikt an explizite Caveat-Signale gebunden.
- `no_op` bleibt explizit erkennbar (neutraler Hint/Caveat-Ausgang trotz ausgewertetem Pfad).

## 4) Runtime-/Selection-Rückbindung (advisory-only)

- Workspace-Hints enthalten jetzt zusätzlich:
  - `KURAMOTO_STATE=...`
  - `KURAMOTO_RUNTIME=...` (inkl. `none`)
  - `KURAMOTO_COHERENCE=...`
  - `KURAMOTO_CAVEAT=...`
- Diese Tokens sind diagnostisch/advisory-only; sie treffen keine direkte Selection-/Runtime-Entscheidung.

## 5) no-direct-* Guards bleiben hart

Kuramoto- und HH-Boundary-Guards bleiben unverändert strikt `false` für:

- action execution / actual action result,
- tool invocation,
- runtime/selection state mutation,
- memory persistence/commit,
- compute invocation,
- policy decision,
- safety override.

## 6) Operative Abgrenzung

- Keine neue Neurodynamikplattform.
- Keine neue Agenten-/Tool-Execution-Logik.
- Keine HH-Produktivübernahme.
- Keine zweite operative Dynamics-Linie neben der bestehenden Kuramoto-Minimallinie.
