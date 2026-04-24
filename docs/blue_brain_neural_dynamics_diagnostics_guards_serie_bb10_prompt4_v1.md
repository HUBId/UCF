# Serie BB10 Prompt 4: Neural-dynamics diagnostics backbind + no-direct-* guards

Status: **minimal integriert** als kanonische Diagnostics-/Guard-Schicht in `runtime/ucf-compute/src/blue_brain_dynamics.rs`.

## 1) Kanonische Dynamics-Diagnostics-Map

`CANONICAL_BLUE_BRAIN_DYNAMICS_DIAGNOSTICS_MAP` unterscheidet explizit:

- `dynamics diagnostic observed`,
- `kuramoto modulation diagnostic`,
- `hodgkin-huxley simulation diagnostic`,
- `dynamics caveated`,
- `dynamics insufficient`,
- `dynamics failed`,
- `dynamics unavailable`,
- `dynamics ignored`,
- `non-canonical/internal-only dynamics diagnostic`.

Damit sind Caveat/Insufficiency/Failure/Unavailable/Ignored sauber getrennt und auditierbar.

## 2) Runtime-/Selection-Backbind ohne direkte Mutation

Kuramoto-/HH-Signale werden als Feedback zurückgebunden, nicht als direkte Entscheidung:

- Runtime: `runtime modulation observed` / `dynamics caveat attached` / `dynamics insufficient for modulation` / `dynamics ignored for current transition`.
- Selection: `selection modulation observed` / `dynamics caveat attached` / `dynamics insufficient for modulation` / `dynamics ignored for current selection`.

Die Backbind-Klassen informieren BB2/BB4-Diagnostik, lösen aber **keine** direkte Runtime-Mutation oder Selection-Entscheidung aus.

## 3) Harte no-direct-action / no-direct-memory / no-direct-compute / no-safety-override Guards

Kuramoto- und HH-Guard-Strukturen setzen kanonisch auf `false`:

- keine direkte Action Execution,
- keine Tool Invocation,
- kein actual action result,
- kein Memory Commit und keine Memory Persistence/Mutation,
- keine Compute Invocation,
- kein Safety Override,
- keine Policy-Entscheidung.

Zusätzlich bleiben Runtime-/Selection-Mutation explizit gesperrt.

## 4) Kuramoto- und HH-Rückbindung im BB10-Scope

- **Kuramoto** bleibt minimaler Modulationspfad:
  - kann advisory Selection-/Runtime-Caveat-Feedback liefern,
  - `simulation-only` wird als `dynamics ignored` markiert,
  - `not implemented/not suitable` wird als `dynamics unavailable` markiert.
- **Hodgkin-Huxley** bleibt simulation-/diagnostic-only:
  - Summary/Caveated/Insufficient bleiben getrennt,
  - `non-canonical/internal-only` wird explizit als non-canonical klassifiziert,
  - Runtime-/Selection-Feedback bleibt `ignored`.

## 5) Bewusst nicht umgesetzt

Weiterhin **nicht** implementiert:

- vollständige Neural-Dynamics- oder SNN-Plattform,
- direkte Action-/Tool-Ausführung aus Dynamics,
- direkte Memory-Persistenz aus Dynamics,
- direkte Compute-Invocation aus Dynamics,
- Safety- oder Policy-Autoritätsübernahme durch Dynamics.
