# Serie BB10 Prompt 3: Hodgkin-Huxley als simulation-only / diagnostic-only boundary

Status: **type-level + contract-level konsolidiert** in `runtime/ucf-compute`; keine produktive Runtime-/Selection-/Action-Kopplung.

## 1) Entscheidung und Scope

Hodgkin-Huxley wird in BB10 Prompt 3 explizit begrenzt auf:

- `simulation-only Hodgkin-Huxley`,
- `diagnostic-only Hodgkin-Huxley`,
- `research/deferred Hodgkin-Huxley`,
- `not suitable for current Blue-Brain runtime`,
- `non-canonical/internal-only HH path`.

Damit ist HH in dieser Linie **kein** produktiver Blue-Brain Runtime-/Selection-/Action-Pfad.

## 2) Input Surface (erlaubt / verboten)

Erlaubte Inputs (`BlueBrainHodgkinHuxleyDiagnosticInput`):

- `diagnostic_run_id` als explizite Laufidentität,
- `context_refs` (nur Referenzen),
- `evidence_refs` (nur Referenzen),
- `simulation_parameters` (bounded integration steps, dt, stimulus),
- `model_parameters` (bounded sodium/potassium/leak proxies).

Explizit verboten (nicht Teil des HH-Input-Surfaces):

- raw compute internals,
- expert/internal-only hooks als kanonischer Input,
- direkte Runtime-State-Mutation (keine direkte Runtime-State-Mutation),
- direkte Selection-Mutation (keine direkte Selection-Mutation),
- direkte Memory-Mutation,
- direkte Action-/Safety-State-Mutation.

## 3) Output Surface (erlaubt / verboten)

Erlaubte Outputs (`BlueBrainHodgkinHuxleyDiagnosticResult`):

- simulation diagnostic summary,
- caveated diagnostic,
- failed/insufficient simulation diagnostic,
- diagnostic trace/reference (`diag:hh:<run_id>`),
- bounded metadata.

Explizit verboten (nicht erzeugbar):

- selection hint mit direkter Wirkung,
- runtime modulation mit direkter Wirkung,
- Memory-Commit (kein Memory-Commit),
- Compute-Invocation (keine Compute-Invocation),
- Action-Ausführung/Tool-Invocation,
- Safety-Entscheidung oder Safety-Override (kein Safety-Override),
- Policy-Result.

## 4) Guards gegen direkte Systemwirkung

`BlueBrainHodgkinHuxleyBoundaryGuard` erzwingt `false` für:

- Runtime-State-Mutation,
- Selection-Mutation,
- Memory-Mutation,
- Action Execution,
- Tool Invocation,
- Compute Invocation,
- Safety Override,
- Policy Decision.

Die HH-Lane bleibt damit diagnostisch/simulativ und nicht-hochmächtig.

## 5) Abgrenzung zu Kuramoto (Prompt 2)

- Kuramoto bleibt der **leichtere advisory modulation path** für Selection/Runtime-Caveat (hint-only / caveat-signal-only).
- Hodgkin-Huxley bleibt detaillierter und schwergewichtiger und damit in dieser Phase simulation-/diagnostic-only.
- HH bekommt **nicht** dieselbe Modulationsmacht wie Kuramoto.

## 6) Umsetzungsgrad in diesem Schritt

- Keine vollständige biophysikalische HH-Großsimulation.
- Keine SNN-/Brain-Simulationsplattform.
- Keine Öffnung des Compute-Core.
- Stattdessen: harte Boundary-Typen, begrenzte I/O-Surface und explizite Guard-Semantik für BB2-BB9-kompatiblen Diagnostic-Use.
