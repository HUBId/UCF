# Serie BB16 Prompt 1: bounded dynamics interaction with real execution (advisory-only)

Diese Linie zieht die erste **begrenzte Rückkopplung** zwischen

- bounded neural-dynamics (BB12 Kuramoto advisory-only),
- real minimal execution (BB13),
- execution-integrity + canonical result references (BB14),
- bounded retrieval/reference basis (BB15)

zusammen, **ohne** Action-Autorität für Dynamics einzuführen.

## Kanonische dynamics-execution feedback states

`CANONICAL_BLUE_BRAIN_DYNAMICS_EXECUTION_FEEDBACK_MAP` unterscheidet:

1. `execution_informed_dynamics_input`
2. `reference_informed_dynamics_input`
3. `caveated_execution_informed_dynamics_input`
4. `insufficient_dynamics_feedback_basis`
5. `blocked_dynamics_feedback_basis`
6. `unavailable_dynamics_feedback_basis`
7. `diagnostic_only_dynamics_feedback`
8. `non_canonical_internal_only_feedback_path`

## Erlaubte Inputs in die Dynamics-Linie (bounded)

Erlaubt sind ausschließlich advisory-only Input-Basen:

- canonical execution result references (z. B. `...:result:completed`),
- failed/cancelled execution references als **caveated basis**,
- blocked execution references als **blocked basis**,
- unavailable/unsupported execution references als **unavailable basis**,
- bounded context/evidence/reference basis,
- diagnostic-only references (z. B. `diag:*`, placeholder refs) nur als diagnostic-only feedback.

## Explizit verboten (no-direct-*)

Die Dynamics-Rückkopplung bleibt nicht-autoritativ:

- kein direct re-execute,
- kein direct re-execute trigger,
- kein direct retry orchestration,
- kein direct action selection,
- kein direct memory commit,
- kein direct compute invocation,
- kein safety override.

Diese Grenzen sind in Kuramoto boundary guards fest verdrahtet (`false`) und werden durch Caveats
sichtbar gehalten.

## Erfolgs-/Fehlertrennung bleibt hart

Execution-Erfolgsbasis und nicht erfolgreiche Basis werden getrennt gehalten:

- `completed` kann execution-informed dynamics input stützen,
- `failed`/`cancelled` bleibt caveated,
- `blocked` bleibt blocked, `unavailable`/`unsupported` bleibt unavailable,
- fehlende Basis bleibt insufficient.

Keine implizite Aufwertung, keine Glättung in „erfolgreich“.

## Runtime-/Selection-Rückbindung (sichtbar, aber advisory-only)

Kuramoto-Hints publizieren den beobachteten execution-feedback-state als
`KURAMOTO_EXEC_FEEDBACK=...`, sodass Runtime/Selection die Rückkopplungsbasis beobachten kann,
ohne daraus direkte Ausführung/Memorieschreibungen abzuleiten.
