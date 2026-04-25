# Serie BB12 Prompt 2: Kuramoto modulation diagnostics / caveat / no-op feedback hardening

Status: **gehärtet** auf der bestehenden operativen BB11/BB12-Linie (kein neuer Dynamics-Stack).

## 1) Kanonische Kuramoto diagnostics map

Die operative Kuramoto-Linie unterscheidet jetzt deterministisch:

1. `modulation_applied_diagnostic`
2. `modulation_caveated_diagnostic`
3. `modulation_insufficient_diagnostic`
4. `modulation_ignored_diagnostic`
5. `modulation_no_op_diagnostic`
6. `modulation_blocked_diagnostic`
7. `modulation_unavailable_diagnostic`
8. `non_canonical_internal_only_dynamics_diagnostic`

Diese Diagnostics-Klassen spiegeln direkt den bestehenden bounded
`modulation_state` wider (kein zweites Dynamics-Vokabular).

## 2) Kompakte deterministische Reason-Tags

Zusätzlich trägt Kuramoto für nicht-triviale Fälle einen kompakten Grund:

- `insufficient_input_group_basis`
- `caveated_partial_or_weak_basis`
- `no_op_neutral_deterministic_result`
- `ignored_by_runtime_selection_context`
- `blocked_by_guard_boundary_condition`
- `unavailable_operational_preconditions`
- `non_canonical_internal_only_path`

`applied_advisory_only` bleibt absichtlich ohne Zusatzgrund (`none`), damit
keine spekulative Begründungsprosa entsteht.

## 3) Runtime-/Selection-Rückbindung (advisory-only)

Der Router publiziert weiter genau einen `BRAIN_NEUROMOD_HINT` entlang des
produktiven Delta-Pfads, jetzt mit zusätzlich gehärteter Diagnostik:

- `KURAMOTO_STATE=...`
- `KURAMOTO_DIAGNOSTIC=...`
- `KURAMOTO_REASON=...`
- `KURAMOTO_RUNTIME=...`
- `KURAMOTO_COHERENCE=...`
- `KURAMOTO_CAVEAT=...`

Die Felder bleiben rein diagnostisch/advisory-only und führen zu keiner
direkten Selection-/Runtime-/Action-/Memory-/Compute-/Policy-Entscheidung.

## 4) Caveat/No-op/Ignored/Blocked/Unavailable Sichtbarkeit

- `insufficient` bleibt strikt an zu wenig Input-Gruppen (`phase_nodes < 2`) gebunden.
- `caveated` bleibt an explizite Caveat-Basis gebunden (z. B. Desynchrony/Posture-Caveat).
- `no_op` bleibt als neutral deterministischer Ausgang explizit sichtbar.
- `ignored` bleibt scope-/Kontext-basiert sichtbar.
- `blocked` bleibt guard-/boundary-basiert sichtbar.
- `unavailable` bleibt an fehlende operative Preconditions gebunden.

## 5) No-direct-* Guards bleiben hart und diagnostisch sichtbar

Für `blocked`, `unavailable` und `non_canonical_internal_only_path` werden
zusätzlich explizite Guard-Caveats getragen:

- `no_direct_action_allowed`
- `no_direct_memory_allowed`
- `no_direct_compute_allowed`
- `no_safety_override_allowed`

Das bleibt strikt diagnostisch; es entsteht **keine** neue Policy-/Autoritätsschicht.

## 6) Delta-/Downstream-Konsistenz

Compute, Router und Workspace benutzen jetzt dieselben kanonischen Tokens für
state/diagnostic/reason/caveat entlang des produktiven Delta-Downstream-Pfads.
Damit gibt es keine konkurrierende zweite Diagnostics-Sprache zwischen Evaluator
und Downstream-Hint.
