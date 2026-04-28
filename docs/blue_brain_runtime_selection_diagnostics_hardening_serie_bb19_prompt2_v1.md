# Serie BB19 Prompt 2: Runtime/Selection Diagnostics & Contract Feedback Hardening

Status: **stable bounded diagnostics hardening** auf der bestehenden BB19-Contract-Line (keine Planner-/Agent-/Policy-/Orchestration-/Explainability-Plattform).

## 1) Kanonische Runtime/Selection-Diagnostics-Map

Die operative Runtime/Selection-Diagnostik bleibt auf acht Klassen begrenzt:

- `runtime_to_selection_contract_diagnostic`
- `selection_to_runtime_contract_diagnostic`
- `deferred_contract_diagnostic`
- `blocked_contract_diagnostic`
- `caveated_contract_diagnostic`
- `insufficient_contract_diagnostic`
- `advisory_only_contract_diagnostic`
- `non_canonical_internal_only_contract_diagnostic`

Diese Klassen bilden den bestehenden Contract ab und erzeugen **keine** zweite Contract-Sprache.

## 2) Kompakte kanonische Contract-Gründe

Diagnostics transportieren deterministische Reason-Tokens:

- `deferred_due_to_bounded_priority_selection_state`
- `blocked_due_to_contract_boundary_or_reference_weakness`
- `caveated_due_to_weak_or_partial_reference_dynamics_execution_basis`
- `insufficient_due_to_missing_bounded_contract_basis`
- `advisory_only_no_direct_action_authority`
- `non_canonical_internal_only_path_excluded`

Kein freies Prosa-Reasoning als Kernmodell.

## 3) Richtungsabgleich Runtime ↔ Selection

- Runtime → Selection bleibt diagnostisch als eigenständige Richtung sichtbar.
- Selection → Runtime bleibt diagnostisch als eigenständige Richtung sichtbar.
- `deferred` bleibt strikt getrennt von `blocked`.
- `blocked` bleibt strikt getrennt von `failed execution`.
- `caveated` bleibt strikt getrennt von starkem Contract-Signal.
- `insufficient` bleibt strikt getrennt von `blocked`.
- `advisory_only` bleibt strikt getrennt von Entscheidungs- oder Action-Autorität.

## 4) Bounded Einbindung von Dynamics/Execution/References

- `execution_informed_dynamics_input` und `reference_informed_dynamics_input` bleiben sichtbare bounded Einflussquellen.
- `caveated_execution_informed_dynamics_input`, blocked/unavailable/insufficient Basis bleiben als geschwächte Grundlagen explizit.
- Diagnostics bleiben observation-/feedback-orientiert ohne direkte Action-Folge.

## 5) No-direct-* Grenzen (verpflichtend)

- `no_direct_action_execution`
- `no_direct_retry_orchestration`
- `no_direct_compute_invocation`
- `no_implicit_memory_persistence`
- keine Policy-/Agenten-Entscheidungsautorität
- keine Neurodynamik-Autoritätserweiterung

## 6) Operative Wirkung in BB19

BB19 Prompt 2 macht Contract-Signale, Diagnostics und Gründe zwischen Runtime und Selection kompakt und konsistent auswertbar, ohne den bounded Contract in Entscheidungslogik umzudeuten. Damit entsteht eine belastbare Basis für weitere BB19-Härtungsschritte.
