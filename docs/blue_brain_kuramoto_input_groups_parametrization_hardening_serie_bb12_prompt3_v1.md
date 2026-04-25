# Serie BB12 Prompt 3: operative Input-Gruppen und Parametrisierung des Kuramoto-Pfads stabilisieren

Status: **gehärtet** auf der bestehenden operativen BB11/BB12-Linie (kein neuer Dynamics-Stack).

## 1) Kanonische Input-Gruppen (operativer Umfang)

Die Kuramoto-Eingangsfläche wird jetzt über eine feste kanonische Gruppenkarte geführt:

1. `runtime_state_group`
2. `selection_attention_group`
3. `context_reference_group`
4. `memory_caveat_reference_group`
5. `evidence_derived_advisory_group`
6. `unsupported_non_canonical_input_group`

Damit ist klar, welche operativen Runtime-/Selection-/Context-/Memory-/Evidence-nahen
Signale in den bounded Pfad gehören, und welche explizit als nicht-kanonisch laufen.

## 2) Deterministische Gruppierungslogik

- `phase_nodes` werden deterministisch sortiert/dedupliziert.
- Nur kanonische Gruppenrefs werden in die Kohärenzberechnung übernommen.
- Nicht-kanonische `phase_nodes` erzeugen explizit Caveats (`unsupported_phase_node_group_ref`).
- Zusätzliche nicht-kanonische Inputflächen werden als `unsupported_input_refs` bzw.
  `blocked_input_refs` getrennt geführt.

Dadurch gibt es keine freie heuristische Gruppierung und keine impliziten Sonderpfade.

## 3) Minimale stabile Parametrisierung

Die bestehende bounded Parametrisierung bleibt minimal:

- `phase_permille` ringgebunden (`0..=999` via `% 1000`),
- `coupling_permille` begrenzt (`<= 1000` in der Kohärenzberechnung),
- deterministische Paar-Distanz + gewichtetes Mittel zur `coherence_permille`.

Der Router speist die kanonischen Gruppen weiterhin aus realen operativen Signalen
(Attention/Runtime/Neuromod-Delta), ohne zweite Tuning-Engine.

## 4) Explizite Input-Basis-Semantik

Zusätzlich zum `modulation_state` wird eine klare Input-Basis geführt:

- `valid_input_basis`
- `caveated_input_basis`
- `insufficient_input_basis`
- `unsupported_input_basis`
- `blocked_input_basis`
- `no_op_neutral_input_basis`

Damit bleiben valid/caveated/insufficient/unsupported/blocked/no-op Fälle
auf der Eingangsseite explizit unterscheidbar.

## 5) Rückbindung und no-direct-* Grenzen

Die Runtime-/Selection-Rückbindung bleibt advisory-only (Hint/Caveat).

`blocked`/`unavailable`/`non_canonical_internal_only_path` tragen weiterhin explizite
no-direct-* Caveats (`no_direct_action_allowed`, `no_direct_memory_allowed`,
`no_direct_compute_allowed`, `no_safety_override_allowed`).

Es wird keine direkte Tool-/Action-/Policy-/Safety-Autorität über Inputwahl geöffnet.
