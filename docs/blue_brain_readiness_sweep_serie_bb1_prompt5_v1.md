# Serie BB1 Prompt 5: Readiness Sweep und Blue-Brain-Integrationsgrundlinie (repo-basiert)

Status: harte Abschlussprüfung der Serie-BB1-Basis auf dem abgeschlossenen Compute-Kern,
ohne neue Compute-Core-Arbeit und ohne neue Architekturspur.

Compute-Referenzlinie (unverändert, kanonisch):
`submit -> compute_canonical -> result/fault/status -> execution_snapshot`
(dieselbe finale Referenzlinie; `CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})` bleibt der Entry-Anchor).

## 1) Repo-basierter Abschlusscheck (BB1-Kern)

- Blue-Brain integration map: vorhanden in
  `docs/blue_brain_integration_map_serie_bb1_prompt1_v1.md` +
  `CANONICAL_BLUE_BRAIN_INTEGRATION_MAP`.
- Blue-Brain-facing contracts: vorhanden in
  `docs/blue_brain_facing_contracts_serie_bb1_prompt2_v1.md` +
  `CANONICAL_BLUE_BRAIN_FACING_CONTRACT_MAP`.
- Blue-Brain-to-compute handoff semantics: vorhanden in
  `docs/blue_brain_compute_handoffs_serie_bb1_prompt3_v1.md` +
  `CANONICAL_BLUE_BRAIN_COMPUTE_HANDOFF_MAP`.
- erster echter Blue-Brain integration candidate: vorhanden in
  `docs/blue_brain_integration_candidate_serie_bb1_prompt4_v1.md` +
  `CANONICAL_BLUE_BRAIN_INTEGRATION_CANDIDATE_MAP`.

## 2) Serie-BB1-Abschlussmatrix (kurz, technisch)

| Bereich | Einstufung | Repo-basierte Aussage |
| --- | --- | --- |
| Blue-Brain-facing inference/status/evidence/state-adjacent contracts | stable Blue-Brain integration foundation | `blue_brain_inference_facing_execution_contract`, `blue_brain_status_health_trust_contract`, `blue_brain_evidence_reference_contract`, `blue_brain_state_facing_context_reference_contract` sind stabil auf derselben finalen Compute-Linie gebunden. |
| Blue-Brain-to-compute handoff map (5 Klassen) | stable Blue-Brain integration foundation | `blue_brain_to_compute_inference_handoff`, `blue_brain_to_compute_status_diagnostics_handoff`, `blue_brain_to_compute_evidence_reference_handoff`, `blue_brain_to_compute_state_adjacent_reference_handoff` + explizite non-canonical boundary sind festgezogen. |
| `runtime_orchestrator_stateful_loop` als erster realer Kandidat | integration-usable with caveats | Als `selected_first_real_blue_brain_integration_candidate` gesetzt, aber mit bewussten Caveats bzgl. progressiver Canonicalisierung und ohne Promotion interner/compat Pfade. |
| `ops_compute_probe` | preparatory / not yet a true integration path | technisch hilfreich als adjacent anchor für Contract/Handoff-Checks, aber kein stateful Blue-Brain-Orchestrierungskern. |
| `integration_runtime_orchestrator`, `runtime_orchestrator_env_bootstrap` | preparatory / not yet a true integration path | gemischte/transitional Flächen; nicht als primäre Blue-Brain-Basis verwenden. |
| `runtime_orchestrator_legacy_engine`, `legacy_runtime_bridge`, `runtime_compute_micro_benchmark`, `compute_status_helper_layer` | intentionally deferred | explizit aus erster realer Blue-Brain-Integrationsbasis ausgeschlossen (compat/internal/helper boundary). |

## 3) Explizite Blue-Brain-Integrationsgrundlinie ab BB1

1. Reale Blue-Brain-nahe Integrationsfläche sitzt auf der finalen Compute-Linie:
   `submit -> result/fault/status` plus outward status/evidence export references.
2. Kanonische BB1-Contracts/Handoffs sind:
   - inference-facing execution contract
   - status health/trust contract
   - evidence reference contract
   - state-adjacent context/handoff-state reference contract
   - handoff lanes: inference/status/evidence/state-adjacent
3. Bewusst akzeptierte Caveats:
   - erster realer Kandidat bleibt `plausible with caveats`, nicht „vollständig produktionsreif“.
   - non-canonical expert/internal lanes bleiben technisch nutzbar, aber nicht Blue-Brain-facing Standard.
4. Ab hier gilt als harte Integrationsregel:
   weitere Blue-Brain-Arbeit muss auf dieser outward Contract/Handoff-Linie aufsetzen;
   kein Rückfall auf compute-interne, legacy- oder helper-dominierte Pfade als primäre Integrationsautorität.

## 4) Nächste Richtungen nach BB1 (nur echter technischer Hebel)

1. **Serie BB2: Blue-Brain state/runtime architecture on top of compute integration**
   - Ziel: runtime_orchestrator_stateful_loop auf den kanonischen state-adjacent + status/evidence handoff bindings weiter stabilisieren.
2. **Serie BB3: Blue-Brain memory/context integration**
   - Ziel: context-digest-/handoff-state-Referenzen konsistent in einen belastbaren Blue-Brain-memory/context-Pfad überführen, ohne neue zweite Compute-Wahrheitsquelle.
3. **Serie BB4: Neural dynamics integration candidates (Hodgkin-Huxley/Kuramoto) erst nach belastbarer Kernanbindung**
   - Ziel: nur nach stabiler BB2/BB3-Kernanbindung evaluieren; davor explizit nachrangig.

## 5) Priorisierte nächste Richtung

**Priorität 1: Serie BB2 (state/runtime architecture auf bestehender Integrationslinie).**

Kurze Begründung:
- höchster unmittelbarer Hebel, weil der erste reale Kandidat (`runtime_orchestrator_stateful_loop`)
  bereits existiert und genau dort Caveats abgebaut werden müssen.
- BB3 ist danach sinnvoll, weil memory/context auf stabiler runtime/state-Linie aufbauen sollte.
- BB4 ist klar nachrangig, weil neural-dynamics Kandidaten ohne belastbare Kernanbindung
  sonst wieder eine zweite Integrationsspur erzeugen würden.
