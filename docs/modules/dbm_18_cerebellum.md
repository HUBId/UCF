# DBM 18 Cerebellum

## Zweck
DBM 18 ist die Cerebellum-nahe Modulfläche für die BR5-Rollenkarte. Die kanonische UCF-Systemrolle ist `cerebellum_like_region`: advisory prediction/timing/correction/mismatch support, bounded execution-support und keine direkte Autorität. Die maßgebliche Regionsdefinition steht in `docs/blue_brain_cerebellum_region_role_map_serie_br5_prompt1_v1.md`; diese Moduldatei erzeugt keine zweite Wahrheitsquelle.

## Inputs
Zulässige Inputs bleiben bounded und diagnostisch: Prediction-, Timing-/Coordination-, correction-/mismatch- und execution-feedback-nahe Lesarten aus bestehenden Runtime-/Selection-/Execution-interface-/Reference-Linien. Nicht zulässig sind direkte Action-, Retry-, Memory-Commit-, Safety-Override- oder Compute-Trigger.

## State
Der aktuelle Zustand ist `abstract functional current mode`. Bounded Kuramoto-like timing coupling bleibt ein späterer Kandidat; Hodgkin-Huxley bleibt simulation-only/diagnostic-only bzw. später selektiv neu zu scopen und ist keine Produktivpflicht.

## Outputs
Outputs sind advisory-only: Kalibrierungs-, Timing-, Correction-/Mismatch- und bounded execution-support Hinweise. Sie dürfen bestehende Contract-/Diagnostic-Lesarten caveated, deferred, blocked oder insufficient machen, aber keine Ausführung freigeben und keine Auswahl treffen.

## Regeln
Deterministische Regeln (sortierte RCs, tighten-only) gelten analog zu den bestehenden BlueBrain-Leitplanken. Cerebellum-Signale bleiben read-only/advisory für Konsumenten und dürfen keine neue Plattform, keine allowed-actions-Erweiterung und keine Compute-Core-Arbeit öffnen.

## Invarianten
- no direct action trigger
- no direct action selection
- no direct execution trigger
- no retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override
- keine biologische Vollsimulation

## Tests
BR5-Tests pinnen die kanonischen Rollen, den Integrationsmodus, die Abgrenzung zu Hippocampus/Amygdala/Thalamus/Basal Ganglia und die no-direct-Scope-Grenzen. Spätere Golden-Stream- oder Anti-Flapping-Tests dürfen nur auf dieser Rollenkarte aufbauen.

## Observability
Observability bleibt diagnostisch: Cerebellum markers dürfen Prediction-/Timing-/Correction-/Mismatch-Lesarten sichtbar machen, aber keine Runtime-, Selection-, Execution-, Retry-, Memory- oder Compute-Autorität erzeugen.

## Microcircuit
Microcircuit-Details wie Purkinje-/Granule-/Deep-Nuclei-/Spiking-Pfade bleiben deferred. Ein späterer Microcircuit-Pfad muss separat als simulation-only/diagnostic-only oder later selective HH deepening begründet werden.

## BR5 Prompt 3 diagnostics/contract hardening

Die gehärtete Cerebellum-Schnitt verwendet die kanonische diagnostics/contract map aus `docs/blue_brain_cerebellum_surface_diagnostics_contracts_hardening_serie_br5_prompt3_v1.md`: advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only, bounded contract signal und non-canonical/internal-only. Runtime, Selection, Execution-interface und Reference lesen denselben canonical contract read. Cerebellum bleibt prediction/timing/correction/mismatch-lastig, advisory-only oder caveated, und erzeugt keine direkte Action-, Execution-, Retry-, Memory-, Compute- oder Safety-Autorität.
