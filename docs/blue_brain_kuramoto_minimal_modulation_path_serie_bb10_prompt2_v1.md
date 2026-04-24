# Serie BB10 Prompt 2: Kuramoto-first minimal dynamics modulation path (Selection/Runtime)

Status: **minimal implementiert und operativ eingehängt** als begrenzter, deterministischer Modulations-/Diagnostikpfad.

Hinweis: Der kanonische BB11-Abschlussstand der operativen Minimal-Dynamics-Linie steht in `docs/blue_brain_readiness_sweep_serie_bb11_prompt4_v1.md`.

Operative kanonische Aufrufstelle (BB11 Prompt 2):

- `core/crates/ucf-router/src/lib.rs` im Verify-Puls beim Konsum von
  `pending_neuromod_delta` (`consume_pending_neuromod_delta`).
- Dort wird `evaluate_blue_brain_kuramoto_modulation(...)` mit realen
  Flow-Signalen gespeist (Evidence-Refs, Lens-/Workspace-Kontextgruppen,
  Attention-State, Runtime-Snapshot, vorhandenes Neuromod-Delta).
- Das Ergebnis wird advisory-only als Runtime-Caveat-Information in den bereits
  produktiven Neuromod-Downstream (`BRAIN_NEUROMOD_HINT`) eingeschrieben.

## 1) Entscheidung und Scope

Kuramoto wird in BB10 Prompt 2 als **kleinster erster Dynamics-Pfad** entlang der Kandidatenkarte aus Prompt 1 geführt:

- `simulation-only Kuramoto`: weiterhin möglich, aber hier nicht erweitert.
- `diagnostic-only Kuramoto`: implementiert.
- `selection-modulating Kuramoto`: implementiert als **Hint-only**.
- `runtime-caveat-modulating Kuramoto`: implementiert als **Caveat-Signal-only**.
- `not implemented / not suitable now`: explizit als Scope-State vorhanden.

Es wurde **keine** globale Dynamics-/SNN-/Brain-Simulationsplattform eingeführt.

## 2) Input Surface (erlaubt / verboten)

Erlaubte Inputs (`BlueBrainKuramotoModulationInput`):

- Selection-Posture,
- Runtime-Posture,
- ausgewählte Context-Referenzen,
- ausgewählte Evidence-Referenzen,
- Memory-Caveats als Inputsignal,
- kleine Phase/Coupling-Nodes (`phase_permille`, `coupling_permille`) für deterministische Kuramoto-nahe Kohärenzschätzung.

Explizit verboten (nicht Teil des Input-Surfaces):

- raw compute internals,
- internal/expert-only hook state,
- direkte Tool/Action-Mutationszustände,
- direkte Memory-Mutationszustände.

## 3) Output Surface (erlaubt / verboten)

Erlaubte Outputs (`BlueBrainKuramotoModulationResult`):

- synchrony diagnostic,
- coherence summary (`coherence_permille`),
- selection modulation hint (optional),
- runtime caveat modulation signal (optional),
- caveated/insufficient Ergebnisstatus,
- expliziter boundary guard mit ausschließlich `false`-Freigaben.

Explizit verboten (nicht erzeugbar):

- direkte Action-Ausführung,
- direkter Memory-Commit,
- direkte Compute-Invocation,
- direkte Safety-Entscheidung oder Safety-Override,
- Policy-Result/Policy-Entscheidung.

## 4) Rückbindung an BB4/BB2

Die Rückbindung ist absichtlich begrenzt:

- BB4: Selection wird nur über `selection_hint` informiert.
- BB2: Runtime wird nur über `runtime_modulation`/Caveat informiert.

Der Pfad ist **advisory only**:

- keine direkte Auswahlentscheidung,
- keine direkte Transition-Klassen-Umschreibung,
- keine Seiteneffekte auf Action/Tool/Memory/Compute/Safety/Policy.

## 5) Determinismus und Sicherheitsgrenzen

- Phase-/Coupling-Verarbeitung läuft auf Integer-Permille-Skala.
- Inputs werden kanonisiert (Sortierung/Deduplizierung) für reproduzierbares Verhalten.
- Unzureichende Dynamics-Inputs führen zu explizitem `InsufficientInput` + Caveat.
- Boundary Guard kodiert die Nicht-Erlaubnis direkter Hochmachtpfade.

## 6) Verhältnis zu BB10 Prompt 1 Candidate Map

Prompt 1 bleibt gültig:

- Kuramoto bleibt der bevorzugte leichte Integrationskandidat.
- Die jetzt eingeführte Implementierung bleibt strikt unterhalb von Execution/Commit/Safety/Policy-Autorität.
- Hodgkin-Huxley bleibt in dieser Linie weiterhin nicht als produktiver Runtime-Pfad umgesetzt.

## 7) Bridge-Phase-Pfad Status (BB11 Prompt 3)

- `domains/ucf-bluebrain-bridge::BrainStimulusEncoder::attach_phase(...)` ist **nicht** Teil
  der operativen Kuramoto-/Neuromod-Linie.
- Operative Callsites in `runtime/ucf-policy` und `runtime/ucf-runtime` verwenden weiterhin
  ausschließlich `encode_to_spikes(...)` (phase-empty spike batches).
- Damit bleibt `attach_phase(...)` derzeit explizit **test-only/deferred**:
  nutzbar für Tests/Experimente, aber ohne produktive Autorität und ohne End-to-end-Claim.
- Eine spätere Aktivierung ist nur sinnvoll mit einer expliziten kanonischen
  Produktions-Callsite und dokumentierter Wirkung innerhalb der bestehenden
  advisory-only/no-direct-\* Grenzen.
