# Architekturpaket MD3 Prompt 1: second-deepening rescope line

Status: kanonische Re-Scope-Entscheidung nach MD2. Diese Linie prüft repo-basiert, ob **genau ein** zweiter Modellvertiefungskandidat jetzt echten Zusatzhebel hat. Sie implementiert noch keine zweite Modellvertiefung, baut keine globale Dynamikplattform, startet keine Compute-Core-Arbeit und erzeugt keine neue Autoritätsschicht.

Canonical code anchor: `CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_DECISION_CLASS_MAP`, `CANONICAL_BLUE_BRAIN_MD3_SECOND_DEEPENING_DECISION_MAP`, `BLUE_BRAIN_MD3_SECOND_DEEPENING_DECISION`, `BLUE_BRAIN_MD3_PRIORITIZED_SECOND_DEEPENING_PAIR`, and `blue_brain_md3_prioritized_second_deepening_candidate` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Current baseline checked before opening anything

The stable baseline remains:

- Real Compute Stack: final technical exit line, maintenance-only.
- Anatomical regions: Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum, Hypothalamus.
- Inter-region architecture: bounded advisory/reference/selection-mediated relations only.
- First model deepening: exactly `Amygdala ↔ Thalamus` as bounded Kuramoto-like advisory/diagnostic relation, maintenance-hardened by MD2.
- Productive default: Maintenance/Bugfix/Cleanup unless an explicit re-scope opens a narrow next candidate.

MD3 confirms that the first deepening is stable enough to allow a **decision-only** second-candidate opening, because the MD2 model surface is frozen, no-direct-* guards are tested, and candidate evaluation can stay relation-local. This does not modify Runtime, Selection, Reference, Execution, Policy, Memory, or Compute authority.

## 2) Canonical second-deepening decision classes

Only these classes are canonical for MD3:

| Class | Meaning | Boundary |
|---|---|---|
| `ready for second deepening consideration` | Exactly one candidate has enough repo leverage and test/doc support to be prioritized for a later minimal implementation prompt. | Decision-only; no implementation yet. |
| `plausible but not yet` | Candidate has real relation or dynamics relevance, but risk, overlap, or freshness means it must not open now. | No wishlist and no parallel opening. |
| `abstract sufficient` | Existing functional contracts remain the right model; extra dynamics would add weight without leverage. | Not a defect and not a deferred failure. |
| `Kuramoto-like candidate` | Bounded coupling/synchrony/gating/timing modulation is the relevant model form. | Advisory-only diagnostics/hints; no direct trigger. |
| `HH simulation-only/diagnostic-only candidate` | HH is useful only for excitability/spiking/membrane-near simulation or diagnostics. | No productive default path. |
| `later selective HH deepening` | HH-like work might be revisited later for a tightly scoped diagnostic need. | Not active in MD3. |
| `no-second-deepening-now` | Maintenance remains better than opening any second candidate. | Available class, but not the MD3 Prompt 1 outcome. |
| `non-canonical/internal-only model path` | Helper, research, legacy, shortcut, or unlisted path. | No canonical consumer read or authority. |

## 3) Candidate assessment summary

The canonical assessment compares regions, implemented relations, and bounded-dynamics-adjacent surfaces. Scores in code are deterministic ordinal evidence for leverage, risk, clarity, test/doc support, guard risk, and model weight.

| Candidate | Surface | Current MD3 class | Model form | Decision rationale |
|---|---|---|---|---|
| Hippocampus | Region | `abstract sufficient` | abstract sufficient | Reference/context role is already carried by BB17/IR1 contracts; dynamics adds little. |
| Amygdala | Region | `plausible but not yet` | bounded Kuramoto-like candidate | Already participates in first deepening; region-level opening would blur relation-local scope. |
| Thalamus | Region | `abstract sufficient` | abstract sufficient | Relay/routing remains contract-sufficient after `Amygdala ↔ Thalamus` first deepening. |
| Basal Ganglia | Region | `plausible but not yet` | abstract/current bounded contracts | Region-level action-gating proximity has too much Selection/Execution scope risk. |
| Cerebellum | Region | `HH simulation-only/diagnostic-only candidate` | HH simulation-only/diagnostic-only | Timing/prediction could be biologically interesting, but productive HH is too heavy and not justified. |
| Hypothalamus | Region | `abstract sufficient` | abstract sufficient | Drive/homeostasis/urgency pressure remains bounded and diagnostic without model deepening. |
| Hippocampus ↔ Thalamus | Relation | `abstract sufficient` | abstract sufficient | Reference-mediated relation is stable as reference/context routing, not dynamics. |
| Amygdala ↔ Thalamus | Relation | `plausible but not yet` | existing bounded Kuramoto-like baseline | This is the first deepening baseline, so it is not eligible as a second candidate. |
| **Amygdala ↔ Basal Ganglia** | Relation | **`ready for second deepening consideration`** | **bounded Kuramoto-like** | Best incremental leverage: salience/caveat to selection-readiness gating/timing, already implemented as selection-mediated relation, with clear advisory-only boundaries. |
| Hippocampus ↔ Hypothalamus | Relation | `abstract sufficient` | abstract sufficient | Reference/context to drive-state pressure is already bounded reference-mediated. |
| Amygdala ↔ Hypothalamus | Relation | `plausible but not yet` | bounded Kuramoto-like candidate | Salience/drive coupling has leverage but newer BR6 surface and higher urgency/safety wording risk. |
| Thalamus ↔ Hypothalamus | Relation | `plausible but not yet` | bounded Kuramoto-like candidate | Relay/urgency timing is plausible but less semantically direct than Amygdala ↔ Basal Ganglia. |
| Basal Ganglia ↔ Hypothalamus | Relation | `plausible but not yet` | abstract/current bounded contracts | Action-gating plus drive pressure is too close to direct action/urgency escalation for this prompt. |
| BB12 bounded advisory Kuramoto surface | Dynamics surface | `Kuramoto-like candidate` | bounded Kuramoto-like | Reusable vocabulary only; not a separate second candidate or platform. |
| BB10 HH diagnostic surface | Dynamics surface | `HH simulation-only/diagnostic-only candidate` | HH simulation-only/diagnostic-only | Remains diagnostic/simulation-only and non-default. |
| Basal Ganglia ↔ Cerebellum | Relation | `later selective HH deepening` | later selective HH only | Possible future excitability/timing diagnostic, but not now and not productive HH. |

## 4) Kuramoto-like vs HH boundary

MD3 keeps the model forms separated:

- **Kuramoto-like** is appropriate only for bounded coupling, synchrony, gating, and timing modulation where existing advisory/diagnostic contracts already constrain consumption.
- **HH simulation-only/diagnostic-only** is appropriate only where excitability, spiking, or membrane-near behavior has concrete diagnostic value.
- **Later selective HH deepening** is a future explicit re-scope class, not an implementation lane.
- **Abstract sufficient** remains the correct answer for many regions/relations and is not a gap.

There is no implicit HH requirement, no global Kuramoto platform, and no general neural-dynamics model platform.

## 5) MD3 decision

MD3 makes exactly one decision:

> Prioritize exactly one second deepening candidate for a later minimal implementation prompt: **`Amygdala ↔ Basal Ganglia` as a relation-level bounded Kuramoto-like candidate**.

No other candidate is opened. `Amygdala ↔ Thalamus` remains the first maintenance baseline and is not re-opened as the second candidate. All other plausible surfaces remain `plausible but not yet`, `abstract sufficient`, `HH simulation-only/diagnostic-only`, or `later selective HH deepening`.

## 6) Minimal hook direction for the prioritized candidate

If a later prompt implements the prioritized candidate, it must stay this narrow:

- **Kind:** relation-level only, `Amygdala ↔ Basal Ganglia`.
- **Model form:** bounded Kuramoto-like, not HH.
- **Allowed inputs:** salience/caveat advisory signal from Amygdala-side surfaces; bounded selection-readiness diagnostic state from Basal-Ganglia-side surfaces; existing caveat/deferred/blocked/insufficient state markers.
- **Allowed outputs:** advisory-only / diagnostic-only modulation evidence for coupling, synchrony, gating, or timing; bounded reads only through existing Runtime/Selection/Reference semantics.
- **States not touched:** action execution state, retry/queue state, memory persistence state, compute job state, policy/governance state, safety override state, allowed-actions definitions, and first-deepening `Amygdala ↔ Thalamus` state.

## 7) No-direct-* and scope boundaries

MD3 explicitly preserves:

- no direct action trigger,
- no direct execution trigger,
- no direct retry trigger,
- no direct memory commit,
- no direct compute invocation,
- no safety override,
- no Planner/Agent platform,
- no Policy/Governance platform,
- no Retry/Queue/Orchestration platform,
- no Retrieval/Consolidation/Reasoning platform,
- no Compute-Core expansion,
- no allowed-actions expansion,
- no productive HH default,
- no global model platform,
- no implicit multiple deepening.

## 8) MD3 follow-up steps

1. Keep MD2 first-deepening maintenance tests intact while adding only narrow MD3 decision-map tests.
2. In a later prompt, define the minimal `Amygdala ↔ Basal Ganglia` input/state/output surface without implementation drift.
3. Prove again that Selection/Execution boundaries remain non-triggering before any runtime consumer can read the new diagnostic evidence.
4. Keep HH as simulation-only/diagnostic-only or later-selective until a separate explicit re-scope exists.
5. Refresh docs lint/readiness artifacts after any future implementation prompt touches the model surface.
