# Serie SC1 Prompt 1: Blue-Brain system audit and consolidation map

Status: systemweiter Audit-/Consolidation-Pass nach BR6. Diese Datei ist eine **Current-Authority-Konsolidierung** für den aktuellen Blue-Brain-Stand; sie baut keine neue Region, keine neue Modellplattform, keine Planner-/Agentenlogik und keine Compute-Core-Arbeit.

## 1) Workspace- und Repro-Basis

Repo-Basis geprüft:

- `git status --short` war zu Beginn leer.
- `AGENTS.md` ist die repo-weite Arbeitsanweisung und fordert offline-first, deterministische Änderungen, canonical checks und keine behavior-changing Änderungen ohne Spec-Intent.
- Auffindbare Audit-Artefakte liegen unter `out/blue_brain_audit_baseline_2026-05-02/`, `out/blue_brain_audit_baseline_2026-05-04/`, `out/blue_brain_audit_baseline_2026-05-08/`, `out/docs_lint_report.json` und `out/gate_report.json`.
- `out/docs_lint_report.json` meldet `ok: true`; `out/gate_report.json` meldet `status: PASS`.

Bewertung: Die Reproduzierbarkeitsbasis reicht für eine belastbare Konsolidierungsentscheidung aus, aber die versionierten Blue-Brain-Baselines sind historisch. Für aktuelle post-BR6-Aussagen wurde der frische Baseline-Refresh in SC1 Prompt 2 umgesetzt; `docs/blue_brain_sc1_prompt2_post_br6_repro_baseline_refresh_v1.md` und `out/blue_brain_audit_baseline_2026-05-08/` sind die aktuelle Konsolidierungsevidenz.

## 2) Autoritätskette und Referenzoberfläche

Audit-Befund:

- `docs/README.md` und `docs/blue_brain_authority_chain_status_map.md` verwiesen vor diesem Pass noch primär auf die BB29-Drei-Regionen-Endlage.
- Nach BR6 ist diese Lesart stale: operativ sind jetzt sechs echte anatomische Regionen plus IR1/MD2-Konsolidierung maßgeblich.
- BB25/BB27/BB29 bleiben historische Snapshots; sie dürfen nicht als gleichrangige aktuelle Autorität gelesen werden.
- BR1-BR6 sind regionale Abschlusslinien; IR1 ist die relationale Abschlusslinie; MD2 ist die Modellvertiefungs-Maintenance-Linie.

Konsolidierungsfix in diesem Pass:

- `docs/README.md` wurde auf post-BR6 authority entrypoints umgestellt.
- `docs/blue_brain_authority_chain_status_map.md` wurde auf BR6/IR1/MD2/System-Audit als Current Authority gehoben; BB29 ist jetzt historische Snapshot-Linie.

Restbewertung: doc authority ist nach dem Fix **clean enough for maintenance**; der zuvor offene Baseline-Refresh ist durch SC1 Prompt 2 als supporting evidence geschlossen oder deutlich reduziert.

## 3) Regionenbestand: operative Status Map

| Region | Current role | Surface/contract status | Diagnostics | Model mode | Runtime/Selection/Reference Wirkung | Audit-Einstufung |
| --- | --- | --- | --- | --- | --- | --- |
| Hippocampus | context/reference/episode indexing | input/state/output/reference surface; bounded contract signals; no direct action/compute/retry/memory path | advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only, non-canonical/internal-only getrennt | abstract functional current mode | bounded advisory/reference read for runtime/selection/reference | stable maintenance-hardened |
| Amygdala | threat/salience/caveat mediation | bounded salience/caveat surface; advisory/caveated contract signals | same cross-line diagnostic classes, amygdala-specific map | bounded Kuramoto-like current mode for model language; contract remains bounded read | advisory/caveat influence; no action/retry/safety authority | stable maintenance-hardened with model-language caveat |
| Thalamus | relay/gating/routing | bounded input/state/output/reference surface; relay/routing diagnostic contracts | advisory-only/caveated/deferred/blocked/insufficient/diagnostic-only separated | abstract functional current mode | bounded routing/relay diagnostic read; no direct execution trigger | stable maintenance-hardened |
| Basal Ganglia | action-channel suppression / selection mediation | bounded action-gating mediation surface, reference-bounded output; direct action control blocked | advisory/caveated/deferred/blocked/insufficient/diagnostic-only separated | abstract functional current mode | selection-mediated advisory only; action gating is not action execution | usable with caveats |
| Cerebellum | prediction/timing/correction | bounded timing/correction reference/advisory surface | advisory/caveated/deferred/blocked/insufficient/diagnostic-only separated | abstract functional current mode | execution-interface diagnostic support only; no execution trigger | usable with caveats |
| Hypothalamus | drive/homeostasis/urgency/state-pressure | BR6 input/state/output/reference surface; bounded contract signal; non-canonical/internal-only paths separated | hypothalamus advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only states | abstract functional current mode | bounded urgency/state-pressure/regulation read; no action, retry, memory, compute, safety authority | stable maintenance-hardened, newest region caveat |

Non-operative anatomical names: Prefrontal Cortex, Anterior Cingulate Cortex and Insula remain canonical map entries or deferred/model-decision references, not current operational integrated regions.

## 4) Inter-region architecture and relations

IR1 has exactly the bounded relation surface currently needed. It is a relation map, not a planner/orchestrator.

Canonical active or consumable relation classes:

- direct bounded advisory relation: bounded read only.
- reference-mediated relation: reference/context read only.
- selection-mediated relation: selection contract read only; no action execution.
- execution-interface-mediated relation: diagnostic/read interface only; no execution trigger.
- caveated relation: visible caveat, not positive promotion.
- deferred/not-yet-active and blocked relations: fail-closed.
- non-canonical/internal-only relation paths: no consumer authority.

Concrete relation status observed in code/docs:

| Pair | Relation class | Current status |
| --- | --- | --- |
| Hippocampus ↔ Amygdala | caveated | active caveated bounded relation |
| Hippocampus ↔ Thalamus | reference-mediated | active reference-mediated relation |
| Hippocampus ↔ Basal Ganglia | blocked | not active |
| Hippocampus ↔ Cerebellum | reference-mediated | active reference-mediated relation |
| Amygdala ↔ Thalamus | direct bounded advisory | active and also the only MD1/MD2 deepened pair |
| Amygdala ↔ Basal Ganglia | selection-mediated | active selection-mediated relation, caveat-sensitive |
| Amygdala ↔ Cerebellum | deferred/not-yet-active | not active |
| Thalamus ↔ Basal Ganglia | selection-mediated | active selection-mediated relation |
| Thalamus ↔ Cerebellum | direct bounded advisory | active bounded advisory relation |
| Basal Ganglia ↔ Cerebellum | execution-interface-mediated | active diagnostic/read interface; no execution trigger |
| Hippocampus ↔ Hypothalamus | reference-mediated | active reference-mediated relation |
| Amygdala ↔ Hypothalamus | caveated | active caveated bounded relation |
| Thalamus ↔ Hypothalamus | direct bounded advisory | active bounded advisory relation |
| Basal Ganglia ↔ Hypothalamus | selection-mediated | active selection-mediated relation |
| Cerebellum ↔ Hypothalamus | deferred/not-yet-active | not active |

Relational risk: The relation taxonomy is rich enough that wording can drift into platform language. Keep IR1 described as a fixed bounded architecture map, not as a generalized inter-region substrate.

## 5) Model-deepening audit

Current model boundary:

- Abstract functional current mode: Hippocampus, Thalamus, Basal Ganglia, Cerebellum, Hypothalamus.
- Bounded Kuramoto-like current/deepened language: Amygdala; exactly one deepened relation, `Amygdala ↔ Thalamus`.
- HH simulation-only/diagnostic-only and later selective HH remain non-operative/deferred.
- Insula remains deferred; Prefrontal Cortex remains later selective HH; Anterior Cingulate Cortex remains a map/model-language entry, not an integrated operational region.

MD1/MD2 audit result:

- Model state is not contract state.
- Diagnostic model output is not operative authority.
- `diagnostic-only`, `deferred`, `blocked`, `insufficient` and `non-canonical/internal-only` carry no consumer support.
- No second model-deepening candidate is open.

Model-boundary drift remaining: medium-low. The main risk is wording around Amygdala/Basal-Ganglia or HH-later candidates being misread as active model expansion.

## 6) Guard-rails and scope boundaries

System-wide guard line remains intact:

- no direct action trigger.
- no direct execution trigger.
- no direct retry trigger.
- no direct memory commit.
- no direct compute invocation.
- no safety override.
- no implicit policy/governance authority.
- no implicit platform formation.
- no implicit new region series.
- no implicit global model platform.

Guard weakness remaining: low in code/tests for explicit no-direct flags, medium in docs because older handoff files can still be found and quoted without the current authority map.

## 7) Cross-line semantic audit

Terminology status:

- `advisory-only`: bounded positive read; never a trigger.
- `caveated`: usable only with caveat; cannot promote to strong support.
- `deferred`: not active now; distinct from blocked.
- `blocked`: fail-closed unavailable/forbidden path.
- `insufficient`: evidence/input basis too weak; no support.
- `diagnostic-only`: visible diagnostic state; not advisory support.
- `reference-only`: read-only reference path; no mutation or action.
- `current model mode`: descriptive model mode; not contract authority.
- `non-canonical/internal-only`: not consumer-readable as operative behavior.

Semantic drift remaining: medium-low. The terms are mostly harmonized, but relation classes, region status classes and model classes are still spread across many files. A compact glossary/checklist should be the next consolidation lever.

## 8) Tests, docs and code consistency

Evidence points:

- Code surfaces for region, relation, model-deepening and guard states are concentrated in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.
- Current generated reports under `out/docs_lint_report.json` and `out/gate_report.json` pass.
- Historical versioned baselines under `out/blue_brain_audit_baseline_2026-05-02/` and `out/blue_brain_audit_baseline_2026-05-04/` remain useful as historical evidence, while `out/blue_brain_audit_baseline_2026-05-08/` is the current post-BR6 consolidation baseline.
- Doku volume is high; authority entrypoints are therefore essential to avoid duplicate-truth drift.

Assessment:

- Region surfaces: maintenance-ready with caveats.
- Relation architecture: maintenance-ready with relation-wording caveat.
- Model deepening: maintenance-ready; no second candidate.
- Guard rails: maintenance-ready.
- Repro baseline: refreshed in SC1 Prompt 2 for the post-BR6 authority line.

## 9) Systemweite consolidation map

| State | System assessment |
| --- | --- |
| stable and maintenance-ready | no-direct guard line, BR1-BR6 region surfaces, IR1 bounded relation map, MD2 single-deepening maintenance map |
| stable but caveated | Basal Ganglia selection-mediated language; Cerebellum execution-interface diagnostic language; newest Hypothalamus relation lane |
| doc authority clean | yes after this pass, with BR6/IR1/MD2/System-Audit as current authority |
| semantic drift remaining | medium-low: terms are consistent but distributed |
| guard weakness remaining | low in guard flags; medium in stale-document discoverability |
| reproducibility gap | low after SC1 Prompt 2: current root reports pass and `out/blue_brain_audit_baseline_2026-05-08/` provides the post-BR6 versioned Blue-Brain baseline |
| relation ambiguity | medium-low: fixed pair map exists, but relation class richness needs periodic checklist review |
| model-boundary drift | medium-low: one deepened pair is safe; HH/Kuramoto wording needs careful maintenance |
| non-canonical residue | present and acceptable when explicitly marked internal-only/deferred |

## 10) Auditdienliche Fixes in diesem Pass

- Authority chain updated from BB29-only current line to post-BR6 BR6/IR1/MD2/System-Audit current authority.
- README current entrypoints updated to six integrated anatomical regions and current post-BR6 baseline evidence; historical BB29 baseline wording remains explicitly residual.
- Minor README Markdown heading drift fixed for BB27/BB28/BB29 sections.

No runtime behavior, region implementation, model implementation, policy logic, planner/agent behavior or compute core code was changed.

## 11) Highest-leverage next consolidation measures

1. **Repro-/Test-Baseline-Refresh for post-BR6 — implemented by SC1 Prompt 2**  
   Fresh Blue-Brain baseline preserved under `out/blue_brain_audit_baseline_2026-05-08/`; `docs/blue_brain_sc1_prompt2_post_br6_repro_baseline_refresh_v1.md` is the canonical action evidence. Historical 2026-05-02/2026-05-04 baselines remain comparison evidence only.

2. **Cross-line terminology/guard checklist consolidation**  
   Add one compact maintenance checklist that maps advisory-only/caveated/deferred/blocked/insufficient/diagnostic-only/reference-only/current-model-mode/non-canonical to allowed and forbidden consumer reads. This reduces semantic drift without adding behavior.

3. **Relation cleanup/hardening review**  
   Review IR1 relation wording and tests around selection-mediated and execution-interface-mediated relations, especially Basal Ganglia/Cerebellum/Hypothalamus edges, to keep them bounded reads and not accidental orchestration language.

## 12) Abschlussnotiz

Geänderte Dateien in diesem Pass:

- `docs/blue_brain_authority_chain_status_map.md`
- `docs/README.md`
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md`

Gesamtentscheidung: Der Blue-Brain-Stand ist systemweit **maintenance-ready with caveats**. Die sechs integrierten anatomischen Regionen und die bounded inter-region architecture sind ausreichend klar, sofern BR6/IR1/MD2/System-Audit als aktuelle Authority gelesen werden. Die größte SC1-Restschwäche, die post-BR6 Repro-Baseline-Lücke, wurde in SC1 Prompt 2 geschlossen oder deutlich reduziert; verbleibende Restschwächen sind verteilte Terminologie und relationale Wortlautrisiken. Nach diesen 1-3 Konsolidierungsmaßnahmen genügt Maintenance; weiterer Ausbau ist erst nach Konsolidierung vertretbar.
