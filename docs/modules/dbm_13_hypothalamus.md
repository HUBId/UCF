# DBM 13 Hypothalamus

Status: BR6 functional role map aligned. DBM 13 remains a bounded roadmap/module surface for Hypothalamus-style regulation signals; it is not a UCF-BlueBrain production HH integration, not a biological full reconstruction, and not a new action, retry, memory, policy, planner, agent, or compute authority.

## Zweck

DBM 13 represents the Hypothalamus-compatible UCF role: bounded drive-state, homeostasis/regulation, urgency modulation, and context-linked state-pressure. Its current BlueBrain interpretation is `hypothalamus_like_region` in `abstract functional current mode`.

## Inputs

Allowed input semantics are bounded diagnostics and caveats from existing lines only:

- runtime/transition/feedback state pressure,
- selection/priority/deferral urgency caveats,
- context/reference-linked state-pressure caveats,
- bounded regulation or cooldown/stability indicators,
- diagnostic-only microcircuit or simulation references when explicitly scoped.

Inputs must not be interpreted as raw biological hypothalamus state, motivational goals, planner state, policy authority, memory authority, action authority, or compute triggers.

## State

State is limited to bounded regulation posture and urgency/homeostasis caveat posture. Any deeper setpoint, nucleus, hormone, autonomic, endocrine, circadian, feeding, thermoregulation, osmotic, sleep/wake, spiking, or microcircuit state remains deferred unless a later Re-Scope makes it diagnostic-only and non-authoritative.

## Outputs

Allowed outputs are advisory-only:

- bounded drive-state caveat,
- homeostasis/regulation caveat,
- urgency modulation caveat,
- context-linked state-pressure caveat,
- diagnostic-only summary or insufficiency marker.

Outputs never grant direct action selection, execution, retry, memory commit, reference write, safety override, policy/governance result, planner/agent control, queue/orchestration behavior, or compute invocation.

## Regeln

- Deterministic, bounded, reproducible processing only.
- Sort externally visible reason codes or keys before hashed or persisted output.
- Tighten-only diagnostics are allowed; authority expansion is not.
- Advisory-only modulation must remain distinguishable from action selection, execution eligibility, memory persistence, policy decisions, and compute-core behavior.
- Hodgkin-Huxley and other biophysical modes remain simulation-only/diagnostic-only unless a later explicit Re-Scope changes documentation and tests.

## Invarianten

- Hypothalamus = bounded drive/homeostasis/urgency modulation.
- Hippocampus = context/reference/episode/indexing.
- Amygdala = salience/valence/caveat/priority.
- Thalamus = relay/gating/routing.
- Basal Ganglia = action-gating/suppression/channel-selection.
- Cerebellum = prediction/timing/correction/mismatch.
- No semantic duplicate or direct authority edge is created between these roles.

## Tests

BR6 targeted checks should assert that:

- Hypothalamus role tokens remain distinct from all established region role tokens,
- current integration mode stays `abstract functional current mode`,
- HH remains simulation-only/diagnostic-only and non-operative,
- docs do not introduce direct action, execution, retry, memory, policy, planner, agent, safety, or compute authority,
- inter-region relation labels remain mediated/advisory/deferred instead of all-to-all coupling.

## Observability

Observability is limited to bounded diagnostics, caveats, contract markers, and readiness summaries. Observability must not expose raw biological state, unstable internal microcircuit details, motivational control state, or execution authority.

## Microcircuit

Microcircuit paths are deferred/diagnostic-only for the UCF-BlueBrain line. Existing setpoint or L4 assets may remain as isolated DBM/replay or diagnostic surfaces, but BR6 does not make them a productive UCF-BlueBrain integration mode and does not require Hodgkin-Huxley.
