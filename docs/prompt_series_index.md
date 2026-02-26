# Prompt Series Index (1–128)

This index is generated from merged prompt-series PR metadata (`#248..#374`) plus this wrap-up prompt (`128`). It is deterministic: prompt IDs map monotonically to the series sequence.

## Table of contents
- [UCF core plumbing (1–37)](#ucf-core-plumbing-137)
- [Real compute onboarding v0 (38–67)](#real-compute-onboarding-v0-3867)
- [LFM + Governance + Replay + Hardening (68–90)](#lfm-+-governance-+-replay-+-hardening-6890)
- [Operator/Release/rc1 (91–120)](#operator/release/rc1-91120)
- [v1.1 plan & prompts (121–127)](#v1.1-plan-&-prompts-121127)
- [Wrap-up (128)](#wrap-up-128)

## UCF core plumbing (1–37)

| ID | Title | One-line intent | Primary modules touched | Key acceptance criteria | Dependencies |
|---:|---|---|---|---|---|
| 1 | create core crate and types modules | Create core crate and types modules | core/ucf-core, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | — |
| 2 | create controlframe schema v1 | Create controlframe schema v1 | domains, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 1 |
| 3 | implement pbm and gem minimal policy gate | Implement pbm and gem minimal policy gate | ucf-policy, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 2 |
| 4 | implement experience stream store ess v1 | Implement experience stream store ess v1 | domains, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 3 |
| 5 | implement runtime orchestrator v0 | Implement runtime orchestrator v0 | ucf-runtime, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 4 |
| 6 | implement policy ruleset v0 with intent support | Implement policy ruleset v0 with intent support | domains, ucf-policy, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 5 |
| 7 | add decision ledger and correlation indexing | Add decision ledger and correlation indexing | domains | merged series step recorded; test/bench or verification assets updated | 6 |
| 8 | implement brainbus with snn/onn support | Implement brainbus with snn/onn support | domains, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 7 |
| 9 | add brainstimulus encoding in ucf bluebrain bridge | Add brainstimulus encoding in ucf bluebrain bridge | domains, ucf-policy, ucf-runtime, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 8 |
| 10 | add bluebrain bridge with spike encoding | Add bluebrain bridge with spike encoding | ucf-runtime | merged series step recorded; test/bench or verification assets updated | 9 |
| 11 | create neuromodulator field and scheduler skeleton | Create neuromodulator field and scheduler skeleton | domains, ucf-runtime, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 10 |
| 12 | add pbm neuromod hooks and decision metadata | Add pbm neuromod hooks and decision metadata | domains, ucf-policy, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 11 |
| 13 | implement onn core with kuramoto phase bus | Implement onn core with kuramoto phase bus | core/crates, Cargo.lock, domains | merged series step recorded | 12 |
| 14 | add iit monitor v0 with phiproxy | Add iit monitor v0 with phiproxy | domains, ucf-runtime, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 13 |
| 15 | implement snn core v0 with event spikes | Implement snn core v0 with event spikes | domains, ucf-runtime, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 14 |
| 16 | add biophys compatible parameter modulation api | Add biophys compatible parameter modulation api | domains, ucf-runtime, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 15 |
| 17 | add hormone ode skeleton and hpa model | Add hormone ode skeleton and hpa model | domains, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 16 |
| 18 | add hhneuron and microcircuit stubs | Add hhneuron and microcircuit stubs | domains, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 17 |
| 19 | implement onn/snn bridge v0 architecture | Implement onn/snn bridge v0 architecture | domains, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 18 |
| 20 | add iit monitor module with feedback | Add iit monitor module with feedback | domains, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 19 |
| 21 | add cde/nsr causal loop module | Add cde/nsr causal loop module | domains, ucf-runtime, engine | merged series step recorded; test/bench or verification assets updated | 20 |
| 22 | add ssm state space working memory | Add ssm state space working memory | domains, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 21 |
| 23 | add onn phase bus and snn event bus v0 | Add onn phase bus and snn event bus v0 | domains, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 22 |
| 24 | add archive store and append only log | Add archive store and append only log | core/ucf-core, domains, ucf-runtime, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 23 |
| 25 | implement causal discovery engine skeleton | Implement causal discovery engine skeleton | ucf-runtime, core/crates, domains, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 24 |
| 26 | add neuro symbolic verifier skeleton v0 | Add neuro symbolic verifier skeleton v0 | ucf-runtime, core/crates, domains, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 25 |
| 27 | implement state space model v0 | Implement state space model v0 | ucf-runtime, core/crates, Cargo.lock, domains | merged series step recorded; test/bench or verification assets updated | 26 |
| 28 | implement temporal coherence framework v0 | Implement temporal coherence framework v0 | domains, ucf-runtime, core/crates, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 27 |
| 29 | implement onn/snn bridge with spikebus and filters | Implement onn/snn bridge with spikebus and filters | ucf-runtime, core/ucf-spikes, domains, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 28 |
| 30 | add iit proxy integration monitor v0 | Add iit proxy integration monitor v0 | domains, ucf-runtime, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 29 |
| 31 | add strange loop engine module | Add strange loop engine module | ucf-runtime, domains, Cargo.lock, core/crates | merged series step recorded; test/bench or verification assets updated | 30 |
| 32 | add continuous time state block module | Add continuous time state block module | domains, ucf-runtime, core/crates, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 31 |
| 33 | add onn core with phase bus | Add onn core with phase bus | core/crates, domains, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 32 |
| 34 | add spike event bus with ttfs coding | Add spike event bus with ttfs coding | domains, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 33 |
| 35 | add hodgkin huxley hormone scaffold v0 | Add hodgkin huxley hormone scaffold v0 | domains, ucf-runtime, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 34 |
| 36 | implement active inference and homeostasis | Implement active inference and homeostasis | domains, ucf-runtime, Cargo.lock, Cargo.toml | merged series step recorded; test/bench or verification assets updated | 35 |
| 37 | implement backend traits and cpu stubs | Implement backend traits and cpu stubs | ucf-runtime, domains, ucf-compute, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 36 |

## Real compute onboarding v0 (38–67)

| ID | Title | One-line intent | Primary modules touched | Key acceptance criteria | Dependencies |
|---:|---|---|---|---|---|
| 38 | implement jepa v0 in compute layer | Implement jepa v0 in compute layer | ucf-compute, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 37 |
| 39 | implement real compute onboarding v0 | Implement real compute onboarding v0 | ucf-compute, ucf-runtime, domains | merged series step recorded; test/bench or verification assets updated | 38 |
| 40 | implement candlebackend v0 with dummy weights | Implement candlebackend v0 with dummy weights | ucf-compute, ucf-runtime, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 39 |
| 41 | finalize adapter layer and subtraits | Finalize adapter layer and subtraits | ucf-compute, ucf-runtime, engine | merged series step recorded; test/bench or verification assets updated | 40 |
| 42 | implement risk signal contract and integration | Implement risk signal contract and integration | domains, ucf-compute, ucf-runtime | merged series step recorded; test/bench or verification assets updated | 41 |
| 43 | implement optional hooks for consolidation and geist | Implement optional hooks for consolidation and geist | ucf-runtime, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 42 |
| 44 | implement replay harness for compute | Implement replay harness for compute | ucf-replay, Cargo.lock, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 43 |
| 45 | implement resource limits and backpressure | Implement resource limits and backpressure | ucf-compute, ucf-runtime, docs, domains | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 44 |
| 46 | implement provenance tracking for compute pipeline | Implement provenance tracking for compute pipeline | ucf-compute, ucf-replay, ucf-runtime, docs | merged series step recorded; documentation artifacts updated | 45 |
| 47 | implement capability model v0 features | Implement capability model v0 features | ucf-policy, domains, ucf-runtime, Cargo.lock | merged series step recorded; test/bench or verification assets updated | 46 |
| 48 | prepare sandboxing v1 with isolation interface | Prepare sandboxing v1 with isolation interface | ucf-runtime, domains, Cargo.lock, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 47 |
| 49 | implement wasm sandbox runtime v0 | Implement wasm sandbox runtime v0 | ucf-runtime, Cargo.lock, domains | merged series step recorded | 48 |
| 50 | implement process isolation runtime v0 | Implement process isolation runtime v0 | ucf-runtime, docs | merged series step recorded; documentation artifacts updated | 49 |
| 51 | implement production hardening tests | Implement production hardening tests | fuzz, ucf-compute, ucf-policy, ucf-runtime | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 50 |
| 52 | implement operator readiness features | Implement operator readiness features | docs, ucf-ops, Cargo.lock | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 51 |
| 53 | create performance benchmarks and latency budgets | Create performance benchmarks and latency budgets | ucf-bench, .github, Cargo.lock, bench | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 52 |
| 54 | implement spikebus and coherence metrics pipeline | Implement spikebus and coherence metrics pipeline | ucf-runtime, ucf-policy, docs, domains | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 53 |
| 55 | implement hormone system v0 with ode core | Implement hormone system v0 with ode core | domains, ucf-runtime, Cargo.lock, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 54 |
| 56 | implement hh lite neuron engine | Implement hh lite neuron engine | domains, ucf-runtime, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 55 |
| 57 | implement structural delta channel v0 | Implement structural delta channel v0 | ucf-runtime, domains, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 56 |
| 58 | stabilize nsr/cde risk score v0 | Stabilize nsr/cde risk score v0 | core/crates, domains, Cargo.lock, docs | merged series step recorded; documentation artifacts updated | 57 |
| 59 | implement decision candidate contracts | Implement decision candidate contracts | ucf-policy, ucf-runtime, domains, Cargo.lock | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 58 |
| 60 | implement llm/text backend with deterministic stub | Implement llm/text backend with deterministic stub | domains, ucf-runtime, docs, ucf-compute | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 59 |
| 61 | implement candle/burn inference adapter | Implement candle/burn inference adapter | ucf-compute, docs, ucf-runtime | merged series step recorded; documentation artifacts updated | 60 |
| 62 | implement jepa predictor adapter with surprise pipeline | Implement jepa predictor adapter with surprise pipeline | ucf-compute, Cargo.lock, docs, ucf-replay | merged series step recorded; documentation artifacts updated | 61 |
| 63 | implement sae feature extractor v0 | Implement sae feature extractor v0 | ucf-compute, docs, ucf-replay | merged series step recorded; documentation artifacts updated | 62 |
| 64 | implement ssm selective scan v0 | Implement ssm selective scan v0 | ucf-compute, docs, ucf-replay | merged series step recorded; documentation artifacts updated | 63 |
| 65 | implement real compute onboarding e2e flow | Implement real compute onboarding e2e flow | fixtures, ucf-runtime, Cargo.lock, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 64 |
| 66 | implement unified ai backend pack | Implement unified ai backend pack | ucf-compute, ucf-runtime, domains, ucf-policy | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 65 |
| 67 | integrate lfm as first class compute stage | Integrate lfm as first class compute stage | ucf-compute, ucf-runtime, domains, ucf-policy | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 66 |

## LFM + Governance + Replay + Hardening (68–90)

| ID | Title | One-line intent | Primary modules touched | Key acceptance criteria | Dependencies |
|---:|---|---|---|---|---|
| 68 | implement lfm candle/burn adapter | Implement lfm candle/burn adapter | ucf-compute, docs | merged series step recorded; documentation artifacts updated | 67 |
| 69 | integrate lfm into llm request system | Integrate lfm into llm request system | ucf-runtime, docs, domains, ucf-compute | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 68 |
| 70 | add lfm memory coupling and indexing | Add lfm memory coupling and indexing | domains, ucf-runtime, docs, ucf-policy | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 69 |
| 71 | implement dynamic capability issuance governance | Implement dynamic capability issuance governance | domains, ucf-policy, ucf-runtime, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 70 |
| 72 | implement deterministic replay mode | Implement deterministic replay mode | ucf-ops, ucf-replay, Cargo.lock, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 71 |
| 73 | harden deterministic float encoding and tests | Harden deterministic float encoding and tests | ucf-compute, ucf-replay, core/crates, domains | merged series step recorded; documentation artifacts updated | 72 |
| 74 | implement lnnodelfmkernel with rk2 | Implement lnnodelfmkernel with rk2 | ucf-compute, docs | merged series step recorded; documentation artifacts updated | 73 |
| 75 | implement liquid plasticity v0 with governance | Implement liquid plasticity v0 with governance | ucf-compute, docs | merged series step recorded; documentation artifacts updated | 74 |
| 76 | implement emergency stability guards | Implement emergency stability guards | domains, ucf-compute, ucf-runtime, ucf-policy | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 75 |
| 77 | implement liquid learning loop v0 proposals | Implement liquid learning loop v0 proposals | ucf-runtime, docs, domains | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 76 |
| 78 | implement ess derived telemetry views | Implement ess derived telemetry views | ucf-ops, docs | merged series step recorded; documentation artifacts updated | 77 |
| 79 | release spine v0 with feature flags and artifacts | Release spine v0 with feature flags and artifacts | ucf-compute, ucf-ops, docs, .github | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 78 |
| 80 | implement production readiness gate tests | Implement production readiness gate tests | ucf-ops, .github, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 79 |
| 81 | implement real weights sandbox | Implement real weights sandbox | ucf-compute, ucf-ops, Cargo.lock, docs | merged series step recorded; documentation artifacts updated | 80 |
| 82 | implement deterministic inference probe | Implement deterministic inference probe | ucf-ops, docs | merged series step recorded; documentation artifacts updated | 81 |
| 83 | implement safe gradual rollout mechanism | Implement safe gradual rollout mechanism | ucf-compute, docs | merged series step recorded; documentation artifacts updated | 82 |
| 84 | implement security hardening features | Implement security hardening features | ucf-policy, ucf-runtime, policies, domains | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 83 |
| 85 | create threat model and adversarial harness | Create threat model and adversarial harness | fixtures, ucf-ops, docs, .github | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 84 |
| 86 | implement process isolation v1 features | Implement process isolation v1 features | ucf-compute, Cargo.lock, docs | merged series step recorded; documentation artifacts updated | 85 |
| 87 | implement remote compute v2 skeleton | Implement remote compute v2 skeleton | ucf-compute, docs, domains, policies | merged series step recorded; documentation artifacts updated | 86 |
| 88 | implement offline benchmark harness | Implement offline benchmark harness | ucf-ops, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 87 |
| 89 | implement data governance for ess retention | Implement data governance for ess retention | domains, policies, ucf-ops, Cargo.lock | merged series step recorded; documentation artifacts updated | 88 |
| 90 | implement multi session and run registry features | Implement multi session and run registry features | ucf-ops, docs | merged series step recorded; documentation artifacts updated | 89 |

## Operator/Release/rc1 (91–120)

| ID | Title | One-line intent | Primary modules touched | Key acceptance criteria | Dependencies |
|---:|---|---|---|---|---|
| 91 | create final p8 sign off bundle | Create final p8 sign off bundle | docs, ucf-ops, release, Cargo.lock | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 90 |
| 92 | kickoff compute backends v1 implementation | Kickoff compute backends v1 implementation | ucf-compute, docs | merged series step recorded; documentation artifacts updated | 91 |
| 93 | implement unified candle safetensors loader | Implement unified candle safetensors loader | ucf-compute, docs, Cargo.lock | merged series step recorded; documentation artifacts updated | 92 |
| 94 | implement jepa/sae/ssm real backends | Implement jepa/sae/ssm real backends | docs, ucf-compute | merged series step recorded; documentation artifacts updated | 93 |
| 95 | implement candle llm backend v1 | Implement candle llm backend v1 | ucf-compute, docs, ucf-runtime | merged series step recorded; documentation artifacts updated | 94 |
| 96 | implement burn backend parity v1 | Implement burn backend parity v1 | ucf-compute, docs, Cargo.lock | merged series step recorded; documentation artifacts updated | 95 |
| 97 | implement fixed point arithmetic for safety scalars | Implement fixed point arithmetic for safety scalars | docs, ucf-compute, ucf-ops, ucf-policy | merged series step recorded; documentation artifacts updated | 96 |
| 98 | implement compute token economics | Implement compute token economics | domains, ucf-runtime, docs | merged series step recorded; documentation artifacts updated | 97 |
| 99 | implement compute contracts schema validation | Implement compute contracts schema validation | ucf-compute, ucf-runtime, ucf-policy, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 98 |
| 100 | add ebm as reasoning layer | Add ebm as reasoning layer | ucf-runtime, domains, ucf-compute, Cargo.lock | merged series step recorded; documentation artifacts updated | 99 |
| 101 | implement candleebmreasonerv1 with variant search | Implement candleebmreasonerv1 with variant search | domains, ucf-compute, ucf-runtime, docs | merged series step recorded; documentation artifacts updated | 100 |
| 102 | implement ebm coupling with fep and governor | Implement ebm coupling with fep and governor | ucf-runtime, docs, domains, ucf-policy | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 101 |
| 103 | add constraint library to ebm | Add constraint library to ebm | domains, docs, policies, ucf-policy | merged series step recorded; documentation artifacts updated | 102 |
| 104 | implement energy tagged experiences for ebm | Implement energy tagged experiences for ebm | domains, docs | merged series step recorded; documentation artifacts updated | 103 |
| 105 | implement offline dataset export for ebm training | Implement offline dataset export for ebm training | ucf-ebm-train, ucf-ops, Cargo.lock, docs | merged series step recorded; documentation artifacts updated | 104 |
| 106 | finalize ebm integration v1 enhancements | Finalize ebm integration v1 enhancements | ucf-ops, docs, fixtures, policies | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 105 |
| 107 | implement lfm ode core with deterministic integrator | Implement lfm ode core with deterministic integrator | ucf-compute, docs | merged series step recorded; documentation artifacts updated | 106 |
| 108 | implement hash locked weights for lfm ode | Implement hash locked weights for lfm ode | fixtures, ucf-compute, docs | merged series step recorded; documentation artifacts updated | 107 |
| 109 | implement neuro symbolic risk engine v1 | Implement neuro symbolic risk engine v1 | ucf-runtime, policies, ucf-compute, ucf-ops | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 108 |
| 110 | stabilize versioned policy packs | Stabilize versioned policy packs | policies, ucf-policy, domains, ucf-ops | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 109 |
| 111 | implement two phase commit for toolintent | Implement two phase commit for toolintent | ucf-policy, domains, docs, ucf-runtime | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 110 |
| 112 | implement minimal causal graph v1 | Implement minimal causal graph v1 | ucf-ops, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 111 |
| 113 | implement iit/tcf monitor v1 features | Implement iit/tcf monitor v1 features | ucf-runtime, domains, ucf-policy, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 112 |
| 114 | implement biophys_runtime v1 features | Implement biophys_runtime v1 features | domains, core/crates, biophys_compartmental_solver, docs | merged series step recorded; documentation artifacts updated | 113 |
| 115 | implement stochasticity lock for determinism | Implement stochasticity lock for determinism | ucf-compute, ucf-ops, ucf-policy, policies | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 114 |
| 116 | implement compute sandbox v1 features | Implement compute sandbox v1 features | ucf-runtime, Cargo.lock, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 115 |
| 117 | finalize production readiness for v1.0 rc1 | Finalize production readiness for v1.0 rc1 | ucf-runtime, docs, ucf-compute, ucf-policy | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 116 |
| 118 | perform post rc1 codebase cleanup | Perform post rc1 codebase cleanup | docs, ucf-ops, core/crates, ucf-compute | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 117 |
| 119 | create v1.1 roadmap and documentation | Create v1.1 roadmap and documentation | docs | merged series step recorded; documentation artifacts updated | 118 |
| 120 | implement weights lifecycle v1.1 | Implement weights lifecycle v1.1 | docs, ucf-ops, ucf-compute, models | merged series step recorded; documentation artifacts updated | 119 |

## v1.1 plan & prompts (121–127)

| ID | Title | One-line intent | Primary modules touched | Key acceptance criteria | Dependencies |
|---:|---|---|---|---|---|
| 121 | add vl jepa slot scaffolding | Add vl jepa slot scaffolding | ucf-compute, ucf-ops, docs | merged series step recorded; documentation artifacts updated | 120 |
| 122 | prepare vl jepa for rollout v1.1 | Prepare vl jepa for rollout v1.1 | ucf-ops, ucf-compute, docs | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 121 |
| 123 | implement sae v1.1 architecture and features | Implement sae v1.1 architecture and features | ucf-compute, fixtures, docs | merged series step recorded; documentation artifacts updated | 122 |
| 124 | implement sae v1.1 architecture and features qt2okt | Implement sae v1.1 architecture and features qt2okt | ucf-compute, docs, ucf-ops | merged series step recorded; documentation artifacts updated | 123 |
| 125 | add optimized ssm kernels v1.1 | Add optimized ssm kernels v1.1 | ucf-compute, docs | merged series step recorded; documentation artifacts updated | 124 |
| 126 | add optional gpu lane support | Add optional gpu lane support | domains, ucf-runtime, ucf-backends-gpu, .github | merged series step recorded; documentation artifacts updated | 125 |
| 127 | extend readiness gate for v1.1 | Extend readiness gate for v1.1 | docs, release, ucf-ops | merged series step recorded; documentation artifacts updated | 126 |

## Wrap-up (128)

| ID | Title | One-line intent | Primary modules touched | Key acceptance criteria | Dependencies |
|---:|---|---|---|---|---|
| 128 | Wrap-Up: Prompt Series Index, Module Map, and Next Prompt Generator Rulebook | Wrap-Up: Prompt Series Index, Module Map, and Next Prompt Generator Rulebook | docs, release, README.md | merged series step recorded; documentation artifacts updated; test/bench or verification assets updated | 127 |
