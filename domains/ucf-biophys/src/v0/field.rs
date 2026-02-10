use crate::v0::{CoherenceState, HpaState};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unit01(pub f32);

impl Unit01 {
    pub fn new(x: f32) -> Self {
        if x.is_nan() {
            return Self(0.0);
        }
        Self(x.clamp(0.0, 1.0))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuromodulatorField {
    pub dopamine: Unit01,
    pub serotonin: Unit01,
    pub oxytocin: Unit01,
    pub endorphin: Unit01,
    pub glutamate: Unit01,
    pub gaba: Unit01,
    pub acetylcholine: Unit01,
    pub noise: Unit01,
    pub stress: Unit01,
}

impl Default for NeuromodulatorField {
    fn default() -> Self {
        Self {
            dopamine: Unit01::new(0.1),
            serotonin: Unit01::new(0.1),
            oxytocin: Unit01::new(0.1),
            endorphin: Unit01::new(0.1),
            glutamate: Unit01::new(0.2),
            gaba: Unit01::new(0.2),
            acetylcholine: Unit01::new(0.1),
            noise: Unit01::new(0.1),
            stress: Unit01::new(0.0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldUpdateCfg {
    pub decay_per_s: f32,
    pub max_delta_per_tick: f32,
}

impl Default for FieldUpdateCfg {
    fn default() -> Self {
        Self {
            decay_per_s: 0.25,
            max_delta_per_tick: 0.2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldEventKind {
    Reward,
    Stress,
    SocialBond,
    PainRelief,
    Excite,
    Inhibit,
    LearnGate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldEvent {
    pub kind: FieldEventKind,
    pub magnitude: Unit01,
}

impl NeuromodulatorField {
    pub fn decay_towards(
        self,
        baseline: NeuromodulatorField,
        dt_s: f32,
        cfg: FieldUpdateCfg,
    ) -> Self {
        let dt = dt_s.max(0.0);
        let alpha = (cfg.decay_per_s.max(0.0) * dt).clamp(0.0, 1.0);

        Self {
            dopamine: step_towards(
                self.dopamine,
                baseline.dopamine,
                alpha,
                cfg.max_delta_per_tick,
            ),
            serotonin: step_towards(
                self.serotonin,
                baseline.serotonin,
                alpha,
                cfg.max_delta_per_tick,
            ),
            oxytocin: step_towards(
                self.oxytocin,
                baseline.oxytocin,
                alpha,
                cfg.max_delta_per_tick,
            ),
            endorphin: step_towards(
                self.endorphin,
                baseline.endorphin,
                alpha,
                cfg.max_delta_per_tick,
            ),
            glutamate: step_towards(
                self.glutamate,
                baseline.glutamate,
                alpha,
                cfg.max_delta_per_tick,
            ),
            gaba: step_towards(self.gaba, baseline.gaba, alpha, cfg.max_delta_per_tick),
            acetylcholine: step_towards(
                self.acetylcholine,
                baseline.acetylcholine,
                alpha,
                cfg.max_delta_per_tick,
            ),
            noise: step_towards(self.noise, baseline.noise, alpha, cfg.max_delta_per_tick),
            stress: step_towards(self.stress, baseline.stress, alpha, cfg.max_delta_per_tick),
        }
    }

    pub fn with_hpa(self, hpa: HpaState) -> NeuromodulatorField {
        let cortisol = Unit01::new(hpa.cortisol).get();
        NeuromodulatorField {
            serotonin: Unit01::new(self.serotonin.get() - 0.2 * cortisol),
            gaba: Unit01::new(self.gaba.get() + 0.2 * cortisol),
            ..self
        }
    }
    pub fn apply_event(self, event: FieldEvent, cfg: FieldUpdateCfg) -> Self {
        let mag = event.magnitude.get();
        let mut next = self;

        // Deterministic event map:
        // Reward -> dopamine += mag, acetylcholine += 0.5*mag
        // Stress -> serotonin -= 0.5*mag, gaba += 0.3*mag
        // SocialBond -> oxytocin += mag
        // PainRelief -> endorphin += mag
        // Excite -> glutamate += mag
        // Inhibit -> gaba += mag
        // LearnGate -> acetylcholine += mag
        match event.kind {
            FieldEventKind::Reward => {
                next.dopamine = rate_limited_add(self.dopamine, mag, cfg.max_delta_per_tick);
                next.acetylcholine =
                    rate_limited_add(self.acetylcholine, 0.5 * mag, cfg.max_delta_per_tick);
            }
            FieldEventKind::Stress => {
                next.serotonin =
                    rate_limited_add(self.serotonin, -0.5 * mag, cfg.max_delta_per_tick);
                next.gaba = rate_limited_add(self.gaba, 0.3 * mag, cfg.max_delta_per_tick);
                next.stress = rate_limited_add(self.stress, 0.3 * mag, cfg.max_delta_per_tick);
            }
            FieldEventKind::SocialBond => {
                next.oxytocin = rate_limited_add(self.oxytocin, mag, cfg.max_delta_per_tick);
            }
            FieldEventKind::PainRelief => {
                next.endorphin = rate_limited_add(self.endorphin, mag, cfg.max_delta_per_tick);
            }
            FieldEventKind::Excite => {
                next.glutamate = rate_limited_add(self.glutamate, mag, cfg.max_delta_per_tick);
            }
            FieldEventKind::Inhibit => {
                next.gaba = rate_limited_add(self.gaba, mag, cfg.max_delta_per_tick);
            }
            FieldEventKind::LearnGate => {
                next.acetylcholine =
                    rate_limited_add(self.acetylcholine, mag, cfg.max_delta_per_tick);
            }
        }

        next
    }
}

fn rate_limited_add(current: Unit01, delta: f32, max_delta_per_tick: f32) -> Unit01 {
    let limit = max_delta_per_tick.max(0.0);
    Unit01::new(current.get() + delta.clamp(-limit, limit))
}

fn step_towards(current: Unit01, target: Unit01, alpha: f32, max_delta_per_tick: f32) -> Unit01 {
    let desired_delta = (target.get() - current.get()) * alpha;
    rate_limited_add(current, desired_delta, max_delta_per_tick)
}

pub fn apply_coherence_feedback(
    field: &mut NeuromodulatorField,
    integration: f32,
    state: CoherenceState,
) {
    let frag_factor = (1.0 - integration).clamp(0.0, 1.0);
    match state {
        CoherenceState::Stable => {
            field.noise = Unit01::new(field.noise.get() - 0.01);
            field.serotonin = Unit01::new(field.serotonin.get() + 0.01);
        }
        CoherenceState::Drifting => {
            field.noise = Unit01::new(field.noise.get() + 0.02 * frag_factor);
            field.dopamine = Unit01::new(field.dopamine.get() - 0.01 * frag_factor);
        }
        CoherenceState::Fragmenting => {
            field.noise = Unit01::new(field.noise.get() + 0.05 * frag_factor);
            field.dopamine = Unit01::new(field.dopamine.get() - 0.03 * frag_factor);
            field.stress = Unit01::new(field.stress.get() + 0.03 * frag_factor);
        }
    }
}
