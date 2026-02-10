use ucf_frames::v1::NeuromodulatorSnapshot;

#[derive(Debug, Clone, Copy)]
pub struct NeuromodulatorField {
    cur: NeuromodulatorSnapshot,
}

impl NeuromodulatorField {
    pub fn new_baseline() -> Self {
        Self {
            cur: NeuromodulatorSnapshot::baseline(),
        }
    }

    pub fn snapshot(&self) -> NeuromodulatorSnapshot {
        self.cur
    }

    pub(crate) fn apply_delta(&mut self, d: NeuromodulatorSnapshot) {
        self.cur.dopamine = apply_step(self.cur.dopamine, d.dopamine);
        self.cur.serotonin = apply_step(self.cur.serotonin, d.serotonin);
        self.cur.norepinephrine = apply_step(self.cur.norepinephrine, d.norepinephrine);
        self.cur.acetylcholine = apply_step(self.cur.acetylcholine, d.acetylcholine);
        self.cur.oxytocin = apply_step(self.cur.oxytocin, d.oxytocin);
        self.cur.endorphin = apply_step(self.cur.endorphin, d.endorphin);
        self.cur.stress = apply_step(self.cur.stress, d.stress);
    }
}

fn apply_step(old: f32, delta: f32) -> f32 {
    clamp01(old + (delta - 0.5) * 0.1)
}

pub(crate) fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}
