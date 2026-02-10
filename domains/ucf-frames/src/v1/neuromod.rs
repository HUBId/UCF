#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuromodulatorSnapshot {
    pub dopamine: f32,
    pub serotonin: f32,
    pub norepinephrine: f32,
    pub acetylcholine: f32,
    pub oxytocin: f32,
    pub endorphin: f32,
    pub stress: f32,
}

impl NeuromodulatorSnapshot {
    pub fn baseline() -> Self {
        Self {
            dopamine: 0.5,
            serotonin: 0.5,
            norepinephrine: 0.5,
            acetylcholine: 0.5,
            oxytocin: 0.5,
            endorphin: 0.5,
            stress: 0.5,
        }
    }
}
