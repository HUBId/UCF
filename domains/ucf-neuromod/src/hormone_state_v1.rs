#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HormoneStateError {
    OutOfRange { field: &'static str, value: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NormalizedHormoneLevelV1(u16);

impl NormalizedHormoneLevelV1 {
    pub const MIN: u16 = 0;
    pub const MAX: u16 = 10_000;
    pub const NEUTRAL: u16 = 5_000;

    pub const fn neutral() -> Self {
        Self(Self::NEUTRAL)
    }

    pub fn try_new(raw: i64) -> Result<Self, HormoneStateError> {
        if !(i64::from(Self::MIN)..=i64::from(Self::MAX)).contains(&raw) {
            return Err(HormoneStateError::OutOfRange {
                field: "normalized_hormone_level_v1",
                value: raw,
            });
        }

        Ok(Self(raw as u16))
    }

    pub fn new_clamped(raw: i64) -> Self {
        let clamped = raw.clamp(i64::from(Self::MIN), i64::from(Self::MAX));
        Self(clamped as u16)
    }

    pub const fn as_units(self) -> u16 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HormoneStateV1 {
    pub dopamine_like: NormalizedHormoneLevelV1,
    pub serotonin_like: NormalizedHormoneLevelV1,
    pub cortisol_like: NormalizedHormoneLevelV1,
    pub arousal_like: NormalizedHormoneLevelV1,
    pub sleep_pressure: NormalizedHormoneLevelV1,
    pub novelty_pressure: NormalizedHormoneLevelV1,
    pub stability_pressure: NormalizedHormoneLevelV1,
}

impl HormoneStateV1 {
    pub const fn neutral() -> Self {
        Self {
            dopamine_like: NormalizedHormoneLevelV1::neutral(),
            serotonin_like: NormalizedHormoneLevelV1::neutral(),
            cortisol_like: NormalizedHormoneLevelV1::neutral(),
            arousal_like: NormalizedHormoneLevelV1::neutral(),
            sleep_pressure: NormalizedHormoneLevelV1::neutral(),
            novelty_pressure: NormalizedHormoneLevelV1::neutral(),
            stability_pressure: NormalizedHormoneLevelV1::neutral(),
        }
    }

    pub fn new(raw: HormoneStateRawV1) -> Result<Self, HormoneStateError> {
        let state = Self {
            dopamine_like: NormalizedHormoneLevelV1::try_new(raw.dopamine_like)?,
            serotonin_like: NormalizedHormoneLevelV1::try_new(raw.serotonin_like)?,
            cortisol_like: NormalizedHormoneLevelV1::try_new(raw.cortisol_like)?,
            arousal_like: NormalizedHormoneLevelV1::try_new(raw.arousal_like)?,
            sleep_pressure: NormalizedHormoneLevelV1::try_new(raw.sleep_pressure)?,
            novelty_pressure: NormalizedHormoneLevelV1::try_new(raw.novelty_pressure)?,
            stability_pressure: NormalizedHormoneLevelV1::try_new(raw.stability_pressure)?,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn new_clamped(raw: HormoneStateRawV1) -> Self {
        Self {
            dopamine_like: NormalizedHormoneLevelV1::new_clamped(raw.dopamine_like),
            serotonin_like: NormalizedHormoneLevelV1::new_clamped(raw.serotonin_like),
            cortisol_like: NormalizedHormoneLevelV1::new_clamped(raw.cortisol_like),
            arousal_like: NormalizedHormoneLevelV1::new_clamped(raw.arousal_like),
            sleep_pressure: NormalizedHormoneLevelV1::new_clamped(raw.sleep_pressure),
            novelty_pressure: NormalizedHormoneLevelV1::new_clamped(raw.novelty_pressure),
            stability_pressure: NormalizedHormoneLevelV1::new_clamped(raw.stability_pressure),
        }
    }

    pub fn validate(&self) -> Result<(), HormoneStateError> {
        for (name, value) in self.fields() {
            NormalizedHormoneLevelV1::try_new(i64::from(value.as_units())).map_err(|_| {
                HormoneStateError::OutOfRange {
                    field: name,
                    value: i64::from(value.as_units()),
                }
            })?;
        }
        Ok(())
    }

    pub const fn policy_mutating() -> bool {
        false
    }

    pub const fn gateway_visible() -> bool {
        false
    }

    pub const fn identity_authority() -> bool {
        false
    }

    pub const fn evidence_archive_authority() -> bool {
        false
    }

    fn fields(&self) -> [(&'static str, NormalizedHormoneLevelV1); 7] {
        [
            ("dopamine_like", self.dopamine_like),
            ("serotonin_like", self.serotonin_like),
            ("cortisol_like", self.cortisol_like),
            ("arousal_like", self.arousal_like),
            ("sleep_pressure", self.sleep_pressure),
            ("novelty_pressure", self.novelty_pressure),
            ("stability_pressure", self.stability_pressure),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HormoneStateRawV1 {
    pub dopamine_like: i64,
    pub serotonin_like: i64,
    pub cortisol_like: i64,
    pub arousal_like: i64,
    pub sleep_pressure: i64,
    pub novelty_pressure: i64,
    pub stability_pressure: i64,
}
