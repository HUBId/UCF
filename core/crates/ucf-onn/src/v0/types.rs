pub type OscId = u16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseDeg(pub f32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmegaHz(pub f32);

impl PhaseDeg {
    pub fn norm(self) -> Self {
        Self(self.0.rem_euclid(360.0))
    }

    pub fn diff(a: Self, b: Self) -> f32 {
        let d = (a.norm().0 - b.norm().0 + 180.0).rem_euclid(360.0) - 180.0;
        if d <= -180.0 {
            180.0
        } else {
            d
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PhaseDeg;

    #[test]
    fn phase_norm_and_diff_work() {
        assert_eq!(PhaseDeg(-30.0).norm(), PhaseDeg(330.0));
        assert_eq!(PhaseDeg(725.0).norm(), PhaseDeg(5.0));

        assert!((PhaseDeg::diff(PhaseDeg(10.0), PhaseDeg(350.0)) - 20.0).abs() < 1e-4);
        assert!((PhaseDeg::diff(PhaseDeg(350.0), PhaseDeg(10.0)) + 20.0).abs() < 1e-4);
        assert!((PhaseDeg::diff(PhaseDeg(0.0), PhaseDeg(180.0)) - 180.0).abs() < 1e-4);
    }
}
