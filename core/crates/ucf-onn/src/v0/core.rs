use super::errors::OnnError;
use super::types::{OscId, PhaseDeg};

#[derive(Clone, Debug)]
pub struct OnnCore {
    cycle_hz: f32,
    k_coupling: f32,
    osc: Vec<(OscId, PhaseDeg)>,
}

impl OnnCore {
    pub fn new(cycle_hz: f32, k_coupling: f32) -> Self {
        Self {
            cycle_hz,
            k_coupling,
            osc: Vec::new(),
        }
    }

    pub fn try_new(cycle_hz: f32, k_coupling: f32) -> Result<Self, OnnError> {
        if !cycle_hz.is_finite() || cycle_hz <= 0.0 {
            return Err(OnnError::InvalidCycleHz);
        }
        Ok(Self::new(cycle_hz, k_coupling))
    }

    pub fn register(&mut self, id: OscId, initial: PhaseDeg) {
        let norm = initial.norm();
        if let Some((_, phase)) = self.osc.iter_mut().find(|(osc_id, _)| *osc_id == id) {
            *phase = norm;
            return;
        }
        self.osc.push((id, norm));
    }

    pub fn set_phase(&mut self, id: OscId, phase: PhaseDeg) {
        if let Some((_, current)) = self.osc.iter_mut().find(|(osc_id, _)| *osc_id == id) {
            *current = phase.norm();
        }
    }

    pub fn phase(&self, id: OscId) -> Option<PhaseDeg> {
        self.osc
            .iter()
            .find(|(osc_id, _)| *osc_id == id)
            .map(|(_, phase)| *phase)
    }

    pub fn step_ms(&mut self, dt_ms: u64) {
        if self.osc.is_empty() || dt_ms == 0 {
            return;
        }

        let dt_sec = dt_ms as f32 / 1_000.0;
        let base_advance_deg = 360.0 * self.cycle_hz * dt_sec;
        let old_phases: Vec<f32> = self.osc.iter().map(|(_, p)| p.norm().0).collect();
        let n = old_phases.len() as f32;

        for (idx, (_, phase)) in self.osc.iter_mut().enumerate() {
            let theta_i = old_phases[idx].to_radians();
            let coupling_term = old_phases
                .iter()
                .map(|theta_j| (theta_j.to_radians() - theta_i).sin())
                .sum::<f32>()
                / n;
            let coupling_deg =
                self.k_coupling * coupling_term * dt_sec * 180.0 / core::f32::consts::PI;
            *phase = PhaseDeg(old_phases[idx] + base_advance_deg + coupling_deg).norm();
        }
    }

    pub fn coherence_pair(&self, a: OscId, b: OscId) -> Option<f32> {
        let phase_a = self.phase(a)?;
        let phase_b = self.phase(b)?;
        let diff = PhaseDeg::diff(phase_a, phase_b).abs();
        let coherence = (1.0 - (diff / 180.0)).clamp(0.0, 1.0);
        Some(coherence)
    }
}

#[cfg(test)]
mod tests {
    use super::OnnCore;
    use crate::v0::types::PhaseDeg;

    #[test]
    fn step_ms_advances_deterministically_without_coupling() {
        let mut core = OnnCore::new(2.0, 0.0);
        core.register(1, PhaseDeg(10.0));
        core.register(2, PhaseDeg(20.0));

        core.step_ms(250);

        let p1 = core.phase(1).expect("oscillator 1 must exist");
        let p2 = core.phase(2).expect("oscillator 2 must exist");
        assert!((p1.0 - 190.0).abs() < 1e-4);
        assert!((p2.0 - 200.0).abs() < 1e-4);
    }

    #[test]
    fn coherence_pair_matches_phase_relationship() {
        let mut core = OnnCore::new(1.0, 0.0);
        core.register(7, PhaseDeg(90.0));
        core.register(9, PhaseDeg(90.0));

        let equal = core.coherence_pair(7, 9).expect("pair must exist");
        assert!((equal - 1.0).abs() < 1e-6);

        core.set_phase(9, PhaseDeg(270.0));
        let opposite = core.coherence_pair(7, 9).expect("pair must exist");
        assert!(opposite < 1e-4);
    }

    #[test]
    fn invalid_cycle_hz_is_rejected() {
        let err = OnnCore::try_new(0.0, 1.0).expect_err("must reject non-positive cycle");
        assert_eq!(err.to_string(), "cycle_hz must be finite and > 0");
    }
}
