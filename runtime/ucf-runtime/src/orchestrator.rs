use crate::errors::RuntimeError;
use ucf_ess::v1::{ExperienceRecord, ExperienceStore, IdAllocator, InMemoryEss};
use ucf_frames::v1::{ControlFrame, DecisionFrame};
use ucf_policy::{adapter::ActionAdapter, gem::Gem, pbm::Pbm};

pub struct RuntimeOrchestrator {
    pub ess: InMemoryEss,
    pub ids: IdAllocator,
    pub pbm: Pbm,
    pub gem: Gem,
}

impl RuntimeOrchestrator {
    pub fn new() -> Self {
        Self {
            ess: InMemoryEss::new(),
            ids: IdAllocator::new(1),
            pbm: Pbm,
            gem: Gem,
        }
    }

    pub fn ingest_and_process<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        ctrl: ControlFrame,
    ) -> Result<DecisionFrame, RuntimeError> {
        let decision = Pbm::decide(&ctrl);
        self.ingest_with_decision(adapter, ctrl, decision)
    }

    pub fn ingest_with_decision<A: ActionAdapter>(
        &mut self,
        adapter: &mut A,
        ctrl: ControlFrame,
        decision: DecisionFrame,
    ) -> Result<DecisionFrame, RuntimeError> {
        let eid1 = self.ids.next();
        self.ess
            .append(ExperienceRecord::from_control(eid1, ctrl.clone()))?;

        let eid2 = self.ids.next();
        self.ess
            .append(ExperienceRecord::from_decision(eid2, decision.clone()))?;

        if let Err(error) = Gem::execute(adapter, &ctrl, Some(&decision)) {
            let mut note = format!("gem_error:{error}");
            if note.chars().count() > 120 {
                note = note.chars().take(120).collect();
            }

            let eid3 = self.ids.next();
            self.ess.append(ExperienceRecord::note(
                eid3,
                decision.time,
                decision.corr,
                note,
            ))?;

            return Err(error.into());
        }

        if let Some((count, target)) = adapter.take_brain_spike_meta() {
            let mut note = format!("brain_spikes:n={count},dst={target}");
            if note.chars().count() > 120 {
                note = note.chars().take(120).collect();
            }

            let eid3 = self.ids.next();
            self.ess.append(ExperienceRecord::note(
                eid3,
                decision.time,
                decision.corr,
                note,
            ))?;
        }

        Ok(decision)
    }
}

impl Default for RuntimeOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}
