pub type NeuronId = u32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SynKind {
    Excitatory,
    Inhibitory,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Synapse {
    pub pre: NeuronId,
    pub post: NeuronId,
    pub kind: SynKind,
    pub weight: f32,
    pub delay_ms: u16,
    pub stp_u: f32,
    pub stp_x: f32,
}

impl Synapse {
    pub fn effective_weight(&self) -> f32 {
        let magnitude = self.weight.abs();
        match self.kind {
            SynKind::Excitatory => magnitude,
            SynKind::Inhibitory => -magnitude,
        }
    }
}

pub trait PlasticityRule {
    fn on_pre_spike(&mut self, syn: &mut Synapse, t_ms: u64);
    fn on_post_spike(&mut self, syn: &mut Synapse, t_ms: u64);
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoPlasticity;

impl PlasticityRule for NoPlasticity {
    fn on_pre_spike(&mut self, _syn: &mut Synapse, _t_ms: u64) {}

    fn on_post_spike(&mut self, _syn: &mut Synapse, _t_ms: u64) {}
}

pub fn stp_step(syn: &mut Synapse, dt_s: f32) {
    let relax_x = 2.0;
    let relax_u = 3.0;

    syn.stp_x += (1.0 - syn.stp_x) * dt_s * relax_x;
    syn.stp_u += (0.2 - syn.stp_u) * dt_s * relax_u;

    syn.stp_x = syn.stp_x.clamp(0.0, 1.0);
    syn.stp_u = syn.stp_u.clamp(0.0, 1.0);
}
