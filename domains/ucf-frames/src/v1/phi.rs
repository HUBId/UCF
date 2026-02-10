#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhiProxySnapshot {
    pub phi: f32,
    pub coherence_mean: f32,
    pub coherence_min: f32,
    pub n_pairs: u16,
}

impl PhiProxySnapshot {
    pub fn baseline() -> Self {
        Self {
            phi: 0.0,
            coherence_mean: 0.0,
            coherence_min: 0.0,
            n_pairs: 0,
        }
    }
}
