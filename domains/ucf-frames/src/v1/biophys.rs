#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiophysHhParams {
    pub g_na: f32,
    pub g_k: f32,
    pub g_l: f32,
    pub threshold_shift_mv: f32,
    pub max_firing_hz: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiophysFrame {
    pub now_ms: u64,
    pub field: [f32; 7],
    pub hh_params: BiophysHhParams,
}
