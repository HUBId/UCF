#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IitConfig {
    pub enabled: bool,
    pub phi_gain: f32,
    pub phi_floor: f32,
    pub phi_ceiling: f32,
}

impl Default for IitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            phi_gain: 1.0,
            phi_floor: 0.0,
            phi_ceiling: 1.0,
        }
    }
}
