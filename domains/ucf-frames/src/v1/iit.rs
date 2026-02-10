#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IitFrame {
    pub now_ms: u64,
    pub integration: f32,
    pub state: u8,
}
