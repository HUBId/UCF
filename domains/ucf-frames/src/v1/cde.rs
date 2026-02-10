#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CdeFrame {
    pub now_ms: u64,
    pub hyps: u32,
    pub top_conf: f32,
    pub acyclic: bool,
}
