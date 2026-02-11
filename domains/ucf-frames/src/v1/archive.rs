#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArchiveAppendFrame {
    pub now_ms: u64,
    pub seq: u64,
    pub bytes: u32,
    pub flushed: bool,
}
