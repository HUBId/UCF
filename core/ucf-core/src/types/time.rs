#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Tick(u64);

impl Tick {
    pub const fn new(v: u64) -> Self {
        Self(v)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct WindowId(u64);

impl WindowId {
    pub const fn new(v: u64) -> Self {
        Self(v)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SimTime {
    pub tick: Tick,
    pub window: WindowId,
}
