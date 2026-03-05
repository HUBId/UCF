#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum ErrorCode {
    RuntimePolicy = 1001,
    RuntimeEss = 1002,
    RuntimeCompute = 1003,
    RuntimePanic = 1004,
    OpsIo = 2001,
    OpsJson = 2002,
    OpsRuntime = 2003,
    OpsCompute = 2004,
    OpsInvalid = 2005,
    OpsReplay = 2006,
    OpsPolicyPack = 2007,
}

impl ErrorCode {
    pub const fn as_u16(self) -> u16 {
        self as u16
    }

    pub const fn stable_key(self) -> &'static str {
        match self {
            Self::RuntimePolicy => "runtime.policy",
            Self::RuntimeEss => "runtime.ess",
            Self::RuntimeCompute => "runtime.compute",
            Self::RuntimePanic => "runtime.panic",
            Self::OpsIo => "ops.io",
            Self::OpsJson => "ops.json",
            Self::OpsRuntime => "ops.runtime",
            Self::OpsCompute => "ops.compute",
            Self::OpsInvalid => "ops.invalid",
            Self::OpsReplay => "ops.replay",
            Self::OpsPolicyPack => "ops.policy_pack",
        }
    }
}
