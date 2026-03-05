use core::fmt::{Display, Formatter};
use ucf_types::error_codes::ErrorCode;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    Policy(ucf_policy::errors::PolicyError),
    Ess(ucf_ess::v1::EssError),
    Compute(ucf_compute::ComputeError),
    Panic {
        stage: &'static str,
        panic_digest: String,
        fail_fast: bool,
    },
}

impl From<ucf_policy::errors::PolicyError> for RuntimeError {
    fn from(value: ucf_policy::errors::PolicyError) -> Self {
        Self::Policy(value)
    }
}

impl From<ucf_ess::v1::EssError> for RuntimeError {
    fn from(value: ucf_ess::v1::EssError) -> Self {
        Self::Ess(value)
    }
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Policy(error) => write!(f, "policy error: {error}"),
            Self::Ess(error) => write!(f, "ess error: {error}"),
            Self::Compute(error) => write!(f, "compute error: {error}"),
            Self::Panic {
                stage,
                panic_digest,
                fail_fast,
            } => write!(
                f,
                "panic caught in stage={stage} digest={} action={}",
                &panic_digest[..panic_digest.len().min(12)],
                if *fail_fast { "shutdown" } else { "degraded" }
            ),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<ucf_compute::ComputeError> for RuntimeError {
    fn from(value: ucf_compute::ComputeError) -> Self {
        Self::Compute(value)
    }
}

impl RuntimeError {
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::Policy(_) => ErrorCode::RuntimePolicy,
            Self::Ess(_) => ErrorCode::RuntimeEss,
            Self::Compute(_) => ErrorCode::RuntimeCompute,
            Self::Panic { .. } => ErrorCode::RuntimePanic,
        }
    }
}
