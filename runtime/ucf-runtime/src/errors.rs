use core::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    Policy(ucf_policy::errors::PolicyError),
    Ess(ucf_ess::v1::EssError),
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
        }
    }
}

impl std::error::Error for RuntimeError {}
