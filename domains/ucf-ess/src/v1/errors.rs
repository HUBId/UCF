use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EssError {
    TimeWentBackwards,
    InvalidAppend(&'static str),
    NotFound,
}

impl fmt::Display for EssError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimeWentBackwards => write!(f, "append rejected: time went backwards"),
            Self::InvalidAppend(msg) => write!(f, "append rejected: {msg}"),
            Self::NotFound => write!(f, "record not found"),
        }
    }
}

impl std::error::Error for EssError {}
