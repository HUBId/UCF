#![forbid(unsafe_code)]

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Error {
    message: String,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Error {}

pub fn to_string<T>(_value: &T) -> Result<String> {
    Ok(String::new())
}

pub fn from_str<T>(_s: &str) -> Result<T>
where
    T: Default,
{
    Ok(T::default())
}
