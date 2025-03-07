#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error("Duplicate value")]
    DuplicateValue { first: usize, second: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum ApplicationError {
    #[error("Invalid take")]
    InvalidTake(usize),
    #[error("Invalid drop")]
    InvalidDrop(usize),
    #[error("Unexpected data")]
    UnexpectedData(usize),
}

impl ApplicationError {
    pub fn position(&self) -> usize {
        match self {
            Self::InvalidTake(position) => *position,
            Self::InvalidDrop(position) => *position,
            Self::UnexpectedData(position) => *position,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodingError {
    #[error("Invalid count")]
    InvalidCount(u64),
    #[error("Invalid value")]
    InvalidValue(u64),
    #[error("Invalid insert count")]
    InvalidInsertCount(u64),
    #[error("Invalid insert value")]
    InvalidInsertValue(u64),
    #[error("Unexpected data")]
    UnexpectedData(u64),
}

impl DecodingError {
    pub fn position(&self) -> u64 {
        match self {
            Self::InvalidCount(position) => *position,
            Self::InvalidValue(position) => *position,
            Self::InvalidInsertCount(position) => *position,
            Self::InvalidInsertValue(position) => *position,
            Self::UnexpectedData(position) => *position,
        }
    }
}
