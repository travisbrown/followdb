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
#[derive(Debug, thiserror::Error)]
pub enum DecodingError {
    #[error("Count error")]
    Count(std::io::Error),
    #[error("Value error")]
    Value(std::io::Error),
    #[error("Insert count error")]
    InsertCount(std::io::Error),
    #[error("Insert value error")]
    InsertValue(std::io::Error),
    #[error("Index error")]
    Index(std::io::Error),
    #[error("Timestamp error")]
    Timestamp(std::io::Error),
}
