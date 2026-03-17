#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error("Duplicate value")]
    DuplicateValue { first: usize, second: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum ApplicationError {
    #[error("Invalid take")]
    InvalidTake(u64),
    #[error("Invalid drop")]
    InvalidDrop(u64),
    #[error("Unexpected source")]
    UnexpectedSource(u64),
    #[error("Unexpected source length")]
    UnexpectedSourceLen { expected: u64, actual: u64 },
    #[error("Unexpected target length")]
    UnexpectedTargetLen { expected: i64 },
}

impl ApplicationError {
    #[must_use]
    pub const fn position(&self) -> u64 {
        match self {
            Self::InvalidTake(position)
            | Self::InvalidDrop(position)
            | Self::UnexpectedSource(position) => *position,
            Self::UnexpectedSourceLen { .. } | Self::UnexpectedTargetLen { .. } => 0,
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
    #[error("ID error")]
    Id(std::io::Error),
    #[error("Timestamp error")]
    Timestamp(std::io::Error),
    #[error("Invalid timestamp")]
    InvalidTimestamp(u32),
}
