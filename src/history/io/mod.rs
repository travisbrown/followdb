use std::path::PathBuf;

pub mod compact;
pub mod expanded;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("Invalid file name")]
    InvalidFileName(PathBuf),
    #[error("Invalid line")]
    InvalidLine(String),
    #[error("BASE64 decoding error")]
    Base64(#[from] base64::DecodeError),
    #[error("Follow input error")]
    FollowInput(#[from] crate::diff::error::InputError),
    #[error("Follow decoding error")]
    FollowDecoding(#[from] crate::diff::error::DecodingError),
    #[error("Follow application error")]
    FollowApplication(#[from] crate::diff::error::ApplicationError),
}
