use thiserror::Error;

#[derive(Error, Debug)]
pub enum MctsError {
    #[error("Board error: {0:?}")]
    BoardError(reversi_core::BoardError),

    #[error("Torch error: {0}")]
    TorchError(#[from] tch::TchError),

    #[error("Terminal position cannot be searched")]
    TerminalPosition,

    #[error("No legal moves available")]
    NoLegalMoves,

    #[error("Invalid node ID: {0}")]
    InvalidNodeId(usize),

    #[error("Root node not initialized")]
    RootNotInitialized,

    #[error("NN evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Dirichlet sampling error: {0}")]
    DirichletError(String),
}

// Manual conversion from BoardError
impl From<reversi_core::BoardError> for MctsError {
    fn from(err: reversi_core::BoardError) -> Self {
        MctsError::BoardError(err)
    }
}

pub type Result<T> = std::result::Result<T, MctsError>;
