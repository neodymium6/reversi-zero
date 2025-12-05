use reversi_core::Board;

/// Result of a game
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    BlackWin,
    WhiteWin,
    Draw,
}

/// Record of a single game
#[derive(Debug, Clone)]
pub struct GameRecord {
    /// Board state at each move
    pub states: Vec<Board>,
    /// MCTS policy distribution (64 elements) at each move
    pub policies: Vec<Vec<f32>>,
    /// Actual move taken (None for pass)
    pub moves: Vec<Option<usize>>,
    /// Final result of the game
    pub winner: GameResult,
}

impl GameRecord {
    /// Create a new empty game record
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            policies: Vec::new(),
            moves: Vec::new(),
            winner: GameResult::Draw,
        }
    }

    /// Set the winner
    pub fn set_winner(&mut self, winner: GameResult) {
        self.winner = winner;
    }

    /// Add a move to the record
    pub fn add_move(&mut self, state: Board, policy: Vec<f32>, action: Option<usize>) {
        self.states.push(state);
        self.policies.push(policy);
        self.moves.push(action);
    }

    /// Get the number of moves in this game
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if the record is empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl Default for GameRecord {
    fn default() -> Self {
        Self::new()
    }
}

/// Training example for neural network
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Board state as flat vector (3*8*8 = 192 elements)
    pub state: Vec<f32>,
    /// Target policy distribution (64 elements)
    pub policy: Vec<f32>,
    /// Target value (-1.0, 0.0, or 1.0)
    pub value: f32,
}

impl TrainingExample {
    /// Create a new training example
    pub fn new(state: Vec<f32>, policy: Vec<f32>, value: f32) -> Self {
        Self {
            state,
            policy,
            value,
        }
    }
}
