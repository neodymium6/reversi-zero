use rust_reversi_core::board::Board as rrcBoard;
use tch::Tensor;

// Re-export types from rust_reversi_core for MCTS
pub use rust_reversi_core::board::{BoardError, Color, Turn};

const BITS: [u64; 64] = {
    let mut bits = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        bits[i] = 1u64 << (63 - i);
        i += 1;
    }
    bits
};

pub const BOARD_FEATURE_COUNT: usize = 3 * 8 * 8;

pub struct Board {
    inner: rrcBoard,
}

impl Board {
    pub fn new() -> Self {
        Self {
            inner: rrcBoard::new(),
        }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn to_features(&self) -> [f32; BOARD_FEATURE_COUNT] {
        const SIZE: usize = 8;
        const NUMEL_PER_PLANE: usize = SIZE * SIZE;
        let mut board_array = [0f32; BOARD_FEATURE_COUNT];
        let (player_board, opponent_board, _) = self.inner.get_board();
        for x in 0..SIZE {
            for y in 0..SIZE {
                let idx2d = x * SIZE + y;
                let bit = BITS[idx2d];
                let idx_player = idx2d; // plane 0
                let idx_opponent = NUMEL_PER_PLANE + idx2d; // plane 1
                let idx_empty = 2 * NUMEL_PER_PLANE + idx2d; // plane 2
                match (player_board & bit, opponent_board & bit) {
                    (0, 0) => board_array[idx_empty] = 1.0,
                    (_, 0) => board_array[idx_player] = 1.0,
                    (0, _) => board_array[idx_opponent] = 1.0,
                    (_, _) => {
                        // This should never happen in a valid Reversi game
                        debug_assert!(false, "Invalid board state: overlapping pieces");
                    }
                }
            }
        }
        board_array
    }

    pub fn to_tensor(&self) -> Tensor {
        Tensor::from_slice(&self.to_features()).view([3, 8, 8])
    }

    // Thin wrappers for MCTS to access game logic
    pub fn get_legal_moves_mask(&mut self) -> u64 {
        self.inner.get_legal_moves()
    }

    pub fn get_legal_moves_vec(&mut self) -> Vec<usize> {
        self.inner
            .get_legal_moves_vec()
            .into_iter()
            .copied()
            .collect()
    }

    pub fn set_board_str(&mut self, board_str: &str, turn: Turn) -> Result<(), BoardError> {
        self.inner.set_board_str(board_str, turn)
    }

    pub fn do_move(&mut self, position: usize) -> Result<(), BoardError> {
        self.inner.do_move(position)
    }

    pub fn is_game_over(&self) -> bool {
        self.inner.is_game_over()
    }

    pub fn get_winner(&self) -> Result<Option<Turn>, BoardError> {
        self.inner.get_winner()
    }

    pub fn is_pass(&self) -> bool {
        self.inner.is_pass()
    }

    pub fn do_pass(&mut self) -> Result<(), BoardError> {
        self.inner.do_pass()
    }

    pub fn get_turn(&self) -> Turn {
        self.inner.get_turn()
    }

    /// Return absolute black/white piece counts regardless of the side to move.
    pub fn piece_counts(&self) -> (u32, u32) {
        let (player, opponent, turn) = self.inner.get_board();
        let player_count = player.count_ones();
        let opponent_count = opponent.count_ones();
        match turn {
            Turn::Black => (player_count, opponent_count),
            Turn::White => (opponent_count, player_count),
        }
    }

    pub fn is_win(&self) -> Result<bool, BoardError> {
        self.inner.is_win()
    }

    pub fn is_lose(&self) -> Result<bool, BoardError> {
        self.inner.is_lose()
    }

    pub fn is_draw(&self) -> Result<bool, BoardError> {
        self.inner.is_draw()
    }
}

impl Clone for Board {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Board")
            .field("turn", &self.get_turn())
            .finish()
    }
}
