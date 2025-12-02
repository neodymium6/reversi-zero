use rust_reversi_core::board::Board as rrcBoard;
use tch::Tensor;

const BITS: [u64; 64] = {
    let mut bits = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        bits[i] = 1u64 << (63 - i);
        i += 1;
    }
    bits
};

pub struct Board {
    inner: rrcBoard,
}

impl Board {
    pub fn new() -> Self {
        Self {
            inner: rrcBoard::new(),
        }
    }
    pub fn to_tensor(&self) -> Tensor {
        const PLANES: usize = 3;
        const SIZE: usize = 8;
        const NUMEL_PER_PLANE: usize = SIZE * SIZE;
        let mut board_array = [0f32; PLANES * NUMEL_PER_PLANE];
        let (player_board, opponent_board, _) = self.inner.get_board();
        for x in 0..SIZE {
            for y in 0..SIZE {
                let idx2d = x * SIZE + y;
                let bit = BITS[idx2d];
                let idx_player = 0 * NUMEL_PER_PLANE + idx2d; // plane 0
                let idx_opponent = 1 * NUMEL_PER_PLANE + idx2d; // plane 1
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
        // Tensor::of_slice(&board_array).view([PLANES as i64, SIZE as i64, SIZE as i64])
        Tensor::from_slice(&board_array).view([PLANES as i64, SIZE as i64, SIZE as i64])
    }
}

