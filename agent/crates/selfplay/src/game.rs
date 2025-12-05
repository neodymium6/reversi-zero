use anyhow::Result;
use reversi_core::{Board, Turn};
use reversi_mcts::{Mcts, MctsConfig, PolicyValueModel};

use crate::data::{GameRecord, GameResult};

/// Play a single self-play game
///
/// # Arguments
/// * `model` - Neural network model for MCTS evaluation
/// * `config` - MCTS configuration
///
/// # Returns
/// A complete game record with all states, policies, and the final result
pub fn play_game<M: PolicyValueModel>(model: &M, config: &MctsConfig) -> Result<GameRecord> {
    let mut board = Board::new();
    let mut mcts = Mcts::new();
    let mut record = GameRecord::new();

    while !board.is_game_over() {
        // Run MCTS search
        let result = mcts.search(&board, model, config)?;

        // Record state, policy, and move
        record.add_move(board.clone(), result.policy_distribution, result.best_move);

        // Apply the move
        match result.best_move {
            Some(m) => board
                .do_move(m)
                .map_err(|e| anyhow::anyhow!("Move failed: {:?}", e))?,
            None => board
                .do_pass()
                .map_err(|e| anyhow::anyhow!("Pass failed: {:?}", e))?,
        }

        // Reset MCTS tree for next move (optional, can also reuse)
        mcts.reset();
    }

    // Determine the winner
    let winner = determine_winner(&board)?;
    record.set_winner(winner);

    Ok(record)
}

/// Determine the winner from a finished board
fn determine_winner(board: &Board) -> Result<GameResult> {
    // Board should be game over
    if !board.is_game_over() {
        anyhow::bail!("Board is not game over");
    }

    // Check win/lose/draw from the current player's perspective
    let is_win = board
        .is_win()
        .map_err(|e| anyhow::anyhow!("Failed to check win: {:?}", e))?;
    let is_lose = board
        .is_lose()
        .map_err(|e| anyhow::anyhow!("Failed to check lose: {:?}", e))?;
    let is_draw = board
        .is_draw()
        .map_err(|e| anyhow::anyhow!("Failed to check draw: {:?}", e))?;

    if is_win {
        // Current player won
        Ok(match board.get_turn() {
            Turn::Black => GameResult::BlackWin,
            Turn::White => GameResult::WhiteWin,
        })
    } else if is_lose {
        // Current player lost
        Ok(match board.get_turn() {
            Turn::Black => GameResult::WhiteWin,
            Turn::White => GameResult::BlackWin,
        })
    } else if is_draw {
        Ok(GameResult::Draw)
    } else {
        anyhow::bail!("Unexpected game state");
    }
}

/// Convert game record to training examples
///
/// Each position in the game is converted to a training example with:
/// - state: the board position
/// - policy: the MCTS policy distribution
/// - value: the game outcome from that player's perspective
pub fn game_to_training_examples(record: &GameRecord) -> Vec<crate::data::TrainingExample> {
    let mut examples = Vec::new();

    for (move_index, (state, policy)) in
        record.states.iter().zip(record.policies.iter()).enumerate()
    {
        // Convert board to vector
        let state_vec = board_to_vec(state);

        // Calculate value based on game result and whose turn it is
        let value = calculate_value(move_index, record);

        examples.push(crate::data::TrainingExample::new(
            state_vec,
            policy.clone(),
            value,
        ));
    }

    examples
}

/// Convert Board to flat vector for neural network input
fn board_to_vec(board: &Board) -> Vec<f32> {
    // Use Board::to_tensor() and extract data
    let tensor = board.to_tensor();
    let mut vec = vec![0.0f32; 3 * 8 * 8];
    let len = vec.len();

    // Copy tensor data to vector
    // Safety: The tensor is contiguous and has exactly 192 elements
    tensor.copy_data(&mut vec, len);

    vec
}

/// Calculate the value (game outcome) for a position
///
/// Returns:
/// - +1.0 if the player at this position won
/// - -1.0 if the player at this position lost
/// - 0.0 for a draw
fn calculate_value(move_index: usize, record: &GameRecord) -> f32 {
    // Determine whose turn it is based on move index
    // Even indices (0, 2, 4, ...) are Black's turns
    // Odd indices (1, 3, 5, ...) are White's turns
    let is_black_turn = move_index % 2 == 0;

    match record.winner {
        GameResult::BlackWin => {
            if is_black_turn {
                1.0 // Black won, and it's Black's turn
            } else {
                -1.0 // Black won, but it's White's turn
            }
        }
        GameResult::WhiteWin => {
            if is_black_turn {
                -1.0 // White won, but it's Black's turn
            } else {
                1.0 // White won, and it's White's turn
            }
        }
        GameResult::Draw => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_value_black_win() {
        let mut record = GameRecord::new();
        record.set_winner(GameResult::BlackWin);

        // Move 0 (Black's turn): should be +1.0
        assert_eq!(calculate_value(0, &record), 1.0);

        // Move 1 (White's turn): should be -1.0
        assert_eq!(calculate_value(1, &record), -1.0);

        // Move 2 (Black's turn): should be +1.0
        assert_eq!(calculate_value(2, &record), 1.0);
    }

    #[test]
    fn test_calculate_value_white_win() {
        let mut record = GameRecord::new();
        record.set_winner(GameResult::WhiteWin);

        // Move 0 (Black's turn): should be -1.0
        assert_eq!(calculate_value(0, &record), -1.0);

        // Move 1 (White's turn): should be +1.0
        assert_eq!(calculate_value(1, &record), 1.0);
    }

    #[test]
    fn test_calculate_value_draw() {
        let mut record = GameRecord::new();
        record.set_winner(GameResult::Draw);

        assert_eq!(calculate_value(0, &record), 0.0);
        assert_eq!(calculate_value(1, &record), 0.0);
        assert_eq!(calculate_value(10, &record), 0.0);
    }

    #[test]
    fn test_board_to_vec_size() {
        let board = Board::new();
        let vec = board_to_vec(&board);
        assert_eq!(vec.len(), 3 * 8 * 8);
    }
}
