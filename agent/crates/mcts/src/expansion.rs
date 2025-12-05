use reversi_core::Board;

use crate::error::Result;
use crate::evaluation::{PolicyValueModel, evaluate_with_nn, softmax_legal_moves_mask};
use crate::tree::{MctsNode, MctsTree, NodeId};

pub fn expand_and_evaluate<M: PolicyValueModel>(
    tree: &mut MctsTree,
    leaf_id: NodeId,
    model: &M,
) -> Result<f32> {
    // Clone state first to avoid borrow checker issues
    let leaf_state = tree.nodes[leaf_id].state.clone();

    // Handle terminal nodes
    if leaf_state.is_game_over() {
        let value = calculate_terminal_value(&leaf_state);
        tree.nodes[leaf_id].is_terminal = true;
        tree.nodes[leaf_id].terminal_value = Some(value);
        tree.nodes[leaf_id].is_expanded = true;
        return Ok(value);
    }

    // Get NN evaluation
    let (policy_logits, value) = evaluate_with_nn(&leaf_state, model)?;

    expand_with_policy_value(tree, leaf_id, leaf_state, &policy_logits, value)
}

/// Expand a node using precomputed policy logits and value.
///
/// This is used by both the standard single-eval path and batched eval paths.
pub fn expand_with_policy_value(
    tree: &mut MctsTree,
    leaf_id: NodeId,
    mut leaf_state: Board,
    policy_logits: &[f32],
    value: f32,
) -> Result<f32> {
    // Pass check is cheaper than generating legal move list, so handle it first
    if leaf_state.is_pass() {
        let mut child_state = leaf_state.clone();
        child_state.do_pass()?;

        let child = MctsNode::new_child(child_state, None, leaf_id, 1.0);
        let child_id = tree.add_node(child);
        tree.nodes[leaf_id].children.push(child_id);

        tree.nodes[leaf_id].is_expanded = true;
        return Ok(value);
    }

    // Get legal moves as bitmask
    let legal_moves_mask = leaf_state.get_legal_moves_mask();

    if legal_moves_mask == 0 {
        // No legal moves and not a pass: unexpected but avoid panic
        tree.nodes[leaf_id].is_expanded = true;
        return Ok(value);
    }

    // Softmax policy over legal moves
    let move_priors = softmax_legal_moves_mask(policy_logits, legal_moves_mask);

    // Create children
    for (move_idx, prior) in move_priors {
        let mut child_state = leaf_state.clone();
        child_state.do_move(move_idx)?;

        let child = MctsNode::new_child(child_state, Some(move_idx), leaf_id, prior);
        let child_id = tree.add_node(child);
        tree.nodes[leaf_id].children.push(child_id);
    }

    tree.nodes[leaf_id].is_expanded = true;
    Ok(value)
}

/// Calculate terminal value from current player's perspective
///
/// Returns:
/// - 1.0 if current player wins
/// - -1.0 if current player loses
/// - 0.0 for draw
pub(crate) fn calculate_terminal_value(board: &Board) -> f32 {
    // Use is_win/is_lose/is_draw for efficiency
    if board.is_win().unwrap_or(false) {
        1.0
    } else if board.is_lose().unwrap_or(false) {
        -1.0
    } else {
        0.0 // Draw or error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::PolicyValueModel;
    use reversi_core::Turn;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_terminal_value_black_wins() {
        // Board full with a majority for current player (Black)
        let mut board = Board::new();
        let mut s = String::new();
        s.extend(std::iter::repeat('X').take(63));
        s.push('O');
        board.set_board_str(&s, Turn::Black).unwrap();
        assert!(board.is_game_over());
        assert_eq!(calculate_terminal_value(&board), 1.0);
    }

    #[test]
    fn test_terminal_value_draw() {
        // Equal pieces -> draw
        let mut board = Board::new();
        let mut s = String::new();
        s.extend(std::iter::repeat('X').take(32));
        s.extend(std::iter::repeat('O').take(32));
        board.set_board_str(&s, Turn::Black).unwrap();
        assert!(board.is_game_over());
        assert_eq!(calculate_terminal_value(&board), 0.0);
    }

    #[test]
    fn test_terminal_value_black_loses() {
        // Board full with a majority for opponent (White from Black's perspective)
        let mut board = Board::new();
        let mut s = String::new();
        s.extend(std::iter::repeat('O').take(63));
        s.push('X');
        board.set_board_str(&s, Turn::Black).unwrap();
        assert!(board.is_game_over());
        assert_eq!(calculate_terminal_value(&board), -1.0);
    }

    #[test]
    fn test_pass_only_position_expands_pass_child() {
        // A position where black has no legal move but game is not over (must pass)
        // Setup: Black has no moves, White occupies edges, but game isn't over.
        // Board string uses player's perspective; Turn::Black means 'X' are black, 'O' are white.
        let mut board = Board::new();
        board
            .set_board_str(
                "OX--------------------------------------------------------------",
                Turn::Black,
            )
            .unwrap();

        let mut tree = MctsTree::new();
        let root_id = tree.initialize_root(board.clone());

        // Minimal dummy model: returns uniform policy and zero value
        let dummy_model = DummyModel;

        let value = expand_and_evaluate(&mut tree, root_id, &dummy_model).unwrap();

        // Should mark expanded and have exactly one child representing pass
        assert_eq!(tree.nodes[root_id].children.len(), 1);
        assert!(tree.nodes[root_id].is_expanded);
        assert!(
            tree.nodes[tree.nodes[root_id].children[0]]
                .move_action
                .is_none()
        );

        // Value should be returned (dummy model returns 0.0)
        assert_eq!(value, 0.0);
    }

    struct DummyModel;

    impl DummyModel {
        fn forward_impl(&self, _x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
            Ok((
                Tensor::zeros([1, 64], (Kind::Float, Device::Cpu)),
                Tensor::zeros([1], (Kind::Float, Device::Cpu)),
            ))
        }

        fn device_impl(&self) -> Device {
            Device::Cpu
        }
    }

    impl PolicyValueModel for DummyModel {
        fn forward(&self, x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
            self.forward_impl(x)
        }

        fn device(&self) -> Device {
            self.device_impl()
        }
    }
}
