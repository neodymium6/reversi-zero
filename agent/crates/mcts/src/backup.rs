use crate::tree::{MctsTree, NodeId};

/// Backup value from leaf to root
///
/// CRITICAL: In zero-sum games, value must be negated when moving to parent
/// because what's good for the child is bad for the parent (they're opponents)
pub fn backup(tree: &mut MctsTree, leaf_id: NodeId, value: f32) {
    let mut current_value = value;
    let mut current_id = Some(leaf_id);

    while let Some(node_id) = current_id {
        let node = &mut tree.nodes[node_id];

        // Update statistics
        node.visit_count += 1;
        node.total_value += current_value;

        // Move to parent
        current_id = node.parent;

        // CRITICAL: Negate value for opponent
        // In zero-sum games, if position is +0.5 for current player,
        // it's -0.5 for opponent
        current_value = -current_value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::MctsNode;
    use reversi_core::Board;

    #[test]
    fn test_backup_single_node() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create root
        tree.nodes.push(MctsNode::new_root(board));
        tree.root_id = 0;

        // Backup value to root
        backup(&mut tree, 0, 0.5);

        assert_eq!(tree.nodes[0].visit_count, 1);
        assert_eq!(tree.nodes[0].total_value, 0.5);
    }

    #[test]
    fn test_backup_negates_value() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create root
        tree.nodes.push(MctsNode::new_root(board.clone()));
        tree.root_id = 0;
        tree.nodes[0].is_expanded = true;

        // Create child
        let child = MctsNode::new_child(board, Some(0), 0, 0.5);
        tree.nodes.push(child);
        tree.nodes[0].children.push(1);

        // Backup from child
        backup(&mut tree, 1, 0.5);

        // Child gets +0.5
        assert_eq!(tree.nodes[1].visit_count, 1);
        assert_eq!(tree.nodes[1].total_value, 0.5);

        // Root gets -0.5 (negated because opponent)
        assert_eq!(tree.nodes[0].visit_count, 1);
        assert_eq!(tree.nodes[0].total_value, -0.5);
    }

    #[test]
    fn test_backup_multiple_visits() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create root
        tree.nodes.push(MctsNode::new_root(board));

        // First backup
        backup(&mut tree, 0, 0.5);
        assert_eq!(tree.nodes[0].visit_count, 1);
        assert_eq!(tree.nodes[0].total_value, 0.5);

        // Second backup
        backup(&mut tree, 0, 0.3);
        assert_eq!(tree.nodes[0].visit_count, 2);
        assert_eq!(tree.nodes[0].total_value, 0.8); // 0.5 + 0.3

        // Q-value should be average
        assert!((tree.nodes[0].q_value() - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_backup_deep_tree() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create a 3-level tree: root -> child1 -> child2
        tree.nodes.push(MctsNode::new_root(board.clone()));
        tree.nodes[0].is_expanded = true;

        let child1 = MctsNode::new_child(board.clone(), Some(0), 0, 0.5);
        tree.nodes.push(child1);
        tree.nodes[0].children.push(1);
        tree.nodes[1].is_expanded = true;

        let child2 = MctsNode::new_child(board, Some(1), 1, 0.5);
        tree.nodes.push(child2);
        tree.nodes[1].children.push(2);

        // Backup from deepest child with value 1.0
        backup(&mut tree, 2, 1.0);

        // child2: +1.0
        assert_eq!(tree.nodes[2].total_value, 1.0);

        // child1: -1.0 (negated once)
        assert_eq!(tree.nodes[1].total_value, -1.0);

        // root: +1.0 (negated twice)
        assert_eq!(tree.nodes[0].total_value, 1.0);

        // All should have visit_count = 1
        assert_eq!(tree.nodes[0].visit_count, 1);
        assert_eq!(tree.nodes[1].visit_count, 1);
        assert_eq!(tree.nodes[2].visit_count, 1);
    }
}
