use crate::tree::{MctsTree, NodeId};

/// Select a leaf node to expand using PUCT formula
///
/// Traverses the tree from root, selecting children with highest PUCT value
/// until reaching a leaf (unexpanded or terminal) node.
pub fn select(tree: &MctsTree, root_id: NodeId, c_puct: f32) -> NodeId {
    let mut current_id = root_id;

    loop {
        let node = &tree.nodes[current_id];

        // Stop at leaf or terminal
        if !node.is_expanded || node.is_terminal {
            return current_id;
        }

        // Select child with max PUCT
        let sqrt_parent = (node.visit_count as f32).sqrt();

        let best_child = node
            .children
            .iter()
            .max_by(|&&a, &&b| {
                let puct_a = puct_value(tree, a, sqrt_parent, c_puct);
                let puct_b = puct_value(tree, b, sqrt_parent, c_puct);
                puct_a
                    .partial_cmp(&puct_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .expect("Expanded node must have children");

        current_id = best_child;
    }
}

/// Calculate PUCT value for a node
///
/// PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
///
/// Where:
/// - Q(s, a) = W(s, a) / N(s, a) is the average action value
/// - P(s, a) is the prior probability from policy network
/// - N(s) is parent visit count
/// - N(s, a) is child visit count
/// - c_puct is the exploration constant
fn puct_value(tree: &MctsTree, node_id: NodeId, sqrt_parent: f32, c_puct: f32) -> f32 {
    let node = &tree.nodes[node_id];

    // Q value (exploitation)
    let q = node.q_value();

    // U value (exploration)
    let u = c_puct * node.prior_probability * sqrt_parent / (1.0 + node.visit_count as f32);

    q + u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::MctsNode;
    use reversi_core::Board;

    #[test]
    fn test_puct_value_unvisited() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create root
        tree.nodes.push(MctsNode::new_root(board.clone()));
        tree.nodes[0].visit_count = 100;
        tree.nodes[0].is_expanded = true;

        // Create child with prior 0.5, unvisited
        let child = MctsNode::new_child(board, Some(0), 0, 0.5);
        tree.nodes.push(child);
        tree.nodes[0].children.push(1);

        let sqrt_parent = (tree.nodes[0].visit_count as f32).sqrt();
        let puct = puct_value(&tree, 1, sqrt_parent, 1.5);

        // PUCT = 0 + 1.5 * 0.5 * sqrt(100) / (1 + 0)
        //      = 1.5 * 0.5 * 10 / 1
        //      = 7.5
        assert!((puct - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_puct_value_visited() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create root
        tree.nodes.push(MctsNode::new_root(board.clone()));
        tree.nodes[0].visit_count = 100;
        tree.nodes[0].is_expanded = true;

        // Create child with visits
        let mut child = MctsNode::new_child(board, Some(0), 0, 0.5);
        child.visit_count = 10;
        child.total_value = 5.0; // Q = 0.5
        tree.nodes.push(child);
        tree.nodes[0].children.push(1);

        let sqrt_parent = (tree.nodes[0].visit_count as f32).sqrt();
        let puct = puct_value(&tree, 1, sqrt_parent, 1.5);

        // Q = 5.0 / 10 = 0.5
        // U = 1.5 * 0.5 * 10 / (1 + 10) ≈ 0.682
        // PUCT ≈ 1.182
        assert!((puct - 1.182).abs() < 0.01);
    }
}
