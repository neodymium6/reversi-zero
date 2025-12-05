use rand::thread_rng;
use rand_distr::{Dirichlet, Distribution};

use crate::config::MctsConfig;
use crate::error::{MctsError, Result};
use crate::tree::{MctsTree, NodeId};

/// Add Dirichlet noise to root node priors for exploration
///
/// This is used during self-play to encourage exploration.
/// The noise is mixed with the original prior: P' = (1-ε)*P + ε*noise
///
/// Only applied to root node, not to other nodes in the tree.
pub fn add_dirichlet_noise_to_root(
    tree: &mut MctsTree,
    root_id: NodeId,
    config: &MctsConfig,
) -> Result<()> {
    let root = &tree.nodes[root_id];

    if !root.is_expanded {
        return Err(MctsError::RootNotInitialized);
    }

    let n = root.children.len();
    if n == 0 {
        // No children, nothing to add noise to
        return Ok(());
    }

    // Sample from Dirichlet distribution
    let alpha_vec = vec![config.dirichlet_alpha as f64; n];
    let dirichlet =
        Dirichlet::new(&alpha_vec).map_err(|e| MctsError::DirichletError(e.to_string()))?;

    let noise: Vec<f32> = dirichlet
        .sample(&mut thread_rng())
        .iter()
        .map(|&x| x as f32)
        .collect();

    // Mix noise with priors: P' = (1-ε)*P + ε*noise
    let eps = config.dirichlet_epsilon;
    let children = tree.nodes[root_id].children.clone();

    for (i, &child_id) in children.iter().enumerate() {
        let child = &mut tree.nodes[child_id];
        child.prior_probability = (1.0 - eps) * child.prior_probability + eps * noise[i];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::MctsNode;
    use approx::assert_relative_eq;
    use reversi_core::Board;

    #[test]
    fn test_dirichlet_noise_changes_priors() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create root with children
        tree.nodes.push(MctsNode::new_root(board.clone()));
        tree.nodes[0].is_expanded = true;

        // Add 3 children with equal priors
        for i in 0..3 {
            let child = MctsNode::new_child(board.clone(), Some(i), 0, 1.0 / 3.0);
            let child_id = tree.nodes.len();
            tree.nodes.push(child);
            tree.nodes[0].children.push(child_id);
        }

        // Store original priors
        let original_priors: Vec<f32> = tree.nodes[0]
            .children
            .iter()
            .map(|&id| tree.nodes[id].prior_probability)
            .collect();

        // Add noise
        let config = MctsConfig::default().with_dirichlet_noise(0.3, 0.25);
        add_dirichlet_noise_to_root(&mut tree, 0, &config).unwrap();

        // Check that priors changed
        let new_priors: Vec<f32> = tree.nodes[0]
            .children
            .iter()
            .map(|&id| tree.nodes[id].prior_probability)
            .collect();

        // At least one prior should be different
        assert!(
            original_priors
                .iter()
                .zip(new_priors.iter())
                .any(|(o, n)| (o - n).abs() > 1e-6),
            "Priors should change after adding noise"
        );

        // Priors should still sum to approximately 1 (within epsilon of mixing)
        let sum: f32 = new_priors.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_dirichlet_noise_no_children() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create root without children
        tree.nodes.push(MctsNode::new_root(board));
        tree.nodes[0].is_expanded = true;

        let config = MctsConfig::default().with_dirichlet_noise(0.3, 0.25);

        // Should not error
        let result = add_dirichlet_noise_to_root(&mut tree, 0, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dirichlet_noise_not_expanded() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Create root that's not expanded
        tree.nodes.push(MctsNode::new_root(board));

        let config = MctsConfig::default().with_dirichlet_noise(0.3, 0.25);

        // Should error
        let result = add_dirichlet_noise_to_root(&mut tree, 0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_dirichlet_preserves_relative_ordering_somewhat() {
        let mut tree = MctsTree::new();
        let board = Board::new();

        tree.nodes.push(MctsNode::new_root(board.clone()));
        tree.nodes[0].is_expanded = true;

        // Add children with very different priors
        let priors = [0.8, 0.15, 0.05];
        for (i, &prior) in priors.iter().enumerate() {
            let child = MctsNode::new_child(board.clone(), Some(i), 0, prior);
            let child_id = tree.nodes.len();
            tree.nodes.push(child);
            tree.nodes[0].children.push(child_id);
        }

        // Add small noise (epsilon = 0.1)
        let config = MctsConfig::default().with_dirichlet_noise(0.3, 0.1);
        add_dirichlet_noise_to_root(&mut tree, 0, &config).unwrap();

        let new_priors: Vec<f32> = tree.nodes[0]
            .children
            .iter()
            .map(|&id| tree.nodes[id].prior_probability)
            .collect();

        // With small epsilon, high prior should still be highest (usually)
        // This is a probabilistic test, but should pass most of the time
        assert!(
            new_priors[0] > new_priors[1] && new_priors[0] > new_priors[2],
            "Highest prior should usually stay highest with small epsilon"
        );
    }
}
