use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use reversi_core::Board;

use crate::backup::backup;
use crate::config::MctsConfig;
use crate::dirichlet::add_dirichlet_noise_to_root;
use crate::error::{MctsError, Result};
use crate::evaluation::PolicyValueModel;
use crate::expansion::expand_and_evaluate;
use crate::search_result::SearchResult;
use crate::selection::select;
use crate::tree::{MctsTree, NodeId};

/// Monte Carlo Tree Search for Reversi using AlphaZero algorithm
pub struct Mcts {
    tree: MctsTree,
}

impl Mcts {
    /// Create a new MCTS instance
    pub fn new() -> Self {
        Self {
            tree: MctsTree::new(),
        }
    }

    /// Run MCTS search from a given board position
    ///
    /// Returns the best move and policy distribution
    pub fn search<M: PolicyValueModel>(
        &mut self,
        board: &Board,
        model: &M,
        config: &MctsConfig,
    ) -> Result<SearchResult> {
        // Check if position is already terminal
        if board.is_game_over() {
            return Err(MctsError::TerminalPosition);
        }

        // 1. Initialize root node
        let root_id = self.initialize_root(board, model)?;

        // 2. Add Dirichlet noise if configured (for self-play)
        if config.add_dirichlet_noise {
            add_dirichlet_noise_to_root(&mut self.tree, root_id, config)?;
        }

        // 3. Run simulations
        for _ in 0..config.num_simulations {
            // Selection: traverse tree using PUCT
            let leaf_id = select(&self.tree, root_id, config.c_puct);

            // Expansion & Evaluation: expand and get NN value
            let value = expand_and_evaluate(&mut self.tree, leaf_id, model)?;

            // Backup: propagate value up tree
            backup(&mut self.tree, leaf_id, value);
        }

        // 4. Extract results
        self.create_search_result(root_id, config)
    }

    /// Initialize the tree with root node
    fn initialize_root<M: PolicyValueModel>(&mut self, board: &Board, model: &M) -> Result<NodeId> {
        let root_id = self.tree.initialize_root(board.clone());

        // Expand root immediately
        expand_and_evaluate(&mut self.tree, root_id, model)?;

        Ok(root_id)
    }

    /// Create search result from root node statistics
    fn create_search_result(&self, root_id: NodeId, config: &MctsConfig) -> Result<SearchResult> {
        let root = &self.tree.nodes[root_id];

        // Collect visit counts for each move
        let mut visit_counts = vec![0u32; 64];
        let mut move_visits = Vec::new();
        let mut has_pass_child = false;

        for &child_id in &root.children {
            let child = &self.tree.nodes[child_id];
            if let Some(m) = child.move_action {
                visit_counts[m] = child.visit_count;
                move_visits.push((m, child.visit_count));
            } else {
                has_pass_child = true;
            }
        }

        // If no moves but pass is available, return a pass result instead of panicking
        if move_visits.is_empty() {
            if has_pass_child {
                return Ok(SearchResult::new(
                    None,
                    vec![0.0; 64],
                    root.q_value(),
                    config.num_simulations,
                    Vec::new(),
                ));
            }

            return Err(MctsError::NoLegalMoves);
        }

        // Select best move based on temperature
        let best_move = self.select_move(&move_visits, config.temperature);

        // Create policy distribution (normalized visit counts)
        let total_visits: u32 = visit_counts.iter().sum();
        let policy_distribution: Vec<f32> = if total_visits > 0 {
            visit_counts
                .iter()
                .map(|&v| v as f32 / total_visits as f32)
                .collect()
        } else {
            vec![0.0; 64]
        };

        // Root Q-value
        let root_value = root.q_value();

        Ok(SearchResult::new(
            Some(best_move),
            policy_distribution,
            root_value,
            config.num_simulations,
            move_visits,
        ))
    }

    /// Select move based on temperature
    ///
    /// - temperature ≈ 0: argmax (deterministic)
    /// - temperature = 1: proportional to visits
    /// - temperature > 1: more exploration (visits^(1/t))
    fn select_move(&self, move_visits: &[(usize, u32)], temperature: f32) -> usize {
        if move_visits.is_empty() {
            panic!("No moves available");
        }

        if temperature < 0.01 {
            // Temperature ≈ 0: select argmax
            move_visits
                .iter()
                .max_by_key(|(_, v)| v)
                .map(|(m, _)| *m)
                .unwrap()
        } else {
            // Temperature > 0: sample proportional to visits^(1/temp)
            let inv_temp = 1.0f64 / temperature as f64;
            let weights: Vec<f64> = move_visits
                .iter()
                .map(|(_, v)| {
                    let w = *v as f64;
                    if w == 0.0 { 0.0 } else { w.powf(inv_temp) }
                })
                .collect();

            if weights.iter().all(|&w| w == 0.0) {
                // Fallback: all weights zero, pick argmax
                return move_visits
                    .iter()
                    .max_by_key(|(_, v)| v)
                    .map(|(m, _)| *m)
                    .unwrap();
            }

            match WeightedIndex::new(&weights) {
                Ok(dist) => {
                    let choice = dist.sample(&mut thread_rng());
                    move_visits[choice].0
                }
                Err(_) => move_visits
                    .iter()
                    .max_by_key(|(_, v)| v)
                    .map(|(m, _)| *m)
                    .unwrap(),
            }
        }
    }

    /// Reset the tree (clear all nodes)
    pub fn reset(&mut self) {
        self.tree.clear();
    }

    /// Get the number of nodes in the tree
    pub fn tree_size(&self) -> usize {
        self.tree.size()
    }
}

impl Default for Mcts {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::tree::MctsTree;
    

    #[test]
    fn test_mcts_creation() {
        let mcts = Mcts::new();
        assert_eq!(mcts.tree_size(), 0);
    }

    #[test]
    fn test_mcts_reset() {
        let mut mcts = Mcts::new();
        mcts.reset();
        assert_eq!(mcts.tree_size(), 0);
    }

    #[test]
    fn test_select_move_argmax() {
        let mcts = Mcts::new();
        let move_visits = vec![(0, 10), (1, 20), (2, 5)];

        let best = mcts.select_move(&move_visits, 0.0);
        assert_eq!(best, 1); // Move 1 has highest visits
    }

    #[test]
    fn test_select_move_single() {
        let mcts = Mcts::new();
        let move_visits = vec![(5, 100)];

        let best = mcts.select_move(&move_visits, 0.0);
        assert_eq!(best, 5);
    }

    #[test]
    fn test_select_move_with_temperature_sampling() {
        let mcts = Mcts::new();
        let move_visits = vec![(0, 10), (1, 10), (2, 10)];

        let mut picked = std::collections::HashSet::new();
        for _ in 0..50 {
            picked.insert(mcts.select_move(&move_visits, 1.0));
        }

        // With non-zero temperature and equal visits, sampling should pick multiple moves
        assert!(picked.len() > 1);
    }

    #[test]
    fn test_create_search_result_pass_only() {
        let mut mcts = Mcts::new();
        let mut tree = MctsTree::new();
        let board = Board::new();

        // Root
        tree.nodes
            .push(crate::tree::MctsNode::new_root(board.clone()));
        tree.nodes[0].is_expanded = true;

        // Only child is a pass (move_action None)
        let pass_child = crate::tree::MctsNode::new_child(board, None, 0, 1.0);
        tree.nodes.push(pass_child);
        tree.nodes[0].children.push(1);

        mcts.tree = tree;
        let cfg = MctsConfig::default().with_simulations(1);

        let result = mcts
            .create_search_result(0, &cfg)
            .expect("pass result should be ok");

        assert!(result.best_move.is_none());
        assert!(result.policy_distribution.iter().all(|&p| p == 0.0));
        assert!(result.root_visit_counts.is_empty());
    }
}
