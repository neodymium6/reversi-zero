/// Result of MCTS search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Best move selected (0-63), or None when the player must pass
    pub best_move: Option<usize>,

    /// Policy distribution over all 64 squares (probability vector)
    pub policy_distribution: Vec<f32>,

    /// Q-value of the root node
    pub root_value: f32,

    /// Number of simulations actually run
    pub num_simulations_run: u32,

    /// Visit counts for each legal move from root: (move, visit_count)
    pub root_visit_counts: Vec<(usize, u32)>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(
        best_move: Option<usize>,
        policy_distribution: Vec<f32>,
        root_value: f32,
        num_simulations_run: u32,
        root_visit_counts: Vec<(usize, u32)>,
    ) -> Self {
        Self {
            best_move,
            policy_distribution,
            root_value,
            num_simulations_run,
            root_visit_counts,
        }
    }

    /// Get the visit count for a specific move
    pub fn visit_count_for_move(&self, move_idx: usize) -> u32 {
        self.root_visit_counts
            .iter()
            .find(|(m, _)| *m == move_idx)
            .map(|(_, v)| *v)
            .unwrap_or(0)
    }

    /// Get the total number of visits to root
    pub fn total_visits(&self) -> u32 {
        self.root_visit_counts.iter().map(|(_, v)| v).sum()
    }
}
