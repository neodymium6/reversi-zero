use reversi_core::Board;

/// Node ID in the arena-style tree
pub type NodeId = usize;

/// A single node in the MCTS tree
pub struct MctsNode {
    /// Board state at this node
    pub state: Board,

    /// Move that led to this state (None for root or pass)
    pub move_action: Option<usize>,

    /// Parent node ID
    pub parent: Option<NodeId>,

    /// Child node IDs
    pub children: Vec<NodeId>,

    /// Number of times this node has been visited
    pub visit_count: u32,

    /// Sum of values backed up through this node
    pub total_value: f32,

    /// Prior probability from policy network
    pub prior_probability: f32,

    /// Whether this is a terminal state
    pub is_terminal: bool,

    /// Whether children have been created
    pub is_expanded: bool,

    /// Terminal value if is_terminal (-1, 0, 1)
    pub terminal_value: Option<f32>,
}

impl MctsNode {
    /// Create a new root node
    pub fn new_root(state: Board) -> Self {
        Self {
            state,
            move_action: None,
            parent: None,
            children: Vec::new(),
            visit_count: 0,
            total_value: 0.0,
            prior_probability: 1.0,
            is_terminal: false,
            is_expanded: false,
            terminal_value: None,
        }
    }

    /// Create a new child node
    pub fn new_child(state: Board, move_action: Option<usize>, parent: NodeId, prior: f32) -> Self {
        Self {
            state,
            move_action,
            parent: Some(parent),
            children: Vec::new(),
            visit_count: 0,
            total_value: 0.0,
            prior_probability: prior,
            is_terminal: false,
            is_expanded: false,
            terminal_value: None,
        }
    }

    /// Get Q-value (average value)
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }
}

/// MCTS tree using arena allocation
pub struct MctsTree {
    /// Arena of all nodes
    pub nodes: Vec<MctsNode>,

    /// Root node ID (usually 0)
    pub root_id: NodeId,
}

impl MctsTree {
    /// Create a new empty tree
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(10000),
            root_id: 0,
        }
    }

    /// Initialize the tree with a root node
    pub fn initialize_root(&mut self, board: Board) -> NodeId {
        self.nodes.clear();
        let root = MctsNode::new_root(board);
        self.nodes.push(root);
        self.root_id = 0;
        self.root_id
    }

    /// Add a new node and return its ID
    pub fn add_node(&mut self, node: MctsNode) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Get the number of nodes in the tree
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Clear the tree
    pub fn clear(&mut self) {
        self.nodes.clear();
    }
}

impl Default for MctsTree {
    fn default() -> Self {
        Self::new()
    }
}
