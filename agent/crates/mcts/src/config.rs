/// Configuration for MCTS search
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Number of simulations to run
    pub num_simulations: u32,

    /// Maximum number of leaf evaluations to batch per forward pass
    pub batch_size: u32,

    /// Maximum time (ms) to wait before flushing a partial batch
    pub batch_timeout_ms: u64,

    /// PUCT exploration constant (typically 1.0-5.0)
    pub c_puct: f32,

    /// Whether to add Dirichlet noise to root (for self-play)
    pub add_dirichlet_noise: bool,

    /// Dirichlet alpha parameter (0.3 for Chess, 0.03 for Go, tune for Reversi)
    pub dirichlet_alpha: f32,

    /// Dirichlet epsilon for mixing noise (typically 0.25)
    pub dirichlet_epsilon: f32,

    /// Temperature for move selection (1.0 = proportional to visits, 0.0 = argmax)
    pub temperature: f32,
}

impl MctsConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of simulations
    pub fn with_simulations(mut self, n: u32) -> Self {
        self.num_simulations = n;
        self
    }

    /// Set maximum batch size for neural network inference
    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Set maximum wait time for forming a batch (milliseconds)
    pub fn with_batch_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.batch_timeout_ms = timeout_ms;
        self
    }

    /// Set PUCT exploration constant
    pub fn with_c_puct(mut self, c: f32) -> Self {
        self.c_puct = c;
        self
    }

    /// Set temperature for move selection
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Enable Dirichlet noise with given parameters
    pub fn with_dirichlet_noise(mut self, alpha: f32, epsilon: f32) -> Self {
        self.add_dirichlet_noise = true;
        self.dirichlet_alpha = alpha;
        self.dirichlet_epsilon = epsilon;
        self
    }

    /// Disable Dirichlet noise
    pub fn without_dirichlet_noise(mut self) -> Self {
        self.add_dirichlet_noise = false;
        self
    }
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            batch_size: 1,
            batch_timeout_ms: 2,
            c_puct: 1.5,
            add_dirichlet_noise: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MctsConfig::default();
        assert_eq!(config.num_simulations, 800);
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.batch_timeout_ms, 2);
        assert_eq!(config.c_puct, 1.5);
        assert!(!config.add_dirichlet_noise);
    }

    #[test]
    fn test_builder_pattern() {
        let config = MctsConfig::default()
            .with_simulations(1000)
            .with_batch_size(4)
            .with_batch_timeout_ms(5)
            .with_c_puct(2.0)
            .with_temperature(0.5)
            .with_dirichlet_noise(0.25, 0.3);

        assert_eq!(config.num_simulations, 1000);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.batch_timeout_ms, 5);
        assert_eq!(config.c_puct, 2.0);
        assert_eq!(config.temperature, 0.5);
        assert!(config.add_dirichlet_noise);
        assert_eq!(config.dirichlet_alpha, 0.25);
        assert_eq!(config.dirichlet_epsilon, 0.3);
    }
}
