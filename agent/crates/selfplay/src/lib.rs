//! Self-play system for AlphaZero-style reinforcement learning
//!
//! This crate provides functionality for:
//! - Playing self-play games using MCTS
//! - Recording game states and policies
//! - Converting games to training examples
//! - Saving training data to NPY files for Python/PyTorch
//!
//! # Example
//!
//! ```no_run
//! use reversi_selfplay::{play_game, game_to_training_examples, save_training_data};
//! use reversi_mcts::MctsConfig;
//! use reversi_nn::NnModel;
//! use tch::Device;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Load model
//! let model = NnModel::load("model.pt", Device::Cpu)?;
//!
//! // Configure MCTS for self-play
//! let config = MctsConfig::default()
//!     .with_simulations(800)
//!     .with_temperature(1.0)
//!     .with_dirichlet_noise(0.3, 0.25);
//!
//! // Play one game
//! let record = play_game(&model, &config)?;
//! println!("Game finished with {} moves", record.len());
//!
//! // Convert to training examples
//! let examples = game_to_training_examples(&record);
//! println!("Generated {} training examples", examples.len());
//!
//! // Save to files
//! save_training_data(&examples, "selfplay_data")?;
//! # Ok(())
//! # }
//! ```

mod data;
mod game;
pub mod storage;

// Re-export public API
pub use data::{GameRecord, GameResult, TrainingExample};
pub use game::{game_to_training_examples, play_game};
pub use storage::save_training_data;
