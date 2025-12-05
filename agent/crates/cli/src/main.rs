use anyhow::Result;
use rayon::prelude::*;
use reversi_mcts::{BatchingModel, MctsConfig};
use reversi_nn::NnModel;
use reversi_selfplay::{game_to_training_examples, play_game, save_training_data};
use std::time::Duration;
use tch::Device;

fn main() -> Result<()> {
    // Configuration
    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "../models/ts/latest.pt".to_string());
    let num_games = std::env::var("NUM_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let game_concurrency = std::env::var("GAME_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_games.min(10));
    let num_simulations = std::env::var("NUM_SIMULATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(800);
    let batch_size = std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let batch_timeout_ms = std::env::var("BATCH_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let output_path = std::env::var("OUTPUT_PATH").unwrap_or_else(|_| "selfplay_data".to_string());

    // Device selection
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // Load model
    println!("Loading model from {}...", model_path);
    let base_model = NnModel::load(&model_path, device)?;
    println!("Model loaded successfully!");

    // Wrap model with a batching worker shared across games
    let batching_model = BatchingModel::new(
        base_model,
        batch_size as usize,
        Duration::from_millis(batch_timeout_ms),
    );

    // Configure MCTS for self-play
    let config = MctsConfig::default()
        .with_simulations(num_simulations)
        .with_batch_size(batch_size)
        .with_batch_timeout_ms(batch_timeout_ms as u64)
        .with_c_puct(1.5)
        .with_temperature(1.0) // Stochastic for exploration
        .with_dirichlet_noise(0.3, 0.25); // Add exploration noise

    println!("\nSelf-play configuration:");
    println!("  Number of games: {}", num_games);
    println!("  Game concurrency: {}", game_concurrency);
    println!("  MCTS simulations: {}", config.num_simulations);
    println!("  C_PUCT: {}", config.c_puct);
    println!("  Temperature: {}", config.temperature);
    println!(
        "  Dirichlet noise: {} (alpha={}, epsilon={})",
        config.add_dirichlet_noise, config.dirichlet_alpha, config.dirichlet_epsilon
    );
    println!(
        "  Batch size: {} (timeout {} ms)",
        batch_size, batch_timeout_ms
    );
    println!("  Output path: {}", output_path);
    println!();

    // Play multiple games
    let mut all_examples = Vec::new();

    // Run games in parallel while sharing the batching model.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(game_concurrency)
        .build()?;

    let game_results: Vec<anyhow::Result<Vec<_>>> = pool.install(|| {
        (0..num_games)
            .into_par_iter()
            .map(|game_num| {
                println!("Playing game {}/{}...", game_num + 1, num_games);
                let local_model = batching_model.clone();
                let record = play_game(&local_model, &config)?;
                println!(
                    "  Game finished: {} moves, result: {:?}",
                    record.len(),
                    record.winner
                );
                let examples = game_to_training_examples(&record);
                println!("  Generated {} training examples", examples.len());
                Ok(examples)
            })
            .collect()
    });

    for res in game_results {
        let examples = res?;
        all_examples.extend(examples);
    }

    // Save all training data
    println!(
        "\nSaving {} total training examples to {}...",
        all_examples.len(),
        output_path
    );
    save_training_data(&all_examples, &output_path)?;

    println!("\nSelf-play completed successfully!");
    println!("Created files:");
    println!("  - {}_states.npy", output_path);
    println!("  - {}_policies.npy", output_path);
    println!("  - {}_values.npy", output_path);

    Ok(())
}
