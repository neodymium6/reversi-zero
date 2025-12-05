use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;
use reversi_mcts::{BatchingModel, MctsConfig};
use reversi_nn::NnModel;
use reversi_selfplay::play_game;
use std::time::Duration;
use tch::Device;

/// Benchmark for a single self-play game with different MCTS simulation counts
fn bench_single_game(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_game");

    // Load model once for all benchmarks
    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "../../../models/ts/latest.pt".to_string());

    let device = Device::Cpu; // Use CPU for consistent benchmarking
    let model = NnModel::load(&model_path, device).expect("Failed to load model");

    // Benchmark with different simulation counts
    for num_sims in [1, 10, 30].iter() {
        let config = MctsConfig::default()
            .with_simulations(*num_sims)
            .with_c_puct(1.5)
            .with_temperature(1.0)
            .with_dirichlet_noise(0.3, 0.25);

        group.bench_with_input(BenchmarkId::from_parameter(num_sims), num_sims, |b, _| {
            b.iter(|| {
                let record = play_game(black_box(&model), black_box(&config)).expect("Game failed");
                black_box(record)
            });
        });
    }

    group.finish();
}

/// Benchmark running multiple games sequential vs parallel (shared batching model)
fn bench_multi_game_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_game");

    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "../../../models/ts/latest.pt".to_string());

    let device = Device::cuda_if_available();
    let num_games: usize = std::env::var("NUM_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let batch_size: usize = std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_games);
    let batch_timeout_ms: u64 = std::env::var("BATCH_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let game_concurrency: usize = std::env::var("GAME_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_games);
    let num_simulations: u32 = std::env::var("NUM_SIMULATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Load two models: one raw for sequential (no batching), one wrapped for parallel batching.
    let seq_model = NnModel::load(&model_path, device).expect("Failed to load model");
    let base_model = NnModel::load(&model_path, device).expect("Failed to load model");
    let batching_model = BatchingModel::new(
        base_model,
        batch_size,
        Duration::from_millis(batch_timeout_ms),
    );

    // Sequential: use a raw model and batch_size=1 (no batching overhead).
    let config_seq = MctsConfig::default()
        .with_simulations(num_simulations)
        .with_batch_size(1)
        .with_batch_timeout_ms(batch_timeout_ms)
        .with_c_puct(1.5)
        .with_temperature(1.0)
        .with_dirichlet_noise(0.3, 0.25);

    // Parallel: shared batching model with the configured batch_size.
    let config_par = MctsConfig::default()
        .with_simulations(num_simulations)
        .with_batch_size(batch_size as u32)
        .with_batch_timeout_ms(batch_timeout_ms)
        .with_c_puct(1.5)
        .with_temperature(1.0)
        .with_dirichlet_noise(0.3, 0.25);

    group.bench_function(
        BenchmarkId::new("sequential", format!("{}-games", num_games)),
        |b| {
            let model = &seq_model;
            b.iter(|| {
                for _ in 0..num_games {
                    let _record = play_game(model, &config_seq).expect("Game failed");
                }
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "parallel",
            format!("{}-games-conc{}", num_games, game_concurrency),
        ),
        |b| {
            let model = batching_model.clone();
            b.iter(|| {
                (0..num_games)
                    .into_par_iter()
                    .with_max_len(1) // keep tasks small to let batching occur
                    .with_min_len(1)
                    .map(|_| play_game(&model, &config_par).expect("Game failed"))
                    .collect::<Vec<_>>();
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_single_game, bench_multi_game_parallel);
criterion_main!(benches);
