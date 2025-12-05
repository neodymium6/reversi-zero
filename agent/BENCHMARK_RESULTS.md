# Selfplay Benchmark Results (2025-12-06)

Model: `models/ts/latest.pt`
Command: `cargo bench -p reversi-selfplay --bench selfplay_bench` (bench.rs defaults)

## single_game (CPU, sims = 1/10/30)

| MCTS Simulations | Median Time |
|------------------|-------------|
| 1                | ~9.79 ms    |
| 10               | ~53.12 ms   |
| 30               | ~150.60 ms  |

- `bench_single_game` uses `Device::Cpu`.
- Criterion default samples (100); some warmup/sample time warnings are expected.

## multi_game (4 games, concurrency = 4, sims = 10, batch_size = 4, timeout = 1ms)

| Mode           | Total Time (4 games) | Per-Game Time | Notes                        |
|----------------|----------------------|---------------|------------------------------|
| Sequential     | ~286.3 ms            | ~71.6 ms      | Raw model, batch size = 1    |
| Parallel (4x)  | ~134.2 ms            | ~33.5 ms      | Shared batching, bs=4, 1 ms  |

- `bench_multi_game_parallel` uses `Device::cuda_if_available` (GPU in this run).
- Sample count follows Criterion defaults (warnings may reduce samples/adjust target time).

Notes:
- Parallel + batching roughly halves per-game time for 4 concurrent games at sims=10 (default bench.rs settings).
- For light models and low simulation counts, overhead can dominate; re-measure under heavier loads (higher sims or heavier model) for production-like scenarios.

## multi_game_high_sims (4 games, concurrency = 4, sims = 100)

Environment: `cargo bench -p reversi-selfplay --bench selfplay_bench` (multi_game_high_sims group), GPU, sample_size=30 (default env vars):
- `NUM_GAMES_HIGH=4`, `GAME_CONCURRENCY_HIGH=4`, `NUM_SIMULATIONS_HIGH=100`, `BATCH_TIMEOUT_MS_HIGH=1`

| Mode                       | Total Time (4 games) | Per-Game Time | Notes                    |
|----------------------------|----------------------|---------------|--------------------------|
| parallel_base (bs=4)       | ~712 ms              | ~178 ms       | Batch size = game count  |
| parallel_big_batch (bs=16) | ~489 ms              | ~122 ms       | Batch size = 4x games    |

Notes:
- Increasing batch size (4 → 16) reduced per-game time by ~1.4× under higher simulation count (100) with 4 concurrent games.
- Sample size was 30; rerun with more samples if you need tighter confidence. Heavier models or larger sims should amplify batching benefits.
