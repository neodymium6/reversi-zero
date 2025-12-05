# Selfplay Benchmark Results

**Date:** 2025-12-06
**Hardware:** CPU
**Model:** `models/ts/latest.pt`
**Samples:** 100 iterations per configuration

## Results

| MCTS Simulations | Mean Time | Median Time | Std Dev | 95% CI Lower | 95% CI Upper |
|-----------------|-----------|-------------|---------|--------------|--------------|
| 1               | 11.17 ms  | 10.69 ms    | 1.35 ms | 10.90 ms     | 11.43 ms     |
| 10              | 56.71 ms  | 54.86 ms    | 5.88 ms | 55.58 ms     | 57.90 ms     |
| 30              | 154.78 ms | 153.29 ms   | 11.11 ms| 152.64 ms    | 156.97 ms    |

## Performance Analysis

### Scaling

- **1 → 10 simulations**: 5.08× slower
- **10 → 30 simulations**: 2.73× slower
- **1 → 30 simulations**: 13.86× slower

Roughly linear scaling with simulation count.

### Throughput

| Simulations | Games/sec | Games/min | Games/hour |
|------------|-----------|-----------|------------|
| 1          | 89.5      | 5,370     | 322,200    |
| 10         | 17.6      | 1,058     | 63,480     |
| 30         | 6.5       | 387       | 23,220     |

### Production Estimate (800 simulations)

- Time per game: ~4.13 seconds
- Throughput: ~0.24 games/second
- 100 games: ~6.9 minutes
- 1,000 games: ~1.15 hours
