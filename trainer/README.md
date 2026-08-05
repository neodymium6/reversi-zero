# Reversi Zero Trainer

Run training from this directory:

```bash
uv run train
```

Each invocation creates an isolated timestamped directory under `runs/`.
Passing an existing directory without `--resume` is an error.

```bash
# Named run
uv run train --run-dir runs/experiment-001

# Continue after its latest complete iteration
uv run train --run-dir runs/experiment-001 --resume

# Small smoke run
uv run train \
  --run-dir runs/smoke \
  --num-iterations 1 \
  --games-per-iteration 4 \
  --simulations 8 \
  --epochs 1 \
  --no-arena
```

See `uv run train --help` for all configuration options.

Self-play data is written as a recoverable three-file transaction. If a write
is interrupted, the next append restores the previous complete dataset before
continuing. Checkpoints and TorchScript models are also replaced atomically.
