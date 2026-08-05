# Reversi Zero Trainer

Initialize the repository from its root before running the trainer:

```bash
make init
make doctor
```

The setup builds `reversi-zero-rs` against the PyTorch installation in this
directory's `.venv`. Prefer `make init` over a standalone `uv sync` when
rebuilding the environment or the Rust extension.

Run training from this directory:

```bash
../scripts/train
```

Each invocation creates an isolated timestamped directory under `runs/`.
Passing an existing directory without `--resume` is an error.

```bash
# Named run
../scripts/train --run-dir runs/experiment-001

# Continue after its latest complete iteration
../scripts/train --run-dir runs/experiment-001 --resume

# Small smoke run
../scripts/train \
  --run-dir runs/smoke \
  --num-iterations 1 \
  --games-per-iteration 4 \
  --simulations 8 \
  --epochs 1 \
  --no-arena
```

See `../scripts/train --help` for all configuration options.

Evaluate a trained snapshot against its initial model with fixed paired
openings:

```bash
../scripts/evaluate \
  --challenger runs/experiment-001/models/ts/model_iter_2.pt \
  --reference-model runs/experiment-001/models/ts/model_iter_0.pt \
  --output runs/experiment-001/evaluations/iter2-vs-iter0.json
```

Use `../scripts/evaluate --help` for fixed Alpha-Beta/Random references,
opening-suite reuse, search settings, and promotion thresholds.

Self-play data is written as a recoverable three-file transaction. If a write
is interrupted, the next append restores the previous complete dataset before
continuing. Checkpoints and TorchScript models are also replaced atomically.
