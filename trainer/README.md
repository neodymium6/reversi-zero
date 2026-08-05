# Reversi Zero Trainer

Initialize the repository from its root before running the trainer:

```bash
make init
make doctor
```

The setup builds `reversi-zero-rs` against the PyTorch installation in this
directory's `.venv`. Prefer `make init` over a standalone `uv sync` when
rebuilding the environment or the Rust extension. `make doctor` verifies that
Python and Rust agree on both CUDA build support and runtime availability.

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

CPU training automatically uses four Torch threads, an NN inference batch of
32, expansion batches of four, and up to 16 concurrent self-play games. These
values avoid CPU oversubscription and can be overridden with `--torch-threads`,
`--selfplay-batch-size`, `--expansion-batch-size`, and `--game-concurrency`.
CUDA retains its previous inference-batch and threading defaults.

The in-memory self-play dataset also defaults to zero DataLoader workers on
CPU, avoiding multiprocessing startup and transfer overhead. CUDA retains four
workers. Use `--num-workers N` to override either default.

CUDA runs export self-play TorchScript models in FP16 by default while retaining
FP32 training weights and checkpoints. CPU runs automatically use FP32:

```bash
../scripts/train --device cuda
```

The Rust inference boundary remains FP32; casting is contained inside the
exported model. Use `--inference-dtype float32` to override the CUDA default.

Evaluate a trained snapshot against its initial model with fixed paired
openings:

```bash
../scripts/evaluate \
  --challenger runs/experiment-001/models/ts/model_iter_2.pt \
  --reference-model runs/experiment-001/models/ts/model_iter_0.pt \
  --output runs/experiment-001/evaluations/iter2-vs-iter0.json
```

CPU evaluation limits each Arena MCTS process to one Torch thread by default,
preventing the concurrent player processes from oversubscribing the machine.
This can be overridden with `--torch-threads N`; CUDA keeps Torch's defaults.

Use `../scripts/evaluate --help` for fixed Alpha-Beta/Random references,
opening-suite reuse, independent challenger/reference expansion batch sizes,
search settings, and promotion thresholds.

Self-play data is written as a recoverable three-file transaction. If a write
is interrupted, the next append restores the previous complete dataset before
continuing. Checkpoints and TorchScript models are also replaced atomically.
