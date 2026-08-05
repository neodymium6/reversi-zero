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
  --no-reference-eval
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

Training lazily applies all eight rotations/reflections to each self-play
example by default. The NPY files are not duplicated, and evaluation uses the
original examples only. Use `--symmetry-augmentation 1`, `2`, `4`, or `8` to
control the training-example multiplier.

Training also uses a rolling window of the five most recent self-play
iterations by default. Their datasets are lazily concatenated without creating
merged NPY files. Use `--replay-window N` to change the history length.

Use `--seed N` to give comparison runs identical PyTorch model initialization
and training-shuffle seeds. Rust self-play remains stochastic.

CUDA runs export self-play TorchScript models in FP16 by default while retaining
FP32 training weights and checkpoints. CPU runs automatically use FP32:

```bash
../scripts/train --device cuda
```

The Rust inference boundary remains FP32; casting is contained inside the
exported model. FP16 exports use channels-last internally to avoid repeated
cuDNN layout conversions. Use `--inference-dtype float32` to override the CUDA
default.

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

Use `../scripts/evaluate --help` for fixed Alpha-Beta/BitMatrix/Random references,
opening-suite reuse, independent challenger/reference expansion batch sizes,
search settings, and promotion thresholds.

Training uses the same paired evaluator as a promotion gate by default. After
each iteration, the candidate must score at least 55% against the incumbent
over 40 openings played with both colors, and the 95% interval lower bound must
exceed 50%. A rejected candidate's model and checkpoint are retained under
`models/ts/candidates/` and `checkpoints/`, while the incumbent model and
optimizer are restored for the next iteration. Use `--no-promotion` to disable
gating or `--no-promotion-require-confidence` to use only the score threshold.

After promotion, the selected incumbent is automatically evaluated against
Random, Alpha-Beta, and BitMatrix with one fixed opening suite shared across
opponents and iterations. The
reference search reuses the self-play simulation and expansion settings; the
opening length and seed reuse the promotion settings. This leaves only two
dedicated controls: `--no-reference-eval` disables the suite, and
`--reference-games N` changes the even total number of games per opponent.

Self-play data is written as a recoverable three-file transaction. If a write
is interrupted, the next append restores the previous complete dataset before
continuing. Checkpoints and TorchScript models are also replaced atomically.
