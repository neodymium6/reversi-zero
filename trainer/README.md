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
Passing an existing directory without `run.resume=true` is an error.

```bash
# Named run
../scripts/train run.dir=runs/experiment-001

# Continue after its latest complete iteration
../scripts/train run.dir=runs/experiment-001 run.resume=true

# Small smoke run
../scripts/train \
  run.dir=runs/smoke \
  run.num_iterations=1 \
  selfplay.games_per_iteration=4 \
  selfplay.simulations=8 \
  training.epochs=1 \
  reference.enabled=false
```

Configuration is composed by Hydra. Select `profile=auto|cpu|gpu` and
`model=dummy|resnet`, then override individual values with `section.key=value`.
The structured schema rejects unknown keys, invalid types, unsupported devices,
inference dtypes, model types, and symmetry multipliers before training starts.
Use `../scripts/train --help` for the complete config and
`../scripts/train --cfg job` to print the resolved application config without
starting a run.

CPU training automatically uses four Torch threads, an NN inference batch of
32, expansion batches of four, and up to 16 concurrent self-play games. These
values avoid CPU oversubscription and can be overridden with
`hardware.torch_threads`, `selfplay.batch_size`,
`selfplay.expansion_batch_size`, and `selfplay.game_concurrency`.
CUDA retains its previous inference-batch and threading defaults.

The in-memory self-play dataset also defaults to zero DataLoader workers on
CPU, avoiding multiprocessing startup and transfer overhead. CUDA retains four
workers. Use `training.num_workers=N` to override either default.

Training lazily applies all eight rotations/reflections to each self-play
example by default. The NPY files are not duplicated, and evaluation uses the
original examples only. Use `training.symmetry_augmentation=1`, `2`, `4`, or
`8` to control the training-example multiplier.

Training also uses a rolling window of the five most recent self-play
iterations by default. Their datasets are lazily concatenated without creating
merged NPY files. Use `training.replay_window=N` to change the history length.

The learning rate uses WSD by default: a linear warmup for the first 2% of the
run, the configured learning rate held through 85%, then a linear decay to 1%
of it over the final 15%. The schedule is updated after every optimizer step
and its state is included in checkpoints and candidate rollback. Set
`training.lr_schedule=constant` for the previous behavior.

Use `run.seed=N` to give comparison runs identical PyTorch model initialization
and training-shuffle seeds. Rust self-play remains stochastic.

CUDA runs export self-play TorchScript models in FP16 by default while retaining
FP32 training weights and checkpoints. CPU runs automatically use FP32:

```bash
../scripts/train profile=gpu model=resnet
```

The Rust inference boundary remains FP32; casting is contained inside the
exported model. FP16 exports use channels-last internally to avoid repeated
cuDNN layout conversions. Use `hardware.inference_dtype=float32` to override
the CUDA default.

Evaluate a trained snapshot against its initial model with fixed paired
openings:

```bash
../scripts/evaluate \
  --challenger runs/experiment-001/models/ts/model_iter_2.pt \
  --reference-model runs/experiment-001/models/ts/model_iter_0.pt \
  --output runs/experiment-001/evaluations/iter2-vs-iter0.json
```

Model-vs-model evaluation uses the native Rust Arena and fixed-shape inference
batches across up to 16 concurrent games. Fixed non-model opponents continue
to use Arena player processes. CPU MCTS player processes use one Torch thread
by default; CUDA keeps Torch's defaults.

Use `../scripts/evaluate --help` for fixed Alpha-Beta/BitMatrix/Random references,
opening-suite reuse, independent challenger/reference expansion batch sizes,
search settings, and promotion thresholds.

Training uses the same paired evaluator as a promotion gate by default. After
each iteration, the candidate must score at least 55% against the incumbent
over 80 openings played with both colors, for 160 total games. A rejected
candidate's model and checkpoint are retained under `models/ts/candidates/`
and `checkpoints/`, while the incumbent model and optimizer are restored for
the next iteration. Use `promotion.enabled=false` to disable gating or
`promotion.require_confidence=true` to additionally require the 95% interval
lower bound to exceed 50%.

After promotion, the selected incumbent is automatically evaluated against
BitMatrix. Random and Alpha-Beta remain available through the standalone
evaluation command for manual smoke tests. The reference search reuses the
self-play simulation and expansion settings; the opening length and seed reuse
the promotion settings. This leaves only two dedicated controls:
`reference.enabled=false` disables the evaluation, and `reference.games=N`
changes the even total number of games against BitMatrix.

Self-play data is written as a recoverable three-file transaction. If a write
is interrupted, the next append restores the previous complete dataset before
continuing. Checkpoints and TorchScript models are also replaced atomically.
