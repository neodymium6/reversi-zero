"""Arena-compatible MCTS player script using reversi_zero_rs.MctsPlayer.

Usage:
    python mcts_player.py BLACK --model /abs/path/to/model.pt [--device cpu|cuda] [--sims 800] [--c-puct 1.5] [--temperature 0.0]

Protocol:
- argv[1]: "BLACK" or "WHITE" (the side to play)
- stdin: one line per position (board string compatible with rust_reversi_core) or "ping"
- stdout: best move index (0-63); responds "pong" to ping; prints error to stderr and exits on failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from reversi_zero_rs import MctsConfigArgs, MctsPlayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arena-compatible MCTS player")
    parser.add_argument("color", choices=["BLACK", "WHITE", "black", "white"])
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to TorchScript model (absolute or relative)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "gpu", None],
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=800,
        help="Number of MCTS simulations",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT constant",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for argmax, recommended for evaluation)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model.resolve()
    mcts_args = MctsConfigArgs(
        num_simulations=args.sims,
        c_puct=args.c_puct,
        temperature=args.temperature,
    )

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    player = MctsPlayer(str(model_path), device=args.device, mcts=mcts_args)
    color = args.color.upper()

    for line in sys.stdin:
        board_str = line.strip()
        if not board_str:
            continue
        if board_str.lower() == "ping":
            print("pong", flush=True)
            continue

        try:
            mv = player.select_move(board_str, color)
        except Exception as exc:  # pragma: no cover - fail fast in Arena
            print(exc, file=sys.stderr, flush=True)
            sys.exit(1)

        print(mv, flush=True)


if __name__ == "__main__":
    main()
