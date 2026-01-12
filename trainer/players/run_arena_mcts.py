"""Run a quick Arena duel between two MCTS players backed by reversi_zero_rs.MctsPlayer.

Assumes `rust_reversi.Arena` passes the color argument to each player command.
Each player command is `python mcts_player.py <COLOR> --model ... --sims ...`.

Example:
    python trainer/players/run_arena_mcts.py \
        --black-model /abs/path/to/model.pt \
        --white-model /abs/path/to/model.pt \
        --games 20 --sims 800 --c-puct 1.5 --temperature 0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rust_reversi import Arena


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arena duel using MctsPlayer.")
    parser.add_argument(
        "--black-model", required=True, type=Path, help="TorchScript model for Black."
    )
    parser.add_argument(
        "--white-model", required=True, type=Path, help="TorchScript model for White."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games (even number recommended).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "gpu"],
        default=None,
        help="Device for both players.",
    )
    parser.add_argument("--sims", type=int, default=800, help="MCTS simulations.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT constant.")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature (0.0 = argmax)."
    )
    return parser.parse_args()


def main() -> None:
    # args = parse_args()
    mcts_script = Path(__file__).parent / "mcts_player.py"

    # common_flags = [
    #     f"--sims={args.sims}",
    #     f"--c-puct={args.c_puct}",
    #     f"--temperature={args.temperature}",
    # ]
    # if args.device:
    #     common_flags.append(f"--device={args.device}")

    # player1 = [
    #     sys.executable,
    #     str(mcts_script),
    #     "--model",
    #     str(args.black_model.resolve()),
    #     *common_flags,
    # ]
    random_script = Path(__file__).parent / "random_player.py"
    player1 = [
        sys.executable,
        str(random_script),
    ]

    # mcts_model_path = "/home/neodymium6/projects/reversi-zero/models/ts/latest.pt"
    mcts_model_path = (
        "/home/neodymium6/projects/reversi-zero/trainer/models/ts/model_iter_0.pt"
    )
    common_mcts_flags = [
        "--sims=100",
        "--c-puct=1.5",
        "--temperature=0.0",
    ]
    player2 = [
        sys.executable,
        str(mcts_script),
        "--model",
        mcts_model_path,
        *common_mcts_flags,
    ]

    arena = Arena(player1, player2)
    games = 10
    # arena.play_n(args.games)
    arena.play_n(games)

    wins1, wins2, draws = arena.get_stats()
    pieces1, pieces2 = arena.get_pieces()

    # print(f"Games: {args.games}")
    print(f"Games: {games}")
    print(f"Player1 (black model) wins: {wins1}")
    print(f"Player2 (white model) wins: {wins2}")
    print(f"Draws: {draws}")
    print(f"Total pieces - P1: {pieces1}, P2: {pieces2}")


if __name__ == "__main__":
    main()
