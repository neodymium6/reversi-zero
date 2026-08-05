"""Arena-compatible random player."""

import argparse
import random
import sys
from pathlib import Path

from rust_reversi import Board, Turn

from reversi_zero_trainer.openings import OpeningController


def parse_args():
    parser = argparse.ArgumentParser(description="Arena-compatible random player")
    parser.add_argument("color", choices=["BLACK", "WHITE", "black", "white"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--openings-file", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    color = args.color.upper()
    turn = Turn.BLACK if color == "BLACK" else Turn.WHITE
    board = Board()
    rng = random.Random(args.seed)  # nosec B311 - reproducible game play
    openings = OpeningController(args.openings_file) if args.openings_file else None

    while True:
        try:
            board_str = input().strip()

            # Handle ping/pong protocol
            if board_str == "ping":
                print("pong", flush=True)
                continue

            # Update board state and get random move
            board.set_board_str(board_str, turn)
            move = openings.select_forced_move(board_str, color) if openings else None
            if move is None:
                move = rng.choice(board.get_legal_moves_vec())

            print(move, flush=True)

        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
