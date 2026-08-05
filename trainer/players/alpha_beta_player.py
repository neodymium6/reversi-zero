"""Simple alpha-beta player using rust_reversi search (baseline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rust_reversi import AlphaBetaSearch, Board, PieceEvaluator, Turn

from reversi_zero_trainer.openings import OpeningController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arena-compatible alpha-beta player")
    parser.add_argument("color", choices=["BLACK", "WHITE", "black", "white"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--openings-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    color = args.color.upper()
    turn = Turn.BLACK if color == "BLACK" else Turn.WHITE
    board = Board()

    evaluator = PieceEvaluator()
    search = AlphaBetaSearch(evaluator, args.depth, win_score=1 << 10)
    openings = OpeningController(args.openings_file) if args.openings_file else None

    while True:
        try:
            board_str = input().strip()

            if board_str.lower() == "ping":
                print("pong", flush=True)
                continue

            board.set_board_str(board_str, turn)
            move = openings.select_forced_move(board_str, color) if openings else None
            if move is None:
                move = search.get_move(board)
            print(move, flush=True)
        except Exception as exc:  # pragma: no cover
            print(exc, file=sys.stderr, flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
